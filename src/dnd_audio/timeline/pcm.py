"""Read mono PCM out of a RIFF/RF64 container as float32, by seeking (ADR-0011, ADR-0030).

The working path needs *random* windowed access — a mix pass asks for
``[start, start + n)`` of a track, not a stream from the beginning — so the reader seeks
into the ``data`` chunk and reads exactly the bytes it was asked for. Piping the file
through FFmpeg would be a stream, and one subprocess per window is not a reader; buffering
the track to make a stream seekable is the INV-07 violation the whole design avoids.

Nothing here reads more than the caller asked for, so the memory bound is the caller's
window and not the file's size. A four-hour source is opened, seeked, and read a fraction
of a second at a time exactly like a two-second one. That includes the integer formats: an
integer window is unpacked into arrays the size of the window, never the size of the file,
which matters because NumPy has no packed 24-bit dtype and the unpacking step is precisely
where a careless implementation expands a whole source.

**Mono, and only formats that convert to float32 exactly** (ADR-0030). A format is accepted
when it is a signed little-endian integer or an IEEE float *and* the conversion loses
nothing: today `pcm_f32le`, `pcm_s24le` and `pcm_s16le`. `pcm_s32le` is still refused —
float32 carries 24 significand bits, so `2147483647` becomes `2147483648.0` — but it is
refused with the reason that is true of *it*, rather than every integer format being
refused with a sentence that is false for 24-bit. The scaling divisor is `2**(bits-1)`, a
power of two, which is why the integer conversions are exact rather than merely close.

The format is validated against the bytes on disk rather than trusted from the manifest.
The manifest says what FFprobe reported; this says what is actually there, and INV-01
means the two should agree — so checking is how a disagreement becomes visible.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, ClassVar, Final, Self

import numpy as np
import numpy.typing as npt

from dnd_audio.errors import DndAudioError
from dnd_audio.inspection.riff import read_inventory

__all__ = [
    "ACCEPTED_CODECS",
    "ACCEPTED_FORMATS",
    "BYTES_PER_SAMPLE",
    "FLOAT32",
    "INT16",
    "INT24",
    "PcmError",
    "PcmReader",
    "PcmSource",
    "SampleFormat",
    "codec_name_for",
    "decode",
    "open_pcm",
    "refusal_reason",
]

#: mono float32 — the format everything this pipeline *writes* uses. The reader derives
#: its arithmetic from the source's own :class:`SampleFormat` instead.
BYTES_PER_SAMPLE: Final = 4

_WAVE_FORMAT_PCM: Final = 1
_WAVE_FORMAT_IEEE_FLOAT: Final = 3
_WAVE_FORMAT_EXTENSIBLE: Final = 0xFFFE
_FMT_FIXED_BYTES: Final = 16
_CHUNK_HEADER_BYTES: Final = 8
_BITS_PER_BYTE: Final = 8

#: float32's significand. A signed integer format converts exactly when its whole width
#: fits here, because the divisor `2**(bits-1)` is a power of two and contributes no error.
_EXACT_SIGNIFICAND_BITS: Final = 24

_SIGNED_LE_INTEGER = re.compile(r"pcm_s(\d+)le\Z")
_IEEE_FLOAT_LE = re.compile(r"pcm_f(\d+)le\Z")


class PcmError(DndAudioError):
    """A source cannot be read as mono float32 PCM.

    The spec's "a source file cannot be decoded" fatal error. Separate from a layout
    failure because it is about the bytes rather than about the timeline.
    """

    default_code: ClassVar[str] = "undecodable_source"


@dataclass(frozen=True, slots=True)
class SampleFormat:
    """One encoding this reader decodes, named the way FFprobe names it.

    Sharing FFprobe's `codec_name` is deliberate: the manifest records what FFprobe said
    and this module reads what is on disk, and a single vocabulary is what lets the two be
    compared rather than translated.
    """

    codec_name: str
    #: The WAVE `fmt ` tag that carries it.
    format_tag: int
    bits: int

    @property
    def bytes_per_sample(self) -> int:
        return self.bits // _BITS_PER_BYTE


FLOAT32: Final = SampleFormat("pcm_f32le", _WAVE_FORMAT_IEEE_FLOAT, 32)
INT24: Final = SampleFormat("pcm_s24le", _WAVE_FORMAT_PCM, 24)
INT16: Final = SampleFormat("pcm_s16le", _WAVE_FORMAT_PCM, 16)

#: Every format that satisfies ADR-0030's two-part rule. Ordered widest first so a
#: diagnostic listing them reads the way an operator thinks about quality.
ACCEPTED_FORMATS: Final = (FLOAT32, INT24, INT16)
ACCEPTED_CODECS: Final = frozenset(fmt.codec_name for fmt in ACCEPTED_FORMATS)

_BY_TAG_AND_BITS: Final = {(fmt.format_tag, fmt.bits): fmt for fmt in ACCEPTED_FORMATS}


def codec_name_for(tag: int, bits: int) -> str:
    """What FFprobe would call the format a WAVE `fmt ` chunk declares.

    Only exact for the families this reader knows about, which is the point: it exists so
    a refusal read from the *bytes* can name the format the same way a refusal read from
    the *manifest* does.
    """
    if tag == _WAVE_FORMAT_PCM:
        # 8-bit WAV PCM is unsigned; every wider width is signed. That asymmetry is in the
        # format, not in this reader, and it is why `pcm_u8` is refused separately.
        return "pcm_u8" if bits == _BITS_PER_BYTE else f"pcm_s{bits}le"
    if tag == _WAVE_FORMAT_IEEE_FLOAT:
        return f"pcm_f{bits}le"
    return f"format tag {tag} at {bits} bits"


def refusal_reason(codec_name: str) -> str:
    """Why a format is not accepted — naming the half of ADR-0030's rule that it fails.

    A refusal that gives a reason which is false for the file in front of the operator is
    worse than no reason at all: it sends them to check the wrong thing. This is the
    defect M8 exists to fix, so the branches are per-family rather than one sentence.
    """
    signed = _SIGNED_LE_INTEGER.match(codec_name)
    if signed is not None:
        bits = int(signed.group(1))
        largest = 2 ** (bits - 1) - 1
        return (
            f"{codec_name} carries {bits} significant bits and float32's significand "
            f"carries {_EXACT_SIGNIFICAND_BITS}, so converting it would round: {largest} "
            f"becomes {float(np.float32(largest)):.1f}. A lossless path is never quietly "
            f"rounded (ADR-0011)."
        )
    if _IEEE_FLOAT_LE.match(codec_name) is not None:
        return (
            f"{codec_name} is wider than the float32 the working path carries, so "
            f"converting it would round. A lossless path is never quietly rounded "
            f"(ADR-0011)."
        )
    if codec_name.startswith("pcm_u"):
        return (
            f"{codec_name} is unsigned PCM, offset by half its range rather than signed. "
            f"It would convert exactly, but through a different convention than the "
            f"signed formats this reader implements, and no DJI file has been seen "
            f"writing it (OQ-007) — so it is refused as untested rather than as "
            f"unrepresentable (ADR-0030)."
        )
    return (
        f"{codec_name} is not a little-endian signed integer or IEEE float, which is the "
        f"family this reader decodes (ADR-0030)."
    )


@dataclass(frozen=True, slots=True)
class PcmSource:
    """Where one file's samples are, established once by walking the container."""

    path: Path
    #: Byte offset of the first sample — the ``data`` chunk's payload, not its header.
    data_offset: int
    n_samples: int
    sample_rate: int
    sample_format: SampleFormat = FLOAT32

    @property
    def bytes_per_sample(self) -> int:
        return self.sample_format.bytes_per_sample

    @property
    def data_bytes(self) -> int:
        return self.n_samples * self.bytes_per_sample


def open_pcm(path: Path) -> PcmSource:
    """Locate and validate a file's PCM data without reading any of it.

    RF64 is handled by the chunk walk, which resolves a sentinel ``data`` size through
    ``ds64`` — so a file past RIFF's 4 GiB limit reports its real length here rather than
    4 294 967 295.

    Raises:
        PcmError: if the container has no ``fmt ``/``data`` pair, is not mono in a format
            that converts to float32 exactly (ADR-0030), or declares a data size that is
            not a whole number of samples.
    """
    inventory = read_inventory(path)
    fmt = inventory.find("fmt ")
    data = inventory.find("data")
    if fmt is None or data is None:
        missing = " and ".join(
            name for name, chunk in (("fmt ", fmt), ("data", data)) if chunk is None
        )
        message = f"{path} has no {missing} chunk, so it holds no readable PCM"
        raise PcmError(message)

    with path.open("rb") as handle:
        handle.seek(fmt.offset + _CHUNK_HEADER_BYTES)
        raw = handle.read(_FMT_FIXED_BYTES)
    if len(raw) < _FMT_FIXED_BYTES:
        message = f"{path} has a truncated fmt chunk ({len(raw)} bytes)"
        raise PcmError(message)

    tag, channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", raw)
    sample_format = _accepted_format(path, tag, channels, bits)

    if data.size % sample_format.bytes_per_sample:
        message = (
            f"{path} declares a data chunk of {data.size} bytes, which is not a whole "
            f"number of {sample_format.bytes_per_sample}-byte samples. Flooring it would "
            f"invent the length of a file this pipeline is about to place on a timeline."
        )
        raise PcmError(message, code="unknown_sample_count")

    return PcmSource(
        path=path,
        data_offset=data.offset + _CHUNK_HEADER_BYTES,
        n_samples=data.size // sample_format.bytes_per_sample,
        sample_rate=sample_rate,
        sample_format=sample_format,
    )


def _accepted_format(path: Path, tag: int, channels: int, bits: int) -> SampleFormat:
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        message = (
            f"{path} is WAVE_FORMAT_EXTENSIBLE. Its sample format lives in a subformat "
            f"GUID this reader does not parse; no DJI file has been seen using it "
            f"(OQ-001), so support waits for one that does rather than being guessed at."
        )
        raise PcmError(message)

    sample_format = _BY_TAG_AND_BITS.get((tag, bits))
    if sample_format is None:
        accepted = ", ".join(fmt.codec_name for fmt in ACCEPTED_FORMATS)
        message = (
            f"{path} is {codec_name_for(tag, bits)}, and the working path reads "
            f"{accepted}. {refusal_reason(codec_name_for(tag, bits))}"
        )
        raise PcmError(message)

    if channels != 1:
        message = (
            f"{path} has {channels} channels; a transmitter records one. Two suggests a "
            f"receiver mixdown rather than a transmitter recording."
        )
        raise PcmError(message)
    return sample_format


class PcmReader:
    """An open file, read in bounded windows.

    A context manager because a mix pass keeps six of these open across a whole session:
    reopening per window would be a syscall per window for no benefit, and leaking handles
    across six tracks and thousands of windows is its own failure.
    """

    def __init__(self, source: PcmSource) -> None:
        self._source = source
        self._handle: BinaryIO | None = None

    def __enter__(self) -> Self:
        self._handle = self._source.path.open("rb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def source(self) -> PcmSource:
        return self._source

    def read(self, start_sample: int, n_samples: int) -> npt.NDArray[np.float32]:
        """Exactly ``[start_sample, start_sample + n_samples)`` of this file's samples.

        Raises:
            PcmError: if the range runs past the data chunk. The segment map guarantees it
                does not, so reaching this means the file on disk no longer matches the
                manifest the map was built from — which INV-01 says cannot happen during a
                run, and which a truncated file between runs would cause.
        """
        if self._handle is None:
            message = f"{self._source.path} is not open; use PcmReader as a context manager"
            raise PcmError(message)
        if start_sample < 0 or n_samples < 0:
            message = f"cannot read {n_samples} samples from {start_sample}"
            raise PcmError(message)
        if start_sample + n_samples > self._source.n_samples:
            message = (
                f"{self._source.path} holds {self._source.n_samples} samples, but "
                f"[{start_sample}, {start_sample + n_samples}) was requested. The file no "
                f"longer matches the manifest this timeline was built from; re-run "
                f"`dnd-audio inspect`."
            )
            raise PcmError(message, code="source_changed")

        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        width = self._source.bytes_per_sample
        self._handle.seek(self._source.data_offset + start_sample * width)
        raw = self._handle.read(n_samples * width)
        if len(raw) != n_samples * width:
            message = (
                f"{self._source.path} returned {len(raw)} bytes where "
                f"{n_samples * width} were expected; the file was truncated "
                f"after it was inspected"
            )
            raise PcmError(message, code="source_changed")
        return decode(raw, self._source.sample_format)


def decode(raw: bytes, sample_format: SampleFormat) -> npt.NDArray[np.float32]:
    """One window of packed samples as float32, exactly (ADR-0030).

    Every intermediate is the size of ``raw`` and nothing here is retained, so the memory
    cost of decoding is the caller's window whatever the width on disk.
    """
    if sample_format is FLOAT32:
        # Little-endian explicitly, so the working path does not depend on the host's
        # byte order the way a bare `float32` view would.
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)

    values = _widen(raw, sample_format)
    # `2**(bits-1)` is a power of two and the values fit float32's significand, so this
    # division introduces no error at all — which is the whole reason the format is
    # accepted. It is also the convention libsndfile and FFmpeg decode by, and the reader
    # is cross-checked against FFmpeg's own output rather than trusted on that point.
    return values.astype(np.float32) / np.float32(2 ** (sample_format.bits - 1))


def _widen(raw: bytes, sample_format: SampleFormat) -> npt.NDArray[np.int32]:
    """Packed little-endian signed integers as int32, sign preserved."""
    if sample_format.bits != _EXACT_SIGNIFICAND_BITS:
        return np.frombuffer(raw, dtype=f"<i{sample_format.bytes_per_sample}").astype(np.int32)

    # NumPy has no packed 24-bit dtype. Placing each sample's three bytes in the *high*
    # three bytes of a little-endian int32 makes the hardware sign-extend it for free;
    # the arithmetic shift back down then recovers the original signed value. The
    # alternative — building the sign bit by hand with a comparison and a subtraction —
    # is a branch per sample and gets the boundary wrong at least once per project.
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    wide = np.zeros((packed.shape[0], 4), dtype=np.uint8)
    wide[:, 1:] = packed
    shifted: npt.NDArray[np.int32] = np.right_shift(wide.view("<i4").reshape(-1), _BITS_PER_BYTE)
    return shifted
