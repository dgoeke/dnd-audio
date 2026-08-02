"""Read mono float32 PCM out of a RIFF/RF64 container, by seeking (ADR-0011).

The working path needs *random* windowed access — a mix pass asks for
``[start, start + n)`` of a track, not a stream from the beginning — so the reader seeks
into the ``data`` chunk and reads exactly the bytes it was asked for. Piping the file
through FFmpeg would be a stream, and one subprocess per window is not a reader; buffering
the track to make a stream seekable is the INV-07 violation the whole design avoids.

Nothing here reads more than the caller asked for, so the memory bound is the caller's
window and not the file's size. A four-hour source is opened, seeked, and read a fraction
of a second at a time exactly like a two-second one.

**Mono 32-bit float only.** That is what the session contract specifies and what dual-file
mode's `orig` is. Integer PCM is refused rather than converted, because `pcm_s32le` cannot
become float32 exactly — float32 carries 24 mantissa bits, so `2147483647` becomes
`2147483648.0` — and a "lossless integer path" is not available to be built. When a real
recovery need appears, this is the module that grows.

The format is validated against the bytes on disk rather than trusted from the manifest.
The manifest says what FFprobe reported; this says what is actually there, and INV-01
means the two should agree — so checking is how a disagreement becomes visible.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, ClassVar, Final, Self

import numpy as np
import numpy.typing as npt

from dnd_audio.errors import DndAudioError
from dnd_audio.inspection.riff import read_inventory

__all__ = ["BYTES_PER_SAMPLE", "PcmError", "PcmReader", "PcmSource", "open_pcm"]

#: mono float32.
BYTES_PER_SAMPLE: Final = 4

_WAVE_FORMAT_IEEE_FLOAT: Final = 3
_WAVE_FORMAT_EXTENSIBLE: Final = 0xFFFE
_FMT_FIXED_BYTES: Final = 16
_CHUNK_HEADER_BYTES: Final = 8


class PcmError(DndAudioError):
    """A source cannot be read as mono float32 PCM.

    The spec's "a source file cannot be decoded" fatal error. Separate from a layout
    failure because it is about the bytes rather than about the timeline.
    """

    default_code: ClassVar[str] = "undecodable_source"


@dataclass(frozen=True, slots=True)
class PcmSource:
    """Where one file's samples are, established once by walking the container."""

    path: Path
    #: Byte offset of the first sample — the ``data`` chunk's payload, not its header.
    data_offset: int
    n_samples: int
    sample_rate: int

    @property
    def data_bytes(self) -> int:
        return self.n_samples * BYTES_PER_SAMPLE


def open_pcm(path: Path) -> PcmSource:
    """Locate and validate a file's PCM data without reading any of it.

    RF64 is handled by the chunk walk, which resolves a sentinel ``data`` size through
    ``ds64`` — so a file past RIFF's 4 GiB limit reports its real length here rather than
    4 294 967 295.

    Raises:
        PcmError: if the container has no ``fmt ``/``data`` pair, is not mono 32-bit
            float, or declares a data size that is not a whole number of samples.
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
    _reject_unreadable_format(path, tag, channels, bits)

    if data.size % BYTES_PER_SAMPLE:
        message = (
            f"{path} declares a data chunk of {data.size} bytes, which is not a whole "
            f"number of {BYTES_PER_SAMPLE}-byte samples. Flooring it would invent the "
            f"length of a file this pipeline is about to place on a timeline."
        )
        raise PcmError(message, code="unknown_sample_count")

    return PcmSource(
        path=path,
        data_offset=data.offset + _CHUNK_HEADER_BYTES,
        n_samples=data.size // BYTES_PER_SAMPLE,
        sample_rate=sample_rate,
    )


def _reject_unreadable_format(path: Path, tag: int, channels: int, bits: int) -> None:
    if tag == _WAVE_FORMAT_EXTENSIBLE:
        message = (
            f"{path} is WAVE_FORMAT_EXTENSIBLE. Its sample format lives in a subformat "
            f"GUID this reader does not parse; no DJI file has been seen using it "
            f"(OQ-001), so support waits for one that does rather than being guessed at."
        )
        raise PcmError(message)
    if tag != _WAVE_FORMAT_IEEE_FLOAT or bits != BYTES_PER_SAMPLE * 8:
        message = (
            f"{path} is format tag {tag} at {bits} bits, and the working path reads "
            f"32-bit IEEE float (tag {_WAVE_FORMAT_IEEE_FLOAT}). An integer format cannot "
            f"be converted to float32 exactly, so it is refused rather than quietly "
            f"rounded (ADR-0011)."
        )
        raise PcmError(message)
    if channels != 1:
        message = (
            f"{path} has {channels} channels; a transmitter records one. Two suggests a "
            f"receiver mixdown rather than a transmitter recording."
        )
        raise PcmError(message)


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

        self._handle.seek(self._source.data_offset + start_sample * BYTES_PER_SAMPLE)
        raw = self._handle.read(n_samples * BYTES_PER_SAMPLE)
        if len(raw) != n_samples * BYTES_PER_SAMPLE:
            message = (
                f"{self._source.path} returned {len(raw)} bytes where "
                f"{n_samples * BYTES_PER_SAMPLE} were expected; the file was truncated "
                f"after it was inspected"
            )
            raise PcmError(message, code="source_changed")
        # Little-endian explicitly, so the working path does not depend on the host's
        # byte order the way a bare `float32` view would.
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
