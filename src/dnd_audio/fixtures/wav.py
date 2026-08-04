"""Assemble the RIFF/RF64 bytes the synthetic fixtures are made of.

A convenience library would be enough to write *audio*. It is not enough to write the
things M1's inspection code has to be tested against: a Broadcast-WAV ``bext`` chunk
carrying a sample-accurate ``time_reference``, an ``INFO``/``ISMP`` timecode tag, a
vendor-private chunk nothing can parse, and RF64's 64-bit sizes. So the bytes are
assembled here directly.

This module is a *writer only*. :mod:`dnd_audio.inspection.riff` parses, and the two
share no table of chunk layouts on purpose: if they did, a malformed-file test would be
exercising the shared table rather than the parser.

What was verified against FFmpeg 8.0 before this was written, because each would have
forced a different design:

* ``ffprobe`` reports a ``bext`` time reference as ``format.tags.time_reference``.
* It reports an ``INFO``/``ISMP`` entry as ``format.tags.timecode`` — which is where a
  timecode string lives in a RIFF file, and why :func:`info_payload` exists.
* It reports **nothing at all** for ``iXML`` or for a private chunk. That asymmetry is
  the entire justification for the generic RIFF walk (OQ-005).
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "BEXT_FIXED_BYTES",
    "RF64_SENTINEL",
    "BroadcastMetadata",
    "ExtraChunk",
    "SampleFormatName",
    "bext_payload",
    "chunk",
    "info_payload",
    "pack_samples",
    "quantize",
    "write_wav",
]

#: What a fixture may be written in. FFprobe's own names, so a fixture spec and the
#: manifest it produces read alike.
SampleFormatName = Literal["pcm_f32le", "pcm_s24le", "pcm_s16le"]

#: A 32-bit size field set to this means "the real size is 64-bit, look in ``ds64``".
RF64_SENTINEL: Final = 0xFFFFFFFF

#: Size of a ``bext`` payload before any coding history: 256 + 32 + 32 + 10 + 8 + 8 + 2
#: + 64 + 10 + 180. Version 2 adds the loudness fields inside that reserved area, so
#: the fixed size is the same for versions 1 and 2.
BEXT_FIXED_BYTES: Final = 602

_UNSET_LOUDNESS: Final = 0x7FFF

#: The formats a fixture can be written in, named the way FFprobe names them, as
#: ``(fmt tag, bits)``. Deliberately **not** imported from
#: :mod:`dnd_audio.timeline.pcm`, for the reason stated above about
#: :mod:`~dnd_audio.inspection.riff`: a reader test that got its header fields from the
#: reader's own table would be exercising the table rather than the reader. The two agree
#: on FFprobe's vocabulary and on nothing else.
_FORMATS: Final[dict[SampleFormatName, tuple[int, int]]] = {
    "pcm_f32le": (3, 32),
    "pcm_s24le": (1, 24),
    "pcm_s16le": (1, 16),
}


@dataclass(frozen=True, slots=True)
class BroadcastMetadata:
    """The ``bext`` fields that matter to the timecode strategy chain.

    ``time_reference`` is a sample count since midnight **at the file's own sample
    rate**, per EBU Tech 3285. It is an integer here and stays one all the way through
    inspection: rounding it through a frame count is exactly what INV-04 forbids.
    """

    time_reference: int
    origination_date: dt.date | None = None
    origination_time: dt.time | None = None
    description: str = ""
    originator: str = ""
    originator_reference: str = ""
    version: int = 2
    coding_history: str = ""

    def __post_init__(self) -> None:
        if self.time_reference < 0:
            message = f"time_reference must not be negative, got {self.time_reference}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ExtraChunk:
    """A chunk written verbatim, whether or not anything can interpret it."""

    chunk_id: bytes
    payload: bytes

    def __post_init__(self) -> None:
        if len(self.chunk_id) != 4:
            message = f"a RIFF chunk id is four bytes, got {self.chunk_id!r}"
            raise ValueError(message)


def chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """One chunk: four-byte id, little-endian 32-bit size, payload, pad to even.

    The pad byte is not counted in the size field. Getting that wrong shifts every
    subsequent chunk offset by one, which is why the parser has an odd-size test.
    """
    if len(chunk_id) != 4:
        message = f"a RIFF chunk id is four bytes, got {chunk_id!r}"
        raise ValueError(message)
    body = chunk_id + struct.pack("<I", len(payload)) + payload
    return body + b"\x00" if len(payload) % 2 else body


def _fixed(text: str, size: int) -> bytes:
    """ASCII, NUL-padded to exactly ``size`` bytes, truncated if too long."""
    return text.encode("ascii", errors="replace")[:size].ljust(size, b"\x00")


def bext_payload(meta: BroadcastMetadata) -> bytes:
    """Serialize a ``bext`` chunk payload."""
    date_text = meta.origination_date.strftime("%Y-%m-%d") if meta.origination_date else ""
    time_text = meta.origination_time.strftime("%H:%M:%S") if meta.origination_time else ""
    loudness = struct.pack("<5h", *([_UNSET_LOUDNESS] * 5))
    payload = (
        _fixed(meta.description, 256)
        + _fixed(meta.originator, 32)
        + _fixed(meta.originator_reference, 32)
        + _fixed(date_text, 10)
        + _fixed(time_text, 8)
        + struct.pack("<Q", meta.time_reference)
        + struct.pack("<H", meta.version)
        + bytes(64)  # UMID
        + loudness
        + bytes(180)  # reserved
        + meta.coding_history.encode("ascii", errors="replace")
    )
    if len(payload) < BEXT_FIXED_BYTES:  # pragma: no cover - arithmetic above forbids it
        message = f"bext payload is {len(payload)} bytes, expected at least {BEXT_FIXED_BYTES}"
        raise ValueError(message)
    return payload


def info_payload(entries: dict[bytes, str]) -> bytes:
    """Serialize a ``LIST``/``INFO`` payload from ``{four-byte id: text}``.

    ``ISMP`` is the SMPTE time code field. FFmpeg surfaces it as the ``timecode``
    format tag, which is what makes the chain's fallback strategy reachable.

    Entries are emitted in sorted id order so a fixture's bytes do not depend on dict
    construction order.
    """
    payload = b"INFO"
    for entry_id in sorted(entries):
        if len(entry_id) != 4:
            message = f"an INFO entry id is four bytes, got {entry_id!r}"
            raise ValueError(message)
        # NUL-terminated, as RIFF INFO strings are.
        payload += chunk(entry_id, entries[entry_id].encode("ascii", errors="replace") + b"\x00")
    return payload


def quantize(
    samples: npt.NDArray[np.float32], sample_format: SampleFormatName
) -> npt.NDArray[np.int32]:
    """Float32 audio as the integers a recorder set to that width would have written.

    Round-half-away-from-zero and a clamp at the positive edge, which is what a converter
    does: `2**(bits-1)` values exist below zero and only `2**(bits-1) - 1` above it, so
    full-scale positive is the one sample that cannot be represented symmetrically.
    """
    scale = 2 ** (_FORMATS[sample_format][1] - 1)
    scaled = np.rint(np.asarray(samples, dtype=np.float64) * scale)
    clamped: npt.NDArray[np.int32] = np.clip(scaled, -scale, scale - 1).astype(np.int32)
    return clamped


def pack_samples(values: npt.NDArray[np.int32], sample_format: SampleFormatName) -> bytes:
    """Little-endian packed bytes for one integer sample format.

    24-bit is written by dropping the *high* byte of each little-endian int32 — the
    inverse of the reader's widen-and-shift, arrived at independently so that a
    round-trip test measures agreement rather than a shared helper.
    """
    bits = _FORMATS[sample_format][1]
    limit = 2 ** (bits - 1)
    if values.size and (int(values.min()) < -limit or int(values.max()) > limit - 1):
        message = (
            f"{sample_format} holds [{-limit}, {limit - 1}]; got "
            f"[{int(values.min())}, {int(values.max())}]"
        )
        raise ValueError(message)
    wide = np.asarray(values, dtype="<i4").reshape(-1, 1).view(np.uint8)
    return wide[:, : bits // 8].tobytes()


def write_wav(
    path: Path,
    samples: npt.NDArray[np.float32] | npt.NDArray[np.int32],
    *,
    sample_rate: int,
    sample_format: SampleFormatName = "pcm_f32le",
    broadcast: BroadcastMetadata | None = None,
    info: dict[bytes, str] | None = None,
    extra: tuple[ExtraChunk, ...] = (),
    trailing: tuple[ExtraChunk, ...] = (),
    rf64: bool = False,
) -> int:
    """Write one mono WAV. Returns the number of bytes written.

    Args:
        samples: Mono. Float32 for ``FLOAT32``; the *integers themselves* for an integer
            format, so a fixture that cares about exact values can state them and
            :func:`quantize` converts when it does not. The fixtures are mono because each
            transmitter records one channel.
        sample_rate: Hertz.
        sample_format: What to write. A settings mismatch across kits produces a 24-bit
            `orig` on real hardware (ADR-0030), so fixtures have to be able to carry one.
        broadcast: Written as a ``bext`` chunk when given.
        info: Written as a ``LIST``/``INFO`` chunk when given.
        extra: Chunks written between the metadata and ``data``.
        trailing: Chunks written *after* ``data``. Some recorders append; a parser that
            stops at ``data`` would silently miss them.
        rf64: Write an RF64 header with a ``ds64`` chunk and a sentinel ``data`` size.
            The fixtures that use it are small — the point is to exercise the 64-bit
            path without a four-gigabyte file.
    """
    if samples.ndim != 1:
        message = f"fixture audio must be mono (1-D), got shape {samples.shape}"
        raise ValueError(message)
    if sample_rate <= 0:
        message = f"sample_rate must be positive, got {sample_rate}"
        raise ValueError(message)

    tag, bits = _FORMATS[sample_format]
    if sample_format == "pcm_f32le":
        frames = np.asarray(samples, dtype="<f4").tobytes()
    else:
        frames = pack_samples(np.asarray(samples, dtype=np.int32), sample_format)
    block_align = bits // 8
    fmt = struct.pack("<HHIIHH", tag, 1, sample_rate, sample_rate * block_align, block_align, bits)

    before_data = [chunk(b"fmt ", fmt)]
    if broadcast is not None:
        before_data.append(chunk(b"bext", bext_payload(broadcast)))
    if info:
        before_data.append(chunk(b"LIST", info_payload(info)))
    before_data.extend(chunk(item.chunk_id, item.payload) for item in extra)
    after_data = [chunk(item.chunk_id, item.payload) for item in trailing]

    data = _data_chunk(frames, rf64=rf64)
    body = b"".join(before_data) + data + b"".join(after_data)

    if rf64:
        sample_count = int(samples.shape[0])
        # ds64 must be the first chunk, and its riffSize covers everything after the
        # eight-byte RF64 header. Written twice because the first pass is what makes
        # that length knowable.
        placeholder = chunk(b"ds64", _ds64_payload(0, len(frames), sample_count))
        riff_size = len(b"WAVE") + len(placeholder) + len(body)
        ds64 = chunk(b"ds64", _ds64_payload(riff_size, len(frames), sample_count))
        blob = b"RF64" + struct.pack("<I", RF64_SENTINEL) + b"WAVE" + ds64 + body
    else:
        blob = b"RIFF" + struct.pack("<I", len(b"WAVE") + len(body)) + b"WAVE" + body

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return len(blob)


def _data_chunk(frames: bytes, *, rf64: bool) -> bytes:
    declared = RF64_SENTINEL if rf64 else len(frames)
    body = b"data" + struct.pack("<I", declared) + frames
    return body + b"\x00" if len(frames) % 2 else body


def _ds64_payload(riff_size: int, data_size: int, sample_count: int) -> bytes:
    """``ds64``: the 64-bit sizes the 32-bit header fields cannot hold."""
    return struct.pack("<QQQI", riff_size, data_size, sample_count, 0)
