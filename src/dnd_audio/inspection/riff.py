"""A generic RIFF/RF64 chunk inventory that does not depend on FFprobe.

The spec forbids assuming FFprobe exposes unknown vendor or iXML chunks as tags, and
that is not a hypothetical caution: against FFmpeg 8.0, a file carrying both an ``iXML``
chunk and a four-byte-named private chunk produces `-show_format -show_streams` output
mentioning neither. If a DJI recorder ever puts timing in a private chunk, FFprobe is
not the thing that will find it (OQ-005).

So this walks the container itself. It reads structure, never audio: the `data` payload
is inventoried by offset and size and is not read, which is both an INV-07 requirement
and the reason inspecting a four-hour session costs the same as inspecting a two-minute
one.

Three contracts worth stating, because each has an obvious wrong version:

* **``offset`` is the offset of the chunk header**, not of its payload. The payload
  begins eight bytes later.
* **A chunk's ``sha256`` covers its complete payload.** The retention cap bounds how
  much text is *kept*, never how much is hashed — a prefix hash presented as a chunk
  hash would be a lie, and M7's archival verification would inherit it.
* **A malformed chunk length stops the walk and is recorded.** It is not repaired.
  Guessing past a bad length turns corruption into plausible-looking metadata, which is
  worse than a short inventory that says why it is short.
"""

from __future__ import annotations

import string
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final

from dnd_audio.determinism import sha256_stream
from dnd_audio.errors import DndAudioError

__all__ = [
    "CHUNK_HEADER_BYTES",
    "MAX_RETAINED_TEXT_BYTES",
    "RF64_SENTINEL",
    "RiffChunk",
    "RiffError",
    "RiffInventory",
    "RiffWarning",
    "read_inventory",
]

#: Four-byte id plus a little-endian 32-bit size.
CHUNK_HEADER_BYTES: Final = 8

#: A 32-bit size of this value means "look the real 64-bit size up in ``ds64``".
RF64_SENTINEL: Final = 0xFFFFFFFF

#: Payloads at or below this are retained as text when they safely decode. Bounded so a
#: pathological file cannot push megabytes of "metadata" into the manifest.
MAX_RETAINED_TEXT_BYTES: Final = 4096

#: Chunks whose payload is audio, and which are therefore inventoried but never read.
#: The file's own SHA-256 already covers these bytes.
_UNHASHED_CHUNKS: Final = frozenset({"data"})

_DS64_FIXED_BYTES: Final = 28
_DS64_TABLE_ENTRY_BYTES: Final = 12
_PRINTABLE: Final = frozenset(string.printable)


class RiffError(DndAudioError):
    """The file is not a RIFF/RF64 container at all.

    Distinct from a warning: a candidate that is not a container cannot be inspected,
    whereas a container with one bad chunk still yields everything before it.
    """


@dataclass(frozen=True, slots=True)
class RiffWarning:
    """Something structurally wrong that did not stop the file being useful."""

    code: str
    message: str
    offset: int


@dataclass(frozen=True, slots=True)
class RiffChunk:
    """One chunk, as found."""

    chunk_id: str
    #: Offset of the chunk **header**. Payload starts at ``offset + CHUNK_HEADER_BYTES``.
    offset: int
    #: Payload size in bytes, resolved through ``ds64`` where RF64 requires it.
    size: int
    #: The ``LIST`` form type this chunk was found inside, e.g. ``"INFO"``. ``None`` at
    #: the top level.
    container: str | None = None
    #: SHA-256 of the **complete** payload. ``None`` for audio payloads, which are
    #: deliberately not read.
    sha256: str | None = None
    #: The payload decoded as text, when it is short enough to retain and safe to
    #: decode. ``None`` when it is binary or over the cap — never a truncated prefix.
    text: str | None = None


@dataclass(frozen=True, slots=True)
class RiffInventory:
    """Everything the walk found, plus everything it could not make sense of."""

    form: str
    form_type: str
    declared_size: int
    file_size: int
    chunks: tuple[RiffChunk, ...] = ()
    warnings: tuple[RiffWarning, ...] = ()
    #: True when the walk stopped early. The chunks before that point are still valid.
    truncated: bool = False

    def find(self, chunk_id: str) -> RiffChunk | None:
        """The first chunk with this id, at any level."""
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)


@dataclass
class _Walk:
    """Mutable state for one file's walk. Not part of the public shape."""

    handle: BinaryIO
    file_size: int
    is_rf64: bool
    max_text_bytes: int
    data_size: int | None = None
    table: dict[str, int] = field(default_factory=dict)
    chunks: list[RiffChunk] = field(default_factory=list)
    warnings: list[RiffWarning] = field(default_factory=list)
    truncated: bool = False


def read_inventory(path: Path, *, max_text_bytes: int = MAX_RETAINED_TEXT_BYTES) -> RiffInventory:
    """Walk ``path``'s chunk structure.

    Raises:
        RiffError: if the file is too short to hold a header, or its magic is neither
            ``RIFF`` nor ``RF64``, or its form type is not ``WAVE``.
    """
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12:
            message = f"{path} is {len(header)} bytes: too short to be a RIFF container"
            raise RiffError(message)

        form = header[:4].decode("ascii", errors="replace")
        declared_size = struct.unpack("<I", header[4:8])[0]
        form_type = header[8:12].decode("ascii", errors="replace")
        if form not in ("RIFF", "RF64"):
            message = f"{path} starts with {form!r}, which is neither RIFF nor RF64"
            raise RiffError(message)
        if form_type != "WAVE":
            message = f"{path} is a {form} container of form {form_type!r}, not WAVE"
            raise RiffError(message)

        walk = _Walk(
            handle=handle,
            file_size=file_size,
            is_rf64=form == "RF64",
            max_text_bytes=max_text_bytes,
        )
        _walk_chunks(walk, start=12, end=file_size, container=None)

    return RiffInventory(
        form=form,
        form_type=form_type,
        declared_size=declared_size,
        file_size=file_size,
        chunks=tuple(walk.chunks),
        warnings=tuple(walk.warnings),
        truncated=walk.truncated,
    )


def _walk_chunks(walk: _Walk, *, start: int, end: int, container: str | None) -> None:
    """Read consecutive chunks in ``[start, end)``, recording each."""
    offset = start
    while offset + CHUNK_HEADER_BYTES <= end:
        walk.handle.seek(offset)
        header = walk.handle.read(CHUNK_HEADER_BYTES)
        if len(header) < CHUNK_HEADER_BYTES:
            _stop(walk, "chunk_header_truncated", "the file ends inside a chunk header", offset)
            return

        raw_id = header[:4]
        if not _is_plausible_id(raw_id):
            _stop(
                walk,
                "chunk_id_not_ascii",
                f"chunk id {raw_id!r} is not printable ASCII, so the walk has lost "
                f"alignment and anything after this point would be invented",
                offset,
            )
            return

        chunk_id = raw_id.decode("ascii")
        declared = struct.unpack("<I", header[4:8])[0]
        size = _resolve_size(walk, chunk_id, declared, offset)
        if size is None:
            return

        payload = offset + CHUNK_HEADER_BYTES
        if payload + size > end:
            _stop(
                walk,
                "chunk_truncated",
                f"chunk {chunk_id!r} claims {size} bytes but only {max(0, end - payload)} remain",
                offset,
            )
            return

        if chunk_id == "ds64" and container is None:
            _read_ds64(walk, payload, size)

        walk.chunks.append(
            RiffChunk(
                chunk_id=chunk_id,
                offset=offset,
                size=size,
                container=container,
                sha256=_hash_payload(walk, chunk_id, payload, size),
                text=_retain_text(walk, chunk_id, payload, size),
            )
        )

        if chunk_id == "LIST" and size >= 4:
            walk.handle.seek(payload)
            form_type = walk.handle.read(4).decode("ascii", errors="replace")
            # One level only. Nested LISTs are legal and vanishingly rare in WAV; a
            # bounded depth is what keeps a crafted file from recursing forever.
            _walk_chunks(walk, start=payload + 4, end=payload + size, container=form_type)

        # The pad byte is not counted in the size field.
        offset = payload + size + (size % 2)


def _resolve_size(walk: _Walk, chunk_id: str, declared: int, offset: int) -> int | None:
    """The chunk's real size, following RF64's indirection. ``None`` stops the walk."""
    if declared != RF64_SENTINEL:
        return declared
    if not walk.is_rf64:
        _stop(
            walk,
            "sentinel_size_in_riff",
            f"chunk {chunk_id!r} uses the RF64 sentinel size in a plain RIFF file",
            offset,
        )
        return None
    if chunk_id == "data" and walk.data_size is not None:
        return walk.data_size
    resolved = walk.table.get(chunk_id)
    if resolved is None:
        _stop(
            walk,
            "rf64_size_unresolved",
            f"chunk {chunk_id!r} defers its size to ds64, which has no entry for it",
            offset,
        )
        return None
    return resolved


def _read_ds64(walk: _Walk, payload: int, size: int) -> None:
    """Record the 64-bit sizes the 32-bit header fields cannot hold."""
    if size < _DS64_FIXED_BYTES:
        walk.warnings.append(
            RiffWarning(
                code="ds64_too_short",
                message=f"ds64 is {size} bytes, below the {_DS64_FIXED_BYTES} it must have",
                offset=payload - CHUNK_HEADER_BYTES,
            )
        )
        return

    walk.handle.seek(payload)
    fixed = walk.handle.read(_DS64_FIXED_BYTES)
    _, data_size, _, table_length = struct.unpack("<QQQI", fixed)
    walk.data_size = data_size

    available = (size - _DS64_FIXED_BYTES) // _DS64_TABLE_ENTRY_BYTES
    for _ in range(min(table_length, available)):
        entry = walk.handle.read(_DS64_TABLE_ENTRY_BYTES)
        entry_id, entry_size = struct.unpack("<4sQ", entry)
        walk.table[entry_id.decode("ascii", errors="replace")] = entry_size

    if table_length > available:
        walk.warnings.append(
            RiffWarning(
                code="ds64_table_truncated",
                message=f"ds64 declares {table_length} table entries but holds {available}",
                offset=payload - CHUNK_HEADER_BYTES,
            )
        )


def _hash_payload(walk: _Walk, chunk_id: str, payload: int, size: int) -> str | None:
    """SHA-256 of the whole payload, streamed. ``None`` for audio."""
    if chunk_id in _UNHASHED_CHUNKS:
        return None
    walk.handle.seek(payload)
    return sha256_stream(_Slice(walk.handle, size))


def _retain_text(walk: _Walk, chunk_id: str, payload: int, size: int) -> str | None:
    """The payload as text, when it is short and safe. Never a truncated prefix."""
    if chunk_id in _UNHASHED_CHUNKS or size == 0 or size > walk.max_text_bytes:
        return None
    walk.handle.seek(payload)
    raw = walk.handle.read(size)
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # RIFF pads strings with NULs; a payload that is *mostly* NUL padding is still text.
    stripped = decoded.rstrip("\x00")
    if not stripped or any(char not in _PRINTABLE for char in stripped):
        return None
    return stripped


def _stop(walk: _Walk, code: str, message: str, offset: int) -> None:
    walk.warnings.append(RiffWarning(code=code, message=message, offset=offset))
    walk.truncated = True


def _is_plausible_id(raw: bytes) -> bool:
    """Chunk ids are printable ASCII. Anything else means the walk lost its place."""
    return all(0x20 <= byte <= 0x7E for byte in raw)


class _Slice:
    """A bounded view of an open file, for streamed hashing (INV-07).

    :func:`~dnd_audio.determinism.sha256_stream` takes anything with ``read``; this
    stops it at the chunk boundary so hashing one chunk never reads the next one, and
    never materializes the payload.
    """

    __slots__ = ("_handle", "_remaining")

    def __init__(self, handle: BinaryIO, size: int) -> None:
        self._handle = handle
        self._remaining = size

    def read(self, size: int = -1, /) -> bytes:
        if self._remaining <= 0:
            return b""
        want = self._remaining if size < 0 else min(size, self._remaining)
        block = self._handle.read(want)
        self._remaining -= len(block)
        return block
