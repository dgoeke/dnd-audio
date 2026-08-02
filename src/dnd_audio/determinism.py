"""The mechanisms INV-02, INV-04, and INV-07 are enforced with.

Three properties, one module, so that no later milestone has to reinvent them:

* **Byte-stability** (INV-02). :func:`canonical_json` is the only JSON serializer this
  project uses for an artifact. Keys are sorted, encoding is UTF-8, and non-finite
  floats are rejected rather than emitted as the invalid JSON tokens ``NaN`` and
  ``Infinity``.
* **Exact time** (INV-04). Timeline arithmetic stays in :class:`~fractions.Fraction`.
  :func:`to_milliseconds` is the one quantizer, with an explicit tie rule, and
  :func:`public_seconds` is the one place a float is produced.
* **Bounded memory** (INV-07). :func:`sha256_file` streams. Nothing here reads a whole
  file into memory, because from M1 onward these helpers are pointed at multi-gigabyte
  recordings.

Writes go through :func:`write_atomic`: temp file in the destination directory, fsync,
rename, fsync the directory. A reader either sees the previous bytes or the new ones,
never a truncated file, which is what INV-13 means by "written atomically even on
partial failure".
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Protocol

__all__ = [
    "HASH_CHUNK_BYTES",
    "BinaryReader",
    "canonical_json",
    "public_seconds",
    "sha256_bytes",
    "sha256_file",
    "sha256_stream",
    "to_milliseconds",
    "write_atomic",
    "write_json_atomic",
]

#: Bytes read per hashing iteration. Small enough that a session-length WAV never
#: lands in memory, large enough that the syscall overhead is irrelevant (INV-07).
HASH_CHUNK_BYTES: Final = 1 << 20

_JSON_INDENT: Final = 2


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to this project's one canonical JSON form.

    Sorted keys, two-space indent, UTF-8 text kept as itself rather than escaped, and a
    trailing newline so the files behave in a text editor and in ``git diff``.

    Raises:
        ValueError: if the value contains NaN or an infinity. Those are not
            representable in JSON, and silently emitting the JavaScript spellings would
            produce an artifact that fails to parse elsewhere.
    """
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=_JSON_INDENT,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    )


def write_atomic(path: Path, data: str | bytes) -> None:
    """Write ``data`` to ``path`` so that a concurrent reader never sees a partial file.

    The temporary file is created in the destination directory — a rename across
    filesystems is not atomic — and is removed if anything fails before the rename.

    ``data`` is held in memory in full, which is right for the JSON artifacts this is
    built for and wrong for audio. INV-07 forbids materializing a session-length
    waveform, so M2's working-audio writes need their own streamed path rather than
    this one; they are not text and do not need canonical serialization anyway.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    handle_fd, temp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates 0600. An artifact is not a secret, and inheriting the
        # temp-file mode would silently make transcripts unreadable to anyone but the
        # invoking user.
        temp_path.chmod(_default_file_mode())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    _fsync_directory(directory)


def write_json_atomic(path: Path, value: Any) -> None:
    """Canonically serialize ``value`` and write it atomically."""
    write_atomic(path, canonical_json(value))


def _default_file_mode() -> int:
    """The mode an ordinary ``open(path, "w")`` would have produced.

    Reading the umask means temporarily setting it; this project is a single-threaded
    CLI, so the window is not a hazard. Respecting it rather than hardcoding 0644 keeps
    a deliberately restrictive umask meaningful for session artifacts.
    """
    current = os.umask(0o022)
    os.umask(current)
    return 0o666 & ~current


def _fsync_directory(directory: Path) -> None:
    """Persist the rename itself, not just the bytes it points at.

    Best effort: some filesystems reject opening a directory for fsync, and failing a
    whole run over that would be worse than the durability it buys.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class BinaryReader(Protocol):
    """Anything that yields bytes a chunk at a time.

    Narrower than ``IO[bytes]`` on purpose: hashing needs one method, and a test that
    instruments the read size should not have to implement a whole file object to
    supply it.
    """

    def read(self, size: int = -1, /) -> bytes: ...


def sha256_stream(stream: BinaryReader, *, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Hash a binary stream in bounded chunks (INV-07).

    Split out from :func:`sha256_file` so a test can hand it an instrumented stream and
    prove the read size is actually bounded, rather than trusting the implementation.
    """
    if chunk_bytes <= 0:
        message = f"chunk_bytes must be positive, got {chunk_bytes}"
        raise ValueError(message)

    digest = hashlib.sha256()
    while chunk := stream.read(chunk_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Hash a file's contents without reading it into memory (INV-07)."""
    with path.open("rb") as handle:
        return sha256_stream(handle, chunk_bytes=chunk_bytes)


def sha256_bytes(data: bytes) -> str:
    """Hash an in-memory value. For file contents use :func:`sha256_file`."""
    return hashlib.sha256(data).hexdigest()


def to_milliseconds(seconds: Fraction | int) -> int:
    """Quantize an exact time to integer milliseconds, halves away from zero.

    The tie rule is stated rather than inherited: Python's :func:`round` is
    banker's rounding, so ``round(0.0005 * 1000)`` and ``round(0.0015 * 1000)`` disagree
    about which way a half goes. An artifact whose timestamps depend on that is not
    byte-stable in any useful sense (INV-02).

    Takes a :class:`~fractions.Fraction` because INV-04 forbids accumulating floats;
    the conversion is exact and happens once, at the boundary.
    """
    scaled = Fraction(seconds) * 1000
    magnitude, remainder = divmod(abs(scaled.numerator), scaled.denominator)
    if 2 * remainder >= scaled.denominator:
        magnitude += 1
    return -magnitude if scaled.numerator < 0 else magnitude


def public_seconds(seconds: Fraction | int) -> float:
    """Convert an exact time to the float seconds a public artifact serializes.

    The single float-producing conversion in the project (INV-04). It is defined in
    terms of :func:`to_milliseconds`, so the value is always an exact number of
    milliseconds and its shortest repr — what :mod:`json` writes — round-trips.
    """
    return to_milliseconds(seconds) / 1000
