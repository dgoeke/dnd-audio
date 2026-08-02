"""Write a float32 WAV without ever holding it in memory (INV-07).

`determinism.write_atomic` is for artifacts: it takes the whole payload as bytes, which is
right for JSON and a direct INV-07 violation for a session-length waveform. M0's closeout
says so explicitly, so the streamed path is here.

Same durability contract, different shape. Samples are appended block by block into a
temporary file in the destination directory, then fsynced and renamed, so a reader sees
either the previous file or a complete one — never a half-written derivative that a cache
would then serve as a hit (INV-08).

**RF64 is chosen from the length, not from a flag.** The total sample count is known before
the first byte is written — the timeline says so — so the container that can hold it is
decided up front rather than discovered at 4 GiB. Above RIFF's 32-bit size field the file
gets an RF64 header and a ``ds64`` chunk; below it, a plain RIFF, because an RF64 file that
did not need to be one is a compatibility problem for no gain.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, Self

import numpy as np
import numpy.typing as npt

from dnd_audio.errors import DndAudioError

__all__ = ["RIFF_SIZE_LIMIT", "WavWriteError", "WavWriter", "needs_rf64"]

_WAVE_FORMAT_IEEE_FLOAT: Final = 3
_RF64_SENTINEL: Final = 0xFFFFFFFF
_BYTES_PER_SAMPLE: Final = 4

#: The largest total a RIFF file's 32-bit size fields can express. A file at or above this
#: needs RF64; the spec names `-rf64 auto` for exactly this reason.
RIFF_SIZE_LIMIT: Final = 0xFFFFFFFF


class WavWriteError(DndAudioError):
    """The stream did not deliver the number of samples it declared."""

    default_code = "wav_write_failed"


def needs_rf64(n_samples: int) -> bool:
    """Whether this length overflows RIFF's 32-bit size fields.

    Judged on the whole file rather than on the ``data`` chunk alone, since the `RIFF`
    size field covers everything after the first eight bytes and overflows first.
    """
    return _riff_size(n_samples) > RIFF_SIZE_LIMIT


class WavWriter:
    """Streams mono float32 samples into a WAV or RF64 file, atomically.

    The declared length is a contract: :meth:`write` refuses to exceed it and closing
    early is an error. A header states how many samples follow, so a file holding fewer
    is malformed in a way that most readers do not notice — they simply return silence at
    the end, which is precisely the failure a cached derivative must never present.
    """

    def __init__(self, path: Path, *, sample_rate: int, n_samples: int) -> None:
        if sample_rate <= 0:
            message = f"sample_rate must be positive, got {sample_rate}"
            raise ValueError(message)
        if n_samples < 0:
            message = f"n_samples must not be negative, got {n_samples}"
            raise ValueError(message)
        self._path = path
        self._sample_rate = sample_rate
        self._declared = n_samples
        self._written = 0
        self._temp: Path | None = None
        self._handle: BinaryIO | None = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp"
        )
        self._temp = Path(name)
        handle: BinaryIO = os.fdopen(descriptor, "wb")
        self._handle = handle
        handle.write(self._header())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle = self._handle
        temp = self._temp
        self._handle = None
        self._temp = None
        if handle is None or temp is None:  # pragma: no cover - __enter__ sets both
            return
        # The temporary file is removed on *every* path that does not publish, including
        # the one where this method itself raises. Keying the cleanup on `exc_type` alone
        # was wrong and left a stray `.wav.tmp` behind whenever a stream came up short —
        # in `work/cache/`, where the next run would find files nothing accounts for.
        published = False
        try:
            if exc_type is None and self._written != self._declared:
                message = (
                    f"{self._path.name} declared {self._declared} samples but received "
                    f"{self._written}. A short file reads as silence at the end rather "
                    f"than as an error, which is exactly what a cache must never serve."
                )
                raise WavWriteError(message)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if exc_type is None:
                temp.chmod(_default_file_mode())
                temp.replace(self._path)
                _fsync_directory(self._path.parent)
                published = True
        finally:
            if not handle.closed:
                handle.close()
            if not published:
                temp.unlink(missing_ok=True)

    def write(self, samples: npt.NDArray[np.float32]) -> None:
        """Append one block. Nothing here retains it."""
        if self._handle is None:
            message = f"{self._path} is not open; use WavWriter as a context manager"
            raise WavWriteError(message)
        if samples.ndim != 1:
            message = f"samples must be mono (1-D), got shape {samples.shape}"
            raise ValueError(message)
        count = int(samples.shape[0])
        if self._written + count > self._declared:
            message = (
                f"{self._path.name} declared {self._declared} samples; writing {count} "
                f"more after {self._written} would exceed it"
            )
            raise WavWriteError(message)
        # Little-endian explicitly: the working path must not depend on the host's byte
        # order, or a derivative built on one machine would be noise on another.
        self._handle.write(np.asarray(samples, dtype="<f4").tobytes())
        self._written += count

    def _header(self) -> bytes:
        data_bytes = self._declared * _BYTES_PER_SAMPLE
        fmt = struct.pack(
            "<HHIIHH",
            _WAVE_FORMAT_IEEE_FLOAT,
            1,
            self._sample_rate,
            self._sample_rate * _BYTES_PER_SAMPLE,
            _BYTES_PER_SAMPLE,
            _BYTES_PER_SAMPLE * 8,
        )
        fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt

        if not needs_rf64(self._declared):
            body_size = len(b"WAVE") + len(fmt_chunk) + 8 + data_bytes
            return (
                b"RIFF"
                + struct.pack("<I", body_size)
                + b"WAVE"
                + fmt_chunk
                + b"data"
                + struct.pack("<I", data_bytes)
            )

        # ds64 must be the first chunk, and its riffSize covers everything after the
        # eight-byte RF64 header.
        ds64_payload = struct.pack("<QQQI", 0, data_bytes, self._declared, 0)
        ds64 = b"ds64" + struct.pack("<I", len(ds64_payload)) + ds64_payload
        riff_size = len(b"WAVE") + len(ds64) + len(fmt_chunk) + 8 + data_bytes
        ds64_payload = struct.pack("<QQQI", riff_size, data_bytes, self._declared, 0)
        ds64 = b"ds64" + struct.pack("<I", len(ds64_payload)) + ds64_payload
        return (
            b"RF64"
            + struct.pack("<I", _RF64_SENTINEL)
            + b"WAVE"
            + ds64
            + fmt_chunk
            + b"data"
            + struct.pack("<I", _RF64_SENTINEL)
        )


def _riff_size(n_samples: int) -> int:
    """What a plain RIFF header's size field would have to hold."""
    fmt_chunk = 8 + 16
    return len(b"WAVE") + fmt_chunk + 8 + n_samples * _BYTES_PER_SAMPLE


def _default_file_mode() -> int:
    """The mode an ordinary ``open(path, "wb")`` would have produced.

    `mkstemp` creates 0600, and inheriting that would make working audio unreadable to
    anyone but the invoking user. Same reasoning as `determinism.write_atomic`.
    """
    current = os.umask(0o022)
    os.umask(current)
    return 0o666 & ~current


def _fsync_directory(directory: Path) -> None:
    """Persist the rename itself, not just the bytes it points at. Best effort."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
