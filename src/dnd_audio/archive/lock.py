"""One upload per session at a time, and an honest account of what that does not cover.

Two uploads of one session racing would both PUT the manifest, and the second would win
silently. The obvious remote protocol — HEAD the manifest, PUT it if absent — is not a
compare-and-swap: both writers HEAD, both see nothing, both PUT. Genuine mutual exclusion
needs conditional create (`If-None-Match: *`), and DigitalOcean's documented `PutObject`
does not expose it.

So this is what is actually available: an advisory `flock` between processes on **one
host**. That covers the supported deployment — the single archive machine — and it is
tested with two real processes contending rather than with one process asserting about
itself.

**It does not cover two machines**, and the charter says so as an operator precondition
rather than pretending otherwise (ADR-0038). Writing down a guarantee the platform does not
give would be worse than writing down the real one, because only the real one makes an
operator careful. If a provider with conditional create ever becomes the target, this
module is where the stronger primitive would land.

The lock lives outside every session directory: one inside a source root would itself
violate INV-01, and one under `work/` would vanish when somebody cleared the caches
mid-upload.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.config import state_dir
from dnd_audio.archive.paths import encode_component
from dnd_audio.determinism import sha256_bytes

__all__ = ["lock_path", "single_writer"]


def lock_path(session_id: str, *, directory: Path | None = None) -> Path:
    """Where a session's upload lock lives.

    Named by the session id's **digest**, so a session called `a/b` cannot name a lock in
    another directory and a very long session id cannot produce a filename past the
    255-byte component limit. The readable form is not needed: nothing reads these back,
    and the digest is stable for a given session (M7a code review).
    """
    base = state_dir() if directory is None else directory
    return base / f"{sha256_bytes(encode_component(session_id).encode('utf-8'))}.upload.lock"


@contextmanager
def single_writer(session_id: str, *, directory: Path | None = None) -> Iterator[Path]:
    """Hold the session's upload lock, or refuse immediately.

    Non-blocking on purpose. An upload takes as long as a session is large, so a second
    invocation that blocked would look indistinguishable from one that had hung — and the
    operator would be left deciding whether to interrupt something that might be halfway
    through a commit.

    The lock file is deliberately **not** removed on release. Unlinking it races: another
    process can open the path, then have the file it holds unlinked out from under it, and
    then acquire a lock on an inode nobody else can reach. An empty file per archived
    session is a rounding error against the recordings themselves.

    Raises:
        ArchiveError: with code ``archive_upload_in_progress`` if another process on this
            machine holds it.
    """
    path = lock_path(session_id, directory=directory)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            message = (
                f"another archive upload of session {session_id!r} is already running on "
                f"this machine. Two uploads of one session would race to publish the "
                f"manifest, and the storage provider offers no conditional create to "
                f"resolve that, so the second is refused rather than allowed to win "
                f"silently (ADR-0038). Wait for the first to finish."
            )
            raise ArchiveError(message, code="archive_upload_in_progress") from exc
        try:
            yield path
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)
