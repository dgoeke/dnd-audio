"""Verified off-site backup of a session's immutable sources (M7a).

The one thing later software cannot repair is a lost original recording. Six milestones
of pipeline are worth nothing against a failed disk, so this package exists to put a
byte-exact copy somewhere else and — the part that makes it a backup rather than a belief
— to prove that copy restores before claiming it does.

**This is the only part of the project permitted to send audio anywhere** (INV-06 as
amended, ADR-0035). The exception is narrow in every direction: only immutable source
files, only to the owner's own private bucket, only from an explicit ``archive``
subcommand, never invoked by ``process``, never publishing an output, and never deleting
anything. Nothing at the far end processes the audio, which is the distinction the
invariant now turns on.

Four rules shape everything here, and each has an ADR:

* **The source set is its own enumeration** (ADR-0036). Not the inspection manifest, which
  inventories candidate *audio* and would silently drop the notes file beside it; and not
  ``raw_guard.snapshot()``, whose ``is_file()`` follows a leaf symlink — correct for
  comparing a tree against itself, and a way to upload ``~/.ssh/id_ed25519`` under an
  innocent key.
* **Archive v1 is frozen** (ADR-0037). One single-threaded zstd recipe, pinned through
  ``uv.lock`` rather than through whatever ``zstd`` is on ``PATH``, versioned in the key
  prefix. A changed recipe is archive v2, never different bytes at an existing key.
* **Nothing is committed until it has been read back** (ADR-0038). Compress, decompress
  locally, upload, download the whole object, decompress again — then, and only then, PUT
  the manifest that marks the session committed. An ETag is not a content checksum.
* **Three words for "checked", never merged** (ADR-0039). ``committed`` is history,
  ``previously_verified_at_commit`` is history with a receipt, and ``verified`` means this
  operation just read those bytes. ``status`` may never say the third.
"""

from __future__ import annotations

from typing import Final

from dnd_audio.errors import DndAudioError

__all__ = [
    "ARCHIVE_MANIFEST_FILENAME",
    "ARCHIVE_OBJECTS_DIRNAME",
    "ARCHIVE_PREFIX",
    "ARCHIVE_VERSION",
    "MAX_OBJECT_KEY_BYTES",
    "ArchiveError",
]


class ArchiveError(DndAudioError):
    """An archive operation cannot proceed, or cannot be trusted.

    Fatal in every case, and deliberately so: this package exists to make one promise —
    that what it stored restores byte-for-byte — and every failure here is that promise
    failing to be establishable. There is no degraded mode where an archive is "probably
    fine", because the whole point is that nobody finds out until the local copy is gone.
    """

    default_code = "archive_failed"


#: Bumped only when the encoding recipe or key layout changes, and then a *new* prefix is
#: written rather than new payloads at old keys (ADR-0037).
ARCHIVE_VERSION: Final = 1

#: Everything this project writes lives under one recognizable, versioned prefix, so a
#: human staring at a bucket during a bad day can tell what they are looking at.
ARCHIVE_PREFIX: Final = f"sessions/archive-v{ARCHIVE_VERSION}"

ARCHIVE_OBJECTS_DIRNAME: Final = "objects"

#: The commit marker, and the only small object a session puts in the bucket. Cold Storage
#: bills anything under 128 KiB as 128 KiB, so one is worth its floor and a dozen sidecars
#: are not.
ARCHIVE_MANIFEST_FILENAME: Final = f"archive-manifest.v{ARCHIVE_VERSION}.json"

#: S3's object-key limit, in **UTF-8 bytes** rather than characters. Percent-encoding a
#: path can triple its length, so this is a bound a real session can reach; an entry whose
#: key would exceed it is refused with a diagnostic rather than truncated into a collision
#: (ADR-0036).
MAX_OBJECT_KEY_BYTES: Final = 1024
