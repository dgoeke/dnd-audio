"""When captured inspection work may be reused, and when it must not be (INV-08).

The identity is the interesting part. It includes everything whose change would change
what inspection produces from the same bytes:

* the source's **path**, because FFprobe echoes the filename into its own output and
  because which recovery override applies is keyed by path — two byte-identical files at
  two paths genuinely have different captures;
* the source's **SHA-256**, the obvious one;
* the **resolved configuration hash**, which carries the frame rate and every override;
* **FFmpeg and FFprobe versions, separately**, because they are separate binaries that
  can be upgraded independently;
* the exact **FFprobe argument vector**;
* :data:`~dnd_audio.inspection.INSPECTION_SEMANTICS_VERSION`, covering every parser in
  this package — probe, naming, RIFF, start-time. A cache identity that varied only the
  RIFF-parser version would happily keep serving the answer a fixed strategy-chain bug
  produced;
* the **manifest schema version**, since the record is stored in that shape.

Entries are staged in memory and written only when the caller commits, and the runner
commits only after it has re-verified that every source is byte-identical to what it
inspected. A file that changed under the pipeline therefore cannot leave a cache entry
describing bytes that no longer exist. Writes go through the atomic writer, so a crash
mid-write leaves the previous entry or nothing — never half an entry that reads as a hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from dnd_audio.artifacts.manifest import MANIFEST_SCHEMA_VERSION
from dnd_audio.determinism import canonical_json, sha256_bytes, write_json_atomic
from dnd_audio.inspection import INSPECTION_SEMANTICS_VERSION
from dnd_audio.inspection.probe import ToolVersions

__all__ = ["CACHE_DIRNAME", "InspectionCache", "cache_key"]

#: Session-relative. Under ``work/`` because it is disposable: deleting it costs a
#: re-probe and nothing else.
CACHE_DIRNAME: Final = "work/cache/inspect"

#: The shape of a stored record. Bumped when the payload's structure changes, which is
#: separate from the semantics version — one is "what we computed", the other is "how we
#: wrote it down".
_CACHE_RECORD_VERSION: Final = 1


def cache_key(
    *,
    relative_path: str,
    source_sha256: str,
    stage_config_hash: str,
    tools: ToolVersions,
    ffprobe_args: tuple[str, ...],
) -> str:
    """The identity of one source's inspection. See the module docstring.

    ``stage_config_hash`` is the ``inspection`` projection of the resolved configuration,
    not the whole of it (ADR-0016): a mix or activity threshold cannot change what FFprobe
    reports about a file, and re-probing every source because one was tuned is cost with no
    corresponding risk. The projection is deliberately generous — placement, roster, and
    recovery all reach inspection — and `tests/test_config.py` proves it changes for every
    section it includes.
    """
    identity = {
        "cache_record_version": _CACHE_RECORD_VERSION,
        "config_hash": stage_config_hash,
        "ffmpeg_version": tools.ffmpeg,
        "ffprobe_args": list(ffprobe_args),
        "ffprobe_version": tools.ffprobe,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "relative_path": relative_path,
        "semantics_version": INSPECTION_SEMANTICS_VERSION,
        "source_sha256": source_sha256,
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


@dataclass
class InspectionCache:
    """A content-addressed store of per-source inspection records.

    Args:
        directory: Where entries live. Created on commit, not on construction, so a
            read-only run leaves no trace.
        read_enabled: Set false to force a full re-inspection. **Writes still happen**:
            `--no-cache` is for distrusting what is stored, and making it also refuse
            to store would turn "one slow run" into "every run slow", which is not what
            anyone reaches for it to do.
    """

    directory: Path
    read_enabled: bool = True
    hits: int = 0
    misses: int = 0
    _staged: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def get(self, key: str) -> dict[str, Any] | None:
        """The record for ``key``, or ``None``. Counts the hit or the miss.

        A record that will not parse is a miss rather than an error: a corrupted cache
        should cost time, not a session.
        """
        if not self.read_enabled:
            self.misses += 1
            return None
        path = self._path(key)
        try:
            raw = path.read_bytes()
        except OSError:
            self.misses += 1
            return None
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            self.misses += 1
            return None
        if not isinstance(record, dict):
            self.misses += 1
            return None
        self.hits += 1
        payload = record.get("payload")
        return payload if isinstance(payload, dict) else None

    def stage(self, key: str, payload: dict[str, Any]) -> None:
        """Hold a record until :meth:`commit`.

        Staging rather than writing immediately is what lets the runner publish only
        after it has proved the sources are unchanged (INV-01): a file altered during a
        run must not leave behind an entry describing bytes that are gone.
        """
        self._staged[key] = payload

    def commit(self) -> int:
        """Write every staged record atomically. Returns how many were written."""
        if not self._staged:
            return 0
        written = 0
        for key, payload in sorted(self._staged.items()):
            write_json_atomic(
                self._path(key),
                {"cache_record_version": _CACHE_RECORD_VERSION, "key": key, "payload": payload},
            )
            written += 1
        self._staged.clear()
        return written

    def discard(self) -> None:
        """Drop everything staged. Used when a run failed after inspecting."""
        self._staged.clear()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"
