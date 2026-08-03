"""The lossless mix intermediate, addressed by everything that could change its samples.

The spec asks for "a lossless mix intermediate in `work/` for debugging/cache reuse, not as
a required user-facing deliverable". This is the cache-reuse half, and it follows M2's
derivative cache exactly — same publication order, same completeness rules, same reason.

**What the identity carries, and what it deliberately does not.**

* **The timeline's own sha256 and the graph's `attribution_cache_key`.** Between them these
  are downstream of every placement and activity setting there is: a parser fix that moves a
  chunk changes the first, a bleed threshold changes the second. Restating those sections as
  a configuration projection as well would put the same facts in two places that could
  disagree — the argument `_FIELD_SCOPES` already makes for `asr` and `transcript`.
* **The `mix` stage projection**, which under ADR-0023 is `mix.envelope` and nothing else.
  The loudness target, the bitrate, the tolerances and the retry budget sit *after* the render
  boundary: they change the MP3, which is regenerated every run and never cached, so keying
  the intermediate on them would re-mix six four-hour tracks to change a number that cannot
  reach a single sample of it.
* **The track order and each track's level correction**, because the correction is applied to
  the audio here rather than being an encode parameter, and because the share depends on how
  many tracks are being divided between.
* **The mix semantics version and NumPy's**, for the reason every other cache in this project
  carries them.

**Not the FFmpeg version.** Nothing external produces this file; it is written by
`timeline.wavwrite` from arrays this package computed. FFmpeg touches only the MP3.

Publication is the same three steps in the same fixed order — write the audio
temp-then-rename, stage the sidecar in memory, commit only once the caller has re-verified
INV-01 — and for the same reason: the mix is the one stage after inspection that reads
*source* audio, so a run that discovers a source changed must not leave behind an entry keyed
on the bytes it read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.determinism import canonical_json, sha256_bytes, write_json_atomic
from dnd_audio.mix import MIX_CACHE_DIRNAME, MIX_SEMANTICS_VERSION
from dnd_audio.mix.levels import LevelCorrections

__all__ = [
    "CACHE_RECORD_VERSION",
    "CachedMix",
    "MixCache",
    "mix_identity",
    "mix_identity_document",
    "mix_relative_path",
]

#: The shape of a sidecar. Separate from the semantics version: one is "what we computed",
#: the other is "how we wrote it down".
CACHE_RECORD_VERSION: Final = 1


def mix_identity_document(
    graph: ActivityGraph,
    *,
    stage_config_hash: str,
    corrections: LevelCorrections,
    track_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Everything the key is derived from, before it is hashed.

    Separate from :func:`mix_identity` so a test can assert *which components are present*
    rather than only that some change produced some different hash. M2's closeout is blunt
    about why: a key that changes for the right reason in one test can still be missing a
    component, and the missing one is always the one that matters later.
    """
    by_id = {item.track_id: item for item in corrections.corrections}
    return {
        "attribution_cache_key": graph.attribution_cache_key,
        "cache_record_version": CACHE_RECORD_VERSION,
        "config_hash": stage_config_hash,
        "corrections_mb": [by_id[track_id].correction_mb for track_id in track_ids],
        "duration_samples": graph.duration_samples,
        "mix_semantics_version": MIX_SEMANTICS_VERSION,
        "numpy_version": np.__version__,
        "sample_rate": graph.sample_rate,
        "timeline_sha256": graph.timeline_sha256,
        "track_ids": list(track_ids),
    }


def mix_identity(
    graph: ActivityGraph,
    *,
    stage_config_hash: str,
    corrections: LevelCorrections,
    track_ids: tuple[str, ...],
) -> str:
    """The full cache identity of one session's mix intermediate (INV-08)."""
    return sha256_bytes(
        canonical_json(
            mix_identity_document(
                graph,
                stage_config_hash=stage_config_hash,
                corrections=corrections,
                track_ids=track_ids,
            )
        ).encode("utf-8")
    )


def mix_relative_path(key: str) -> str:
    """Where the intermediate lives, session-relative and named by its identity.

    Content-addressed rather than named for the session, for the reason the derivatives are:
    two versions can coexist while one is being rebuilt, and nothing ever overwrites an
    artifact something else might be reading.
    """
    return f"{MIX_CACHE_DIRNAME}/{key}.wav"


@dataclass(frozen=True, slots=True)
class CachedMix:
    """A mix intermediate that is present and complete."""

    key: str
    relative_path: str
    sample_rate: int
    n_samples: int
    size_bytes: int


@dataclass
class MixCache:
    """Reads and publishes the cached intermediate for one session.

    Args:
        read_enabled: Set false to re-mix without deleting anything. Writes still happen —
            `--no-cache` is for distrusting what is stored, and making it also refuse to
            store turns "one slow run" into "every run slow".
    """

    session_dir: Path
    read_enabled: bool = True
    hits: int = 0
    misses: int = 0
    _staged: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def get(self, key: str, *, expected_samples: int | None = None) -> CachedMix | None:
        """The complete artifact for ``key``, or ``None``. Counts the hit or the miss.

        Incompleteness is a miss rather than an error, in every form it takes: no sidecar, an
        unparsable one, one that disagrees with itself, a sidecar whose audio is missing, or
        audio whose size does not match what the sidecar recorded. **The size check is what
        makes "an incomplete entry is never a hit" true rather than merely intended** — a
        truncated float32 WAV reads as a shorter mix, which is silence at the end of a
        session and nothing an operator would attribute to a cache.
        """
        if not self.read_enabled:
            self.misses += 1
            return None

        record = self._read_sidecar(key)
        wrong_length = (
            expected_samples is not None
            and record is not None
            and record.n_samples != expected_samples
        )
        if record is None or wrong_length:
            self.misses += 1
            return None

        try:
            size = (self.session_dir / record.relative_path).stat().st_size
        except OSError:
            self.misses += 1
            return None
        if size != record.size_bytes:
            self.misses += 1
            return None

        self.hits += 1
        return record

    def publish(self, key: str, *, sample_rate: int, n_samples: int) -> CachedMix:
        """Stage the record that will make ``key`` findable.

        Called *after* the audio has been renamed into place, never before: the sidecar is
        what makes an entry a hit, so writing it first would advertise a file that does not
        exist yet. Nothing reaches disk until :meth:`commit`, which the runner calls only once
        INV-01 has been re-verified.
        """
        relative = mix_relative_path(key)
        record = CachedMix(
            key=key,
            relative_path=relative,
            sample_rate=sample_rate,
            n_samples=n_samples,
            size_bytes=(self.session_dir / relative).stat().st_size,
        )
        self._staged[key] = {
            "cache_record_version": CACHE_RECORD_VERSION,
            "key": record.key,
            "n_samples": record.n_samples,
            "relative_path": record.relative_path,
            "sample_rate": record.sample_rate,
            "size_bytes": record.size_bytes,
        }
        return record

    def commit(self) -> int:
        """Write every staged sidecar atomically. Returns how many were written."""
        written = 0
        for key, payload in sorted(self._staged.items()):
            write_json_atomic(self._sidecar_path(key), payload)
            written += 1
        self._staged.clear()
        return written

    def discard(self) -> None:
        """Drop everything staged. The audio remains, and without a sidecar it is inert."""
        self._staged.clear()

    def audio_path(self, key: str) -> Path:
        return self.session_dir / mix_relative_path(key)

    def _sidecar_path(self, key: str) -> Path:
        return self.session_dir / f"{MIX_CACHE_DIRNAME}/{key}.json"

    def _read_sidecar(self, key: str) -> CachedMix | None:
        try:
            raw = self._sidecar_path(key).read_bytes()
        except OSError:
            return None
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(document, dict) or document.get("key") != key:
            return None
        try:
            record = CachedMix(
                key=key,
                relative_path=str(document["relative_path"]),
                sample_rate=int(document["sample_rate"]),
                n_samples=int(document["n_samples"]),
                size_bytes=int(document["size_bytes"]),
            )
            version = int(document["cache_record_version"])
        except (KeyError, TypeError, ValueError):
            return None

        # A sidecar that disagrees with itself is not a usable entry: the caller reads the
        # *canonical* path, so a record naming a different file would grant a hit on the
        # strength of a file nothing goes on to read.
        if version != CACHE_RECORD_VERSION or record.relative_path != mix_relative_path(key):
            return None
        return record
