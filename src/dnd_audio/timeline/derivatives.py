"""Cached working audio, addressed by everything that could change its contents (INV-08).

A derivative is regenerable, so the only thing that can go wrong is serving a stale one —
and a stale 16 kHz track is not obviously wrong when you look at it. It has the right
length and the right speech in it; it is simply aligned to a timeline that has since moved.
Every VAD span and every word timestamp built on it would then be off by a constant nobody
would attribute to a cache.

So the identity is deliberately broad. It carries:

* **the track's segment map**, canonically serialized — every source path, hash, offset,
  and placed position. This is the component the first draft omitted, and it is the one
  that matters: a parser fix in M1 moves a chunk without changing a single source byte;
* **the resolved configuration hash**, which carries the frame rate, the origin, and every
  recovery override;
* **both semantics versions**. `INSPECTION_SEMANTICS_VERSION` covers the code that produced
  the timing evidence, `TIMELINE_SEMANTICS_VERSION` the code that placed it;
* **the filter's identity** — the SHA-256 of the checked-in coefficient file — for anything
  resampled, and nothing for a 48 kHz copy, which passes through no filter;
* **NumPy and SciPy versions**, because they are external implementations whose upgrade can
  legitimately change the samples;
* **the target rate**, so the 16 kHz and 48 kHz artifacts of one track cannot collide.

Publication is two steps in a fixed order: the audio is written temp-then-rename, and only
then is a sidecar written naming it. An entry is a hit only when the sidecar parses *and*
the audio it names exists at the recorded size — so a crash between the two leaves a miss,
never a truncated file dressed as a hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import scipy

from dnd_audio.artifacts.timeline import TimelineTrack
from dnd_audio.determinism import canonical_json, sha256_bytes, write_json_atomic
from dnd_audio.inspection import INSPECTION_SEMANTICS_VERSION
from dnd_audio.timeline import TIMELINE_DIRNAME, TIMELINE_SEMANTICS_VERSION

__all__ = [
    "CACHE_RECORD_VERSION",
    "CachedDerivative",
    "DerivativeCache",
    "derivative_identity",
    "derivative_relative_path",
]

#: The shape of a sidecar. Separate from the semantics version: one is "what we computed",
#: the other is "how we wrote it down".
CACHE_RECORD_VERSION: Final = 1


def derivative_identity(
    track: TimelineTrack,
    *,
    config_hash: str,
    target_rate: int,
    filter_identity: str | None,
) -> str:
    """The full cache identity of one track's derived audio. See the module docstring."""
    identity: dict[str, Any] = {
        "cache_record_version": CACHE_RECORD_VERSION,
        "config_hash": config_hash,
        "filter_identity": filter_identity,
        "inspection_semantics_version": INSPECTION_SEMANTICS_VERSION,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "segments": [segment.model_dump(mode="json") for segment in track.segments],
        "target_rate": target_rate,
        "timeline_semantics_version": TIMELINE_SEMANTICS_VERSION,
        "track_id": track.track_id,
        "track_extent": [track.start_sample, track.end_sample],
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def derivative_relative_path(target_rate: int, key: str) -> str:
    """Where a derivative lives, session-relative and named by its identity.

    Content-addressed rather than named for the track: two timelines of the same session
    can coexist while one is being rebuilt, and nothing ever overwrites an artifact that
    something else might be reading.
    """
    return f"{TIMELINE_DIRNAME}/{target_rate}/{key}.wav"


@dataclass(frozen=True, slots=True)
class CachedDerivative:
    """A derivative that is present and complete."""

    key: str
    relative_path: str
    sample_rate: int
    n_samples: int
    size_bytes: int


@dataclass
class DerivativeCache:
    """Reads and publishes cached working audio for one session.

    Args:
        read_enabled: Set false to rebuild without deleting anything. Writes still happen,
            for the reason M1's inspection cache gives: `--no-cache` is for distrusting
            what is stored, and making it also refuse to store turns "one slow run" into
            "every run slow".
    """

    session_dir: Path
    read_enabled: bool = True
    hits: int = 0
    misses: int = 0

    def get(self, key: str, target_rate: int) -> CachedDerivative | None:
        """The complete artifact for ``key``, or ``None``. Counts the hit or the miss.

        Incompleteness is a miss rather than an error, in every form it takes: no sidecar,
        an unparsable one, a sidecar whose audio is missing, or audio whose size does not
        match what the sidecar recorded. A corrupted cache should cost time, not a session
        — and the size check is what makes "an incomplete entry is never a hit" true rather
        than merely intended.
        """
        if not self.read_enabled:
            self.misses += 1
            return None

        record = self._read_sidecar(key, target_rate)
        if record is None:
            self.misses += 1
            return None

        audio = self.session_dir / record.relative_path
        try:
            size = audio.stat().st_size
        except OSError:
            self.misses += 1
            return None
        if size != record.size_bytes:
            self.misses += 1
            return None

        self.hits += 1
        return record

    def publish(self, key: str, *, target_rate: int, n_samples: int) -> CachedDerivative:
        """Record that the audio for ``key`` is now complete.

        Called *after* the audio has been renamed into place, never before: the sidecar is
        what makes an entry a hit, so writing it first would advertise a file that does not
        exist yet.
        """
        relative = derivative_relative_path(target_rate, key)
        audio = self.session_dir / relative
        record = CachedDerivative(
            key=key,
            relative_path=relative,
            sample_rate=target_rate,
            n_samples=n_samples,
            size_bytes=audio.stat().st_size,
        )
        write_json_atomic(
            self._sidecar_path(key, target_rate),
            {
                "cache_record_version": CACHE_RECORD_VERSION,
                "key": record.key,
                "n_samples": record.n_samples,
                "relative_path": record.relative_path,
                "sample_rate": record.sample_rate,
                "size_bytes": record.size_bytes,
            },
        )
        return record

    def audio_path(self, key: str, target_rate: int) -> Path:
        return self.session_dir / derivative_relative_path(target_rate, key)

    def _sidecar_path(self, key: str, target_rate: int) -> Path:
        return self.session_dir / f"{TIMELINE_DIRNAME}/{target_rate}/{key}.json"

    def _read_sidecar(self, key: str, target_rate: int) -> CachedDerivative | None:
        try:
            raw = self._sidecar_path(key, target_rate).read_bytes()
        except OSError:
            return None
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(document, dict) or document.get("key") != key:
            return None
        try:
            return CachedDerivative(
                key=key,
                relative_path=str(document["relative_path"]),
                sample_rate=int(document["sample_rate"]),
                n_samples=int(document["n_samples"]),
                size_bytes=int(document["size_bytes"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
