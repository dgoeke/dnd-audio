"""Two caches, because detection and attribution invalidate for different reasons.

The spec defines `activity` as the shared cached operation `transcribe`, `mix`, and
`process` all invoke, so the *whole* thing has to be cacheable — not just the expensive
inference half. But caching it as one unit is wrong in the other direction: raising
`min_correlation` by a hundredth cannot change a single per-frame probability, and re-running
six tracks of inference to find that out is the tuning loop OQ-017 guarantees gets walked
repeatedly.

So there are two identities (ADR-0016):

* **Detection**, per track — the derivative's own cache key, the detector and model identity
  including the *interface* it was called through, and the `detection` projection of the
  configuration. Changing a VAD threshold invalidates this; changing a score weight does not.
* **Attribution**, per session — every detection key that fed it, the `attribution`
  projection, the speech-band filter's identity, and the timeline the candidates are placed
  on. Changing a score weight invalidates this and nothing else.

Both follow M2's publication order exactly, and for the reason its closeout gives: the data
file is written temp-then-rename, the sidecar that makes the entry findable is **staged in
memory**, and staged sidecars are committed only once the caller has re-verified that every
source is byte-identical to what it read (INV-01). A run that correctly *fails* on a changed
source must not leave behind an entry keyed on the bytes it read, because restoring the file
makes that key match again forever.

An entry is a hit only when the sidecar parses, agrees with itself about its own key, path,
and record version, *and* the data it names exists at exactly the recorded size. Each of
those is a way a half-written or hand-edited entry could otherwise pass for a good one — and
the size check in particular is what makes "an incomplete entry is never a hit" true rather
than merely intended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import scipy
from pydantic import ValidationError

from dnd_audio.activity import (
    ACTIVITY_SEMANTICS_VERSION,
    ATTRIBUTION_DIRNAME,
    DETECTION_DIRNAME,
    DETECTOR_FRAME_SAMPLES,
)
from dnd_audio.activity.detect import DetectionResult, SpeechRegion
from dnd_audio.artifacts.activity import ActivityGraph, DetectorIdentity
from dnd_audio.determinism import canonical_json, sha256_bytes, write_atomic, write_json_atomic
from dnd_audio.timeline import TIMELINE_SEMANTICS_VERSION

__all__ = [
    "CACHE_RECORD_VERSION",
    "PROBABILITY_DTYPE",
    "AttributionCache",
    "CachedDetection",
    "DetectionCache",
    "attribution_identity",
    "attribution_identity_document",
    "detection_identity",
    "detection_identity_document",
    "probability_relative_path",
]

#: The shape of a sidecar. Separate from the semantics version: one is "what we computed",
#: the other is "how we wrote it down".
CACHE_RECORD_VERSION: Final = 1

#: Per-frame probabilities on disk: little-endian ``uint16`` per-mille, one per frame.
#: Explicitly little-endian rather than native, so a cache written on one machine is not
#: silently misread on another — the same reason the PCM reader spells out ``<f4``.
PROBABILITY_DTYPE: Final = "<u2"


def detection_identity_document(
    *,
    track_id: str,
    derivative_cache_key: str,
    detector: DetectorIdentity,
    stage_config_hash: str,
) -> dict[str, Any]:
    """Everything one track's detection depends on, before it is hashed.

    Separate from :func:`detection_identity` so a test can assert *which components are
    present* rather than only that some change produced some different hash. A key that
    changes for the right reason in one test can still be missing a component, and the
    missing one is always the one that matters later.

    The derivative's own cache key is the load-bearing entry: it already carries the source
    hashes, the segment map, the filter, and both upstream semantics versions, so a placement
    fix that moves a chunk without changing a source byte invalidates detection too.
    """
    return {
        "activity_semantics_version": ACTIVITY_SEMANTICS_VERSION,
        "cache_record_version": CACHE_RECORD_VERSION,
        "derivative_cache_key": derivative_cache_key,
        "detector": detector.model_dump(mode="json"),
        "frame_samples": DETECTOR_FRAME_SAMPLES,
        "numpy_version": np.__version__,
        "stage_config_hash": stage_config_hash,
        "timeline_semantics_version": TIMELINE_SEMANTICS_VERSION,
        "track_id": track_id,
    }


def detection_identity(
    *,
    track_id: str,
    derivative_cache_key: str,
    detector: DetectorIdentity,
    stage_config_hash: str,
) -> str:
    """The full cache identity of one track's detection (INV-08)."""
    return sha256_bytes(
        canonical_json(
            detection_identity_document(
                track_id=track_id,
                derivative_cache_key=derivative_cache_key,
                detector=detector,
                stage_config_hash=stage_config_hash,
            )
        ).encode("utf-8")
    )


def attribution_identity_document(
    *,
    detection_keys: list[str],
    timeline_sha256: str,
    speech_band_identity: str,
    stage_config_hash: str,
) -> dict[str, Any]:
    """Everything the assembled graph depends on, before it is hashed.

    The detection keys are carried rather than re-derived, so this identity inherits
    everything each of them covers without restating it — and a graph built from five fresh
    detections and one stale one is impossible to spell.
    """
    return {
        "activity_semantics_version": ACTIVITY_SEMANTICS_VERSION,
        "cache_record_version": CACHE_RECORD_VERSION,
        "detection_keys": sorted(detection_keys),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "speech_band_identity": speech_band_identity,
        "stage_config_hash": stage_config_hash,
        "timeline_semantics_version": TIMELINE_SEMANTICS_VERSION,
        "timeline_sha256": timeline_sha256,
    }


def attribution_identity(
    *,
    detection_keys: list[str],
    timeline_sha256: str,
    speech_band_identity: str,
    stage_config_hash: str,
) -> str:
    """The full cache identity of one session's activity graph (INV-08)."""
    return sha256_bytes(
        canonical_json(
            attribution_identity_document(
                detection_keys=detection_keys,
                timeline_sha256=timeline_sha256,
                speech_band_identity=speech_band_identity,
                stage_config_hash=stage_config_hash,
            )
        ).encode("utf-8")
    )


def probability_relative_path(key: str) -> str:
    """Where one track's per-frame probabilities live, session-relative.

    Content-addressed rather than named for the track, for the reason M2's derivatives are:
    two runs of the same session can coexist while one is being rebuilt, and nothing ever
    overwrites an artifact something else might be reading.
    """
    return f"{DETECTION_DIRNAME}/{key}.probs"


@dataclass(frozen=True, slots=True)
class CachedDetection:
    """A detection entry that is present and complete."""

    key: str
    track_id: str
    regions: tuple[SpeechRegion, ...]
    frame_count: int
    from_detector: bool
    probability_relative_path: str

    def probabilities(self, session_dir: Path) -> npt.NDArray[np.uint16]:
        """Read the per-frame probabilities back. Bounded: two bytes per 32 ms."""
        raw = (session_dir / self.probability_relative_path).read_bytes()
        return np.frombuffer(raw, dtype=PROBABILITY_DTYPE).astype(np.uint16, copy=False)


@dataclass
class DetectionCache:
    """Per-track detection results, addressed by everything that could change them."""

    session_dir: Path
    read_enabled: bool = True
    hits: int = 0
    misses: int = 0
    _staged: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def get(self, key: str) -> CachedDetection | None:
        """The complete entry for ``key``, or ``None``. Counts the hit or the miss."""
        if not self.read_enabled:
            self.misses += 1
            return None

        record = self._read_sidecar(key)
        if record is None:
            self.misses += 1
            return None

        expected = record.frame_count * 2
        try:
            size = (self.session_dir / record.probability_relative_path).stat().st_size
        except OSError:
            self.misses += 1
            return None
        if size != expected:
            self.misses += 1
            return None

        self.hits += 1
        return record

    def publish(self, key: str, result: DetectionResult) -> CachedDetection:
        """Write the probabilities and stage the sidecar that will make ``key`` findable.

        The data lands first and the sidecar is staged, never the other way round: the
        sidecar is what makes an entry a hit, so writing it first would advertise a file
        that does not exist yet.
        """
        relative = probability_relative_path(key)
        payload = result.frame_probabilities.astype(PROBABILITY_DTYPE).tobytes()
        write_atomic(self.session_dir / relative, payload)

        entry = CachedDetection(
            key=key,
            track_id=result.track_id,
            regions=result.regions,
            frame_count=int(result.frame_probabilities.shape[0]),
            from_detector=result.from_detector,
            probability_relative_path=relative,
        )
        self._staged[key] = {
            "cache_record_version": CACHE_RECORD_VERSION,
            "frame_count": entry.frame_count,
            "frame_samples": DETECTOR_FRAME_SAMPLES,
            "from_detector": entry.from_detector,
            "key": key,
            "probability_dtype": PROBABILITY_DTYPE,
            "probability_relative_path": relative,
            "regions": [
                {
                    "start_sample": region.start_sample,
                    "end_sample": region.end_sample,
                    "probability_permille": region.probability_permille,
                    "peak_probability_permille": region.peak_probability_permille,
                }
                for region in entry.regions
            ],
            "track_id": entry.track_id,
        }
        return entry

    def commit(self) -> int:
        """Write every staged sidecar atomically. Returns how many were written."""
        written = 0
        for key, payload in sorted(self._staged.items()):
            write_json_atomic(self._sidecar_path(key), payload)
            written += 1
        self._staged.clear()
        return written

    def discard(self) -> None:
        """Drop everything staged. The data remains, and without a sidecar it is inert."""
        self._staged.clear()

    def _sidecar_path(self, key: str) -> Path:
        return self.session_dir / f"{DETECTION_DIRNAME}/{key}.json"

    def _read_sidecar(self, key: str) -> CachedDetection | None:
        document = _parse(self._sidecar_path(key))
        if document is None or document.get("key") != key:
            return None
        try:
            version = int(document["cache_record_version"])
            relative = str(document["probability_relative_path"])
            frames = int(document["frame_count"])
            regions = tuple(
                SpeechRegion(
                    start_sample=int(item["start_sample"]),
                    end_sample=int(item["end_sample"]),
                    probability_permille=int(item["probability_permille"]),
                    peak_probability_permille=int(item["peak_probability_permille"]),
                )
                for item in document["regions"]
            )
            entry = CachedDetection(
                key=key,
                track_id=str(document["track_id"]),
                regions=regions,
                frame_count=frames,
                from_detector=bool(document["from_detector"]),
                probability_relative_path=relative,
            )
        except (KeyError, TypeError, ValueError):
            return None

        # A sidecar that disagrees with itself is not a usable entry: the reader looks at the
        # *canonical* path, so a record naming another file would grant a hit on the strength
        # of a file nothing goes on to read.
        if (
            version != CACHE_RECORD_VERSION
            or relative != probability_relative_path(key)
            or document.get("frame_samples") != DETECTOR_FRAME_SAMPLES
            or document.get("probability_dtype") != PROBABILITY_DTYPE
            or frames < 0
        ):
            return None
        return entry


@dataclass
class AttributionCache:
    """The assembled graph, addressed by every detection and every threshold behind it."""

    session_dir: Path
    read_enabled: bool = True
    hits: int = 0
    misses: int = 0
    _staged: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def get(self, key: str) -> ActivityGraph | None:
        """The graph for ``key``, or ``None``.

        A stored graph is re-validated through the model rather than trusted, because a
        document that no longer satisfies the frozen contract is a miss, not a crash — the
        artifact is regenerable and refusing to rebuild it would turn a schema change into
        an unrecoverable session.
        """
        if not self.read_enabled:
            self.misses += 1
            return None

        document = _parse(self._entry_path(key))
        if document is None:
            self.misses += 1
            return None
        try:
            graph = ActivityGraph.model_validate(document)
        except ValidationError:
            self.misses += 1
            return None
        if graph.attribution_cache_key != key:
            self.misses += 1
            return None

        self.hits += 1
        return graph

    def publish(self, key: str, graph: ActivityGraph) -> None:
        """Stage the graph. Nothing reaches disk until :meth:`commit`."""
        self._staged[key] = graph.model_dump(mode="json")

    def commit(self) -> int:
        written = 0
        for key, payload in sorted(self._staged.items()):
            write_json_atomic(self._entry_path(key), payload)
            written += 1
        self._staged.clear()
        return written

    def discard(self) -> None:
        self._staged.clear()

    def _entry_path(self, key: str) -> Path:
        return self.session_dir / f"{ATTRIBUTION_DIRNAME}/{key}.json"


def _parse(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or ``None`` for anything that is not one.

    Every form of unreadability is a miss rather than an error, in both caches: a corrupted
    cache should cost time, not a session.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None
