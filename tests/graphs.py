"""Hand-built activity graphs, for tests that need a specific shape rather than a session.

M4's request and collapse tests each grew their own copy of these three builders. This is the
third caller, so it lives in one place — M5's tests only, deliberately: rewriting a closed
milestone's tests to import it would churn code that is passing for no gain, and the two copies
are already load-bearing where they are.

Every default here is a *plausible* value, never a boundary one. A builder whose defaults sit on
a threshold makes every test that does not override them a test of that threshold, silently.
"""

from __future__ import annotations

from typing import Any

from dnd_audio.artifacts.activity import (
    ActivityCandidate,
    ActivityGraph,
    ActivityProvenance,
    ActivityTrack,
    DetectorIdentity,
    candidate_id,
)
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE, DERIVATIVE_SAMPLE_RATE

__all__ = ["HASH", "a_candidate", "a_graph", "a_track"]

HASH = "c" * 64

_DECIMATION = CANONICAL_SAMPLE_RATE // DERIVATIVE_SAMPLE_RATE


def a_track(track_id: str = "tx-a", **overrides: Any) -> ActivityTrack:
    fields: dict[str, Any] = {
        "track_id": track_id,
        "speaker_id": track_id.replace("tx-", "spk-"),
        "speaker_name": track_id.upper(),
        "detection_cache_key": HASH,
        "probability_relative_path": f"work/cache/activity/detect/{track_id}.probs",
        "probability_frames": 100,
        "frame_samples": 512,
        "speech_reference_mbfs": -2800,
    }
    return ActivityTrack(**{**fields, **overrides})


def a_candidate(track_id: str, start: int, end: int, **overrides: Any) -> ActivityCandidate:
    """A candidate whose 48 kHz interval maps exactly onto its derivative one."""
    fields: dict[str, Any] = {
        "candidate_id": candidate_id(track_id, start),
        "track_id": track_id,
        "start_sample": start,
        "end_sample": end,
        "derivative_start_sample": start // _DECIMATION,
        "derivative_end_sample": -(-end // _DECIMATION),
        "probability_permille": 900,
        "peak_probability_permille": 950,
        "band_level_mbfs": -2000,
        "relative_level_mb": 0,
        "score_permille": 800,
        "score_level_permille": 800,
        "score_confidence_permille": 900,
        "score_dominance_permille": 700,
        "score_correlation_permille": 500,
        "decision": "retained",
    }
    return ActivityCandidate(**{**fields, **overrides})


def a_graph(
    candidates: list[ActivityCandidate] | None = None,
    tracks: list[ActivityTrack] | None = None,
    duration_samples: int = CANONICAL_SAMPLE_RATE * 10,
) -> ActivityGraph:
    return ActivityGraph(
        session_id="2026-08-15",
        config_hash=HASH,
        timeline_sha256=HASH,
        attribution_cache_key=HASH,
        provenance=ActivityProvenance(
            activity_semantics_version=1,
            timeline_semantics_version=1,
            inspection_semantics_version=1,
            numpy_version="2.3.4",
            scipy_version="1.18.0",
            detector=DetectorIdentity(name="scripted", variant_digest=HASH),
            speech_band_filter_name="speechband",
            speech_band_filter_identity=HASH,
        ),
        sample_rate=CANONICAL_SAMPLE_RATE,
        derivative_sample_rate=DERIVATIVE_SAMPLE_RATE,
        duration_samples=duration_samples,
        tracks=tracks or [a_track()],
        candidates=candidates or [],
    )
