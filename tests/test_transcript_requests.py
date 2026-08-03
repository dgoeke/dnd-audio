"""Segment requests: what is submitted, what it owns, and what it may never exceed.

The gate criteria this file proves are the first three of M4's: requests come from retained
candidates rather than from six full-length files, short adjacent regions merge, every request
has padding and an unpadded ownership interval, and **the submitted padded waveform never
exceeds `max_segment_s`**.

The last one has two failure modes and only one of them is obvious. A core longer than the cap
is the obvious one. The one a review had to point out is a core comfortably *inside* the cap
whose padding pushes the submitted waveform over it — which is the ordinary case for any
session configured with generous padding.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from dnd_audio.artifacts.activity import (
    ActivityCandidate,
    ActivityGraph,
    ActivityProvenance,
    ActivityTrack,
    CandidateEvidence,
    DetectorIdentity,
    candidate_id,
)
from dnd_audio.config import AsrConfig, TranscriptConfig
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.transcript.requests import core_cap_samples, plan_requests, request_id

HASH = "c" * 64
DECIMATION = 3


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
    """A candidate whose 48 kHz interval is exactly three times its derivative one."""
    fields: dict[str, Any] = {
        "candidate_id": candidate_id(track_id, start),
        "track_id": track_id,
        "start_sample": start,
        "end_sample": end,
        "derivative_start_sample": start // DECIMATION,
        "derivative_end_sample": -(-end // DECIMATION),
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
    candidates: list[ActivityCandidate],
    tracks: list[ActivityTrack] | None = None,
    duration_samples: int = 48_000 * 600,
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
        sample_rate=48_000,
        derivative_sample_rate=DERIVATIVE_SAMPLE_RATE,
        duration_samples=duration_samples,
        tracks=tracks or [a_track()],
        candidates=candidates,
    )


def plan(
    candidates: list[ActivityCandidate],
    *,
    tracks: list[ActivityTrack] | None = None,
    duration_samples: int = 48_000 * 600,
    asr: AsrConfig | None = None,
    transcript: TranscriptConfig | None = None,
) -> list[Any]:
    return plan_requests(
        a_graph(candidates, tracks, duration_samples),
        asr=asr or AsrConfig(),
        transcript=transcript or TranscriptConfig(),
    )


class TestFromTheGraph:
    def test_requests_come_from_retained_candidates_only(self) -> None:
        """A suppressed candidate is another track's voice; transcribing it is the waste
        M3's bleed gate exists to avoid."""
        winner = a_candidate("tx-a", 48_000, 96_000)
        loser = a_candidate(
            "tx-b",
            48_000,
            96_000,
            decision="suppressed",
            suppressed_by_candidate_id=winner.candidate_id,
            evidence=[
                CandidateEvidence(
                    other_candidate_id=winner.candidate_id,
                    other_track_id="tx-a",
                    overlap_start_sample=48_000,
                    overlap_end_sample=96_000,
                    compared_derivative_samples=16_000,
                    correlation_permille=900,
                    lag_derivative_samples=48,
                    score_margin_permille=300,
                    level_delta_mb=1500,
                    outcome="suppresses",
                )
            ],
        )
        plans = plan(
            [winner, loser],
            tracks=[a_track("tx-a"), a_track("tx-b")],
        )
        assert [p.track_id for p in plans] == ["tx-a"]
        assert [o.candidate_id for p in plans for o in p.ownership] == [winner.candidate_id]

    def test_an_ambiguous_candidate_is_always_planned(self) -> None:
        """`ambiguous` means the veto overrode the numbers (ADR-0014) — not "skip it"."""
        candidates = [a_candidate("tx-a", 48_000, 96_000, ambiguous=True)]
        assert len(plan(candidates)) == 1

    def test_adjacent_regions_merge_into_one_request(self) -> None:
        """Half a second apart, well inside the default 1.5 s merge gap."""
        first = a_candidate("tx-a", 48_000, 96_000)
        second = a_candidate("tx-a", 120_000, 168_000)
        plans = plan([first, second])
        assert len(plans) == 1
        assert [o.candidate_id for o in plans[0].ownership] == [
            first.candidate_id,
            second.candidate_id,
        ]

    def test_a_long_silence_between_regions_keeps_them_apart(self) -> None:
        first = a_candidate("tx-a", 48_000, 96_000)
        second = a_candidate("tx-a", 48_000 * 30, 48_000 * 31)
        plans = plan([first, second])
        assert len(plans) == 2

    def test_merging_joins_the_audio_and_not_the_ownership(self) -> None:
        """ADR-0017: one retained candidate still produces one ownership interval."""
        first = a_candidate("tx-a", 48_000, 96_000)
        second = a_candidate("tx-a", 120_000, 168_000)
        (only,) = plan([first, second])
        assert len(only.ownership) == 2
        assert only.ownership[0].session_start_sample == 48_000
        assert only.ownership[0].session_end_sample == 96_000
        assert only.ownership[1].session_start_sample == 120_000
        assert only.ownership[1].session_end_sample == 168_000
        # The silence between them belongs to no ownership interval, which is what makes a
        # word returned there dropped rather than attributed to whichever side is nearer.
        assert only.ownership[0].end_sample < only.ownership[1].start_sample

    def test_two_tracks_do_not_share_a_request(self) -> None:
        plans = plan(
            [a_candidate("tx-a", 48_000, 96_000), a_candidate("tx-b", 60_000, 96_000)],
            tracks=[a_track("tx-a"), a_track("tx-b")],
        )
        assert sorted(p.track_id for p in plans) == ["tx-a", "tx-b"]

    def test_every_request_has_padding_around_its_core(self) -> None:
        (only,) = plan([a_candidate("tx-a", 48_000 * 10, 48_000 * 11)])
        pad = TranscriptConfig().pad_ms * DERIVATIVE_SAMPLE_RATE // 1000
        assert only.core_start_sample - only.padded_start_sample == pad
        assert only.padded_end_sample - only.core_end_sample == pad

    def test_padding_is_clamped_at_the_start_of_the_session(self) -> None:
        (only,) = plan([a_candidate("tx-a", 0, 48_000)])
        assert only.padded_start_sample == 0

    def test_padding_is_clamped_at_the_end_of_the_session(self) -> None:
        duration = 48_000 * 12
        (only,) = plan([a_candidate("tx-a", 48_000 * 11, duration)], duration_samples=duration)
        assert only.padded_end_sample == -(-duration // DECIMATION)

    def test_requests_are_ordered_by_time_then_track(self) -> None:
        plans = plan(
            [
                a_candidate("tx-b", 48_000 * 30, 48_000 * 31),
                a_candidate("tx-a", 48_000 * 30, 48_000 * 31),
                a_candidate("tx-a", 48_000, 96_000),
            ],
            tracks=[a_track("tx-a"), a_track("tx-b")],
        )
        assert [(p.core_start_sample, p.track_id) for p in plans] == sorted(
            (p.core_start_sample, p.track_id) for p in plans
        )
        assert [p.track_id for p in plans] == ["tx-a", "tx-a", "tx-b"]

    def test_ids_derive_from_track_and_session_position(self) -> None:
        (only,) = plan([a_candidate("tx-a", 48_000, 96_000)])
        assert only.request_id == request_id("tx-a", 48_000) == "req_tx-a_000000048000"

    def test_a_session_with_nothing_retained_plans_nothing(self) -> None:
        assert plan([]) == []


class TestTheCap:
    """`max_segment_s` bounds the *padded* waveform. Both ways it can be exceeded."""

    def test_a_core_longer_than_the_cap_is_split(self) -> None:
        asr = AsrConfig(max_segment_s=10)
        plans = plan([a_candidate("tx-a", 0, 48_000 * 25)], asr=asr)
        assert len(plans) > 1
        assert all(p.padded_samples <= 10 * DERIVATIVE_SAMPLE_RATE for p in plans)

    def test_a_core_inside_the_cap_is_cut_so_its_padding_still_fits(self) -> None:
        """The case the plan review had to point out.

        A five-second cap with two seconds of padding on each side leaves one second of core.
        A two-second candidate is comfortably *inside* the cap and its padded waveform is not:
        the obvious test — a core longer than the cap — never reaches this.
        """
        asr = AsrConfig(max_segment_s=5)
        transcript = TranscriptConfig(pad_ms=2_000)
        cap, pad = core_cap_samples(asr, transcript)
        assert (cap, pad) == (16_000, 32_000)

        plans = plan(
            [a_candidate("tx-a", 48_000 * 10, 48_000 * 12)], asr=asr, transcript=transcript
        )
        assert len(plans) == 2
        for request in plans:
            assert request.core_samples <= cap
            assert request.padded_samples <= 5 * DERIVATIVE_SAMPLE_RATE

    def test_padding_shrinks_rather_than_the_cap_being_exceeded(self) -> None:
        """`pad_ms` and `max_segment_s` are bounded independently, so this is configurable.

        When no core would fit, the padding gives way: a submitted waveform over the cap is a
        request the adapter refuses, and a shorter pad is only a worse chance at a boundary
        word.
        """
        asr = AsrConfig(max_segment_s=1)
        transcript = TranscriptConfig(pad_ms=5_000)
        cap, pad = core_cap_samples(asr, transcript)
        assert pad < 5_000 * DERIVATIVE_SAMPLE_RATE // 1000
        assert cap >= 1
        assert cap + 2 * pad <= 1 * DERIVATIVE_SAMPLE_RATE

    def test_every_split_piece_keeps_its_candidate_identity(self) -> None:
        """A candidate cut across requests is still one candidate, and still one segment."""
        long = a_candidate("tx-a", 0, 48_000 * 25)
        plans = plan([long], asr=AsrConfig(max_segment_s=10))
        owned = {o.candidate_id for p in plans for o in p.ownership}
        assert owned == {long.candidate_id}

    def test_split_pieces_tile_the_candidate_without_gaps_or_overlap(self) -> None:
        long = a_candidate("tx-a", 0, 48_000 * 25)
        plans = plan([long], asr=AsrConfig(max_segment_s=10))
        pieces = [o for p in plans for o in p.ownership]
        assert pieces[0].start_sample == long.derivative_start_sample
        assert pieces[-1].end_sample == long.derivative_end_sample
        for earlier, later in itertools.pairwise(pieces):
            assert earlier.end_sample == later.start_sample

    def test_the_outer_session_bounds_are_the_candidates_own(self) -> None:
        """Reconverting them would shrink the candidate — M2's floor/ceil trap."""
        long = a_candidate("tx-a", 1, 48_000 * 25 + 2)
        plans = plan([long], asr=AsrConfig(max_segment_s=10))
        pieces = [o for p in plans for o in p.ownership]
        assert pieces[0].session_start_sample == long.start_sample
        assert pieces[-1].session_end_sample == long.end_sample

    def test_a_merged_group_is_cut_at_a_candidate_boundary_when_it_can_be(self) -> None:
        """An ordinary long conversation is cut where somebody stopped talking."""
        asr = AsrConfig(max_segment_s=10)
        transcript = TranscriptConfig(merge_gap_ms=60_000)
        candidates = [
            a_candidate("tx-a", 48_000 * index * 4, 48_000 * (index * 4 + 3)) for index in range(5)
        ]
        plans = plan(candidates, asr=asr, transcript=transcript)
        for request in plans:
            assert request.padded_samples <= 10 * DERIVATIVE_SAMPLE_RATE
            for ownership in request.ownership:
                # Never cut mid-candidate: each piece is a whole candidate.
                matching = next(c for c in candidates if c.candidate_id == ownership.candidate_id)
                assert ownership.start_sample == matching.derivative_start_sample
                assert ownership.end_sample == matching.derivative_end_sample

    def test_the_cap_holds_for_every_planned_request(self) -> None:
        """The property, over a session with merges, splits and edges all at once."""
        asr = AsrConfig(max_segment_s=8)
        candidates = [
            a_candidate("tx-a", 0, 48_000 * 30),
            a_candidate("tx-a", 48_000 * 31, 48_000 * 32),
            a_candidate("tx-a", 48_000 * 33, 48_000 * 40),
        ]
        plans = plan(candidates, asr=asr)
        assert all(p.padded_samples <= 8 * DERIVATIVE_SAMPLE_RATE for p in plans)
        assert all(p.core_samples >= 1 for p in plans)


class TestPlansCarryNoAudio:
    def test_a_plan_is_intervals_and_ids(self) -> None:
        """INV-07: building every request's samples up front would hold the session."""
        (only,) = plan([a_candidate("tx-a", 48_000, 96_000)])
        assert not hasattr(only, "audio")
        assert not hasattr(only, "samples")


class TestMalformedPlansAreRefused:
    def test_a_request_must_own_something(self) -> None:
        from dnd_audio.transcript.requests import RequestPlan

        with pytest.raises(ValueError, match="owns nothing"):
            RequestPlan(
                request_id="req-1",
                track_id="tx-a",
                core_start_sample=0,
                core_end_sample=100,
                padded_start_sample=0,
                padded_end_sample=100,
                ownership=(),
            )

    def test_padding_may_not_reach_inside_the_core(self) -> None:
        from dnd_audio.transcript.requests import Ownership, RequestPlan

        ownership = (
            Ownership(
                candidate_id="cand_tx-a_000000000000",
                start_sample=0,
                end_sample=100,
                session_start_sample=0,
                session_end_sample=300,
            ),
        )
        with pytest.raises(ValueError, match="inside its own core"):
            RequestPlan(
                request_id="req-1",
                track_id="tx-a",
                core_start_sample=0,
                core_end_sample=100,
                padded_start_sample=0,
                padded_end_sample=50,
                ownership=ownership,
            )
