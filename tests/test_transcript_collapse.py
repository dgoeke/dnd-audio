"""Duplicate collapse: three independent conditions, and the cases that must survive.

This is the most dangerous code in the milestone, because a wrong collapse deletes speech and
leaves nothing in the audio to show it. So most of these tests are about what is **kept**: the
short matching utterance the spec names by example, the two people saying different things at
once, and the pair the activity graph never found any acoustic relationship between.
"""

from __future__ import annotations

from typing import Any

from dnd_audio.artifacts.activity import (
    ActivityCandidate,
    ActivityGraph,
    ActivityProvenance,
    ActivityTrack,
    CandidateEvidence,
    DetectorIdentity,
    candidate_id,
)
from dnd_audio.artifacts.records import segment_id
from dnd_audio.config import DuplicateConfig
from dnd_audio.transcript.collapse import collapse
from dnd_audio.transcript.segments import SegmentDraft

HASH = "e" * 64
RATE = 48_000


def a_track(track_id: str, **overrides: Any) -> ActivityTrack:
    fields: dict[str, Any] = {
        "track_id": track_id,
        "speaker_id": track_id.replace("tx-", ""),
        "speaker_name": track_id.upper(),
        "detection_cache_key": HASH,
        "probability_relative_path": f"work/cache/activity/detect/{track_id}.probs",
        "probability_frames": 10,
        "frame_samples": 512,
    }
    return ActivityTrack(**{**fields, **overrides})


def a_candidate(track_id: str, start: int, end: int, score: int = 800, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "candidate_id": candidate_id(track_id, start),
        "track_id": track_id,
        "start_sample": start,
        "end_sample": end,
        "derivative_start_sample": start // 3,
        "derivative_end_sample": -(-end // 3),
        "probability_permille": 900,
        "peak_probability_permille": 950,
        "band_level_mbfs": -2000,
        "score_permille": score,
        "score_level_permille": score,
        "score_confidence_permille": 900,
        "score_dominance_permille": 700,
        "score_correlation_permille": 500,
        "decision": "retained",
    }
    return ActivityCandidate(**{**fields, **overrides})


def an_evidence(other: str, other_track: str, correlation: int, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "other_candidate_id": other,
        "other_track_id": other_track,
        "overlap_start_sample": RATE,
        "overlap_end_sample": RATE * 2,
        "compared_derivative_samples": 16_000,
        "correlation_permille": correlation,
        "lag_derivative_samples": 24,
        "score_margin_permille": 0,
        "level_delta_mb": 0,
        "outcome": "insufficient_margin",
    }
    return CandidateEvidence(**{**fields, **overrides})


def a_graph(candidates: list[Any], tracks: list[str]) -> ActivityGraph:
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
        derivative_sample_rate=16_000,
        duration_samples=RATE * 600,
        tracks=[a_track(track) for track in tracks],
        candidates=candidates,
    )


def a_draft(track_id: str, start: int, end: int, text: str) -> SegmentDraft:
    return SegmentDraft(
        candidate_ids=(candidate_id(track_id, start),),
        track_id=track_id,
        start_sample=start,
        end_sample=end,
        ownership_start_sample=start,
        ownership_end_sample=end,
        text=text,
        words=(),
        alignment_status="segment_only",
        request_ids=(f"req_{track_id}_{start:012d}",),
        truncation_submissions=0,
    )


LONG = "We should go back to Zephyrine before the gate closes"
LONG_MISHEARD = "We should go back to Zephyrin before the gate closes"


def two_tracks(
    *,
    first_text: str = LONG,
    second_text: str = LONG,
    correlation: int = 900,
    first_score: int = 800,
    second_score: int = 750,
    second_start: int = RATE,
    second_end: int = RATE * 2,
    evidence: bool = True,
) -> tuple[list[SegmentDraft], ActivityGraph]:
    """Alice and Bob, overlapping, with whatever the test needs to be different."""
    first = a_candidate("tx-a", RATE, RATE * 2, score=first_score)
    second = a_candidate("tx-b", second_start, second_end, score=second_score)
    if evidence:
        first = a_candidate(
            "tx-a",
            RATE,
            RATE * 2,
            score=first_score,
            evidence=[an_evidence(second.candidate_id, "tx-b", correlation)],
        )
        second = a_candidate(
            "tx-b",
            second_start,
            second_end,
            score=second_score,
            evidence=[an_evidence(first.candidate_id, "tx-a", correlation)],
        )
    drafts = [
        a_draft("tx-a", RATE, RATE * 2, first_text),
        a_draft("tx-b", second_start, second_end, second_text),
    ]
    return drafts, a_graph([first, second], ["tx-a", "tx-b"])


def run(
    drafts: list[SegmentDraft],
    graph: ActivityGraph,
    *,
    settings: DuplicateConfig | None = None,
    overlap_min_samples: int = RATE // 4,
) -> Any:
    return collapse(
        drafts,
        graph,
        settings=settings or DuplicateConfig(),
        overlap_min_samples=overlap_min_samples,
    )


class TestAllThreeConditionsAreRequired:
    def test_overlap_similar_text_and_correlation_collapse(self) -> None:
        drafts, graph = two_tracks()
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "duplicate"]
        assert result.verdicts[1].duplicate_of_segment_id == segment_id(0)

    def test_materially_different_text_keeps_both(self) -> None:
        drafts, graph = two_tracks(second_text="Absolutely not, we are going north")
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]
        assert all(verdict.overlap for verdict in result.verdicts)

    def test_weak_correlation_keeps_both(self) -> None:
        """The numbers say the two lavs did not hear the same sound, whatever the text says."""
        drafts, graph = two_tracks(correlation=100)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]

    def test_weak_correlation_with_compelling_dominance_still_collapses(self) -> None:
        """The spec's own "or compelling source-dominance evidence"."""
        drafts, graph = two_tracks(correlation=100, first_score=900, second_score=400)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "duplicate"]

    def test_no_measured_pair_at_all_keeps_both(self) -> None:
        """Candidates M3 never compared did not overlap in time, and two utterances that did
        not overlap are not one utterance heard twice."""
        drafts, graph = two_tracks(evidence=False)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]

    def test_slight_overlap_keeps_both(self) -> None:
        drafts, graph = two_tracks(second_start=RATE * 2 - 1000, second_end=RATE * 3)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]

    def test_a_misheard_word_still_collapses(self) -> None:
        """Two lavs never transcribe identically; the threshold is similarity, not equality."""
        drafts, graph = two_tracks(second_text=LONG_MISHEARD)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "duplicate"]


class TestShortUtterancesNeverCollapseOnText:
    def test_two_people_saying_yes_both_survive(self) -> None:
        """The spec names this case by example. Perfect text similarity, strong correlation,
        substantial overlap — and they are two people agreeing with each other."""
        drafts, graph = two_tracks(first_text="Yes.", second_text="Yes.")
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]
        assert all(verdict.overlap for verdict in result.verdicts)

    def test_the_word_floor_is_what_stops_it(self) -> None:
        """Lowering the floor collapses the same pair, so the floor is load-bearing rather
        than incidentally true of this fixture."""
        drafts, graph = two_tracks(first_text="Yes, absolutely.", second_text="Yes, absolutely.")
        lenient = DuplicateConfig(min_text_words=1, min_text_chars=1)
        assert [v.decision for v in run(drafts, graph, settings=lenient).verdicts] == [
            "retained",
            "duplicate",
        ]
        assert [v.decision for v in run(drafts, graph).verdicts] == ["retained", "retained"]

    def test_a_long_enough_utterance_is_not_protected_by_the_floor(self) -> None:
        drafts, graph = two_tracks()
        assert run(drafts, graph).verdicts[1].decision == "duplicate"


class TestTheSurvivorIsTheBestSourceScore:
    def test_the_higher_scoring_track_wins(self) -> None:
        drafts, graph = two_tracks(first_score=600, second_score=900)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["duplicate", "retained"]
        assert result.verdicts[0].duplicate_of_segment_id == segment_id(1)

    def test_a_tie_goes_to_the_lower_segment_id(self) -> None:
        """Which is a function of time and track, never of iteration order (INV-02)."""
        drafts, graph = two_tracks(first_score=800, second_score=800)
        result = run(drafts, graph)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "duplicate"]

    def test_the_text_does_not_decide_the_winner(self) -> None:
        """A longer or more confident-sounding transcript must not out-vote the graph."""
        drafts, graph = two_tracks(
            first_text=LONG, second_text=LONG + " and then we can rest", second_score=900
        )
        result = run(drafts, graph)
        assert result.verdicts[1].decision == "retained"


class TestWhatIsRecorded:
    def test_a_rejected_alternative_carries_the_numbers_that_rejected_it(self) -> None:
        drafts, graph = two_tracks()
        (winner, _) = run(drafts, graph).verdicts
        (alternative,) = winner.rejected_alternatives
        assert alternative.segment_id == segment_id(1)
        assert alternative.track_id == "tx-b"
        assert alternative.speaker_id == "b"
        assert alternative.text == LONG
        assert alternative.overlap_permille == 1000
        assert alternative.text_similarity_permille == 1000
        assert alternative.correlation_permille == 900
        assert alternative.score_margin_permille == 50

    def test_only_actual_rejections_are_recorded(self) -> None:
        """Every evaluated pair would be quadratic growth in the artifact for no audit value."""
        drafts, graph = two_tracks(second_text="Absolutely not, we are going north")
        assert all(not verdict.rejected_alternatives for verdict in run(drafts, graph).verdicts)

    def test_a_collapse_produces_an_auditable_decision(self) -> None:
        drafts, graph = two_tracks()
        (decision,) = run(drafts, graph).decisions
        assert decision.code == "duplicate_collapsed"
        assert decision.subject == segment_id(1)
        assert "1000/1000" in decision.detail
        assert "900/1000" in decision.detail

    def test_a_duplicate_is_never_marked_overlapping(self) -> None:
        drafts, graph = two_tracks()
        assert run(drafts, graph).verdicts[1].overlap is False


class TestOverlapFlag:
    def test_two_retained_speakers_overlapping_are_both_marked(self) -> None:
        drafts, graph = two_tracks(second_text="Absolutely not, we are going north")
        assert [verdict.overlap for verdict in run(drafts, graph).verdicts] == [True, True]

    def test_an_overlap_below_the_threshold_is_not_one(self) -> None:
        drafts, graph = two_tracks(
            second_text="Absolutely not, we are going north",
            second_start=RATE * 2 - 100,
            second_end=RATE * 3,
        )
        assert [verdict.overlap for verdict in run(drafts, graph).verdicts] == [False, False]

    def test_overlapping_only_a_collapsed_duplicate_does_not_count(self) -> None:
        """The spec's definition, exactly: another *retained, non-duplicate* speaker segment.

        The collapsed one is not in the transcript to overlap with.
        """
        drafts, graph = two_tracks()
        result = run(drafts, graph)
        assert result.verdicts[0].decision == "retained"
        assert result.verdicts[1].decision == "duplicate"
        assert result.verdicts[0].overlap is False

    def test_two_segments_on_one_track_never_overlap_each_other(self) -> None:
        """Two candidates on one track are two utterances — the bleed gate's own rule."""
        first = a_candidate("tx-a", RATE, RATE * 2)
        second = a_candidate("tx-a", RATE * 2, RATE * 3)
        drafts = [
            a_draft("tx-a", RATE, RATE * 2 + RATE // 2, LONG),
            a_draft("tx-a", RATE * 2, RATE * 3, "Absolutely not, we are going north"),
        ]
        result = run(drafts, a_graph([first, second], ["tx-a"]))
        assert [verdict.overlap for verdict in result.verdicts] == [False, False]
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]


class TestChainsAreImpossible:
    def test_a_segment_that_absorbed_another_is_never_itself_absorbed(self) -> None:
        """A chain of duplicates has no surviving text at the end of it."""
        first = a_candidate(
            "tx-a",
            RATE,
            RATE * 2,
            score=800,
            evidence=[
                an_evidence(candidate_id("tx-b", RATE), "tx-b", 900),
                an_evidence(candidate_id("tx-c", RATE), "tx-c", 900),
            ],
        )
        second = a_candidate(
            "tx-b",
            RATE,
            RATE * 2,
            score=700,
            evidence=[
                an_evidence(candidate_id("tx-a", RATE), "tx-a", 900),
                an_evidence(candidate_id("tx-c", RATE), "tx-c", 900),
            ],
        )
        third = a_candidate(
            "tx-c",
            RATE,
            RATE * 2,
            score=600,
            evidence=[
                an_evidence(candidate_id("tx-a", RATE), "tx-a", 900),
                an_evidence(candidate_id("tx-b", RATE), "tx-b", 900),
            ],
        )
        drafts = [
            a_draft("tx-a", RATE, RATE * 2, LONG),
            a_draft("tx-b", RATE, RATE * 2, LONG),
            a_draft("tx-c", RATE, RATE * 2, LONG),
        ]
        result = run(drafts, a_graph([first, second, third], ["tx-a", "tx-b", "tx-c"]))
        assert [verdict.decision for verdict in result.verdicts] == [
            "retained",
            "duplicate",
            "duplicate",
        ]
        assert {verdict.duplicate_of_segment_id for verdict in result.verdicts[1:]} == {
            segment_id(0)
        }


class TestDeterminism:
    def test_the_same_input_gives_the_same_verdicts(self) -> None:
        drafts, graph = two_tracks()
        assert run(drafts, graph) == run(drafts, graph)
