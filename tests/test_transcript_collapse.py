"""Duplicate collapse: three independent conditions, and the cases that must survive.

This is the most dangerous code in the milestone, because a wrong collapse deletes speech and
leaves nothing in the audio to show it. So most of these tests are about what is **kept**: the
short matching utterance the spec names by example, the two people saying different things at
once, and the pair the activity graph never found any acoustic relationship between.
"""

from __future__ import annotations

from dataclasses import replace
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


class TestContainedFragmentsAreASeparateConservativeRule:
    CONTAINER = "Finally here's the fourth microphone"
    FRAGMENT = "the fourth microphone"

    def test_a_compelling_proper_fragment_collapses_and_names_its_rule(self) -> None:
        drafts, graph = two_tracks(
            first_text=self.CONTAINER,
            second_text=self.FRAGMENT,
            first_score=900,
            second_score=500,
            correlation=100,
        )
        result = run(drafts, graph)
        assert [item.decision for item in result.verdicts] == ["retained", "duplicate"]
        assert result.decisions[0].code == "contained_fragment_collapsed"
        assert "properly contained" in result.decisions[0].detail

    def test_genuine_overlap_with_unrelated_text_survives(self) -> None:
        drafts, graph = two_tracks(
            first_text=self.CONTAINER,
            second_text="No, I was answering somebody else",
            first_score=900,
            second_score=500,
        )
        assert [item.decision for item in run(drafts, graph).verdicts] == [
            "retained",
            "retained",
        ]

    def test_noncontiguous_words_are_not_containment(self) -> None:
        drafts, graph = two_tracks(
            first_text=self.CONTAINER,
            second_text="Finally fourth microphone",
            first_score=900,
            second_score=500,
        )
        assert [item.decision for item in run(drafts, graph).verdicts] == [
            "retained",
            "retained",
        ]

    def test_absent_graph_evidence_keeps_the_fragment(self) -> None:
        drafts, graph = two_tracks(
            first_text=self.CONTAINER,
            second_text=self.FRAGMENT,
            first_score=900,
            second_score=500,
            evidence=False,
        )
        assert [item.decision for item in run(drafts, graph).verdicts] == [
            "retained",
            "retained",
        ]

    def test_weak_dominance_keeps_the_fragment(self) -> None:
        drafts, graph = two_tracks(
            first_text=self.CONTAINER,
            second_text=self.FRAGMENT,
            first_score=800,
            second_score=550,
        )
        assert [item.decision for item in run(drafts, graph).verdicts] == [
            "retained",
            "retained",
        ]

    def test_the_shorter_better_source_cannot_delete_the_containing_text(self) -> None:
        drafts, graph = two_tracks(
            first_text=self.CONTAINER,
            second_text=self.FRAGMENT,
            first_score=500,
            second_score=900,
        )
        assert [item.decision for item in run(drafts, graph).verdicts] == [
            "retained",
            "retained",
        ]

    def test_exact_yes_and_okay_remain_protected_even_under_extreme_dominance(self) -> None:
        for text in ("Yes", "Okay"):
            drafts, graph = two_tracks(
                first_text=text,
                second_text=text,
                first_score=950,
                second_score=100,
            )
            assert [item.decision for item in run(drafts, graph).verdicts] == [
                "retained",
                "retained",
            ]

    def test_legacy_similarity_finishes_before_containment(self) -> None:
        """A→B containment must not preempt the old C→B similarity decision.

        The first-pass B→C decision remains visible even though the second pass then collapses
        C into A. The resulting terminating audit chain is the concrete M9 plan-review
        counterexample: running containment pair-by-pair would have skipped B→C altogether.
        """
        texts = {
            "tx-a": "alpha we should go home omega",
            "tx-b": "we should go home",
            "tx-c": "we should go home",
        }
        scores = {"tx-a": 900, "tx-b": 400, "tx-c": 500}
        candidates = [
            a_candidate(
                track,
                RATE,
                RATE * 2,
                score=score,
                evidence=[
                    an_evidence(candidate_id(other, RATE), other, 900)
                    for other in scores
                    if other != track
                ],
            )
            for track, score in scores.items()
        ]
        drafts = [a_draft(track, RATE, RATE * 2, texts[track]) for track in scores]

        result = run(drafts, a_graph(candidates, list(scores)))

        assert [item.decision for item in result.verdicts] == [
            "retained",
            "duplicate",
            "duplicate",
        ]
        assert result.verdicts[1].duplicate_of_segment_id == segment_id(2)
        assert result.verdicts[2].duplicate_of_segment_id == segment_id(0)
        assert [item.code for item in result.decisions] == [
            "contained_fragment_collapsed",
            "duplicate_collapsed",
        ]


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

    def test_the_best_score_absorbs_first_however_the_drafts_are_ordered(self) -> None:
        """M4's deferred defect, and ADR-0032's fix.

        A=800, B=700, C=900 in canonical order. Taking pairs in that order, A absorbs B and
        is then forbidden from being absorbed by C — so A *and* C both reach the transcript,
        contradicting this module's own rule that the survivor is the best source score.
        Resolving `(C, A)` first makes the shape unreachable.

        The 2026-08-03 capture is why this is a fix rather than a deleted docstring: three
        tracks within 32 ms carrying identical text, so three lavs do agree closely enough
        for the shape to occur.
        """
        scores = {"tx-a": 800, "tx-b": 700, "tx-c": 900}
        others = {name: [other for other in scores if other != name] for name in scores}
        candidates = [
            a_candidate(
                name,
                RATE,
                RATE * 2,
                score=score,
                evidence=[
                    an_evidence(candidate_id(other, RATE), other, 900) for other in others[name]
                ],
            )
            for name, score in scores.items()
        ]
        drafts = [a_draft(name, RATE, RATE * 2, LONG) for name in scores]

        result = run(drafts, a_graph(candidates, list(scores)))
        assert [verdict.decision for verdict in result.verdicts] == [
            "duplicate",
            "duplicate",
            "retained",
        ]
        # And both losers point at the survivor, so nothing is left dangling.
        assert {result.verdicts[index].duplicate_of_segment_id for index in (0, 1)} == {
            segment_id(2)
        }


class TestDeterminism:
    def test_the_same_input_gives_the_same_verdicts(self) -> None:
        drafts, graph = two_tracks()
        assert run(drafts, graph) == run(drafts, graph)


class TestASegmentCoveringSeveralCandidates:
    """The wordless case ADR-0017 names, and the one every other test here avoids.

    When alignment fails on a merged request its text cannot be divided, so the candidates
    that shared it share one segment. Collapse then has to reason about a segment backed by
    *several* candidates, and both halves of that reasoning were untested: the suite passed
    with the weakest correlation replaced by the strongest, and it passed while a segment was
    collapsed on evidence covering only part of it (M4's verify phase).
    """

    @staticmethod
    def _merged(
        *,
        first_correlation: int,
        second_correlation: int | None,
        # Close enough that the spec's "or compelling source-dominance" escape hatch
        # cannot fire, so correlation is what decides and this tests what it says it does.
        merged_score: int = 850,
        other_score: int = 900,
    ) -> tuple[list[SegmentDraft], ActivityGraph]:
        """`tx-a`'s two candidates share one wordless segment; `tx-b` has one of its own."""
        a1, a2 = candidate_id("tx-a", RATE * 10), candidate_id("tx-a", RATE * 11)
        b1 = candidate_id("tx-b", RATE * 11)
        other_evidence = [an_evidence(a2, "tx-a", first_correlation)]
        first_evidence = []
        if second_correlation is not None:
            other_evidence.append(an_evidence(a1, "tx-a", second_correlation))
            first_evidence.append(an_evidence(b1, "tx-b", second_correlation))

        graph = a_graph(
            [
                a_candidate(
                    "tx-a", RATE * 10, RATE * 11, score=merged_score, evidence=first_evidence
                ),
                a_candidate(
                    "tx-a",
                    RATE * 11,
                    RATE * 13,
                    score=merged_score,
                    evidence=[an_evidence(b1, "tx-b", first_correlation)],
                ),
                a_candidate(
                    "tx-b", RATE * 11, RATE * 13, score=other_score, evidence=other_evidence
                ),
            ],
            ["tx-a", "tx-b"],
        )
        merged = replace(
            a_draft("tx-a", RATE * 10, RATE * 13, "Yeah " + LONG), candidate_ids=(a1, a2)
        )
        return [merged, a_draft("tx-b", RATE * 11, RATE * 13, LONG)], graph

    def test_every_pair_must_clear_the_threshold_not_the_best_one(self) -> None:
        """One candidate correlating strongly cannot vouch for the other."""
        drafts, graph = self._merged(first_correlation=950, second_correlation=100)
        verdicts = collapse(
            drafts, graph, settings=DuplicateConfig(), overlap_min_samples=1000
        ).verdicts
        assert [verdict.decision for verdict in verdicts] == ["retained", "retained"]

    def test_all_pairs_clearing_it_does_collapse(self) -> None:
        """The control: the same shape with both pairs measured and both strong."""
        drafts, graph = self._merged(first_correlation=950, second_correlation=950)
        verdicts = collapse(
            drafts, graph, settings=DuplicateConfig(), overlap_min_samples=1000
        ).verdicts
        assert [verdict.decision for verdict in verdicts] == ["duplicate", "retained"]

    def test_a_candidate_the_graph_never_compared_keeps_both(self) -> None:
        """The hole: `A1` was never measured against anything on `tx-b`, so `A2`'s strong
        evidence was silently taken to cover it — and collapsing the merged record deleted
        `A1`'s words with nothing in the transcript to show it happened."""
        drafts, graph = self._merged(first_correlation=950, second_correlation=None)
        result = collapse(drafts, graph, settings=DuplicateConfig(), overlap_min_samples=1000)
        assert [verdict.decision for verdict in result.verdicts] == ["retained", "retained"]
        assert "Yeah" in drafts[0].text
        assert result.decisions == ()
