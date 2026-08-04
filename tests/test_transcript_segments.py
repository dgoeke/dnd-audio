"""Who owns each word, and how a merged request comes back apart.

ADR-0017 says requests merge and ownership does not; ADR-0020 says a word belongs to the
ownership interval containing its start and a word in padding belongs to nobody. This file is
those two decisions, checked on the shapes that actually occur: one candidate per request,
several candidates in one request, one candidate cut across several requests, and the wordless
result that cannot be divided at all.
"""

from __future__ import annotations

from typing import Any

from dnd_audio.artifacts.transcript import AlignmentStatus
from dnd_audio.interfaces import TranscribedWord
from dnd_audio.transcript.asr import RequestOutcome
from dnd_audio.transcript.requests import Ownership, RequestPlan
from dnd_audio.transcript.segments import draft_segments

RATE = 16_000
DECIMATION = 3


def an_ownership(candidate: str, start: int, end: int) -> Ownership:
    return Ownership(
        candidate_id=candidate,
        start_sample=start,
        end_sample=end,
        session_start_sample=start * DECIMATION,
        session_end_sample=end * DECIMATION,
    )


def a_plan(request_id: str, ownership: tuple[Ownership, ...], track: str = "tx-a") -> RequestPlan:
    core_start = ownership[0].start_sample
    core_end = ownership[-1].end_sample
    return RequestPlan(
        request_id=request_id,
        track_id=track,
        core_start_sample=core_start,
        core_end_sample=core_end,
        padded_start_sample=max(0, core_start - RATE // 2),
        padded_end_sample=core_end + RATE // 2,
        ownership=ownership,
    )


def a_word(start: int, text: str, length: int = 200) -> TranscribedWord:
    return TranscribedWord(start_sample=start, end_sample=start + length, text=text)


def an_outcome(
    plan: RequestPlan,
    text: str,
    words: tuple[TranscribedWord, ...] = (),
    alignment_status: AlignmentStatus | None = None,
    **overrides: Any,
) -> RequestOutcome:
    fields: dict[str, Any] = {
        "plan": plan,
        "text": text,
        "words": words,
        "alignment_status": alignment_status or ("aligned" if words else "not_attempted"),
        "request_ids": (plan.request_id,),
        "truncation_submissions": 0,
        "truncated": False,
    }
    return RequestOutcome(**{**fields, **overrides})


def draft(*outcomes: RequestOutcome) -> Any:
    """The drafts and notes. Diagnostic 9's third return value has its own tests below."""
    drafts, notes, _ = draft_segments(tuple(outcomes), decimation=DECIMATION)
    return drafts, notes


def dropped(*outcomes: RequestOutcome) -> Any:
    return draft_segments(tuple(outcomes), decimation=DECIMATION)[2]


class TestOneCandidatePerRequest:
    def test_a_segment_carries_the_words_it_owns(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        outcome = an_outcome(
            plan,
            "We should go",
            (a_word(RATE, "We"), a_word(RATE + 400, "should"), a_word(RATE + 800, "go")),
        )
        drafts, notes = draft(outcome)
        (only,) = drafts
        assert only.candidate_ids == ("cand-a",)
        assert only.text == "We should go"
        assert [word.text for word in only.words] == ["We", "should", "go"]
        assert notes == []

    def test_times_cross_to_the_session_grid(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        drafts, _ = draft(an_outcome(plan, "hi", (a_word(RATE + 300, "hi"),)))
        assert drafts[0].words[0].start_sample == (RATE + 300) * DECIMATION

    def test_a_words_start_is_clamped_into_the_interval_that_owns_it(self) -> None:
        """The graph's 48 kHz interval covers its derivative one, so converting the very
        first derivative sample back can land before the candidate begins."""
        ownership = Ownership(
            candidate_id="cand-a",
            start_sample=100,
            end_sample=200,
            # 301 floors to derivative sample 100, and 100 converts back to 300 — one sample
            # before the candidate starts, which the records artifact refuses.
            session_start_sample=301,
            session_end_sample=600,
        )
        plan = a_plan("req-1", (ownership,))
        drafts, _ = draft(an_outcome(plan, "hi", (a_word(100, "hi"),)))
        assert drafts[0].words[0].start_sample == 301

    def test_the_segment_interval_is_where_the_words_are(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 3),))
        drafts, _ = draft(an_outcome(plan, "hi", (a_word(RATE * 2, "hi"),)))
        (only,) = drafts
        assert only.start_sample == RATE * 2 * DECIMATION
        assert only.ownership_start_sample == RATE * DECIMATION
        assert only.ownership_end_sample == RATE * 3 * DECIMATION

    def test_a_candidate_the_model_found_nothing_in_produces_no_segment(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        drafts, notes = draft(an_outcome(plan, "   ", alignment_status="not_attempted"))
        assert drafts == []
        assert [note.code for note in notes] == ["candidate_transcribed_to_nothing"]
        assert "1 retained" in notes[0].message


class TestPaddingIsContextAndNotContent:
    def test_a_word_inside_no_ownership_interval_is_dropped(self) -> None:
        """The whole reason padding exists is that the model hears more than it owns."""
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        outcome = an_outcome(
            plan,
            "before We after",
            (
                a_word(RATE - 400, "before"),
                a_word(RATE + 100, "We"),
                a_word(RATE * 2 + 100, "after"),
            ),
        )
        drafts, _ = draft(outcome)
        assert [word.text for word in drafts[0].words] == ["We"]
        assert drafts[0].text == "We"

    def test_a_word_starting_exactly_on_the_end_belongs_to_the_next_interval(self) -> None:
        """Half-open, so a boundary word is never in two segments at once."""
        first = an_ownership("cand-a", RATE, RATE * 2)
        second = an_ownership("cand-b", RATE * 2, RATE * 3)
        plan = a_plan("req-1", (first, second))
        outcome = an_outcome(plan, "edge", (a_word(RATE * 2, "edge"),))
        drafts, _ = draft(outcome)
        (only,) = [item for item in drafts if item.words]
        assert only.candidate_ids == ("cand-b",)


class TestAMergedRequestComesBackApart:
    def test_each_candidate_gets_the_words_it_owns(self) -> None:
        first = an_ownership("cand-a", RATE, RATE * 2)
        second = an_ownership("cand-b", RATE * 3, RATE * 4)
        plan = a_plan("req-1", (first, second))
        outcome = an_outcome(
            plan,
            "yes it is Absolutely not",
            (
                a_word(RATE + 100, "yes"),
                a_word(RATE + 400, "it"),
                a_word(RATE + 700, "is"),
                a_word(RATE * 3 + 100, "Absolutely"),
                a_word(RATE * 3 + 900, "not"),
            ),
        )
        drafts, _ = draft(outcome)
        assert len(drafts) == 2
        assert drafts[0].candidate_ids == ("cand-a",)
        assert drafts[0].text == "yes it is"
        assert drafts[1].candidate_ids == ("cand-b",)
        assert drafts[1].text == "Absolutely not"

    def test_a_word_in_the_silence_between_two_candidates_is_dropped(self) -> None:
        first = an_ownership("cand-a", RATE, RATE * 2)
        second = an_ownership("cand-b", RATE * 3, RATE * 4)
        plan = a_plan("req-1", (first, second))
        outcome = an_outcome(
            plan,
            "yes hmm no",
            (a_word(RATE + 100, "yes"), a_word(RATE * 2 + 500, "hmm"), a_word(RATE * 3 + 1, "no")),
        )
        drafts, _ = draft(outcome)
        assert [item.text for item in drafts] == ["yes", "no"]

    def test_one_candidate_still_produces_one_segment(self) -> None:
        """ADR-0017's whole point: `source_candidate_id` stays singular."""
        plan = a_plan(
            "req-1",
            (an_ownership("cand-a", RATE, RATE * 2), an_ownership("cand-b", RATE * 3, RATE * 4)),
        )
        outcome = an_outcome(plan, "a b", (a_word(RATE + 1, "a"), a_word(RATE * 3 + 1, "b")))
        drafts, _ = draft(outcome)
        assert [len(item.candidate_ids) for item in drafts] == [1, 1]


class TestACandidateCutAcrossRequests:
    def test_its_words_are_stitched_back_into_one_segment(self) -> None:
        first = a_plan("req-1.0", (an_ownership("cand-a", RATE, RATE * 2),))
        second = a_plan("req-1.1", (an_ownership("cand-a", RATE * 2, RATE * 3),))
        drafts, _ = draft(
            an_outcome(first, "first half", (a_word(RATE + 100, "first"),)),
            an_outcome(second, "second half", (a_word(RATE * 2 + 100, "second"),)),
        )
        (only,) = drafts
        assert only.candidate_ids == ("cand-a",)
        assert only.text == "first second"
        assert only.request_ids == ("req-1.0", "req-1.1")
        assert only.ownership_start_sample == RATE * DECIMATION
        assert only.ownership_end_sample == RATE * 3 * DECIMATION

    def test_truncation_submissions_add_up_across_the_pieces(self) -> None:
        first = a_plan("req-1.0", (an_ownership("cand-a", RATE, RATE * 2),))
        second = a_plan("req-1.1", (an_ownership("cand-a", RATE * 2, RATE * 3),))
        drafts, _ = draft(
            an_outcome(first, "a", (a_word(RATE + 1, "a"),), truncation_submissions=2),
            an_outcome(second, "b", (a_word(RATE * 2 + 1, "b"),), truncation_submissions=4),
        )
        assert drafts[0].truncation_submissions == 6


class TestAWordlessResult:
    def test_its_text_is_kept_whole_against_the_ownership_interval(self) -> None:
        """Trimming it is impossible and dropping it loses speech the spec says must survive."""
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        drafts, _ = draft(
            an_outcome(plan, "we lost the word times", alignment_status="segment_only")
        )
        (only,) = drafts
        assert only.text == "we lost the word times"
        assert only.words == ()
        assert only.alignment_status == "segment_only"
        assert (only.start_sample, only.end_sample) == (
            RATE * DECIMATION,
            RATE * 2 * DECIMATION,
        )

    def test_candidates_that_shared_it_share_one_segment(self) -> None:
        """The degenerate case ADR-0017 names, and the only reason the field is a list."""
        plan = a_plan(
            "req-1",
            (an_ownership("cand-a", RATE, RATE * 2), an_ownership("cand-b", RATE * 3, RATE * 4)),
        )
        drafts, _ = draft(an_outcome(plan, "one string for both", alignment_status="segment_only"))
        (only,) = drafts
        assert only.candidate_ids == ("cand-a", "cand-b")
        assert only.text == "one string for both"

    def test_a_group_is_not_aligned_when_any_contributor_was_not(self) -> None:
        first = a_plan("req-1.0", (an_ownership("cand-a", RATE, RATE * 2),))
        second = a_plan("req-1.1", (an_ownership("cand-a", RATE * 2, RATE * 3),))
        drafts, _ = draft(
            an_outcome(first, "first", (a_word(RATE + 100, "first"),)),
            an_outcome(second, "second", alignment_status="segment_only"),
        )
        (only,) = drafts
        assert only.alignment_status == "segment_only"
        assert only.words == ()
        assert only.text == "first second"


class TestAlignmentFailureWarns:
    """The spec: retain the segment-level transcript and *emit a warning* (not fail)."""

    def test_a_failed_alignment_produces_a_warning(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        drafts, notes = draft(an_outcome(plan, "no word times", alignment_status="segment_only"))
        assert [note.code for note in notes] == ["alignment_failed"]
        assert notes[0].path == "tx-a"
        assert "1 segment(s) on tx-a" in notes[0].message
        assert drafts[0].text == "no word times"

    def test_an_aligner_that_never_ran_is_not_a_failure(self) -> None:
        """`not_attempted` is a different state, and warning about it would be noise."""
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        _, notes = draft(an_outcome(plan, "text", alignment_status="not_attempted"))
        assert [note.code for note in notes] == []

    def test_one_warning_per_track_rather_than_per_segment(self) -> None:
        """An aligner that fails does not fail once; thousands of lines hide the problem."""
        first = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        second = a_plan("req-2", (an_ownership("cand-b", RATE * 5, RATE * 6),))
        third = a_plan("req-3", (an_ownership("cand-c", RATE, RATE * 2),), track="tx-b")
        _, notes = draft(
            an_outcome(first, "one", alignment_status="segment_only"),
            an_outcome(second, "two", alignment_status="segment_only"),
            an_outcome(third, "three", alignment_status="segment_only"),
        )
        assert [(note.path, note.code) for note in notes] == [
            ("tx-a", "alignment_failed"),
            ("tx-b", "alignment_failed"),
        ]
        assert "2 segment(s) on tx-a" in notes[0].message

    def test_a_successful_alignment_warns_about_nothing(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        _, notes = draft(an_outcome(plan, "hi", (a_word(RATE + 1, "hi"),)))
        assert notes == []


class TestOrdering:
    def test_drafts_are_ordered_by_time_then_track(self) -> None:
        early = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),), track="tx-b")
        late = a_plan("req-2", (an_ownership("cand-b", RATE * 5, RATE * 6),), track="tx-a")
        same = a_plan("req-3", (an_ownership("cand-c", RATE, RATE * 2),), track="tx-a")
        drafts, _ = draft(
            an_outcome(late, "late", (a_word(RATE * 5 + 1, "late"),)),
            an_outcome(early, "early", (a_word(RATE + 1, "early"),)),
            an_outcome(same, "same", (a_word(RATE + 1, "same"),)),
        )
        assert [(item.start_sample, item.track_id) for item in drafts] == sorted(
            (item.start_sample, item.track_id) for item in drafts
        )
        assert [item.text for item in drafts] == ["same", "early", "late"]

    def test_the_result_does_not_depend_on_the_order_outcomes_arrive_in(self) -> None:
        first = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        second = a_plan("req-2", (an_ownership("cand-b", RATE * 5, RATE * 6),))
        one = an_outcome(first, "a", (a_word(RATE + 1, "a"),))
        two = an_outcome(second, "b", (a_word(RATE * 5 + 1, "b"),))
        assert draft(one, two)[0] == draft(two, one)[0]


class TestAdjacentPiecesDoNotDuplicateAWord:
    """ADR-0020's rule 3, on the boundary the ADR originally said did not exist.

    A candidate longer than `max_segment_s` is cut by `requests._divide` into pieces that tile
    it exactly, each submitted as its own independently padded request. Each request's padding
    reaches across the boundary into the other's core, so the model can return the boundary
    word on both sides — and a rule that only looks at where a word *starts* then keeps one
    copy in each piece. ADR-0020 called a truncation stitch "the only place two ownership
    intervals are genuinely adjacent"; that was wrong, and this is the case it missed
    (M4's verify phase).
    """

    @staticmethod
    def _cut_in_two(
        left_words: tuple[TranscribedWord, ...], right_words: tuple[TranscribedWord, ...]
    ) -> Any:
        """One candidate across two adjacent, separately padded requests."""
        return (
            an_outcome(
                a_plan("req_tx-a_000000000000", (an_ownership("cand-a", 0, RATE),)),
                "left",
                left_words,
            ),
            an_outcome(
                a_plan("req_tx-a_000000048000", (an_ownership("cand-a", RATE, RATE * 2),)),
                "right",
                right_words,
            ),
        )

    def test_the_same_word_returned_either_side_appears_once(self) -> None:
        """The two copies are at *different* times, which is what a start rule cannot see."""
        outcomes = self._cut_in_two(
            (a_word(500, "hello"), a_word(RATE - 100, "Zephyrine")),
            (a_word(RATE + 20, "Zephyrine"), a_word(RATE + 900, "again")),
        )
        (draft,) = draft_segments(outcomes, decimation=DECIMATION)[0]

        assert [word.text for word in draft.words] == ["hello", "Zephyrine", "again"]
        assert draft.text == "hello Zephyrine again"

    def test_a_genuinely_repeated_word_survives(self) -> None:
        """ "No, no." is two words. Only a repeat that also *overlaps in time* is one."""
        outcomes = self._cut_in_two(
            (a_word(RATE - 400, "no", length=200),),
            (a_word(RATE + 400, "no", length=200),),
        )
        (draft,) = draft_segments(outcomes, decimation=DECIMATION)[0]

        assert [word.text for word in draft.words] == ["no", "no"]

    def test_a_different_word_at_the_boundary_is_kept(self) -> None:
        outcomes = self._cut_in_two(
            (a_word(RATE - 100, "Zephyrine"),),
            (a_word(RATE + 20, "Zephyrus"),),
        )
        (draft,) = draft_segments(outcomes, decimation=DECIMATION)[0]

        assert [word.text for word in draft.words] == ["Zephyrine", "Zephyrus"]

    def test_two_pieces_of_one_candidate_separated_by_a_gap_keep_both(self) -> None:
        """The rule applies only where the two pieces actually touch.

        Padding is what makes the same word reachable from both sides, and padding only spans
        a boundary the pieces *share*. Two pieces with silence between them are two moments,
        and a word repeated across them is a word somebody said twice — deleting it would be
        inventing a correction, which is the one thing this milestone may not do.

        The planner as built never produces this shape: within one group the pieces of a
        candidate tile exactly. The guard is what keeps that an assumption of the *planner*
        rather than a silent assumption of this function, so it is driven directly.

        Ordinary spacing is caught by the time-overlap half of the rule on its own; see
        :meth:`test_a_word_whose_end_reaches_across_a_gap_still_keeps_both` for the input that
        needs the adjacency half.
        """
        outcomes = (
            an_outcome(
                a_plan("req_tx-a_000000000000", (an_ownership("cand-a", 0, RATE),)),
                "left",
                (a_word(RATE - 100, "yes"),),
            ),
            an_outcome(
                a_plan("req_tx-a_000000144000", (an_ownership("cand-a", RATE * 3, RATE * 4),)),
                "right",
                (a_word(RATE * 3 + 10, "yes"),),
            ),
        )
        (draft,) = draft_segments(outcomes, decimation=DECIMATION)[0]

        assert [word.text for word in draft.words] == ["yes", "yes"]


class TestAWordBelongsToTheIntervalContainingItsStart:
    """The half-open start rule itself, on the input that distinguishes it from an overlap
    rule: a word that *straddles* the boundary between two adjacent ownership intervals.

    An overlap rule assigns such a word to both intervals, so it reaches the transcript twice.
    The whole suite passed with that rule substituted, which is why this test exists
    (M4's verify phase).
    """

    def test_a_word_straddling_two_candidates_belongs_only_to_the_first(self) -> None:
        merged = a_plan(
            "req_tx-a_000000000000",
            (an_ownership("cand-a", 0, RATE), an_ownership("cand-b", RATE, RATE * 2)),
        )
        # Starts 100 samples before the boundary and ends 100 after it.
        outcomes = (an_outcome(merged, "one two", (a_word(RATE - 100, "straddling"),)),)
        drafts, _, _ = draft_segments(outcomes, decimation=DECIMATION)

        assert [(draft.candidate_ids, [w.text for w in draft.words]) for draft in drafts] == [
            (("cand-a",), ["straddling"])
        ]

    def test_a_word_whose_end_reaches_across_a_gap_still_keeps_both(self) -> None:
        """The case the time-overlap check alone gets wrong.

        A model that returns an absurdly long end time for one word makes it overlap a word
        two seconds later. Same text, overlapping intervals — the repeat rule would fire and
        delete real speech. Adjacency is what stops it: these two pieces do not share a
        boundary, so no padding could have carried one word into both.
        """
        outcomes = (
            an_outcome(
                a_plan("req_tx-a_000000000000", (an_ownership("cand-a", 0, RATE),)),
                "left",
                (a_word(RATE - 100, "yes", length=RATE * 3),),
            ),
            an_outcome(
                a_plan("req_tx-a_000000144000", (an_ownership("cand-a", RATE * 3, RATE * 4),)),
                "right",
                (a_word(RATE * 3 + 10, "yes"),),
            ),
        )
        (draft,) = draft_segments(outcomes, decimation=DECIMATION)[0]

        assert [word.text for word in draft.words] == ["yes", "yes"]


class TestDroppedWordsAreCounted:
    """Diagnostic 9. ADR-0020 rule 2 is right and it was invisible.

    A word inside a request's padding but inside no ownership interval belongs to nobody and
    is discarded — which is what stops padding from becoming content. On the 2026-08-02
    capture that silently removed the opening word of five of eleven segments, and the
    transcript read as ordinary prose. Nothing here changes the behaviour; it makes the
    number visible, which is what turns `activity.vad.pad_ms` from a guess into a
    measurement on the first real session.
    """

    def test_a_word_just_before_its_interval_is_counted_once(self) -> None:
        """50 ms early, which is the shape a detector starting a moment late produces."""
        early = RATE - RATE // 20
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        outcome = an_outcome(plan, "we should go", (a_word(early, "we"), a_word(RATE, "should")))

        found = dropped(outcome)
        assert [(item.track_id, item.count) for item in found] == [("tx-a", 1)]
        assert found[0].candidate_ids == ("cand-a",)
        assert found[0].before_ownership_count == 1
        assert found[0].after_ownership_count == 0
        assert found[0].leading_word_count == 1
        assert found[0].nonleading_word_count == 0
        assert found[0].edge_distance_derivative_samples == ((RATE // 20, 1),)
        assert found[0].max_edge_distance_derivative_samples == RATE // 20

        # And the note an operator actually sees carries the count.
        _, notes = draft(outcome)
        assert [note.code for note in notes] == ["words_dropped_outside_ownership"]
        assert "1 word(s)" in notes[0].message
        assert "1 were the request's leading returned word" in notes[0].message
        assert f"needs {RATE // 20} additional derivative sample(s)" in notes[0].message

    def test_nothing_is_dropped_when_every_word_is_owned(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        outcome = an_outcome(plan, "we should go", (a_word(RATE, "we"), a_word(RATE + 500, "go")))
        assert dropped(outcome) == ()
        assert draft(outcome)[1] == []

    def test_a_merged_request_owning_two_candidates_drops_nothing(self) -> None:
        """The reason this is computed per *outcome* rather than inside `_owned_words`.

        That function runs once per candidate group and sees only that group's intervals, so
        counting there would call every word of group B a drop while assembling group A —
        a diagnostic that fired on every merged request, which is the ordinary case with six
        lavs in a room.
        """
        plan = a_plan(
            "req-1",
            (
                an_ownership("cand-a", RATE, RATE * 2),
                an_ownership("cand-b", RATE * 2, RATE * 3),
            ),
        )
        outcome = an_outcome(
            plan,
            "mine yours",
            (a_word(RATE + 10, "mine"), a_word(RATE * 2 + 10, "yours")),
        )
        assert dropped(outcome) == ()

    def test_a_word_in_the_gap_between_two_candidates_is_a_real_drop(self) -> None:
        """Padding covers the gap; no interval does. That word is genuinely gone."""
        plan = a_plan(
            "req-1",
            (
                an_ownership("cand-a", RATE, RATE * 2),
                an_ownership("cand-b", RATE * 3, RATE * 4),
            ),
        )
        outcome = an_outcome(
            plan,
            "mine cough yours",
            (
                a_word(RATE + 10, "mine"),
                a_word(RATE * 2 + 500, "cough"),
                a_word(RATE * 3 + 10, "yours"),
            ),
        )
        found = dropped(outcome)
        assert [(item.track_id, item.count) for item in found] == [("tx-a", 1)]
        # Nearest by distance: the gap word sits 500 samples past cand-a's end and 11 500
        # before cand-b's start.
        assert found[0].candidate_ids == ("cand-a",)
        assert found[0].before_ownership_count == 0
        assert found[0].after_ownership_count == 1
        assert found[0].leading_word_count == 0
        assert found[0].nonleading_word_count == 1
        # A half-open end needs one more sample than the distance to the edge itself.
        assert found[0].edge_distance_derivative_samples == ((501, 1),)

    def test_one_padding_word_two_requests_both_drop_counts_twice(self) -> None:
        """The metric is dropped `(request, word)` pairs, and it says so.

        Two requests whose padding both reaches the same moment each lost a word. Collapsing
        that to one would need a rule for when two words at slightly different times are the
        same word — which is precisely the judgement this measurement must not quietly make
        on the operator's behalf.
        """
        stray = RATE * 2 + 200
        outcomes = (
            an_outcome(
                a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),)),
                "left",
                (a_word(stray, "cough"),),
            ),
            an_outcome(
                a_plan("req-2", (an_ownership("cand-b", RATE * 3, RATE * 4),)),
                "right",
                (a_word(stray, "cough"),),
            ),
        )
        found = dropped(*outcomes)
        assert [(item.track_id, item.count) for item in found] == [("tx-a", 2)]
        assert found[0].candidate_ids == ("cand-a", "cand-b")
        assert found[0].before_ownership_count == 1
        assert found[0].after_ownership_count == 1
        assert found[0].leading_word_count == 2
        assert found[0].nonleading_word_count == 0
        assert found[0].edge_distance_derivative_samples == ((201, 1), (RATE - 200, 1))

    def test_the_distance_histogram_is_sorted_and_counts_repeated_distances(self) -> None:
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        outcome = an_outcome(
            plan,
            "one two three",
            (
                a_word(RATE - 800, "one"),
                a_word(RATE - 800, "two"),
                a_word(RATE * 2 + 99, "three"),
            ),
        )

        (found,) = dropped(outcome)
        assert found.count == 3
        assert found.edge_distance_derivative_samples == ((100, 1), (800, 2))
        assert found.max_edge_distance_derivative_samples == 800
        assert found.before_ownership_count == 2
        assert found.after_ownership_count == 1
        assert found.leading_word_count == 1
        assert found.nonleading_word_count == 2

    def test_a_wordless_outcome_drops_nothing(self) -> None:
        """No word times came back at all, so no word was assigned or discarded."""
        plan = a_plan("req-1", (an_ownership("cand-a", RATE, RATE * 2),))
        assert dropped(an_outcome(plan, "no word times", alignment_status="segment_only")) == ()
