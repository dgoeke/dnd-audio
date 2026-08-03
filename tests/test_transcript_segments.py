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
    return draft_segments(tuple(outcomes), decimation=DECIMATION)


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
