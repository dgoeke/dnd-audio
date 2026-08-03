"""Submitting requests, and surviving a truncated answer.

The truncation path is the one that can quietly lose speech, so most of this file is about
what happens when a response comes back cut off: where it is split, how many attempts that is
allowed to cost, and what is kept when the budget runs out. ADR-0020 is the decision these
tests pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from dnd_audio.artifacts.records import TranscriberIdentity
from dnd_audio.config import TranscriptConfig
from dnd_audio.fakes import ScriptedTranscriber
from dnd_audio.interfaces import TranscribedWord, TranscriptionResult
from dnd_audio.transcript.asr import SPLIT_FRAME_SAMPLES, split_point, transcribe_plans
from dnd_audio.transcript.cache import AsrCache
from dnd_audio.transcript.requests import Ownership, PlanContext, RequestPlan

RATE = 16_000
DECIMATION = 3
CONTEXT = PlanContext(pad=RATE // 2, limit=RATE * 600, decimation=DECIMATION, core_cap=RATE * 60)


def an_identity(**overrides: Any) -> TranscriberIdentity:
    fields: dict[str, Any] = {
        "name": "scripted",
        "max_new_tokens": 1024,
        "language": "English",
        "variant_digest": "a" * 64,
    }
    return TranscriberIdentity(**{**fields, **overrides})


def a_plan(
    start: int = RATE * 10,
    end: int = RATE * 20,
    *,
    request_id: str = "req_tx-a_000000480000",
    ownership: tuple[Ownership, ...] | None = None,
    pad: int | None = None,
) -> RequestPlan:
    padding = CONTEXT.pad if pad is None else pad
    return RequestPlan(
        request_id=request_id,
        track_id="tx-a",
        core_start_sample=start,
        core_end_sample=end,
        padded_start_sample=max(0, start - padding),
        padded_end_sample=min(CONTEXT.limit, end + padding),
        ownership=ownership
        or (
            Ownership(
                candidate_id="cand_tx-a_000000480000",
                start_sample=start,
                end_sample=end,
                session_start_sample=start * DECIMATION,
                session_end_sample=end * DECIMATION,
            ),
        ),
    )


def loud(_track: str, _start: int, n_samples: int) -> npt.NDArray[np.float32]:
    """A reader whose audio is uniform, so no split point is more attractive than another."""
    return np.full(n_samples, 0.25, dtype=np.float32)


def a_word(start: int, text: str) -> TranscribedWord:
    return TranscribedWord(start_sample=start, end_sample=start + 100, text=text)


def run(
    plans: list[RequestPlan],
    responses: dict[str, TranscriptionResult],
    tmp_path: Path,
    *,
    read: Any = loud,
    settings: TranscriptConfig | None = None,
    transcriber: ScriptedTranscriber | None = None,
    identity: TranscriberIdentity | None = None,
    cache: AsrCache | None = None,
    language: str = "English",
    glossary: str | None = None,
) -> Any:
    return transcribe_plans(
        plans,
        read=read,
        transcriber=transcriber or ScriptedTranscriber(responses),
        cache=cache or AsrCache(session_dir=tmp_path),
        identity=identity or an_identity(),
        context=CONTEXT,
        settings=settings or TranscriptConfig(),
        language=language,
        glossary=glossary,
    )


class TestAnOrdinaryRequest:
    def test_a_complete_response_is_returned_as_it_came(self, tmp_path: Path) -> None:
        plan = a_plan()
        outcome = run(
            [plan],
            {plan.request_id: TranscriptionResult(request_id=plan.request_id, text="hello")},
            tmp_path,
        )
        (only,) = outcome.outcomes
        assert only.text == "hello"
        assert only.truncated is False
        assert only.truncation_submissions == 0
        assert only.request_ids == (plan.request_id,)
        assert outcome.warnings == ()

    def test_the_submitted_window_is_the_padded_one(self, tmp_path: Path) -> None:
        plan = a_plan()
        transcriber = ScriptedTranscriber(
            {plan.request_id: TranscriptionResult(request_id=plan.request_id, text="hello")}
        )
        run([plan], {}, tmp_path, transcriber=transcriber)
        (request,) = transcriber.requests
        assert request.audio.start_sample == plan.padded_start_sample
        assert len(request.audio) == plan.padded_samples
        assert request.core_start_sample == plan.core_start_sample
        assert request.core_end_sample == plan.core_end_sample

    def test_the_language_and_glossary_reach_the_transcriber(self, tmp_path: Path) -> None:
        """The spec: force English by default, keep it configurable, pass `glossary.txt`."""
        plan = a_plan()
        transcriber = ScriptedTranscriber(
            {plan.request_id: TranscriptionResult(request_id=plan.request_id, text="hallo")}
        )
        run(
            [plan],
            {},
            tmp_path,
            transcriber=transcriber,
            language="German",
            glossary="Zephyrine\nWaterdeep",
        )
        (request,) = transcriber.requests
        assert request.language == "German"
        assert request.context == "Zephyrine\nWaterdeep"

    def test_an_absent_glossary_does_not_block_a_run(self, tmp_path: Path) -> None:
        plan = a_plan()
        transcriber = ScriptedTranscriber(
            {plan.request_id: TranscriptionResult(request_id=plan.request_id, text="hi")}
        )
        run([plan], {}, tmp_path, transcriber=transcriber, glossary=None)
        assert transcriber.requests[0].context is None

    def test_max_new_tokens_comes_from_the_identity(self, tmp_path: Path) -> None:
        plan = a_plan()
        transcriber = ScriptedTranscriber(
            {plan.request_id: TranscriptionResult(request_id=plan.request_id, text="hi")}
        )
        run([plan], {}, tmp_path, transcriber=transcriber, identity=an_identity(max_new_tokens=64))
        assert transcriber.requests[0].max_new_tokens == 64

    def test_a_second_run_is_served_from_the_cache(self, tmp_path: Path) -> None:
        plan = a_plan()
        responses = {plan.request_id: TranscriptionResult(request_id=plan.request_id, text="hi")}
        cache = AsrCache(session_dir=tmp_path)
        run([plan], responses, tmp_path, cache=cache)
        cache.commit()

        warm_cache = AsrCache(session_dir=tmp_path)
        transcriber = ScriptedTranscriber({})
        outcome = run([plan], {}, tmp_path, cache=warm_cache, transcriber=transcriber)
        assert outcome.outcomes[0].text == "hi"
        assert transcriber.requests == []
        assert (warm_cache.hits, warm_cache.misses) == (1, 0)


class TestTruncation:
    def _truncated(self, request_id: str) -> TranscriptionResult:
        return TranscriptionResult(request_id=request_id, text="cut off mid", truncated=True)

    def test_a_truncated_response_is_split_retried_and_stitched(self, tmp_path: Path) -> None:
        plan = a_plan()
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0", text="the first half"
            ),
            f"{plan.request_id}.1": TranscriptionResult(
                request_id=f"{plan.request_id}.1", text="and the second"
            ),
        }
        (only,) = run([plan], responses, tmp_path).outcomes
        assert only.text == "the first half and the second"
        assert only.truncated is False
        assert only.truncation_submissions == 2
        assert only.request_ids == (
            plan.request_id,
            f"{plan.request_id}.0",
            f"{plan.request_id}.1",
        )

    def test_every_child_request_obeys_the_cap(self, tmp_path: Path) -> None:
        plan = a_plan()
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0", text="a"
            ),
            f"{plan.request_id}.1": TranscriptionResult(
                request_id=f"{plan.request_id}.1", text="b"
            ),
        }
        transcriber = ScriptedTranscriber(responses)
        run([plan], {}, tmp_path, transcriber=transcriber)
        cap = CONTEXT.core_cap + 2 * CONTEXT.pad
        assert [len(request.audio) <= cap for request in transcriber.requests] == [True] * 3

    def test_the_children_tile_the_parents_core(self, tmp_path: Path) -> None:
        plan = a_plan()
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0", text="a"
            ),
            f"{plan.request_id}.1": TranscriptionResult(
                request_id=f"{plan.request_id}.1", text="b"
            ),
        }
        transcriber = ScriptedTranscriber(responses)
        run([plan], {}, tmp_path, transcriber=transcriber)
        _, left, right = transcriber.requests
        assert left.core_start_sample == plan.core_start_sample
        assert left.core_end_sample == right.core_start_sample
        assert right.core_end_sample == plan.core_end_sample

    def test_an_unresolvable_split_keeps_the_original_atomically(self, tmp_path: Path) -> None:
        """One half resolved and one still truncated is not a partial answer — it is the
        original plus a warning, because a partial stitch looks complete."""
        plan = a_plan()
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0", text="resolved"
            ),
            f"{plan.request_id}.1": self._truncated(f"{plan.request_id}.1"),
            f"{plan.request_id}.1.0": self._truncated(f"{plan.request_id}.1.0"),
            f"{plan.request_id}.1.1": self._truncated(f"{plan.request_id}.1.1"),
        }
        outcome = run(
            [plan], responses, tmp_path, settings=TranscriptConfig(max_truncation_retries=4)
        )
        (only,) = outcome.outcomes
        assert only.text == "cut off mid"
        assert only.truncated is True
        assert "resolved" not in only.text
        assert [note.code for note in outcome.warnings] == ["asr_truncation_unresolved"]

    def test_the_budget_counts_submissions_globally_not_depth(self, tmp_path: Path) -> None:
        """`max_truncation_retries=2` is two extra submissions, not two levels of splitting.

        Depth doubles: two levels of binary splitting is six extra calls, and a model that
        takes a minute each is why this is counted the way it is (ADR-0020).
        """
        plan = a_plan()
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": self._truncated(f"{plan.request_id}.0"),
            f"{plan.request_id}.1": self._truncated(f"{plan.request_id}.1"),
        }
        transcriber = ScriptedTranscriber(responses)
        outcome = run(
            [plan],
            {},
            tmp_path,
            transcriber=transcriber,
            settings=TranscriptConfig(max_truncation_retries=2),
        )
        assert len(transcriber.requests) == 3
        assert outcome.outcomes[0].truncated is True

    def test_no_retry_at_all_keeps_the_original_and_warns(self, tmp_path: Path) -> None:
        plan = a_plan()
        transcriber = ScriptedTranscriber({plan.request_id: self._truncated(plan.request_id)})
        outcome = run(
            [plan],
            {},
            tmp_path,
            transcriber=transcriber,
            settings=TranscriptConfig(max_truncation_retries=0),
        )
        assert len(transcriber.requests) == 1
        assert outcome.outcomes[0].text == "cut off mid"
        assert outcome.outcomes[0].truncated is True
        assert outcome.warnings[0].code == "asr_truncation_unresolved"
        assert "budget" in outcome.warnings[0].message

    def test_a_core_too_short_to_divide_is_not_divided(self, tmp_path: Path) -> None:
        """The rule that stops the recursion producing sub-word requests."""
        plan = a_plan(RATE * 10, RATE * 12)
        transcriber = ScriptedTranscriber({plan.request_id: self._truncated(plan.request_id)})
        outcome = run(
            [plan],
            {},
            tmp_path,
            transcriber=transcriber,
            settings=TranscriptConfig(min_split_core_ms=5_000),
        )
        assert len(transcriber.requests) == 1
        assert outcome.warnings[0].code == "asr_truncation_unresolved"
        assert "too short" in outcome.warnings[0].message

    def test_a_stitched_word_repeated_across_the_boundary_appears_once(
        self, tmp_path: Path
    ) -> None:
        """ADR-0020 rule 3: the same word at 99 in one child and 101 in the other.

        A rule keyed on the word's start alone keeps both copies, because each start falls in
        its own child's core. This is the case the plan review had to point out.
        """
        plan = a_plan()
        at = split_point(
            loud("tx-a", plan.padded_start_sample, plan.padded_samples),
            plan,
            settings=TranscriptConfig(),
        )
        assert at is not None
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0",
                text="back to Zephyrine",
                words=(
                    a_word(at - 400, "back"),
                    a_word(at - 200, "to"),
                    a_word(at - 60, "Zephyrine"),
                ),
                alignment_status="aligned",
            ),
            f"{plan.request_id}.1": TranscriptionResult(
                request_id=f"{plan.request_id}.1",
                text="Zephyrine now",
                words=(a_word(at - 20, "Zephyrine."), a_word(at + 200, "now")),
                alignment_status="aligned",
            ),
        }
        (only,) = run([plan], responses, tmp_path).outcomes
        assert [word.text for word in only.words] == ["back", "to", "Zephyrine", "now"]

    def test_a_different_word_at_the_boundary_is_kept(self, tmp_path: Path) -> None:
        plan = a_plan()
        at = split_point(
            loud("tx-a", plan.padded_start_sample, plan.padded_samples),
            plan,
            settings=TranscriptConfig(),
        )
        assert at is not None
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0",
                text="back to",
                words=(a_word(at - 200, "to"),),
                alignment_status="aligned",
            ),
            f"{plan.request_id}.1": TranscriptionResult(
                request_id=f"{plan.request_id}.1",
                text="Zephyrine",
                words=(a_word(at - 20, "Zephyrine"),),
                alignment_status="aligned",
            ),
        }
        (only,) = run([plan], responses, tmp_path).outcomes
        assert [word.text for word in only.words] == ["to", "Zephyrine"]

    def test_a_stitch_is_aligned_only_when_both_halves_were(self, tmp_path: Path) -> None:
        """A word list covering half a request would be serialized as covering all of it."""
        plan = a_plan()
        responses = {
            plan.request_id: self._truncated(plan.request_id),
            f"{plan.request_id}.0": TranscriptionResult(
                request_id=f"{plan.request_id}.0",
                text="first",
                words=(a_word(RATE * 11, "first"),),
                alignment_status="aligned",
            ),
            f"{plan.request_id}.1": TranscriptionResult(
                request_id=f"{plan.request_id}.1",
                text="second",
                alignment_status="segment_only",
            ),
        }
        (only,) = run([plan], responses, tmp_path).outcomes
        assert only.alignment_status == "segment_only"
        assert only.words == ()
        assert only.text == "first second"


class TestTheSplitPoint:
    def test_it_lands_in_the_quiet_band_rather_than_the_middle(self) -> None:
        """ "A natural low-energy boundary", as the gate puts it — not the midpoint."""
        plan = a_plan(RATE * 10, RATE * 30)
        quiet_at = plan.core_start_sample + RATE * 5

        def read(_track: str, start: int, n_samples: int) -> npt.NDArray[np.float32]:
            samples = np.full(n_samples, 0.5, dtype=np.float32)
            first = quiet_at - start
            samples[first : first + RATE] = 0.0
            return samples

        audio = read("tx-a", plan.padded_start_sample, plan.padded_samples)
        at = split_point(audio, plan, settings=TranscriptConfig())
        assert at is not None
        assert quiet_at <= at < quiet_at + RATE

    def test_a_tie_breaks_toward_the_middle(self) -> None:
        """Uniform audio has no quiet point; a long silence would otherwise push the split
        to whichever end argmin happened to reach first."""
        plan = a_plan(RATE * 10, RATE * 30)
        audio = loud("tx-a", plan.padded_start_sample, plan.padded_samples)
        at = split_point(audio, plan, settings=TranscriptConfig())
        middle = (plan.core_start_sample + plan.core_end_sample) // 2
        assert at is not None
        assert abs(at - middle) <= SPLIT_FRAME_SAMPLES

    def test_a_short_core_has_no_split_point(self) -> None:
        plan = a_plan(RATE * 10, RATE * 12)
        audio = loud("tx-a", plan.padded_start_sample, plan.padded_samples)
        assert split_point(audio, plan, settings=TranscriptConfig(min_split_core_ms=5_000)) is None

    def test_the_point_is_strictly_interior(self) -> None:
        """Guaranteed progress: neither child may be empty."""
        plan = a_plan(RATE * 10, RATE * 30)
        audio = loud("tx-a", plan.padded_start_sample, plan.padded_samples)
        at = split_point(audio, plan, settings=TranscriptConfig())
        assert at is not None
        assert plan.core_start_sample < at < plan.core_end_sample

    def test_it_is_deterministic(self) -> None:
        plan = a_plan(RATE * 10, RATE * 30)
        audio = loud("tx-a", plan.padded_start_sample, plan.padded_samples)
        settings = TranscriptConfig()
        assert split_point(audio, plan, settings=settings) == split_point(
            audio, plan, settings=settings
        )
