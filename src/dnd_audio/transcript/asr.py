"""Submitting requests, surviving a truncated answer, and deciding who owns each word.

Three responsibilities, and the second two are where the damage happens.

**Submission is one request at a time.** A plan carries no audio; the samples for one padded
window are read, hashed, submitted, and dropped before the next plan is touched. Nothing here
holds more than one request's audio, which is what keeps a four-hour session inside INV-07 —
and `tests/test_memory.py`'s technique, an ordered event log asserting a transcription happens
before the last read, is what proves it over the composed path rather than over one function.

**Truncation is bounded by a submission budget, not by a recursion depth** (ADR-0020). Depth
doubles: a "bounded" retry configured as 3 could mean fifteen calls to a model that takes a
minute each. The budget counts *attempts* — cached or not, so that a warm run and a cold run
produce the same records (INV-02) — and stops the recursion when it runs out. A child core
below `min_split_core_ms` is not split again, so the recursion terminates on short input even
with budget to spare, and the split point is strictly interior, so every retry makes progress.

The fallback is **atomic per original request**: if any descendant is still truncated when the
budget runs out, the original response is kept with a warning naming it. A partially stitched
result looks complete and is missing an unknown amount of speech in the middle, which is worse
than one response that is visibly truncated and says so.

**A word belongs to the ownership interval containing its start**, half-open, and a word inside
padding but inside no ownership interval is dropped — padding is context, never content. At a
truncation stitch boundary, the one place two ownership intervals are genuinely adjacent, a
word from the later child that repeats the earlier child's last word and overlaps it in time is
dropped as well: a start-based rule alone keeps both copies when the model reports the same
word at 99 in one request and 101 in the other, which is the failure a plan review had to
point out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import numpy.typing as npt

from dnd_audio.artifacts.records import TranscriberIdentity, TranscriptNote
from dnd_audio.artifacts.transcript import AlignmentStatus
from dnd_audio.config import TranscriptConfig
from dnd_audio.interfaces import (
    AudioWindow,
    TranscribedWord,
    Transcriber,
    TranscriptionRequest,
    TranscriptionResult,
)
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.transcript.cache import AsrCache, asr_identity, audio_sha256
from dnd_audio.transcript.normalize import comparison_key
from dnd_audio.transcript.requests import PlanContext, RequestPlan, split_plan

__all__ = [
    "SPLIT_FRAME_SAMPLES",
    "AsrOutcome",
    "AudioReader",
    "RequestContribution",
    "RequestOutcome",
    "split_point",
    "transcribe_plans",
    "without_boundary_repeat",
]

#: Frames the energy search for a split point works in: 512 derivative samples is 32 ms, long
#: enough that one glottal pulse does not look like silence and short enough to land a split
#: inside a pause. It is the same number the detector's frame happens to be, and deliberately
#: not imported from there — this is a search granularity, not a model's protocol.
#:
#: Both halves of that claim are guesses about real speech at a real table rather than measured
#: facts, so the number is registered with the rest of M4's request-shaping constants
#: (**OQ-018**): too long and the search cannot see a short pause at all, too short and a
#: single glottal pulse reads as silence and the split lands mid-word.
SPLIT_FRAME_SAMPLES: Final = 512

#: ``(track_id, start, n_samples)`` of the 16 kHz derivative, in that track's own samples.
#: The seam that keeps this module testable with synthetic audio and every read bounded.
AudioReader = Callable[[str, int, int], npt.NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class RequestContribution:
    """One submitted request whose answer contributes to the final outcome.

    A normal outcome has one. A resolved truncation has the leaf submissions, not the
    truncated parent response that was discarded. Keeping the plan beside its returned words
    preserves the actual padded bounds and sliced ownership at every retry seam (ADR-0033).
    """

    plan: RequestPlan
    words: tuple[TranscribedWord, ...]


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    """What one planned request finally produced, retries included."""

    plan: RequestPlan
    text: str
    #: On the 16 kHz derivative grid, session-absolute. Empty unless ``alignment_status`` is
    #: ``aligned``, which is the seam's own rule.
    words: tuple[TranscribedWord, ...]
    alignment_status: AlignmentStatus
    #: Every request id attempted while resolving this result, the original first, in
    #: submission order. The contributor records below distinguish answers actually retained.
    request_ids: tuple[str, ...]
    #: The actual submitted leaf requests whose answers make up ``text``/``words``. This is
    #: piece-specific because a retry child's padding and ownership differ from its parent.
    contributing_submissions: tuple[RequestContribution, ...]
    #: Attempts beyond the original. Zero in the ordinary case, and recorded so "bounded" is
    #: checkable rather than asserted.
    truncation_submissions: int
    #: Still truncated after everything the budget allowed. The text is the original's.
    truncated: bool

    def __post_init__(self) -> None:
        if not self.contributing_submissions:
            raise ValueError(
                f"request outcome {self.plan.request_id} has no contributing submission"
            )
        known = set(self.request_ids)
        for contribution in self.contributing_submissions:
            if contribution.plan.request_id not in known:
                message = (
                    f"request outcome {self.plan.request_id} keeps contribution "
                    f"{contribution.plan.request_id} outside its request lineage"
                )
                raise ValueError(message)
            if contribution.plan.track_id != self.plan.track_id:
                message = (
                    f"request outcome {self.plan.request_id} mixes contribution track "
                    f"{contribution.plan.track_id} with {self.plan.track_id}"
                )
                raise ValueError(message)
        if self.alignment_status == "aligned":
            contributed_words = tuple(
                word
                for contribution in self.contributing_submissions
                for word in contribution.words
            )
            if contributed_words != self.words:
                message = (
                    f"request outcome {self.plan.request_id}'s aligned words disagree with its "
                    "submission-specific occurrences"
                )
                raise ValueError(message)


@dataclass(frozen=True, slots=True)
class AsrOutcome:
    """Every request's outcome, and what an operator should be told about them."""

    outcomes: tuple[RequestOutcome, ...]
    warnings: tuple[TranscriptNote, ...]


@dataclass
class _Budget:
    """Attempts still available for resolving truncation, across one original request.

    Counts cached attempts too. A budget that only counted model calls would let a warm run
    explore further than a cold one and produce different records from the same inputs, which
    is exactly the class of nondeterminism INV-02 exists to keep out of artifacts.
    """

    remaining: int

    def spend(self, count: int) -> bool:
        if self.remaining < count:
            return False
        self.remaining -= count
        return True


@dataclass
class _Session:
    """Everything one pass over the plans needs, gathered so the recursion stays readable."""

    read: AudioReader
    transcriber: Transcriber
    cache: AsrCache
    identity: TranscriberIdentity
    context: PlanContext
    settings: TranscriptConfig
    language: str
    glossary: str | None
    warnings: list[TranscriptNote] = field(default_factory=list)


def transcribe_plans(
    plans: list[RequestPlan],
    *,
    read: AudioReader,
    transcriber: Transcriber,
    cache: AsrCache,
    identity: TranscriberIdentity,
    context: PlanContext,
    settings: TranscriptConfig,
    language: str,
    glossary: str | None = None,
) -> AsrOutcome:
    """Transcribe every plan, in order, resolving truncation within the configured budget."""
    session = _Session(
        read=read,
        transcriber=transcriber,
        cache=cache,
        identity=identity,
        context=context,
        settings=settings,
        language=language,
        glossary=glossary,
    )
    outcomes = [
        _resolve(plan, session, _Budget(settings.max_truncation_retries), original=True)
        for plan in plans
    ]
    return AsrOutcome(
        outcomes=tuple(outcomes),
        warnings=tuple(
            sorted(session.warnings, key=lambda note: (note.code, note.path or "", note.message))
        ),
    )


def _resolve(
    plan: RequestPlan, session: _Session, budget: _Budget, *, original: bool
) -> RequestOutcome:
    """One request, split and retried while it comes back truncated and the budget allows.

    Only the **original** request warns. The fallback is atomic per original request, so a
    descendant that could not be resolved is why the original was kept — reporting each level
    of a failed recursion separately would put four lines in front of an operator about one
    utterance.
    """
    audio = session.read(plan.track_id, plan.padded_start_sample, plan.padded_samples)
    result = _submit(plan, audio, session)
    if not result.truncated:
        return _outcome(plan, result, request_ids=(plan.request_id,), spent=0)

    at = split_point(audio, plan, settings=session.settings)
    if at is None or not budget.spend(2):
        if original:
            session.warnings.append(_unresolved(plan, at is None))
        return _outcome(plan, result, request_ids=(plan.request_id,), spent=0, truncated=True)

    left_plan, right_plan = split_plan(plan, at, session.context)
    left = _resolve(left_plan, session, budget, original=False)
    right = _resolve(right_plan, session, budget, original=False)
    spent = 2 + left.truncation_submissions + right.truncation_submissions
    ids = (plan.request_id, *left.request_ids, *right.request_ids)

    if left.truncated or right.truncated:
        # Atomic: the original, not a mixture of a resolved half and a truncated one. A
        # partial stitch looks complete and is missing an unknown amount of speech.
        if original:
            session.warnings.append(_unresolved(plan, False))
        return _outcome(plan, result, request_ids=ids, spent=spent, truncated=True)

    return _stitch(plan, left, right, request_ids=ids, spent=spent)


def _submit(
    plan: RequestPlan, audio: npt.NDArray[np.float32], session: _Session
) -> TranscriptionResult:
    """The cache, then the model. The request is built here and nowhere else."""
    key = asr_identity(
        audio_hash=audio_sha256(audio),
        request_id=plan.request_id,
        track_id=plan.track_id,
        core_start_sample=plan.core_start_sample,
        core_end_sample=plan.core_end_sample,
        transcriber=session.identity,
    )
    cached = session.cache.get(key)
    if cached is not None:
        return cached.as_result()

    request = TranscriptionRequest(
        request_id=plan.request_id,
        audio=AudioWindow(
            track_id=plan.track_id,
            sample_rate=DERIVATIVE_SAMPLE_RATE,
            start_sample=plan.padded_start_sample,
            samples=audio,
        ),
        core_start_sample=plan.core_start_sample,
        core_end_sample=plan.core_end_sample,
        language=session.language,
        context=session.glossary,
        max_new_tokens=session.identity.max_new_tokens,
    )
    result = session.transcriber.transcribe(request)
    session.cache.publish(key, result)
    return result


def split_point(
    audio: npt.NDArray[np.float32], plan: RequestPlan, *, settings: TranscriptConfig
) -> int | None:
    """The quietest interior frame boundary of ``plan``'s core, or ``None`` if it has none.

    ``None`` means the core is too short to divide into two children that both clear
    `min_split_core_ms` — the rule that stops the recursion from producing sub-word requests.
    It never means "no quiet point exists": among the boundaries that *are* allowed, the
    quietest always exists, and the tie is broken toward the middle so a long silence does not
    push the split to one end. Earlier wins a remaining tie, so the choice is not a function
    of NumPy's argmin order.

    Args:
        audio: The whole padded window, as submitted.
    """
    minimum = max(1, settings.min_split_core_ms * DERIVATIVE_SAMPLE_RATE // 1000)
    first = plan.core_start_sample + minimum
    last = plan.core_end_sample - minimum
    if last <= first:
        return None

    offset = plan.core_start_sample - plan.padded_start_sample
    boundaries = range(_round_up(first, SPLIT_FRAME_SAMPLES), last + 1, SPLIT_FRAME_SAMPLES)
    if not boundaries:
        # The window between the two minimum-length bounds is narrower than one frame; the
        # midpoint is still a legal split and still makes progress.
        return (first + last) // 2

    middle = (plan.core_start_sample + plan.core_end_sample) // 2
    energies = [
        (
            _energy(audio, offset + position - plan.core_start_sample),
            abs(position - middle),
            position,
        )
        for position in boundaries
    ]
    return min(energies)[2]


def _energy(audio: npt.NDArray[np.float32], at: int) -> float:
    """Mean square of the frame beginning at ``at``, clamped to what exists."""
    window = audio[max(0, at) : max(0, at) + SPLIT_FRAME_SAMPLES]
    if window.size == 0:
        return 0.0
    return float(np.mean(np.square(window.astype(np.float64))))


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _outcome(
    plan: RequestPlan,
    result: TranscriptionResult,
    *,
    request_ids: tuple[str, ...],
    spent: int,
    truncated: bool = False,
) -> RequestOutcome:
    return RequestOutcome(
        plan=plan,
        text=result.text,
        words=result.words,
        alignment_status=result.alignment_status,
        request_ids=request_ids,
        contributing_submissions=(RequestContribution(plan=plan, words=result.words),),
        truncation_submissions=spent,
        truncated=truncated,
    )


def _stitch(
    plan: RequestPlan,
    left: RequestOutcome,
    right: RequestOutcome,
    *,
    request_ids: tuple[str, ...],
    spent: int,
) -> RequestOutcome:
    """Join two resolved halves into the answer their parent request was asked for.

    Alignment is all or nothing: a stitched result is ``aligned`` only when both halves were,
    because a word list covering half a request would be serialized as though it covered the
    whole of it. The text is joined either way, since text is never the thing to discard.
    """
    aligned = left.alignment_status == "aligned" and right.alignment_status == "aligned"
    text = " ".join(part for part in (left.text.strip(), right.text.strip()) if part)
    if not aligned:
        status: AlignmentStatus = (
            "segment_only"
            if "segment_only" in (left.alignment_status, right.alignment_status)
            else "not_attempted"
        )
        return RequestOutcome(
            plan=plan,
            text=text,
            words=(),
            alignment_status=status,
            request_ids=request_ids,
            contributing_submissions=(
                *left.contributing_submissions,
                *right.contributing_submissions,
            ),
            truncation_submissions=spent,
            truncated=False,
        )

    right_words = without_boundary_repeat(left.words, right.words)
    right_contributions = right.contributing_submissions
    if len(right_words) != len(right.words):
        right_contributions = _without_first_contributed_word(right_contributions)
    return RequestOutcome(
        plan=plan,
        text=text,
        words=(*left.words, *right_words),
        alignment_status="aligned",
        request_ids=request_ids,
        contributing_submissions=(*left.contributing_submissions, *right_contributions),
        truncation_submissions=spent,
        truncated=False,
    )


def _without_first_contributed_word(
    contributions: tuple[RequestContribution, ...],
) -> tuple[RequestContribution, ...]:
    """Mirror a boundary-repeat removal in the leaf occurrence that supplied the word."""
    found: list[RequestContribution] = []
    removed = False
    for contribution in contributions:
        if not removed and contribution.words:
            found.append(replace(contribution, words=contribution.words[1:]))
            removed = True
        else:
            found.append(contribution)
    if not removed:  # pragma: no cover - caller observed a removed word in the aggregate
        raise AssertionError("a stitched boundary word had no contributing submission")
    return tuple(found)


def without_boundary_repeat(
    left: tuple[TranscribedWord, ...], right: tuple[TranscribedWord, ...]
) -> tuple[TranscribedWord, ...]:
    """``right`` with a word that repeats ``left``'s last one across the split removed.

    ADR-0020's rule 3. Comparison is on the comparison key, so capitalization or a trailing
    comma cannot hide the repeat, and the two must also overlap in time — a genuinely repeated
    word ("no, no") is two words at two positions and must survive.

    ADR-0020 originally called a truncation stitch "the only place two ownership intervals are
    genuinely adjacent", and that was wrong. A candidate longer than `max_segment_s` is cut by
    `requests._divide` into pieces that tile it exactly, each submitted as its own
    independently padded request — adjacent in precisely the same way, and reassembled in
    `segments._owned_words` rather than here. So this rule lives in one function called from
    both places (M4's verify phase, found by independent review).
    """
    if not left or not right:
        return right
    last, first = left[-1], right[0]
    repeated = comparison_key(last.text) == comparison_key(first.text)
    overlapping = first.start_sample < last.end_sample and last.start_sample < first.end_sample
    return right[1:] if repeated and overlapping else right


def _unresolved(plan: RequestPlan, too_short: bool) -> TranscriptNote:
    reason = (
        "it is too short to split into two halves that both clear `transcript.min_split_core_ms`"
        if too_short
        else "the retry budget in `transcript.max_truncation_retries` ran out"
    )
    return TranscriptNote(
        code="asr_truncation_unresolved",
        message=(
            f"{plan.request_id} came back truncated and could not be resolved because {reason}. "
            f"The original response is kept in full; the utterance may be cut off mid-sentence. "
            f"Raising `asr.max_new_tokens` is the fix if this recurs."
        ),
        path=plan.track_id,
    )
