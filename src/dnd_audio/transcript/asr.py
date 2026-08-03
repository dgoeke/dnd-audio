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
from dataclasses import dataclass, field
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
    "RequestOutcome",
    "split_point",
    "transcribe_plans",
]

#: Frames the energy search for a split point works in: 512 derivative samples is 32 ms, long
#: enough that one glottal pulse does not look like silence and short enough to land a split
#: inside a pause. It is the same number the detector's frame happens to be, and deliberately
#: not imported from there — this is a search granularity, not a model's protocol.
SPLIT_FRAME_SAMPLES: Final = 512

#: ``(track_id, start, n_samples)`` of the 16 kHz derivative, in that track's own samples.
#: The seam that keeps this module testable with synthetic audio and every read bounded.
AudioReader = Callable[[str, int, int], npt.NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    """What one planned request finally produced, retries included."""

    plan: RequestPlan
    text: str
    #: On the 16 kHz derivative grid, session-absolute. Empty unless ``alignment_status`` is
    #: ``aligned``, which is the seam's own rule.
    words: tuple[TranscribedWord, ...]
    alignment_status: AlignmentStatus
    #: Every request id that contributed, the original first, in submission order.
    request_ids: tuple[str, ...]
    #: Attempts beyond the original. Zero in the ordinary case, and recorded so "bounded" is
    #: checkable rather than asserted.
    truncation_submissions: int
    #: Still truncated after everything the budget allowed. The text is the original's.
    truncated: bool


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
            truncation_submissions=spent,
            truncated=False,
        )

    return RequestOutcome(
        plan=plan,
        text=text,
        words=(*left.words, *_without_boundary_repeat(left.words, right.words)),
        alignment_status="aligned",
        request_ids=request_ids,
        truncation_submissions=spent,
        truncated=False,
    )


def _without_boundary_repeat(
    left: tuple[TranscribedWord, ...], right: tuple[TranscribedWord, ...]
) -> tuple[TranscribedWord, ...]:
    """``right`` with a word that repeats ``left``'s last one across the split removed.

    ADR-0020's rule 3, and the only place it applies: two children's cores are the one case
    where ownership intervals are genuinely adjacent, so it is the only boundary at which the
    same physical word can be returned on both sides at slightly different times. Comparison
    is on the comparison key, so capitalization or a trailing comma cannot hide the repeat.
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
