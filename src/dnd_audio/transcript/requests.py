"""From retained activity candidates to transcription requests.

The spec's instruction is to "transcribe retained VAD segments from their owner's lav rather
than transcribing the six full-length files blindly", merging very short adjacent regions and
capping a request well below the adapter's limit. Four things that sounds simpler than it is:

**Merging joins the audio; it does not join ownership** (ADR-0017). A request may cover several
adjacent candidates so the model hears a sentence with its pauses rather than eight fragments,
but each candidate keeps its own ownership interval inside that request. One retained candidate
still produces one segment, `source_candidate_id` stays singular, and post-ASR collapse can
still look up the exact pairwise evidence M3 measured.

**The cap applies to the padded waveform**, not to the core. `max_segment_s` is what is
actually submitted, so the core is cut to leave room for padding on both sides — and when the
configured padding is itself larger than the cap allows, the padding shrinks rather than the
cap being quietly exceeded.

**Nothing here reads audio.** A plan is intervals and ids; the samples are attached one
request at a time when it is submitted. Building every request's audio up front would hold
the session in memory, which is exactly what INV-07 forbids and exactly what a plan that
returned `TranscriptionRequest` objects would invite.

**Everything is on the 16 kHz derivative grid**, the grid the model consumes and the detector
decided on (ADR-0017). Each ownership interval carries its 48 kHz session bounds too, taken
from the graph rather than recomputed, so nothing in this project converts between the grids
twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dnd_audio.artifacts.activity import ActivityCandidate, ActivityGraph
from dnd_audio.config import AsrConfig, TranscriptConfig
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.resample import to_source_sample

__all__ = [
    "Ownership",
    "PlanContext",
    "RequestPlan",
    "core_cap_samples",
    "plan_context",
    "plan_requests",
    "request_id",
    "slice_ownership",
    "split_plan",
]

#: Width of the zero-padded sample position in a request id, matching `candidate_id`'s, so
#: request ids sort lexically in the same order they sort numerically.
_ID_WIDTH: Final = 12


def request_id(track_id: str, session_start_sample: int) -> str:
    """The deterministic id of a request, from its track and its core's session position.

    Quoted on the 48 kHz session grid rather than the derivative one, so a request id and a
    candidate id name the same instant in the same units (INV-02).
    """
    return f"req_{track_id}_{session_start_sample:0{_ID_WIDTH}d}"


@dataclass(frozen=True, slots=True)
class Ownership:
    """The part of one activity candidate that one request owns.

    Usually the whole candidate. It is a *part* when a candidate longer than the cap had to
    be cut across several requests, which is why the segment assembler groups ownership by
    ``candidate_id`` rather than assuming one of each.
    """

    candidate_id: str
    #: Half-open, on the 16 kHz derivative grid — the grid words come back on.
    start_sample: int
    end_sample: int
    #: The same interval on the canonical 48 kHz session grid, and the candidate's own bounds.
    session_start_sample: int
    session_end_sample: int

    def __post_init__(self) -> None:
        if self.end_sample <= self.start_sample:
            message = f"ownership of {self.candidate_id} spans an empty interval"
            raise ValueError(message)

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """One transcription request, before any audio has been read.

    ``core`` is what this request owns and ``padded`` is what will be submitted. The two
    differ by the padding that exists so the model does not clip the first and last word;
    a word inside the padding and inside no ownership interval is dropped (ADR-0020).
    """

    request_id: str
    track_id: str
    #: Half-open, derivative grid.
    core_start_sample: int
    core_end_sample: int
    padded_start_sample: int
    padded_end_sample: int
    ownership: tuple[Ownership, ...]

    def __post_init__(self) -> None:
        if self.core_end_sample <= self.core_start_sample:
            message = f"request {self.request_id} has an empty core"
            raise ValueError(message)
        if self.padded_start_sample > self.core_start_sample:
            message = (
                f"request {self.request_id} pads to {self.padded_start_sample}, inside its own "
                f"core at {self.core_start_sample}"
            )
            raise ValueError(message)
        if self.padded_end_sample < self.core_end_sample:
            message = (
                f"request {self.request_id} pads to {self.padded_end_sample}, inside its own "
                f"core at {self.core_end_sample}"
            )
            raise ValueError(message)
        if not self.ownership:
            message = f"request {self.request_id} owns nothing, so nothing could use its words"
            raise ValueError(message)

    @property
    def padded_samples(self) -> int:
        """What will actually be submitted, and what `max_segment_s` bounds."""
        return self.padded_end_sample - self.padded_start_sample

    @property
    def core_samples(self) -> int:
        return self.core_end_sample - self.core_start_sample


def core_cap_samples(asr: AsrConfig, transcript: TranscriptConfig) -> tuple[int, int]:
    """``(core_cap, pad)`` in derivative samples, both honouring `max_segment_s`.

    The cap is on the padded waveform, so the core gets what is left after padding both ends.
    When the configured padding is so large that no core would fit — legal configuration, since
    `pad_ms` and `max_segment_s` are bounded independently — the *padding* is what gives way.
    A submitted waveform over the cap is a request the adapter refuses; a shorter pad is a
    slightly worse chance of recovering a boundary word.
    """
    segment_cap = asr.max_segment_s * DERIVATIVE_SAMPLE_RATE
    pad = transcript.pad_ms * DERIVATIVE_SAMPLE_RATE // 1000
    if segment_cap - 2 * pad < 1:
        pad = max(0, (segment_cap - 1) // 2)
    return segment_cap - 2 * pad, pad


@dataclass(frozen=True, slots=True)
class PlanContext:
    """What a request needs to know about the session it sits in.

    Carried rather than recomputed because a truncation retry builds *new* requests from an
    old one's core, and it needs the same padding, the same session bound, and the same grid
    ratio the original was built with. Deriving them a second time at the retry site is how
    the two quietly stop agreeing.
    """

    #: Derivative samples of padding on each side of a core.
    pad: int
    #: The session's length on the derivative grid: no padded window may reach past it.
    limit: int
    #: 48 kHz session samples per derivative sample. Three, everywhere in this project.
    decimation: int
    #: The longest core that leaves room for padding inside `max_segment_s`.
    core_cap: int


def plan_context(
    graph: ActivityGraph, *, asr: AsrConfig, transcript: TranscriptConfig
) -> PlanContext:
    """The padding, bounds, and cap every request in one session is built against."""
    decimation = graph.sample_rate // graph.derivative_sample_rate
    cap, pad = core_cap_samples(asr, transcript)
    return PlanContext(
        pad=pad,
        limit=-(-graph.duration_samples // decimation),
        decimation=decimation,
        core_cap=cap,
    )


def slice_ownership(
    ownership: tuple[Ownership, ...], start: int, end: int, decimation: int
) -> tuple[Ownership, ...]:
    """The parts of ``ownership`` inside ``[start, end)``, keeping every identity.

    An interval the boundary cuts through becomes a shorter interval with the same
    ``candidate_id``, which is what lets a candidate split across a truncation retry still
    assemble into one segment. Session bounds at an *untouched* edge are the candidate's own;
    only an edge this slice actually moved is reconverted, and then through the exact
    direction (`to_source_sample`) rather than by inverting the covering rule.
    """
    sliced: list[Ownership] = []
    for item in ownership:
        first = max(item.start_sample, start)
        last = min(item.end_sample, end)
        if last <= first:
            continue
        sliced.append(
            Ownership(
                candidate_id=item.candidate_id,
                start_sample=first,
                end_sample=last,
                session_start_sample=(
                    item.session_start_sample
                    if first == item.start_sample
                    else to_source_sample(first, decimation)
                ),
                session_end_sample=(
                    item.session_end_sample
                    if last == item.end_sample
                    else to_source_sample(last, decimation)
                ),
            )
        )
    return tuple(sliced)


def split_plan(plan: RequestPlan, at: int, context: PlanContext) -> tuple[RequestPlan, RequestPlan]:
    """Two requests covering ``plan``'s core, divided at ``at``, each padded in its own right.

    Used only by truncation retry (ADR-0020). The children's cores tile the parent's exactly,
    so nothing is lost between them and nothing is covered twice — which is what makes the
    stitch a concatenation rather than a merge. Their ids extend the parent's, so a records
    file says which request a word came from and how it got there.
    """
    if not plan.core_start_sample < at < plan.core_end_sample:
        message = (
            f"cannot split {plan.request_id} at {at}: the point must be strictly inside its "
            f"core [{plan.core_start_sample}, {plan.core_end_sample}), or a child would be "
            f"empty and the retry would not make progress"
        )
        raise ValueError(message)

    return (
        _child(plan, 0, plan.core_start_sample, at, context),
        _child(plan, 1, at, plan.core_end_sample, context),
    )


def _child(
    plan: RequestPlan, index: int, start: int, end: int, context: PlanContext
) -> RequestPlan:
    return RequestPlan(
        request_id=f"{plan.request_id}.{index}",
        track_id=plan.track_id,
        core_start_sample=start,
        core_end_sample=end,
        padded_start_sample=max(0, start - context.pad),
        padded_end_sample=min(context.limit, end + context.pad),
        ownership=slice_ownership(plan.ownership, start, end, context.decimation),
    )


def plan_requests(
    graph: ActivityGraph, *, asr: AsrConfig, transcript: TranscriptConfig
) -> list[RequestPlan]:
    """Every request one session needs, in canonical order.

    Retained candidates only: a suppressed candidate is another track's voice, and
    transcribing it is exactly the extra work M3's bleed gate exists to avoid. An
    **ambiguous** candidate is retained and therefore planned — the flag means the numbers
    said bleed and the track-level veto overrode them (ADR-0014), which is a reason to look
    hard at it after ASR, never a reason to skip it before.

    Ordered by ``(core start, track)``, the same canonical order the graph itself uses, so a
    scripted transcriber and a real one see the same sequence.
    """
    context = plan_context(graph, asr=asr, transcript=transcript)
    gap = transcript.merge_gap_ms * DERIVATIVE_SAMPLE_RATE // 1000

    plans: list[RequestPlan] = []
    for track in graph.tracks:
        retained = graph.retained(track.track_id)
        for group in _merge(retained, gap):
            for ownership in _within_cap(group, context.core_cap, context.decimation):
                plans.append(_plan(track.track_id, ownership, pad=context.pad, limit=context.limit))
    return sorted(plans, key=lambda plan: (plan.core_start_sample, plan.track_id))


def _merge(candidates: list[ActivityCandidate], gap: int) -> list[list[Ownership]]:
    """Group adjacent candidates on one track into the requests they will share.

    The gap is measured on the derivative grid between one candidate's end and the next's
    start. Candidates arrive in graph order, which is sorted by start sample, and a track's
    candidates are disjoint after M3's own merging — so a single pass is enough.
    """
    groups: list[list[Ownership]] = []
    for candidate in candidates:
        ownership = Ownership(
            candidate_id=candidate.candidate_id,
            start_sample=candidate.derivative_start_sample,
            end_sample=candidate.derivative_end_sample,
            session_start_sample=candidate.start_sample,
            session_end_sample=candidate.end_sample,
        )
        if groups and ownership.start_sample - groups[-1][-1].end_sample <= gap:
            groups[-1].append(ownership)
        else:
            groups.append([ownership])
    return groups


def _within_cap(group: list[Ownership], cap: int, decimation: int) -> list[list[Ownership]]:
    """Cut one merged group into requests no longer than ``cap``.

    Two rules, in this order:

    1. **Prefer candidate boundaries.** A group is closed before the candidate that would push
       it over, so an ordinary long conversation is cut where somebody stopped talking.
    2. **A candidate longer than the cap is divided.** Into as few equal pieces as fit, by
       integer arithmetic rather than by repeatedly subtracting the cap: equal pieces keep the
       last request from being a sliver, and a sliver at the end of a long utterance is the
       request most likely to transcribe to nothing useful. Its ownership pieces all carry the
       same ``candidate_id``, and the segment assembler stitches them back together.
    """
    requests: list[list[Ownership]] = []
    current: list[Ownership] = []
    for ownership in group:
        for piece in _divide(ownership, cap, decimation):
            length = piece.end_sample - piece.start_sample
            if current and piece.end_sample - current[0].start_sample > cap:
                requests.append(current)
                current = []
            if length > cap:  # pragma: no cover - `_divide` has already bounded every piece
                message = f"a piece of {piece.candidate_id} is {length} samples, over the cap"
                raise ValueError(message)
            current.append(piece)
    if current:
        requests.append(current)
    return requests


def _divide(ownership: Ownership, cap: int, decimation: int) -> list[Ownership]:
    """One candidate as pieces no longer than ``cap``, each keeping its identity.

    Interior boundaries convert through `to_source_sample` — the one direction that is exact,
    since output ``k`` sits at input ``k * decimation``. The outer edges keep the candidate's
    *own* session bounds rather than being reconverted: the graph's 48 kHz interval covers its
    derivative one (the start floors and the end ceils), so converting back would shrink the
    candidate by up to two samples at each end, which is the trap M2's closeout names.
    """
    total = ownership.n_samples
    if total <= cap:
        return [ownership]

    pieces = -(-total // cap)
    divided: list[Ownership] = []
    for index in range(pieces):
        start = ownership.start_sample + index * total // pieces
        end = ownership.start_sample + (index + 1) * total // pieces
        divided.append(
            Ownership(
                candidate_id=ownership.candidate_id,
                start_sample=start,
                end_sample=end,
                session_start_sample=(
                    ownership.session_start_sample
                    if index == 0
                    else to_source_sample(start, decimation)
                ),
                session_end_sample=(
                    ownership.session_end_sample
                    if index == pieces - 1
                    else to_source_sample(end, decimation)
                ),
            )
        )
    return divided


def _plan(track_id: str, ownership: list[Ownership], *, pad: int, limit: int) -> RequestPlan:
    """One request over an ordered, non-empty run of ownership intervals."""
    core_start = ownership[0].start_sample
    core_end = ownership[-1].end_sample
    return RequestPlan(
        request_id=request_id(track_id, ownership[0].session_start_sample),
        track_id=track_id,
        core_start_sample=core_start,
        core_end_sample=core_end,
        # Clamped to the session, so a candidate at either end is padded with the audio that
        # exists rather than with a window reaching past it. The reader would return silence
        # there, which is harmless, but a request claiming to start before sample zero is not.
        padded_start_sample=max(0, core_start - pad),
        padded_end_sample=min(limit, core_end + pad),
        ownership=tuple(ownership),
    )
