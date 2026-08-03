"""From request outcomes to one draft segment per retained candidate.

This is where ADR-0017's "requests merge; ownership does not" becomes code. A request may
have covered several adjacent candidates, and a candidate longer than the cap may have been
cut across several requests. Both are undone here, so that what comes out is one segment per
retained candidate, carrying only what that candidate owns.

**Words decide the text when there are words.** A merged request returns one string covering
several candidates, and there is no honest way to divide a string. What *is* divisible is the
word list: each word goes to the ownership interval containing its start, and a segment's text
is the words it owns. When no word times came back the string cannot be divided, so the
candidates that shared that request share one segment — the degenerate case ADR-0017 names,
and the only reason `source_candidate_ids` is a list.

**A candidate that owns no words produces no segment.** A VAD fires on coughs and door
closes; a retained candidate the model found nothing in is ordinary, not an error. They are
counted into one warning rather than one per candidate, so an operator can see that it
happened without reading six hundred lines about it.

Times cross to the canonical 48 kHz grid here and nowhere else in this package, through
`to_source_sample`, and are clamped into the candidate's own bounds — the graph's 48 kHz
interval *covers* its derivative one, so an unclamped conversion of the very first derivative
sample can land up to two samples before the candidate starts.
"""

from __future__ import annotations

from dataclasses import dataclass

from dnd_audio.artifacts.records import TranscriptNote, WordRecord
from dnd_audio.artifacts.transcript import AlignmentStatus
from dnd_audio.interfaces import TranscribedWord
from dnd_audio.timeline.resample import to_source_sample
from dnd_audio.transcript.asr import RequestOutcome
from dnd_audio.transcript.normalize import normalize_text
from dnd_audio.transcript.requests import Ownership

__all__ = ["SegmentDraft", "draft_segments"]


@dataclass(frozen=True, slots=True)
class SegmentDraft:
    """One candidate's transcript, before collapse has decided whether to keep it."""

    #: Usually one. More only where a wordless result covering a merged request could not be
    #: divided, which is the case that forces those candidates to share a segment.
    candidate_ids: tuple[str, ...]
    track_id: str
    #: Where the speech is, on the 48 kHz session grid: the span of the owned words, or the
    #: ownership interval when there are none.
    start_sample: int
    end_sample: int
    ownership_start_sample: int
    ownership_end_sample: int
    text: str
    words: tuple[WordRecord, ...]
    alignment_status: AlignmentStatus
    request_ids: tuple[str, ...]
    truncation_submissions: int


def draft_segments(
    outcomes: tuple[RequestOutcome, ...], *, decimation: int
) -> tuple[list[SegmentDraft], list[TranscriptNote]]:
    """Every retained candidate's draft segment, in canonical order, plus what to warn about."""
    groups = _grouped_candidates(outcomes)
    contributions = _contributions(outcomes, groups)

    drafts: list[SegmentDraft] = []
    empty = 0
    for candidates in sorted(groups, key=sorted):
        draft = _draft(candidates, contributions[candidates], decimation=decimation)
        if draft is None:
            empty += 1
            continue
        drafts.append(draft)

    drafts.sort(key=lambda item: (item.start_sample, item.track_id, item.candidate_ids))
    return drafts, [*_notes(empty), *_alignment_notes(drafts)]


def _grouped_candidates(outcomes: tuple[RequestOutcome, ...]) -> set[tuple[str, ...]]:
    """Which candidates must share a segment, as sorted tuples of candidate id.

    A candidate is its own group unless a wordless outcome covered it together with others:
    that outcome's text cannot be divided, so those candidates are one segment. Groups are
    merged transitively, because a candidate cut across two requests can be joined to a
    different neighbour by each of them.
    """
    groups: dict[str, tuple[str, ...]] = {}
    for outcome in outcomes:
        owned = _ordered({item.candidate_id for item in outcome.plan.ownership})
        for candidate in owned:
            groups.setdefault(candidate, (candidate,))
        if outcome.words or len(owned) < 2:
            continue
        merged = _ordered({name for candidate in owned for name in groups[candidate]})
        for name in merged:
            groups[name] = merged
    return set(groups.values())


def _contributions(
    outcomes: tuple[RequestOutcome, ...], groups: set[tuple[str, ...]]
) -> dict[tuple[str, ...], list[RequestOutcome]]:
    """Every outcome touching each group, in the order the requests were submitted."""
    by_candidate = {candidate: group for group in groups for candidate in group}
    found: dict[tuple[str, ...], list[RequestOutcome]] = {group: [] for group in groups}
    for outcome in outcomes:
        touched = {by_candidate[item.candidate_id] for item in outcome.plan.ownership}
        for group in sorted(touched):
            found[group].append(outcome)
    return found


def _draft(
    candidates: tuple[str, ...], outcomes: list[RequestOutcome], *, decimation: int
) -> SegmentDraft | None:
    """One group's segment, or ``None`` when the model found nothing in it."""
    pieces = [
        item
        for outcome in outcomes
        for item in outcome.plan.ownership
        if item.candidate_id in set(candidates)
    ]
    if not pieces:  # pragma: no cover - a group exists only because an outcome owned it
        return None

    ownership_start = min(item.session_start_sample for item in pieces)
    ownership_end = max(item.session_end_sample for item in pieces)
    aligned = all(outcome.alignment_status == "aligned" for outcome in outcomes)

    if aligned:
        words = _owned_words(candidates, outcomes, decimation=decimation)
        text = normalize_text(" ".join(word.text for word in words))
        if not text:
            return None
        return SegmentDraft(
            candidate_ids=candidates,
            track_id=outcomes[0].plan.track_id,
            start_sample=min(word.start_sample for word in words),
            end_sample=max(word.end_sample for word in words),
            ownership_start_sample=ownership_start,
            ownership_end_sample=ownership_end,
            text=text,
            words=tuple(words),
            alignment_status="aligned",
            request_ids=_ordered_ids(outcomes),
            truncation_submissions=sum(o.truncation_submissions for o in outcomes),
        )

    text = normalize_text(" ".join(outcome.text for outcome in outcomes))
    if not text:
        return None
    return SegmentDraft(
        candidate_ids=candidates,
        track_id=outcomes[0].plan.track_id,
        start_sample=ownership_start,
        end_sample=ownership_end,
        ownership_start_sample=ownership_start,
        ownership_end_sample=ownership_end,
        text=text,
        words=(),
        # `segment_only` wins over `not_attempted`: an aligner that ran and failed is the
        # thing the spec wants warned about, and a mixture that included one is one.
        alignment_status=(
            "segment_only"
            if any(outcome.alignment_status == "segment_only" for outcome in outcomes)
            else "not_attempted"
        ),
        request_ids=_ordered_ids(outcomes),
        truncation_submissions=sum(o.truncation_submissions for o in outcomes),
    )


def _owned_words(
    candidates: tuple[str, ...], outcomes: list[RequestOutcome], *, decimation: int
) -> list[WordRecord]:
    """The words this group owns, on the 48 kHz grid, in order.

    A word belongs to the ownership interval containing its **start** (ADR-0020); a word
    inside padding but inside no ownership interval belongs to nobody and is dropped, which
    is what stops padding from becoming content.
    """
    wanted = set(candidates)
    found: list[WordRecord] = []
    for outcome in outcomes:
        for piece in outcome.plan.ownership:
            if piece.candidate_id not in wanted:
                continue
            found.extend(
                _record(word, piece, decimation=decimation)
                for word in outcome.words
                if piece.start_sample <= word.start_sample < piece.end_sample
            )
    return sorted(found, key=lambda word: (word.start_sample, word.end_sample))


def _record(word: TranscribedWord, piece: Ownership, *, decimation: int) -> WordRecord:
    """One word on the session grid, clamped into the interval that owns it.

    The clamp is not cosmetic. A candidate's 48 kHz interval *covers* its derivative one, so
    the derivative sample the candidate starts at converts back to up to two samples *before*
    the candidate begins — and a word starting outside the interval that owns it is a state
    the records artifact refuses, correctly.
    """
    start = min(
        max(to_source_sample(word.start_sample, decimation), piece.session_start_sample),
        piece.session_end_sample - 1,
    )
    end = max(to_source_sample(max(word.end_sample, word.start_sample + 1), decimation), start + 1)
    return WordRecord(start_sample=start, end_sample=end, text=word.text)


def _ordered_ids(outcomes: list[RequestOutcome]) -> tuple[str, ...]:
    """Every contributing request id, first occurrence first, without repeats."""
    seen: dict[str, None] = {}
    for outcome in outcomes:
        for request_id in outcome.request_ids:
            seen.setdefault(request_id, None)
    return tuple(seen)


def _ordered(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(values))


def _alignment_notes(drafts: list[SegmentDraft]) -> list[TranscriptNote]:
    """Warn about segments the aligner ran on and failed. The spec requires exactly this.

    *"If alignment fails for one segment, retain the segment-level transcript and emit a
    warning rather than failing the entire session."* The retention is above; this is the
    warning.

    One warning per **track**, not per segment. An aligner that fails does not usually fail
    once — a four-hour session where it fails throughout would put thousands of lines in front
    of an operator, which is a way of hiding the problem rather than reporting it. Which
    individual segments lost their word times is in the records, where `alignment_status` says
    so per segment; the report gets the number and the tracks.
    """
    affected: dict[str, int] = {}
    for draft in drafts:
        if draft.alignment_status == "segment_only":
            affected[draft.track_id] = affected.get(draft.track_id, 0) + 1
    return [
        TranscriptNote(
            code="alignment_failed",
            message=(
                f"{count} segment(s) on {track_id} kept their text but have no word times: "
                f"forced alignment ran and did not produce them. The transcript is complete "
                f"and its word-level timings are not."
            ),
            path=track_id,
        )
        for track_id, count in sorted(affected.items())
    ]


def _notes(empty: int) -> list[TranscriptNote]:
    if not empty:
        return []
    return [
        TranscriptNote(
            code="candidate_transcribed_to_nothing",
            message=(
                f"{empty} retained activity candidate(s) produced no text and are absent from "
                f"the transcript. A detector fires on coughs and door closes as well as on "
                f"speech, so this is ordinary; a large number of them means the VAD "
                f"thresholds are worth revisiting (OQ-017)."
            ),
        )
    ]
