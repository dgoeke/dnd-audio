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
`to_source_sample`, and are clamped into the effective transcript bounds. The original
candidate bounds remain beside them as separate audit evidence (ADR-0033).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from dnd_audio.artifacts.records import TranscriptNote, WordRecord
from dnd_audio.artifacts.transcript import AlignmentStatus
from dnd_audio.interfaces import TranscribedWord
from dnd_audio.timeline.resample import to_source_sample
from dnd_audio.transcript.asr import RequestContribution, RequestOutcome, without_boundary_repeat
from dnd_audio.transcript.normalize import normalize_text

__all__ = ["DroppedWords", "OwnershipPiece", "SegmentDraft", "draft_segments"]


@dataclass(frozen=True, slots=True)
class OwnershipPiece:
    """One activity interval and the post-ASR interval that may claim its words.

    Piece-specific rather than aggregate because one alignment-fallback segment can span
    several candidates and gaps. Keeping the request and submitted bounds beside each piece
    makes every grace clamp auditable without rewriting the original plan (ADR-0033).
    """

    candidate_id: str
    request_id: str
    activity_start_derivative_sample: int
    activity_end_derivative_sample: int
    effective_start_derivative_sample: int
    effective_end_derivative_sample: int
    submitted_start_derivative_sample: int
    submitted_end_derivative_sample: int
    activity_start_sample: int
    activity_end_sample: int
    effective_start_sample: int
    effective_end_sample: int


@dataclass(frozen=True, slots=True)
class _AssemblyOccurrence:
    contribution: RequestContribution
    ownership: tuple[OwnershipPiece, ...]


@dataclass(frozen=True, slots=True)
class _AssemblyOutcome:
    outcome: RequestOutcome
    occurrences: tuple[_AssemblyOccurrence, ...]

    @property
    def ownership(self) -> tuple[OwnershipPiece, ...]:
        return tuple(piece for occurrence in self.occurrences for piece in occurrence.ownership)


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
    #: Candidate/piece-specific original and effective ownership (ADR-0033). Empty only in
    #: hand-built legacy test drafts that never become a new records artifact.
    ownership_pieces: tuple[OwnershipPiece, ...] = ()


@dataclass(frozen=True, slots=True)
class DroppedWords:
    """Words one track's requests returned that no ownership interval claimed.

    ADR-0020's rule 2, made visible. The behaviour it describes is right — padding must not
    become content — but until M8 it happened in silence: on the 2026-08-02 capture five of
    eleven segments lost their opening word and the transcript read as plausible prose.
    """

    track_id: str
    #: Dropped ``(request, word)`` pairs, not distinct words. One padding word that two
    #: overlapping requests both return and both drop is counted twice, because two requests
    #: each lost a word — and because deduplicating would need a rule for when two words at
    #: slightly different times are the same word, which is exactly what this metric must not
    #: quietly assume.
    count: int
    #: The candidates nearest the dropped words, so an operator can find the segments.
    candidate_ids: tuple[str, ...]
    #: Starts before versus at/after the nearest half-open ownership interval. The two
    #: counts sum to ``count`` and distinguish a late-opening detector from a candidate
    #: that ended too soon.
    before_ownership_count: int
    after_ownership_count: int
    #: Position in the request's returned word list. A leading word is where the aligner's
    #: request-boundary bias can act (OQ-027); a non-leading word needs another explanation.
    leading_word_count: int
    nonleading_word_count: int
    #: ``(minimum additional derivative samples, count)`` in ascending distance order.
    #: For an after-edge word this includes the one sample needed to cross a half-open end.
    #: Holding the current request result fixed, this is the exact ownership expansion that
    #: would include the word — not a claim that the word *should* be included.
    edge_distance_derivative_samples: tuple[tuple[int, int], ...]

    @property
    def max_edge_distance_derivative_samples(self) -> int:
        """The farthest dropped start from its nearest ownership edge."""
        return self.edge_distance_derivative_samples[-1][0]


def draft_segments(
    outcomes: tuple[RequestOutcome, ...], *, decimation: int, leading_grace_samples: int = 0
) -> tuple[list[SegmentDraft], list[TranscriptNote], tuple[DroppedWords, ...]]:
    """Every retained candidate's draft segment, in canonical order, plus what to report."""
    assembled = _effective_ownership(
        outcomes, decimation=decimation, leading_grace_samples=leading_grace_samples
    )
    groups = _grouped_candidates(assembled)
    contributions = _contributions(assembled, groups)

    drafts: list[SegmentDraft] = []
    empty = 0
    for candidates in sorted(groups, key=sorted):
        draft = _draft(candidates, contributions[candidates], decimation=decimation)
        if draft is None:
            empty += 1
            continue
        drafts.append(draft)

    drafts.sort(key=lambda item: (item.start_sample, item.track_id, item.candidate_ids))
    dropped = _dropped_words(assembled)
    return (
        drafts,
        [*_notes(empty), *_alignment_notes(drafts), *_dropped_notes(dropped)],
        dropped,
    )


def _effective_ownership(
    outcomes: tuple[RequestOutcome, ...], *, decimation: int, leading_grace_samples: int
) -> tuple[_AssemblyOutcome, ...]:
    """Canonical post-ASR ownership, globally clipped on each track (ADR-0033, OQ-027)."""
    if leading_grace_samples < 0:
        message = f"leading ownership grace must not be negative, got {leading_grace_samples}"
        raise ValueError(message)

    entries = sorted(
        (
            (
                contribution.plan.track_id,
                piece.start_sample,
                piece.end_sample,
                contribution.plan.request_id,
                piece.candidate_id,
                outcome_index,
                contribution_index,
                piece_index,
                contribution,
                piece,
            )
            for outcome_index, outcome in enumerate(outcomes)
            for contribution_index, contribution in enumerate(outcome.contributing_submissions)
            for piece_index, piece in enumerate(contribution.plan.ownership)
        ),
        key=lambda item: item[:8],
    )
    by_outcome: list[list[list[OwnershipPiece | None]]] = [
        [
            [None] * len(contribution.plan.ownership)
            for contribution in outcome.contributing_submissions
        ]
        for outcome in outcomes
    ]
    previous_end: dict[str, int] = {}
    for (
        track_id,
        _,
        _,
        _,
        _,
        outcome_index,
        contribution_index,
        piece_index,
        contribution,
        piece,
    ) in entries:
        predecessor = previous_end.get(track_id, 0)
        if piece.start_sample < predecessor:
            message = (
                f"activity ownership overlaps on {track_id}: {piece.candidate_id} starts at "
                f"{piece.start_sample} before the preceding half-open end {predecessor}"
            )
            raise ValueError(message)
        effective_start = max(
            contribution.plan.padded_start_sample,
            predecessor,
            piece.start_sample - leading_grace_samples,
        )
        by_outcome[outcome_index][contribution_index][piece_index] = OwnershipPiece(
            candidate_id=piece.candidate_id,
            request_id=contribution.plan.request_id,
            activity_start_derivative_sample=piece.start_sample,
            activity_end_derivative_sample=piece.end_sample,
            effective_start_derivative_sample=effective_start,
            effective_end_derivative_sample=piece.end_sample,
            submitted_start_derivative_sample=contribution.plan.padded_start_sample,
            submitted_end_derivative_sample=contribution.plan.padded_end_sample,
            activity_start_sample=piece.session_start_sample,
            activity_end_sample=piece.session_end_sample,
            effective_start_sample=(
                piece.session_start_sample
                if effective_start == piece.start_sample
                else to_source_sample(effective_start, decimation)
            ),
            effective_end_sample=piece.session_end_sample,
        )
        previous_end[track_id] = piece.end_sample

    return tuple(
        _AssemblyOutcome(
            outcome=outcome,
            occurrences=tuple(
                _AssemblyOccurrence(
                    contribution=contribution,
                    ownership=tuple(
                        piece
                        for piece in by_outcome[index][contribution_index]
                        if piece is not None
                    ),
                )
                for contribution_index, contribution in enumerate(outcome.contributing_submissions)
            ),
        )
        for index, outcome in enumerate(outcomes)
    )


def _dropped_words(outcomes: tuple[_AssemblyOutcome, ...]) -> tuple[DroppedWords, ...]:
    """Per submitted occurrence, words returned outside that occurrence's own intervals.

    **Per occurrence, not per candidate group or stitched parent**, and that is the whole
    design. `_owned_words` runs once per group, while a resolved retry has child submissions
    with different ownership and padding. Looking through the parent would let one child's word
    be claimed by another child's candidate; looking per group would report false positives on
    every merged request (ADR-0033, M9 code review).
    """
    observations: dict[str, list[tuple[OwnershipPiece, Literal["before", "after"], int, bool]]] = {}
    for assembled in outcomes:
        if not assembled.outcome.words:
            continue
        for occurrence in assembled.occurrences:
            intervals = occurrence.ownership
            words = occurrence.contribution.words
            if not words or not intervals:
                continue
            track_id = occurrence.contribution.plan.track_id
            for index, word in enumerate(words):
                if any(
                    piece.effective_start_derivative_sample
                    <= word.start_sample
                    < piece.effective_end_derivative_sample
                    for piece in intervals
                ):
                    continue
                piece, side, distance = _nearest_edge(word, intervals)
                observations.setdefault(track_id, []).append((piece, side, distance, index == 0))
    return tuple(
        DroppedWords(
            track_id=track_id,
            count=len(items),
            candidate_ids=tuple(sorted({piece.candidate_id for piece, _, _, _ in items})),
            before_ownership_count=sum(side == "before" for _, side, _, _ in items),
            after_ownership_count=sum(side == "after" for _, side, _, _ in items),
            leading_word_count=sum(leading for _, _, _, leading in items),
            nonleading_word_count=sum(not leading for _, _, _, leading in items),
            edge_distance_derivative_samples=tuple(
                sorted(Counter(distance for _, _, distance, _ in items).items())
            ),
        )
        for track_id, items in sorted(observations.items())
    )


def _nearest_edge(
    word: TranscribedWord, intervals: tuple[OwnershipPiece, ...]
) -> tuple[OwnershipPiece, Literal["before", "after"], int]:
    """Whose segment is nearest, on which side, and by how many derivative samples.

    A dropped word is almost always just outside one interval's edge — the opening word of an
    utterance the detector started a moment late — so naming the nearest candidate points an
    operator at the segment that reads wrong, rather than at every candidate in the request.

    Distance is the minimum outward expansion that would make the word's **start** owned while
    holding this request result fixed. Because ownership is half-open, a word starting exactly
    at an interval's end needs one additional sample, not zero.
    """
    measured: list[tuple[int, str, Literal["before", "after"], OwnershipPiece]] = []
    for piece in intervals:
        if word.start_sample < piece.effective_start_derivative_sample:
            side: Literal["before", "after"] = "before"
            distance = piece.effective_start_derivative_sample - word.start_sample
        else:
            side = "after"
            distance = word.start_sample - piece.effective_end_derivative_sample + 1
        measured.append((distance, piece.candidate_id, side, piece))
    distance, _, side, piece = min(measured)
    return piece, side, distance


def _dropped_notes(dropped: tuple[DroppedWords, ...]) -> list[TranscriptNote]:
    return [
        TranscriptNote(
            code="words_dropped_outside_ownership",
            message=(
                f"{item.count} word(s) returned for {item.track_id} started inside a request's "
                f"padding but inside no ownership interval, so they are not in the transcript "
                f"(ADR-0020). {item.leading_word_count} were the request's leading returned "
                f"word; {item.before_ownership_count} started before and "
                f"{item.after_ownership_count} after the nearest interval. The farthest start "
                f"needs {item.max_edge_distance_derivative_samples} additional derivative "
                f"sample(s) of ownership to be included, holding this model result fixed. "
                f"Counted as (request, word) pairs. This is boundary geometry, not evidence "
                f"that every dropped word should become content: a weak lav's padding can "
                f"contain another speaker's words."
            ),
            path=item.track_id,
        )
        for item in dropped
    ]


def _grouped_candidates(outcomes: tuple[_AssemblyOutcome, ...]) -> set[tuple[str, ...]]:
    """Which candidates must share a segment, as sorted tuples of candidate id.

    A candidate is its own group unless a wordless outcome covered it together with others:
    that outcome's text cannot be divided, so those candidates are one segment. Groups are
    merged transitively, because a candidate cut across two requests can be joined to a
    different neighbour by each of them.
    """
    groups: dict[str, tuple[str, ...]] = {}
    for assembled in outcomes:
        outcome = assembled.outcome
        owned = _ordered({item.candidate_id for item in assembled.ownership})
        for candidate in owned:
            groups.setdefault(candidate, (candidate,))
        if outcome.words or len(owned) < 2:
            continue
        merged = _ordered({name for candidate in owned for name in groups[candidate]})
        for name in merged:
            groups[name] = merged
    return set(groups.values())


def _contributions(
    outcomes: tuple[_AssemblyOutcome, ...], groups: set[tuple[str, ...]]
) -> dict[tuple[str, ...], list[_AssemblyOutcome]]:
    """Every outcome touching each group, in the order the requests were submitted."""
    by_candidate = {candidate: group for group in groups for candidate in group}
    found: dict[tuple[str, ...], list[_AssemblyOutcome]] = {group: [] for group in groups}
    for outcome in outcomes:
        touched = {by_candidate[item.candidate_id] for item in outcome.ownership}
        for group in sorted(touched):
            found[group].append(outcome)
    return found


def _draft(
    candidates: tuple[str, ...], outcomes: list[_AssemblyOutcome], *, decimation: int
) -> SegmentDraft | None:
    """One group's segment, or ``None`` when the model found nothing in it."""
    pieces = [
        item
        for outcome in outcomes
        for item in outcome.ownership
        if item.candidate_id in set(candidates)
    ]
    if not pieces:  # pragma: no cover - a group exists only because an outcome owned it
        return None

    ownership_start = min(item.activity_start_sample for item in pieces)
    ownership_end = max(item.activity_end_sample for item in pieces)
    aligned = all(outcome.outcome.alignment_status == "aligned" for outcome in outcomes)

    if aligned:
        words = _owned_words(candidates, outcomes, decimation=decimation)
        text = normalize_text(" ".join(word.text for word in words))
        if not text:
            return None
        return SegmentDraft(
            candidate_ids=candidates,
            track_id=outcomes[0].outcome.plan.track_id,
            start_sample=min(word.start_sample for word in words),
            end_sample=max(word.end_sample for word in words),
            ownership_start_sample=ownership_start,
            ownership_end_sample=ownership_end,
            text=text,
            words=tuple(words),
            alignment_status="aligned",
            request_ids=_ordered_ids(outcomes),
            truncation_submissions=sum(o.outcome.truncation_submissions for o in outcomes),
            ownership_pieces=tuple(pieces),
        )

    text = normalize_text(" ".join(outcome.outcome.text for outcome in outcomes))
    if not text:
        return None
    return SegmentDraft(
        candidate_ids=candidates,
        track_id=outcomes[0].outcome.plan.track_id,
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
            if any(outcome.outcome.alignment_status == "segment_only" for outcome in outcomes)
            else "not_attempted"
        ),
        request_ids=_ordered_ids(outcomes),
        truncation_submissions=sum(o.outcome.truncation_submissions for o in outcomes),
        ownership_pieces=tuple(pieces),
    )


def _owned_words(
    candidates: tuple[str, ...], outcomes: list[_AssemblyOutcome], *, decimation: int
) -> list[WordRecord]:
    """The words this group owns, on the 48 kHz grid, in order.

    A word belongs to the ownership interval containing its **start** (ADR-0020); a word
    inside padding but inside no ownership interval belongs to nobody and is dropped, which
    is what stops padding from becoming content.

    The start rule alone is not enough where two pieces are genuinely **adjacent** — a
    candidate longer than the cap, cut by `requests._divide` into separately padded requests.
    Each request's padded window reaches past the boundary into the other's core, so the model
    can return the boundary word on both sides at slightly different times, and a start rule
    then puts one copy in each piece: `"Zephyrine Zephyrine"`. ADR-0020's rule 3 is applied
    across that boundary too, through the same function the truncation stitch uses.
    """
    wanted = set(candidates)
    owned = sorted(
        (
            (
                piece,
                tuple(
                    word
                    for word in occurrence.contribution.words
                    if piece.effective_start_derivative_sample
                    <= word.start_sample
                    < piece.effective_end_derivative_sample
                ),
            )
            for outcome in outcomes
            for occurrence in outcome.occurrences
            for piece in occurrence.ownership
            if piece.candidate_id in wanted
        ),
        key=lambda item: (
            item[0].effective_start_derivative_sample,
            item[0].effective_end_derivative_sample,
            item[0].request_id,
        ),
    )

    found: list[WordRecord] = []
    previous: tuple[OwnershipPiece, tuple[TranscribedWord, ...]] | None = None
    for piece, words in owned:
        # Only across a boundary the two pieces actually share. Two pieces separated by
        # silence are two utterances, and a word repeated across a gap is a word said twice.
        if (
            previous is not None
            and previous[0].effective_end_derivative_sample
            == piece.effective_start_derivative_sample
        ):
            words = without_boundary_repeat(previous[1], words)
        found.extend(_record(word, piece, decimation=decimation) for word in words)
        previous = (piece, words)
    return sorted(found, key=lambda word: (word.start_sample, word.end_sample))


def _record(word: TranscribedWord, piece: OwnershipPiece, *, decimation: int) -> WordRecord:
    """One word on the session grid, clamped into the interval that owns it.

    The clamp is not cosmetic. A candidate's 48 kHz interval *covers* its derivative one, so
    the derivative sample the candidate starts at converts back to up to two samples *before*
    the candidate begins — and a word starting outside the interval that owns it is a state
    the records artifact refuses, correctly.
    """
    start = min(
        max(to_source_sample(word.start_sample, decimation), piece.effective_start_sample),
        piece.effective_end_sample - 1,
    )
    end = max(to_source_sample(max(word.end_sample, word.start_sample + 1), decimation), start + 1)
    return WordRecord(start_sample=start, end_sample=end, text=word.text)


def _ordered_ids(outcomes: list[_AssemblyOutcome]) -> tuple[str, ...]:
    """Every contributing request id, first occurrence first, without repeats."""
    seen: dict[str, None] = {}
    for outcome in outcomes:
        for request_id in outcome.outcome.request_ids:
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
