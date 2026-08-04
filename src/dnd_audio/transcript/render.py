"""Records to `transcript.json` and `transcript.md`.

Both outputs are functions of the records artifact and of nothing else (ADR-0019), which is
what makes `render` provably free of ASR, of the activity graph, and of the mixer.

**Times become floats here and nowhere else in this package.** Everything upstream is integer
samples; the public boundary quantizes once, through `determinism.public_seconds`, so every
value in `transcript.json` is an exact number of milliseconds whose shortest repr round-trips
(INV-04). The Markdown timestamp is computed from the *same* integer millisecond count rather
than by reformatting the float, so the two documents cannot disagree about a rounding.

**Only retained segments are rendered.** A collapsed duplicate stays in the records with the
evidence that condemned it; the transcript is what survived. That leaves gaps in the segment
numbering, and a gap is informative — it says a collapse happened there.

**User and model text is escaped.** A speaker named `*DM*` and a model that emitted a
backtick are both ordinary, and neither may reach a Markdown document as formatting. The
escape is applied to the name and to the text, and any whitespace that survived normalization
is collapsed again here: the format is one turn per line, and a newline in the middle of a
line breaks the document rather than the sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Final

from dnd_audio.artifacts.records import SegmentRecord, TranscriptRecords, WordRecord
from dnd_audio.artifacts.transcript import (
    AlignmentStatus,
    SegmentProvenance,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from dnd_audio.determinism import public_seconds, to_milliseconds

__all__ = [
    "MARKDOWN_ESCAPED",
    "PresentationTurn",
    "build_transcript",
    "presentation_turns",
    "render_markdown",
    "timestamp",
]

#: Characters that would otherwise be read as Markdown. Deliberately not a general HTML
#: escape: this is a Markdown document, and turning an ampersand into an entity would change
#: what the transcript says a person said.
MARKDOWN_ESCAPED: Final = "\\`*_[]<>"

_ESCAPE = re.compile(f"([{re.escape(MARKDOWN_ESCAPED)}])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class PresentationTurn:
    """One public turn, with every granular record still named (ADR-0034)."""

    segment_id: str
    source_segment_ids: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    request_ids: frozenset[str]
    start_sample: int
    end_sample: int
    speaker_id: str
    speaker_name: str
    track_id: str
    text: str
    overlap: bool
    words: tuple[WordRecord, ...]
    alignment_status: AlignmentStatus


def presentation_turns(records: TranscriptRecords) -> tuple[PresentationTurn, ...]:
    """The shared JSON/Markdown view over granular retained records.

    Legacy records have no presentation threshold and retain their original one-record-per-line
    semantics. New records group first, then recompute public overlap over the resulting exact
    intervals (ADR-0034, OQ-018).
    """
    turns: list[PresentationTurn] = []
    for segment in records.retained():
        current = _turn(segment)
        if turns and _may_join(turns[-1], segment, records.presentation_join_gap_samples):
            turns[-1] = _join(turns[-1], segment)
        else:
            turns.append(current)

    if records.overlap_min_samples is None:
        return tuple(turns)
    return tuple(
        replace(
            turn,
            overlap=any(
                _public_overlap(turn, other, records.overlap_min_samples) for other in turns
            ),
        )
        for turn in turns
    )


def _turn(segment: SegmentRecord) -> PresentationTurn:
    return PresentationTurn(
        segment_id=segment.segment_id,
        source_segment_ids=(segment.segment_id,),
        source_candidate_ids=tuple(segment.source_candidate_ids),
        request_ids=frozenset(segment.request_ids),
        start_sample=segment.start_sample,
        end_sample=segment.end_sample,
        speaker_id=segment.speaker_id,
        speaker_name=segment.speaker_name,
        track_id=segment.track_id,
        text=segment.text,
        overlap=segment.overlap,
        words=tuple(segment.words),
        alignment_status=segment.alignment_status,
    )


def _public_overlap(first: PresentationTurn, second: PresentationTurn, minimum: int) -> bool:
    if first.speaker_id == second.speaker_id:
        return False
    shared = min(first.end_sample, second.end_sample) - max(first.start_sample, second.start_sample)
    return shared > 0 and shared >= minimum


def _may_join(
    turn: PresentationTurn, segment: SegmentRecord, maximum_gap_samples: int | None
) -> bool:
    if maximum_gap_samples is None:
        return False
    gap = segment.start_sample - turn.end_sample
    return (
        turn.track_id == segment.track_id
        and turn.speaker_id == segment.speaker_id
        and turn.alignment_status == segment.alignment_status
        and not turn.overlap
        and not segment.overlap
        and bool(turn.request_ids.intersection(segment.request_ids))
        and 0 <= gap <= maximum_gap_samples
    )


def _join(turn: PresentationTurn, segment: SegmentRecord) -> PresentationTurn:
    candidates = dict.fromkeys((*turn.source_candidate_ids, *segment.source_candidate_ids))
    return replace(
        turn,
        source_segment_ids=(*turn.source_segment_ids, segment.segment_id),
        source_candidate_ids=tuple(candidates),
        request_ids=turn.request_ids.union(segment.request_ids),
        end_sample=max(turn.end_sample, segment.end_sample),
        text=" ".join(part for part in (turn.text, segment.text) if part),
        words=tuple(
            sorted(
                (*turn.words, *segment.words),
                key=lambda word: (word.start_sample, word.end_sample),
            )
        ),
    )


def build_transcript(records: TranscriptRecords) -> Transcript:
    """`transcript.json`, from the records and nothing else."""
    return Transcript(
        session_id=records.session_id,
        title=records.title,
        duration_s=_seconds(records.duration_samples, records.sample_rate),
        speakers=list(records.speakers),
        segments=[_segment(turn, records) for turn in presentation_turns(records)],
    )


def _segment(segment: PresentationTurn, records: TranscriptRecords) -> TranscriptSegment:
    transcriber = records.provenance.transcriber
    return TranscriptSegment(
        segment_id=segment.segment_id,
        start_s=_seconds(segment.start_sample, records.sample_rate),
        end_s=_seconds(segment.end_sample, records.sample_rate),
        speaker_id=segment.speaker_id,
        speaker_name=segment.speaker_name,
        track_id=segment.track_id,
        text=segment.text,
        overlap=segment.overlap,
        words=[
            TranscriptWord(
                start_s=_seconds(word.start_sample, records.sample_rate),
                end_s=_seconds(word.end_sample, records.sample_rate),
                text=word.text,
            )
            for word in segment.words
        ],
        provenance=SegmentProvenance(
            # A transcriber with no model name is a fake, and saying so is better than an
            # empty string that reads like a missing value (ADR-0018).
            asr_model=transcriber.model or transcriber.name,
            asr_model_revision=transcriber.model_revision,
            alignment_status=segment.alignment_status,
            # The first, because ordinarily there is only one: ownership survives a merge
            # (ADR-0017), and the several-candidate case is the wordless one the records
            # artifact records in full.
            source_candidate_id=segment.source_candidate_ids[0],
            source_candidate_ids=list(segment.source_candidate_ids),
            source_segment_ids=list(segment.source_segment_ids),
        ),
    )


def render_markdown(records: TranscriptRecords) -> str:
    """`transcript.md`, in the spec's format, sorted by start time.

    Built from the records rather than from the assembled `Transcript` so the timestamps come
    from integer samples. Overlapping turns stay separate entries — this is a transcript of a
    conversation, and merging two people talking at once into one line would be inventing a
    turn nobody took.
    """
    lines = [f"# {_escape(records.title)}", ""]
    lines.extend(_line(segment, records.sample_rate) for segment in presentation_turns(records))
    return "\n".join(lines) + "\n"


def _line(segment: PresentationTurn, sample_rate: int) -> str:
    marker = " [overlap]" if segment.overlap else ""
    stamp = timestamp(segment.start_sample, sample_rate)
    return f"**[{stamp}] {_escape(segment.speaker_name)}{marker}:** {_escape(segment.text)}\n"


def timestamp(sample: int, sample_rate: int) -> str:
    """``HH:MM:SS.mmm`` for a sample position, from exact integer milliseconds.

    Hours are not truncated at 24: a session that ran past midnight is one session, and its
    transcript should say hour 25 rather than restart at zero.
    """
    total = to_milliseconds(Fraction(sample, sample_rate))
    milliseconds = total % 1000
    seconds = total // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}.{milliseconds:03d}"


def _seconds(sample: int, sample_rate: int) -> float:
    return public_seconds(Fraction(sample, sample_rate))


def _escape(text: str) -> str:
    """Make text safe to place inside a Markdown line, without changing what it says."""
    return _ESCAPE.sub(r"\\\1", _WHITESPACE.sub(" ", text).strip())
