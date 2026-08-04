"""Assembling `work/transcript-records.json` from everything the stage decided.

The last step before anything is written: drafts, verdicts, the graph they came from, and the
provenance that says what produced them become one validated document. Nothing here decides
anything — the artifact's own validators are what refuse a state that would make a transcript
lie, and this is the code that hands them the whole picture at once.

A segment's id is its position in the canonical order (ADR-0019), and the drafts arrive in
that order, so the index *is* the id — the same index the collapse verdicts were computed
against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dnd_audio.activity import ACTIVITY_SEMANTICS_VERSION
from dnd_audio.artifacts.activity import ActivityGraph, ActivityTrack
from dnd_audio.artifacts.records import (
    OwnershipPieceRecord,
    SegmentRecord,
    TranscriberIdentity,
    TranscriptDecision,
    TranscriptNote,
    TranscriptRecords,
    TranscriptRecordsProvenance,
    segment_id,
)
from dnd_audio.artifacts.transcript import TranscriptSpeaker
from dnd_audio.config import SessionConfig, config_hash
from dnd_audio.transcript import (
    TRANSCRIPT_ASSEMBLY_SEMANTICS_VERSION,
    TRANSCRIPT_SEMANTICS_VERSION,
)
from dnd_audio.transcript.collapse import SegmentVerdict
from dnd_audio.transcript.segments import SegmentDraft

__all__ = ["build_records"]


def build_records(
    config: SessionConfig,
    graph: ActivityGraph,
    drafts: Sequence[SegmentDraft],
    verdicts: Sequence[SegmentVerdict],
    *,
    transcriber: TranscriberIdentity,
    timeline_sha256: str,
    warnings: Sequence[TranscriptNote] = (),
    decisions: Sequence[TranscriptDecision] = (),
) -> TranscriptRecords:
    """One validated records document. Raises if the pieces do not agree with each other."""
    speakers = {track.track_id: track for track in graph.tracks}
    return TranscriptRecords(
        session_id=config.session_id,
        title=config.title,
        language=config.language,
        config_hash=config_hash(config),
        timeline_sha256=timeline_sha256,
        activity_cache_key=graph.attribution_cache_key,
        sample_rate=graph.sample_rate,
        duration_samples=graph.duration_samples,
        presentation_join_gap_samples=(
            config.transcript.presentation_join_gap_ms * graph.sample_rate // 1000
        ),
        overlap_min_samples=config.transcript.overlap_min_ms * graph.sample_rate // 1000,
        speakers=[
            TranscriptSpeaker(
                speaker_id=track.speaker_id,
                speaker_name=track.speaker_name,
                track_id=track.track_id,
            )
            for track in graph.tracks
        ],
        segments=[
            _segment(index, draft, verdict, speakers[draft.track_id])
            for index, (draft, verdict) in enumerate(zip(drafts, verdicts, strict=True))
        ],
        warnings=list(warnings),
        decisions=list(decisions),
        provenance=TranscriptRecordsProvenance(
            transcript_semantics_version=TRANSCRIPT_SEMANTICS_VERSION,
            transcript_assembly_semantics_version=TRANSCRIPT_ASSEMBLY_SEMANTICS_VERSION,
            activity_semantics_version=ACTIVITY_SEMANTICS_VERSION,
            timeline_semantics_version=graph.provenance.timeline_semantics_version,
            inspection_semantics_version=graph.provenance.inspection_semantics_version,
            numpy_version=np.__version__,
            transcriber=transcriber,
        ),
    )


def _segment(
    index: int,
    draft: SegmentDraft,
    verdict: SegmentVerdict,
    track: ActivityTrack,
) -> SegmentRecord:
    return SegmentRecord(
        segment_id=segment_id(index),
        track_id=draft.track_id,
        speaker_id=track.speaker_id,
        speaker_name=track.speaker_name,
        start_sample=draft.start_sample,
        end_sample=draft.end_sample,
        ownership_start_sample=draft.ownership_start_sample,
        ownership_end_sample=draft.ownership_end_sample,
        text=draft.text,
        words=list(draft.words),
        alignment_status=draft.alignment_status,
        decision=verdict.decision,
        collapse_rule=verdict.collapse_rule,
        duplicate_of_segment_id=verdict.duplicate_of_segment_id,
        overlap=verdict.overlap,
        source_candidate_ids=list(draft.candidate_ids),
        request_ids=list(draft.request_ids),
        ownership_pieces=(
            [
                OwnershipPieceRecord(
                    **{field: getattr(piece, field) for field in OwnershipPieceRecord.model_fields}
                )
                for piece in draft.ownership_pieces
            ]
            if draft.ownership_pieces
            else None
        ),
        truncation_submissions=draft.truncation_submissions,
        rejected_alternatives=list(verdict.rejected_alternatives),
    )
