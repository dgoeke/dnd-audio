"""`transcript.json` — the diarized transcript in its machine-readable form.

The shape follows the spec's baseline exactly. Two properties matter more than the
fields themselves:

* Times are floats here and nowhere else. They arrive through
  :func:`dnd_audio.determinism.public_seconds`, so every value is an exact number of
  milliseconds whose shortest repr round-trips (INV-04).
* ``segment_id`` derives from sorted source identity and time, never from the order
  tasks finished in, so a rerun with the same inputs produces the same IDs (INV-02).
  M4 owns that derivation; the format is fixed here.

There is deliberately no ASR confidence field. The spec forbids manufacturing one when
the model does not expose a meaningful value, and signal-quality scores are a different
thing that belongs in the report.
"""

from __future__ import annotations

import re
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "TRANSCRIPT_SCHEMA_VERSION",
    "AlignmentStatus",
    "SegmentProvenance",
    "Transcript",
    "TranscriptSegment",
    "TranscriptSpeaker",
    "TranscriptWord",
]

#: Provisional until M4 closes. See the package docstring.
TRANSCRIPT_SCHEMA_VERSION: Final = 1

#: ``aligned`` — word times came from the forced aligner. ``segment_only`` — alignment
#: failed for this segment and the segment-level transcript was kept with a warning,
#: which the spec requires instead of failing the session. ``not_attempted`` — no
#: aligner ran.
AlignmentStatus = Literal["aligned", "segment_only", "not_attempted"]

_SEGMENT_ID = re.compile(r"^seg_\d{6,}$")

SegmentId = Annotated[str, Field(pattern=_SEGMENT_ID.pattern)]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TranscriptWord(_Artifact):
    """One word, with times in seconds at millisecond precision."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    text: str

    @model_validator(mode="after")
    def _check_order(self) -> TranscriptWord:
        if self.end_s < self.start_s:
            message = f"word {self.text!r} ends at {self.end_s} before it starts at {self.start_s}"
            raise ValueError(message)
        return self


class SegmentProvenance(_Artifact):
    """Where a segment came from, so a surprising line can be traced back."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    asr_model: str
    #: The resolved Hugging Face commit, not the mutable branch name.
    asr_model_revision: str | None = None
    alignment_status: AlignmentStatus
    #: The pre-ASR activity candidate this segment was transcribed from. The link
    #: between the model-independent graph (INV-09) and the text.
    source_candidate_id: str
    #: Plural lineage for a public turn coalesced from granular records. Optional additive
    #: fields preserve schema-1 compatibility (ADR-0034).
    source_candidate_ids: list[str] | None = None
    source_segment_ids: list[SegmentId] | None = None


class TranscriptSpeaker(_Artifact):
    """A participant, and the transmitter that captured them."""

    speaker_id: str
    speaker_name: str
    track_id: str


class TranscriptSegment(_Artifact):
    """One attributed utterance."""

    segment_id: SegmentId
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    speaker_id: str
    speaker_name: str
    track_id: str
    text: str
    #: True when this segment overlaps another retained, non-duplicate speaker's
    #: segment by at least the configured threshold.
    overlap: bool = False
    words: list[TranscriptWord] = Field(default_factory=list)
    provenance: SegmentProvenance

    @model_validator(mode="after")
    def _check_order(self) -> TranscriptSegment:
        if self.end_s < self.start_s:
            message = (
                f"segment {self.segment_id} ends at {self.end_s} before it starts at {self.start_s}"
            )
            raise ValueError(message)
        return self


class Transcript(_Artifact):
    """The whole transcript, sorted and stable."""

    schema_version: Literal[1] = TRANSCRIPT_SCHEMA_VERSION
    session_id: str
    title: str
    duration_s: float = Field(ge=0.0)
    speakers: list[TranscriptSpeaker] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort(self) -> Transcript:
        """Sort by start time, then by segment id.

        Overlapping turns stay separate entries, so start time alone is not a total
        order; the id breaks ties and is itself derived from source identity, which
        makes the whole ordering independent of completion order (INV-02).
        """
        speakers = sorted(self.speakers, key=lambda speaker: speaker.speaker_id)
        segments = sorted(self.segments, key=lambda segment: (segment.start_s, segment.segment_id))
        object.__setattr__(self, "speakers", speakers)
        object.__setattr__(self, "segments", segments)
        return self
