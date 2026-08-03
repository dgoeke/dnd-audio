"""`work/transcript-records.json` — what ASR produced, and what was decided about it.

The artifact `render` reads, and the only thing it reads (ADR-0019). Duplicate collapse and
overlap marking have already happened by the time this is written, so regenerating
`transcript.json` and `transcript.md` needs no model, no activity graph, no timeline, and no
mixer — which is what makes the spec's "render without invoking ASR or the mixer" a property
of the input rather than a claim about which code paths happen not to run.

Five shapes are deliberate.

**Times are integer samples, on the canonical 48 kHz session grid.** No floats anywhere in
this document, for the reason `timeline.json` and `activity.json` have none: it is byte-stable
on an unchanged rerun (INV-02), and the millisecond boundary belongs to the *public*
transcript, where `determinism.public_seconds` is the one conversion that produces one.

**A segment's interval is where its words are, when it has words.** The ownership interval it
came from is kept beside it rather than instead of it: a segment whose speech occupies the
middle two seconds of a five-second candidate should say so, and an operator chasing a bad
attribution needs the candidate's own bounds to compare against.

**A collapsed duplicate stays in the document.** It keeps its id, its text, and the evidence
that condemned it, and names the segment that beat it. `transcript.json` carries only the
survivors, so its numbering has gaps — a gap is a collapse, and this file says which.

**Rejected alternatives are recorded only where something was actually rejected.** Every
evaluated pair would be quadratic growth for no audit value.

**Provenance names the graph and the configuration this describes**, so a records file sitting
beside an activity graph it does not describe is detectable rather than merely unlikely.
"""

from __future__ import annotations

import re
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.artifacts.report import RuntimeProvenance
from dnd_audio.artifacts.transcript import AlignmentStatus, TranscriptSpeaker

__all__ = [
    "TRANSCRIPT_RECORDS_SCHEMA_VERSION",
    "RejectedAlternative",
    "SegmentDecision",
    "SegmentRecord",
    "TranscriberIdentity",
    "TranscriptDecision",
    "TranscriptNote",
    "TranscriptRecords",
    "TranscriptRecordsProvenance",
    "WordRecord",
    "segment_id",
]

#: Provisional until M4 closes; additive optional fields only thereafter (ADR-0005).
TRANSCRIPT_RECORDS_SCHEMA_VERSION: Final = 1

_SEGMENT_ID = re.compile(r"^seg_\d{6,}$")

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SegmentId = Annotated[str, Field(pattern=_SEGMENT_ID.pattern)]

#: A ratio as thousandths, the same unit the activity graph quotes every score in.
Permille = Annotated[int, Field(ge=0, le=1000)]

#: A signed difference of two per-mille quantities.
SignedPermille = Annotated[int, Field(ge=-1000, le=1000)]

#: ``retained`` reaches `transcript.json`; ``duplicate`` was collapsed into another segment
#: and stays here for the audit trail. There is deliberately no third value: a segment is
#: either in the transcript or it is accounted for by one that is.
SegmentDecision = Literal["retained", "duplicate"]


def segment_id(index: int) -> str:
    """The id of the ``index``-th segment in canonical order.

    Position in the sort by ``(start_sample, track_id)`` — so it derives from sorted source
    identity and time rather than from the order tasks finished in (INV-02) — and in the shape
    the spec's own example uses, which `tests/data/transcript-spec-example.json` holds as
    independent ground truth (ADR-0019).
    """
    if index < 0:
        message = f"segment index must not be negative, got {index}"
        raise ValueError(message)
    return f"seg_{index:06d}"


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TranscriptNote(_Artifact):
    """A warning: worth a human's attention, did not stop the run."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None


class TranscriptDecision(_Artifact):
    """A transcript choice worth auditing. Deterministic: no counters, no timings."""

    code: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class TranscriberIdentity(_Artifact):
    """Everything that decides what the transcriber would answer (INV-08).

    A fake carries a ``variant_digest`` over its whole script and no model, the same shape
    `DetectorIdentity` uses: two scripted transcribers with different scripts are different
    transcribers, and a cache that could not tell them apart would serve one test's answers
    to another.
    """

    # `model` and `model_revision` collide with pydantic's reserved prefix; the names come
    # from the spec's own vocabulary, so the namespace guard goes instead.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str = Field(min_length=1)
    model: str | None = None
    #: The resolved commit, never a mutable branch name.
    model_revision: str | None = None
    aligner: str | None = None
    aligner_revision: str | None = None
    #: Explicit rather than inherited from the wrapper's default, and part of the ASR cache
    #: key, so changing it re-runs the work (spec, Milestone 4).
    max_new_tokens: int = Field(gt=0)
    language: str = Field(min_length=1)
    #: Hash of the glossary text passed as the model's context. ``None`` when the session has
    #: no glossary, which must not block a run.
    context_sha256: Sha256Hex | None = None
    #: Distinguishes two instances of one implementation. Scripted transcribers hash their
    #: script into it; a real one leaves it unset because its revision already differs.
    variant_digest: Sha256Hex | None = None
    #: The compute runtime the adapter resolved: python, torch, HIP, device, device name,
    #: dtype, attention. **Nested rather than flattened into fields beside these**, which is
    #: the point: M6a defined that vocabulary once, in `RuntimeProvenance`, precisely so
    #: M6b would not build a second one for the cache key to drift from (INV-08). The same
    #: audio transcribed in BF16 on gfx1151 and in float32 on a CPU are not the same result,
    #: and a Torch or HIP upgrade can change a kernel's rounding.
    #:
    #: ``None`` for a transcriber that resolved no runtime — every fake, which is why this
    #: is optional rather than required (ADR-0005's additive rule).
    runtime: RuntimeProvenance | None = None
    #: The `qwen-asr` distribution version. Its release notes are not the model's, but its
    #: prompt construction, chunking and output parsing all change what a request returns.
    package_version: str | None = None
    #: Transformers' version. Reaches the key for the same reason: generation is its code.
    transformers_version: str | None = None
    #: How close to `max_new_tokens` a response must land to be called truncated. In the key
    #: because it decides whether a split-and-retry happened, and therefore what the text is
    #: (ADR-0028). ``None`` for a transcriber with no such notion.
    truncation_margin_tokens: int | None = Field(default=None, ge=0)


class TranscriptRecordsProvenance(_Artifact):
    """What produced these records. Deterministic, and no wall clock (INV-03)."""

    transcript_semantics_version: int = Field(ge=1)
    #: The graph these segments were built from, and the placement underneath it. A records
    #: file whose upstream moved is not obviously wrong without them.
    activity_semantics_version: int = Field(ge=1)
    timeline_semantics_version: int = Field(ge=1)
    inspection_semantics_version: int = Field(ge=1)
    numpy_version: str = Field(min_length=1)
    transcriber: TranscriberIdentity


class WordRecord(_Artifact):
    """One word, in canonical 48 kHz session samples, half-open."""

    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end_sample <= self.start_sample:
            message = f"word {self.text!r} spans an empty interval"
            raise ValueError(message)
        return self


class RejectedAlternative(_Artifact):
    """A segment collapsed into this one, and the evidence that condemned it.

    The spec requires rejected alternatives to be recorded; recording only the id would make
    the decision unauditable, which is the same argument ADR-0012 made for keeping the bleed
    gate's evidence per pair rather than summarized.
    """

    segment_id: SegmentId
    track_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    #: What the loser said, kept verbatim: the whole question a reader has is whether the two
    #: really were the same utterance.
    text: str
    #: Shared samples as a fraction of the *shorter* segment.
    overlap_permille: Permille
    text_similarity_permille: Permille
    #: The weakest peak correlation among the candidate pairs behind these two segments, or
    #: ``None`` when the graph measured no pair — which is itself a reason not to collapse.
    correlation_permille: Permille | None = None
    #: The winner's source score minus the loser's.
    score_margin_permille: SignedPermille


class SegmentRecord(_Artifact):
    """One attributed utterance, normalized, with what was decided about it."""

    segment_id: SegmentId
    track_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    #: Where the speech is: the span of the words this segment owns, or the ownership
    #: interval itself when no word times came back.
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    #: The activity candidate's own interval, kept beside the tighter one above so a
    #: surprising segment can be compared against the region it was cut from.
    ownership_start_sample: int = Field(ge=0)
    ownership_end_sample: int = Field(gt=0)
    text: str
    words: list[WordRecord] = Field(default_factory=list)
    alignment_status: AlignmentStatus
    decision: SegmentDecision
    #: Set exactly when ``decision`` is ``duplicate``.
    duplicate_of_segment_id: SegmentId | None = None
    #: Overlaps another *retained, non-duplicate* segment of a different speaker by at least
    #: the configured threshold. False on a duplicate, which is not in the transcript to
    #: overlap anything.
    overlap: bool = False
    #: The activity candidates this segment owns. One in every ordinary case; more only when
    #: a wordless result spanning a merged request could not be split (ADR-0017).
    source_candidate_ids: list[str] = Field(min_length=1)
    #: The ASR requests that produced it — more than one when a truncated response was split
    #: and stitched.
    request_ids: list[str] = Field(min_length=1)
    #: Extra submissions spent resolving truncation for this segment's requests. Zero is the
    #: ordinary case and the number is here so "bounded" is checkable (ADR-0020).
    truncation_submissions: int = Field(default=0, ge=0)
    rejected_alternatives: list[RejectedAlternative] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end_sample <= self.start_sample:
            message = f"segment {self.segment_id} spans an empty interval"
            raise ValueError(message)
        if self.ownership_end_sample <= self.ownership_start_sample:
            message = f"segment {self.segment_id} spans an empty ownership interval"
            raise ValueError(message)

        for word in self.words:
            if not self.ownership_start_sample <= word.start_sample < self.ownership_end_sample:
                message = (
                    f"segment {self.segment_id} owns the word {word.text!r} starting at "
                    f"{word.start_sample}, outside its ownership interval "
                    f"[{self.ownership_start_sample}, {self.ownership_end_sample}). A word "
                    f"belongs to the interval containing its start (ADR-0020)."
                )
                raise ValueError(message)

        object.__setattr__(
            self, "words", sorted(self.words, key=lambda w: (w.start_sample, w.end_sample))
        )
        object.__setattr__(
            self,
            "rejected_alternatives",
            sorted(self.rejected_alternatives, key=lambda item: item.segment_id),
        )
        return self._check_decision()

    def _check_decision(self) -> Self:
        """A decision must be consistent with everything recorded beside it."""
        if self.decision == "duplicate":
            if self.duplicate_of_segment_id is None:
                message = (
                    f"segment {self.segment_id} is a duplicate but names nothing it duplicates. "
                    f"A collapsed alternative that cannot be traced to what absorbed it is not "
                    f"auditable."
                )
                raise ValueError(message)
            if self.duplicate_of_segment_id == self.segment_id:
                message = f"segment {self.segment_id} claims to be a duplicate of itself"
                raise ValueError(message)
            if self.overlap:
                message = (
                    f"segment {self.segment_id} is collapsed and also marked as overlapping. "
                    f"`overlap` is about retained, non-duplicate speakers."
                )
                raise ValueError(message)
            if self.rejected_alternatives:
                message = (
                    f"segment {self.segment_id} was itself collapsed and also claims to have "
                    f"rejected alternatives"
                )
                raise ValueError(message)
            return self

        if self.duplicate_of_segment_id is not None:
            message = f"segment {self.segment_id} is retained but names a segment it duplicates"
            raise ValueError(message)
        return self

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample


class TranscriptRecords(_Artifact):
    """Every normalized segment of one session, at `work/transcript-records.json`."""

    schema_version: Literal[1] = TRANSCRIPT_RECORDS_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = Field(min_length=1)
    config_hash: Sha256Hex
    #: The timeline these samples are counted on, and the graph the segments came from. A
    #: records file read beside either one it does not describe is then a detectable mistake.
    timeline_sha256: Sha256Hex
    activity_cache_key: Sha256Hex
    sample_rate: int = Field(gt=0)
    duration_samples: int = Field(ge=0)
    speakers: list[TranscriptSpeaker] = Field(default_factory=list)
    segments: list[SegmentRecord] = Field(default_factory=list)
    warnings: list[TranscriptNote] = Field(default_factory=list)
    decisions: list[TranscriptDecision] = Field(default_factory=list)
    provenance: TranscriptRecordsProvenance

    def retained(self) -> list[SegmentRecord]:
        """The segments `transcript.json` carries, in document order.

        Provided here so the renderer and any later consumer cannot disagree about what
        "retained" means — the same reason `ActivityGraph.retained` exists.
        """
        return [segment for segment in self.segments if segment.decision == "retained"]

    @model_validator(mode="after")
    def _sort_and_check(self) -> Self:
        object.__setattr__(
            self, "speakers", sorted(self.speakers, key=lambda speaker: speaker.speaker_id)
        )
        object.__setattr__(
            self,
            "segments",
            sorted(self.segments, key=lambda segment: (segment.start_sample, segment.segment_id)),
        )
        object.__setattr__(
            self, "warnings", sorted(self.warnings, key=lambda w: (w.code, w.path or "", w.message))
        )
        object.__setattr__(
            self, "decisions", sorted(self.decisions, key=lambda d: (d.code, d.subject, d.detail))
        )
        self._check_segments()
        return self

    def _check_segments(self) -> None:
        """Ids are unique, speakers exist, intervals fit, and every reference resolves."""
        ids = [segment.segment_id for segment in self.segments]
        if len(set(ids)) != len(ids):
            message = "two segments share an id, so a duplicate reference is ambiguous"
            raise ValueError(message)
        known_ids = set(ids)
        known_speakers = {speaker.speaker_id for speaker in self.speakers}
        retained = {segment.segment_id for segment in self.retained()}

        for segment in self.segments:
            if segment.speaker_id not in known_speakers:
                message = (
                    f"segment {segment.segment_id} is attributed to speaker "
                    f"{segment.speaker_id!r}, who is not in this document"
                )
                raise ValueError(message)
            if segment.end_sample > self.duration_samples:
                message = (
                    f"segment {segment.segment_id} ends at {segment.end_sample}, past the "
                    f"session's {self.duration_samples} aligned samples"
                )
                raise ValueError(message)
            self._check_references(segment, known_ids, retained)

    @staticmethod
    def _check_references(segment: SegmentRecord, known_ids: set[str], retained: set[str]) -> None:
        if segment.duplicate_of_segment_id is not None:
            if segment.duplicate_of_segment_id not in known_ids:
                message = (
                    f"segment {segment.segment_id} was collapsed into "
                    f"{segment.duplicate_of_segment_id}, which is not in this document"
                )
                raise ValueError(message)
            if segment.duplicate_of_segment_id not in retained:
                message = (
                    f"segment {segment.segment_id} was collapsed into "
                    f"{segment.duplicate_of_segment_id}, which was itself collapsed. A chain "
                    f"of duplicates has no surviving text at the end of it."
                )
                raise ValueError(message)

        for alternative in segment.rejected_alternatives:
            if alternative.segment_id not in known_ids:
                message = (
                    f"segment {segment.segment_id} rejected {alternative.segment_id}, which "
                    f"is not in this document"
                )
                raise ValueError(message)
            if alternative.segment_id == segment.segment_id:
                message = f"segment {segment.segment_id} rejected itself"
                raise ValueError(message)
