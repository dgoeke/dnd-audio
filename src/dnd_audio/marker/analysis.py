"""`work/sync-marker-analysis.json` — every occurrence, every lag, and what it does not mean.

Deterministic and byte-stable (INV-02), which here means **no floats anywhere in the
document**, exactly as `timeline.json` carries none. Every position is an integer sample,
every score is integer permille, and the only concession to human reading is an integer
millisecond alongside a sample count. A threshold compared as a float is the defect M8 removed
from `sync_qa`, and keeping floats out of the artifact is what stops it coming back through
the side door.

**The vocabulary is the correctness property**, in the same way ADR-0039's three words for
"checked" were M7a's. A start-to-end change in a track's lag is
:attr:`ArrivalOutcome.DIFFERENTIAL_ARRIVAL` unless the event log asserts one unchanged
geometry for the phone and every compared transmitter; only then may it be
:attr:`ArrivalOutcome.CLOCK_DRIFT_EVIDENCE`. Six lavs at a table are 0.5–3 m from any source,
so a wearer leaning back moves the acoustic term by the same order as the drift being looked
for (ADR-0040). Merging the two would be a confident wrong answer, which is worse than an
honest inconclusive.

**Nothing here corrects anything.** The timeline is not touched, no sample moves, and valid
timecode is never overridden. The analysis is evidence an operator reads.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.determinism import canonical_json, sha256_bytes

__all__ = [
    "MARKER_ANALYSIS_SCHEMA_VERSION",
    "AnalysisIdentity",
    "ArrivalComparison",
    "ArrivalOutcome",
    "DetectedOccurrence",
    "GroupMember",
    "OccurrenceGroup",
    "SyncMarkerAnalysis",
    "TimecodeComparison",
]

MARKER_ANALYSIS_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArrivalOutcome(StrEnum):
    """What a start-to-end change in one track's lag is permitted to mean (ADR-0040)."""

    #: Geometry unasserted or known to have changed. The measurement stands; the cause does
    #: not. This is the normal outcome for any session where people wear the microphones.
    DIFFERENTIAL_ARRIVAL = "differential_arrival"
    #: The event log asserts one unchanged geometry ID across both occurrences, for the
    #: source and every compared transmitter. Only here may the clocks be blamed.
    CLOCK_DRIFT_EVIDENCE = "clock_drift_evidence"
    #: One end is missing, weak, clipped or ambiguous. Reported rather than guessed at: the
    #: analyzer never fabricates a lag because the report has a field for one.
    INCONCLUSIVE = "inconclusive"


class DetectionOutcome(StrEnum):
    """Why a track has, or does not have, a usable arrival for one occurrence."""

    DETECTED = "detected"
    #: Nothing reached the sequence threshold in this occurrence's neighbourhood.
    MISSING = "missing"
    #: Found, but the window is at or near full scale, so the position is usable and the
    #: score is not. A different fact from a weak match, and kept distinct for the reason
    #: M8 had to separate "nobody clapped" from "the jam failed".
    CLIPPED = "clipped"
    #: Found in a window with essentially no signal.
    WEAK = "weak"
    #: A comparable alternative sits inside the association window, so which arrival belongs
    #: to this occurrence is not decidable.
    AMBIGUOUS = "ambiguous"


class AnalysisIdentity(_Artifact):
    """Everything that can change this analysis, named rather than hashed together.

    Kept separate from :meth:`digest` deliberately — the `derivative_identity_document`
    pattern M2 established. A key that changes for the right reason can still be missing the
    component that matters later, and only a test that reads the components by name can
    notice that.

    Three semantic versions rather than one, because three independent things move the
    result and a single number would let a change to grouping hide behind a detector bump
    (ADR-0041, second plan review finding 6).
    """

    marker_semantics_version: int = Field(ge=1)
    detector_semantics_version: int = Field(ge=1)
    #: Occurrence grouping, cross-track association, role assignment, geometry
    #: classification and source-coordinate mapping. The one the first draft omitted.
    marker_analysis_semantics_version: int = Field(ge=1)

    marker_name: str = Field(min_length=1)
    #: Digest of the canonical **WAV file**, header included — the same value
    #: `marker-manifest.json` publishes and the operator copied to the phone, so the two can
    #: be compared directly. Named for what it is: an earlier draft called this
    #: `marker_pcm_sha256`, which described the samples rather than the file.
    marker_wav_sha256: Sha256Hex

    #: Schema versions of everything consumed, so a format change is visible here too.
    timeline_schema_version: int = Field(ge=1)
    event_log_schema_version: int | None = Field(default=None, ge=1)
    #: Digest of the event log's *model*, so reformatting the YAML changes nothing and
    #: editing a number changes everything. ``None`` when no log was supplied.
    event_log_sha256: Sha256Hex | None = None

    config_hash: Sha256Hex
    manifest_sha256: Sha256Hex
    timeline_config_hash: Sha256Hex

    reference_track: str = Field(min_length=1)
    #: Every threshold and tie-break, in permille or samples. Integers only.
    thresholds: dict[str, int] = Field(default_factory=dict)
    #: The exact half-open searched set, after canonicalization, as sample pairs.
    searched_intervals: list[tuple[int, int]] = Field(default_factory=list)

    numpy_version: str = Field(min_length=1)
    scipy_version: str = Field(min_length=1)

    def digest(self) -> str:
        """One hash over every component above."""
        return sha256_bytes(canonical_json(self.model_dump(mode="json")).encode("utf-8"))


class DetectedOccurrence(_Artifact):
    """One accepted marker sequence on one track, with where it maps to in a source file."""

    track_id: str = Field(min_length=1)
    #: Session sample of the marker's frozen anchor — the first sample of the first chirp.
    anchor_sample: int = Field(ge=0)
    #: The same instant in the operator's units. Present for reading, never for comparing.
    anchor_ms: int = Field(ge=0)
    score_permille: int = Field(ge=0, le=1000)
    runner_up_permille: int = Field(ge=0, le=1000)
    #: Measured minus canonical, per inter-chirp gap. The quantity OQ-029 asks about, and
    #: the one that distinguishes a clock problem from a scheduling one at the bench.
    gap_errors_samples: list[int] = Field(default_factory=list)
    #: Where the anchor falls in a real recording, when it falls in one. ``None`` when the
    #: anchor lands in silence — before the transmitter started, inside a gap, or after it
    #: stopped — which is a real answer rather than a missing one.
    source_relative_path: str | None = None
    source_sample: int | None = Field(default=None, ge=0)
    clipped: bool = False
    weak: bool = False

    @model_validator(mode="after")
    def _source_coordinates_come_as_a_pair(self) -> Self:
        if (self.source_relative_path is None) != (self.source_sample is None):
            message = (
                f"{self.track_id} has a half-populated source coordinate; a path without a "
                f"sample names a file and no position in it"
            )
            raise ValueError(message)
        return self


class GroupMember(_Artifact):
    """One track's participation in a reference-anchored occurrence group."""

    track_id: str = Field(min_length=1)
    outcome: DetectionOutcome
    anchor_sample: int | None = Field(default=None, ge=0)
    #: ``track_anchor - reference_anchor``. Positive means this track heard it later.
    #: ``None`` for anything but a usable detection.
    relative_lag_samples: int | None = None
    score_permille: int | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def _only_a_detection_carries_a_lag(self) -> Self:
        detected = self.outcome is DetectionOutcome.DETECTED
        if not detected and self.relative_lag_samples is not None:
            message = (
                f"{self.track_id} is `{self.outcome.value}` but carries a lag. A lag that is "
                f"not from a clean detection is a number nobody should compare."
            )
            raise ValueError(message)
        if detected and (self.anchor_sample is None or self.relative_lag_samples is None):
            message = f"{self.track_id} is `detected` without an anchor and a lag"
            raise ValueError(message)
        return self


class OccurrenceGroup(_Artifact):
    """One acoustic event, as heard on every track that heard it."""

    group_index: int = Field(ge=0)
    reference_anchor_sample: int = Field(ge=0)
    #: From the event log, or from the one-event-per-default-window rule. **Never** from peak
    #: strength — a louder detection is not a more start-like one (ADR-0041).
    role: str | None = None
    role_source: Literal["event_log", "default_window", "unassigned"] = "unassigned"
    #: The operator's assertion that nothing moved. Only a group carrying one may take part
    #: in a drift classification.
    geometry_id: str | None = None
    members: list[GroupMember] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_members(self) -> Self:
        object.__setattr__(self, "members", sorted(self.members, key=lambda m: m.track_id))
        return self


class TimecodeComparison(_Artifact):
    """What the metadata predicted against what the audio shows, for one track.

    Jam QA, and nothing else: a disagreement is reported, the timeline is not adjusted, and
    valid timecode is never overridden by a correlation (ADR-0040).
    """

    track_id: str = Field(min_length=1)
    measured_lag_samples: int
    #: The offset `bext.time_reference` implies for this track against the reference. Zero
    #: when both were placed from the same evidence.
    predicted_lag_samples: int
    disagreement_samples: int
    #: The finest offset this session's own timing evidence could express, from
    #: `syncqa.offset_floor_samples`. M8's measured quantization floor — a matched filter is
    #: far more precise than a 33.3 ms timecode quantum, and a healthy within-one-quantum
    #: offset must not become a failed jam because the instrument improved.
    quantum_floor_samples: int = Field(gt=0)
    beyond_quantum: bool


class ArrivalComparison(_Artifact):
    """How one track's lag changed between two occurrences, and what that is allowed to mean."""

    track_id: str = Field(min_length=1)
    start_lag_samples: int | None = None
    end_lag_samples: int | None = None
    #: End minus start. ``None`` when either end is missing.
    change_samples: int | None = None
    change_ms: int | None = None
    outcome: ArrivalOutcome
    #: Why it is not stronger than it is, in the operator's words.
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def _drift_requires_both_ends(self) -> Self:
        if self.outcome is not ArrivalOutcome.INCONCLUSIVE and self.change_samples is None:
            message = (
                f"{self.track_id} reports `{self.outcome.value}` with no measured change; "
                f"only `inconclusive` may lack one"
            )
            raise ValueError(message)
        return self


class SyncMarkerAnalysis(_Artifact):
    """The deterministic record of one `marker analyze` run."""

    schema_version: Literal[1] = MARKER_ANALYSIS_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    identity: AnalysisIdentity
    #: Every accepted occurrence on every track, not only the chosen start/end pair. The
    #: charter is explicit: a repeat, a miss, or a moved-phone diagnostic must be visible
    #: rather than silently absorbed.
    occurrences: list[DetectedOccurrence] = Field(default_factory=list)
    groups: list[OccurrenceGroup] = Field(default_factory=list)
    #: Detections on non-reference tracks that no group claimed. Kept rather than dropped:
    #: an unmatched arrival is evidence about the reference track, not noise.
    unmatched: list[DetectedOccurrence] = Field(default_factory=list)
    timecode: list[TimecodeComparison] = Field(default_factory=list)
    arrival: list[ArrivalComparison] = Field(default_factory=list)
    #: Operator-facing observations that did not stop the analysis.
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _canonical_order(self) -> Self:
        """Sorted by position then track, so two runs cannot differ by iteration order."""
        object.__setattr__(
            self,
            "occurrences",
            sorted(self.occurrences, key=lambda item: (item.anchor_sample, item.track_id)),
        )
        object.__setattr__(
            self,
            "unmatched",
            sorted(self.unmatched, key=lambda item: (item.anchor_sample, item.track_id)),
        )
        object.__setattr__(self, "groups", sorted(self.groups, key=lambda item: item.group_index))
        object.__setattr__(self, "timecode", sorted(self.timecode, key=lambda item: item.track_id))
        object.__setattr__(self, "arrival", sorted(self.arrival, key=lambda item: item.track_id))
        return self

    @property
    def conclusive(self) -> bool:
        """Whether anything was measured at all. Drives the report's inconclusive note."""
        return bool(self.groups)
