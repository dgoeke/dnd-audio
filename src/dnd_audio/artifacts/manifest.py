"""`manifest.json` — the immutable record of what was ingested.

M1 owns this artifact. The schema version stays 1 while M1 is open; after it closes,
only additive optional fields, and anything else bumps the version (ADR-0005).

Three shapes here are deliberate and each has a tempting wrong version:

**Timing evidence is a discriminated union** (ADR-0006). A BWF sample reference, a
timecode, and an operator's session-relative offset do not share a coordinate system —
different units, different rates, different origins, and only one of them signed. One
"start_samples" integer would be wrong for at least one of them in any session whose
sources do not all start at their own origin, and would erase the distinction M2 needs to
reconcile them.

**Sources that nothing will read are still recorded** — ignored `edit` files,
duplicates, and files found where no track is configured. The gate requires per-file
capture of every candidate, and the last group needs :attr:`Manifest.unassigned`,
because attaching them to a track would attribute an unconfigured directory to a
speaker (INV-11).

**Nothing here is a wall clock or a counter** (INV-03). There is no run time, no cache
hit count, and no hostname; the tool versions that *are* here are provenance, not
telemetry — an FFmpeg upgrade genuinely changes what was captured, and INV-08 requires
it to invalidate the work.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.artifacts.roster import RosterSummary

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "BwfSampleReferenceRecord",
    "ContainerRecord",
    "DeclinedStrategyRecord",
    "FilenameHintsRecord",
    "InspectionProvenance",
    "Manifest",
    "ManifestDecision",
    "ManifestNote",
    "ManifestSource",
    "ManifestTrack",
    "ProbeRecord",
    "RationalRate",
    "RiffChunkRecord",
    "RiffRecord",
    "SessionOffsetRecord",
    "StartEvidenceRecord",
    "StartTimeRecord",
    "TimecodeRecord",
]

#: Provisional until M1 closes. See the module docstring.
MANIFEST_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

SourceRole = Literal["selected", "associated_edit", "duplicate", "unassigned", "unusable"]

#: What a source found outside every configured track directory may be. ``duplicate`` is
#: allowed alongside ``unassigned`` because a stray copy of a track's recording is worth
#: saying so about — noting that a file duplicates another is a statement about bytes,
#: not an attribution to a speaker, and INV-11 is about the latter. What must never
#: appear here is ``selected`` or ``associated_edit``: those mean a track is using the
#: file, and no track was configured for it.
_TRACK_INDEPENDENT_ROLES: Final = frozenset({"unassigned", "duplicate"})
Variant = Literal["orig", "edit", "unknown"]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RationalRate(_Artifact):
    """An exact rate, as the two integers it actually is.

    INV-04: 30000/1001 is never serialized as 29.97. A consumer that wants a float can
    divide; a consumer that wants exactness cannot recover it from one.
    """

    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)


class BwfSampleReferenceRecord(_Artifact):
    """Samples from the recorder's own origin, at the **file's own** sample rate.

    Not midnight, whatever EBU Tech 3285 says the field means: OQ-004 measured this
    hardware writing a device-local count, and ADR-0031 records what follows. What matters
    to placement is that receivers *share* the origin, which a jam supplies (OQ-023).

    The rate is recorded because it need not be the session's: a 44.1 kHz source counts
    44100ths of a second, and reading it as 48000ths misplaces the file by 8.75%.
    """

    kind: Literal["bwf_sample_reference"] = "bwf_sample_reference"
    samples: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    origination_date: dt.date | None = None


class TimecodeRecord(_Artifact):
    """A timecode, as an exact frame index plus the rational rate it counts in.

    Not a sample position: at 30000/1001 fps a frame is 8008/5 samples, and the rule for
    rounding that onto the working grid belongs to M2 (ADR-0006).
    """

    kind: Literal["timecode"] = "timecode"
    text: str = Field(min_length=1)
    frames: int = Field(ge=0)
    frame_rate_label: str = Field(min_length=1)
    frame_rate: RationalRate
    drop_frame: bool
    recording_date: dt.date | None = None


class SessionOffsetRecord(_Artifact):
    """An operator-supplied offset: signed, 48 kHz, relative to session zero.

    Not a time of day, and not convertible into one without knowing where session zero
    is — which is M2's to determine.
    """

    kind: Literal["session_offset_samples"] = "session_offset_samples"
    samples: int
    sample_rate: int = Field(gt=0)
    recording_date: dt.date | None = None


StartEvidenceRecord = Annotated[
    BwfSampleReferenceRecord | TimecodeRecord | SessionOffsetRecord,
    Field(discriminator="kind"),
]


class DeclinedStrategyRecord(_Artifact):
    """A strategy that ran and found nothing, and what it was looking for.

    Recorded so real-capture evidence for OQ-001 is readable from these reasons rather than
    requiring the metadata investigation to be repeated.
    """

    strategy: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class StartTimeRecord(_Artifact):
    """Where this source's start time came from, and what that rests on."""

    strategy: str = Field(min_length=1)
    evidence: StartEvidenceRecord
    #: Each tagged with the open question it depends on, so `rg OQ-004` finds every
    #: manifest that would change if it were answered.
    assumptions: list[str] = Field(default_factory=list)
    declined: list[DeclinedStrategyRecord] = Field(default_factory=list)
    #: Present only when a recovery override supplied the time. The spec requires
    #: overrides to be recorded prominently: the manifest has to be able to say why a
    #: time was not read from the file.
    override_reason: str | None = None


class FilenameHintsRecord(_Artifact):
    """What the filename suggested. Hints only — never identity (INV-11), never a
    time (INV-12)."""

    recognized: bool
    variant: Variant
    tx_label: str | None = None
    sequence: int | None = None
    named_date: dt.date | None = None
    named_time: dt.time | None = None


class ContainerRecord(_Artifact):
    """The container facts, captured before any project-specific interpretation."""

    codec_name: str
    sample_format: str
    bits_per_sample: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_ts: int | None = None
    time_base: str | None = None
    #: FFprobe's displayed duration, as the string it printed. Not parsed into a float:
    #: it is a human cross-check, not arithmetic (INV-04).
    duration_text: str | None = None
    #: The exact PCM frame count where one could be established.
    sample_count: int | None = None
    sample_count_source: Literal["data_chunk", "duration_ts", "none"] = "none"
    #: Whether the `data` chunk and `duration_ts` agreed, when both existed. The
    #: synthetic half of OQ-011.
    sample_count_agrees: bool | None = None


class ProbeRecord(_Artifact):
    """Where the verbatim FFprobe output was kept, and what produced it."""

    #: Session-relative, content-hash-addressed.
    sidecar_path: str = Field(min_length=1)
    sha256: Sha256Hex
    #: The exact argument vector, so a capture taken under different options is
    #: identifiable as such (INV-08).
    command: list[str] = Field(default_factory=list)


class RiffChunkRecord(_Artifact):
    """One chunk of the container, found without FFprobe's help."""

    chunk_id: str = Field(min_length=1)
    #: Offset of the chunk **header**; the payload begins eight bytes later.
    offset: int = Field(ge=0)
    size: int = Field(ge=0)
    #: The `LIST` form type this was found inside, if any.
    container: str | None = None
    #: SHA-256 of the complete payload. Absent for audio, which is never read.
    sha256: Sha256Hex | None = None
    #: The payload as text when it was short enough to keep and safe to decode. Never
    #: a truncated prefix.
    text: str | None = None


class ManifestNote(_Artifact):
    """A warning: something worth a human's attention that did not stop the run."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None


class RiffRecord(_Artifact):
    """The generic chunk inventory (OQ-005)."""

    form: str = Field(min_length=1)
    form_type: str = Field(min_length=1)
    declared_size: int = Field(ge=0)
    file_size: int = Field(ge=0)
    #: True when the walk stopped early. The chunks recorded before that point are
    #: still valid; what follows them was not guessed at.
    truncated: bool = False
    chunks: list[RiffChunkRecord] = Field(default_factory=list)
    warnings: list[ManifestNote] = Field(default_factory=list)


class ManifestDecision(_Artifact):
    """A choice worth auditing. Deterministic by construction — no counts, no timings."""

    code: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class ManifestSource(_Artifact):
    """One candidate file, whatever became of it. Never modified — INV-01."""

    relative_path: str = Field(min_length=1)
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)
    role: SourceRole
    #: Stable and machine-readable, so a consumer branches on this rather than on prose.
    reason_code: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    #: For an ignored `edit`, the original it belongs to; for a duplicate, the copy kept.
    associated_with: str | None = None
    filename: FilenameHintsRecord
    #: Absent when the file was never probed — a duplicate or an ignored edit is
    #: recorded and left alone.
    container: ContainerRecord | None = None
    probe: ProbeRecord | None = None
    riff: RiffRecord | None = None
    start_time: StartTimeRecord | None = None
    warnings: list[ManifestNote] = Field(default_factory=list)


class ManifestTrack(_Artifact):
    """One roster track, and everything found in its directory."""

    track_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    input_path: str = Field(min_length=1)
    #: Whether discovery found a usable original. An inactive roster track is reported
    #: with a reason, never silently dropped.
    active: bool
    inactive_reason: str | None = None
    sources: list[ManifestSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_sources(self) -> Self:
        """Sort by path so directory iteration order cannot reach the artifact (INV-02).

        Done during validation rather than at each call site, so no future caller can
        forget. Frozen models still permit this here.
        """
        object.__setattr__(self, "sources", sorted(self.sources, key=lambda s: s.relative_path))
        return self

    @model_validator(mode="after")
    def _inactive_tracks_say_why(self) -> Self:
        if not self.active and not self.inactive_reason:
            message = f"track {self.track_id} is inactive without a reason"
            raise ValueError(message)
        if self.active and self.inactive_reason is not None:
            message = f"track {self.track_id} is active but carries an inactive reason"
            raise ValueError(message)
        return self


class InspectionProvenance(_Artifact):
    """What produced this manifest. Deterministic, and part of the cache identity.

    Tool versions live here rather than in the report's telemetry because they are not
    per-run measurements: the same tools on unchanged bytes produce the same capture,
    and a different FFmpeg legitimately produces a different one (INV-08). Nothing here
    varies between two identical runs, which is what INV-03 asks.
    """

    ffmpeg_version: str = Field(min_length=1)
    ffprobe_version: str = Field(min_length=1)
    ffprobe_args: list[str] = Field(default_factory=list)
    #: Covers every parser in the inspection package. A fix in any of them must
    #: re-inspect, not re-serve the answer the bug produced.
    semantics_version: int = Field(ge=1)


class Manifest(_Artifact):
    """The deterministic inventory `inspect` writes to ``work/manifest.json``."""

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    #: Ties this manifest to the resolved configuration that produced it (INV-08).
    config_hash: Sha256Hex
    inspection: InspectionProvenance
    roster: RosterSummary
    tracks: list[ManifestTrack] = Field(default_factory=list)
    #: Candidates found where no track is configured. Captured in full, attributed to
    #: nobody (INV-11). Every entry has ``role: unassigned``.
    unassigned: list[ManifestSource] = Field(default_factory=list)
    warnings: list[ManifestNote] = Field(default_factory=list)
    decisions: list[ManifestDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_and_check(self) -> Self:
        object.__setattr__(self, "tracks", sorted(self.tracks, key=lambda t: t.track_id))
        object.__setattr__(
            self, "unassigned", sorted(self.unassigned, key=lambda s: s.relative_path)
        )
        object.__setattr__(
            self, "warnings", sorted(self.warnings, key=lambda w: (w.code, w.path or "", w.message))
        )
        object.__setattr__(
            self, "decisions", sorted(self.decisions, key=lambda d: (d.code, d.subject))
        )

        misfiled = [
            source.relative_path
            for source in self.unassigned
            if source.role not in _TRACK_INDEPENDENT_ROLES
        ]
        if misfiled:
            message = (
                f"a source outside every configured track may only be "
                f"{' or '.join(sorted(_TRACK_INDEPENDENT_ROLES))}, but "
                f"{', '.join(sorted(misfiled))} is not. A file nobody configured a track "
                f"for cannot be selected or paired here (INV-11)."
            )
            raise ValueError(message)
        return self
