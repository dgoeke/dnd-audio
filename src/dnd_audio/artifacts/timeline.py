"""`timeline.json` — where every source sample sits on the session's clock.

M2 owns this artifact. It is the **authoritative** 48 kHz working path: a segment map, not
a pile of materialized audio (ADR-0011). A contiguous 48 kHz file is an optional,
disposable cache artifact; this document is the thing that is true.

Four shapes here are deliberate, and each has a tempting wrong version.

**There are no floats in this document at all.** Not one. Sample counts are integers and
rates are `{numerator, denominator}`, because INV-04 forbids a fractional rate becoming a
binary float anywhere in timestamp arithmetic and a "just for display" seconds field is
how that rule dies. `tests/test_timeline_artifact.py` walks the serialized document and
fails on any float it finds. Human-facing seconds belong to the report.

**Every interval is half-open**, `[start, start + n_samples)`. Stated once here rather than
implied by each consumer's arithmetic, because M3 and M5 both index into these and an
off-by-one that only shows up as a clipped word is expensive to find later.

**A track's map tiles its own extent with no holes**, gaps included as explicit `silence`
segments. Enforced by a validator rather than trusted: a map with a hole in it has two
readings — silence, or "the builder forgot" — and a reader cannot tell them apart.

**Both the rasterized start and the placed start are kept.** `evidence_start_sample` is
where ADR-0008's arithmetic put the chunk; `session_start_sample` is where the layout put
it after applying ADR-0010's overlap policy; `shift_samples` is the difference. Recording
only the second would erase the fact that the evidence disagreed, which is precisely what
an operator debugging a bad sync needs to see.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.artifacts.manifest import RationalRate

__all__ = [
    "TIMELINE_SCHEMA_VERSION",
    "DerivativeRecord",
    "SessionZero",
    "Timeline",
    "TimelineDecision",
    "TimelineNote",
    "TimelineProvenance",
    "TimelineSegment",
    "TimelineTrack",
]

#: Provisional until M2 closes. After that, only additive optional fields; anything else
#: bumps the version (ADR-0005).
TIMELINE_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

#: How session zero was decided. ``configured_origin`` means the operator stated it;
#: ``earliest_source`` means it was derived, and the whole timeline was shifted so the
#: earliest valid source start lands at zero (ADR-0009).
ZeroSource = Literal["configured_origin", "earliest_source"]

#: The coordinate system the day origin belongs to. ``real_time`` is midnight, from a BWF
#: sample reference. ``timecode`` is the recorder's ``00:00:00:00``, which is *not* real
#: midnight at a fractional non-drop rate (OQ-015). ``relative`` means no absolute evidence
#: existed and the origin is the offsets' own.
ZeroDomain = Literal["real_time", "timecode", "relative"]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimelineNote(_Artifact):
    """A warning: worth a human's attention, did not stop the run."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None


class TimelineDecision(_Artifact):
    """A placement choice worth auditing — a rollover inferred, an overlap nudged.

    Deterministic by construction: no counters, no timings. Two identical runs produce
    identical decisions (INV-02, INV-03).
    """

    code: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class SessionZero(_Artifact):
    """Where session time zero is, and how that was decided (ADR-0009)."""

    source: ZeroSource
    domain: ZeroDomain
    #: The calendar day the timeline is anchored to, when one is known. Never inferred
    #: from a date-shaped ``session_id``.
    origin_date: dt.date | None = None
    #: The configured origin timecode, when there was one.
    origin_timecode: str | None = None
    #: Session zero's own position, in samples since its domain's day origin at the
    #: canonical rate. ``None`` when the domain is ``relative`` and there is no day.
    since_day_origin_samples: int | None = Field(default=None, ge=0)
    detail: str = Field(min_length=1)


class TimelineSegment(_Artifact):
    """One contiguous run on a track's timeline: audio from a source, or silence.

    Silence is a first-class segment rather than an absence. A transmitter switched off
    and back on leaves a real hole, and representing it explicitly is what stops a reader
    from sliding the later audio earlier to close it.
    """

    kind: Literal["audio", "silence"]
    session_start_sample: int = Field(ge=0)
    n_samples: int = Field(gt=0)
    #: Audio only: which file, verified by hash, and the PCM frame offset within it.
    source_relative_path: str | None = None
    source_sha256: Sha256Hex | None = None
    source_start_sample: int | None = Field(default=None, ge=0)
    #: Audio only: where ADR-0008's rasterization put this chunk, before the layout
    #: applied any overlap policy. Equals ``session_start_sample`` when nothing shifted.
    evidence_start_sample: int | None = None
    #: What the layout added. Non-zero only for a nudged overlap (ADR-0010).
    shift_samples: int = 0

    @property
    def session_end_sample(self) -> int:
        """One past the last sample. Intervals are half-open."""
        return self.session_start_sample + self.n_samples

    @model_validator(mode="after")
    def _audio_and_silence_carry_different_fields(self) -> Self:
        audio_fields = (self.source_relative_path, self.source_sha256, self.source_start_sample)
        if self.kind == "audio":
            if any(value is None for value in audio_fields):
                message = (
                    f"an audio segment at {self.session_start_sample} must name its source "
                    f"file, that file's hash, and the sample offset within it"
                )
                raise ValueError(message)
            if self.evidence_start_sample is None:
                message = (
                    f"an audio segment at {self.session_start_sample} must record where its "
                    f"evidence rasterized to, so a shift is visible rather than absorbed"
                )
                raise ValueError(message)
            if self.session_start_sample != self.evidence_start_sample + self.shift_samples:
                message = (
                    f"segment at {self.session_start_sample} claims evidence "
                    f"{self.evidence_start_sample} plus shift {self.shift_samples}, which "
                    f"do not sum to it"
                )
                raise ValueError(message)
            return self

        if any(value is not None for value in (*audio_fields, self.evidence_start_sample)):
            message = (
                f"a silence segment at {self.session_start_sample} must not name a source: "
                f"silence is a hole in the recording, not a file"
            )
            raise ValueError(message)
        if self.shift_samples:
            message = "a silence segment cannot be shifted; it is defined by its neighbours"
            raise ValueError(message)
        return self


class DerivativeRecord(_Artifact):
    """A cached resampling of one track, and everything needed to map back to 48 kHz.

    Output sample `k` corresponds to input sample `k * decimation`, exactly, because the
    group delay divides by the decimation factor — see :mod:`dnd_audio.timeline.fir`. The
    reverse direction lands between grid points, so an interval is converted by flooring
    its start and ceiling its end (`timeline.resample.to_derivative_interval`) rather than
    by one rounding rule for both ends.
    """

    sample_rate: int = Field(gt=0)
    #: Session-relative. Under ``work/cache/``: regenerable, and named by its identity.
    relative_path: str = Field(min_length=1)
    #: The full INV-08 identity this artifact was built under. A change to the segment
    #: map, the sources, the configuration, either semantics version, the filter, or
    #: NumPy/SciPy produces a different key and therefore a rebuild.
    cache_key: Sha256Hex
    size_bytes: int = Field(ge=0)
    input_samples: int = Field(ge=0)
    output_samples: int = Field(ge=0)
    decimation: int = Field(gt=0)
    filter_name: str = Field(min_length=1)
    filter_identity: Sha256Hex
    group_delay_input_samples: int = Field(ge=0)
    group_delay_output_samples: int = Field(ge=0)
    #: How a length that does not divide evenly is handled. ``ceil`` keeps the tail by
    #: zero-padding; truncating would silently shorten every track whose length is not a
    #: multiple of the decimation factor.
    length_rule: Literal["ceil"] = "ceil"

    @model_validator(mode="after")
    def _length_follows_the_rule(self) -> Self:
        expected = -(-self.input_samples // self.decimation)
        if self.output_samples != expected:
            message = (
                f"{self.input_samples} input samples decimated by {self.decimation} is "
                f"{expected} output samples under the {self.length_rule} rule, not "
                f"{self.output_samples}"
            )
            raise ValueError(message)
        if self.group_delay_output_samples * self.decimation != self.group_delay_input_samples:
            message = (
                f"a group delay of {self.group_delay_input_samples} input samples is not "
                f"{self.group_delay_output_samples} whole output samples"
            )
            raise ValueError(message)
        return self


class TimelineTrack(_Artifact):
    """One person's reconstructed virtual track."""

    track_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    #: First and last-plus-one sample of this track's own extent. A track that started
    #: late begins after zero, and one that stopped early ends before the session does;
    #: neither is padded here, because padding would put invented silence in the map.
    #: A reader returns silence outside this range up to the session duration.
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    segments: list[TimelineSegment] = Field(default_factory=list)
    derivatives: list[DerivativeRecord] = Field(default_factory=list)
    warnings: list[TimelineNote] = Field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    @model_validator(mode="after")
    def _segments_tile_the_extent(self) -> Self:
        """The map has no holes and no overlaps, and covers exactly the stated extent.

        Checked here rather than trusted to the builder. A hole in a segment map has two
        readings — silence, or a builder that forgot — and a consumer cannot distinguish
        them, so the shape that admits the ambiguity is rejected at the boundary.
        """
        if not self.segments:
            if self.start_sample != self.end_sample:
                message = (
                    f"track {self.track_id} spans "
                    f"[{self.start_sample}, {self.end_sample}) with no segments"
                )
                raise ValueError(message)
            return self

        ordered = sorted(self.segments, key=lambda s: s.session_start_sample)
        object.__setattr__(self, "segments", ordered)

        if ordered[0].session_start_sample != self.start_sample:
            message = (
                f"track {self.track_id} starts at {self.start_sample} but its first "
                f"segment starts at {ordered[0].session_start_sample}"
            )
            raise ValueError(message)

        position = self.start_sample
        for segment in ordered:
            if segment.session_start_sample != position:
                shape = "hole" if segment.session_start_sample > position else "overlap"
                message = (
                    f"track {self.track_id} has a {shape} "
                    f"at sample {position}: the next segment starts at "
                    f"{segment.session_start_sample}. A gap is an explicit silence "
                    f"segment, never an absence."
                )
                raise ValueError(message)
            position = segment.session_end_sample

        if position != self.end_sample:
            message = (
                f"track {self.track_id} ends at {self.end_sample} but its segments run "
                f"to {position}"
            )
            raise ValueError(message)

        if ordered[0].kind == "silence" or ordered[-1].kind == "silence":
            message = (
                f"track {self.track_id} begins or ends with silence. A track's extent is "
                f"defined by its audio; leading or trailing silence is the reader's job, "
                f"not the map's."
            )
            raise ValueError(message)
        return self


class TimelineProvenance(_Artifact):
    """What produced this timeline. Deterministic, and part of every cache identity.

    NumPy and SciPy are here for the same reason M1 records FFmpeg's version: they are
    external implementations whose upgrade can legitimately change what the derivatives
    contain, and INV-08 requires that to invalidate the work rather than serve it stale.
    """

    timeline_semantics_version: int = Field(ge=1)
    #: The version of the package that produced the timing evidence underneath. A parser
    #: fix in M1 moves a chunk without changing a single source byte.
    inspection_semantics_version: int = Field(ge=1)
    numpy_version: str = Field(min_length=1)
    scipy_version: str = Field(min_length=1)


class Timeline(_Artifact):
    """The deterministic segment map `ingest` writes to ``work/timeline.json``."""

    schema_version: Literal[1] = TIMELINE_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    config_hash: Sha256Hex
    #: Hash of the `manifest.json` this was built from. Ties the two artifacts together,
    #: so a timeline cannot be read alongside a manifest that no longer describes it.
    manifest_sha256: Sha256Hex
    provenance: TimelineProvenance
    #: The grid every sample index in this document is counted on.
    sample_rate: int = Field(gt=0)
    #: Set by the latest track end, never by the shortest track.
    duration_samples: int = Field(ge=0)
    session_zero: SessionZero
    frame_rate_label: str = Field(min_length=1)
    frame_rate: RationalRate
    tracks: list[TimelineTrack] = Field(default_factory=list)
    warnings: list[TimelineNote] = Field(default_factory=list)
    decisions: list[TimelineDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_and_check(self) -> Self:
        object.__setattr__(self, "tracks", sorted(self.tracks, key=lambda t: t.track_id))
        object.__setattr__(
            self, "warnings", sorted(self.warnings, key=lambda w: (w.code, w.path or "", w.message))
        )
        object.__setattr__(
            self, "decisions", sorted(self.decisions, key=lambda d: (d.code, d.subject, d.detail))
        )

        latest = max((track.end_sample for track in self.tracks), default=0)
        if self.duration_samples != latest:
            message = (
                f"duration_samples is {self.duration_samples} but the latest track ends at "
                f"{latest}. The aligned duration is set by the latest track end, not by the "
                f"shortest track and not by anything else."
            )
            raise ValueError(message)
        return self
