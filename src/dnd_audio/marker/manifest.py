"""`marker-manifest.json` — what was built, and what its bytes are.

Published **last**, and only after both artifacts exist and have been validated, so its
presence is the completeness marker for the set (ADR-0041). Deliberately deterministic: two
builds of the same spec produce identical manifests, which means the file carries no wall
clock, no hostname, no absolute path, and no output directory — an operator can compare two
machines' manifests and a difference is a real difference.

**It does not hash itself.** Writing the hash would change the bytes the hash describes;
ADR-0003 established that for `ingest-report.json` and the same fixed point applies here.
What a consumer gets instead is every *other* artifact's digest, which is the useful half:
the question is whether the WAV on the phone is the WAV the analyzer looks for.

The chirp and gap intervals are published because they are the marker's code, and an operator
confirming what was played against what was searched for should not have to read Python to do
it. The detector derives its templates from the same spec, so these are the intervals that
were actually built rather than a description maintained beside them.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.marker import MARKER_SAMPLE_RATE, MARKER_SEMANTICS_VERSION
from dnd_audio.marker.spec import MarkerSpec
from dnd_audio.marker.wav import BITS_PER_SAMPLE, CHANNELS, SAMPLE_FORMAT

__all__ = ["MARKER_MANIFEST_SCHEMA_VERSION", "ChirpRecord", "MarkerArtifact", "MarkerManifest"]

MARKER_MANIFEST_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarkerArtifact(_Artifact):
    """One published file, by name and content.

    The filename is relative and carries no directory: the manifest sits beside what it
    describes, and an absolute path would name the machine that built it.
    """

    filename: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256Hex

    @model_validator(mode="after")
    def _reject_a_path(self) -> Self:
        if "/" in self.filename or "\\" in self.filename:
            message = f"{self.filename!r} is a path; the manifest names files beside itself"
            raise ValueError(message)
        return self


class ChirpRecord(_Artifact):
    """One chirp, where it sits and what it sweeps."""

    start_hz: int = Field(gt=0)
    end_hz: int = Field(gt=0)
    #: Half-open ``[start, end)`` relative to the WAV's first sample, like every interval in
    #: this project's artifacts.
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    fade_samples: int = Field(ge=0)


class MarkerManifest(_Artifact):
    """The complete description of one `marker build`."""

    schema_version: Literal[1] = MARKER_MANIFEST_SCHEMA_VERSION
    #: Which waveform. Public builds use bench-selected v1; candidate names remain for the
    #: reproducible M10 evidence (ADR-0042).
    marker_name: str = Field(min_length=1)
    #: Bumping this changes the bytes, so it belongs beside the digests rather than in a
    #: separate provenance section nobody reads.
    marker_semantics_version: int = Field(ge=1)
    #: Why this waveform exists, carried through from the spec so a WAV on a phone can be
    #: traced back to the question it was built to answer.
    rationale: str = Field(min_length=1)

    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    bits_per_sample: int = Field(gt=0)
    sample_format: str = Field(min_length=1)

    total_samples: int = Field(gt=0)
    #: The frozen anchor: the first sample of the first chirp, relative to the WAV's start.
    #: Every lag this project reports is measured from it (ADR-0041).
    anchor_sample: int = Field(ge=0)
    peak_amplitude: int = Field(gt=0)
    lead_silence_samples: int = Field(ge=0)
    trail_silence_samples: int = Field(ge=0)

    chirps: list[ChirpRecord] = Field(min_length=2)
    #: Half-open ``[start, end)`` per inter-chirp gap. The asymmetry is the code.
    gap_intervals: list[tuple[int, int]] = Field(min_length=1)

    wav: MarkerArtifact
    page: MarkerArtifact

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> Self:
        if len(self.gap_intervals) != len(self.chirps) - 1:
            message = (
                f"{len(self.chirps)} chirps imply {len(self.chirps) - 1} gaps, "
                f"got {len(self.gap_intervals)}"
            )
            raise ValueError(message)
        if self.anchor_sample != self.chirps[0].start_sample:
            message = (
                f"anchor_sample={self.anchor_sample} but the first chirp starts at "
                f"{self.chirps[0].start_sample}; the anchor is defined as that sample"
            )
            raise ValueError(message)
        if len({end - start for start, end in self.gap_intervals}) != len(self.gap_intervals):
            message = "the gaps are equal, which makes a reversed sequence indistinguishable"
            raise ValueError(message)
        if self.wav.filename == self.page.filename:
            message = "the WAV and the page cannot be the same file"
            raise ValueError(message)
        return self

    @property
    def duration_seconds(self) -> float:
        """For a human reading the manifest. Never used to decide anything."""
        return self.total_samples / self.sample_rate


def describe(spec: MarkerSpec, *, wav: MarkerArtifact, page: MarkerArtifact) -> MarkerManifest:
    """The manifest for ``spec`` and the two artifacts just written.

    Built from the spec rather than from the files, so the intervals published are the ones
    synthesis used — there is no second description to drift from what was generated.
    """
    return MarkerManifest(
        marker_name=spec.name,
        marker_semantics_version=MARKER_SEMANTICS_VERSION,
        rationale=spec.rationale,
        sample_rate=MARKER_SAMPLE_RATE,
        channels=CHANNELS,
        bits_per_sample=BITS_PER_SAMPLE,
        sample_format=SAMPLE_FORMAT,
        total_samples=spec.total_samples,
        anchor_sample=spec.anchor_sample,
        peak_amplitude=spec.peak_amplitude,
        lead_silence_samples=spec.lead_silence_samples,
        trail_silence_samples=spec.trail_silence_samples,
        chirps=[
            ChirpRecord(
                start_hz=chirp.start_hz,
                end_hz=chirp.end_hz,
                start_sample=start,
                end_sample=end,
                fade_samples=chirp.fade_samples,
            )
            for chirp, (start, end) in zip(spec.chirps, spec.chirp_intervals(), strict=True)
        ],
        gap_intervals=list(spec.gap_intervals()),
        wav=wav,
        page=page,
    )
