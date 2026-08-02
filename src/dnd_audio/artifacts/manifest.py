"""`manifest.json` — the immutable record of what was ingested.

**Skeleton. M1 owns this artifact and will extend it** with the FFprobe capture, the
RIFF/RF64 chunk inventory, `orig`/`edit` association, parsed timecode, and the parser's
own warnings. What is here is the envelope those fields hang off, plus the properties
that must hold from the start: sorted output, no wall-clock (INV-03), and a config hash
that ties the manifest to the configuration that produced it (INV-08).
"""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "ManifestSource",
    "ManifestTrack",
]

#: Provisional until M1 closes. See the package docstring.
MANIFEST_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManifestSource(BaseModel):
    """One source file as it was found. Never modified — INV-01."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)


class ManifestTrack(_Artifact):
    """One roster track, and whichever originals were discovered for it."""

    track_id: str
    speaker_id: str
    #: Whether discovery found a usable original. Under ``active_tracks: auto`` an
    #: inactive roster track is reported with a warning, never silently dropped.
    active: bool
    sources: list[ManifestSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_sources(self) -> ManifestTrack:
        """Sort by path so the manifest cannot depend on directory iteration order.

        Frozen models still allow this during validation, and doing it here rather than
        at each call site means no future caller can forget (INV-02).
        """
        ordered = sorted(self.sources, key=lambda source: source.relative_path)
        object.__setattr__(self, "sources", ordered)
        return self


class Manifest(_Artifact):
    """The deterministic inventory `inspect` writes to ``work/manifest.json``."""

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    session_id: str
    #: Ties this manifest to the resolved configuration that produced it (INV-08).
    config_hash: Sha256Hex
    tracks: list[ManifestTrack] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_tracks(self) -> Manifest:
        ordered = sorted(self.tracks, key=lambda track: track.track_id)
        object.__setattr__(self, "tracks", ordered)
        return self
