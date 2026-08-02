"""What the roster looked like, as both the manifest and the report must state it.

Defined once and embedded in both artifacts rather than written twice. The spec asks
for the same facts in each — known roster, observed active, per-track file counts, and
the missing, empty, and extra directories — and two independently drifting copies of
one truth is how a report ends up disagreeing with the manifest it describes.

The cost is that a change here bumps both artifacts' schema versions. That is the right
coupling: it really is the same information.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["RosterSummary"]


class RosterSummary(BaseModel):
    """Who was configured, who was recording, and what was found where."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Every track in `session.yaml`, whether or not it recorded anything.
    known_tracks: list[str] = Field(default_factory=list)
    #: The subset with at least one usable original — "observed active", in the spec's
    #: words. Under an explicit `active_tracks` list, the subset the operator required.
    active_tracks: list[str] = Field(default_factory=list)
    #: Configured but not recording. Present rather than absent, because "Erin did not
    #: come" and "Erin's recorder failed" are different problems and only a human can
    #: tell them apart.
    inactive_tracks: list[str] = Field(default_factory=list)
    #: Candidate files found per track, including the ones that were not selected.
    file_counts: dict[str, int] = Field(default_factory=dict)
    #: Configured, but the directory does not exist.
    missing_directories: list[str] = Field(default_factory=list)
    #: The directory exists and holds no candidate.
    empty_directories: list[str] = Field(default_factory=list)
    #: A directory beside the configured ones that no track claims. Its files are
    #: captured and attributed to nobody (INV-11).
    extra_directories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_everything(self) -> Self:
        """Sort here rather than at each call site, so no caller can forget (INV-02)."""
        for field in (
            "known_tracks",
            "active_tracks",
            "inactive_tracks",
            "missing_directories",
            "empty_directories",
            "extra_directories",
        ):
            object.__setattr__(self, field, sorted(getattr(self, field)))
        object.__setattr__(self, "file_counts", dict(sorted(self.file_counts.items())))
        return self
