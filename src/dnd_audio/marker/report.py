"""`output/marker-report.json` — INV-13 at the marker command's own boundary.

Not a seventh stage in `ingest-report.json`. ADR-0039 settled the shape of this argument for
the archive and it applies unchanged here: the processing report accounts for six named stages
and refuses to build with a gap, so adding one would mean five commands skipping it with a
reason, and `overall_status` would stop describing a pipeline run. Marker analysis is not in
the stage DAG and never will be.

**An inconclusive measurement is a completed command.** The single most important distinction
in this file. "Nobody played the marker", "it was played and the room swallowed it", and "the
timeline is stale so nothing could be read" are three different things, and only the last is a
failure. A report that called a quiet room a failure would train an operator to ignore the
exit code — and one that called a stale timeline a success would hide the thing they need to
fix. Missing, weak, clipped or ambiguous evidence is `complete` with warnings; invalid inputs,
corrupt sources and unsafe paths are `failed`.

Written atomically whichever way the run went, and **not written at all** when its own
resolved path would land under a session's sources: INV-01 outranks INV-13 there, for the
reason M1 established — a report is regenerable and a source recording is not.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.determinism import write_json_atomic
from dnd_audio.errors import ExitCode

__all__ = [
    "MARKER_REPORT_SCHEMA_VERSION",
    "AnalysisStatus",
    "MarkerReport",
    "MarkerReportError",
    "MarkerReportWarning",
    "ReportDeliverable",
    "write_marker_report",
]

MARKER_REPORT_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AnalysisStatus(StrEnum):
    """The marker-analysis stage's outcome. One stage, because there is one."""

    #: It ran and produced an analysis — including one that measured nothing, which is a
    #: result about the room rather than a failure of the command.
    COMPLETE = "complete"
    #: It could not run: stale artifacts, a corrupt source, an unusable event log.
    FAILED = "failed"
    #: Not attempted. Present so the vocabulary matches `ingest-report.json`'s.
    SKIPPED = "skipped"


class OverallStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarkerReportError(_Artifact):
    """A failure a caller can branch on without parsing prose."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    #: Session-relative when it concerns a file. Never an absolute path: a report is a thing
    #: people paste, and an absolute path names the machine.
    path: str | None = None


class MarkerReportWarning(_Artifact):
    """Something an operator must read, which did not stop the measurement."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None


class ReportDeliverable(_Artifact):
    """A file this run produced, and its hash — every one except this report (ADR-0003)."""

    relative_path: str = Field(min_length=1)
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)


class MarkerReport(_Artifact):
    """The structured record of one `marker analyze`.

    Carries a wall clock, so it is not byte-stable and is not required to be — the
    deterministic artifact is `work/sync-marker-analysis.json`, and nothing downstream
    consumes this one.
    """

    schema_version: Literal[1] = MARKER_REPORT_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    marker_name: str | None = None
    overall_status: OverallStatus
    analysis_status: AnalysisStatus
    #: Why the stage was skipped. Required when skipped, forbidden otherwise — the rule
    #: `StageReport` enforces, for the same reason: "skipped" with no reason is
    #: indistinguishable from a stage nobody remembered.
    skip_reason: str | None = None
    #: True when the command completed and the evidence did not support a conclusion. The
    #: distinction this report exists to keep: a quiet room is not a broken command.
    inconclusive: bool = False
    #: Counts an operator reads before opening the analysis.
    occurrences_found: int = Field(default=0, ge=0)
    groups_formed: int = Field(default=0, ge=0)
    errors: list[MarkerReportError] = Field(default_factory=list)
    warnings: list[MarkerReportWarning] = Field(default_factory=list)
    deliverables: list[ReportDeliverable] = Field(default_factory=list)
    started_at: dt.datetime
    finished_at: dt.datetime

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        for name, value in (("started_at", self.started_at), ("finished_at", self.finished_at)):
            if value.tzinfo is None or value.utcoffset() is None:
                message = f"{name} must be timezone-aware; a naive time is ambiguous"
                raise ValueError(message)
        if self.finished_at < self.started_at:
            message = "finished_at precedes started_at"
            raise ValueError(message)

        if self.analysis_status is AnalysisStatus.SKIPPED and not self.skip_reason:
            message = "a skipped analysis must say why (INV-13)"
            raise ValueError(message)
        if self.analysis_status is not AnalysisStatus.SKIPPED and self.skip_reason is not None:
            message = f"analysis is {self.analysis_status.value} but carries a skip reason"
            raise ValueError(message)
        if self.analysis_status is AnalysisStatus.FAILED and not self.errors:
            message = "a failed analysis must carry a structured error (INV-13)"
            raise ValueError(message)

        expected = _roll_up(self.analysis_status, bool(self.deliverables))
        if self.overall_status is not expected:
            message = (
                f"overall_status is {self.overall_status.value} but the analysis status "
                f"rolls up to {expected.value}"
            )
            raise ValueError(message)

        if self.inconclusive and self.analysis_status is not AnalysisStatus.COMPLETE:
            message = (
                "`inconclusive` describes a measurement that ran and settled nothing; a "
                "failed or skipped analysis measured nothing at all, which is different"
            )
            raise ValueError(message)

        object.__setattr__(
            self, "deliverables", sorted(self.deliverables, key=lambda item: item.relative_path)
        )
        return self

    def exit_code(self) -> ExitCode:
        """INV-13: partial success never exits zero.

        An inconclusive measurement exits **zero**: the command did what it was asked, and
        the answer is that the evidence does not support a conclusion. Making that nonzero
        would put a quiet room and a stale timeline in the same bucket.
        """
        if self.overall_status is OverallStatus.COMPLETE:
            return ExitCode.OK
        if self.overall_status is OverallStatus.PARTIAL:
            return ExitCode.PARTIAL
        return ExitCode.FATAL


def _roll_up(status: AnalysisStatus, produced_deliverable: bool) -> OverallStatus:
    """One stage, so the rollup is nearly an identity — stated once rather than inlined."""
    if status is AnalysisStatus.FAILED:
        return OverallStatus.PARTIAL if produced_deliverable else OverallStatus.FAILED
    return OverallStatus.COMPLETE


def write_marker_report(report: MarkerReport, path: Path) -> Path:
    """Write the report atomically, whichever way the run went. Returns its path.

    Takes no "only if successful" flag, for the reason `ReportBuilder.write` does not: a
    report that is only written on success is missing exactly when it is wanted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, report.model_dump(mode="json"))
    return path
