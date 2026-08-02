"""`ingest-report.json` — what happened, and what can be trusted.

This is the artifact INV-13 is about. Three things it must get right, and they are the
reason it exists in M0 rather than being bolted on at the end:

* **Every stage has a status.** ``complete``, ``failed``, or ``skipped``, for all of
  them, so a missing deliverable is never ambiguous between "not asked for" and "broke".
* **Provenance and telemetry are separate structures, not a convention.** Provenance is
  deterministic and belongs to the inputs; telemetry is per-run and belongs to the
  machine. A wall-clock time cannot reach provenance by accident because there is no
  field there to put it in (INV-03).
* **Partial success is visible.** A failed transcript alongside a good mix rolls up to
  ``partial``, and :meth:`IngestReport.exit_code` turns that into a nonzero exit so
  automation cannot mistake it for a full run.

The report does not contain its own hash. See ADR-0003: writing it would change the
bytes it describes, and there is no fixed point.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.determinism import sha256_file, write_json_atomic
from dnd_audio.errors import ExitCode

__all__ = [
    "REPORT_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "STAGE_ORDER",
    "Deliverable",
    "IngestReport",
    "OverallStatus",
    "Provenance",
    "ReportBuilder",
    "ReportWarning",
    "StageName",
    "StageReport",
    "StageStatus",
    "StructuredError",
    "Telemetry",
]

#: Provisional. The report accretes fields in every milestone; see the package
#: docstring for the versioning policy.
REPORT_SCHEMA_VERSION: Final = 1

#: The report is written here, relative to the session's ``output/`` directory. Named
#: so :meth:`ReportBuilder.add_deliverable` can refuse to hash it (ADR-0003).
REPORT_FILENAME: Final = "ingest-report.json"

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StageName(StrEnum):
    """The executable stage DAG, in dependency order.

    Declaration order is the serialization order — see :data:`STAGE_ORDER`. Report
    finalization is not here: it is an always-run internal sink, not a stage that can
    itself be skipped.
    """

    INSPECT = "inspect"
    RECONSTRUCT = "reconstruct"
    ACTIVITY = "activity"
    TRANSCRIBE = "transcribe"
    RENDER = "render"
    MIX = "mix"


#: Stages always serialize in this order, never in the order they finished. The
#: transcript and mix branches race by design (INV-09), so completion order is exactly
#: the kind of nondeterminism INV-02 exists to keep out of artifacts.
STAGE_ORDER: Final[tuple[StageName, ...]] = tuple(StageName)


class StageStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class OverallStatus(StrEnum):
    COMPLETE = "complete"
    #: At least one stage failed and at least one produced a deliverable.
    PARTIAL = "partial"
    FAILED = "failed"


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StructuredError(_Artifact):
    """A failure a caller can branch on without parsing prose."""

    #: Stable, machine-readable, lowercase-with-underscores. Never reworded once
    #: something depends on it.
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    #: Session-relative path this concerns, when it concerns one.
    path: str | None = None
    details: dict[str, str] = Field(default_factory=dict)


class ReportWarning(_Artifact):
    """Something the operator should look at that did not stop the run."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None
    details: dict[str, str] = Field(default_factory=dict)


class Deliverable(_Artifact):
    """A file this run produced, and its hash."""

    relative_path: str
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)


class StageReport(_Artifact):
    """One stage's outcome."""

    stage: StageName
    status: StageStatus
    #: Why the stage was skipped. Required when skipped, absent otherwise — "skipped"
    #: with no reason is indistinguishable from a stage nobody remembered to run.
    skip_reason: str | None = None
    errors: list[StructuredError] = Field(default_factory=list)
    warnings: list[ReportWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.status is StageStatus.SKIPPED and not self.skip_reason:
            message = f"stage {self.stage} is skipped without a reason"
            raise ValueError(message)
        if self.status is not StageStatus.SKIPPED and self.skip_reason is not None:
            message = f"stage {self.stage} is {self.status} but carries a skip reason"
            raise ValueError(message)
        if self.status is StageStatus.FAILED and not self.errors:
            message = f"stage {self.stage} failed without a structured error"
            raise ValueError(message)
        return self


class Provenance(_Artifact):
    """Deterministic facts about the inputs and the software.

    INV-03: nothing in here may vary between two identical runs. No timestamps, no
    durations, no cache counters, no hostname. Those live in :class:`Telemetry`, and
    ``tests/test_report.py`` asserts that no field here is time-typed.
    """

    #: Ties the report to the resolved configuration (INV-08).
    config_hash: Sha256Hex
    #: External tools whose version changes the output — FFmpeg, FFprobe, SoX.
    tool_versions: dict[str, str] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    #: Resolved model and aligner revisions, not mutable branch names.
    model_identity: dict[str, str] = Field(default_factory=dict)
    #: Every deliverable this run produced, except the report itself (ADR-0003).
    deliverables: list[Deliverable] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_deliverables(self) -> Self:
        ordered = sorted(self.deliverables, key=lambda item: item.relative_path)
        object.__setattr__(self, "deliverables", ordered)
        return self


class Telemetry(_Artifact):
    """Per-run measurements. Legitimately different every time.

    The report as a whole is exempt from byte-stability precisely because of this
    section; its provenance and decisions must still be semantically stable (INV-02).
    """

    started_at: dt.datetime
    finished_at: dt.datetime
    stage_seconds: dict[StageName, float] = Field(default_factory=dict)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_aware(self) -> Self:
        for name, value in (("started_at", self.started_at), ("finished_at", self.finished_at)):
            if value.tzinfo is None or value.utcoffset() is None:
                message = f"telemetry.{name} must be timezone-aware; a naive time is ambiguous"
                raise ValueError(message)
        if self.finished_at < self.started_at:
            message = "telemetry.finished_at precedes telemetry.started_at"
            raise ValueError(message)
        return self


class IngestReport(_Artifact):
    """The structured record of one run."""

    schema_version: Literal[1] = REPORT_SCHEMA_VERSION
    session_id: str
    overall_status: OverallStatus
    stages: list[StageReport]
    provenance: Provenance
    telemetry: Telemetry

    @model_validator(mode="after")
    def _check_stages(self) -> Self:
        seen = [stage.stage for stage in self.stages]
        if len(set(seen)) != len(seen):
            message = "a stage appears more than once in the report"
            raise ValueError(message)
        ordered = sorted(self.stages, key=lambda item: STAGE_ORDER.index(item.stage))
        object.__setattr__(self, "stages", ordered)
        return self

    def exit_code(self) -> ExitCode:
        """The process exit code this outcome deserves.

        INV-13: partial success never exits zero.
        """
        if self.overall_status is OverallStatus.COMPLETE:
            return ExitCode.OK
        if self.overall_status is OverallStatus.PARTIAL:
            return ExitCode.PARTIAL
        return ExitCode.FATAL


def roll_up(stages: list[StageReport]) -> OverallStatus:
    """Reduce per-stage outcomes to one status.

    A skipped stage is not a failure — running only `mix` skips `transcribe` on
    purpose. What matters is whether anything failed, and whether anything survived.
    """
    failed = any(stage.status is StageStatus.FAILED for stage in stages)
    if not failed:
        return OverallStatus.COMPLETE
    completed = any(stage.status is StageStatus.COMPLETE for stage in stages)
    return OverallStatus.PARTIAL if completed else OverallStatus.FAILED


class ReportBuilder:
    """Accumulates stage outcomes and writes the report atomically.

    Stages are held keyed by name and emitted in :data:`STAGE_ORDER`, so a report is
    identical whether the mix branch or the transcript branch finished first.

    The report is written even when the run failed — that is the whole point of INV-13,
    and it is why :meth:`write` takes no "only if successful" flag.
    """

    def __init__(self, session_id: str, *, config_hash: str, started_at: dt.datetime) -> None:
        self._session_id = session_id
        self._config_hash = config_hash
        self._started_at = started_at
        self._stages: dict[StageName, StageReport] = {}
        self._deliverables: dict[str, Deliverable] = {}
        self._tool_versions: dict[str, str] = {}
        self._package_versions: dict[str, str] = {}
        self._model_identity: dict[str, str] = {}
        self._stage_seconds: dict[StageName, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def stage_complete(
        self, stage: StageName, *, warnings: list[ReportWarning] | None = None
    ) -> None:
        self._record(StageReport(stage=stage, status=StageStatus.COMPLETE, warnings=warnings or []))

    def stage_failed(
        self,
        stage: StageName,
        errors: list[StructuredError],
        *,
        warnings: list[ReportWarning] | None = None,
    ) -> None:
        self._record(
            StageReport(
                stage=stage,
                status=StageStatus.FAILED,
                errors=errors,
                warnings=warnings or [],
            )
        )

    def stage_skipped(self, stage: StageName, reason: str) -> None:
        self._record(StageReport(stage=stage, status=StageStatus.SKIPPED, skip_reason=reason))

    def add_deliverable(self, path: Path, *, relative_to: Path) -> Deliverable:
        """Hash a produced file and record it.

        Raises:
            ValueError: if asked to hash the report itself. A file cannot contain the
                hash of its own final bytes (ADR-0003), and a caller reaching for this
                has misunderstood what the field means.
        """
        if path.name == REPORT_FILENAME:
            message = (
                f"{REPORT_FILENAME} is not one of its own deliverables: writing its hash "
                f"would change the bytes that hash describes (ADR-0003)"
            )
            raise ValueError(message)

        relative = path.relative_to(relative_to).as_posix()
        deliverable = Deliverable(
            relative_path=relative,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        self._deliverables[relative] = deliverable
        return deliverable

    def record_tool_version(self, name: str, version: str) -> None:
        self._tool_versions[name] = version

    def record_package_version(self, name: str, version: str) -> None:
        self._package_versions[name] = version

    def record_model_identity(self, name: str, revision: str) -> None:
        self._model_identity[name] = revision

    def record_stage_seconds(self, stage: StageName, seconds: float) -> None:
        self._stage_seconds[stage] = seconds

    def record_cache(self, *, hits: int = 0, misses: int = 0) -> None:
        self._cache_hits += hits
        self._cache_misses += misses

    def build(self, finished_at: dt.datetime) -> IngestReport:
        stages = [self._stages[name] for name in STAGE_ORDER if name in self._stages]
        return IngestReport(
            session_id=self._session_id,
            overall_status=roll_up(stages),
            stages=stages,
            provenance=Provenance(
                config_hash=self._config_hash,
                tool_versions=dict(self._tool_versions),
                package_versions=dict(self._package_versions),
                model_identity=dict(self._model_identity),
                deliverables=list(self._deliverables.values()),
            ),
            telemetry=Telemetry(
                started_at=self._started_at,
                finished_at=finished_at,
                stage_seconds=dict(self._stage_seconds),
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
            ),
        )

    def write(self, path: Path, finished_at: dt.datetime) -> IngestReport:
        """Build and write the report atomically. Returns what was written."""
        report = self.build(finished_at)
        write_json_atomic(path, report.model_dump(mode="json"))
        return report

    def _record(self, stage: StageReport) -> None:
        if stage.stage in self._stages:
            message = f"stage {stage.stage} already has an outcome in this report"
            raise ValueError(message)
        self._stages[stage.stage] = stage
