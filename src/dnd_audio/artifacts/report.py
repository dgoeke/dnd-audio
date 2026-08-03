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
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.artifacts.roster import RosterSummary
from dnd_audio.determinism import sha256_file, write_json_atomic
from dnd_audio.errors import ExitCode

__all__ = [
    "REPORT_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "STAGE_ORDER",
    "Decision",
    "Deliverable",
    "IngestReport",
    "OverallStatus",
    "Provenance",
    "ReportBuilder",
    "ReportWarning",
    "RosterSummary",
    "RuntimeProvenance",
    "StageName",
    "StageOrigin",
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


class Decision(_Artifact):
    """A choice the pipeline made that a reader may need to audit.

    Deterministic by construction: no timings, no counters, nothing that varies between
    two identical runs. The spec asks for the report's decision subsection to be
    semantically stable even though the report as a whole is exempt (INV-02).

    ``subject`` is what the decision was about — a track id, a source path, a segment
    id — and is what makes two decisions with the same ``code`` orderable.
    """

    code: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    details: dict[str, str] = Field(default_factory=dict)


class Deliverable(_Artifact):
    """A file this run produced, and its hash."""

    relative_path: str
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)


class StageOrigin(StrEnum):
    """Whether a completed stage did its work or was served from cache.

    Separate from :class:`StageStatus` because they answer different questions. A stage
    served entirely from cache genuinely *completed* — its outputs are current and its
    deliverables are real — so calling it `skipped` would be false, and a caller checking
    whether the manifest reflects the sources on disk would read the wrong answer.
    """

    EXECUTED = "executed"
    #: Every unit of work was a cache hit. The outputs are current; nothing was recomputed.
    REUSED = "reused"


class StageReport(_Artifact):
    """One stage's outcome."""

    stage: StageName
    status: StageStatus
    #: Whether the work ran or came from cache. Present only on a completed stage: a
    #: skipped or failed one did not produce outputs, so "executed" would be noise on
    #: four stages out of six.
    origin: StageOrigin | None = None
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
        if self.origin is not None and self.status is not StageStatus.COMPLETE:
            message = (
                f"stage {self.stage} is {self.status} and cannot carry an origin: only a "
                f"completed stage either ran or was served from cache"
            )
            raise ValueError(message)
        return self


class RuntimeProvenance(_Artifact):
    """The compute runtime a stage resolved: which device, which precision, which build.

    Present only on a run that actually resolved one. That is deliberate rather than
    defensive: ``inspect``, ``ingest``, ``activity`` and ``mix`` never load Torch, so
    filling this in for them would mean probing a GPU those stages have no use for, and
    recording an answer no stage acted on. An absent section says "nothing here chose a
    device", which is true and is not the same as "there is no GPU".

    Every field reaches an ASR cache key (INV-08): the same audio transcribed in BF16 on
    gfx1151 and in float32 on a CPU are not the same result, and a Torch or HIP upgrade
    can change a kernel's rounding. Defining them here once means M6b adds them to
    ``TranscriberIdentity`` without a second vocabulary to drift from this one.
    """

    #: Always known — it is the interpreter running this code.
    python: str
    #: ``None`` when the resolution ran without Torch installed, which is the CPU-fallback
    #: case a machine with no ``asr-qwen`` group is in.
    torch: str | None = None
    #: ``torch.version.hip``. ``None`` on a CPU-only or CUDA build, and that distinction
    #: is load-bearing: a CUDA build here would mean the AMD index routing failed.
    hip: str | None = None
    #: The resolved device, not the requested one. ``cpu`` or ``cuda:0``.
    device: str
    #: What the driver calls the GPU. ``None`` on CPU.
    device_name: str | None = None
    #: The resolved dtype, not the requested one.
    dtype: str
    #: The attention implementation the ASR model was loaded with — ``sdpa`` today. M6a had
    #: nothing to put here because nothing in it loaded a model; M6b's adapter fills it.
    #: Output-affecting like everything else in this class: SDPA and the math fallback are
    #: not required to produce identical numbers, so it belongs in the cache key.
    #: ``None`` on a run that resolved a device without loading a model.
    attention: str | None = None


class Provenance(_Artifact):
    """Deterministic facts about the inputs and the software.

    INV-03: nothing in here may vary between two identical runs. No timestamps, no
    durations, no cache counters, no hostname. Those live in :class:`Telemetry`, and
    ``tests/test_report.py`` asserts that no field here is time-typed.
    """

    #: Ties the report to the resolved configuration (INV-08). ``None`` only when the
    #: run failed before a configuration could be resolved — a fabricated hash there
    #: would be syntactically valid and untrue, which is worse than an absence.
    config_hash: Sha256Hex | None = None
    #: External tools whose version changes the output — FFmpeg, FFprobe, SoX.
    tool_versions: dict[str, str] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    #: The exact external commands whose parameters affect an output, as the spec's
    #: observability section requires. Recorded as the invariant part of the invocation
    #: with the varying operand written as a placeholder: twelve near-identical FFprobe
    #: lines are noise, and the parameters are the thing that changes a capture.
    commands: list[str] = Field(default_factory=list)
    #: Resolved model and aligner revisions, not mutable branch names.
    model_identity: dict[str, str] = Field(default_factory=dict)
    #: The compute runtime, when a stage resolved one. ``None`` on every run that loaded
    #: no model — see :class:`RuntimeProvenance`.
    runtime: RuntimeProvenance | None = None
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
    #: Deterministic choices the run made. INV-02 requires this subsection to be
    #: semantically stable across an unchanged rerun even though the report as a whole
    #: is exempt. M1 onward fill it; the envelope and its ordering are fixed here.
    decisions: list[Decision] = Field(default_factory=list)
    #: Who was configured, who was recording, and what was found where. The spec
    #: requires the report to show this; a typed section rather than free-text
    #: decisions, so a consumer reads counts instead of parsing prose. ``None`` when no
    #: stage that discovers files ran — which is not the same as an empty roster.
    roster: RosterSummary | None = None
    telemetry: Telemetry

    @model_validator(mode="after")
    def _check_stages(self) -> Self:
        """Every stage must be accounted for, and the rollup must match them.

        INV-13 says every stage reports `complete`, `failed`, or `skipped` — which is
        only true if a stage that did not run says `skipped` rather than being absent.
        An absent stage is indistinguishable from a stage nobody thought about, and a
        report with no stages at all would otherwise roll up to `complete` and exit 0.
        """
        seen = [stage.stage for stage in self.stages]
        if len(set(seen)) != len(seen):
            message = "a stage appears more than once in the report"
            raise ValueError(message)

        missing = [stage for stage in STAGE_ORDER if stage not in set(seen)]
        if missing:
            names = ", ".join(stage.value for stage in missing)
            message = (
                f"the report accounts for no outcome of: {names}. Every stage needs a "
                f"status; a stage that did not run is `skipped` with a reason (INV-13)."
            )
            raise ValueError(message)

        ordered = sorted(self.stages, key=lambda item: STAGE_ORDER.index(item.stage))
        object.__setattr__(self, "stages", ordered)

        expected = roll_up(ordered)
        if self.overall_status is not expected:
            message = (
                f"overall_status is {self.overall_status.value} but the stages roll up "
                f"to {expected.value}"
            )
            raise ValueError(message)

        object.__setattr__(
            self, "decisions", sorted(self.decisions, key=lambda item: (item.code, item.subject))
        )
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

    Raises:
        ValueError: if handed no stages. A run that recorded nothing is not a complete
            run, and returning `complete` for an empty list would let a builder that
            forgot to record anything exit zero (INV-13).
    """
    if not stages:
        message = "cannot roll up an empty stage list: a run with no recorded stage is not complete"
        raise ValueError(message)
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

    def __init__(
        self, session_id: str, *, config_hash: str | None, started_at: dt.datetime
    ) -> None:
        self._session_id = session_id
        self._config_hash: str | None = config_hash
        self._started_at = started_at
        self._stages: dict[StageName, StageReport] = {}
        self._deliverables: dict[str, Deliverable] = {}
        self._tool_versions: dict[str, str] = {}
        # The interpreter, recorded for every run rather than by whichever stage
        # remembers. The spec's observability list asks for the Python version and it is
        # true of the whole process, not of a stage — a run that failed in `inspect`
        # should still say which Python produced that failure. Deterministic on one
        # machine, so INV-03 is satisfied: this is provenance, not telemetry.
        self._package_versions: dict[str, str] = {
            "python": ".".join(str(part) for part in sys.version_info[:3])
        }
        self._commands: list[str] = []
        self._model_identity: dict[str, str] = {}
        self._runtime: RuntimeProvenance | None = None
        self._stage_seconds: dict[StageName, float] = {}
        self._decisions: list[Decision] = []
        self._roster: RosterSummary | None = None
        self._cache_hits = 0
        self._cache_misses = 0

    def stage_complete(
        self,
        stage: StageName,
        *,
        warnings: list[ReportWarning] | None = None,
        origin: StageOrigin = StageOrigin.EXECUTED,
    ) -> None:
        self._record(
            StageReport(
                stage=stage,
                status=StageStatus.COMPLETE,
                origin=origin,
                warnings=warnings or [],
            )
        )

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

    def recorded(self, stage: StageName) -> bool:
        """Whether this stage already has an outcome.

        Exists so a caller failing partway through a multi-stage run can account for the
        stages that never got the chance to report. :meth:`build` refuses a report with a
        gap in it, so without this a failure early in `ingest` produced no report at all —
        the exact outcome INV-13 exists to prevent.
        """
        return stage in self._stages

    def completed(self, stage: StageName) -> bool:
        """Whether this stage finished successfully.

        Stronger than :meth:`recorded`, and the distinction is load-bearing for a run that
        fails partway through a composed pipeline: an artifact belonging to a stage that
        genuinely completed has already been hashed as a deliverable, so deleting it during
        failure cleanup would leave the report advertising the hash of a file that is gone
        (INV-13, M4's verify phase).
        """
        recorded = self._stages.get(stage)
        return recorded is not None and recorded.status is StageStatus.COMPLETE

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

    def record_command(self, command: str) -> None:
        """Record an external command's exact parameters. Recorded once, not per file."""
        if command not in self._commands:
            self._commands.append(command)

    def record_runtime(self, runtime: RuntimeProvenance) -> None:
        """Record the compute runtime a stage resolved.

        One run resolves one device and one dtype, so recording the same answer twice is
        fine and recording a *different* one is a bug. This used to say exactly that and
        then overwrite silently, which would have left M6b's report authoritative-looking
        and carrying only whichever stage recorded last — with the disagreement, the thing
        worth knowing, gone. Now it fails, because a partial merge would describe a machine
        that does not exist and a silent overwrite describes one that might not.

        Raises:
            ValueError: if a different runtime was already recorded.
        """
        if self._runtime is not None and self._runtime != runtime:
            message = (
                f"two stages resolved different compute runtimes in one run: "
                f"{self._runtime.device}/{self._runtime.dtype} then "
                f"{runtime.device}/{runtime.dtype}. One run resolves one runtime; a "
                f"disagreement is a bug, not something to overwrite."
            )
            raise ValueError(message)
        self._runtime = runtime

    def record_model_identity(self, name: str, revision: str) -> None:
        self._model_identity[name] = revision

    def record_stage_seconds(self, stage: StageName, seconds: float) -> None:
        self._stage_seconds[stage] = seconds

    def record_cache(self, *, hits: int = 0, misses: int = 0) -> None:
        self._cache_hits += hits
        self._cache_misses += misses

    def record_decision(self, decision: Decision) -> None:
        self._decisions.append(decision)

    def record_roster(self, roster: RosterSummary) -> None:
        """Record who was configured and who was recording.

        Set once, by whichever stage discovered the files. A second call replaces it
        rather than merging: two partial rosters would be worse than one complete one.
        """
        self._roster = roster

    def build(self, finished_at: dt.datetime) -> IngestReport:
        """Assemble the report.

        Raises:
            ValueError: if any stage has no recorded outcome. Filling those in
                automatically would invent a reason nobody wrote; the caller knows why
                it did not run a stage and INV-13 says the report has to say.
        """
        missing = [name for name in STAGE_ORDER if name not in self._stages]
        if missing:
            names = ", ".join(name.value for name in missing)
            message = (
                f"no outcome recorded for: {names}. Call stage_skipped() with a reason "
                f"for each stage this run did not perform (INV-13)."
            )
            raise ValueError(message)

        stages = [self._stages[name] for name in STAGE_ORDER]
        return IngestReport(
            session_id=self._session_id,
            overall_status=roll_up(stages),
            stages=stages,
            decisions=list(self._decisions),
            roster=self._roster,
            provenance=Provenance(
                config_hash=self._config_hash,
                tool_versions=dict(self._tool_versions),
                package_versions=dict(self._package_versions),
                commands=list(self._commands),
                model_identity=dict(self._model_identity),
                runtime=self._runtime,
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
