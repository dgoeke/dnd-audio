"""The archive operation report, and the three words for "checked" (ADR-0039).

`ingest-report.json` is shaped for the processing DAG: six named stages, a rollup, and a
refusal to build with a gap. An archive operation is not one of those stages and never will
be, and three of the five commands have no session directory to write into — `list`,
`verify` and `restore` exist precisely for the case where the local session is gone. So
this is a separate artifact, and `ingest-report.json` is left alone.

**The vocabulary is the correctness property.** An operator asking "is my archive good?"
can be answered three genuinely different ways, and merging them produces the exact failure
this milestone exists to prevent — a green display describing an archive nobody has read:

* ``committed`` — a manifest exists remotely. History.
* ``previously_verified_at_commit`` — the upload that wrote it read every object back at
  the time. History with a receipt.
* ``verified`` — **this operation** downloaded these bytes and decompressed them, just now.

`status` may report the first two and may never report the third; only
:class:`ArchiveOperation.VERIFY` and a full-readback upload can produce it. That constraint
is enforced here rather than left to each caller, because it is one rule and there are five
commands.

Exit codes and the `complete`/`failed`/`partial` vocabulary are borrowed deliberately from
:mod:`dnd_audio.artifacts.report`, so an operator reads the same words in both places and
INV-13's "partial success never exits zero" holds identically.
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
    "ARCHIVE_REPORT_SCHEMA_VERSION",
    "ArchiveObjectOutcome",
    "ArchiveOperation",
    "ArchiveReport",
    "ArchiveReportError",
    "ArchiveScope",
    "ObjectResult",
    "OperationStatus",
    "VerificationState",
    "write_report",
]

ARCHIVE_REPORT_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ArchiveOperation(StrEnum):
    """Which command produced this report."""

    UPLOAD = "upload"
    STATUS = "status"
    LIST = "list"
    VERIFY = "verify"
    RESTORE = "restore"


class VerificationState(StrEnum):
    """How much is actually known about the archive's integrity right now.

    Ordered weakest to strongest, and never collapsed. See the module docstring: this
    distinction is the single most important word in the milestone.
    """

    #: No manifest for this session exists remotely.
    ABSENT = "absent"
    #: Objects exist but no manifest does — an upload was interrupted before it committed.
    PENDING = "pending"
    #: A manifest exists. Says nothing about whether its objects are readable today.
    COMMITTED = "committed"
    #: A local operation report records that the committing upload read every object back.
    PREVIOUSLY_VERIFIED_AT_COMMIT = "previously_verified_at_commit"
    #: These bytes were downloaded and decompressed by the operation writing this report.
    VERIFIED = "verified"
    #: The remote archive and the local source set disagree.
    DIVERGENT = "divergent"


class OperationStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ArchiveObjectOutcome(StrEnum):
    """What happened to one object."""

    #: Compressed, uploaded, and read back successfully.
    UPLOADED = "uploaded"
    #: Already present with identical content, confirmed by full download.
    ALREADY_PRESENT = "already_present"
    #: Downloaded and decompressed to its original digest, by this operation.
    VERIFIED = "verified"
    #: Written to the restore destination and verified there.
    RESTORED = "restored"
    FAILED = "failed"
    #: Not attempted, because an earlier entry failed and the operation stopped.
    SKIPPED = "skipped"


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchiveReportError(_Artifact):
    """A failure a caller can branch on without parsing prose (INV-13)."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    #: The session-relative path this concerns, when it concerns one. Never an absolute
    #: local path: those name the machine, and a report is a thing people paste.
    path: str | None = None


class ObjectResult(_Artifact):
    """One object's outcome within an operation."""

    path: str = Field(min_length=1)
    outcome: ArchiveObjectOutcome
    size_bytes: int | None = Field(default=None, ge=0)
    compressed_size_bytes: int | None = Field(default=None, ge=0)
    error: ArchiveReportError | None = None

    @model_validator(mode="after")
    def _failed_results_say_why(self) -> Self:
        if self.outcome is ArchiveObjectOutcome.FAILED and self.error is None:
            message = f"{self.path} is marked failed without a structured error (INV-13)"
            raise ValueError(message)
        return self


class ArchiveScope(_Artifact):
    """Exactly what this operation covered, so a report cannot overstate itself."""

    session_id: str = Field(min_length=1)
    #: ``None`` means the whole session. A track id means only entries attributed to it —
    #: and therefore *not* unassigned files, which whole-session restore alone recovers.
    track_id: str | None = None
    #: How many entries were in scope, so "3 verified" can be read against "of 3" rather
    #: than looking complete on its own.
    entries_in_scope: int = Field(ge=0)


class ArchiveReport(_Artifact):
    """The structured record of one archive operation.

    Not byte-stable and not required to be: it carries a wall-clock time, like
    `ingest-report.json`'s telemetry and for the same reason. Nothing deterministic
    downstream consumes it.
    """

    schema_version: Literal[1] = ARCHIVE_REPORT_SCHEMA_VERSION
    operation: ArchiveOperation
    status: OperationStatus
    scope: ArchiveScope
    verification: VerificationState
    #: The committed manifest's digest, when one was read or written. This is where
    #: ADR-0003's fixed point is resolved: a manifest cannot hold its own hash, so the
    #: report that accompanies it does.
    manifest_sha256: Sha256Hex | None = None
    objects: list[ObjectResult] = Field(default_factory=list)
    errors: list[ArchiveReportError] = Field(default_factory=list)
    #: Operator-facing notes — retrieval cost, what a scope did not cover. Never a value
    #: that identifies the bucket, the endpoint, or this machine.
    notes: list[str] = Field(default_factory=list)
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

        if self.status is OperationStatus.FAILED and not self.errors:
            message = "a failed operation must carry a structured error (INV-13)"
            raise ValueError(message)

        # The rule the module docstring exists for, enforced once here rather than in five
        # command implementations. A cheap `status` that could say `verified` is precisely
        # the lie this artifact was designed to make impossible.
        if (
            self.verification is VerificationState.VERIFIED
            and self.operation is not ArchiveOperation.VERIFY
            and self.operation is not ArchiveOperation.UPLOAD
        ):
            message = (
                f"a `{self.operation.value}` operation may not report `verified`. Only a "
                f"current full download and decompression establishes that, which is "
                f"`verify` and the readback inside `upload`. Use `committed` or "
                f"`previously_verified_at_commit` (ADR-0039)."
            )
            raise ValueError(message)

        object.__setattr__(self, "objects", sorted(self.objects, key=lambda item: item.path))
        return self

    def exit_code(self) -> ExitCode:
        """INV-13: partial success never exits zero."""
        if self.status is OperationStatus.COMPLETE:
            return ExitCode.OK
        if self.status is OperationStatus.PARTIAL:
            return ExitCode.PARTIAL
        return ExitCode.FATAL


def write_report(report: ArchiveReport, path: Path) -> Path:
    """Write the report atomically, whichever way the operation went. Returns its path.

    Takes no "only if successful" flag, for `ReportBuilder.write`'s reason: a report that
    is only written on success is missing exactly when it is wanted.
    """
    write_json_atomic(path, report.model_dump(mode="json"))
    return path
