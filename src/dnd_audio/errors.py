"""Errors and exit codes.

INV-13 requires fatal and recoverable to be distinguished explicitly, and requires
partial success never to exit zero. That starts here: every deliberate failure in this
project raises a subclass of :class:`DndAudioError`, and the CLI maps it to a distinct
exit code so automation can tell the cases apart without parsing text.

Unimplemented stages deliberately raise the builtin ``NotImplementedError`` instead of
a project error, annotated ``DEFERRED: M<n>`` at the raise site so that
``scripts/scan_placeholders.py`` can see them. A custom exception type would hide
placeholder work from the very check that exists to surface it.
"""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

__all__ = [
    "ConfigError",
    "DiscoveryError",
    "DndAudioError",
    "ExitCode",
    "RecoveryError",
    "TimecodeError",
]


class DndAudioError(Exception):
    """Base for every error this project raises on purpose.

    Every one carries a stable machine-readable ``code``, because INV-13 requires the
    report to hold *structured* errors: a caller has to be able to branch on what went
    wrong without matching against prose that will be reworded. The class-level default
    is the usual case; pass ``code=`` where one exception type covers several distinct
    failures a consumer would want to tell apart.
    """

    #: Lowercase-with-underscores, and never reworded once something depends on it.
    default_code: ClassVar[str] = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or self.default_code


class ConfigError(DndAudioError):
    """`session.yaml` is missing, unreadable, or does not describe a usable session."""

    default_code: ClassVar[str] = "invalid_configuration"


class DiscoveryError(DndAudioError):
    """The session's files do not satisfy a rule the operator can only fix themselves.

    A missing required track and a processed-only source are both this: nothing about
    the run can be adjusted to proceed, and proceeding anyway would mean guessing at
    what the operator meant.
    """

    default_code: ClassVar[str] = "discovery_failed"


class TimecodeError(DndAudioError):
    """A frame-rate label or timecode string is malformed or internally inconsistent.

    Raised by configuration parsing and, from M1 onward, by source-metadata parsing.
    INV-12 forbids inventing a time when this happens: it is fatal, not a fallback.
    """

    default_code: ClassVar[str] = "no_reliable_timecode"


class RecoveryError(DndAudioError):
    """A `recovery` escape hatch was configured but cannot be applied as written.

    Separate from :class:`ConfigError` because the configuration is well formed: an
    override whose hash does not match, or which names a file that was never found, is
    valid YAML describing something that is not true. Both are fatal. A silently ignored
    override is precisely the failure the recovery mechanism exists to prevent
    (ADR-0005), and an override aimed at a mistyped path is that failure with a typo in
    front of it (ADR-0007).
    """

    default_code: ClassVar[str] = "recovery_override_unusable"


class ExitCode(IntEnum):
    """Process exit codes.

    2 is deliberately absent: Click uses it for usage errors, and shadowing that would
    make a typo indistinguishable from a pipeline failure.
    """

    OK = 0
    FATAL = 1
    NOT_IMPLEMENTED = 3
    #: At least one stage failed while another produced a deliverable. INV-13: a
    #: partial run must never look like success to a caller checking the exit code.
    PARTIAL = 4
