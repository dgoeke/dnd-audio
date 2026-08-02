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

__all__ = [
    "ConfigError",
    "DndAudioError",
    "ExitCode",
    "TimecodeError",
]


class DndAudioError(Exception):
    """Base for every error this project raises on purpose."""


class ConfigError(DndAudioError):
    """`session.yaml` is missing, unreadable, or does not describe a usable session."""


class TimecodeError(DndAudioError):
    """A frame-rate label or timecode string is malformed or internally inconsistent.

    Raised by configuration parsing and, from M1 onward, by source-metadata parsing.
    INV-12 forbids inventing a time when this happens: it is fatal, not a fallback.
    """


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
