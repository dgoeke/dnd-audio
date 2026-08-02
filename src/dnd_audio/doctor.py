"""Host checks that do not touch session audio.

`doctor` answers "is this machine able to run a session at all", which is a different
question from "did this session work". It runs before anything is ingested, so it may
not read, write, or probe anything under a session's ``raw/`` (INV-01).

Invoking ``ffmpeg -version`` is part of the job: the spec requires the report to record
exact tool versions, and INV-08 makes a tool upgrade a cache-invalidating event. Reading
a tool's version is not the "no ffprobe invocation" boundary M1 owns — that boundary is
about probing session audio.

GPU checks — ``/dev/kfd`` and render-node openability, ``torch.cuda``, a BF16
operation — land in M6a. The spec is emphatic that openability must be *tested* rather
than inferred from group membership, so there is no half-check of it here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

__all__ = [
    "REQUIRED_TOOLS",
    "CheckResult",
    "CheckStatus",
    "overall_status",
    "run_checks",
]

#: Free space below this earns a warning. Order of magnitude, for a four-hour
#: six-transmitter session:
#:
#:     48 kHz float32 working audio   6 tracks * 14400 s * 48000 Hz * 4 B  ≈ 15.4 GiB
#:     16 kHz derivatives for VAD/ASR 6 tracks * 14400 s * 16000 Hz * 4 B  ≈  5.1 GiB
#:     lossless mono mix intermediate            14400 s * 48000 Hz * 4 B  ≈  2.6 GiB
#:
#: ~23 GiB before caches, retries, or the MP3, so a 20 GiB floor would warn only after
#: the disk was already gone. 40 GiB leaves room to finish. The real preflight belongs
#: to M2, which knows the session's actual length rather than assuming four hours.
#: The intermediate count is a guess about a pipeline that does not exist yet — OQ-013.
MIN_FREE_GIB: Final = 40.0

#: ``(executable, version-flag)``. SoX spells it differently from the FFmpeg tools, and
#: is the canary for the flake environment: the target host has no system SoX.
REQUIRED_TOOLS: Final[tuple[tuple[str, str], ...]] = (
    ("ffmpeg", "-version"),
    ("ffprobe", "-version"),
    ("sox", "--version"),
)

_SUBPROCESS_TIMEOUT_S: Final = 15.0
_BYTES_PER_GIB: Final = 1024**3


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


def run_checks(path: Path, *, min_free_gib: float = MIN_FREE_GIB) -> list[CheckResult]:
    """Run every non-GPU check against ``path``, in a stable order."""
    results = [_check_interpreter()]
    results.extend(_check_tool(name, flag) for name, flag in REQUIRED_TOOLS)
    results.append(_check_writable(path))
    results.append(_check_free_space(path, min_free_gib))
    return results


def overall_status(results: list[CheckResult]) -> CheckStatus:
    """The worst status present. A warning is not a failure; a failure is."""
    if any(result.status is CheckStatus.FAIL for result in results):
        return CheckStatus.FAIL
    if any(result.status is CheckStatus.WARN for result in results):
        return CheckStatus.WARN
    return CheckStatus.OK


def _check_interpreter() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    detail = f"{version} at {sys.executable}"
    if sys.version_info[:2] != (3, 12):
        return CheckResult(
            name="python",
            status=CheckStatus.FAIL,
            detail=f"{detail} — expected 3.12; run `direnv allow` to enter the project shell",
        )
    return CheckResult(name="python", status=CheckStatus.OK, detail=detail)


def _check_tool(name: str, version_flag: str) -> CheckResult:
    executable = shutil.which(name)
    if executable is None:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail="not on PATH — run `direnv allow` to enter the project shell",
        )

    version = _tool_version(executable, version_flag)
    if version is None:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"{executable} did not report a version",
        )
    return CheckResult(name=name, status=CheckStatus.OK, detail=f"{version} ({executable})")


def _tool_version(executable: str, version_flag: str) -> str | None:
    """First line of the tool's version output, or None if it could not be read."""
    try:
        # Fixed argv, no shell, executable already resolved through shutil.which.
        completed = subprocess.run(
            [executable, version_flag],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    # SoX writes its banner to stderr; the FFmpeg tools use stdout.
    for stream in (completed.stdout, completed.stderr):
        lines = stream.strip().splitlines()
        if lines:
            return _strip_program_prefix(lines[0].strip(), executable)
    return None


def _strip_program_prefix(line: str, executable: str) -> str:
    """Drop a leading ``<program>:`` that SoX prepends to its own banner.

    Without this the detail line names the same store path twice, which reads as a
    bug in the check rather than as a quirk of the tool.
    """
    prefix = f"{executable}:"
    if line.startswith(prefix):
        return line[len(prefix) :].strip()
    basename_prefix = f"{Path(executable).name}:"
    if line.startswith(basename_prefix):
        return line[len(basename_prefix) :].strip()
    return line


def _check_writable(path: Path) -> CheckResult:
    if not path.is_dir():
        return CheckResult(
            name="writable path", status=CheckStatus.FAIL, detail=f"{path} is not a directory"
        )
    try:
        handle_fd, name = tempfile.mkstemp(dir=path, prefix=".dnd-audio-doctor-")
    except OSError as exc:
        return CheckResult(name="writable path", status=CheckStatus.FAIL, detail=f"{path}: {exc}")

    os.close(handle_fd)
    Path(name).unlink(missing_ok=True)
    return CheckResult(name="writable path", status=CheckStatus.OK, detail=str(path))


def _check_free_space(path: Path, min_free_gib: float) -> CheckResult:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return CheckResult(name="free space", status=CheckStatus.FAIL, detail=f"{path}: {exc}")

    free_gib = usage.free / _BYTES_PER_GIB
    detail = f"{free_gib:.1f} GiB free at {path}"
    if free_gib < min_free_gib:
        return CheckResult(
            name="free space",
            status=CheckStatus.WARN,
            detail=f"{detail} — below the {min_free_gib:.0f} GiB a long session needs",
        )
    return CheckResult(name="free space", status=CheckStatus.OK, detail=detail)
