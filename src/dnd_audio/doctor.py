"""Host checks that do not touch session audio.

`doctor` answers "is this machine able to run a session at all", which is a different
question from "did this session work". It runs before anything is ingested, so it may
not read, write, or probe anything under a session's ``raw/`` (INV-01).

Invoking ``ffmpeg -version`` is part of the job: the spec requires the report to record
exact tool versions, and INV-08 makes a tool upgrade a cache-invalidating event. Reading
a tool's version is not the "no ffprobe invocation" boundary M1 owns — that boundary is
about probing session audio.

The model-availability check is a **warning** when the model is absent, not a failure.
A host with no models can still inspect, ingest, mix, and run the entire default test
suite; the one thing it cannot do is run activity detection against the real detector.
Failing the whole check would tell an operator their machine is broken when it is
merely incomplete, and the fix is one named command away.

GPU checks — ``/dev/kfd`` and render-node openability, ``torch.cuda``, a BF16
operation — land in M6a, and the ASR models in M6b. The spec is emphatic that GPU
openability must be *tested* rather than inferred from group membership, so there is no
half-check of it here.
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

from dnd_audio.models import SILERO_VAD, find_model, models_dir

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


def run_checks(
    path: Path,
    *,
    min_free_gib: float = MIN_FREE_GIB,
    models_directory: Path | None = None,
) -> list[CheckResult]:
    """Run every non-GPU check against ``path``, in a stable order.

    ``models_directory`` overrides where models are looked for; the default is the one
    :func:`~dnd_audio.models.models_dir` resolves, which is what the CLI uses.
    """
    results = [_check_interpreter()]
    results.extend(_check_tool(name, flag) for name, flag in REQUIRED_TOOLS)
    results.append(_check_writable(path))
    results.append(_check_free_space(path, min_free_gib))
    results.append(_check_vad_model(models_directory))
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


def _check_vad_model(models_directory: Path | None) -> CheckResult:
    """Is the pinned VAD model present *and* verifying?

    :func:`~dnd_audio.models.find_model` answers the second half too, so a truncated or
    substituted file reports as absent here rather than as available-but-broken. That is
    the failure this check is worth having for: a missing file announces itself the first
    time activity runs, and a wrong one does not.
    """
    directory = models_dir() if models_directory is None else models_directory
    path = find_model(SILERO_VAD, directory=directory)
    if path is None:
        return CheckResult(
            name="vad model",
            status=CheckStatus.WARN,
            detail=(
                f"{SILERO_VAD.filename} is absent or does not match its pinned sha256 in "
                f"{directory} — run `dnd-audio models fetch`"
            ),
        )
    return CheckResult(
        name="vad model",
        status=CheckStatus.OK,
        detail=f"{SILERO_VAD.key} {SILERO_VAD.release} at {path}",
    )


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
