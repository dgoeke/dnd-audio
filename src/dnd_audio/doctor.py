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

The GPU checks follow the same rule and it is worth stating where the line falls, because
"fail" here means "something is broken", not "something is missing":

* **No GPU, or no Torch → warning.** That machine can still inspect, ingest, mix, and run
  the whole default suite. It is incomplete, not broken.
* **A device node that exists but will not open → failure.** That is a real regression and
  the one the spec's optional-hardening note is about: on this host the compute nodes are
  mode ``0666`` and the invoking user is in neither ``render`` nor ``video``, so a future
  distro default that tightens them breaks GPU access with nothing else to show for it.
* **Torch that reports a device and then computes the wrong answer → failure.** So is a
  Torch with no HIP version, because in this project Torch only ever comes from the AMD
  ``gfx1151`` index: a CUDA or CPU-only build means the per-package routing stopped
  working, which is exactly the silent failure M6a exists to prevent.

Openability is tested by **opening** the nodes. The spec says so in as many words, and
inferring it from group membership would report no access on the target host, where
access plainly works. See :mod:`dnd_audio.runtime`, which owns the probing.

ASR model checks land in M6b.
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
from typing import Final, Literal

from dnd_audio.models import SILERO_VAD, find_model, models_dir
from dnd_audio.runtime import (
    KFD_NODE,
    ROCM_ENV_VARS,
    SMOKE_DTYPES,
    ComputeError,
    DeviceNode,
    RuntimeProbe,
    missing_rocm_env_vars,
    probe_runtime,
    resolve_runtime,
)

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
    probe: RuntimeProbe | None = None,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    dtype: Literal["auto", "float32", "bfloat16"] = "auto",
) -> list[CheckResult]:
    """Run every check against ``path``, in a stable order.

    ``models_directory`` overrides where models are looked for; the default is the one
    :func:`~dnd_audio.models.models_dir` resolves, which is what the CLI uses.

    ``probe`` overrides the hardware measurement. The default measures this machine, which
    is what the CLI wants and what no test should ever do — a default test suite that
    probed a real GPU would pass or fail on which machine ran it (INV-05).

    ``device`` and ``dtype`` are the requested preferences, defaulting to what an
    unconfigured `session.yaml` asks for. Passing an explicit pair is how an operator finds
    out whether *their* combination works before starting a four-hour session, rather than
    finding out during it.
    """
    measured = probe_runtime() if probe is None else probe
    results = [_check_interpreter()]
    results.extend(_check_tool(name, flag) for name, flag in REQUIRED_TOOLS)
    results.append(_check_writable(path))
    results.append(_check_free_space(path, min_free_gib))
    results.append(_check_vad_model(models_directory))
    results.append(_check_kfd(measured))
    results.append(_check_render_node(measured))
    results.append(_check_torch(measured))
    results.append(_check_gpu(measured))
    results.append(_check_resolution(measured, device=device, dtype=dtype))
    results.append(_check_rocm_env())
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


def _node_result(name: str, nodes: tuple[DeviceNode, ...], absent_detail: str) -> CheckResult:
    """Shared verdict for a group of device nodes: opened, refused, or not there.

    One node that opens is a pass, and the others are reported rather than fatal. Failing
    because a *second* GPU's node is inaccessible would condemn a machine whose compute
    device works perfectly, and nothing here can yet say which node backs that device —
    that is OQ-021. Refusal only fails when nothing opened at all.
    """
    present = [node for node in nodes if node.exists]
    if not present:
        return CheckResult(name=name, status=CheckStatus.WARN, detail=absent_detail)

    refused = [node for node in present if not node.openable]
    opened = [node for node in present if node.openable]

    if not opened:
        detail = "; ".join(f"{node.path}: {node.error}" for node in refused)
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=(
                f"{detail} — the node exists but this user cannot open it. ROCm compute "
                f"needs it; adding the user to `render` and `video`, or pinning udev "
                f"permissions declaratively, is the fix"
            ),
        )

    detail = f"opened {', '.join(str(node.path) for node in opened)}"
    if refused:
        detail += f" (but not {', '.join(str(node.path) for node in refused)})"
    return CheckResult(name=name, status=CheckStatus.OK, detail=detail)


def _check_kfd(probe: RuntimeProbe) -> CheckResult:
    nodes = tuple(node for node in probe.nodes if node.path == KFD_NODE)
    return _node_result(
        "kfd node",
        nodes,
        f"{KFD_NODE} is absent — no AMD GPU is exposed for compute on this machine",
    )


def _check_render_node(probe: RuntimeProbe) -> CheckResult:
    nodes = tuple(node for node in probe.nodes if node.path != KFD_NODE)
    return _node_result(
        "render node",
        nodes,
        "no /dev/dri/renderD* node is present — ROCm compute needs a render node "
        "(it does not need the display card node)",
    )


def _check_torch(probe: RuntimeProbe) -> CheckResult:
    if not probe.installed:
        return CheckResult(
            name="torch",
            status=CheckStatus.WARN,
            detail=(
                "not installed — it lives in the opt-in `asr-qwen` group, which the "
                "project environment deliberately does not carry (INV-05). Install it "
                "with `nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm "
                "uv sync --group asr-qwen'`"
            ),
        )
    if probe.hip_version is None:
        return CheckResult(
            name="torch",
            status=CheckStatus.FAIL,
            detail=(
                f"{probe.version} reports no HIP version, so this is a CPU-only or CUDA "
                f"build. Torch here comes only from the gfx1151 index, so this means the "
                f"`[tool.uv.sources]` routing is not being applied (ADR-0025)"
            ),
        )
    return CheckResult(
        name="torch", status=CheckStatus.OK, detail=f"{probe.version} (HIP {probe.hip_version})"
    )


def _check_gpu(probe: RuntimeProbe) -> CheckResult:
    """`torch.cuda`, the device name, the gfx target, and the arithmetic itself."""
    if not probe.installed:
        return CheckResult(
            name="gpu", status=CheckStatus.WARN, detail="not checked — torch is not installed"
        )
    if probe.hip_version is None:
        # Before the `cuda_available` branch below, and deliberately. A CUDA or CPU-only
        # build reports no device on this host, so severity decided on `cuda_available`
        # alone called a *routing failure* an incomplete machine — while printing text
        # that said the routing was broken. Same priority `_check_torch` already uses;
        # they disagreed until M6a's verify phase.
        return CheckResult(name="gpu", status=CheckStatus.FAIL, detail=probe.unavailability())
    if probe.device_usable:
        target = f", {probe.gfx_target}" if probe.gfx_target else ""
        verified = ", ".join(sorted(probe.gpu_dtypes))
        detail = f"{probe.device_name}{target} — verified {verified}"
        missing = sorted(set(SMOKE_DTYPES) - probe.gpu_dtypes)
        if missing:
            # A device that computes float32 correctly and BF16 wrongly is usable and
            # degraded, which is a different thing from broken and reads differently.
            return CheckResult(
                name="gpu",
                status=CheckStatus.WARN,
                detail=f"{detail}; {', '.join(missing)} did not: {probe.error}",
            )
        return CheckResult(name="gpu", status=CheckStatus.OK, detail=detail)
    if not probe.cuda_available:
        # No device at all is the "incomplete machine" case the module docstring
        # describes, and the node checks above have already said why in more detail.
        return CheckResult(name="gpu", status=CheckStatus.WARN, detail=probe.unavailability())
    return CheckResult(name="gpu", status=CheckStatus.FAIL, detail=probe.unavailability())


def _check_resolution(
    probe: RuntimeProbe,
    *,
    device: Literal["auto", "cpu", "cuda"],
    dtype: Literal["auto", "float32", "bfloat16"],
) -> CheckResult:
    """What the requested device and dtype actually resolve to on this machine.

    The one line an operator most wants before starting a four-hour session, and the only
    place the resolution rules run against real hardware rather than a constructed probe.
    A request that cannot be delivered raises, and that is reported as a failure here
    rather than propagating: `doctor` exists to tell someone their machine is wrong, so an
    unhandled traceback would be the check failing to do its job.
    """
    requested = f"device: {device}, dtype: {dtype}"
    try:
        resolution = resolve_runtime(device=device, dtype=dtype, probe=probe)
    except ComputeError as exc:
        return CheckResult(name="device/dtype", status=CheckStatus.FAIL, detail=str(exc))

    detail = f"{requested} resolves to {resolution.device} / {resolution.dtype}"
    if resolution.device == "cpu" and device == "auto":
        return CheckResult(
            name="device/dtype",
            status=CheckStatus.WARN,
            detail=f"{detail} — GPU inference is unavailable, so a session will be slow",
        )
    if resolution.warnings:
        # `RuntimeResolution.warnings` is documented as "not decoration"; reporting `ok`
        # over a non-empty one made this line — the one an operator most wants — say
        # nothing about a GPU that had silently lost BF16. Nothing outside this function
        # read the field at all until M6a's verify phase.
        return CheckResult(
            name="device/dtype",
            status=CheckStatus.WARN,
            detail=f"{detail} — {' '.join(resolution.warnings)}",
        )
    return CheckResult(name="device/dtype", status=CheckStatus.OK, detail=detail)


def _check_rocm_env() -> CheckResult:
    """The two gfx1151 variables the dev shell sets.

    A warning rather than a failure: both matter only once a GPU is doing real work, and
    both fail *silently* when unset, which is the entire reason this check exists rather
    than trusting a comment in `flake.nix`.
    """
    missing = missing_rocm_env_vars()
    if not missing:
        return CheckResult(
            name="rocm env",
            status=CheckStatus.OK,
            detail=", ".join(f"{name}={value}" for name, value in ROCM_ENV_VARS),
        )
    return CheckResult(
        name="rocm env",
        status=CheckStatus.WARN,
        detail=(
            f"{', '.join(missing)} not set as the dev shell sets them — SDPA falls back "
            f"to the slow math backend without the AOTriton flag, and gfx1151 transfer "
            f"stability wants SDMA off. Enter the project shell"
        ),
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
