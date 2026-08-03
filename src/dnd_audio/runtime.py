"""The GPU, and the rules that turn a configured preference into a device and a dtype.

This module has two halves, and keeping them apart is the whole design (ADR-0026).

**Probing is impure and total.** :func:`probe_runtime` imports Torch, opens character
devices, and runs arithmetic on a GPU. It can only tell the truth on a machine that has
one, it never raises, and it returns a single frozen :class:`RuntimeProbe` describing
everything found.

**Resolution is pure.** :func:`resolve_runtime` is a total function of that probe. So every
rule the spec states — ``device: cuda`` requested and failing is fatal, ``auto`` falls back
to CPU with a prominent warning, ``dtype: auto`` resolves together with the *final* device,
BF16 on CPU only behind its own smoke test, and an explicitly requested combination that
fails is rejected rather than quietly downgraded — is exercised offline, over the whole
matrix, on a machine with no GPU and no Torch installed.

The alternative is a resolver reachable only through a real device, which can be tested on
exactly one host along exactly the path that host happens to take. That is the shape of
"it works on my machine" this milestone exists to avoid, and the states that matter most
are the ones no machine can be asked to enter on demand: a driver that enumerates a device
and then computes the wrong answer, or a GPU whose BF16 fails while its float32 works.

**The smoke test is per device *and* per dtype**, which is why :class:`RuntimeProbe` records
sets of dtypes rather than one verdict. A single BF16 result standing in for every
combination is wrong in both directions: it refuses an explicit ``cuda`` + ``float32``
request that would have worked, and it accepts an explicit ``float32`` request having never
tested it.

**Torch is imported lazily and only here.** The default test suite runs with no GPU, no
model weights, and no Torch at all (INV-05): the heavyweight runtime lives in the opt-in
``asr-qwen`` group, which the project environment deliberately never installs (ADR-0025).
Importing this module must therefore stay free, and ``tests/test_runtime.py`` proves it in
a subprocess over the whole CLI — the same technique :mod:`dnd_audio.activity.silero` uses
for ``onnxruntime``.

Nothing in M6a *resolves* a runtime during a pipeline run: ``mix`` and ``activity`` never
load a model. ``doctor`` was this module's only production caller until M6b; the Qwen
adapter is the second, and it is the one that resolves a runtime *during* a run and records
it in the report (`transcript/runner.py::_default_transcriber`).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from dnd_audio.artifacts.report import RuntimeProvenance
from dnd_audio.errors import DndAudioError

__all__ = [
    "KFD_NODE",
    "ROCM_ENV_VARS",
    "SMOKE_DTYPES",
    "ComputeError",
    "DeviceNode",
    "RuntimeProbe",
    "RuntimeResolution",
    "missing_rocm_env_vars",
    "open_device_nodes",
    "probe_runtime",
    "render_nodes",
    "resolve_runtime",
]

#: The ROCm compute node. Present whenever an AMD GPU is exposed for compute at all.
KFD_NODE: Final = Path("/dev/kfd")

#: ROCm compute needs a *render* node and specifically does not need the display card
#: node (`card0`, `card1`), which is why this glob is narrow: asking for the card node
#: would fail on a host whose display device is restricted while compute works fine.
_RENDER_NODE_GLOB: Final = "renderD*"
_DRI_DIR: Final = Path("/dev/dri")

#: Environment variables the dev shell sets, checked here so the documentation is
#: verifiable rather than aspirational. Both are gfx1151-specific:
#:
#: * ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` — gfx1151 is not on AOTriton's
#:   officially supported list, so its SDPA kernels are gated behind this flag. Without
#:   it Torch falls back to the *math* SDPA backend, which is correct and slow. Silent is
#:   the problem: nothing in the output says the fast path was skipped.
#: * ``HSA_ENABLE_SDMA=0`` — a stability measure, not a performance one. gfx1151's SDMA
#:   copy engines are implicated in ring timeouts and GPU resets during large transfers;
#:   with SDMA off, copies go through compute-queue blits, which on a UMA host costs
#:   approximately nothing.
#:
#: Promotion to host defaults deliberately waits for M6b's real transcription smoke test.
ROCM_ENV_VARS: Final[tuple[tuple[str, str], ...]] = (
    ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
    ("HSA_ENABLE_SDMA", "0"),
)

#: The dtypes a smoke test is run for, most-preferred first. ``_resolve_dtype``
#: iterates this for ``auto``, so adding one here changes what ``auto`` may pick.
SMOKE_DTYPES: Final[tuple[str, ...]] = ("bfloat16", "float32")

#: Operands for the smoke test, chosen so every input **and** every product is exactly
#: representable in bfloat16's 8-bit significand — the narrowest type tested. That is what
#: lets the check assert equality rather than a tolerance, and a tolerance is precisely how
#: a smoke test passes on a device that computed something plausible and wrong.
_SMOKE_LEFT: Final = (1.5, -2.25, 0.5, 8.0)
_SMOKE_RIGHT: Final = (2.0, 4.0, -8.0, 0.25)
_SMOKE_EXPECTED: Final = (3.0, -9.0, -4.0, 2.0)


class ComputeError(DndAudioError):
    """A requested device or dtype cannot be delivered on this machine.

    Not ``RuntimeError``: that is a builtin, and shadowing it inside a module about
    compute runtimes would be a genuinely confusing thing to do.

    Fatal by construction. The spec is explicit that an explicitly requested device or
    dtype which fails its smoke test is rejected rather than silently downgraded: a run
    that quietly moved from BF16 to float32 would produce different numbers than the
    operator asked for, and nothing in the output would say so.
    """

    default_code = "asr_runtime_unavailable"


@dataclass(frozen=True, slots=True)
class DeviceNode:
    """One character device, and whether this process could actually open it.

    ``openable`` is measured by opening it, never inferred from the file mode or from the
    invoking user's group membership. The spec says so in as many words, and the target
    host is why: its compute nodes are mode ``0666`` and the invoking user belongs to
    neither ``render`` nor ``video``, so a group-membership check would report no access
    on a machine where access plainly works. The reverse error is worse — a tightened
    distro default would leave a group check reporting success at the exact moment GPU
    access broke.
    """

    path: Path
    exists: bool
    openable: bool
    #: ``None`` when the node opened, or when it is simply absent.
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    """Everything measured about this machine's compute runtime, in one document.

    One record rather than several because :mod:`dnd_audio.doctor` and M6b's adapter both
    resolve from it, and two half-documents are two things to keep in step. Populated by
    :func:`probe_runtime`, which never raises: a machine with no Torch, no GPU, or a
    broken ROCm stack is a fact to report, not an exception to handle at every call site.
    """

    installed: bool
    #: The device nodes, opened. Recorded and reported, but deliberately **not** a
    #: precondition of :func:`resolve_runtime`: arithmetic that came out right on the
    #: device is strictly stronger evidence than an ``open()`` on a character device, and
    #: refusing a working GPU because a node glob came back empty would be inventing a
    #: failure (ADR-0026).
    nodes: tuple[DeviceNode, ...] = ()
    version: str | None = None
    #: ``torch.version.hip``. ``None`` on a CPU-only or CUDA build — the check that
    #: distinguishes "Torch is installed" from "the *ROCm* Torch is installed". A CUDA
    #: build reaching this project would mean the AMD index routing stopped applying,
    #: which is the silent failure ADR-0025 is mostly about.
    hip_version: str | None = None
    cuda_available: bool = False
    device_name: str | None = None
    #: ``gfx1151`` on the target host. Read from the device properties because the device
    #: *name* is a marketing string and the gfx target is not. Recorded and reported;
    #: never asserted here, because a resolver that hardcoded one architecture would be
    #: wrong on every other machine.
    gfx_target: str | None = None
    #: Dtypes that produced *exactly* the right answer on the GPU.
    gpu_dtypes: frozenset[str] = frozenset()
    #: Dtypes that produced exactly the right answer on the CPU. Only the ones that need
    #: proving are here — see :meth:`permits`.
    cpu_dtypes: frozenset[str] = frozenset()
    #: What went wrong, in the tool's own words, for the first thing that did.
    error: str | None = None

    @property
    def device_usable(self) -> bool:
        """Is this a ROCm GPU that has demonstrably computed something correctly?

        Every condition, not any of them. ``cuda_available`` alone is what a naive check
        tests, and it is true on a stack that reports a device and then fails every kernel
        launch; the arithmetic is what makes this a check rather than a claim.
        """
        return (
            self.installed
            and self.hip_version is not None
            and self.cuda_available
            and bool(self.gpu_dtypes)
        )

    def permits(self, device: Literal["cpu", "cuda:0"], dtype: str) -> bool:
        """May ``dtype`` be used on ``device``, on the evidence gathered?

        float32 on the CPU is the one combination that needs no proof: it is what every
        Python numeric stack does, it is the fallback a machine without Torch resolves to,
        and the spec asks only for a *separate CPU BF16 smoke test*. Everything else must
        have been measured.
        """
        if device == "cpu":
            return dtype == "float32" or dtype in self.cpu_dtypes
        return dtype in self.gpu_dtypes

    def unavailability(self) -> str:
        """Why the GPU is not usable, phrased for an operator to act on.

        Ordered from the outermost cause inward, so the first true statement is the one
        worth fixing. Returns the empty string when the device *is* usable.
        """
        if not self.installed:
            return (
                "torch is not installed — it lives in the opt-in `asr-qwen` group, which "
                "the project environment deliberately does not carry. Install it with "
                "`nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm "
                "uv sync --group asr-qwen'`"
            )
        if self.hip_version is None:
            return (
                f"torch {self.version} reports no HIP version, so this is a CPU-only or "
                f"CUDA build rather than the gfx1151 ROCm one — the `[tool.uv.sources]` "
                f"routing is not being applied (ADR-0025)"
            )
        if not self.cuda_available:
            detail = f": {self.error}" if self.error else ""
            blocked = [node for node in self.nodes if node.exists and not node.openable]
            if blocked:
                paths = ", ".join(str(node.path) for node in blocked)
                return (
                    f"torch {self.version} is a ROCm build (HIP {self.hip_version}) but "
                    f"reports no usable device, and {paths} could not be opened — that is "
                    f"very likely the cause"
                )
            return (
                f"torch {self.version} is a ROCm build (HIP {self.hip_version}) but "
                f"reports no usable device{detail}"
            )
        if not self.gpu_dtypes:
            detail = f": {self.error}" if self.error else ""
            return (
                f"{self.device_name or 'the device'} is present but no dtype produced the "
                f"expected result{detail} — the arithmetic is wrong, not merely absent"
            )
        return ""


@dataclass(frozen=True, slots=True)
class RuntimeResolution:
    """The device and dtype a run will actually use, and what to tell the operator.

    ``warnings`` is not decoration. A CPU fallback changes how long a session takes by
    orders of magnitude, and the spec requires it to be prominent rather than inferred
    from a slow run.
    """

    device: Literal["cpu", "cuda:0"]
    dtype: Literal["float32", "bfloat16"]
    warnings: tuple[str, ...] = ()
    probe: RuntimeProbe | None = None

    def provenance(self) -> RuntimeProvenance:
        """The subsection the report carries (INV-08: all of this reaches a cache key)."""
        probe = self.probe
        return RuntimeProvenance(
            python=".".join(str(part) for part in sys.version_info[:3]),
            torch=probe.version if probe is not None else None,
            hip=probe.hip_version if probe is not None else None,
            device=self.device,
            # Only when the GPU is the device actually being used. The probe knows the
            # name whenever the machine *has* a GPU, so copying it unconditionally made a
            # deliberate `device: cpu` run on this host record
            # `device='cpu', device_name='Radeon 8060S Graphics'` — provenance that is
            # false, and that reaches M6b's cache key under INV-08.
            device_name=probe.device_name if probe is not None and self.device != "cpu" else None,
            dtype=self.dtype,
        )


def render_nodes(directory: Path = _DRI_DIR) -> tuple[Path, ...]:
    """Every DRM render node on this machine, sorted.

    Discovered rather than hardcoded. The spec names ``/dev/dri/renderD128`` and qualifies
    it with the word *currently*: that numbering shifts with how many DRM devices the
    kernel enumerated first. On a host with a second GPU there are several and only one
    backs the compute device — that is **OQ-021**, and until it is answered every node is
    opened and one that opens is enough.
    """
    try:
        return tuple(sorted(directory.glob(_RENDER_NODE_GLOB)))
    except OSError:
        return ()


def open_device_nodes(
    kfd: Path = KFD_NODE, nodes: tuple[Path, ...] | None = None
) -> tuple[DeviceNode, ...]:
    """Open ``/dev/kfd`` and every render node, and report what happened to each."""
    candidates = (kfd, *(render_nodes() if nodes is None else nodes))
    return tuple(_open_node(path) for path in candidates)


def _open_node(path: Path) -> DeviceNode:
    """Open one node read-write and close it again immediately.

    Read-write because that is how the HSA runtime opens both of these, and a node that
    opens read-only while refusing read-write would pass a weaker check and then fail the
    first real allocation.
    """
    if not path.exists():
        return DeviceNode(path=path, exists=False, openable=False)
    try:
        descriptor = os.open(path, os.O_RDWR)
    except OSError as exc:
        return DeviceNode(path=path, exists=True, openable=False, error=str(exc))
    os.close(descriptor)
    return DeviceNode(path=path, exists=True, openable=True)


def probe_runtime() -> RuntimeProbe:
    """Measure this machine: device nodes, Torch identity, and the arithmetic itself.

    Never raises. Every failure mode — Torch absent, a CUDA build where a ROCm one was
    meant, a driver that enumerates a device and then refuses to launch a kernel — comes
    back as a populated :class:`RuntimeProbe` for :func:`resolve_runtime` to rule on.

    Slow the first time it finds a GPU: it starts HIP and runs real kernels.
    """
    nodes = open_device_nodes()

    try:
        import torch
    except Exception as exc:
        # `except Exception`, not `except ImportError`. A ROCm build with a missing or
        # mismatched shared library raises `OSError` from the dynamic loader — which is
        # exactly the environment failure this milestone exists to diagnose, and the one
        # state in which an uncaught traceback would replace the actionable diagnostic the
        # charter asks for. Found in M6a's verify phase by injecting one; it escaped.
        return RuntimeProbe(installed=False, nodes=nodes, error=f"{type(exc).__name__}: {exc}")

    version = str(torch.__version__)
    hip_version = None if torch.version.hip is None else str(torch.version.hip)
    # The CPU BF16 smoke test the spec asks for, kept separate from the GPU's on purpose:
    # CPU BF16 is a different code path that succeeds on a different set of machines — it
    # is emulated without AVX-512 BF16 or equivalent, and emulated-but-correct is a pass.
    # What is not a pass is assuming it works because the GPU's did.
    cpu_dtypes = frozenset(
        dtype for dtype in SMOKE_DTYPES if dtype != "float32" and _smoke("cpu", dtype)[0]
    )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # a broken driver raises anything at all
        return RuntimeProbe(
            installed=True,
            nodes=nodes,
            version=version,
            hip_version=hip_version,
            cpu_dtypes=cpu_dtypes,
            error=str(exc),
        )

    if not cuda_available:
        return RuntimeProbe(
            installed=True,
            nodes=nodes,
            version=version,
            hip_version=hip_version,
            cpu_dtypes=cpu_dtypes,
        )

    try:
        properties = torch.cuda.get_device_properties(0)
        device_name = str(torch.cuda.get_device_name(0))
        # `gcnArchName` is what carries `gfx1151`. Absent on a CUDA build, which is one
        # more way this probe distinguishes the two without trusting a version string.
        gfx_target = getattr(properties, "gcnArchName", None)
    except Exception as exc:  # enumeration can fail after availability says otherwise
        return RuntimeProbe(
            installed=True,
            nodes=nodes,
            version=version,
            hip_version=hip_version,
            cuda_available=True,
            cpu_dtypes=cpu_dtypes,
            error=str(exc),
        )

    results = {dtype: _smoke("cuda:0", dtype) for dtype in SMOKE_DTYPES}
    failures = [error for _, error in results.values() if error is not None]
    return RuntimeProbe(
        installed=True,
        nodes=nodes,
        version=version,
        hip_version=hip_version,
        cuda_available=True,
        device_name=device_name,
        gfx_target=None if gfx_target is None else str(gfx_target),
        gpu_dtypes=frozenset(dtype for dtype, (ok, _) in results.items() if ok),
        cpu_dtypes=cpu_dtypes,
        error=failures[0] if failures else None,
    )


def _smoke(device: str, dtype: str) -> tuple[bool, str | None]:
    """Multiply two vectors of ``dtype`` on ``device`` and check the result exactly.

    Returns ``(ok, error)``. Both operands and all four products are exact in bfloat16 —
    the narrowest type tested — so this compares for equality. A tolerance would let a
    device that returned nearly the right numbers pass, and nearly-right is the signature
    of a miscompiled or wrongly targeted kernel: the one failure this exists to catch.
    """
    try:
        import torch

        torch_dtype = getattr(torch, dtype)
        left = torch.tensor(_SMOKE_LEFT, dtype=torch_dtype, device=device)
        right = torch.tensor(_SMOKE_RIGHT, dtype=torch_dtype, device=device)
        # Back to float32 on the CPU before comparing: the comparison must not itself be
        # the thing under test, and `.cpu()` forces the synchronization that turns an
        # asynchronous kernel failure into an exception here rather than later.
        product = tuple((left * right).float().cpu().tolist())
    except Exception as exc:  # a kernel launch failure raises anything
        return False, f"{dtype} on {device}: {exc}"

    if product != _SMOKE_EXPECTED:
        return False, f"{dtype} on {device}: expected {_SMOKE_EXPECTED}, got {product}"
    return True, None


def resolve_runtime(
    *,
    device: Literal["auto", "cpu", "cuda"],
    dtype: Literal["auto", "float32", "bfloat16"],
    probe: RuntimeProbe,
) -> RuntimeResolution:
    """Turn a configured preference into the device and dtype a run will actually use.

    Pure: every argument is data, so the whole matrix is testable with no GPU present.

    Raises:
        ComputeError: when an explicitly requested device or dtype cannot be delivered.
            Never downgrades one silently — that is the spec's rule, and the reason this
            returns a resolution *or* raises rather than returning a best effort.
    """
    resolved_device, warnings = _resolve_device(device, probe)
    resolved_dtype, dtype_warnings = _resolve_dtype(dtype, resolved_device, probe)
    return RuntimeResolution(
        device=resolved_device,
        dtype=resolved_dtype,
        warnings=(*warnings, *dtype_warnings),
        probe=probe,
    )


def _resolve_device(
    requested: Literal["auto", "cpu", "cuda"], probe: RuntimeProbe
) -> tuple[Literal["cpu", "cuda:0"], list[str]]:
    if requested == "cpu":
        return "cpu", []

    if requested == "cuda":
        if not probe.device_usable:
            message = (
                f"asr.device: cuda was requested but this machine cannot provide it — "
                f"{probe.unavailability()}. Set asr.device: auto to fall back to CPU, or "
                f"asr.device: cpu to ask for CPU deliberately."
            )
            raise ComputeError(message, code="asr_device_unavailable")
        return "cuda:0", []

    if probe.device_usable:
        return "cuda:0", []
    return "cpu", [
        f"GPU inference is unavailable and asr.device is `auto`, so this run falls back "
        f"to CPU: {probe.unavailability()}. On a session-length recording that is the "
        f"difference between minutes and hours."
    ]


def _resolve_dtype(
    requested: Literal["auto", "float32", "bfloat16"],
    device: Literal["cpu", "cuda:0"],
    probe: RuntimeProbe,
) -> tuple[Literal["float32", "bfloat16"], list[str]]:
    """Resolved *together with the final device*, which is the whole point.

    Resolving dtype against the requested device rather than the resolved one is how a run
    that fell back to CPU ends up asking for BF16 on a machine that cannot do it.
    """
    if requested != "auto":
        if probe.permits(device, requested):
            return requested, []
        message = (
            f"asr.dtype: {requested} was requested and this run resolved to {device}, "
            f"where its smoke test did not succeed. Refusing rather than silently "
            f"computing in another precision: the request would have been honoured in "
            f"name only. Set asr.dtype: auto to take whatever this machine can prove."
        )
        raise ComputeError(message, code="asr_dtype_unavailable")

    if device == "cpu":
        # `auto` on CPU is float32. BF16 on CPU is something to ask for deliberately, not
        # a default to be handed over because a smoke test happened to pass.
        return "float32", []

    # Iterate the preference order rather than hardcoding it, and take only a dtype this
    # device actually proved. Hardcoding `bfloat16 else float32` gave the right answer for
    # today's two-entry SMOKE_DTYPES and would quietly return *unproven* float32 the day a
    # third dtype is added and a device passes only that one — a trap laid for M6b rather
    # than a bug today. It also made SMOKE_DTYPES' "preference order" comment a claim the
    # body did not honour.
    for candidate in SMOKE_DTYPES:
        if candidate not in probe.gpu_dtypes:
            continue
        if candidate == "bfloat16":
            return "bfloat16", []
        return candidate, [  # type: ignore[return-value]
            f"BF16 did not produce the expected result on this GPU, so `auto` resolved to "
            f"{candidate} on it rather than falling back to CPU. The device works; that "
            f"precision does not."
        ]

    # Unreachable while the device is usable, since `device_usable` requires a non-empty
    # `gpu_dtypes` drawn from SMOKE_DTYPES. Stated rather than assumed, because the thing
    # that makes it unreachable lives in another function.
    message = (
        f"asr.dtype: auto on {device}, but no dtype produced the expected result there. "
        f"Set asr.device: cpu to run without the GPU."
    )
    raise ComputeError(message, code="asr_dtype_unavailable")


def missing_rocm_env_vars() -> tuple[str, ...]:
    """Which of :data:`ROCM_ENV_VARS` are not set to their intended value.

    The dev shell sets both. This exists so ``doctor`` can say when something else is
    running the process — a bare shell, a systemd unit, an editor's integrated terminal —
    because the failure both variables prevent is silent by nature: the AOTriton one costs
    speed with no message, and the SDMA one shows up as an unexplained GPU reset under
    load rather than as anything anyone would connect to a missing variable.
    """
    return tuple(name for name, value in ROCM_ENV_VARS if os.environ.get(name) != value)
