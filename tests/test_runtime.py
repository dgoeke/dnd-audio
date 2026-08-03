"""Device discovery, and the rules that turn a preference into a device and a dtype.

Almost everything here runs on a machine with no GPU and no Torch, which is the point.
:func:`~dnd_audio.runtime.resolve_runtime` is a pure function of a :class:`RuntimeProbe`,
so the whole matrix — including the states no real machine can be asked to enter on
demand, like a driver that enumerates a device and then computes the wrong answer, or a
GPU whose BF16 fails while its float32 works — is exercised by constructing the probe
rather than by owning the hardware (ADR-0026).

The node checks are the exception and go the other way: they call the real :func:`os.open`
against real temporary files, because "can this process open it" is exactly the question a
mocked filesystem would stop answering. The spec is emphatic that openability is tested
rather than inferred, and a test that inferred it would be the same mistake one level up.

Two tests are marked ``host_smoke``: the arithmetic on the real device, and the real nodes.
They are the only ones that need the target host, and they run from the ROCm environment.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from dnd_audio.runtime import (
    KFD_NODE,
    ROCM_ENV_VARS,
    SMOKE_DTYPES,
    ComputeError,
    RuntimeProbe,
    missing_rocm_env_vars,
    open_device_nodes,
    probe_runtime,
    render_nodes,
    resolve_runtime,
)

# --- probes standing in for machines this one is not ---------------------------


def working() -> RuntimeProbe:
    """A healthy gfx1151: ROCm build, a device, and both dtypes exactly right."""
    return RuntimeProbe(
        installed=True,
        version="2.9.1+rocm7.13.0",
        hip_version="7.13.99004-3309c6114a",
        cuda_available=True,
        device_name="Radeon 8060S Graphics",
        gfx_target="gfx1151",
        gpu_dtypes=frozenset(SMOKE_DTYPES),
        cpu_dtypes=frozenset({"bfloat16"}),
    )


def absent() -> RuntimeProbe:
    """No `asr-qwen` group installed — the state the project environment is always in."""
    return RuntimeProbe(installed=False, error="No module named 'torch'")


def cuda_build() -> RuntimeProbe:
    """Torch resolved from ordinary PyPI. The failure ADR-0025's lock guards against."""
    return RuntimeProbe(installed=True, version="2.9.1+cu128", hip_version=None)


def no_device() -> RuntimeProbe:
    """The ROCm build, but nothing to run it on — an unopenable node, usually."""
    return RuntimeProbe(
        installed=True,
        version="2.9.1+rocm7.13.0",
        hip_version="7.13.99004-3309c6114a",
        cuda_available=False,
    )


def bad_arithmetic() -> RuntimeProbe:
    """A device that enumerates and then computes the wrong answer.

    The state that makes `torch.cuda.is_available()` insufficient on its own, and the
    reason `device_usable` requires a smoke result rather than the flag.
    """
    return RuntimeProbe(
        installed=True,
        version="2.9.1+rocm7.13.0",
        hip_version="7.13.99004-3309c6114a",
        cuda_available=True,
        device_name="Radeon 8060S Graphics",
        gfx_target="gfx1151",
        gpu_dtypes=frozenset(),
        error="bfloat16 on cuda:0: expected (3.0, -9.0, -4.0, 2.0), got (3.0, -9.0, -4.0, 0.0)",
    )


def bf16_broken() -> RuntimeProbe:
    """A usable device whose float32 is right and whose BF16 is not.

    The case a single BF16 verdict gets wrong in both directions, and the reason the probe
    records a set of dtypes per device rather than one boolean.
    """
    return RuntimeProbe(
        installed=True,
        version="2.9.1+rocm7.13.0",
        hip_version="7.13.99004-3309c6114a",
        cuda_available=True,
        device_name="Radeon 8060S Graphics",
        gfx_target="gfx1151",
        gpu_dtypes=frozenset({"float32"}),
        error="bfloat16 on cuda:0: expected (3.0, -9.0, -4.0, 2.0), got (3.0, -9.0, -4.0, 0.0)",
    )


#: Every state in which the GPU cannot be used at all.
UNUSABLE: list[Callable[[], RuntimeProbe]] = [absent, cuda_build, no_device, bad_arithmetic]


def _id(build: Callable[[], RuntimeProbe]) -> str:
    return build.__name__


# --- device nodes: opened, not inferred ----------------------------------------


class TestDeviceNodes:
    def test_a_node_that_opens_is_openable(self, tmp_path: Path) -> None:
        node = tmp_path / "kfd"
        node.write_bytes(b"")
        (result,) = open_device_nodes(kfd=node, nodes=())
        assert result.exists
        assert result.openable
        assert result.error is None

    def test_a_node_that_exists_but_refuses_is_not_openable(self, tmp_path: Path) -> None:
        """The regression the spec's optional-hardening note is about.

        On the target host the compute nodes are mode 0666 and the invoking user is in
        neither `render` nor `video`. A permission check based on group membership would
        report no access today, and — far worse — would keep reporting success on the day
        a distro default tightened the modes. Only opening the node can tell.
        """
        node = tmp_path / "kfd"
        node.write_bytes(b"")
        node.chmod(0o000)
        try:
            (result,) = open_device_nodes(kfd=node, nodes=())
            assert result.exists
            assert not result.openable
            assert result.error is not None
        finally:
            node.chmod(0o600)

    def test_an_absent_node_is_neither(self, tmp_path: Path) -> None:
        (result,) = open_device_nodes(kfd=tmp_path / "nope", nodes=())
        assert not result.exists
        assert not result.openable
        assert result.error is None

    def test_the_kfd_node_comes_first(self, tmp_path: Path) -> None:
        """Stable ordering, so `doctor`'s output does not shuffle between runs."""
        results = open_device_nodes(kfd=tmp_path / "kfd", nodes=(tmp_path / "renderD128",))
        assert [result.path.name for result in results] == ["kfd", "renderD128"]

    def test_render_nodes_are_discovered_and_the_card_node_is_not(self, tmp_path: Path) -> None:
        """Discovered rather than hardcoded, and narrow.

        `renderD128` is the current node on one machine, not a constant — hence OQ-021.
        The card node is deliberately excluded: ROCm compute does not need it, and asking
        for it would fail on a host whose display node is restricted while compute works.
        """
        for name in ("renderD128", "renderD129", "card1", "by-path"):
            (tmp_path / name).write_bytes(b"")

        assert [path.name for path in render_nodes(tmp_path)] == ["renderD128", "renderD129"]

    def test_a_missing_dri_directory_is_not_an_error(self, tmp_path: Path) -> None:
        """A machine with no DRM devices at all. Nothing to report, nothing to raise."""
        assert render_nodes(tmp_path / "absent") == ()

    def test_the_real_kfd_path_is_the_default(self) -> None:
        """Reads the constant rather than the machine, so it holds on any host."""
        assert Path("/dev/kfd") == KFD_NODE


# --- what a probe means --------------------------------------------------------


class TestProbeInterpretation:
    def test_a_working_gpu_is_usable(self) -> None:
        assert working().device_usable
        assert working().unavailability() == ""

    @pytest.mark.parametrize("build", UNUSABLE, ids=_id)
    def test_every_broken_state_is_unusable(self, build: Callable[[], RuntimeProbe]) -> None:
        assert not build().device_usable

    @pytest.mark.parametrize("build", UNUSABLE, ids=_id)
    def test_every_broken_state_explains_itself(self, build: Callable[[], RuntimeProbe]) -> None:
        """An operator has to be able to act on this without reading the source."""
        assert build().unavailability().strip()

    def test_a_device_with_one_working_dtype_is_usable(self) -> None:
        """Degraded is not broken. A GPU whose float32 is right is worth using."""
        assert bf16_broken().device_usable

    def test_a_cuda_build_is_named_as_a_routing_failure(self) -> None:
        """The one broken state that is a project misconfiguration rather than a machine.

        Torch here comes only from the gfx1151 index, so a build with no HIP version means
        `[tool.uv.sources]` stopped being applied — and it would otherwise present as a
        merely slow run.
        """
        detail = cuda_build().unavailability()
        assert "CUDA" in detail
        assert "2.9.1+cu128" in detail

    def test_availability_alone_is_not_enough(self) -> None:
        """`cuda_available` is true here and the device still cannot be used."""
        probe = bad_arithmetic()
        assert probe.cuda_available
        assert not probe.device_usable
        assert "arithmetic is wrong" in probe.unavailability()

    def test_a_blocked_node_is_offered_as_the_likely_cause(self) -> None:
        """The diagnostic that turns "no device" into something to go and fix."""
        blocked = open_device_nodes(kfd=Path("/dev/kfd"), nodes=())
        probe = RuntimeProbe(
            installed=True,
            version="2.9.1+rocm7.13.0",
            hip_version="7.13.0",
            cuda_available=False,
            nodes=tuple(
                node.__class__(path=node.path, exists=True, openable=False, error="EACCES")
                for node in blocked
            ),
        )
        assert "could not be opened" in probe.unavailability()


class TestPermits:
    """Which dtype may run where, on the evidence gathered."""

    def test_float32_on_cpu_needs_no_proof(self) -> None:
        """It is what every Python numeric stack does, and it is the fallback a machine
        without Torch resolves to. The spec asks only for a separate CPU *BF16* test."""
        assert absent().permits("cpu", "float32")

    def test_bf16_on_cpu_needs_proof(self) -> None:
        assert not absent().permits("cpu", "bfloat16")
        assert working().permits("cpu", "bfloat16")

    def test_every_gpu_dtype_needs_proof(self) -> None:
        """Including float32 — the GPU is the thing being validated, not the dtype."""
        assert not bad_arithmetic().permits("cuda:0", "float32")
        assert bf16_broken().permits("cuda:0", "float32")
        assert not bf16_broken().permits("cuda:0", "bfloat16")


# --- device resolution ---------------------------------------------------------


class TestDeviceResolution:
    def test_auto_takes_the_gpu_when_it_is_usable(self) -> None:
        resolution = resolve_runtime(device="auto", dtype="auto", probe=working())
        assert resolution.device == "cuda:0"
        assert resolution.warnings == ()

    @pytest.mark.parametrize("build", UNUSABLE, ids=_id)
    def test_auto_falls_back_to_cpu_and_says_so_prominently(
        self, build: Callable[[], RuntimeProbe]
    ) -> None:
        """The spec asks for a prominent warning, and it is prominent because it is
        structured: a caller can surface it rather than having to notice a slow run."""
        resolution = resolve_runtime(device="auto", dtype="auto", probe=build())
        assert resolution.device == "cpu"
        assert len(resolution.warnings) == 1

    @pytest.mark.parametrize("build", UNUSABLE, ids=_id)
    def test_cuda_requested_and_unavailable_is_fatal(
        self, build: Callable[[], RuntimeProbe]
    ) -> None:
        """Never a fallback. `auto` is how an operator asks for one."""
        with pytest.raises(ComputeError) as raised:
            resolve_runtime(device="cuda", dtype="auto", probe=build())
        assert raised.value.code == "asr_device_unavailable"

    def test_the_cuda_diagnostic_is_actionable(self) -> None:
        """It names what failed and what to change, which is what `actionable` means."""
        with pytest.raises(ComputeError) as raised:
            resolve_runtime(device="cuda", dtype="auto", probe=absent())
        message = str(raised.value)
        assert "asr-qwen" in message
        assert "asr.device: auto" in message

    def test_cuda_requested_and_available_is_taken(self) -> None:
        assert resolve_runtime(device="cuda", dtype="auto", probe=working()).device == "cuda:0"

    def test_cpu_requested_is_honoured_even_with_a_working_gpu(self) -> None:
        """A deliberate request is not second-guessed."""
        resolution = resolve_runtime(device="cpu", dtype="auto", probe=working())
        assert resolution.device == "cpu"
        assert resolution.warnings == ()


# --- dtype resolution ----------------------------------------------------------


class TestDtypeResolution:
    def test_auto_is_bf16_on_a_working_gpu(self) -> None:
        resolution = resolve_runtime(device="auto", dtype="auto", probe=working())
        assert (resolution.device, resolution.dtype) == ("cuda:0", "bfloat16")

    def test_auto_is_float32_after_a_cpu_fallback(self) -> None:
        """Resolved with the *final* device. Resolving it against the requested one is
        how a fallback ends up asking for BF16 on a machine that cannot do it."""
        resolution = resolve_runtime(device="auto", dtype="auto", probe=absent())
        assert (resolution.device, resolution.dtype) == ("cpu", "float32")

    def test_auto_stays_float32_on_cpu_even_when_cpu_bf16_works(self) -> None:
        """`auto` on CPU means float32. BF16 on CPU is something to ask for, not a
        default to be handed over because a smoke test happened to pass."""
        assert resolve_runtime(device="cpu", dtype="auto", probe=working()).dtype == "float32"

    def test_auto_stays_on_a_gpu_whose_bf16_is_broken(self) -> None:
        """The device works; that precision does not. Falling all the way back to CPU
        would trade a working GPU for a slow one over a dtype that has an alternative."""
        resolution = resolve_runtime(device="auto", dtype="auto", probe=bf16_broken())
        assert (resolution.device, resolution.dtype) == ("cuda:0", "float32")
        assert len(resolution.warnings) == 1

    def test_bf16_on_cpu_is_allowed_only_behind_its_own_smoke_test(self) -> None:
        probe = RuntimeProbe(installed=True, cpu_dtypes=frozenset({"bfloat16"}))
        resolution = resolve_runtime(device="cpu", dtype="bfloat16", probe=probe)
        assert (resolution.device, resolution.dtype) == ("cpu", "bfloat16")

    def test_bf16_on_cpu_without_that_smoke_test_is_rejected(self) -> None:
        """Rejected, not downgraded. A run that quietly became float32 would produce
        different numbers than were asked for and say nothing about it."""
        with pytest.raises(ComputeError) as raised:
            resolve_runtime(device="cpu", dtype="bfloat16", probe=absent())
        assert raised.value.code == "asr_dtype_unavailable"
        assert "asr.dtype: auto" in str(raised.value)

    def test_bf16_requested_after_an_auto_fallback_is_still_checked(self) -> None:
        """The combination that only exists because the device moved underneath it."""
        with pytest.raises(ComputeError):
            resolve_runtime(device="auto", dtype="bfloat16", probe=no_device())

    def test_an_explicit_float32_request_is_smoke_tested_too(self) -> None:
        """The half a single BF16 verdict gets wrong in the permissive direction: this
        device's float32 was never shown to work, so accepting it would be a claim."""
        with pytest.raises(ComputeError) as raised:
            resolve_runtime(device="cuda", dtype="float32", probe=bad_arithmetic())
        # It fails at the device, because no dtype worked at all there.
        assert raised.value.code == "asr_device_unavailable"

    def test_cuda_plus_float32_is_accepted_when_only_bf16_is_broken(self) -> None:
        """And the half it gets wrong in the restrictive direction: the requested
        combination works, and a BF16-only probe would have refused it."""
        resolution = resolve_runtime(device="cuda", dtype="float32", probe=bf16_broken())
        assert (resolution.device, resolution.dtype) == ("cuda:0", "float32")

    def test_cuda_plus_bf16_is_refused_when_bf16_is_the_broken_one(self) -> None:
        with pytest.raises(ComputeError) as raised:
            resolve_runtime(device="cuda", dtype="bfloat16", probe=bf16_broken())
        assert raised.value.code == "asr_dtype_unavailable"

    def test_float32_is_honoured_on_a_working_gpu(self) -> None:
        """Asking for less precision than the device offers is a legitimate request."""
        resolution = resolve_runtime(device="auto", dtype="float32", probe=working())
        assert (resolution.device, resolution.dtype) == ("cuda:0", "float32")


# --- what reaches the report ---------------------------------------------------


class TestProvenance:
    def test_a_gpu_resolution_records_the_whole_stack(self) -> None:
        """Every field here reaches an ASR cache key in M6b (INV-08)."""
        provenance = resolve_runtime(device="auto", dtype="auto", probe=working()).provenance()

        assert provenance.torch == "2.9.1+rocm7.13.0"
        assert provenance.hip == "7.13.99004-3309c6114a"
        assert provenance.device == "cuda:0"
        assert provenance.dtype == "bfloat16"
        assert provenance.device_name == "Radeon 8060S Graphics"

    def test_the_python_version_is_always_known(self) -> None:
        provenance = resolve_runtime(device="auto", dtype="auto", probe=absent()).provenance()
        assert provenance.python == ".".join(str(part) for part in sys.version_info[:3])

    def test_a_cpu_fallback_records_no_torch_rather_than_a_guess(self) -> None:
        provenance = resolve_runtime(device="auto", dtype="auto", probe=absent()).provenance()
        assert provenance.torch is None
        assert provenance.hip is None
        assert provenance.device == "cpu"
        assert provenance.device_name is None


# --- the dev shell's variables -------------------------------------------------


class TestRocmEnvironment:
    def test_both_variables_set_leaves_nothing_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in ROCM_ENV_VARS:
            monkeypatch.setenv(name, value)
        assert missing_rocm_env_vars() == ()

    def test_an_unset_variable_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in ROCM_ENV_VARS:
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(ROCM_ENV_VARS[0][0])
        assert missing_rocm_env_vars() == (ROCM_ENV_VARS[0][0],)

    def test_a_wrong_value_counts_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`HSA_ENABLE_SDMA=1` is not the same as unset, and it is not what the shell
        sets. Checking presence rather than value would call that configured."""
        for name, value in ROCM_ENV_VARS:
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("HSA_ENABLE_SDMA", "1")
        assert "HSA_ENABLE_SDMA" in missing_rocm_env_vars()

    def test_the_flake_sets_exactly_these(self, repo_root: Path) -> None:
        """The `doctor` check and the shell that satisfies it, kept in step.

        A variable renamed in one place and not the other would leave `doctor` warning
        forever about something the shell believes it set — which is worse than not
        checking, because the warning trains an operator to ignore it.
        """
        flake = (repo_root / "flake.nix").read_text(encoding="utf-8")
        for name, value in ROCM_ENV_VARS:
            assert f'{name} = "{value}"' in flake, name


# --- INV-05: none of this may drag Torch into the default suite -----------------


class TestLazyImport:
    def test_importing_the_cli_does_not_import_torch(self) -> None:
        """The invariant that keeps the default suite runnable without the `asr-qwen`
        group — asserted over the whole CLI, not just this module.

        `dnd_audio.cli` imports every runner, so this covers the pre-ASR commands an
        operator actually runs. Proving only that `dnd_audio.runtime` is lazy would be too
        narrow: the question is whether `ingest` or `mix` drags Torch in, not whether the
        module that owns Torch does.

        A subprocess because `sys.modules` cannot be un-rung: the `host_smoke` tests below
        load Torch, and an in-process assertion would then pass or fail on test ordering
        rather than on the code. The same technique `test_silero.py` uses for
        `onnxruntime`.
        """
        source = (
            "import sys; import dnd_audio.cli; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules "
            "if m.startswith('torch'))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr

    def test_resolution_works_in_a_process_that_has_no_torch(self) -> None:
        """The CPU-fallback path, end to end, where Torch is genuinely unavailable."""
        source = (
            "import sys; import dnd_audio.runtime as m; "
            "assert 'torch' not in sys.modules, 'imported eagerly'; "
            "probe = m.RuntimeProbe(installed=False); "
            "r = m.resolve_runtime(device='auto', dtype='auto', probe=probe); "
            "assert (r.device, r.dtype) == ('cpu', 'float32'), r"
        )
        completed = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr

    def test_probing_a_machine_reports_rather_than_raising(self) -> None:
        """Deliberately does not assert `installed is False`: on the ROCm environment
        Torch *is* importable and this must still not raise. What is under test is that a
        probe always returns a probe."""
        probe = probe_runtime()
        assert isinstance(probe, RuntimeProbe)
        if not probe.installed:
            assert probe.error is not None
            assert not probe.device_usable


# --- the real device ------------------------------------------------------------


@pytest.mark.host_smoke
def test_bf16_runs_on_the_real_gfx1151_device() -> None:
    """The charter's `host_smoke` criterion, on the target host.

    Asserts the arithmetic came out exactly right, not merely that a device exists — every
    operand and product is exact in bfloat16, so there is no tolerance to hide behind. Also
    pins the gfx target, because a wheel built for a different architecture can enumerate
    this GPU and then miscompute on it, which is what `bad_arithmetic` stands in for in the
    offline tests above. That assertion belongs here rather than in the resolver: it is a
    claim about *this* host, and a resolver that hardcoded one architecture would be wrong
    on every other.
    """
    probe = probe_runtime()

    assert probe.installed, (
        "run this from the ROCm environment: "
        "nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm uv run pytest -m host_smoke'"
    )
    assert probe.hip_version is not None, f"torch {probe.version} is not a ROCm build"
    assert probe.cuda_available, probe.unavailability()
    assert probe.gfx_target == "gfx1151", f"expected gfx1151, got {probe.gfx_target}"
    assert probe.gpu_dtypes == frozenset(SMOKE_DTYPES), (
        f"not every dtype computed correctly: {sorted(probe.gpu_dtypes)}, {probe.error}"
    )
    assert "bfloat16" in probe.cpu_dtypes, "the separate CPU BF16 smoke test did not pass"

    resolution = resolve_runtime(device="cuda", dtype="auto", probe=probe)
    assert (resolution.device, resolution.dtype) == ("cuda:0", "bfloat16")
    assert resolution.warnings == ()


@pytest.mark.host_smoke
def test_the_real_device_nodes_open_on_the_target_host() -> None:
    """Openability, against the machine rather than against a temporary file.

    `host_smoke` because a host with no AMD GPU is not a failing host — that case is a
    warning in `doctor` and is covered offline. This is the assertion that the target
    host's mode-0666 compute nodes really are usable by a user in neither `render` nor
    `video`, which is the arrangement the spec's optional-hardening note describes.
    """
    present = [node for node in open_device_nodes() if node.exists]

    assert present, "no /dev/kfd and no render node — this is not the target host"
    assert any(node.path == KFD_NODE for node in present), "no /dev/kfd"
    assert any(node.path != KFD_NODE for node in present), "no render node"
    assert all(node.openable for node in present), [
        (str(node.path), node.error) for node in present if not node.openable
    ]
