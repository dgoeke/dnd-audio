"""`doctor` genuinely checks the host, without touching session audio.

These run the real tools. That is the point: a mocked `ffmpeg -version` would prove
only that the mock returns a string, and the check exists to catch a host where the
flake environment is not active.

The model check is the exception, and for the opposite reason: it is pointed at a
temporary directory in every test here, so the results do not depend on whether whoever
is running the suite happens to have fetched the model — and so no test reads or writes
the invoking user's real cache.

**The GPU checks are the same exception, and it is not optional.** Every test here injects
a constructed :class:`~dnd_audio.runtime.RuntimeProbe`, so no default test imports Torch,
starts HIP, or opens a real device node. Letting them measure the machine would make this
file pass or fail on whether the person running it owns an AMD GPU — which is precisely
what INV-05 exists to prevent, and it would do it while looking like a thorough test.

The states the probes stand in for come from `test_runtime.py`, which owns them.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dnd_audio import doctor as doctor_module
from dnd_audio.determinism import sha256_bytes
from dnd_audio.doctor import (
    REQUIRED_TOOLS,
    CheckResult,
    CheckStatus,
    overall_status,
    run_checks,
)
from dnd_audio.models import ModelDescriptor
from dnd_audio.runtime import ROCM_ENV_VARS, DeviceNode, RuntimeProbe
from tests.test_runtime import bad_arithmetic, bf16_broken, cuda_build, working


@pytest.fixture(autouse=True)
def _empty_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No model, by default, wherever this suite is run."""
    monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path / "models"))


@pytest.fixture(autouse=True)
def _no_rocm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither variable set, by default, so the `rocm env` check does not depend on
    whether whoever ran the suite happened to be inside the project shell."""
    for name, _ in ROCM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def node(path: str, *, exists: bool = True, openable: bool = True) -> DeviceNode:
    return DeviceNode(
        path=Path(path),
        exists=exists,
        openable=openable,
        error=None if openable or not exists else "[Errno 13] Permission denied",
    )


#: The nodes a healthy target host presents.
HEALTHY_NODES = (node("/dev/kfd"), node("/dev/dri/renderD128"))


#: The default machine for this file: no GPU, no Torch. Never measured — see the module
#: docstring. Every `run_checks` call below passes a probe for that reason.
NO_GPU = RuntimeProbe(installed=False, error="No module named 'torch'")


@pytest.fixture
def results(tmp_path: Path) -> list[CheckResult]:
    return run_checks(tmp_path, probe=NO_GPU)


def _named(results: list[CheckResult], name: str) -> CheckResult:
    return next(result for result in results if result.name == name)


class TestToolChecks:
    def test_every_required_tool_is_checked(self, results: list[CheckResult]) -> None:
        checked = {result.name for result in results}
        assert {name for name, _ in REQUIRED_TOOLS} <= checked

    @pytest.mark.parametrize("tool", [name for name, _ in REQUIRED_TOOLS])
    def test_reports_a_real_version(self, results: list[CheckResult], tool: str) -> None:
        """Reads the actual version. INV-08 makes a tool upgrade cache-invalidating."""
        result = _named(results, tool)
        assert result.status is CheckStatus.OK, result.detail
        assert result.detail.strip()
        executable = shutil.which(tool)
        assert executable is not None
        assert executable in result.detail

    def test_sox_is_present(self, results: list[CheckResult]) -> None:
        """The canary for the flake shell: the target host has no system SoX."""
        assert _named(results, "sox").status is CheckStatus.OK

    def test_ffmpeg_version_looks_like_a_version(self, results: list[CheckResult]) -> None:
        assert "ffmpeg version" in _named(results, "ffmpeg").detail


class TestInterpreterCheck:
    def test_reports_the_running_interpreter(self, results: list[CheckResult]) -> None:
        result = _named(results, "python")
        assert sys.executable in result.detail

    def test_passes_on_312(self, results: list[CheckResult]) -> None:
        """If this fails, the tests are running outside the flake shell."""
        assert _named(results, "python").status is CheckStatus.OK
        assert sys.version_info[:2] == (3, 12)


class TestPathChecks:
    def test_a_writable_directory_passes(self, results: list[CheckResult]) -> None:
        assert _named(results, "writable path").status is CheckStatus.OK

    def test_leaves_nothing_behind(self, tmp_path: Path) -> None:
        run_checks(tmp_path, probe=NO_GPU)
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_directory_fails(self, tmp_path: Path) -> None:
        result = _named(run_checks(tmp_path / "absent", probe=NO_GPU), "writable path")
        assert result.status is CheckStatus.FAIL
        assert "not a directory" in result.detail

    def test_a_read_only_directory_fails(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            result = _named(run_checks(locked, probe=NO_GPU), "writable path")
            assert result.status is CheckStatus.FAIL
        finally:
            locked.chmod(0o700)

    def test_free_space_is_reported(self, results: list[CheckResult]) -> None:
        result = _named(results, "free space")
        assert "GiB free" in result.detail

    def test_low_free_space_warns_rather_than_fails(self, tmp_path: Path) -> None:
        """Not enough disk is a problem to know about now, not a reason to refuse."""
        result = _named(run_checks(tmp_path, min_free_gib=10_000_000.0, probe=NO_GPU), "free space")
        assert result.status is CheckStatus.WARN
        assert "below" in result.detail


class TestOverallStatus:
    def test_worst_status_wins(self) -> None:
        ok = CheckResult(name="a", status=CheckStatus.OK, detail="")
        warn = CheckResult(name="b", status=CheckStatus.WARN, detail="")
        fail = CheckResult(name="c", status=CheckStatus.FAIL, detail="")

        assert overall_status([ok, ok]) is CheckStatus.OK
        assert overall_status([ok, warn]) is CheckStatus.WARN
        assert overall_status([ok, warn, fail]) is CheckStatus.FAIL

    def test_a_healthy_host_passes(self, results: list[CheckResult]) -> None:
        assert overall_status(results) is not CheckStatus.FAIL, [
            (r.name, r.detail) for r in results if r.status is CheckStatus.FAIL
        ]


class TestModelCheck:
    """The spec's model-availability check, warning rather than failing."""

    def test_an_absent_model_warns_and_names_the_fix(self, results: list[CheckResult]) -> None:
        result = _named(results, "vad model")
        assert result.status is CheckStatus.WARN
        assert "dnd-audio models fetch" in result.detail

    def test_an_absent_model_does_not_condemn_the_host(self, results: list[CheckResult]) -> None:
        """Inspection, ingest, and the whole default suite run without any model."""
        assert overall_status(results) is not CheckStatus.FAIL

    def test_a_present_model_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Substituting the descriptor is the only way to test this offline.

        The real pin is 2.3 MB of weights that may not be committed, so the check is
        exercised against a stand-in of the same shape. What is under test is the
        check's logic, not Silero.
        """
        directory = tmp_path / "present"
        directory.mkdir()
        payload = b"stand-in for the ONNX graph\n"
        monkeypatch.setattr(doctor_module, "SILERO_VAD", _fake_model(payload))
        (directory / "fake_vad.onnx").write_bytes(payload)

        result = _named(run_checks(tmp_path, models_directory=directory, probe=NO_GPU), "vad model")

        assert result.status is CheckStatus.OK
        assert str(directory / "fake_vad.onnx") in result.detail

    def test_a_corrupted_model_warns_like_an_absent_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that does not verify must not be reported as available."""
        directory = tmp_path / "corrupt"
        directory.mkdir()
        payload = b"stand-in for the ONNX graph\n"
        monkeypatch.setattr(doctor_module, "SILERO_VAD", _fake_model(payload))
        (directory / "fake_vad.onnx").write_bytes(b"y" * len(payload))

        result = _named(run_checks(tmp_path, models_directory=directory, probe=NO_GPU), "vad model")

        assert result.status is CheckStatus.WARN


def _fake_model(payload: bytes) -> ModelDescriptor:
    commit = "0" * 40
    return ModelDescriptor(
        key="fake-vad",
        filename="fake_vad.onnx",
        repository="example/fake-vad",
        release="v0.0.1",
        commit=commit,
        path_in_repository="data/fake_vad.onnx",
        url=f"https://raw.githubusercontent.com/example/fake-vad/{commit}/data/fake_vad.onnx",
        size_bytes=len(payload),
        sha256=sha256_bytes(payload),
    )


class TestGpuChecks:
    """The line between "incomplete" and "broken", which is where these earn their keep.

    A warning tells an operator their machine cannot do one thing. A failure tells them
    something is wrong. Getting that backwards in either direction is how `doctor` becomes
    noise an operator learns to skip past.
    """

    def test_a_machine_with_no_gpu_only_warns(self, tmp_path: Path) -> None:
        """It can still inspect, ingest, mix, and run the whole default suite."""
        results = run_checks(tmp_path, probe=NO_GPU)
        assert overall_status(results) is not CheckStatus.FAIL
        for name in ("kfd node", "render node", "torch", "gpu"):
            assert _named(results, name).status is CheckStatus.WARN, name

    def test_the_absent_torch_message_names_the_way_to_install_it(self, tmp_path: Path) -> None:
        detail = _named(run_checks(tmp_path, probe=NO_GPU), "torch").detail
        assert "asr-qwen" in detail
        assert ".venv-rocm" in detail

    def test_a_healthy_gpu_reports_the_device_and_the_gfx_target(self, tmp_path: Path) -> None:
        probe = replace(working(), nodes=HEALTHY_NODES)
        results = run_checks(tmp_path, probe=probe)

        assert _named(results, "kfd node").status is CheckStatus.OK
        assert _named(results, "render node").status is CheckStatus.OK
        assert _named(results, "torch").status is CheckStatus.OK
        gpu = _named(results, "gpu")
        assert gpu.status is CheckStatus.OK
        assert "gfx1151" in gpu.detail
        assert "Radeon 8060S Graphics" in gpu.detail

    def test_an_unopenable_node_fails_rather_than_warns(self, tmp_path: Path) -> None:
        """The regression the spec's optional-hardening note is about: a distro default
        that tightens the compute nodes breaks GPU access with nothing else to show."""
        probe = replace(working(), nodes=(node("/dev/kfd", openable=False),))
        result = _named(run_checks(tmp_path, probe=probe), "kfd node")

        assert result.status is CheckStatus.FAIL
        assert "render" in result.detail
        assert "video" in result.detail

    def test_one_openable_render_node_is_enough(self, tmp_path: Path) -> None:
        """A second, unrelated GPU whose node is restricted must not condemn a machine
        whose compute device works. Which node backs device 0 is OQ-021."""
        probe = replace(
            working(),
            nodes=(
                node("/dev/kfd"),
                node("/dev/dri/renderD128"),
                node("/dev/dri/renderD129", openable=False),
            ),
        )
        result = _named(run_checks(tmp_path, probe=probe), "render node")

        assert result.status is CheckStatus.OK
        assert "renderD129" in result.detail

    def test_a_cuda_build_fails_and_names_the_routing(self, tmp_path: Path) -> None:
        """Torch here comes only from the gfx1151 index, so a build with no HIP version
        is a project misconfiguration rather than a merely incomplete machine."""
        result = _named(run_checks(tmp_path, probe=cuda_build()), "torch")
        assert result.status is CheckStatus.FAIL
        assert "ADR-0025" in result.detail

    def test_a_device_that_computes_wrongly_fails(self, tmp_path: Path) -> None:
        """`cuda_available` is true here. Only the arithmetic distinguishes this from a
        working GPU, which is the whole reason the probe runs it."""
        result = _named(run_checks(tmp_path, probe=bad_arithmetic()), "gpu")
        assert result.status is CheckStatus.FAIL

    def test_a_device_with_one_broken_dtype_warns_rather_than_fails(self, tmp_path: Path) -> None:
        """Degraded, not broken — the device is usable in float32."""
        result = _named(run_checks(tmp_path, probe=bf16_broken()), "gpu")
        assert result.status is CheckStatus.WARN
        assert "float32" in result.detail
        assert "bfloat16" in result.detail


class TestRequestedResolution:
    """`--device` / `--dtype`: does *this* configuration work on *this* machine."""

    def test_auto_on_a_bare_machine_reports_the_cpu_fallback(self, tmp_path: Path) -> None:
        result = _named(run_checks(tmp_path, probe=NO_GPU), "device/dtype")
        assert result.status is CheckStatus.WARN
        assert "cpu / float32" in result.detail

    def test_auto_on_a_working_gpu_reports_bf16(self, tmp_path: Path) -> None:
        result = _named(run_checks(tmp_path, probe=working()), "device/dtype")
        assert result.status is CheckStatus.OK
        assert "cuda:0 / bfloat16" in result.detail

    def test_an_explicit_cuda_request_that_cannot_be_met_fails(self, tmp_path: Path) -> None:
        """The charter's criterion, through the check an operator actually runs. `auto`
        on the same machine only warns — the difference is that they asked."""
        results = run_checks(tmp_path, probe=NO_GPU, device="cuda")
        result = _named(results, "device/dtype")

        assert result.status is CheckStatus.FAIL
        assert overall_status(results) is CheckStatus.FAIL
        assert "asr.device: auto" in result.detail

    def test_an_explicit_bf16_request_on_cpu_fails(self, tmp_path: Path) -> None:
        results = run_checks(tmp_path, probe=NO_GPU, device="cpu", dtype="bfloat16")
        assert _named(results, "device/dtype").status is CheckStatus.FAIL

    def test_an_explicit_combination_that_works_passes(self, tmp_path: Path) -> None:
        """float32 on a GPU whose BF16 is broken: the combination a single BF16 verdict
        would have refused."""
        results = run_checks(tmp_path, probe=bf16_broken(), device="cuda", dtype="float32")
        result = _named(results, "device/dtype")

        assert result.status is CheckStatus.OK
        assert "cuda:0 / float32" in result.detail

    def test_the_requested_pair_is_echoed_back(self, tmp_path: Path) -> None:
        """So the line is readable without knowing what was asked for."""
        detail = _named(run_checks(tmp_path, probe=working(), dtype="float32"), "device/dtype")
        assert "dtype: float32" in detail.detail


class TestRocmEnvCheck:
    def test_unset_variables_warn_and_name_the_shell(self, results: list[CheckResult]) -> None:
        result = _named(results, "rocm env")
        assert result.status is CheckStatus.WARN
        assert "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" in result.detail
        assert "project shell" in result.detail

    def test_both_set_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in ROCM_ENV_VARS:
            monkeypatch.setenv(name, value)
        assert _named(run_checks(tmp_path, probe=NO_GPU), "rocm env").status is CheckStatus.OK


class TestBoundaries:
    def test_no_default_check_measures_the_real_machine(self, tmp_path: Path) -> None:
        """INV-05, asserted rather than trusted: `run_checks` must accept a probe, and
        every test in this file must pass one. A default that measured would make this
        suite depend on the hardware of whoever ran it."""
        import inspect

        signature = inspect.signature(run_checks)
        assert "probe" in signature.parameters
        assert signature.parameters["probe"].default is None

        source = Path(__file__).read_text(encoding="utf-8")
        calls = [
            line for line in source.splitlines() if "run_checks(" in line and "def " not in line
        ]
        unprobed = [line for line in calls if "probe=" not in line]
        assert not unprobed, unprobed

    def test_no_asr_model_checks_yet(self, tmp_path: Path) -> None:
        """M6b owns those. Only the VAD model is fetchable today."""
        names = {result.name for result in run_checks(tmp_path, probe=NO_GPU)}
        assert not names & {"asr model", "qwen", "aligner"}
