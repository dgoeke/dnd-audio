"""`doctor` genuinely checks the host, without touching session audio.

These run the real tools. That is the point: a mocked `ffmpeg -version` would prove
only that the mock returns a string, and the check exists to catch a host where the
flake environment is not active.

The model check is the exception, and for the opposite reason: it is pointed at a
temporary directory in every test here, so the results do not depend on whether whoever
is running the suite happens to have fetched the model — and so no test reads or writes
the invoking user's real cache.

GPU checks belong to M6a; there is deliberately no placeholder for them here.
"""

from __future__ import annotations

import shutil
import sys
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


@pytest.fixture(autouse=True)
def _empty_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No model, by default, wherever this suite is run."""
    monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path / "models"))


@pytest.fixture
def results(tmp_path: Path) -> list[CheckResult]:
    return run_checks(tmp_path)


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
        run_checks(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_directory_fails(self, tmp_path: Path) -> None:
        result = _named(run_checks(tmp_path / "absent"), "writable path")
        assert result.status is CheckStatus.FAIL
        assert "not a directory" in result.detail

    def test_a_read_only_directory_fails(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            result = _named(run_checks(locked), "writable path")
            assert result.status is CheckStatus.FAIL
        finally:
            locked.chmod(0o700)

    def test_free_space_is_reported(self, results: list[CheckResult]) -> None:
        result = _named(results, "free space")
        assert "GiB free" in result.detail

    def test_low_free_space_warns_rather_than_fails(self, tmp_path: Path) -> None:
        """Not enough disk is a problem to know about now, not a reason to refuse."""
        result = _named(run_checks(tmp_path, min_free_gib=10_000_000.0), "free space")
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

        result = _named(run_checks(tmp_path, models_directory=directory), "vad model")

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

        result = _named(run_checks(tmp_path, models_directory=directory), "vad model")

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


class TestBoundaries:
    def test_no_gpu_checks_yet(self, tmp_path: Path) -> None:
        """M6a owns those, and must test openability rather than infer it."""
        names = {result.name for result in run_checks(tmp_path)}
        assert not names & {"gpu", "torch", "/dev/kfd", "render node"}

    def test_no_asr_model_checks_yet(self, tmp_path: Path) -> None:
        """M6b owns those. Only the VAD model is fetchable today."""
        names = {result.name for result in run_checks(tmp_path)}
        assert not names & {"asr model", "qwen", "aligner"}
