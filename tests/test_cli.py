"""Every command exists, stubs fail clearly, and the installed console script works.

The `CliRunner` tests exercise the command tree. They are not sufficient on their own:
a `CliRunner` test imports the app directly, so it passes even when the build backend,
the `src/` layout, or `[project.scripts]` is wrong and `uv run dnd-audio` does not
exist. The subprocess tests cover that, and they are what the spec's user contract —
`uv run dnd-audio process /path/to/session` — actually is.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnd_audio import cli, models
from dnd_audio.cli import app
from dnd_audio.determinism import sha256_bytes
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.models import ModelDescriptor

runner = CliRunner()

CONSOLE_SCRIPT = Path(sys.executable).parent / "dnd-audio"


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    return tmp_path


class TestCommandSurface:
    def test_every_spec_command_is_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for name in ("process", "inspect", "ingest", "transcribe", "mix", "render", "doctor"):
            assert name in result.output
        assert "models" in result.output

    def test_models_fetch_is_registered(self) -> None:
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        assert "fetch" in result.output

    def test_no_command_is_a_stub_any_more(self, session_dir: Path) -> None:
        """Every command the spec names is implemented from M5 on.

        A session with no `session.yaml` is a *fatal* failure for all of them now, never a
        `NotImplementedError` — which is the distinction ADR-0005 spends a whole exit code on:
        exit 3 means "this pipeline has not built that yet" and exit 1 means "your session is
        broken". The remaining exit-3 path is a missing ASR adapter (M6b), below.
        """
        for command in ("process", "inspect", "ingest", "activity", "transcribe", "mix", "render"):
            result = runner.invoke(app, [command, str(session_dir)])
            assert not isinstance(result.exception, NotImplementedError), command

    @pytest.mark.parametrize("command", ["transcribe", "process"])
    def test_a_command_that_needs_the_absent_asr_adapter_says_which_milestone(
        self, canonical_fixture: FixtureTruth, command: str
    ) -> None:
        """The one place the `DEFERRED` shape still lives, and the reason it still lives.

        Both commands run against a *valid* session, so this is not "your session is broken"
        by any reading — it is a pipeline that has not built the adapter yet, and an operator
        who wants the audio branch runs `mix` (ADR-0005, ADR-0024).
        """
        result = runner.invoke(app, [command, str(canonical_fixture.session_dir)])
        assert isinstance(result.exception, NotImplementedError)
        assert "M6b" in str(result.exception)

    def test_models_fetch_says_the_asr_half_is_still_to_come(self) -> None:
        """It is implemented, so the stub message is gone — but the boundary is not."""
        result = runner.invoke(app, ["models", "fetch", "--help"])
        assert result.exit_code == 0
        assert "M6b" in result.output

    def test_a_missing_session_directory_is_a_usage_error(self, tmp_path: Path) -> None:
        """Click's exit code 2, which is why ExitCode does not use 2."""
        result = runner.invoke(app, ["inspect", str(tmp_path / "absent")])
        assert result.exit_code == 2

    def test_a_file_is_not_a_session_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "session.yaml"
        path.write_text("", encoding="utf-8")
        assert runner.invoke(app, ["inspect", str(path)]).exit_code == 2


class TestDoctorCommand:
    def test_runs_and_reports(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path)])
        assert result.exit_code == ExitCode.OK
        assert "python" in result.output
        assert "ffmpeg" in result.output

    def test_json_output_is_machine_readable(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path), "--json"])
        assert result.exit_code == ExitCode.OK
        payload = json.loads(result.output)
        assert payload["status"] in {"ok", "warn"}
        names = {check["name"] for check in payload["checks"]}
        assert {"python", "ffmpeg", "ffprobe", "sox", "writable path", "free space"} <= names

    def test_fails_on_an_unwritable_target(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", str(tmp_path / "absent")])
        assert result.exit_code == ExitCode.FATAL

    def test_defaults_to_the_working_directory(self) -> None:
        assert runner.invoke(app, ["doctor"]).exit_code == ExitCode.OK


class TestModelsFetchCommand:
    """The one command permitted to reach the network, exercised without reaching it.

    Both the descriptor and the downloader are substituted: the real pin is 2.3 MB of
    weights that may not be committed, and the socket block in `conftest.py` would fail
    any test that actually fetched it (INV-05). What is under test is the command's
    behaviour — what it reports, and what it exits with — not Silero.
    """

    PAYLOAD = b"stand-in for the ONNX graph\n"

    @pytest.fixture
    def models_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        directory = tmp_path / "models"
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(directory))
        monkeypatch.setattr(cli, "SILERO_VAD", self._descriptor())
        return directory

    @classmethod
    def _descriptor(cls) -> ModelDescriptor:
        commit = "0" * 40
        return ModelDescriptor(
            key="fake-vad",
            filename="fake_vad.onnx",
            repository="example/fake-vad",
            release="v0.0.1",
            commit=commit,
            path_in_repository="data/fake_vad.onnx",
            url=f"https://raw.githubusercontent.com/example/fake-vad/{commit}/data/fake.onnx",
            size_bytes=len(cls.PAYLOAD),
            sha256=sha256_bytes(cls.PAYLOAD),
        )

    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
        monkeypatch.setattr(models, "default_download", lambda url: payload)  # noqa: ARG005

    def test_it_fetches_and_says_where_it_landed(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._serve(monkeypatch, self.PAYLOAD)

        result = runner.invoke(app, ["models", "fetch"])

        assert result.exit_code == ExitCode.OK, result.output
        assert "fetched" in result.output
        assert str(models_dir / "fake_vad.onnx") in result.output
        assert "0" * 40 in result.output
        assert (models_dir / "fake_vad.onnx").read_bytes() == self.PAYLOAD

    def test_a_second_run_reports_it_as_already_present(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._serve(monkeypatch, self.PAYLOAD)
        assert runner.invoke(app, ["models", "fetch"]).exit_code == ExitCode.OK

        result = runner.invoke(app, ["models", "fetch"])

        assert result.exit_code == ExitCode.OK
        assert "already present" in result.output

    def test_a_verification_failure_exits_nonzero_and_writes_nothing(
        self, models_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._serve(monkeypatch, b"z" * len(self.PAYLOAD))

        result = runner.invoke(app, ["models", "fetch"])

        assert result.exit_code == ExitCode.FATAL
        assert "model_hash_mismatch" in result.output
        assert not (models_dir / "fake_vad.onnx").exists()


class TestInstalledConsoleScript:
    """The user contract is `uv run dnd-audio ...`, not `from dnd_audio.cli import app`.

    These run the entry point as a real process. Deliberately not guarded by a
    `skipif` on the script's existence: a missing console script is precisely the
    packaging failure these tests exist to catch, and skipping would hide it.

    Note that the socket block in `conftest.py` does not reach into a subprocess — the
    boundary is documented there.
    """

    def test_the_console_script_is_installed(self) -> None:
        assert CONSOLE_SCRIPT.exists(), (
            f"{CONSOLE_SCRIPT} is missing: [project.scripts], the build backend, or the "
            f"src/ layout is wrong. Run `uv sync` if the environment is simply stale."
        )

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CONSOLE_SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_help_works(self) -> None:
        completed = self._run("--help")
        assert completed.returncode == 0
        assert "dnd-audio" in completed.stdout

    def test_doctor_works(self, tmp_path: Path) -> None:
        completed = self._run("doctor", str(tmp_path), "--json")
        assert completed.returncode == 0
        assert json.loads(completed.stdout)["checks"]

    def test_an_unbuilt_adapter_exits_with_the_not_implemented_code(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """A traceback would be a bad message; a distinct exit code is a usable one."""
        completed = self._run("process", str(canonical_fixture.session_dir))
        assert completed.returncode == ExitCode.NOT_IMPLEMENTED
        assert "not implemented yet" in completed.stderr
        assert "M6b" in completed.stderr
        assert "Traceback" not in completed.stderr

    def test_the_not_implemented_code_is_distinct_from_a_usage_error(
        self, canonical_fixture: FixtureTruth, tmp_path: Path
    ) -> None:
        usage = self._run("process", str(tmp_path / "absent"))
        deferred = self._run("process", str(canonical_fixture.session_dir))
        assert usage.returncode == 2
        assert deferred.returncode == ExitCode.NOT_IMPLEMENTED
        assert usage.returncode != deferred.returncode

    def test_ingest_fails_like_an_implemented_command(self, session_dir: Path) -> None:
        """`ingest` is implemented now, so a session with no config is a fatal exit.

        The distinction matters to a caller: exit 3 means "this pipeline has not built
        that yet" and exit 1 means "your session is broken". Confusing the two sends an
        operator looking in entirely the wrong place.
        """
        completed = self._run("ingest", str(session_dir))
        assert completed.returncode == ExitCode.FATAL
        assert "not implemented yet" not in completed.stderr
        assert "Traceback" not in completed.stderr

    def test_a_session_with_no_config_fails_without_a_traceback(self, session_dir: Path) -> None:
        """`inspect` is implemented now, so its failure mode is a fatal exit and a
        report — not a stub message, and never a stack trace."""
        completed = self._run("inspect", str(session_dir))
        assert completed.returncode == ExitCode.FATAL
        assert "invalid_configuration" in completed.stderr
        assert "Traceback" not in completed.stderr
