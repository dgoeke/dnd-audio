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

from dnd_audio.cli import app
from dnd_audio.errors import ExitCode

runner = CliRunner()

#: Command, and the milestone its stub names. `doctor`, `inspect`, and `ingest` are
#: absent: they are implemented, in M0, M1, and M2 respectively.
STUBS = [
    ("process", "M5"),
    ("transcribe", "M4"),
    ("mix", "M5"),
    ("render", "M4"),
]

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

    @pytest.mark.parametrize(("command", "milestone"), STUBS)
    def test_stub_names_its_milestone(
        self, session_dir: Path, command: str, milestone: str
    ) -> None:
        result = runner.invoke(app, [command, str(session_dir)])
        assert isinstance(result.exception, NotImplementedError)
        assert milestone in str(result.exception)

    def test_models_fetch_names_its_milestone(self) -> None:
        result = runner.invoke(app, ["models", "fetch"])
        assert isinstance(result.exception, NotImplementedError)
        assert "M6b" in str(result.exception)

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

    def test_a_stub_exits_with_the_not_implemented_code(self, session_dir: Path) -> None:
        """A traceback would be a bad message; a distinct exit code is a usable one."""
        completed = self._run("transcribe", str(session_dir))
        assert completed.returncode == ExitCode.NOT_IMPLEMENTED
        assert "not implemented yet" in completed.stderr
        assert "M4" in completed.stderr
        assert "Traceback" not in completed.stderr

    def test_stub_exit_code_is_distinct_from_usage_error(self, tmp_path: Path) -> None:
        usage = self._run("transcribe", str(tmp_path / "absent"))
        stub = self._run("transcribe", str(tmp_path))
        assert usage.returncode == 2
        assert stub.returncode == ExitCode.NOT_IMPLEMENTED
        assert usage.returncode != stub.returncode

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
