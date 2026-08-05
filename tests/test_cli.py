"""Every command exists, stubs fail clearly, and the installed console script works.

The `CliRunner` tests exercise the command tree. They are not sufficient on their own:
a `CliRunner` test imports the app directly, so it passes even when the build backend,
the `src/` layout, or `[project.scripts]` is wrong and `uv run dnd-audio` does not
exist. The subprocess tests cover that, and they are what the spec's user contract —
`uv run dnd-audio process /path/to/session` — actually is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from dnd_audio import cli, models
from dnd_audio import doctor as doctor_module
from dnd_audio.cli import app
from dnd_audio.determinism import sha256_bytes
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.models import (
    QWEN3_ALIGNER,
    QWEN3_ASR,
    QWEN_SNAPSHOTS,
    SNAPSHOT_FETCH_COMMAND,
    ModelDescriptor,
    snapshot_dir,
)
from dnd_audio.runtime import RuntimeProbe
from tests.test_runtime import shadow

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

    def test_the_archive_command_group_is_registered(self) -> None:
        """M7a's five subcommands, and the group itself on the top-level help.

        This list is the cross-check for `tests/test_archive_isolation.py`, which names
        every *non*-archive command explicitly so a new one cannot pass its
        network-boundary proof by construction. A command added here and forgotten there
        is visible as a difference between the two lists.
        """
        assert "archive" in runner.invoke(app, ["--help"]).output
        result = runner.invoke(app, ["archive", "--help"])
        assert result.exit_code == 0
        for name in ("upload", "status", "list", "verify", "restore"):
            assert name in result.output

    def test_the_archive_group_offers_no_delete_or_publish(self) -> None:
        """The operator surface matches the authority ADR-0035 grants, and no more.

        M7b owns publication and any reclamation. A subcommand appearing here before that
        milestone has justified it would be authority arriving without a decision.
        """
        # Checked against the *registered command names*, not the help text. The group's
        # own help says "never publishes or deletes", so a substring search over the output
        # finds those words and fails for the opposite of the right reason.
        from dnd_audio.cli import archive_app

        registered = {command.name for command in archive_app.registered_commands}
        assert registered == {"upload", "status", "list", "verify", "restore"}
        for forbidden in ("delete", "prune", "publish", "reclaim", "remove"):
            assert not any(forbidden in str(name) for name in registered)

    def test_an_unconfigured_machine_fails_cleanly_rather_than_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No bucket configured is an ordinary, actionable failure — not a traceback."""
        for name in list(os.environ):
            if name.startswith("DND_AUDIO_ARCHIVE_"):
                monkeypatch.delenv(name, raising=False)
        result = runner.invoke(app, ["archive", "list"])
        assert result.exit_code == ExitCode.FATAL
        assert "invalid_archive_configuration" in result.output
        assert "DND_AUDIO_ARCHIVE_BUCKET" in result.output

    def test_a_destination_outside_any_session_has_nothing_to_protect(self, tmp_path: Path) -> None:
        from dnd_audio.cli import _sessions_above

        elsewhere = tmp_path / "backups" / "recovered"
        elsewhere.mkdir(parents=True)
        assert _sessions_above(elsewhere) == []


def _tiny_session(root: Path) -> Path:
    """A session directory with one recording, enough to be a protected source root."""
    import yaml

    session = root / "session"
    (session / "raw" / "tx-a").mkdir(parents=True)
    yaml.safe_dump(
        {
            "session_id": "s",
            "title": "t",
            "tracks": [
                {
                    "track_id": "tx-a",
                    "receiver_id": "rx-a",
                    "receiver_channel": 1,
                    "speaker_id": "sp",
                    "speaker_name": "Speaker",
                    "input": "raw/tx-a",
                }
            ],
        },
        (session / "session.yaml").open("w"),
    )
    (session / "raw" / "tx-a" / "DJI_01.WAV").write_bytes(b"RIFF....WAVEirreplaceable")
    return session


class TestTheArchiveCommandsRunAndAreGuarded:
    """`_run_archive`'s body, driven through the actual commands.

    Nothing reached this function before. `tests/test_cli.py` invoked `archive --help` and
    an unconfigured `archive list`, which exits at `load_archive_config` several lines
    above everything interesting, and every other archive test called `run_upload`/
    `run_restore` directly. So the CLI's own INV-01 guard, its protected-root wiring, its
    report path and its exit codes were carried entirely by a test that asserted on a
    helper function in isolation — while a P0 sat in the block that test named.

    These drive `runner.invoke(app, ["archive", ...])` against a deterministic fake, which
    is the only place the wiring is the thing under test. Found by M7a's second review.
    """

    ENVIRONMENT: ClassVar[dict[str, str]] = {
        "DND_AUDIO_ARCHIVE_ENDPOINT_URL": "https://nyc3.digitaloceanspaces.com",
        "DND_AUDIO_ARCHIVE_REGION": "nyc3",
        "DND_AUDIO_ARCHIVE_BUCKET": "example-cold",
        "DND_AUDIO_ARCHIVE_ACCESS_KEY_ID": "DO00EXAMPLEACCESSKEY",
        "DND_AUDIO_ARCHIVE_SECRET_ACCESS_KEY": "wJalrXUtnFEMI-EXAMPLEKEY",
    }

    @pytest.fixture
    def fake_bucket(self, monkeypatch: pytest.MonkeyPatch) -> object:
        """A fake storage in place of the provider client, with the environment set.

        Patched at `dnd_audio.archive.spaces.build_storage`, which is where `_run_archive`
        imports it from — so the substitution happens at the seam the real command uses
        rather than beside it.
        """
        from dnd_audio.archive import spaces
        from dnd_audio.archive.fakes import FakeArchiveStorage

        for name, value in self.ENVIRONMENT.items():
            monkeypatch.setenv(name, value)
        bucket = FakeArchiveStorage()
        monkeypatch.setattr(spaces, "build_storage", lambda _settings: bucket)
        return bucket

    def test_verify_may_not_write_its_report_into_a_source_directory(
        self, tmp_path: Path, fake_bucket: object
    ) -> None:
        """The P0 M7a's second review found, and the reason this class exists.

        `--report` is an ordinary documented flag, and `verify` is remote-only, so it never
        has a session directory — which is exactly the condition the INV-01 guard was
        written under. Pointed at a recording, it replaced 50 bytes of WAV with 637 bytes
        of JSON: the backup tool destroying the thing it exists to protect.
        """
        session = _tiny_session(tmp_path)
        recording = session / "raw" / "tx-a" / "DJI_01.WAV"
        original = recording.read_bytes()

        result = runner.invoke(
            app,
            ["archive", "verify", "--session-id", "any", "--report", str(recording)],
        )

        assert result.exit_code == ExitCode.FATAL
        assert "INV-01" in result.output
        assert recording.read_bytes() == original, "the archive overwrote a source recording"

    def test_restore_may_not_write_its_report_into_a_source_directory(
        self, tmp_path: Path, fake_bucket: object
    ) -> None:
        """The same hole, through the other remote-only command."""
        session = _tiny_session(tmp_path)
        recording = session / "raw" / "tx-a" / "DJI_01.WAV"
        original = recording.read_bytes()
        destination = tmp_path / "recovered"
        destination.mkdir()

        result = runner.invoke(
            app,
            [
                "archive",
                "restore",
                "--session-id",
                "any",
                "--to",
                str(destination),
                "--report",
                str(recording),
            ],
        )

        assert result.exit_code == ExitCode.FATAL
        assert recording.read_bytes() == original

    def test_restore_refuses_a_destination_inside_raw_through_the_command(
        self, tmp_path: Path, fake_bucket: object
    ) -> None:
        """The first review's P0, now asserted where it actually lived.

        The previous version of this test called `_sessions_above` and never invoked the
        command, so deleting `protected_session_dirs=...` from `cli.py` left it green.
        """
        session = _tiny_session(tmp_path)
        destination = session / "raw" / "restore-here"
        destination.mkdir()

        result = runner.invoke(
            app,
            ["archive", "restore", "--session-id", "any", "--to", str(destination)],
        )

        assert result.exit_code != 0
        assert "archive_destination_protected" in result.output
        assert not any(destination.iterdir()), "a refused restore wrote into a source root"

    def test_status_runs_end_to_end_and_writes_its_report(
        self, tmp_path: Path, fake_bucket: object
    ) -> None:
        """The positive control: without it the refusals above could be refusing everything.

        A command that failed for an unrelated reason — a bad fixture, an unparsed
        argument — would satisfy every assertion in this class. This one has to succeed.
        """
        session = _tiny_session(tmp_path)
        result = runner.invoke(app, ["archive", "status", str(session)])

        assert result.exit_code == ExitCode.OK, result.output
        report = session / "work" / "archive-status-report.json"
        assert report.is_file()
        assert json.loads(report.read_bytes())["verification"] == "absent"

    def test_a_report_path_outside_every_session_is_written_where_asked(
        self, tmp_path: Path, fake_bucket: object
    ) -> None:
        """The guard refuses source directories, not `--report` as such."""
        elsewhere = tmp_path / "reports" / "verify.json"
        result = runner.invoke(
            app,
            ["archive", "verify", "--session-id", "missing", "--report", str(elsewhere)],
        )

        # The session is not in the fake bucket, so the operation fails — but it fails
        # *having written a report*, which is what INV-13 asks of it.
        assert result.exit_code != 0
        assert elsewhere.is_file()
        assert json.loads(elsewhere.read_bytes())["errors"][0]["code"] == "archive_not_committed"

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
    def test_a_command_that_cannot_find_its_asr_models_fails_cleanly(
        self, session_without_asr_models: FixtureTruth, command: str
    ) -> None:
        """M6b retires the last `DEFERRED` raise, and this is what replaces it.

        Both commands run against a *valid* session whose configured ASR revision is not
        installed. That used to be "this pipeline has not built the adapter yet"; it is now
        "this machine cannot run it", which is an ordinary failure and must behave like one:
        an exit code rather than an exception, and no traceback.
        """
        result = runner.invoke(app, [command, str(session_without_asr_models.session_dir)])

        assert not isinstance(result.exception, NotImplementedError)
        assert result.exit_code not in (0, ExitCode.NOT_IMPLEMENTED)

    def test_nothing_in_the_pipeline_defers_to_a_later_milestone_any_more(self) -> None:
        """The exit code ADR-0005 spends on "not built yet" is now unreachable, and that is
        the milestone: every command the spec names is implemented, adapter included.

        Asserted on `raise NotImplementedError` rather than on the `DEFERRED:` marker,
        because the marker also appears in prose explaining the convention and prose is not
        a placeholder. `scripts/scan_placeholders.py` pairs the two the same way.
        """
        unbuilt = [
            path
            for path in Path("src/dnd_audio").rglob("*.py")
            if "raise NotImplementedError" in path.read_text(encoding="utf-8")
        ]
        assert unbuilt == []

    def test_models_fetch_offers_the_qwen_half(self) -> None:
        """M4's version of this asserted that `fetch` said the ASR half was still to come.

        It is here now, so what replaces that assertion is the flag that installs it —
        and the fact that it is a *flag*, because the two snapshots are about six
        gigabytes and most reasons to run `models fetch` are not about them.
        """
        result = runner.invoke(app, ["models", "fetch", "--help"])
        assert result.exit_code == 0
        assert "--qwen" in result.output

    def test_models_plan_names_the_pinned_commits_and_touches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single statement of the pin, in the form the setup script reads.

        Run against an empty models directory: it must report rather than install, which
        is what makes it safe for a wrapper script to call before deciding anything.
        """
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path))
        result = runner.invoke(app, ["models", "plan", "--json"])

        assert result.exit_code == 0
        plan = json.loads(result.output)
        rows = {row["key"]: row for row in plan["models"]}
        assert rows["qwen3-asr"]["revision"] == QWEN3_ASR.revision
        assert rows["qwen3-forced-aligner"]["revision"] == QWEN3_ALIGNER.revision
        assert all(row["present"] is False for row in plan["models"])
        assert list(tmp_path.iterdir()) == []

    def test_a_configured_revision_can_actually_be_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The half of *"explicit model/aligner revisions may be set in configuration"*
        that M6b's first pass left unbuildable.

        `session.yaml` accepted `asr.model_revision` and `_default_transcriber` required a
        snapshot at exactly that commit — but `models fetch --qwen` always installed the
        revision pinned in the build and took no argument. So an operator who set one had a
        `process` that reported "revision not installed" forever and no permitted command
        able to fix it: the criterion was satisfiable in configuration and unreachable in
        practice. Found by M6b's code review.

        Asserted through `plan`, which reads the same override and reaches no network.
        """
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path))
        override = "1" * 40
        result = runner.invoke(app, ["models", "plan", "--json", "--asr-revision", override])

        assert result.exit_code == 0
        rows = {row["key"]: row for row in json.loads(result.output)["models"]}
        assert rows["qwen3-asr"]["revision"] == override
        assert rows["qwen3-asr"]["target"].endswith(override)
        # The aligner is untouched: the two are overridden independently.
        assert rows["qwen3-forced-aligner"]["revision"] == QWEN3_ALIGNER.revision

    def test_fetch_takes_the_same_overrides_plan_does(self) -> None:
        """The pair has to exist on the command that installs, not only on the one that
        reports — a plan naming a revision nothing can fetch is the defect, restated."""
        result = runner.invoke(app, ["models", "fetch", "--help"])
        assert result.exit_code == 0
        assert "--asr-revision" in result.output
        assert "--aligner-revision" in result.output

    def test_a_revision_that_is_not_a_commit_is_refused_at_the_command_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`AsrConfig`'s rule, restated for the entry point that does not read a config
        file: a branch name would install into a directory `process` never looks in."""
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path))
        result = runner.invoke(app, ["models", "plan", "--asr-revision", "main"])

        # Click's usage code — a typo, not a pipeline failure, which is why `ExitCode`
        # deliberately leaves 2 undefined.
        assert result.exit_code == 2
        assert result.exit_code not in set(ExitCode)
        assert list(tmp_path.iterdir()) == []

    def test_models_plan_agrees_with_the_descriptors_it_is_supposed_to_restate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`scripts/fetch-models.sh` consumes this instead of naming a repository or a
        commit of its own. If the two could disagree, the wrapper would be a second place
        for the pin to live — and the one that drifts is the one nobody is looking at."""
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path))
        plan = json.loads(runner.invoke(app, ["models", "plan", "--json"]).output)
        rows = {row["key"]: row for row in plan["models"]}

        for descriptor in QWEN_SNAPSHOTS:
            row = rows[descriptor.key]
            assert row["repository"] == descriptor.repository
            assert row["revision"] == descriptor.revision
            assert row["target"] == str(snapshot_dir(descriptor))

    def test_a_missing_session_directory_is_a_usage_error(self, tmp_path: Path) -> None:
        """Click's exit code 2, which is why ExitCode does not use 2."""
        result = runner.invoke(app, ["inspect", str(tmp_path / "absent")])
        assert result.exit_code == 2

    def test_a_file_is_not_a_session_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "session.yaml"
        path.write_text("", encoding="utf-8")
        assert runner.invoke(app, ["inspect", str(path)]).exit_code == 2


class TestDoctorCommand:
    """The command wiring, with the hardware measurement substituted.

    `doctor` genuinely probes the GPU — that is its job, and it is the one command allowed
    to. But these tests run it *in process*, so an unsubstituted probe would import Torch
    and launch kernels inside the default suite, which INV-05 forbids. On the project
    environment that is invisible, because there is no Torch to import; on the ROCm
    environment it is real, and it first showed up as an unrelated failure in
    `test_silero.py` that depended on run order. `conftest.py`'s `no_torch_import` fixture
    is the general guard; this is the local fix.

    What is under test here is the CLI: exit codes, output shape, argument handling. The
    checks themselves are `test_doctor.py`'s, and the real device is `host_smoke`'s.
    """

    @pytest.fixture(autouse=True)
    def _bare_machine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            doctor_module,
            "probe_runtime",
            lambda: RuntimeProbe(installed=False, error="No module named 'torch'"),
        )

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
        """The installed script really runs — with Torch shadowed out of the child.

        `doctor` probes the GPU, which is its job. But this is the default suite, and a
        subprocess is exactly where `conftest.py`'s `no_torch_import` fixture cannot look:
        it watches the parent's `sys.modules`. Unshadowed, this test launched real HIP
        kernels whenever it ran from the ROCm environment, and nothing noticed. Found by
        the verify phase's independent review; `test_runtime.py` owns the helper.
        """
        shadow(tmp_path / "shadow")
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(tmp_path / "shadow") + (os.pathsep + existing if existing else "")
        completed = subprocess.run(
            [str(CONSOLE_SCRIPT), "doctor", str(tmp_path), "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=env,
        )

        assert completed.returncode == 0, completed.stderr
        checks = {check["name"]: check for check in json.loads(completed.stdout)["checks"]}
        assert checks
        assert "not installed" in checks["torch"]["detail"], "the shadow did not take"

    def test_a_host_without_the_asr_runtime_fails_like_an_implemented_command(
        self, session_without_asr_models: FixtureTruth
    ) -> None:
        """M6b changes what this means, and the change is the milestone.

        Until now `process` with no ASR adapter exited 3 — "this pipeline has not built that
        yet" — because that was true. It is not any more. The adapter exists; what is missing
        here is the weights, which is an ordinary environment failure. So it is a *failed
        stage with a written report* and a nonzero exit, which is what INV-13 asks for and
        what ADR-0005 always reserved exit 3 against.

        Still no traceback, and the message must still say what to do about it.
        """
        completed = self._run("process", str(session_without_asr_models.session_dir))

        assert completed.returncode != 0
        assert completed.returncode != ExitCode.NOT_IMPLEMENTED
        assert "Traceback" not in completed.stderr
        assert SNAPSHOT_FETCH_COMMAND in completed.stdout + completed.stderr

    def test_that_failure_still_produces_the_mp3_and_the_report(
        self, session_without_asr_models: FixtureTruth
    ) -> None:
        """INV-09, through the installed console script rather than through an import.

        A host that cannot transcribe is exactly the case the invariant exists for. The
        audio branch must still deliver.
        """
        self._run("process", str(session_without_asr_models.session_dir))
        session = session_without_asr_models.session_dir

        assert (session / "output" / "session.mp3").is_file()
        report = json.loads((session / "output" / "ingest-report.json").read_text())
        stages = {stage["stage"]: stage["status"] for stage in report["stages"]}
        assert stages["mix"] == "complete"
        assert stages["transcribe"] == "failed"

    def test_a_usage_error_is_still_distinct_from_a_pipeline_failure(
        self, session_without_asr_models: FixtureTruth, tmp_path: Path
    ) -> None:
        usage = self._run("process", str(tmp_path / "absent"))
        failed = self._run("process", str(session_without_asr_models.session_dir))
        assert usage.returncode == 2
        assert failed.returncode not in (0, 2)

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
