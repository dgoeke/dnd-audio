"""The project metadata, the lock, and the ignore rules — asserted, not reviewed.

Each of these is a charter criterion that would otherwise be "checked" by someone
looking at a file. Reading `pyproject.toml` with `tomllib` and running `git
check-ignore` costs milliseconds and cannot rot.

The no-Torch assertion is the one that matters most in the long run: the charter warns
that keeping heavyweight dependencies out of the base environment is easy to get wrong
now and expensive to untangle in M6a.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest


@pytest.fixture
def pyproject(repo_root: Path) -> dict[str, object]:
    with (repo_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture
def lock(repo_root: Path) -> dict[str, object]:
    with (repo_root / "uv.lock").open("rb") as handle:
        return tomllib.load(handle)


class TestProjectMetadata:
    def test_python_is_pinned_to_312_excluding_313(self, pyproject: dict[str, object]) -> None:
        """The host's own interpreter is 3.13; the upper bound is what makes that fail."""
        project = pyproject["project"]
        assert isinstance(project, dict)
        assert project["requires-python"] == ">=3.12,<3.13"

    def test_license_matches_the_repository(self, pyproject: dict[str, object]) -> None:
        project = pyproject["project"]
        assert isinstance(project, dict)
        assert project["license"] == "Apache-2.0"

    def test_python_version_file_agrees(self, repo_root: Path) -> None:
        assert (repo_root / ".python-version").read_text(encoding="utf-8").strip() == "3.12"

    def test_console_script_is_declared(self, pyproject: dict[str, object]) -> None:
        project = pyproject["project"]
        assert isinstance(project, dict)
        assert project["scripts"] == {"dnd-audio": "dnd_audio.cli:main"}

    def test_committed_files_exist(self, repo_root: Path) -> None:
        for name in ("uv.lock", "flake.lock", "flake.nix", ".envrc", ".python-version"):
            assert (repo_root / name).is_file(), name

    def test_committed_files_are_tracked_by_git(self, repo_root: Path) -> None:
        """A lock file that exists but is not committed is not a lock file."""
        completed = subprocess.run(
            ["git", "ls-files", "--", "uv.lock", "flake.lock", ".envrc", ".python-version"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        tracked = set(completed.stdout.split())
        assert {"uv.lock", "flake.lock", ".envrc", ".python-version"} <= tracked


class TestDependencyIsolation:
    def test_asr_qwen_group_is_declared_and_empty(self, pyproject: dict[str, object]) -> None:
        """Declaring it is M0's job; filling it is M6a's (INV-05)."""
        groups = pyproject["dependency-groups"]
        assert isinstance(groups, dict)
        assert "asr-qwen" in groups
        assert groups["asr-qwen"] == []

    def test_the_lock_contains_no_torch(self, lock: dict[str, object]) -> None:
        """The failure mode M6a is designed around: accelerate's transitive
        `torch>=2.0.0` resolving a CUDA build from ordinary PyPI."""
        packages = lock["package"]
        assert isinstance(packages, list)
        names = {package["name"] for package in packages if isinstance(package, dict)}
        assert not names & {
            "torch",
            "torchaudio",
            "torchvision",
            "accelerate",
            "transformers",
            "qwen-asr",
            "nvidia-cublas-cu12",
        }

    def test_the_lock_contains_what_the_project_needs(self, lock: dict[str, object]) -> None:
        packages = lock["package"]
        assert isinstance(packages, list)
        names = {package["name"] for package in packages if isinstance(package, dict)}
        assert {"typer", "pydantic", "pyyaml", "numpy", "pytest", "ruff", "mypy"} <= names

    def test_no_amd_index_is_configured_yet(self, pyproject: dict[str, object]) -> None:
        """M6a adds it with explicit per-package sourcing; M0 must not pre-empt that."""
        tool = pyproject.get("tool", {})
        assert isinstance(tool, dict)
        assert "uv" not in tool or "index" not in tool.get("uv", {})


class TestMarkerWiring:
    """`host_smoke` and `allow_network` only work if both ends agree."""

    def test_markers_are_registered(self, pyproject: dict[str, object]) -> None:
        """`--strict-markers` turns a typo into an error instead of a silent no-op."""
        tool = pyproject["tool"]
        assert isinstance(tool, dict)
        options = tool["pytest"]["ini_options"]
        declared = {marker.split(":", 1)[0] for marker in options["markers"]}
        assert {"host_smoke", "allow_network"} <= declared
        assert "--strict-markers" in options["addopts"]

    def test_the_gate_excludes_host_smoke(self, repo_root: Path) -> None:
        """Registering the marker means nothing if the gate still runs those tests."""
        gate = (repo_root / "scripts" / "gate.sh").read_text(encoding="utf-8")
        assert "-m 'not host_smoke'" in gate

    def test_the_gate_never_invokes_nix(self, repo_root: Path) -> None:
        """On a cold store that would need the network (ADR-0002)."""
        gate = (repo_root / "scripts" / "gate.sh").read_text(encoding="utf-8")
        code = [
            line for line in gate.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not [
            line for line in code if " nix " in f" {line} " or line.strip().startswith("nix ")
        ]

    def test_the_gate_runs_tools_without_touching_an_index(self, repo_root: Path) -> None:
        """INV-05: `uv run` alone may resolve and download; `--no-sync` cannot."""
        gate = (repo_root / "scripts" / "gate.sh").read_text(encoding="utf-8")
        bare = [
            line
            for line in gate.splitlines()
            if "uv run" in line and "--no-sync" not in line and not line.lstrip().startswith("#")
        ]
        assert not bare, bare


class TestIgnoreRules:
    @pytest.mark.parametrize(
        "candidate",
        [
            "sessions/session-2026-08-15/raw/tx-a/TX01_MIC002_orig.wav",
            "sessions/session-2026-08-15/work/manifest.json",
            "sessions/session-2026-08-15/output/session.mp3",
            "recording.flac",
            "models/Qwen3-ASR-1.7B/model.safetensors",
            "weights.pt",
            "silero.onnx",
            ".hf-cache/hub/blob",
            ".env",
            "LOCAL.md",
            "docs/plan/reviews/M0-plan-20260802.raw.md",
            ".direnv/flake-profile",
            ".venv/bin/python",
        ],
    )
    def test_session_material_and_secrets_are_ignored(
        self, repo_root: Path, candidate: str
    ) -> None:
        """`--no-index` so these are judged as paths, not as files that must exist."""
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", candidate],
            cwd=repo_root,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, f"{candidate} is not ignored"

    @pytest.mark.parametrize(
        "candidate",
        [
            "src/dnd_audio/cli.py",
            "schemas/transcript.schema.json",
            "docs/plan/reviews/M0-plan-20260802.md",
            "flake.nix",
            "uv.lock",
        ],
    )
    def test_project_files_are_not_ignored(self, repo_root: Path, candidate: str) -> None:
        """An over-broad rule that swallowed the source would be just as bad."""
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", candidate],
            cwd=repo_root,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 1, f"{candidate} is unexpectedly ignored"
