"""The project metadata, the lock, and the ignore rules — asserted, not reviewed.

Each of these is a charter criterion that would otherwise be "checked" by someone
looking at a file. Reading `pyproject.toml` with `tomllib` and running `git
check-ignore` costs milliseconds and cannot rot.

The lock assertions are the ones that matter most, and they are deliberately stated in
**both** directions: exactly these packages come from AMD's index, and everything else
comes from PyPI. The failure M6a exists to prevent is not a package that is missing — it
is a package that resolved from the wrong registry at a plausible version, silently,
because a `[tool.uv.sources]` entry did nothing. Only the second direction can see that
(ADR-0025).
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from typing import Any

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


#: Exactly what may come from AMD's index, at exactly which version. An allowlist rather
#: than a "contains torch" check, because the failure this milestone is about is a package
#: silently resolving from the *wrong* registry, which only an exact set in both directions
#: can see (ADR-0025).
AMD_INDEX = "https://repo.amd.com/rocm/whl/gfx1151/"
EXPECTED_AMD_PACKAGES = {
    "torch": "2.9.1+rocm7.13.0",
    "rocm": "7.13.0",
    "rocm-sdk-core": "7.13.0",
    "rocm-sdk-libraries-gfx1151": "7.13.0",
    "triton": "3.5.1+rocm7.13.0",
}


def _table(value: object) -> dict[str, Any]:
    """Narrow a `tomllib` value to a table.

    The fixtures are deliberately typed `dict[str, object]` so a test cannot quietly
    assume a shape it never checked; these helpers do the checking in one place.
    """
    assert isinstance(value, dict), value
    return value


def _rows(value: object) -> list[Any]:
    assert isinstance(value, list), value
    return value


def _uv(pyproject: dict[str, object]) -> dict[str, Any]:
    return _table(_table(pyproject["tool"])["uv"])


def _group(pyproject: dict[str, object], name: str) -> set[str]:
    """The distribution names a dependency group declares, without versions or extras."""
    requirements = _rows(_table(pyproject["dependency-groups"])[name])
    return {str(item).split("==")[0].split("[")[0].strip() for item in requirements}


def _packages(lock: dict[str, object]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in _rows(lock["package"]) if isinstance(entry, dict)}


def _registry(package: dict[str, Any]) -> str | None:
    source = package.get("source")
    return source.get("registry") if isinstance(source, dict) else None


def _dependency_names(package: dict[str, Any]) -> list[str]:
    return [str(entry["name"]) for entry in package.get("dependencies", [])]


class TestAmdIndex:
    """The index declaration, and the routing that only works if it is complete."""

    def test_the_index_is_declared_explicit(self, pyproject: dict[str, object]) -> None:
        """`explicit = true` is what stops this index becoming a candidate source for
        every package — it carries its own numpy, setuptools and jinja2."""
        indexes = _rows(_uv(pyproject)["index"])
        amd = [entry for entry in indexes if entry["name"] == "amd-gfx1151"]
        assert len(amd) == 1, indexes
        assert amd[0]["url"] == AMD_INDEX
        assert amd[0]["explicit"] is True

    def test_torch_is_routed_to_it_in_configuration(self, pyproject: dict[str, object]) -> None:
        """In `pyproject.toml`, never as a command-line index choice a later sync can
        forget — which is the spec's own wording."""
        sources = _table(_uv(pyproject)["sources"])
        assert sources["torch"] == {"index": "amd-gfx1151"}

    def test_every_routed_package_is_also_a_direct_dependency(
        self, pyproject: dict[str, object]
    ) -> None:
        """The regression test for the thing that cost this milestone its first attempt.

        `[tool.uv.sources]` only applies to packages that are also direct members of a
        dependency list. Routing a transitive-only requirement is **silently ignored** —
        no warning, no error, just the wrong registry recorded in the lock. So a sources
        entry without a matching group entry is not a mistake that shows up as a failure;
        it shows up as a package quietly arriving from PyPI (ADR-0025).
        """
        sources = _table(_uv(pyproject)["sources"])
        orphaned = sorted(set(sources) - _group(pyproject, "asr-qwen"))
        assert not orphaned, (
            f"{orphaned} are routed to an index but are not direct dependencies, so the "
            f"routing does nothing and they will resolve from PyPI"
        )


class TestDependencyIsolation:
    def test_asr_qwen_group_holds_the_heavyweight_runtime(
        self, pyproject: dict[str, object]
    ) -> None:
        assert {"torch", "transformers", "accelerate"} <= _group(pyproject, "asr-qwen")

    def test_asr_qwen_is_not_a_default_group(self, pyproject: dict[str, object]) -> None:
        """A plain `uv sync` must install none of it, so the everyday gate keeps running
        the group-absent case rather than proving it once (INV-05)."""
        assert "asr-qwen" not in _uv(pyproject).get("default-groups", [])

    def test_nothing_the_project_needs_reaches_torch(self, lock: dict[str, object]) -> None:
        """The offline half of INV-05's proof: walk the lock's own dependency graph from
        the project and dev roots and show that no path arrives at torch.

        A `uv sync` in a clean environment would prove the same thing and needs the
        network, so it cannot live in this suite. This can, and it fails for the same
        reason: if torch ever became reachable without asking for the group, the default
        environment would start carrying a multi-gigabyte GPU stack.
        """
        packages = _packages(lock)
        root = packages["dnd-audio"]
        dev = _table(root["dev-dependencies"])

        frontier = _dependency_names(root)
        frontier += [str(entry["name"]) for entry in _rows(dev["dev"])]
        seen: set[str] = set()
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            frontier += _dependency_names(packages.get(name, {}))

        assert "torch" not in seen
        assert not seen & set(EXPECTED_AMD_PACKAGES)

    def test_setuptools_is_not_constrained_below_70_2(
        self, pyproject: dict[str, object], lock: dict[str, object]
    ) -> None:
        """The spec's instruction: `rocm[libraries]` builds at install time and its build
        needs setuptools >= 70.2, so nothing here may pin it lower."""
        text = str(pyproject)
        assert "setuptools<" not in text.replace(" ", "")
        setuptools = _packages(lock).get("setuptools")
        if setuptools is not None:
            major, minor = (int(part) for part in str(setuptools["version"]).split(".")[:2])
            assert (major, minor) >= (70, 2), setuptools["version"]


class TestTheLockIsWhatWeAskedFor:
    """The completion gate's central claim, asserted in both directions."""

    def test_exactly_the_expected_packages_come_from_the_amd_index(
        self, lock: dict[str, object]
    ) -> None:
        from_amd = {
            name: str(package["version"])
            for name, package in _packages(lock).items()
            if _registry(package) == AMD_INDEX
        }
        assert from_amd == EXPECTED_AMD_PACKAGES

    def test_everything_else_comes_from_pypi(self, lock: dict[str, object]) -> None:
        """The direction that catches a silently-ignored routing entry. A package that
        should have come from AMD and did not is invisible in the other direction: it is
        simply present, at a plausible version, from the wrong place."""
        strays = {
            name: _registry(package)
            for name, package in _packages(lock).items()
            if name not in EXPECTED_AMD_PACKAGES
            and _registry(package) not in (None, "https://pypi.org/simple")
        }
        assert strays == {}

    def test_the_locked_versions_are_the_ones_pyproject_pins(
        self, pyproject: dict[str, object], lock: dict[str, object]
    ) -> None:
        """Ties two independently-edited files, and de-circularises the allowlist above.

        Three of the five expected versions were *discovered* by the resolver rather than
        chosen by hand, so asserting the lock against a constant copied out of the lock
        would only prove the lock equals itself. The group's `==` pins are the independent
        statement of intent; this checks the lock honours them.
        """
        pins = {
            str(item).split("==")[0].split("[")[0].strip(): str(item).split("==")[1].strip()
            for item in _rows(_table(pyproject["dependency-groups"])["asr-qwen"])
            if "==" in str(item)
        }
        packages = _packages(lock)
        for name, pinned in pins.items():
            assert str(packages[name]["version"]) == pinned, name
        for name in EXPECTED_AMD_PACKAGES:
            assert pins[name] == EXPECTED_AMD_PACKAGES[name], name

    def test_the_amd_index_publishes_no_hashes_so_the_lock_pins_versions_not_bytes(
        self, lock: dict[str, object]
    ) -> None:
        """A property of the supply chain, asserted so it stops being a surprise.

        Every PyPI artifact in this lock carries a sha256; not one AMD artifact does,
        because the index publishes none. So "locked" here means the exact *version* is
        pinned, and the *bytes* are not — a re-upload at the same version would be
        invisible. Recorded in ADR-0025's consequences; this test is what makes the
        statement true rather than remembered, and it will fail loudly on the good day AMD
        starts publishing hashes, which is a change worth noticing.
        """
        amd_artifacts, hashed = [], []
        for name, package in _packages(lock).items():
            if _registry(package) != AMD_INDEX:
                continue
            entries = list(package.get("wheels") or [])
            if package.get("sdist"):
                entries.append(package["sdist"])
            amd_artifacts += [(name, entry) for entry in entries]
            hashed += [(name, entry) for entry in entries if "hash" in entry]

        assert amd_artifacts, "no AMD artifacts in the lock at all"
        assert hashed == [], hashed

        pypi_hashed = any(
            "hash" in entry
            for package in _packages(lock).values()
            if _registry(package) == "https://pypi.org/simple"
            for entry in (package.get("wheels") or [])
        )
        assert pypi_hashed, "PyPI artifacts should carry hashes; the contrast is the point"

    def test_torch_is_a_rocm_build_and_not_a_cuda_one(self, lock: dict[str, object]) -> None:
        torch = _packages(lock)["torch"]
        assert "+rocm" in str(torch["version"])
        assert _registry(torch) == AMD_INDEX

    def test_no_cuda_wheels_are_in_the_lock(self, lock: dict[str, object]) -> None:
        """`torch` from PyPI drags in a dozen `nvidia-*` packages on this platform, so
        their absence is a second, independent signal that the routing held."""
        cuda = sorted(
            name
            for name in _packages(lock)
            if name.startswith("nvidia-") or name.endswith(("-cu12", "-cu126", "-cu128"))
        )
        assert cuda == []

    def test_accelerate_is_present_and_wants_torch(self, lock: dict[str, object]) -> None:
        """Without this the assertions above are vacuous.

        The failure the spec names is `accelerate`'s transitive `torch>=2.0.0` resolving a
        CUDA build. With torch alone in the lock nothing competes for it and "no CUDA
        build won" is true because nobody asked. This test is what makes the rest mean
        something, and it is why `accelerate` is in M6a rather than M6b.
        """
        packages = _packages(lock)
        wants = set(_dependency_names(packages["accelerate"]))

        assert "torch" in wants
        assert "+rocm" in str(packages["torch"]["version"])

    def test_the_lock_still_contains_what_the_project_needs(self, lock: dict[str, object]) -> None:
        names = set(_packages(lock))
        assert {"typer", "pydantic", "pyyaml", "numpy", "pytest", "ruff", "mypy"} <= names


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

    def test_the_gate_excludes_both_markers(self, repo_root: Path) -> None:
        """Registering a marker means nothing if the gate still runs those tests.

        `allow_network` matters as much as `host_smoke`: it is the socket block's own
        escape hatch, so a suite that runs opted-out tests is not the offline suite
        INV-05 describes.
        """
        gate = (repo_root / "scripts" / "gate.sh").read_text(encoding="utf-8")
        assert "-m 'not host_smoke and not allow_network'" in gate

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
