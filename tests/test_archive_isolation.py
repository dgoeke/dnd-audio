"""M7a moves no processing identity, and opens no socket outside `archive` (INV-06, INV-08).

Two properties, and both were rewritten after the plan review found the first drafts could
pass while the property they named was false.

**The cache freeze compares keys, not outputs.** The first draft compared `config_hash`,
the five stage projections, and the final artifact bytes. That is not INV-08: a run whose
key moved recomputes the work and produces byte-identical artifacts anyway, so the
comparison passes on exactly the failure it exists to catch. What is compared here is the
**materialized cache identity** — every sidecar under `work/cache/`, by path and by bytes,
across every cache a run touches — plus the behavioural signal that no key moved, which is
a warm re-run reporting zero misses. ASR is the reason the projection comparison alone was
never enough: its identity is content-addressed on the audio submitted and the inference
parameters used (ADR-0019), and it is not a `stage_config_hash` at all.

Asserted by **glob over the whole cache tree** rather than by naming the caches known
today, which is M3's lesson: a test that enumerates the caches its own milestone added
cannot see the one added next.

**The network boundary is proved in a subprocess.** "`boto3` is not in `sys.modules`" is a
weaker claim than "no socket was opened" — an implementation reaching the network through
stdlib would satisfy it — and the first draft also covered four commands out of eight. A
subprocess has its own address space and escapes the autouse socket fixture, exactly as
INV-05 records, so the trap travels on the child's `PYTHONPATH` the way
`tests/test_runtime.py::shadow` shadows Torch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dnd_audio.archive.config import ENV_PREFIX
from dnd_audio.config import StageScope, config_hash, load_session_config, stage_config_hash
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.mix.runner import run_mix
from dnd_audio.transcript.runner import run_transcribe

#: A complete, plausible archive environment. Values are fictional; what matters is that
#: every variable the archive layer reads is set, so a leak into a processing identity has
#: something to leak.
ARCHIVE_ENVIRONMENT = {
    f"{ENV_PREFIX}ENDPOINT_URL": "https://nyc3.digitaloceanspaces.com",
    f"{ENV_PREFIX}REGION": "nyc3",
    f"{ENV_PREFIX}BUCKET": "example-cold-bucket",
    f"{ENV_PREFIX}ACCESS_KEY_ID": "DO00EXAMPLEACCESSKEY",
    f"{ENV_PREFIX}SECRET_ACCESS_KEY": "wJalrXUtnFEMI-K7MDENG-bPxRfiCYEXAMPLEKEY",
    f"{ENV_PREFIX}MULTIPART_THRESHOLD_BYTES": "5242880",
    f"{ENV_PREFIX}MULTIPART_PART_BYTES": "5242880",
    f"{ENV_PREFIX}MAX_RETRIES": "2",
}

#: Every command that must remain network-denied. Stated explicitly rather than derived
#: from the Typer app, because deriving it would let a newly added command pass by
#: construction — which is the failure this list exists to prevent. `tests/test_cli.py`
#: asserts the full command surface, so a command added without appearing here is visible
#: there.
NETWORK_DENIED_COMMANDS = (
    "inspect",
    "ingest",
    "activity",
    "transcribe",
    "render",
    "mix",
    "process",
    "doctor",
)


def cache_tree(session_dir: Path) -> dict[str, bytes]:
    """Every cached artifact and sidecar, by session-relative path.

    The sidecar *is* the materialized cache identity: its path is derived from the key and
    its body records what the key was computed from. Comparing the whole tree therefore
    compares every key, including ones this test does not know the name of.
    """
    root = session_dir / "work" / "cache"
    return {
        path.relative_to(session_dir).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_both_branches(session_dir: Path) -> tuple[int, int]:
    """Run the transcript and mix branches. Returns aggregate `(hits, misses)`.

    Both branches, because between them they touch all six caches: inspection, derivative,
    detection, attribution, ASR, and mix.
    """
    transcribed = run_transcribe(session_dir, fake_models=True)
    mixed = run_mix(session_dir)
    assert transcribed.records is not None, "the transcript branch must succeed to compare it"
    assert mixed.encode is not None, "the mix branch must succeed to compare it"
    return (
        transcribed.report.telemetry.cache_hits + mixed.report.telemetry.cache_hits,
        transcribed.report.telemetry.cache_misses + mixed.report.telemetry.cache_misses,
    )


class TestArchiveConfigurationMovesNoProcessingIdentity:
    """INV-08: archive settings must reach no cache key of any stage."""

    def test_a_warm_rerun_under_archive_settings_misses_nothing(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The behavioural proof, and the one that cannot be satisfied by accident.

        Build every cache with no archive configuration present, then set all of it and
        run again. A single moved key is a miss, and misses are counted across every cache
        the run touched. Comparing artifacts instead would pass here even if every key had
        changed, because recomputing the work produces the same bytes.
        """
        session_dir = canonical_fixture.session_dir
        _, cold_misses = run_both_branches(session_dir)
        assert cold_misses > 0, "the first run must actually populate the caches"

        for name, value in ARCHIVE_ENVIRONMENT.items():
            monkeypatch.setenv(name, value)

        warm_hits, warm_misses = run_both_branches(session_dir)
        assert warm_misses == 0, (
            "setting archive configuration moved a cache key: the second run recomputed "
            "work it should have served from cache (INV-08)"
        )
        assert warm_hits > 0

    def test_a_cold_run_under_archive_settings_writes_the_same_cache_tree(
        self, canonical_fixture: FixtureTruth, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The structural proof: identical sidecar paths and identical sidecar bytes.

        Two *cold* runs on two copies, so nothing is served from a cache the other built
        and the identities are computed from scratch both times. A key that moved changes a
        sidecar's path; an identity document that gained a field changes its bytes.
        """
        without = canonical_fixture.session_dir
        with_settings = tmp_path / "with-archive-settings"
        # Copied before either run, so both start cold from the same sources rather than
        # one inheriting the other's caches.
        shutil.copytree(without, with_settings)

        run_both_branches(without)
        baseline = cache_tree(without)
        assert baseline, "the fixture must produce cached work for this to compare anything"

        for name, value in ARCHIVE_ENVIRONMENT.items():
            monkeypatch.setenv(name, value)
        run_both_branches(with_settings)

        assert cache_tree(with_settings) == baseline, (
            "archive configuration changed the cache tree: some stage's identity document "
            "or sidecar path depends on a setting that must not reach it (INV-08)"
        )

    def test_the_configuration_hashes_are_unmoved(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cheap direct check, kept alongside the two above rather than instead.

        It is the one that names *which* projection moved when something does, which the
        tree comparison cannot.
        """
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        stages: tuple[StageScope, ...] = (
            "inspection",
            "derivative",
            "detection",
            "attribution",
            "mix",
        )
        before = (config_hash(config), tuple(stage_config_hash(config, s) for s in stages))

        for name, value in ARCHIVE_ENVIRONMENT.items():
            monkeypatch.setenv(name, value)

        reloaded = load_session_config(canonical_fixture.session_dir / "session.yaml")
        after = (config_hash(reloaded), tuple(stage_config_hash(reloaded, s) for s in stages))
        assert after == before

    def test_session_config_has_no_archive_section(self, canonical_fixture: FixtureTruth) -> None:
        """The structural reason the above hold: there is nothing to leak.

        `SessionConfig` forbids unknown keys, so this is really asserting that nobody added
        one — which is the change that would make every test above start failing at once,
        and this is the one that says why.
        """
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        assert not any("archive" in field for field in type(config).model_fields), (
            "archive settings belong outside SessionConfig entirely (ADR-0035)"
        )


#: Fails the child process on socket construction *or* storage-client construction, before
#: either can reach the network. Written to the child's `PYTHONPATH` as `sitecustomize`,
#: which the interpreter imports during startup — earlier than any project module, so a
#: command cannot open a socket before the trap is armed.
_TRAP = '''
import socket
import sys


class NetworkReached(RuntimeError):
    pass


def _fail(*args, **kwargs):
    raise NetworkReached(
        "this command constructed a socket or a storage client. Only `models fetch` "
        "and `archive` may touch the network (INV-06)."
    )


socket.socket.connect = _fail
socket.socket.connect_ex = _fail
socket.create_connection = _fail
socket.getaddrinfo = _fail
socket.gethostbyname = _fail


class _Guard:
    """Refuses to hand out an S3 client, however it is asked for."""

    def __getattr__(self, name):
        if name in ("client", "resource", "Session"):
            return _fail
        raise AttributeError(name)


sys.modules["boto3"] = _Guard()
'''


@pytest.fixture(scope="module")
def trap_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding the `sitecustomize` trap, for the child's `PYTHONPATH`."""
    directory = tmp_path_factory.mktemp("network-trap")
    (directory / "sitecustomize.py").write_text(_TRAP, encoding="utf-8")
    return directory


def invoke(command: str, session_dir: Path, trap_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run one CLI command in a child whose network access fails loudly."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(trap_dir)
    arguments = [sys.executable, "-m", "dnd_audio.cli", command]
    if command != "doctor":
        arguments.append(str(session_dir))
    if command in ("transcribe", "process"):
        arguments.append("--fake-models")
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
        check=False,
    )


class TestNoProcessingCommandTouchesTheNetwork:
    """INV-06, proved where the autouse socket fixture cannot reach."""

    @pytest.mark.parametrize("command", NETWORK_DENIED_COMMANDS)
    def test_the_command_neither_opens_a_socket_nor_builds_a_client(
        self, command: str, canonical_fixture: FixtureTruth, trap_dir: Path
    ) -> None:
        completed = invoke(command, canonical_fixture.session_dir, trap_dir)
        combined = completed.stdout + completed.stderr
        assert "NetworkReached" not in combined, (
            f"`dnd-audio {command}` reached the network. Only `models fetch` and "
            f"`archive` may (INV-06, ADR-0035).\n{combined}"
        )

    def test_the_trap_can_actually_fire(
        self, canonical_fixture: FixtureTruth, trap_dir: Path
    ) -> None:
        """Otherwise the eight assertions above are eight ways of proving nothing.

        M1's closeout records the shape: a check that is present, looks right, and verifies
        nothing. A trap that failed to install would make every test in this class pass.
        """
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(trap_dir)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket; socket.create_connection(('example.invalid', 80))",
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
            check=False,
        )
        assert "NetworkReached" in completed.stdout + completed.stderr
        assert completed.returncode != 0
