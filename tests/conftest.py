"""Shared fixtures, and the thing that makes INV-05 true rather than aspirational.

The default suite must pass with no network. Enforcing that by convention does not
work: a dependency reaches out, the developer's machine has connectivity, the test
passes, and the failure only appears on a machine that does not — or, worse, quietly
sends something somewhere (INV-06).

So outbound network access is blocked for every test, and a violation raises
:class:`NetworkAccessBlockedError` rather than timing out or silently succeeding.

**What is blocked.** ``connect``, ``connect_ex``, ``sendto``, and ``sendmsg`` on any
``AF_INET``/``AF_INET6`` socket, plus name resolution (``getaddrinfo``,
``gethostbyname``, ``gethostbyname_ex``) and ``create_connection``. Unconnected UDP and
DNS matter as much as TCP: both leave the machine, and blocking only ``connect`` would
miss them.

**What is not blocked.** ``AF_UNIX``. A Unix socket cannot reach the network, pytest
internals and some libraries use them, and blocking them buys nothing.

**The honest boundary.** A subprocess has its own address space, so this fixture cannot
constrain one. Nothing in the default suite spawns a network-capable subprocess other
than this project's own CLI. OS-level isolation would close the gap and is not worth its
complexity here; if that changes, this is the comment to revisit.

**The one opt-out.** ``@pytest.mark.allow_network``, reserved for `models fetch` — the
only command INV-06 permits to touch the network. It is deliberately *not* tied to
``host_smoke``: needing a GPU is not a reason to need the internet, and a real Qwen
smoke test runs against models that were already fetched.

``./scripts/gate.sh`` excludes ``allow_network`` as well as ``host_smoke``. Without
that, an opted-out test would still run in the suite the gate calls offline, and the
invariant would hold only until the first test used its own escape hatch.
"""

from __future__ import annotations

import datetime as dt
import shutil
import socket
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.fixtures import FixtureTruth, build_session, canonical_session

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent

#: Address families that cannot leave the machine.
_LOCAL_ONLY_FAMILIES = frozenset({getattr(socket, "AF_UNIX", None)}) - {None}

#: Socket methods that can move bytes to another host.
_BLOCKED_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")

#: Module-level functions that resolve names or open connections. Reverse lookups
#: (`gethostbyaddr`, `getnameinfo`) go out through NSS and DNS exactly like forward
#: ones, so leaving them open would leave a hole the forward-lookup block only appears
#: to close.
_BLOCKED_MODULE_FUNCTIONS = (
    "create_connection",
    "getaddrinfo",
    "gethostbyaddr",
    "gethostbyname",
    "gethostbyname_ex",
    "getnameinfo",
)


class NetworkAccessBlockedError(RuntimeError):
    """Raised when a test attempts to reach the network (INV-05)."""


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block outbound network access for the duration of one test."""
    if request.node.get_closest_marker("allow_network"):
        return

    for name in _BLOCKED_SOCKET_METHODS:
        original = getattr(socket.socket, name)
        monkeypatch.setattr(socket.socket, name, _guard_socket_method(name, original))

    for name in _BLOCKED_MODULE_FUNCTIONS:
        monkeypatch.setattr(socket, name, _blocked_function(name))


def _guard_socket_method(name: str, original: Any) -> Any:
    def guarded(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
        if self.family in _LOCAL_ONLY_FAMILIES:
            return original(self, *args, **kwargs)
        message = (
            f"socket.{name}() on an {self.family!r} socket is blocked in the default test "
            f"suite (INV-05). If this is `models fetch`, mark the test "
            f"@pytest.mark.allow_network; otherwise the code under test should not be "
            f"reaching the network."
        )
        raise NetworkAccessBlockedError(message)

    return guarded


def _blocked_function(name: str) -> Any:
    def blocked(*args: Any, **kwargs: Any) -> Any:
        message = (
            f"socket.{name}() is blocked in the default test suite (INV-05): "
            f"name resolution and connection setup both leave this machine."
        )
        raise NetworkAccessBlockedError(message)

    return blocked


@pytest.fixture(autouse=True)
def no_torch_import(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail any default-suite test that leaves Torch resident in ``sys.modules``.

    INV-05's other half, and it exists because the first version of M6a broke it. `doctor`
    legitimately probes the GPU — that is its job — so the four `test_cli.py` invocations
    of it started importing Torch and running kernels *inside the default suite*. On the
    project environment nothing happened, because there is no Torch to import. On the ROCm
    environment the suite began doing GPU work it is specified not to do, and the only
    symptom was an unrelated test in `test_silero.py` failing on run order.

    That is the shape this project keeps rediscovering: a rule held by convention, honoured
    everywhere it was thought about, and broken by the one place nobody did. So it is a
    fixture now, like the socket block above, and it names the test that did it rather than
    the test that noticed.

    Scoped to *newly* resident: a test that runs after one which already imported Torch is
    not the culprit, and blaming it would send the next person to the wrong file.
    ``host_smoke`` is exempt, since needing the real device is the whole point of the mark.
    """
    if request.node.get_closest_marker("host_smoke"):
        yield
        return

    before = "torch" in sys.modules
    yield
    if not before and "torch" in sys.modules:
        pytest.fail(
            "this test imported torch, which the default suite must not do (INV-05). "
            "Torch is in the opt-in `asr-qwen` group, so on the project environment this "
            "would have been an ImportError instead — the failure only appears on the "
            "ROCm environment, which is why it needs a fixture rather than a habit. "
            "Inject a `RuntimeProbe` instead of measuring the machine, or mark the test "
            "`host_smoke`.",
            pytrace=False,
        )


@pytest.fixture
def repo_root() -> Path:
    """The repository root. Only meaningful in a source checkout, which tests are."""
    return REPO_ROOT


@pytest.fixture
def valid_session_yaml() -> Path:
    """A `session.yaml` exercising every field, including the non-default ones."""
    return TESTS_ROOT / "data" / "session-valid.yaml"


@pytest.fixture
def minimal_session_yaml() -> Path:
    """The same session with every default omitted. Must hash identically (INV-08)."""
    return TESTS_ROOT / "data" / "session-minimal.yaml"


@pytest.fixture
def instant() -> dt.datetime:
    """A fixed timezone-aware instant, so report tests do not depend on the clock."""
    return dt.datetime(2026, 8, 15, 19, 0, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="session")
def _canonical_build(tmp_path_factory: pytest.TempPathFactory) -> FixtureTruth:
    """The canonical six-transmitter fixture, generated once for the whole run.

    Building it costs a few megabytes of synthesis per call, and nearly every test in
    M1 onward wants the same one. Private because it is shared mutable state on disk:
    depend on :func:`canonical_fixture` instead, which hands out a private copy.
    """
    return build_session(canonical_session(), tmp_path_factory.mktemp("canonical"))


@pytest.fixture
def canonical_fixture(_canonical_build: FixtureTruth, tmp_path: Path) -> FixtureTruth:
    """A private copy of the canonical fixture, safe to write `work/` into.

    `inspect` writes `work/` and `output/` inside the session directory, so tests that
    run it would otherwise see each other's caches — which is precisely how a cache test
    passes for the wrong reason.
    """
    session_dir = tmp_path / "session"
    shutil.copytree(_canonical_build.session_dir, session_dir)
    return replace(_canonical_build, session_dir=session_dir)


#: A commit that is valid in shape and installed nowhere. `asr.model_revision` is
#: validated as a 40-character hex sha (ADR-0027), so this is an *accepted* configuration
#: whose weights cannot be found — which is the point.
UNINSTALLED_REVISION: Final = "0" * 40


@pytest.fixture
def session_without_asr_models(canonical_fixture: FixtureTruth) -> FixtureTruth:
    """The canonical fixture, configured to want an ASR revision nothing has installed.

    **Every test about "a host that cannot transcribe" must use this rather than relying on
    the ambient environment**, and that is not a style preference — it is the lesson M6a's
    closeout records and that M6b then repeated. The first version of those tests simply ran
    `transcribe` and expected it to fail, which is true on `.venv` (no Torch, deliberately)
    and false on `.venv-rocm`, where the runtime and six gigabytes of weights are present and
    the run succeeds. They passed the gate and failed the moment the suite was run from the
    other environment — an assertion about the *machine* wearing the clothes of an assertion
    about the code.

    Pinning an uninstalled revision makes the failure a property of the configuration:
    `require_snapshot` refuses it identically on both environments, at the same point, with
    the same code. Silero is untouched, so the detector still loads and the mix branch still
    runs — which is what lets these tests check INV-09 rather than merely a total failure.
    """
    path = canonical_fixture.session_dir / "session.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document.setdefault("asr", {})["model_revision"] = UNINSTALLED_REVISION
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return canonical_fixture


@pytest.fixture(scope="session")
def canonical_activity_graph(tmp_path_factory: pytest.TempPathFactory) -> ActivityGraph:
    """The canonical fixture's real activity graph, built once for the whole run.

    Not hand-assembled: this is what `activity` actually produces from the fixture's audio,
    through the **leaky** scripted detector — the one that fires on bleed as well as speech,
    which is what a real detector does and what makes the bleed gate's decisions real. INV-10
    forbids expecting a learned Silero release to fire on synthetic noise, so the detector is
    scripted from the fixture's own declared truth.

    Shared without copying because `ActivityGraph` is frozen. The session directory it was
    built in is not shared, and nothing here should want it: a test that needs to *run* the
    stage wants `canonical_fixture`.
    """
    from dnd_audio.activity.runner import DetectorBundle, run_activity
    from dnd_audio.fakes import ScriptedActivityDetector
    from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

    truth = build_session(canonical_session(), tmp_path_factory.mktemp("graph"))
    detector = ScriptedActivityDetector(
        truth.leaky_activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE)
    )
    result = run_activity(
        truth.session_dir,
        detector=DetectorBundle(identity=detector.identity(), make=lambda _track: detector),
    )
    assert result.graph is not None, [
        f"{error.code}: {error.message}" for stage in result.report.stages for error in stage.errors
    ]
    return result.graph
