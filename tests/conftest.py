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
import socket
from pathlib import Path
from typing import Any

import pytest

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
