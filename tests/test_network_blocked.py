"""INV-05: the default suite is offline, and a violation fails loudly.

These tests exercise the block itself. They live in the default suite rather than
behind ``host_smoke`` on purpose: a proof the gate does not run is not a proof.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from tests.conftest import NetworkAccessBlockedError

# A port nothing listens on, so a test that somehow got through would fail on connect
# rather than accidentally talking to something real.
_DISCARD = ("127.0.0.1", 9)


def test_tcp_connect_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessBlockedError),
    ):
        sock.connect(_DISCARD)


def test_tcp_connect_ex_is_blocked() -> None:
    """`connect_ex` returns an error code rather than raising, so it needs its own guard."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessBlockedError),
    ):
        sock.connect_ex(_DISCARD)


def test_unconnected_udp_send_is_blocked() -> None:
    """UDP never calls connect. Blocking only connect would leave this wide open."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(NetworkAccessBlockedError),
    ):
        sock.sendto(b"leak", _DISCARD)


def test_ipv6_is_blocked_too() -> None:
    with (
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessBlockedError),
    ):
        sock.connect(("::1", 9))


def test_name_resolution_is_blocked() -> None:
    """A DNS lookup leaves the machine even when no connection follows."""
    with pytest.raises(NetworkAccessBlockedError):
        socket.getaddrinfo("example.invalid", 80)


def test_reverse_name_resolution_is_blocked() -> None:
    """Reverse lookups go out through NSS and DNS exactly like forward ones."""
    with pytest.raises(NetworkAccessBlockedError):
        socket.gethostbyaddr("192.0.2.1")


def test_getnameinfo_is_blocked() -> None:
    with pytest.raises(NetworkAccessBlockedError):
        socket.getnameinfo(("192.0.2.1", 80), 0)


def test_forward_lookup_by_name_is_blocked() -> None:
    with pytest.raises(NetworkAccessBlockedError):
        socket.gethostbyname("example.invalid")


def test_create_connection_is_blocked() -> None:
    with pytest.raises(NetworkAccessBlockedError):
        socket.create_connection(_DISCARD, timeout=0.1)


def test_unix_sockets_still_work(tmp_path: Path) -> None:
    """AF_UNIX cannot reach the network, so blocking it would only break tooling."""
    address = str(tmp_path / "socket")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(address)
        server.listen(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(address)
            connection, _ = server.accept()
            with connection:
                client.sendall(b"local")
                assert connection.recv(5) == b"local"


def test_blocking_is_undone_between_tests() -> None:
    """The guard is a monkeypatch, not a permanent mutation of the socket module.

    If it leaked, an `allow_network` test in M6b would be blocked anyway and the
    opt-out would be a lie.
    """
    assert socket.getaddrinfo.__name__ == "blocked"


@pytest.mark.allow_network
def test_marker_opts_out() -> None:
    """`allow_network` restores the real functions.

    The marker is reserved for `models fetch` (INV-06). This test only proves the
    mechanism works — it makes no connection.
    """
    assert socket.getaddrinfo.__name__ != "blocked"
