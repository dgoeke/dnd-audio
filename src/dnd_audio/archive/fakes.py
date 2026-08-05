"""A deterministic storage backend, and the faults a real bucket eventually produces.

INV-10's principle applied one layer out: the thing that fails in interesting ways sits
behind an interface, and the default suite drives a fake so every failure path is a test
rather than a story. What is faked here is not a model but a network — and the failures
worth rehearsing are the ones that happen at 2 a.m. on a large session, which is exactly
when nobody is watching.

Faults are **scheduled**, not random: a run that fails differently each time cannot be
debugged, and a flaky test would train someone to re-run it. Each fault names a key and an
occurrence, so "the third read of this object returns corrupt bytes" is a thing a test can
state exactly.

Pagination is deliberately small. `list_keys` yields in pages of two, so a caller that
forgot to follow markers is wrong on any prefix holding three objects rather than wrong
only at a thousand — the size at which nobody would have written the fixture.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.storage import ObjectHead
from dnd_audio.determinism import BinaryReader, sha256_bytes

__all__ = ["FakeArchiveStorage", "SlowDownError", "StorageFault"]

#: How many keys a listing page holds. Two, so a caller that ignores the marker fails on
#: three objects instead of on a thousand.
_PAGE = 2


class SlowDownError(Exception):
    """The provider's `503 Slow Down`, which is retryable and ordinary under load.

    A distinct type rather than an `ArchiveError`, because the adapter's whole job at this
    boundary is to tell "retry this" from "stop": raising the project's fatal error here
    would make the retry loop unwritable.
    """


@dataclass(frozen=True, slots=True)
class StorageFault:
    """One scheduled failure.

    Args:
        key: Which object. Matched exactly.
        operation: ``put``, ``get``, ``head`` or ``list``.
        occurrence: 1-based. ``2`` means "succeed once, then fail", which is how a
            bounded-retry test proves the retry actually retried rather than never failing.
        kind: ``slow_down`` for a retryable `503`; ``corrupt`` to return bytes that are not
            what was stored; ``truncate`` to return a short body; ``error`` for a fatal
            provider error; ``interrupt`` to raise mid-transfer after some bytes moved.
    """

    key: str
    operation: str
    occurrence: int = 1
    kind: str = "error"


@dataclass
class FakeArchiveStorage:
    """An in-memory bucket that records what it was asked to do.

    Satisfies :class:`~dnd_audio.archive.storage.ArchiveStorage` structurally. Not
    registered as an implementation of it anywhere: the protocol is what the runner depends
    on, and a fake that had to be declared would be one more place to keep in step.
    """

    objects: dict[str, bytes] = field(default_factory=dict)
    faults: list[StorageFault] = field(default_factory=list)
    #: Every operation performed, in order, as ``(operation, key)``. What lets a test assert
    #: that a readback actually happened rather than trusting a return value.
    calls: list[tuple[str, str]] = field(default_factory=list)
    _seen: dict[tuple[str, str], int] = field(default_factory=dict)

    def arm(self, *faults: StorageFault) -> None:
        """Schedule faults counted from **now**, not from this object's creation.

        Appending to :attr:`faults` directly counts occurrences over the fake's whole
        lifetime, which is right for a fault set up before any traffic and a trap for one
        set up after. A test that uploads a session and then wants "the next read of this
        object is corrupt" would otherwise have to know that the upload's own readback
        already consumed occurrence 1 — and would silently pass by never firing the fault
        at all, which is the worst way for a fault-injection test to be wrong.
        """
        for fault in faults:
            self._seen.pop((fault.operation, fault.key), None)
            self.faults.append(fault)

    # --- the protocol ----------------------------------------------------------------

    def put_object(self, key: str, source: Path) -> None:
        self.calls.append(("put", key))
        self._maybe_fail("put", key)
        # Read in chunks rather than `read_bytes()`, so the fake is not the thing that
        # makes a bounded-memory test pass.
        buffer = bytearray()
        with source.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                buffer += chunk
        self.objects[key] = bytes(buffer)

    def head_object(self, key: str) -> ObjectHead | None:
        self.calls.append(("head", key))
        self._maybe_fail("head", key)
        stored = self.objects.get(key)
        if stored is None:
            return None
        # A plausible ETag that is deliberately *not* the content sha256, so a caller that
        # tried to use it as one gets a wrong answer in a test rather than in production.
        return ObjectHead(key=key, size_bytes=len(stored), etag=f'"{sha256_bytes(stored)[:32]}-3"')

    @contextmanager
    def open_object(self, key: str) -> Iterator[BinaryReader]:
        self.calls.append(("get", key))
        fault = self._maybe_fail("get", key)
        stored = self.objects.get(key)
        if stored is None:
            message = f"no archived object at {key}"
            raise ArchiveError(message, code="archive_object_missing")

        if fault is not None and fault.kind == "corrupt":
            corrupted = bytearray(stored)
            corrupted[len(corrupted) // 2] ^= 0xFF
            stored = bytes(corrupted)
        elif fault is not None and fault.kind == "truncate":
            stored = stored[: max(1, len(stored) // 2)]

        if fault is not None and fault.kind == "interrupt":
            yield _InterruptingReader(stored)
            return
        yield io.BytesIO(stored)

    def list_keys(self, prefix: str) -> Iterator[str]:
        self.calls.append(("list", prefix))
        self._maybe_fail("list", prefix)
        matching = sorted(key for key in self.objects if key.startswith(prefix))
        # Paged deliberately, and yielded page by page, so a caller that stops after the
        # first page is visibly wrong.
        for start in range(0, len(matching), _PAGE):
            yield from matching[start : start + _PAGE]

    # --- fault scheduling ------------------------------------------------------------

    def _maybe_fail(self, operation: str, key: str) -> StorageFault | None:
        seen = self._seen.get((operation, key), 0) + 1
        self._seen[(operation, key)] = seen
        for fault in self.faults:
            if fault.operation != operation or fault.key != key or fault.occurrence != seen:
                continue
            if fault.kind == "slow_down":
                message = f"503 Slow Down on {operation} {key}"
                raise SlowDownError(message)
            if fault.kind == "error":
                message = f"the storage provider refused {operation} on {key}"
                raise ArchiveError(message, code="archive_storage_error")
            return fault
        return None


class _InterruptingReader:
    """Returns some bytes, then fails — a connection dropping mid-transfer.

    The failure mode a `read()`-it-all implementation hides completely: it either gets
    everything or raises, and never has to think about the half it already consumed.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1, /) -> bytes:
        if self._offset >= len(self._payload) // 2:
            message = "the connection dropped partway through reading the object"
            raise ArchiveError(message, code="archive_transfer_interrupted")
        take = len(self._payload) if size < 0 else size
        chunk = self._payload[self._offset : self._offset + take]
        self._offset += len(chunk)
        return chunk
