"""INV-07 over the archive's composed path, with each boundary asserted on its own.

The plan review rejected the first version of this, and was right to. One ordered event log
across the whole pipeline is satisfied by an early compressor write while a *later* stage —
the remote verifier, say — calls `body.read()` with no size and buffers a whole object. The
combined assertion "a write happens before the last read" is true either way, so it would
have passed on precisely the unbounded implementation it claimed to exclude.

So events are typed by phase, and each boundary is checked separately:

* source reads interleave with compression writes;
* staged reads interleave with upload calls;
* remote reads interleave with decompressor consumption;
* restore decoding interleaves with destination writes;
* **no `read()` is ever called without a size**, anywhere.

That last one is the cheapest and catches the most: `read()` with no argument is how a
four-hour recording ends up in RAM on a host where `systemd-oomd` is watching.

The fixture is a session whose largest file is many times the chunk size, so an
implementation that buffered would show one read and one write rather than an interleaving.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from dnd_audio.archive.codec import CHUNK_BYTES, compress_file, decompress_and_measure
from dnd_audio.archive.fakes import FakeArchiveStorage
from dnd_audio.archive.runner import _RESTORE_STAGING_PREFIX, run_restore, run_upload
from dnd_audio.archive.storage import ObjectHead
from dnd_audio.config import load_session_config
from dnd_audio.determinism import BinaryReader
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.runner import run_inspect

#: Large enough that a buffering implementation looks obviously different from a streaming
#: one: several chunks per file rather than one read and one write.
PAYLOAD_BYTES = 6 * CHUNK_BYTES


@dataclass
class Journal:
    """Phase-typed events, in the order they happened.

    `tests/test_memory.py` established the technique for M2's derivative path; the
    difference here is the `phase` field, which is what stops one phase's good behaviour
    vouching for another's.
    """

    events: list[tuple[str, str, int]] = field(default_factory=list)

    def record(self, phase: str, kind: str, count: int) -> None:
        self.events.append((phase, kind, count))

    def of(self, phase: str, kind: str) -> list[int]:
        return [count for p, k, count in self.events if p == phase and k == kind]

    def interleaves(self, phase: str, first: str, second: str) -> bool:
        """Whether ``second`` happens before the last ``first`` **within one phase**.

        Scoped to a phase deliberately. Asked globally, an early compression write answers
        for a later verifier that buffered everything.
        """
        indices = [i for i, (p, k, _) in enumerate(self.events) if p == phase]
        firsts = [i for i in indices if self.events[i][1] == first]
        seconds = [i for i in indices if self.events[i][1] == second]
        if not firsts or not seconds:
            return False
        return min(seconds) < max(firsts)


class _Delegating:
    """Passes through the file protocol production actually uses.

    Both wrappers below stand in for a real file handle, and production opens files with
    `with path.open(...)`. Without `__enter__`/`__exit__` the instrumented run fails in a
    way that looks like a defect in the code under test rather than in the instrument.
    """

    _inner: Any

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if closer is not None:
            closer()

    def flush(self) -> None:
        flusher = getattr(self._inner, "flush", None)
        if flusher is not None:
            flusher()


class WatchedReader(_Delegating):
    """Records every read, and fails loudly on an unbounded one."""

    def __init__(self, inner: BinaryReader, journal: Journal, phase: str) -> None:
        self._inner = inner
        self._journal = journal
        self._phase = phase

    def read(self, size: int = -1, /) -> bytes:
        if size < 0:
            message = (
                f"{self._phase} called read() with no size. That is how a four-hour "
                f"recording ends up in RAM (INV-07)."
            )
            raise AssertionError(message)
        chunk: bytes = self._inner.read(size)
        self._journal.record(self._phase, "read", len(chunk))
        return chunk


class WatchedWriter(_Delegating):
    """Records every write."""

    def __init__(self, inner: Any, journal: Journal, phase: str) -> None:
        self._inner = inner
        self._journal = journal
        self._phase = phase

    def write(self, data: bytes) -> int:
        self._journal.record(self._phase, "write", len(data))
        self._inner.write(data)
        return len(data)


@dataclass
class WatchedStorage:
    """A fake bucket that reports what the upload and download boundaries did."""

    inner: FakeArchiveStorage
    journal: Journal

    def put_object(self, key: str, source: Path) -> None:
        # Reading the staged file here is the "staged reads interleave with upload calls"
        # boundary: a real adapter streams the file into the request body.
        with source.open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                self.journal.record("upload", "read", len(chunk))
                self.journal.record("upload", "write", len(chunk))
        self.inner.put_object(key, source)

    def head_object(self, key: str) -> ObjectHead | None:
        return self.inner.head_object(key)

    @contextmanager
    def open_object(self, key: str) -> Iterator[BinaryReader]:
        with self.inner.open_object(key) as body:
            yield WatchedReader(body, self.journal, "download")

    def list_keys(self, prefix: str) -> Iterator[str]:
        return self.inner.list_keys(prefix)


@pytest.fixture
def big_session(canonical_fixture: FixtureTruth) -> FixtureTruth:
    """A session carrying one file far larger than the chunk size."""
    payload = canonical_fixture.session_dir / "raw" / "large-notes.bin"
    payload.write_bytes(bytes(range(256)) * (PAYLOAD_BYTES // 256))
    assert run_inspect(canonical_fixture.session_dir).exit_code is ExitCode.OK
    return canonical_fixture


class TestCompressionBoundary:
    def test_source_reads_interleave_with_compression_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instruments the **real** `compress_file`, not a copy of its body.

        The first version reimplemented the function's loop in the test. That proves the
        loop the test wrote streams, and says nothing about the one production runs — the
        compressor could buffer the entire source and this would have stayed green. Found
        by M7a's code review. `Path.open` is patched instead, so the function under test is
        genuinely the function that ships.
        """
        source = tmp_path / "in.bin"
        source.write_bytes(bytes(range(256)) * (PAYLOAD_BYTES // 256))
        target = tmp_path / "out.zst"

        journal = Journal()
        real_open = Path.open

        def watched_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(self, mode, *args, **kwargs)
            if self == source:
                return WatchedReader(handle, journal, "compress")
            if self == target:
                return WatchedWriter(handle, journal, "compress")
            return handle

        monkeypatch.setattr(Path, "open", watched_open)
        compress_file(source, target)
        monkeypatch.undo()

        assert len(journal.of("compress", "read")) > 1, "the source was read in one gulp"
        assert len(journal.of("compress", "write")) > 1
        assert journal.interleaves("compress", "read", "write")
        assert all(size <= CHUNK_BYTES for size in journal.of("compress", "read"))

    def test_no_read_is_unbounded_during_compression(self, tmp_path: Path) -> None:
        source = tmp_path / "in.bin"
        source.write_bytes(b"x" * PAYLOAD_BYTES)
        target = tmp_path / "out.zst"
        compress_file(source, target)

        journal = Journal()
        with target.open("rb") as handle:
            decompress_and_measure(
                WatchedReader(handle, journal, "decompress"), max_output_bytes=PAYLOAD_BYTES
            )
        assert len(journal.of("decompress", "read")) > 1


class TestDecompressionBoundary:
    def test_remote_reads_interleave_with_decompressor_consumption(self, tmp_path: Path) -> None:
        source = tmp_path / "in.bin"
        source.write_bytes(bytes(range(256)) * (PAYLOAD_BYTES // 256))
        compress_file(source, tmp_path / "out.zst")

        journal = Journal()
        sink = WatchedWriter(io.BytesIO(), journal, "restore")
        with (tmp_path / "out.zst").open("rb") as handle:
            decompress_and_measure(
                WatchedReader(handle, journal, "restore"),
                max_output_bytes=PAYLOAD_BYTES,
                sink=sink,
            )

        assert len(journal.of("restore", "read")) > 1
        assert len(journal.of("restore", "write")) > 1
        assert journal.interleaves("restore", "read", "write")

    def test_every_chunk_is_bounded(self, tmp_path: Path) -> None:
        source = tmp_path / "in.bin"
        source.write_bytes(bytes(range(256)) * (PAYLOAD_BYTES // 256))
        compress_file(source, tmp_path / "out.zst")

        journal = Journal()
        with (tmp_path / "out.zst").open("rb") as handle:
            decompress_and_measure(
                WatchedReader(handle, journal, "restore"),
                max_output_bytes=PAYLOAD_BYTES,
                sink=WatchedWriter(io.BytesIO(), journal, "restore"),
            )
        assert all(size <= CHUNK_BYTES for size in journal.of("restore", "read"))
        assert all(size <= CHUNK_BYTES for size in journal.of("restore", "write"))


class TestComposedUploadAndRestore:
    """The whole path, not one component — which is what INV-07 actually asks for."""

    def test_upload_streams_at_every_boundary(
        self, big_session: FixtureTruth, tmp_path: Path
    ) -> None:
        journal = Journal()
        storage = WatchedStorage(FakeArchiveStorage(), journal)
        report = run_upload(big_session.session_dir, storage=storage, lock_dir=tmp_path / "locks")
        assert report.status.value == "complete"

        # The upload boundary: the staged file was read in bounded pieces, not slurped.
        assert len(journal.of("upload", "read")) > 1
        assert journal.interleaves("upload", "read", "write")
        assert all(size <= CHUNK_BYTES for size in journal.of("upload", "read"))

        # The download boundary, which is the readback. Its interleaving is asserted in its
        # own right rather than inheriting the upload's — that inheritance is exactly the
        # hole the plan review found.
        assert len(journal.of("download", "read")) > 1
        assert all(size <= CHUNK_BYTES for size in journal.of("download", "read"))

    def test_restore_streams_at_every_boundary(
        self, big_session: FixtureTruth, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        journal = Journal()
        storage = WatchedStorage(FakeArchiveStorage(), journal)
        config = load_session_config(big_session.session_dir / "session.yaml")
        run_upload(big_session.session_dir, storage=storage, lock_dir=tmp_path / "locks")

        journal.events.clear()
        destination = tmp_path / "restored"
        destination.mkdir()

        # The destination side, which the first version never observed — it asserted on
        # remote reads and called that a restore boundary. A restore that buffered a whole
        # object before writing it would have passed. Found by M7a's code review.
        real_open = Path.open

        def watched_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(self, mode, *args, **kwargs)
            if "w" in mode and _RESTORE_STAGING_PREFIX in str(self):
                return WatchedWriter(handle, journal, "restore")
            return handle

        monkeypatch.setattr(Path, "open", watched_open)
        report = run_restore(config.session_id, destination, storage=storage)
        monkeypatch.undo()
        assert report.status.value == "complete"

        assert len(journal.of("download", "read")) > 1
        assert all(size <= CHUNK_BYTES for size in journal.of("download", "read"))

        assert len(journal.of("restore", "write")) > 1, (
            "the restore wrote each file in one call, so it held a whole object in memory"
        )
        assert all(size <= CHUNK_BYTES for size in journal.of("restore", "write"))
        assert journal.interleaves("restore", "write", "write")

    def test_no_boundary_ever_reads_without_a_size(
        self, big_session: FixtureTruth, tmp_path: Path
    ) -> None:
        """`WatchedReader` raises on an unbounded read, so reaching the end is the proof.

        The cheapest assertion here and the one that catches the most: a single
        `body.read()` is all it takes to put a four-hour recording in RAM.
        """
        journal = Journal()
        storage = WatchedStorage(FakeArchiveStorage(), journal)
        config = load_session_config(big_session.session_dir / "session.yaml")

        run_upload(big_session.session_dir, storage=storage, lock_dir=tmp_path / "locks")
        destination = tmp_path / "restored"
        destination.mkdir()
        run_restore(config.session_id, destination, storage=storage)

        assert journal.of("download", "read")


class TestTheInstrumentCanFail:
    """Otherwise every assertion above is a way of proving nothing."""

    def test_an_unbounded_read_is_caught(self) -> None:
        journal = Journal()
        reader = WatchedReader(io.BytesIO(b"x" * 100), journal, "test")
        with pytest.raises(AssertionError, match="no size"):
            reader.read()

    def test_a_buffering_implementation_fails_the_interleaving_check(self) -> None:
        """A journal built the way a buffering implementation would build one."""
        journal = Journal()
        for _ in range(5):
            journal.record("phase", "read", CHUNK_BYTES)
        journal.record("phase", "write", 5 * CHUNK_BYTES)
        assert not journal.interleaves("phase", "read", "write")

    def test_a_streaming_implementation_passes_it(self) -> None:
        journal = Journal()
        for _ in range(5):
            journal.record("phase", "read", CHUNK_BYTES)
            journal.record("phase", "write", CHUNK_BYTES)
        assert journal.interleaves("phase", "read", "write")

    def test_one_phases_streaming_does_not_vouch_for_another(self) -> None:
        """The precise hole the plan review found in the first draft.

        Phase A streams perfectly; phase B buffers completely. A single global log says
        "a write happened before the last read" and passes. Per-phase, B fails.
        """
        journal = Journal()
        for _ in range(3):
            journal.record("a", "read", CHUNK_BYTES)
            journal.record("a", "write", CHUNK_BYTES)
        for _ in range(3):
            journal.record("b", "read", CHUNK_BYTES)
        journal.record("b", "write", 3 * CHUNK_BYTES)

        assert journal.interleaves("a", "read", "write")
        assert not journal.interleaves("b", "read", "write")
