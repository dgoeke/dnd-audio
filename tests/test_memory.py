"""INV-07, proven over the whole path rather than over one component.

The obvious version of this test — assert `TrackReader` never returns more than a window —
is worthless on its own. It passes while a derivative builder collects every window it is
handed, or every `upfirdn` result, and writes at the end. The bound has to be a property of
the reader **and** the resampler **and** the writer composed, because that is what runs.

The property asserted here is ordering, not size: **a write happens before the last read**.
Nothing that accumulates a session-length array can satisfy it, because such an
implementation must finish reading before it writes anything. Size bounds are asserted
alongside, but the ordering is the one that cannot be faked.

The input is a twenty-minute virtual track backed by ten seconds of real audio. That is not
a trick — it is what a segment map *is*: a track that was switched off for nineteen minutes
costs nineteen minutes of derivative and no source bytes at all. It makes the test fast
while still being far larger than any window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.artifacts.timeline import TimelineSegment, TimelineTrack
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.reader import TrackReader
from dnd_audio.timeline.resample import decimate_stream, output_length
from dnd_audio.timeline.wavwrite import WavWriter

#: Twenty minutes at 48 kHz, against a 48 000-sample window: 1200 windows, so an
#: implementation that buffered would need a thousand times the window's memory.
LONG_SESSION_SAMPLES = 20 * 60 * CANONICAL_SAMPLE_RATE
WINDOW = CANONICAL_SAMPLE_RATE


@dataclass
class Journal:
    """An ordered log of what the path did, in the order it did it."""

    events: list[tuple[str, int]] = field(default_factory=list)

    def record(self, kind: str, count: int) -> None:
        self.events.append((kind, count))

    @property
    def reads(self) -> list[int]:
        return [count for kind, count in self.events if kind == "read"]

    @property
    def writes(self) -> list[int]:
        return [count for kind, count in self.events if kind == "write"]

    def first_write_index(self) -> int:
        return next(i for i, (kind, _) in enumerate(self.events) if kind == "write")

    def last_read_index(self) -> int:
        return max(i for i, (kind, _) in enumerate(self.events) if kind == "read")


@pytest.fixture
def long_track(canonical_fixture: FixtureTruth) -> tuple[Path, TimelineTrack, int]:
    """`tx-a`'s real chunks, read as a twenty-minute track.

    Everything past the track's own extent is silence, which is exactly what the reader
    returns for a transmitter that stopped early — so this is an ordinary session shape,
    not a special case built for the test.
    """
    chunks = sorted(canonical_fixture.for_track("tx-a"), key=lambda c: c.start_sample)
    track = TimelineTrack(
        track_id="tx-a",
        speaker_id="alice",
        speaker_name="Alice",
        start_sample=chunks[0].start_sample,
        end_sample=chunks[-1].start_sample + chunks[-1].n_samples,
        segments=[
            TimelineSegment(
                kind="audio",
                session_start_sample=chunk.start_sample,
                n_samples=chunk.n_samples,
                source_relative_path=chunk.relative_path,
                source_sha256=chunk.sha256,
                source_start_sample=0,
                evidence_start_sample=chunk.start_sample,
            )
            for chunk in chunks
        ],
    )
    return canonical_fixture.session_dir, track, LONG_SESSION_SAMPLES


class TestTheWholePathStreams:
    def test_a_write_happens_before_the_last_read(
        self, long_track: tuple[Path, TimelineTrack, int], tmp_path: Path
    ) -> None:
        """The assertion an accumulating implementation cannot pass.

        Reader → resampler → writer, instrumented at both ends. If any stage collected the
        session before handing it on, every read would precede every write and this would
        fail — which is precisely the shape the naive implementation has.
        """
        session_dir, track, duration = long_track
        journal = Journal()
        output = tmp_path / "derivative.wav"
        expected = output_length(duration, 3)

        with TrackReader(session_dir, track, duration) as reader:

            def instrumented() -> object:
                for _, block in reader.windows(window_samples=WINDOW):
                    journal.record("read", int(block.shape[0]))
                    yield block

            with WavWriter(output, sample_rate=16000, n_samples=expected) as writer:
                for produced in decimate_stream(instrumented(), duration):  # type: ignore[arg-type]
                    journal.record("write", int(produced.shape[0]))
                    writer.write(produced)

        assert journal.first_write_index() < journal.last_read_index()
        assert sum(journal.writes) == expected
        assert output.stat().st_size == expected * 4 + 44

    def test_no_single_read_or_write_exceeds_its_window(
        self, long_track: tuple[Path, TimelineTrack, int], tmp_path: Path
    ) -> None:
        """The size bound, alongside the ordering one.

        A window's worth of float32 is 192 kB; the whole track would be 230 MB, and six of
        them 1.4 GB. On a UMA host that difference is the one that kills a process.
        """
        session_dir, track, duration = long_track
        journal = Journal()

        with TrackReader(session_dir, track, duration) as reader:

            def instrumented() -> object:
                for _, block in reader.windows(window_samples=WINDOW):
                    journal.record("read", int(block.shape[0]))
                    yield block

            for produced in decimate_stream(instrumented(), duration):  # type: ignore[arg-type]
                journal.record("write", int(produced.shape[0]))

        assert max(journal.reads) <= WINDOW
        assert sum(journal.reads) == duration
        # Far more than one window in total, so the bound is doing work rather than
        # describing a track that happened to be short.
        assert sum(journal.reads) > 100 * WINDOW
        assert max(journal.writes) <= output_length(WINDOW, 3) + 1

    def test_a_long_gap_costs_no_source_reads(
        self, long_track: tuple[Path, TimelineTrack, int]
    ) -> None:
        """Silence is generated, not read.

        The nineteen minutes past `tx-a`'s extent touch no file at all, which is what makes
        a four-hour session with sparse audio cheap and is why the segment map is the
        working path rather than a materialized file.
        """
        session_dir, track, duration = long_track
        with TrackReader(session_dir, track, duration) as reader:
            tail = reader.read(track.end_sample, WINDOW)
        assert not tail.any()
        assert tail.shape[0] == WINDOW


class TestTheWriterDoesNotBuffer:
    def test_the_declared_length_is_enforced_rather_than_grown(self, tmp_path: Path) -> None:
        """A writer that accepted any length would have to size the header at the end.

        Which would mean either buffering the payload or rewriting the header, and the
        first is the INV-07 violation this whole path exists to avoid.
        """
        from dnd_audio.timeline.wavwrite import WavWriteError

        def over_write() -> None:
            with WavWriter(tmp_path / "short.wav", sample_rate=16000, n_samples=100) as writer:
                writer.write(np.zeros(200, dtype=np.float32))

        with pytest.raises(WavWriteError, match="declared 100 samples"):
            over_write()

    def test_an_incomplete_stream_is_never_published(self, tmp_path: Path) -> None:
        """The half-written file must not survive as a cache hit (INV-08)."""
        from dnd_audio.timeline.wavwrite import WavWriteError

        path = tmp_path / "incomplete.wav"

        def short_write() -> None:
            with WavWriter(path, sample_rate=16000, n_samples=100) as writer:
                writer.write(np.zeros(10, dtype=np.float32))

        with pytest.raises(WavWriteError, match="received 10"):
            short_write()
        assert not path.exists()
        assert not list(tmp_path.glob(".*tmp"))
