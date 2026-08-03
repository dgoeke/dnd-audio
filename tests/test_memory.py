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

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.timeline import Timeline, TimelineSegment, TimelineTrack
from dnd_audio.config import EnvelopeConfig
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.mix.envelope import EnvelopeChunk, EnvelopeStream
from dnd_audio.mix.levels import LevelCorrections, level_corrections
from dnd_audio.mix.render import DEFAULT_MIX_WINDOW, render_mix
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.reader import TrackReader
from dnd_audio.timeline.resample import decimate_stream, output_length
from dnd_audio.timeline.runner import run_ingest
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


@dataclass(frozen=True, slots=True)
class _LongMix:
    """Everything one long-session mix needs, resolved once."""

    session_dir: Path
    timeline: Timeline
    graph: ActivityGraph
    settings: EnvelopeConfig
    corrections: LevelCorrections
    track_ids: tuple[str, ...]


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


class TestTheDetectionPathStreams:
    """The same proof for M3's path, which composes differently and could break differently.

    `detect_track` reads a 16 kHz derivative and hands windows to a detector. Bounding the
    reader says nothing about a detector that keeps every window it is given, or about an
    assembler that materializes the track before deciding anything — and the *whole* point of
    M2's technique is that only an ordered event log over the composed path can tell.
    """

    def test_a_detection_happens_before_the_last_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assertion an accumulating detector cannot pass.

        A detector handed the session one window at a time must have been *called* before the
        reader finished, and one that buffered would be called once, at the end.
        """
        from dnd_audio.activity.detect import detect_track
        from dnd_audio.config import VadConfig
        from dnd_audio.interfaces import ActivityDetector, AudioWindow, SpeechSpan
        from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
        from dnd_audio.timeline.pcm import PcmReader

        # Ten minutes at 16 kHz. Silence is honest here: a scripted detector's answer does not
        # depend on the samples, and what is under test is how many of them are resident.
        duration = 10 * 60 * DERIVATIVE_SAMPLE_RATE
        path = tmp_path / "derivative.wav"
        window = DERIVATIVE_SAMPLE_RATE
        with WavWriter(path, sample_rate=DERIVATIVE_SAMPLE_RATE, n_samples=duration) as writer:
            written = 0
            while written < duration:
                block = min(window, duration - written)
                writer.write(np.zeros(block, dtype=np.float32))
                written += block

        journal = Journal()
        original = PcmReader.read

        def watched_read(self: PcmReader, start: int, n: int) -> npt.NDArray[np.float32]:
            journal.record("read", n)
            return original(self, start, n)

        class Watchful:
            """Records every window it is given, without keeping any of them."""

            def detect(self, audio: AudioWindow) -> tuple[SpeechSpan, ...]:
                journal.record("detect", len(audio))
                return ()

        detector: ActivityDetector = Watchful()
        # Patched by name so the instrumentation sits on the real reader the detection pass
        # opens for itself: there is no seam to inject one through, and that is deliberate —
        # the pass owns its file handle so nothing can hand it a buffered stand-in.
        monkeypatch.setattr("dnd_audio.timeline.pcm.PcmReader.read", watched_read)
        result = detect_track(
            path,
            track_id="tx-a",
            detector=detector,
            settings=VadConfig(),
            window_samples=window,
        )

        detections = [i for i, (kind, _) in enumerate(journal.events) if kind == "detect"]
        assert detections[0] < journal.last_read_index()
        assert max(journal.reads) <= window + 512
        assert sum(journal.reads) == duration
        assert sum(journal.reads) > 100 * window
        # Two bytes per 32 ms frame is the only thing that grows with the session, and it is
        # the artifact that makes a bad attribution debuggable.
        assert result.frame_probabilities.nbytes < duration // 100

    def test_the_bleed_gate_reads_only_what_it_compares(self, tmp_path: Path) -> None:
        """A candidate may be minutes long; a comparison may not.

        `correlation_window_ms` is the bound, and without it one long candidate pulls its
        whole span into memory — six times over, on a host where memory pressure kills
        processes.
        """
        from dnd_audio.activity.bleed import CandidateInput, attribute
        from dnd_audio.config import ActivityConfig
        from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

        minutes = 10 * 60 * DERIVATIVE_SAMPLE_RATE
        reads: list[int] = []

        def read(track_id: str, start: int, n_samples: int) -> npt.NDArray[np.float32]:
            reads.append(n_samples)
            return np.zeros(n_samples, dtype=np.float32)

        candidates = [
            CandidateInput(
                track_id=track,
                start_sample=0,
                end_sample=minutes * 3,
                derivative_start_sample=0,
                derivative_end_sample=minutes,
                probability_permille=900,
                peak_probability_permille=950,
            )
            for track in ("tx-a", "tx-b")
        ]
        config = ActivityConfig()
        attribute(candidates, read=read, config=config)

        cap = config.bleed.correlation_window_ms * DERIVATIVE_SAMPLE_RATE // 1000
        assert reads, "the gate read nothing, so this proves nothing"
        assert max(reads) <= cap
        assert cap < minutes // 100


class TestTheMixPathStreams:
    """The same proof for M5's path, which has **two** things that could accumulate.

    The audio is the obvious one. The gains are the one M5's plan review had to point out: a
    four-hour envelope at 1 kHz over six tracks is 690 MB, and a renderer that built all of it
    first and only then interleaved reads and writes would pass a proof written over the audio
    path alone. So the envelope's own chunk production goes into the same ordered log, and the
    assertion is that a write happens before the *last chunk is produced*.
    """

    @pytest.fixture
    def long_mix(
        self, canonical_fixture: FixtureTruth, canonical_activity_graph: ActivityGraph
    ) -> _LongMix:
        """The real session, answering to twenty minutes instead of ten seconds.

        The same trick `long_track` uses and for the same reason: a track switched off for
        nineteen minutes costs nineteen minutes of mix and no source bytes at all. That is
        what a segment map *is*, so this is an ordinary session shape rather than a
        contrivance.
        """
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        timeline = result.timeline.model_copy(update={"duration_samples": LONG_SESSION_SAMPLES})
        graph = canonical_activity_graph.model_copy(
            update={"duration_samples": LONG_SESSION_SAMPLES}
        )
        settings = EnvelopeConfig()
        return _LongMix(
            session_dir=canonical_fixture.session_dir,
            timeline=timeline,
            graph=graph,
            settings=settings,
            corrections=level_corrections(graph, settings=settings),
            track_ids=tuple(track.track_id for track in timeline.tracks),
        )

    @staticmethod
    def _render(long_mix: _LongMix, destination: Path, journal: Journal, measure: str) -> None:
        """Render the long session with the envelope and the writer both instrumented."""

        class Watched(EnvelopeStream):
            def chunks(self, *, chunk_frames: int) -> Iterator[EnvelopeChunk]:
                for chunk in super().chunks(chunk_frames=chunk_frames):
                    journal.record(
                        "envelope",
                        chunk.n_frames if measure == "frames" else chunk.applied.nbytes,
                    )
                    yield chunk

        original = WavWriter.write

        def watched_write(self: WavWriter, samples: npt.NDArray[np.float32]) -> None:
            journal.record("write", int(samples.shape[0]))
            original(self, samples)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(WavWriter, "write", watched_write)
            render_mix(
                destination,
                session_dir=long_mix.session_dir,
                timeline=long_mix.timeline,
                track_ids=long_mix.track_ids,
                envelope=Watched(
                    long_mix.graph,
                    settings=long_mix.settings,
                    corrections=long_mix.corrections,
                    track_ids=long_mix.track_ids,
                ),
            )

    def test_a_write_happens_before_the_last_envelope_chunk_is_produced(
        self, long_mix: _LongMix, tmp_path: Path
    ) -> None:
        """The assertion neither an accumulating reader nor an accumulating envelope passes.

        Instrumenting only the audio would leave the 690 MB of gains invisible, which is
        exactly the hole M5's plan review found in the first draft of this proof.
        """
        journal = Journal()
        self._render(long_mix, tmp_path / "mix.wav", journal, measure="frames")

        produced = [i for i, (kind, _) in enumerate(journal.events) if kind == "envelope"]
        assert journal.first_write_index() < produced[-1]
        assert sum(journal.writes) == LONG_SESSION_SAMPLES

    def test_no_envelope_chunk_or_write_exceeds_one_window(
        self, long_mix: _LongMix, tmp_path: Path
    ) -> None:
        """The size bound on both paths.

        A window's gains are 1000 frames across six tracks — 48 kB. The whole session's would
        be 57 MB here and 690 MB for four real hours, on the host whose free-space warning
        this pipeline is supposed to reduce.
        """
        journal = Journal()
        self._render(long_mix, tmp_path / "mix.wav", journal, measure="bytes")

        gains = [count for kind, count in journal.events if kind == "envelope"]
        assert max(journal.writes) <= DEFAULT_MIX_WINDOW
        assert max(gains) <= 1000 * len(long_mix.track_ids) * 8
        # Far more than one window in total, so the bounds are doing work rather than
        # describing a session that happened to be short.
        assert len(gains) > 100
        assert sum(journal.writes) > 100 * DEFAULT_MIX_WINDOW
