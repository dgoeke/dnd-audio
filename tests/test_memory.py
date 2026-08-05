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

import tracemalloc
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.timeline import Timeline, TimelineSegment, TimelineTrack
from dnd_audio.config import EnvelopeConfig
from dnd_audio.fixtures import FixtureTruth, build_session
from dnd_audio.fixtures.variants import mixed_format_session
from dnd_audio.fixtures.wav import write_wav
from dnd_audio.mix.envelope import EnvelopeChunk, EnvelopeStream
from dnd_audio.mix.levels import LevelCorrections, level_corrections
from dnd_audio.mix.render import DEFAULT_MIX_WINDOW, render_mix
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.pcm import PcmReader, open_pcm
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


def _long_track(
    truth: FixtureTruth, track_id: str, speaker: str
) -> tuple[Path, TimelineTrack, int]:
    chunks = sorted(truth.for_track(track_id), key=lambda c: c.start_sample)
    track = TimelineTrack(
        track_id=track_id,
        speaker_id=speaker,
        speaker_name=speaker.title(),
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
    return truth.session_dir, track, LONG_SESSION_SAMPLES


@pytest.fixture
def long_track(canonical_fixture: FixtureTruth) -> tuple[Path, TimelineTrack, int]:
    """`tx-a`'s real chunks, read as a twenty-minute track.

    Everything past the track's own extent is silence, which is exactly what the reader
    returns for a transmitter that stopped early — so this is an ordinary session shape,
    not a special case built for the test.
    """
    return _long_track(canonical_fixture, "tx-a", "alice")


@pytest.fixture
def long_24_bit_track(tmp_path: Path) -> tuple[Path, TimelineTrack, int]:
    """The same shape over a source that has to be *unpacked* rather than viewed.

    NumPy has no packed 24-bit dtype, so decoding is the one place in the working path
    where an implementation naturally reaches for the whole `data` chunk — three bytes per
    sample cannot be reinterpreted in place the way float32 can (ADR-0030). Running the
    composed path over both widths is what makes the bound a property of the *reader*
    rather than of float32's convenient memory layout.
    """
    truth = build_session(mixed_format_session(), tmp_path / "mixed-format")
    return _long_track(truth, "tx-b", "bob")


TRACK_WIDTHS = pytest.mark.parametrize(
    "track_fixture", ["long_track", "long_24_bit_track"], ids=["f32", "s24"]
)


class TestTheWholePathStreams:
    @TRACK_WIDTHS
    def test_a_write_happens_before_the_last_read(
        self, request: pytest.FixtureRequest, track_fixture: str, tmp_path: Path
    ) -> None:
        """The assertion an accumulating implementation cannot pass.

        Reader → resampler → writer, instrumented at both ends. If any stage collected the
        session before handing it on, every read would precede every write and this would
        fail — which is precisely the shape the naive implementation has.
        """
        session_dir, track, duration = request.getfixturevalue(track_fixture)
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

    @TRACK_WIDTHS
    def test_no_single_read_or_write_exceeds_its_window(
        self, request: pytest.FixtureRequest, track_fixture: str
    ) -> None:
        """The size bound, alongside the ordering one.

        A window's worth of float32 is 192 kB; the whole track would be 230 MB, and six of
        them 1.4 GB. On a UMA host that difference is the one that kills a process.
        """
        session_dir, track, duration = request.getfixturevalue(track_fixture)
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


class _CountingHandle:
    """A file handle that remembers how many bytes were actually pulled off the disk."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.total = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        position: int = self._wrapped.seek(offset, whence)
        return position

    def read(self, size: int = -1) -> bytes:
        payload: bytes = self._wrapped.read(size)
        self.total += len(payload)
        return payload

    def close(self) -> None:
        self._wrapped.close()


class TestTwentyFourBitUnpackingStaysInsideTheWindow:
    """INV-07 in the one place the ordered event log above cannot see.

    That log measures what the reader *hands back*, so a `read()` that decoded the entire
    `data` chunk and returned a slice of it would satisfy every assertion in this file.
    Three-byte samples are exactly where that implementation is tempting, because NumPy has
    no packed 24-bit dtype and cannot reinterpret them in place (ADR-0030). So these two
    measure what the log cannot: how many bytes leave the disk, and how much memory the
    unpack allocates.
    """

    #: Twelve megabytes on disk at three bytes a sample — eighty-three windows, and large
    #: enough that expanding it would be unmistakable against the bound below.
    SAMPLES = 4_000_000

    def _big_source(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(24)
        values = rng.integers(-(2**23), 2**23, size=self.SAMPLES, dtype=np.int64).astype(np.int32)
        path = tmp_path / "big-s24.wav"
        write_wav(path, values, sample_rate=CANONICAL_SAMPLE_RATE, sample_format="pcm_s24le")
        return path

    def test_a_window_pulls_only_its_own_bytes_off_the_disk(self, tmp_path: Path) -> None:
        source = open_pcm(self._big_source(tmp_path))
        assert source.n_samples == self.SAMPLES

        with PcmReader(source) as reader:
            # Instrumenting the handle is the only way to see this: everything above the
            # handle has already been narrowed to the window by the time it is observable.
            counted = _CountingHandle(reader._handle)
            reader._handle = cast("BinaryIO", counted)
            reader.read(1_000_000, WINDOW)

        assert counted.total == WINDOW * 3

    def test_the_unpack_allocates_a_window_and_not_a_file(self, tmp_path: Path) -> None:
        """The bound is on the *unpacking*, which the byte count above cannot constrain."""
        with PcmReader(open_pcm(self._big_source(tmp_path))) as reader:
            reader.read(0, WINDOW)  # any one-time cost, outside the measurement
            tracemalloc.start()
            reader.read(1_000_000, WINDOW)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

        # The honest path allocates about 900 kB: the raw bytes, the widened int32, and the
        # float32 result, each one window long.
        limit = 2 * 1024 * 1024
        assert peak < limit
        # And the array a whole-file unpack would have built is several times that, so the
        # bound separates the two implementations rather than merely being generous.
        assert 4 * limit < self.SAMPLES * 4


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
        original_read = TrackReader.read

        def watched_write(self: WavWriter, samples: npt.NDArray[np.float32]) -> None:
            journal.record("write", int(samples.shape[0]))
            original(self, samples)

        def watched_read(self: TrackReader, start: int, count: int) -> npt.NDArray[np.float32]:
            journal.record("read", count)
            return original_read(self, start, count)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(WavWriter, "write", watched_write)
            patch.setattr(TrackReader, "read", watched_read)
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
        exactly the hole M5's plan review found in the first draft of this proof. Instrumenting
        only the *envelope* leaves the six waveforms invisible, which is the hole M5's code
        review found in the second — a renderer that collected every track first and only then
        interleaved lazy envelope chunks with writes would have passed. Both paths are in one
        ordered log and the write has to beat the last event of each.
        """
        journal = Journal()
        self._render(long_mix, tmp_path / "mix.wav", journal, measure="frames")

        produced = [i for i, (kind, _) in enumerate(journal.events) if kind == "envelope"]
        assert journal.first_write_index() < produced[-1]
        assert journal.first_write_index() < journal.last_read_index()
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
        reads = [count for kind, count in journal.events if kind == "read"]
        assert max(journal.writes) <= DEFAULT_MIX_WINDOW
        assert reads, "the mix read nothing, so bounding the reads proves nothing"
        assert max(reads) <= DEFAULT_MIX_WINDOW
        assert max(gains) <= 1000 * len(long_mix.track_ids) * 8
        # Far more than one window in total, so the bounds are doing work rather than
        # describing a session that happened to be short.
        assert len(gains) > 100
        assert sum(journal.writes) > 100 * DEFAULT_MIX_WINDOW


class TestTheMarkerAnalysisPathStreams:
    """INV-07 over M10's composed path, where two things can accumulate rather than one.

    The correlator is the obvious half and the easy one: fixed blocks with a template-length
    carry, so the working set is a property of the block size rather than of how much audio
    was asked for. Proving *that* alone is the trap M10's second plan review caught — the
    analyzer also retains every accepted occurrence, and non-maximum suppression bounds
    nearby candidates while saying nothing about the number of separated ones. A longer
    *sparse* search passes a read-size assertion while those lists grow without limit.

    So both halves are asserted: reads stay bounded and interleaved with work (M2's ordered
    event log), and a **dense** input hits the versioned occurrence ceiling and fails rather
    than truncating.
    """

    def test_correlation_happens_before_the_last_read(self) -> None:
        """An implementation that buffered the search range would correlate once, at the end."""
        from dnd_audio.marker.detect import detect_occurrences
        from dnd_audio.marker.spec import MARKER_SPECS
        from dnd_audio.marker.synth import marker_samples

        spec = MARKER_SPECS["cand-a"]
        journal = Journal()
        marker = marker_samples(spec).astype(np.float32) / 32768.0
        # Five minutes at 48 kHz, with the marker once near the start.
        duration = 5 * 60 * 48_000
        track = np.zeros(duration, dtype=np.float32)
        track[100_000 : 100_000 + marker.size] = marker

        class Watched:
            def read(self, start: int, count: int, /) -> npt.NDArray[np.float32]:
                journal.record("read", count)
                window = np.zeros(count, dtype=np.float32)
                low, high = max(0, start), min(duration, start + count)
                if high > low:
                    window[low - start : high - start] = track[low:high]
                return window

        import dnd_audio.marker.detect as detect_module

        original = detect_module._normalized_scores

        def watched_scores(
            signal: npt.NDArray[np.float64], template: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            journal.record("write", int(signal.size))
            return original(signal, template)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(detect_module, "_normalized_scores", watched_scores)
            found = detect_occurrences(Watched(), spec, interval=(0, duration))

        assert [item.anchor_sample for item in found] == [100_000 + spec.anchor_sample]
        assert journal.first_write_index() < journal.last_read_index(), (
            "every correlation happened after the last read, so the whole search range was "
            "resident at once"
        )

    def test_the_largest_read_is_a_property_of_the_block_not_of_the_range(self) -> None:
        """Ten times the audio, the same peak read."""
        from dnd_audio.marker.detect import BLOCK_SAMPLES, detect_occurrences
        from dnd_audio.marker.spec import MARKER_SPECS

        spec = MARKER_SPECS["cand-a"]
        longest = max(chirp.duration_samples for chirp in spec.chirps)

        class Silent:
            """Bound to its own journal, so the closure cannot capture a loop variable."""

            def __init__(self, journal: Journal) -> None:
                self.journal = journal

            def read(self, start: int, count: int, /) -> npt.NDArray[np.float32]:
                self.journal.record("read", count)
                return np.zeros(count, dtype=np.float32)

        peaks = []
        for minutes in (1, 10):
            duration = minutes * 60 * 48_000
            journal = Journal()
            detect_occurrences(Silent(journal), spec, interval=(0, duration))
            peaks.append(max(journal.reads))
            assert len(journal.reads) > minutes, "the range was not read in blocks at all"

        assert peaks[0] == peaks[1]
        assert peaks[0] <= BLOCK_SAMPLES + longest

    def test_dense_occurrences_fail_rather_than_accumulating(self) -> None:
        """The half a sparse long search cannot see (second plan review, P0-2).

        The ceiling is what bounds the *retained* set, and it fails explicitly: a truncated
        occurrence list is indistinguishable from a session that genuinely had that many.
        """
        from dnd_audio.marker.detect import (
            DetectorThresholds,
            OccurrenceCeilingError,
            detect_occurrences,
        )
        from dnd_audio.marker.spec import MARKER_SPECS
        from dnd_audio.marker.synth import marker_samples

        spec = MARKER_SPECS["cand-a"]
        marker = marker_samples(spec).astype(np.float32) / 32768.0
        stride = marker.size + 20_000
        count = 40
        track = np.zeros(count * stride + 10_000, dtype=np.float32)
        for index in range(count):
            track[index * stride : index * stride + marker.size] = marker

        class Dense:
            last_read_end = 0

            def read(self, start: int, n: int, /) -> npt.NDArray[np.float32]:
                self.last_read_end = max(self.last_read_end, start + n)
                window = np.zeros(n, dtype=np.float32)
                low, high = max(0, start), min(track.size, start + n)
                if high > low:
                    window[low - start : high - start] = track[low:high]
                return window

        reader = Dense()
        with pytest.raises(OccurrenceCeilingError):
            detect_occurrences(
                reader,
                spec,
                interval=(0, track.size),
                thresholds=DetectorThresholds(max_occurrences_per_track=8),
            )
        assert reader.last_read_end < track.size, "the ceiling was checked only after the last read"

    def test_dense_partial_candidates_also_fail_before_the_last_read(self) -> None:
        """A ceiling on complete occurrences cannot bound a long series of isolated chirps."""
        from dnd_audio.marker.detect import (
            DetectorThresholds,
            OccurrenceCeilingError,
            detect_occurrences,
        )
        from dnd_audio.marker.spec import MARKER_SPECS
        from dnd_audio.marker.synth import marker_templates

        spec = MARKER_SPECS["v1"]
        chirp = marker_templates(spec)[0].astype(np.float32) / 32768.0
        stride = 20_000
        count = 40
        track = np.zeros(count * stride + 10_000, dtype=np.float32)
        for index in range(count):
            track[index * stride : index * stride + chirp.size] = chirp

        class DensePartials:
            last_read_end = 0

            def read(self, start: int, n: int, /) -> npt.NDArray[np.float32]:
                self.last_read_end = max(self.last_read_end, start + n)
                window = np.zeros(n, dtype=np.float32)
                low, high = max(0, start), min(track.size, start + n)
                if high > low:
                    window[low - start : high - start] = track[low:high]
                return window

        reader = DensePartials()
        with pytest.raises(OccurrenceCeilingError):
            detect_occurrences(
                reader,
                spec,
                interval=(0, track.size),
                thresholds=DetectorThresholds(max_peak_candidates_per_chirp=8),
            )
        assert reader.last_read_end < track.size, "candidate peaks accumulated to the final read"
