"""The seekable PCM reader and the virtual track built on it.

The PCM reader is where "a source file cannot be decoded" is decided, so its refusals get
the same treatment M1 gave its own: each one is driven with the exact malformed input it
exists to catch, because a guard that has never rejected anything is a guard nobody has
tested.

The `TrackReader` tests are about the three kinds of silence — before a track started,
inside a real gap, and after it stopped — being indistinguishable to a caller. They have to
be: all three mean "this transmitter recorded nothing here", and a consumer that could tell
them apart would be tempted to treat them differently.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.artifacts.timeline import TimelineSegment, TimelineTrack
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.fixtures.wav import ExtraChunk, write_wav
from dnd_audio.timeline.pcm import PcmError, PcmReader, open_pcm
from dnd_audio.timeline.reader import TrackReader
from dnd_audio.timeline.wavwrite import WavWriter

RATE = 48000


def a_wav(path: Path, samples: np.ndarray, **kwargs: object) -> Path:
    write_wav(path, samples, sample_rate=RATE, **kwargs)  # type: ignore[arg-type]
    return path


class TestPcmSourceDiscovery:
    def test_it_finds_the_data_chunk_past_the_metadata(self, tmp_path: Path) -> None:
        """DJI files carry `bext`, `iXML`, and a private chunk before `data`.

        A reader that assumed a fixed header offset would read metadata as audio, which
        sounds like a click and is not obviously a bug.
        """
        rng = np.random.default_rng(1)
        samples = rng.standard_normal(1000).astype(np.float32)
        path = a_wav(
            tmp_path / "meta.wav",
            samples,
            info={b"ISMP": "19:00:00:00"},
            extra=(ExtraChunk(b"XPRV", bytes(range(64))),),
        )
        source = open_pcm(path)
        assert source.n_samples == 1000
        with PcmReader(source) as reader:
            assert np.array_equal(reader.read(0, 1000), samples)

    def test_it_reads_chunks_written_after_data(self, tmp_path: Path) -> None:
        """Some recorders append; the data offset must not depend on being last."""
        rng = np.random.default_rng(2)
        samples = rng.standard_normal(500).astype(np.float32)
        path = a_wav(
            tmp_path / "trailing.wav",
            samples,
            trailing=(ExtraChunk(b"ZZZZ", b"appended"),),
        )
        with PcmReader(open_pcm(path)) as reader:
            assert np.array_equal(reader.read(0, 500), samples)


class TestPcmRefusals:
    def test_a_non_riff_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "not.wav"
        path.write_bytes(b"this is not a RIFF file at all")
        with pytest.raises(Exception, match="neither RIFF nor RF64"):
            open_pcm(path)

    def test_a_file_with_no_data_chunk_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "nodata.wav"
        path.write_bytes(b"RIFF" + struct.pack("<I", 4) + b"WAVE")
        with pytest.raises(PcmError, match="no fmt  and data chunk"):
            open_pcm(path)

    def test_an_integer_pcm_file_is_refused_rather_than_rounded(self, tmp_path: Path) -> None:
        """s32 cannot become float32 exactly, so it is named rather than silently lost."""
        path = tmp_path / "s16.wav"
        fmt = struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 4) + bytes(4)
        )
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)
        with pytest.raises(PcmError, match="32-bit IEEE float") as caught:
            open_pcm(path)
        assert caught.value.code == "undecodable_source"

    def test_an_extensible_format_says_why_it_is_not_supported(self, tmp_path: Path) -> None:
        """Named rather than guessed at: no DJI file has been seen using it (OQ-001)."""
        path = tmp_path / "ext.wav"
        fmt = struct.pack("<HHIIHH", 0xFFFE, 1, RATE, RATE * 4, 4, 32)
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 4) + bytes(4)
        )
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)
        with pytest.raises(PcmError, match="EXTENSIBLE"):
            open_pcm(path)

    def test_a_stereo_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "stereo.wav"
        fmt = struct.pack("<HHIIHH", 3, 2, RATE, RATE * 8, 8, 32)
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 8) + bytes(8)
        )
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)
        with pytest.raises(PcmError, match="2 channels"):
            open_pcm(path)

    def test_a_data_size_that_is_not_whole_samples_is_refused(self, tmp_path: Path) -> None:
        """Flooring would invent the length of a file about to be placed on a timeline."""
        path = tmp_path / "ragged.wav"
        fmt = struct.pack("<HHIIHH", 3, 1, RATE, RATE * 4, 4, 32)
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 6) + bytes(6)
        )
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)
        with pytest.raises(PcmError, match="whole number") as caught:
            open_pcm(path)
        assert caught.value.code == "unknown_sample_count"


class TestPcmReads:
    def test_it_seeks_rather_than_streaming_from_the_start(self, tmp_path: Path) -> None:
        """A window from the middle must equal that slice, not the first N samples."""
        rng = np.random.default_rng(3)
        samples = rng.standard_normal(10000).astype(np.float32)
        path = a_wav(tmp_path / "seek.wav", samples)
        with PcmReader(open_pcm(path)) as reader:
            assert np.array_equal(reader.read(7000, 500), samples[7000:7500])
            assert np.array_equal(reader.read(0, 500), samples[:500])
            # Backwards, to prove it is random access and not a cursor.
            assert np.array_equal(reader.read(3000, 100), samples[3000:3100])

    def test_reading_past_the_end_names_the_file_as_changed(self, tmp_path: Path) -> None:
        """The map guarantees the range fits, so a mismatch means the file moved."""
        path = a_wav(tmp_path / "short.wav", np.zeros(100, dtype=np.float32))
        with (
            PcmReader(open_pcm(path)) as reader,
            pytest.raises(PcmError, match="no longer matches"),
        ):
            reader.read(50, 100)

    def test_a_truncated_file_is_detected_rather_than_zero_filled(self, tmp_path: Path) -> None:
        """Truncation after inspection reads as silence unless someone checks the count."""
        path = a_wav(tmp_path / "cut.wav", np.ones(1000, dtype=np.float32))
        source = open_pcm(path)
        with path.open("r+b") as handle:
            handle.truncate(source.data_offset + 400)
        with PcmReader(source) as reader, pytest.raises(PcmError, match="truncated") as caught:
            reader.read(0, 1000)
        assert caught.value.code == "source_changed"

    def test_reading_outside_a_context_manager_is_refused(self, tmp_path: Path) -> None:
        path = a_wav(tmp_path / "closed.wav", np.zeros(10, dtype=np.float32))
        reader = PcmReader(open_pcm(path))
        with pytest.raises(PcmError, match="not open"):
            reader.read(0, 1)

    def test_an_empty_read_is_empty(self, tmp_path: Path) -> None:
        path = a_wav(tmp_path / "e.wav", np.zeros(10, dtype=np.float32))
        with PcmReader(open_pcm(path)) as reader:
            assert reader.read(0, 0).shape[0] == 0


def _track(session_dir: Path, truth: FixtureTruth, track_id: str) -> TimelineTrack:
    """The fixture's own chunk table, as a timeline track, gaps filled in."""
    chunks = sorted(truth.for_track(track_id), key=lambda c: c.start_sample)
    segments: list[TimelineSegment] = []
    position = chunks[0].start_sample
    for chunk in chunks:
        if chunk.start_sample > position:
            segments.append(
                TimelineSegment(
                    kind="silence",
                    session_start_sample=position,
                    n_samples=chunk.start_sample - position,
                )
            )
        segments.append(
            TimelineSegment(
                kind="audio",
                session_start_sample=chunk.start_sample,
                n_samples=chunk.n_samples,
                source_relative_path=chunk.relative_path,
                source_sha256=chunk.sha256,
                source_start_sample=0,
                evidence_start_sample=chunk.start_sample,
            )
        )
        position = chunk.start_sample + chunk.n_samples
    return TimelineTrack(
        track_id=track_id,
        speaker_id=track_id,
        speaker_name=track_id.upper(),
        start_sample=chunks[0].start_sample,
        end_sample=position,
        segments=segments,
    )


class TestVirtualTrack:
    def test_a_gap_reads_as_silence(self, canonical_fixture: FixtureTruth) -> None:
        """`tx-c`'s three-second hole, read from the map rather than from a file."""
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-c")
        gap = next(segment for segment in track.segments if segment.kind == "silence")
        with TrackReader(canonical_fixture.session_dir, track, track.end_sample) as reader:
            samples = reader.read(gap.session_start_sample, gap.n_samples)
        assert samples.shape[0] == gap.n_samples
        assert not samples.any()

    def test_before_the_track_started_is_silence(self, canonical_fixture: FixtureTruth) -> None:
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-c")
        with TrackReader(canonical_fixture.session_dir, track, track.end_sample) as reader:
            assert not reader.read(0, track.start_sample).any()

    def test_after_the_track_stopped_is_silence(self, canonical_fixture: FixtureTruth) -> None:
        """So a track that ended early still answers to the session's aligned duration."""
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-a")
        duration = track.end_sample + 5 * RATE
        with TrackReader(canonical_fixture.session_dir, track, duration) as reader:
            assert not reader.read(track.end_sample, 5 * RATE).any()
            assert reader.duration_samples == duration

    def test_a_window_spanning_a_chunk_boundary_is_continuous(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """`tx-a`'s two chunks are contiguous, and the join must be invisible.

        The fixture generator seeds its noise floor by *timeline position* rather than per
        chunk, precisely so this can be asserted as exact equality rather than hidden
        inside a tolerance. A reader that mis-set the offset into the second file would
        show a discontinuity here and nowhere else.
        """
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-a")
        boundary = canonical_fixture.for_track("tx-a")[0].n_samples
        with TrackReader(canonical_fixture.session_dir, track, track.end_sample) as reader:
            across = reader.read(boundary - 500, 1000)
            before = reader.read(boundary - 500, 500)
            after = reader.read(boundary, 500)
        assert np.array_equal(across[:500], before)
        assert np.array_equal(across[500:], after)

    def test_windows_reassemble_into_the_whole_track(self, canonical_fixture: FixtureTruth) -> None:
        """At a window size that divides nothing evenly, so no boundary is lucky."""
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-c")
        duration = track.end_sample
        with TrackReader(canonical_fixture.session_dir, track, duration) as reader:
            whole = reader.read(0, duration)
        with TrackReader(canonical_fixture.session_dir, track, duration) as reader:
            joined = np.concatenate([block for _, block in reader.windows(window_samples=7777)])
        assert np.array_equal(joined, whole)
        assert joined.shape[0] == duration

    def test_the_track_reads_the_audio_the_file_holds(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Against the source file directly, so the map's offsets are checked end to end."""
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-e")
        chunk = sorted(canonical_fixture.for_track("tx-e"), key=lambda c: c.start_sample)[1]
        with PcmReader(open_pcm(canonical_fixture.session_dir / chunk.relative_path)) as direct:
            expected = direct.read(0, chunk.n_samples)
        with TrackReader(canonical_fixture.session_dir, track, track.end_sample) as reader:
            found = reader.read(chunk.start_sample, chunk.n_samples)
        assert np.array_equal(found, expected)

    def test_an_invalid_window_size_is_refused(self, canonical_fixture: FixtureTruth) -> None:
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-a")
        with (
            TrackReader(canonical_fixture.session_dir, track, track.end_sample) as reader,
            pytest.raises(ValueError, match="must be positive"),
        ):
            list(reader.windows(window_samples=0))

    def test_writing_a_track_out_round_trips(
        self, canonical_fixture: FixtureTruth, tmp_path: Path
    ) -> None:
        """The `--materialize-48k` path, in miniature: read the map, write it, read it back."""
        track = _track(canonical_fixture.session_dir, canonical_fixture, "tx-c")
        duration = track.end_sample
        path = tmp_path / "tx-c.wav"
        with (
            TrackReader(canonical_fixture.session_dir, track, duration) as reader,
            WavWriter(path, sample_rate=RATE, n_samples=duration) as writer,
        ):
            for _, block in reader.windows(window_samples=4096):
                writer.write(block)
        with TrackReader(canonical_fixture.session_dir, track, duration) as reader:
            expected = reader.read(0, duration)
        with PcmReader(open_pcm(path)) as materialized:
            assert np.array_equal(materialized.read(0, duration), expected)
