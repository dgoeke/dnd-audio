"""The streamed float32 writer: RF64 when needed, atomic always.

Two things are being protected. **INV-07**, because `determinism.write_atomic` holds its
whole payload and a session-length waveform must never be held. **INV-08**, because a
derivative that is published half-written reads as a valid file with silence at the end,
and a cache would then serve it as a hit forever.

The RF64 cases are tested through the header rather than by writing four gigabytes. The
header is where the format decision lives, and it is read back with M1's own RIFF walker —
so the assertion is that this project's writer and this project's parser agree, not that
the bytes match a table written by the same person.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.inspection.riff import read_inventory
from dnd_audio.timeline.pcm import open_pcm
from dnd_audio.timeline.wavwrite import (
    RIFF_SIZE_LIMIT,
    WavWriteError,
    WavWriter,
    needs_rf64,
)


def write(path: Path, samples: np.ndarray, *, sample_rate: int = 48000) -> Path:
    with WavWriter(path, sample_rate=sample_rate, n_samples=int(samples.shape[0])) as writer:
        writer.write(samples)
    return path


class TestRoundTrip:
    def test_what_was_written_is_what_is_read_back(self, tmp_path: Path) -> None:
        """Through this project's own reader, which is what will actually read it."""
        rng = np.random.default_rng(5)
        samples = rng.standard_normal(12345).astype(np.float32)
        path = write(tmp_path / "round.wav", samples)

        source = open_pcm(path)
        assert source.n_samples == 12345
        assert source.sample_rate == 48000
        from dnd_audio.timeline.pcm import PcmReader

        with PcmReader(source) as reader:
            assert np.array_equal(reader.read(0, 12345), samples)

    def test_blocks_are_concatenated_in_order(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(6)
        samples = rng.standard_normal(9000).astype(np.float32)
        path = tmp_path / "blocks.wav"
        with WavWriter(path, sample_rate=16000, n_samples=9000) as writer:
            for start in range(0, 9000, 700):
                writer.write(samples[start : start + 700])

        from dnd_audio.timeline.pcm import PcmReader

        with PcmReader(open_pcm(path)) as reader:
            assert np.array_equal(reader.read(0, 9000), samples)

    def test_an_empty_track_writes_a_valid_header(self, tmp_path: Path) -> None:
        path = write(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))
        assert open_pcm(path).n_samples == 0


class TestRf64:
    def test_the_threshold_is_riffs_own_size_field(self) -> None:
        """Pure arithmetic, so the boundary is testable without a four-gigabyte file."""
        assert not needs_rf64(0)
        assert not needs_rf64(1_000_000)
        # One sample past what a 32-bit RIFF size can describe.
        just_over = (RIFF_SIZE_LIMIT - 4 - 24 - 8) // 4 + 1
        assert not needs_rf64(just_over - 1)
        assert needs_rf64(just_over)

    def test_a_small_file_stays_plain_riff(self, tmp_path: Path) -> None:
        """An RF64 file that did not need to be one is a compatibility problem for free."""
        path = write(tmp_path / "small.wav", np.zeros(1000, dtype=np.float32))
        assert path.read_bytes()[:4] == b"RIFF"
        assert read_inventory(path).form == "RIFF"

    def test_a_large_declared_length_produces_an_rf64_header(self, tmp_path: Path) -> None:
        """Written through the real header path and read with M1's own chunk walker.

        The payload is never written — four hours of six tracks is what this format exists
        for, and materializing it to check a header would be its own INV-07 violation. So
        the file is the header alone, and the walker rightly calls it truncated. What it
        says *about* the truncation is the point: to report that 8 000 000 000 bytes are
        missing, it had to resolve the sentinel `data` size through `ds64`, which is the
        indirection under test. A plain RIFF header could not express that number at all.
        """
        huge = 2_000_000_000  # 8 GB of float32
        assert needs_rf64(huge)
        header = WavWriter(tmp_path / "big.wav", sample_rate=48000, n_samples=huge)._header()

        path = tmp_path / "header-only.wav"
        path.write_bytes(header)
        assert header[:4] == b"RF64"
        assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF

        inventory = read_inventory(path)
        assert inventory.form == "RF64"
        assert inventory.form_type == "WAVE"
        assert inventory.find("ds64") is not None
        assert inventory.find("fmt ") is not None

        assert inventory.truncated
        resolved = [note for note in inventory.warnings if str(huge * 4) in note.message]
        assert resolved, [note.message for note in inventory.warnings]

    def test_the_ds64_sample_count_matches_the_declared_length(self, tmp_path: Path) -> None:
        huge = 2_000_000_000
        header = WavWriter(tmp_path / "big.wav", sample_rate=48000, n_samples=huge)._header()
        offset = header.index(b"ds64") + 8
        riff_size, data_size, sample_count, _ = struct.unpack("<QQQI", header[offset : offset + 28])
        assert data_size == huge * 4
        assert sample_count == huge
        assert riff_size > data_size


class TestAtomicity:
    def test_the_destination_does_not_exist_until_the_stream_completes(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "late.wav"
        with WavWriter(path, sample_rate=16000, n_samples=100) as writer:
            writer.write(np.zeros(100, dtype=np.float32))
            assert not path.exists()
        assert path.exists()

    def test_a_failure_mid_stream_leaves_the_previous_file_intact(self, tmp_path: Path) -> None:
        """A rebuild that dies must not destroy the cached artifact it was replacing."""
        path = tmp_path / "existing.wav"
        write(path, np.full(50, 0.25, dtype=np.float32))
        before = path.read_bytes()

        def dies_midway() -> None:
            with WavWriter(path, sample_rate=16000, n_samples=100) as writer:
                writer.write(np.zeros(10, dtype=np.float32))
                raise RuntimeError("synthetic failure")

        with pytest.raises(RuntimeError, match="synthetic"):
            dies_midway()

        assert path.read_bytes() == before
        assert not list(tmp_path.glob(".*.tmp"))

    def test_writing_past_the_declared_length_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "over.wav"

        def too_many() -> None:
            with WavWriter(path, sample_rate=16000, n_samples=10) as writer:
                writer.write(np.zeros(11, dtype=np.float32))

        with pytest.raises(WavWriteError, match="would exceed it"):
            too_many()
        assert not path.exists()

    def test_writing_outside_the_context_is_refused(self, tmp_path: Path) -> None:
        writer = WavWriter(tmp_path / "closed.wav", sample_rate=16000, n_samples=10)
        with pytest.raises(WavWriteError, match="not open"):
            writer.write(np.zeros(1, dtype=np.float32))

    def test_stereo_input_is_refused(self, tmp_path: Path) -> None:
        def two_channels() -> None:
            with WavWriter(tmp_path / "stereo.wav", sample_rate=16000, n_samples=10) as writer:
                writer.write(np.zeros((5, 2), dtype=np.float32))

        with pytest.raises(ValueError, match="mono"):
            two_channels()


class TestByteStability:
    def test_the_same_samples_produce_the_same_bytes(self, tmp_path: Path) -> None:
        """No timestamps, no padding that varies, nothing from the machine (INV-02).

        Working audio is not on INV-02's list of deterministic artifacts, but a writer
        that produced different bytes for identical input would make the derivative cache
        untestable — and would mean something in the path was reading the clock.
        """
        rng = np.random.default_rng(9)
        samples = rng.standard_normal(4321).astype(np.float32)
        first = write(tmp_path / "a.wav", samples).read_bytes()
        second = write(tmp_path / "b.wav", samples).read_bytes()
        assert first == second

    def test_the_bytes_do_not_depend_on_the_block_sizes(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(10)
        samples = rng.standard_normal(4321).astype(np.float32)

        one_shot = write(tmp_path / "one.wav", samples).read_bytes()
        path = tmp_path / "many.wav"
        with WavWriter(path, sample_rate=48000, n_samples=4321) as writer:
            for start in range(0, 4321, 37):
                writer.write(samples[start : start + 37])
        assert path.read_bytes() == one_shot
