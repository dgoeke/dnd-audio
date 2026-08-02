"""Per-file capture: what FFprobe said, kept exactly, and read correctly.

The tests that carry the most weight are the two about *bytes*: that the sidecar is
FFprobe's own output rather than a reserialization of it, and that probing the same
session from a different directory produces the same bytes. The second is the one that
would otherwise be discovered in M2, as a manifest that changes when a session is moved.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dnd_audio.determinism import sha256_bytes
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.probe import (
    ProbeError,
    exact_sample_count,
    format_tags,
    parse_probe,
    read_audio_properties,
    run_ffprobe,
    tool_versions,
)
from dnd_audio.inspection.riff import read_inventory


@pytest.fixture
def probed(canonical_fixture: FixtureTruth) -> tuple[FixtureTruth, str, dict[str, object]]:
    """The first tx-a chunk, probed and parsed."""
    source = canonical_fixture.for_track("tx-a")[0].relative_path
    result = run_ffprobe(canonical_fixture.session_dir, source)
    return canonical_fixture, source, parse_probe(result.raw)


class TestCapture:
    def test_every_field_the_gate_names_is_captured(
        self, probed: tuple[FixtureTruth, str, dict[str, object]]
    ) -> None:
        truth, source, document = probed
        chunk = next(c for c in truth.chunks if c.relative_path == source)
        properties = read_audio_properties(document)

        assert properties.codec_name == "pcm_f32le"
        assert properties.sample_format == "flt"
        assert properties.bits_per_sample == 32
        assert properties.sample_rate == 48000
        assert properties.channels == 1
        assert properties.duration_ts == chunk.n_samples
        assert properties.time_base == "1/48000"
        assert properties.duration_text == "3.000000"
        assert properties.block_align == 4

    def test_the_probe_names_the_file_relatively(
        self, probed: tuple[FixtureTruth, str, dict[str, object]]
    ) -> None:
        """The mechanism behind byte-stability across a relocated session."""
        _, source, document = probed
        container = document["format"]
        assert isinstance(container, dict)
        assert container["filename"] == source
        assert not Path(str(container["filename"])).is_absolute()

    def test_format_tags_expose_the_bwf_reference(
        self, probed: tuple[FixtureTruth, str, dict[str, object]]
    ) -> None:
        truth, source, document = probed
        chunk = next(c for c in truth.chunks if c.relative_path == source)
        assert format_tags(document)["time_reference"] == str(chunk.time_reference)

    def test_format_tags_expose_a_timecode_tag(self, canonical_fixture: FixtureTruth) -> None:
        source = canonical_fixture.for_track("tx-f")[0].relative_path
        document = parse_probe(run_ffprobe(canonical_fixture.session_dir, source).raw)
        assert format_tags(document)["timecode"] == "19:00:03:15"

    def test_tags_are_lowercased_so_a_lookup_cannot_miss_one(self) -> None:
        assert format_tags({"format": {"tags": {"Time_Reference": "7"}}}) == {"time_reference": "7"}

    def test_a_document_with_no_tags_is_not_an_error(self) -> None:
        assert format_tags({"format": {}}) == {}
        assert format_tags({}) == {}


class TestVerbatimSidecar:
    def test_the_sidecar_bytes_are_exactly_what_ffprobe_wrote(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Canonical reserialization would change whitespace, key order, and the
        trailing newline. "Verbatim" has to mean the bytes."""
        source = canonical_fixture.for_track("tx-a")[0].relative_path
        result = run_ffprobe(canonical_fixture.session_dir, source)

        assert result.sha256 == sha256_bytes(result.raw)
        assert result.raw != json.dumps(json.loads(result.raw)).encode("utf-8")
        assert json.loads(result.raw)["format"]["filename"] == source

    def test_the_sidecar_name_is_its_own_content_hash(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        source = canonical_fixture.for_track("tx-a")[0].relative_path
        result = run_ffprobe(canonical_fixture.session_dir, source)
        assert result.sidecar_name == f"{sha256_bytes(result.raw)}.json"

    def test_probing_the_same_session_elsewhere_gives_identical_bytes(
        self, canonical_fixture: FixtureTruth, tmp_path: Path
    ) -> None:
        """INV-02's real hazard in this module. If the probe recorded an absolute path,
        moving a session would change its manifest and nothing would explain why."""
        source = canonical_fixture.for_track("tx-b")[0].relative_path
        elsewhere = tmp_path / "moved" / "deeper" / "session"
        shutil.copytree(canonical_fixture.session_dir, elsewhere)

        here = run_ffprobe(canonical_fixture.session_dir, source)
        there = run_ffprobe(elsewhere, source)
        assert here.raw == there.raw
        assert here.sha256 == there.sha256

    def test_two_different_sources_do_not_collide(self, canonical_fixture: FixtureTruth) -> None:
        names = {
            run_ffprobe(canonical_fixture.session_dir, chunk.relative_path).sidecar_name
            for chunk in canonical_fixture.chunks
        }
        assert len(names) == len(canonical_fixture.chunks)


class TestExactSampleCount:
    def test_the_data_chunk_is_preferred_and_duration_ts_confirms_it(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The synthetic half of OQ-011, answered for every file in the fixture."""
        for chunk in canonical_fixture.chunks:
            path = canonical_fixture.session_dir / chunk.relative_path
            inventory = read_inventory(path)
            document = parse_probe(
                run_ffprobe(canonical_fixture.session_dir, chunk.relative_path).raw
            )
            properties = read_audio_properties(document)
            data = inventory.find("data")
            assert data is not None

            count = exact_sample_count(
                data_size=data.size,
                block_align=properties.block_align,
                duration_ts=properties.duration_ts,
            )
            assert count.samples == chunk.n_samples
            assert count.source == "data_chunk"
            assert count.agrees is True, f"{chunk.relative_path}: ffprobe disagrees"

    def test_disagreement_is_recorded_rather_than_hidden(self) -> None:
        count = exact_sample_count(data_size=400, block_align=4, duration_ts=99)
        assert count.samples == 100
        assert count.source == "data_chunk"
        assert count.agrees is False

    def test_duration_ts_is_the_fallback_when_the_data_size_is_unusable(self) -> None:
        count = exact_sample_count(data_size=None, block_align=4, duration_ts=64)
        assert (count.samples, count.source) == (64, "duration_ts")
        assert count.agrees is None, "there is nothing for it to agree with"

    def test_a_ragged_data_size_does_not_produce_a_fractional_count(self) -> None:
        """A `data` size that is not a whole number of frames means something is wrong
        with the file; silently flooring it would invent a sample."""
        count = exact_sample_count(data_size=401, block_align=4, duration_ts=100)
        assert count.source == "duration_ts"

    def test_no_evidence_yields_no_count(self) -> None:
        count = exact_sample_count(data_size=None, block_align=4, duration_ts=None)
        assert count.samples is None
        assert count.source == "none"


class TestFailures:
    def test_a_missing_file_fails_with_ffprobes_own_message(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        with pytest.raises(ProbeError, match="ffprobe failed"):
            run_ffprobe(canonical_fixture.session_dir, "raw/tx-a/absent.wav")

    def test_a_file_that_is_not_audio_fails(self, tmp_path: Path) -> None:
        (tmp_path / "notes.wav").write_text("this is not a wav", encoding="utf-8")
        with pytest.raises(ProbeError):
            run_ffprobe(tmp_path, "notes.wav")

    def test_unparseable_output_is_an_error_not_an_empty_document(self) -> None:
        with pytest.raises(ProbeError, match="not valid JSON"):
            parse_probe(b"{oh no")

    def test_a_json_array_is_rejected(self) -> None:
        with pytest.raises(ProbeError, match="expected an object"):
            parse_probe(b"[]")

    def test_a_document_with_no_audio_stream_is_an_error(self) -> None:
        with pytest.raises(ProbeError, match="no audio stream"):
            read_audio_properties({"streams": [{"codec_type": "video"}]})

    def test_a_missing_sample_rate_is_an_error_not_a_default(self) -> None:
        """Defaulting it would put an invented rate under every timestamp (INV-12)."""
        with pytest.raises(ProbeError, match="sample_rate"):
            read_audio_properties(
                {"streams": [{"codec_type": "audio", "bits_per_sample": 32, "channels": 1}]}
            )

    def test_multiple_audio_streams_are_rejected(self) -> None:
        stream = {"codec_type": "audio", "bits_per_sample": 32, "channels": 1, "sample_rate": 48000}
        with pytest.raises(ProbeError, match="2 audio streams"):
            read_audio_properties({"streams": [stream, dict(stream)]})


class TestToolVersions:
    def test_both_tools_report_a_version(self) -> None:
        versions = tool_versions()
        assert versions.ffmpeg.startswith("ffmpeg version")
        assert versions.ffprobe.startswith("ffprobe version")

    def test_the_two_are_captured_separately(self) -> None:
        """They are separate binaries and can be upgraded independently, so a cache
        identity that recorded only one could serve a stale capture (INV-08)."""
        versions = tool_versions()
        assert versions.ffmpeg != versions.ffprobe
