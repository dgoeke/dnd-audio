"""The generic RIFF/RF64 walk finds what FFprobe does not, and says when it cannot.

The asymmetry test is the load-bearing one: it runs both `ffprobe` and the walker over
the same bytes and asserts that only one of them sees the private chunk. Without it,
"do not assume FFprobe exposes unknown chunks" is a claim in a document rather than a
property of this code (OQ-005).
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.determinism import sha256_bytes
from dnd_audio.fixtures.wav import (
    RF64_SENTINEL,
    BroadcastMetadata,
    ExtraChunk,
    chunk,
    write_wav,
)
from dnd_audio.inspection.riff import (
    CHUNK_HEADER_BYTES,
    MAX_RETAINED_TEXT_BYTES,
    RiffError,
    read_inventory,
)


@pytest.fixture
def tone() -> np.ndarray:
    return np.zeros(480, dtype=np.float32)


@pytest.fixture
def rich_wav(tmp_path: Path, tone: np.ndarray) -> Path:
    """A file with every shape the walker has to cope with."""
    path = tmp_path / "rich.wav"
    write_wav(
        path,
        tone,
        sample_rate=48000,
        broadcast=BroadcastMetadata(time_reference=123456, description="hello"),
        info={b"ISMP": "19:00:00:00", b"INAM": "Session 01"},
        extra=(ExtraChunk(b"XPRV", bytes(range(64))), ExtraChunk(b"iXML", b"<BWFXML/>")),
        trailing=(ExtraChunk(b"XTRL", b"after the data chunk"),),
    )
    return path


class TestStructure:
    def test_every_top_level_chunk_is_found(self, rich_wav: Path) -> None:
        inventory = read_inventory(rich_wav)
        top = [c.chunk_id for c in inventory.chunks if c.container is None]
        assert top == ["fmt ", "bext", "LIST", "XPRV", "iXML", "data", "XTRL"]
        assert inventory.form == "RIFF"
        assert inventory.form_type == "WAVE"
        assert not inventory.truncated
        assert inventory.warnings == ()

    def test_a_chunk_after_data_is_not_missed(self, rich_wav: Path) -> None:
        """A walker that stops at `data` would silently lose an appended chunk."""
        assert read_inventory(rich_wav).find("XTRL") is not None

    def test_offset_points_at_the_header_not_the_payload(self, rich_wav: Path) -> None:
        blob = rich_wav.read_bytes()
        for found in read_inventory(rich_wav).chunks:
            header = blob[found.offset : found.offset + 4]
            assert header.decode("ascii") == found.chunk_id
            declared = struct.unpack("<I", blob[found.offset + 4 : found.offset + 8])[0]
            assert declared in (found.size, RF64_SENTINEL)

    def test_list_children_are_walked_and_attributed(self, rich_wav: Path) -> None:
        inventory = read_inventory(rich_wav)
        children = {c.chunk_id: c for c in inventory.chunks if c.container == "INFO"}
        assert sorted(children) == ["INAM", "ISMP"]
        assert children["ISMP"].text == "19:00:00:00"

    def test_an_odd_sized_chunk_does_not_shift_the_next_offset(self, tmp_path: Path) -> None:
        """The pad byte is not counted in the size field. Off by one here misaligns
        every chunk that follows, which then fails as a bogus id rather than as a
        length bug — an hour of confusion the assertion prevents."""
        path = tmp_path / "odd.wav"
        write_wav(
            path,
            np.zeros(4, dtype=np.float32),
            sample_rate=48000,
            extra=(ExtraChunk(b"ODD1", b"abc"), ExtraChunk(b"NXT2", b"defg")),
        )
        inventory = read_inventory(path)
        assert [c.chunk_id for c in inventory.chunks] == ["fmt ", "ODD1", "NXT2", "data"]
        assert inventory.find("ODD1") is not None
        assert inventory.find("ODD1").size == 3  # type: ignore[union-attr]
        assert inventory.warnings == ()


class TestHashingAndText:
    def test_a_chunk_hash_covers_the_complete_payload(self, rich_wav: Path) -> None:
        """Not a prefix. A prefix hash presented as a chunk hash would be a lie, and
        M7's archival verification would inherit it."""
        blob = rich_wav.read_bytes()
        found = read_inventory(rich_wav).find("XPRV")
        assert found is not None
        payload_start = found.offset + CHUNK_HEADER_BYTES
        payload = blob[payload_start : payload_start + found.size]
        assert payload == bytes(range(64))
        assert found.sha256 == sha256_bytes(payload)

    def test_a_large_payload_is_still_hashed_in_full(self, tmp_path: Path) -> None:
        """The cap bounds retained text, never hashing."""
        payload = bytes(range(256)) * 64  # 16 KiB, four times the retention cap
        assert len(payload) > MAX_RETAINED_TEXT_BYTES
        path = tmp_path / "big.wav"
        write_wav(
            path,
            np.zeros(4, dtype=np.float32),
            sample_rate=48000,
            extra=(ExtraChunk(b"BIGC", payload),),
        )
        found = read_inventory(path).find("BIGC")
        assert found is not None
        assert found.sha256 == sha256_bytes(payload)
        assert found.text is None

    def test_the_audio_payload_is_inventoried_but_not_hashed(self, rich_wav: Path) -> None:
        """INV-07: the file's own hash already covers those bytes, and a four-hour
        session must not be read twice to learn nothing new."""
        found = read_inventory(rich_wav).find("data")
        assert found is not None
        assert found.size == 480 * 4
        assert found.sha256 is None

    def test_textual_payloads_are_retained(self, rich_wav: Path) -> None:
        found = read_inventory(rich_wav).find("iXML")
        assert found is not None
        assert found.text == "<BWFXML/>"

    def test_binary_payloads_are_not_pretending_to_be_text(self, rich_wav: Path) -> None:
        found = read_inventory(rich_wav).find("XPRV")
        assert found is not None
        assert found.text is None
        assert found.sha256 is not None

    def test_null_padding_does_not_make_a_string_binary(self, tmp_path: Path) -> None:
        path = tmp_path / "padded.wav"
        write_wav(
            path,
            np.zeros(4, dtype=np.float32),
            sample_rate=48000,
            extra=(ExtraChunk(b"NOTE", b"field log\x00\x00\x00\x00"),),
        )
        found = read_inventory(path).find("NOTE")
        assert found is not None
        assert found.text == "field log"


class TestRf64:
    def test_a_sentinel_data_size_is_resolved_through_ds64(self, tmp_path: Path) -> None:
        """Exercising the 64-bit path without a four-gigabyte file."""
        path = tmp_path / "big.wav"
        samples = np.zeros(1200, dtype=np.float32)
        write_wav(path, samples, sample_rate=48000, rf64=True)

        inventory = read_inventory(path)
        assert inventory.form == "RF64"
        assert inventory.warnings == ()
        assert not inventory.truncated
        data = inventory.find("data")
        assert data is not None
        assert data.size == 1200 * 4

        blob = path.read_bytes()
        declared = struct.unpack("<I", blob[data.offset + 4 : data.offset + 8])[0]
        assert declared == RF64_SENTINEL, "the fixture should be exercising the sentinel"

    def test_a_sentinel_with_no_table_entry_stops_the_walk_loudly(self, tmp_path: Path) -> None:
        """The one case where guessing would be tempting and wrong."""
        path = tmp_path / "unresolvable.wav"
        body = (
            b"WAVE"
            + chunk(b"ds64", struct.pack("<QQQI", 0, 16, 4, 0))
            + b"XPRV"
            + struct.pack("<I", RF64_SENTINEL)
            + b"junk"
        )
        path.write_bytes(b"RF64" + struct.pack("<I", RF64_SENTINEL) + body)

        inventory = read_inventory(path)
        assert inventory.truncated
        assert [w.code for w in inventory.warnings] == ["rf64_size_unresolved"]

    def test_a_sentinel_in_a_plain_riff_file_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "confused.wav"
        body = b"WAVE" + b"XPRV" + struct.pack("<I", RF64_SENTINEL) + b"junk"
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)

        inventory = read_inventory(path)
        assert inventory.truncated
        assert [w.code for w in inventory.warnings] == ["sentinel_size_in_riff"]


class TestMalformed:
    def test_a_truncated_chunk_is_recorded_and_the_walk_stops(self, tmp_path: Path) -> None:
        """Recorded, not repaired. Reading past a bad length turns corruption into
        plausible-looking metadata, which is worse than a short inventory."""
        path = tmp_path / "truncated.wav"
        write_wav(path, np.zeros(400, dtype=np.float32), sample_rate=48000)
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) - 600])

        inventory = read_inventory(path)
        assert inventory.truncated
        assert [w.code for w in inventory.warnings] == ["chunk_truncated"]
        assert [c.chunk_id for c in inventory.chunks] == ["fmt "], (
            "chunks before the damage are still valid and should be kept"
        )

    def test_a_nonsense_chunk_id_stops_the_walk(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.wav"
        body = b"WAVE" + chunk(b"fmt ", b"\x00" * 16) + b"\x00\x01\x02\x03" + struct.pack("<I", 4)
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body + b"junk")

        inventory = read_inventory(path)
        assert inventory.truncated
        assert [w.code for w in inventory.warnings] == ["chunk_id_not_ascii"]

    def test_a_file_that_is_not_riff_is_an_error_not_a_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "notawav.wav"
        path.write_bytes(b"OggS" + bytes(64))
        with pytest.raises(RiffError, match="neither RIFF nor RF64"):
            read_inventory(path)

    def test_a_non_wave_riff_file_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "avi.wav"
        path.write_bytes(b"RIFF" + struct.pack("<I", 4) + b"AVI ")
        with pytest.raises(RiffError, match="not WAVE"):
            read_inventory(path)

    def test_an_empty_file_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")
        with pytest.raises(RiffError, match="too short"):
            read_inventory(path)


class TestIndependenceFromFfprobe:
    def test_the_walker_finds_chunks_ffprobe_never_mentions(self, rich_wav: Path) -> None:
        """OQ-005, demonstrated on the same bytes by both tools.

        This is the justification for the module existing. If FFprobe reported private
        chunks, the RIFF walk would be redundant; it does not, so it is not.
        """
        probed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-i",
                rich_wav.name,
            ],
            cwd=rich_wav.parent,
            capture_output=True,
            check=True,
        ).stdout
        document = json.dumps(json.loads(probed))

        assert "XPRV" not in document
        assert "BWFXML" not in document
        assert "XTRL" not in document

        found = {c.chunk_id for c in read_inventory(rich_wav).chunks}
        assert {"XPRV", "iXML", "XTRL"} <= found

    def test_ffprobe_still_agrees_about_the_things_it_does_report(self, rich_wav: Path) -> None:
        """The walker is not reading a different file: both see the bext reference."""
        probed = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-i",
                    rich_wav.name,
                ],
                cwd=rich_wav.parent,
                capture_output=True,
                check=True,
            ).stdout
        )
        assert probed["format"]["tags"]["time_reference"] == "123456"
        assert read_inventory(rich_wav).find("bext") is not None
