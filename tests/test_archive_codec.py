"""Archive v1's recipe is frozen, and its ceilings hold.

The freeze is the point of this file. A content-addressed archive whose compressed bytes
drift with a dependency bump is writing different payloads at keys that claim to identify
content, and nothing downstream would notice — the original digest still matches on
restore, so the corruption is purely in the claim that two uploads of the same file are the
same object.

**The vector is chosen so the freeze can actually fail.** The first one tried here was
250 KB of text-and-noise, and levels 9, 10 and 11 compressed it to byte-identical output —
a freeze that would have passed on a changed level while claiming to pin one. The vector
below produces five distinct outputs across levels 8 to 12, so the assertion has something
to catch. It is generated rather than checked in: no audio or binary fixture belongs in
this repository, and an LCG plus a reuse pool is reproducible anywhere.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest
import zstandard

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.codec import (
    ARCHIVE_CODEC_V1,
    ArchiveCodec,
    compress_bound,
    compress_file,
    decompress_and_measure,
)

#: What `frozen_vector()` produces. Stated here so a change to the generator is a visible
#: change to a checked-in constant rather than a silently different freeze.
VECTOR_SIZE = 2_140_000
VECTOR_SHA256 = "6725698b78a02088f8a4d2174c52f27b4ae1e16ed2ed085a652335b4f5089fe2"

#: What archive v1 compresses that vector to. **If this changes, the recipe changed.**
#: The response is a new archive version (ADR-0037), never editing this line so the gate
#: goes green again — existing objects were written with the old recipe and their keys
#: claim to identify content.
FROZEN_COMPRESSED_SIZE = 121_366
FROZEN_COMPRESSED_SHA256 = "66db1e09ca77ae9d24cc1e71a73ae9b33765a941fb64d18b90a5de3b16d2777a"


def frozen_vector() -> bytes:
    """Deterministic, compressible, and sensitive to the compression level.

    Long-range repeats drawn from a growing pool are what higher levels search harder for,
    so level 8 through 12 all produce different frames. Interleaved counters keep it from
    collapsing into a handful of matches.
    """
    state = 1
    out = bytearray()
    pool: list[bytes] = []
    for i in range(20_000):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        if pool and state % 3:
            out += pool[state % len(pool)]
        else:
            block = bytes((state >> (8 * (j % 3)) & 0xFF) for j in range(97))
            pool.append(block)
            out += block
        out += b" seg%05d " % i
    return bytes(out)


@pytest.fixture
def vector_file(tmp_path: Path) -> Path:
    data = frozen_vector()
    assert len(data) == VECTOR_SIZE, "the generator changed; the frozen digests below are stale"
    assert hashlib.sha256(data).hexdigest() == VECTOR_SHA256
    path = tmp_path / "vector.bin"
    path.write_bytes(data)
    return path


class TestTheRecipeIsFrozen:
    def test_v1_produces_exactly_these_bytes(self, vector_file: Path, tmp_path: Path) -> None:
        """The whole reason archive v1 can content-address anything."""
        fact = compress_file(vector_file, tmp_path / "out.zst")
        assert fact.size_bytes == FROZEN_COMPRESSED_SIZE
        assert fact.sha256 == FROZEN_COMPRESSED_SHA256

    def test_compressing_twice_gives_identical_bytes(
        self, vector_file: Path, tmp_path: Path
    ) -> None:
        first = compress_file(vector_file, tmp_path / "a.zst")
        second = compress_file(vector_file, tmp_path / "b.zst")
        assert first == second
        assert (tmp_path / "a.zst").read_bytes() == (tmp_path / "b.zst").read_bytes()

    @pytest.mark.parametrize("level", [8, 9, 11, 12])
    def test_a_different_level_produces_different_bytes(
        self, level: int, vector_file: Path, tmp_path: Path
    ) -> None:
        """Proves the freeze above can fail, which is the only thing that makes it a freeze.

        Without this, a vector that compressed identically at every level would let the
        recipe drift while `test_v1_produces_exactly_these_bytes` stayed green.
        """
        altered = ArchiveCodec(
            level=level,
            threads=0,
            write_checksum=True,
            write_content_size=True,
            write_dict_id=False,
        )
        fact = compress_file(vector_file, tmp_path / "alt.zst", codec=altered)
        assert fact.sha256 != FROZEN_COMPRESSED_SHA256

    def test_dropping_the_checksum_produces_different_bytes(
        self, vector_file: Path, tmp_path: Path
    ) -> None:
        """Every frame parameter is pinned, not just the level."""
        altered = ArchiveCodec(
            level=10,
            threads=0,
            write_checksum=False,
            write_content_size=True,
            write_dict_id=False,
        )
        fact = compress_file(vector_file, tmp_path / "alt.zst", codec=altered)
        assert fact.sha256 != FROZEN_COMPRESSED_SHA256

    def test_the_recipe_is_single_threaded(self) -> None:
        """`threads=0` is *no worker threads*, and it is not a tuning knob.

        The four-file trial used `-T0`, which means one worker per core and partitions the
        input differently on a 32-core box than on a 4-core one. Both decompress; both are
        wrong for content-addressed storage (ADR-0037).
        """
        assert ARCHIVE_CODEC_V1.threads == 0

    def test_the_description_names_the_library_that_produced_it(self) -> None:
        described = ARCHIVE_CODEC_V1.describe()
        assert described["format"] == "zstd"
        assert described["level"] == 10
        assert described["threads"] == 0
        assert described["libzstd_version"] == ".".join(
            str(part) for part in zstandard.ZSTD_VERSION
        )
        assert described["zstandard_version"] == zstandard.__version__


class TestRoundTrip:
    def test_it_restores_the_original_digest(self, vector_file: Path, tmp_path: Path) -> None:
        original = vector_file.read_bytes()
        compress_file(vector_file, tmp_path / "out.zst")
        with (tmp_path / "out.zst").open("rb") as handle:
            restored = decompress_and_measure(handle, max_output_bytes=len(original))
        assert restored.size_bytes == len(original)
        assert restored.sha256 == hashlib.sha256(original).hexdigest()

    def test_it_can_write_the_bytes_out_as_well_as_measure_them(
        self, vector_file: Path, tmp_path: Path
    ) -> None:
        """Verification discards; restore keeps. One code path, one `sink` argument."""
        compress_file(vector_file, tmp_path / "out.zst")
        sink = io.BytesIO()
        with (tmp_path / "out.zst").open("rb") as handle:
            decompress_and_measure(handle, max_output_bytes=VECTOR_SIZE, sink=sink)
        assert sink.getvalue() == vector_file.read_bytes()

    @pytest.mark.parametrize("size", [0, 1, 1000, 200_000])
    def test_it_handles_small_and_empty_files(self, size: int, tmp_path: Path) -> None:
        """An empty `notes.txt` is a real file in a real session, and must round-trip."""
        source = tmp_path / "small.bin"
        source.write_bytes(os.urandom(size))
        compress_file(source, tmp_path / "small.zst")
        with (tmp_path / "small.zst").open("rb") as handle:
            restored = decompress_and_measure(handle, max_output_bytes=size)
        assert restored.size_bytes == size
        assert restored.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


class TestCeilings:
    def test_a_frame_that_expands_past_its_declared_size_is_stopped(self, tmp_path: Path) -> None:
        """The ceiling is checked per chunk, so a bomb stops *at* the limit.

        Checking only at the end would mean the whole expansion has already been produced —
        and, during a restore, already written — before anything objected.
        """
        bomb = tmp_path / "bomb.bin"
        bomb.write_bytes(b"\x00" * (64 << 20))
        compress_file(bomb, tmp_path / "bomb.zst")

        with (tmp_path / "bomb.zst").open("rb") as handle, pytest.raises(ArchiveError) as caught:
            decompress_and_measure(handle, max_output_bytes=4096)
        assert caught.value.code == "archive_decoded_size_exceeded"

    def test_nothing_is_written_past_the_ceiling(self, tmp_path: Path) -> None:
        """A restore under attack must not fill the disk before failing."""
        bomb = tmp_path / "bomb.bin"
        bomb.write_bytes(b"\x00" * (64 << 20))
        compress_file(bomb, tmp_path / "bomb.zst")

        sink = io.BytesIO()
        with (tmp_path / "bomb.zst").open("rb") as handle, pytest.raises(ArchiveError):
            decompress_and_measure(handle, max_output_bytes=4096, sink=sink)
        assert len(sink.getvalue()) <= 4096 + (1 << 20)

    def test_corrupt_bytes_are_refused_rather_than_returned(self, tmp_path: Path) -> None:
        source = tmp_path / "a.bin"
        source.write_bytes(b"real session audio " * 5000)
        compress_file(source, tmp_path / "a.zst")

        raw = bytearray((tmp_path / "a.zst").read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        with pytest.raises(ArchiveError) as caught:
            decompress_and_measure(io.BytesIO(bytes(raw)), max_output_bytes=source.stat().st_size)
        assert caught.value.code in ("archive_frame_unreadable", "archive_decoded_size_exceeded")

    def test_something_that_is_not_a_frame_at_all_is_refused(self) -> None:
        with pytest.raises(ArchiveError) as caught:
            decompress_and_measure(io.BytesIO(b"this is not zstd"), max_output_bytes=100)
        assert caught.value.code == "archive_frame_unreadable"


class TestCompressBound:
    @pytest.mark.parametrize("size", [0, 1, 1000, 200_000, 5_000_000])
    def test_the_bound_holds_for_incompressible_data(self, size: int, tmp_path: Path) -> None:
        """The case that makes the charter forbid budgeting from the 30.4% saving.

        Already-compressed input comes out slightly *larger*, so a preflight computed from
        an observed ratio passes and then runs out of disk partway through.
        """
        source = tmp_path / "random.bin"
        source.write_bytes(os.urandom(size))
        fact = compress_file(source, tmp_path / "random.zst")
        assert fact.size_bytes > size or size == 0
        assert fact.size_bytes <= compress_bound(size)

    def test_the_bound_holds_for_compressible_data(self, vector_file: Path, tmp_path: Path) -> None:
        fact = compress_file(vector_file, tmp_path / "out.zst")
        assert fact.size_bytes <= compress_bound(VECTOR_SIZE)

    def test_a_negative_size_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            compress_bound(-1)
