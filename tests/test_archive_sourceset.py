"""The archive's own inventory, and the encoding that turns it into object keys.

Two properties are asserted here that no other test in the project covers:

* **Every irreplaceable file is in the set** — including ones inspection has no opinion
  about, and specifically including `raw/tx-a/work/notes.txt`, which INV-01's snapshot was
  once silently dropping (M2's verify phase).
* **Nothing outside the session can get in.** A symlink is refused at every component, not
  filtered afterwards, because a filter over `rglob` never sees inside a symlinked
  directory at all.
"""

from __future__ import annotations

import os

import pytest

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.paths import (
    decode_component,
    encode_component,
    key_length_bytes,
    require_key_within_limit,
)
from dnd_audio.archive.sourceset import (
    ArchiveEntry,
    ArchiveSourceSet,
    build_source_set,
    object_key,
)
from dnd_audio.config import load_session_config
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.raw_guard import raw_roots, snapshot


def source_set(fixture: FixtureTruth) -> ArchiveSourceSet:
    """Build the set for a fixture, reading its configuration the way a command would."""
    config = load_session_config(fixture.session_dir / "session.yaml")
    return build_source_set(fixture.session_dir, config)


#: One entry with a plausible digest, for the key tests that need no file on disk.
DETACHED_ENTRY = ArchiveEntry(relative_path="raw/tx-a/x.wav", size_bytes=1, sha256="0" * 64)


class TestEncoding:
    """Reversible over bytes, canonical, and ASCII whatever the filename was."""

    @pytest.mark.parametrize(
        "original",
        [
            "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav",
            "raw/tx-a/a file with spaces.wav",
            "raw/tx-a/100% real.wav",
            "raw/tx-a/line\nbreak.wav",
            "raw/tx-a/naïve.wav",
            "raw/tx-a/テスト.wav",
            "raw/tx-a/é-combining.wav",
            "raw/tx-a/é-precomposed.wav",
            "Session 01",
            "session/with/slashes",
        ],
    )
    def test_it_round_trips(self, original: str) -> None:
        assert decode_component(encode_component(original)) == original

    def test_it_round_trips_a_filename_that_is_not_valid_utf8(self) -> None:
        """The case that makes this encode bytes rather than text.

        Linux permits it, Python surfaces it as a surrogate escape, and `canonical_json`
        raises on a surrogate — so a text-based encoding could not put this file in a
        manifest at all, and the crash would come partway through an upload.
        """
        original = os.fsdecode(b"raw/tx-a/broken-\xff\xfe.wav")
        encoded = encode_component(original)
        assert encoded.isascii()
        assert "%FF%FE" in encoded
        assert decode_component(encoded) == original

    def test_every_encoded_component_is_ascii_and_holds_no_separator(self) -> None:
        """A whole path collapses into one opaque component; nothing reads as a directory."""
        encoded = encode_component("raw/tx-a/nested/deep.wav")
        assert encoded.isascii()
        assert "/" not in encoded
        assert "%2F" in encoded

    def test_two_unicode_normalizations_stay_distinct(self) -> None:
        """Because on disk they are two different files, and both must be restorable.

        Built from escapes rather than typed literally. The two forms are visually
        identical, so a literal version asserts something the reader cannot see — and an
        editor or tool that normalizes on save would quietly turn it into a tautology
        that passes forever.
        """
        precomposed = "\u00e9.wav"
        combining = "e\u0301.wav"
        assert precomposed != combining

        assert encode_component(precomposed) == "%C3%A9.wav"
        assert encode_component(combining) == "e%CC%81.wav"
        assert decode_component(encode_component(precomposed)) == precomposed
        assert decode_component(encode_component(combining)) == combining

    def test_lowercase_hex_is_refused_rather_than_accepted(self) -> None:
        """`%2f` and `%2F` decode alike and are different keys.

        Accepting both would mean the same file has two valid object keys, which is not
        what "content-addressed" can mean.
        """
        with pytest.raises(ArchiveError) as caught:
            decode_component("raw%2ftx-a")
        assert caught.value.code == "archive_key_not_canonical"

    def test_an_unescaped_byte_that_should_have_been_escaped_is_refused(self) -> None:
        with pytest.raises(ArchiveError):
            decode_component("raw/tx-a")

    def test_a_non_ascii_key_is_refused(self) -> None:
        with pytest.raises(ArchiveError):
            decode_component("naïve")


class TestKeyLength:
    def test_the_limit_is_counted_in_utf8_bytes(self) -> None:
        assert key_length_bytes("abc") == 3
        assert key_length_bytes("%C3%A9") == 6

    def test_a_key_within_the_limit_passes_through(self) -> None:
        assert require_key_within_limit("a" * 1024, subject="x") == "a" * 1024

    def test_an_over_long_key_is_refused_and_never_truncated(self) -> None:
        """Truncating would make two long paths sharing a prefix collide into one key.

        The second upload would then overwrite the first while both manifests reported
        success — a silent loss of exactly the thing being backed up.
        """
        with pytest.raises(ArchiveError) as caught:
            require_key_within_limit("a" * 1025, subject="raw/tx-a/very-long.wav")
        assert caught.value.code == "archive_key_too_long"
        assert "raw/tx-a/very-long.wav" in str(caught.value)

    def test_encoding_expansion_can_reach_the_limit(self) -> None:
        """Not a hypothetical bound: 400 non-ASCII characters already exceed it."""
        long_name = "raw/tx-a/" + ("é" * 400) + ".wav"
        assert key_length_bytes(encode_component(long_name)) > 1024


class TestWhatIsIncluded:
    def test_it_finds_every_source_the_fixture_wrote(self, canonical_fixture: FixtureTruth) -> None:
        found = source_set(canonical_fixture)
        assert found.entries
        assert found.total_bytes > 0
        assert all(entry.size_bytes > 0 for entry in found.entries)
        assert all(len(entry.sha256) == 64 for entry in found.entries)

    def test_it_agrees_exactly_with_the_inv01_snapshot(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """On a tree with no symlinks the two enumerations must not diverge.

        They are separate implementations on purpose (ADR-0036), which means they are two
        things that can drift. This is the test that makes a drift loud. On a tree *with* a
        symlink they legitimately differ, and that difference is fatal on this side rather
        than reconciled — see `TestSymlinksAreRefused`.
        """
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        expected = snapshot(canonical_fixture.session_dir, raw_roots(config))
        found = source_set(canonical_fixture)
        assert {
            entry.relative_path: (entry.sha256, entry.size_bytes) for entry in found.entries
        } == expected

    def test_it_includes_a_non_audio_file_nobody_configured(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Irreplaceable is the criterion, not "audio the pipeline reads"."""
        notes = canonical_fixture.session_dir / "raw" / "field-notes.txt"
        notes.write_text("battery swap at 20:14\n", encoding="utf-8")
        paths = [entry.relative_path for entry in source_set(canonical_fixture).entries]
        assert "raw/field-notes.txt" in paths

    def test_it_includes_a_nested_directory_named_work(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The carve-out is at the session root **only**.

        M2's verify phase found INV-01's snapshot excluding any component named `work` at
        any depth, which left `raw/tx-a/work/notes.txt` unhashed and freely mutable. The
        archive would have left it unbacked-up, which is the same defect costing more.
        """
        nested = canonical_fixture.session_dir / "raw" / "tx-a" / "work"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "notes.txt").write_text("do not lose me\n", encoding="utf-8")
        paths = [entry.relative_path for entry in source_set(canonical_fixture).entries]
        assert "raw/tx-a/work/notes.txt" in paths

    def test_it_excludes_the_sessions_own_generated_directories(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        found = source_set(canonical_fixture)
        assert not any(
            entry.relative_path.startswith(("work/", "output/")) for entry in found.entries
        )

    def test_entries_are_in_deterministic_path_order(self, canonical_fixture: FixtureTruth) -> None:
        paths = [entry.relative_path for entry in source_set(canonical_fixture).entries]
        assert paths == sorted(paths)


class TestTrackAttribution:
    def test_a_track_directorys_audio_is_attributed(self, canonical_fixture: FixtureTruth) -> None:
        attributed = source_set(canonical_fixture).for_track("tx-a")
        assert attributed
        assert all(entry.relative_path.startswith("raw/tx-a/") for entry in attributed)

    def test_a_file_outside_every_track_directory_stays_unassigned(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-11: an unconfigured location is never attributed to a speaker."""
        (canonical_fixture.session_dir / "raw" / "stray.wav").write_bytes(b"RIFF" + b"\x00" * 64)
        found = source_set(canonical_fixture)
        stray = next(e for e in found.entries if e.relative_path == "raw/stray.wav")
        assert stray.track_id is None

    def test_whole_session_covers_what_track_scope_cannot(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The documented ergonomic edge, asserted so it stays documented.

        `--track` under-recovering is the honest cost of not inventing identity; a track
        restore quietly containing an unassigned file would be the dishonest alternative.
        """
        (canonical_fixture.session_dir / "raw" / "stray.wav").write_bytes(b"RIFF" + b"\x00" * 64)
        found = source_set(canonical_fixture)
        every_track = {
            entry.relative_path
            for track_id in {e.track_id for e in found.entries if e.track_id}
            for entry in found.for_track(str(track_id))
        }
        assert "raw/stray.wav" in {e.relative_path for e in found.entries}
        assert "raw/stray.wav" not in every_track


class TestSymlinksAreRefused:
    def test_a_symlinked_file_is_refused(self, canonical_fixture: FixtureTruth) -> None:
        """`raw_guard.snapshot` follows this one, which is why the archive re-implements."""
        secret = canonical_fixture.session_dir.parent / "private-key"
        secret.write_text("not session data\n", encoding="utf-8")
        (canonical_fixture.session_dir / "raw" / "innocent.wav").symlink_to(secret)
        with pytest.raises(ArchiveError) as caught:
            source_set(canonical_fixture)
        assert caught.value.code == "archive_symlink_refused"

    def test_a_symlinked_directory_is_refused(self, canonical_fixture: FixtureTruth) -> None:
        """The one a post-hoc filter cannot catch: `rglob` never descends into it."""
        elsewhere = canonical_fixture.session_dir.parent / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "secret.txt").write_text("not session data\n", encoding="utf-8")
        (canonical_fixture.session_dir / "raw" / "linked").symlink_to(elsewhere)
        with pytest.raises(ArchiveError) as caught:
            source_set(canonical_fixture)
        assert caught.value.code == "archive_symlink_refused"

    def test_a_symlink_deep_inside_a_track_is_refused(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        target = canonical_fixture.session_dir.parent / "target.txt"
        target.write_text("x\n", encoding="utf-8")
        nested = canonical_fixture.session_dir / "raw" / "tx-a" / "sub"
        nested.mkdir(parents=True)
        (nested / "link.wav").symlink_to(target)
        with pytest.raises(ArchiveError) as caught:
            source_set(canonical_fixture)
        assert caught.value.code == "archive_symlink_refused"

    def test_the_refusal_names_the_offending_path(self, canonical_fixture: FixtureTruth) -> None:
        target = canonical_fixture.session_dir.parent / "target.txt"
        target.write_text("x\n", encoding="utf-8")
        (canonical_fixture.session_dir / "raw" / "obvious.wav").symlink_to(target)
        with pytest.raises(ArchiveError) as caught:
            source_set(canonical_fixture)
        assert "raw/obvious.wav" in str(caught.value)


class TestIrregularFiles:
    def test_a_fifo_is_refused_rather_than_silently_skipped(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Skipping it would make "the archive holds everything" quietly false."""
        os.mkfifo(canonical_fixture.session_dir / "raw" / "pipe")
        with pytest.raises(ArchiveError) as caught:
            source_set(canonical_fixture)
        assert caught.value.code == "archive_irregular_file"


class TestVerifyUnchanged:
    def test_an_unchanged_tree_verifies(self, canonical_fixture: FixtureTruth) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        found = build_source_set(canonical_fixture.session_dir, config)
        found.verify_unchanged(config)

    def test_a_modified_source_is_caught(self, canonical_fixture: FixtureTruth) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        found = build_source_set(canonical_fixture.session_dir, config)
        target = canonical_fixture.session_dir / found.entries[0].relative_path
        target.write_bytes(target.read_bytes() + b"\x00")
        with pytest.raises(ArchiveError) as caught:
            found.verify_unchanged(config)
        assert caught.value.code == "archive_sources_modified"
        assert "modified" in str(caught.value)

    def test_an_appeared_file_is_caught(self, canonical_fixture: FixtureTruth) -> None:
        """A set that grew mid-upload was not the set that was hashed."""
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        found = build_source_set(canonical_fixture.session_dir, config)
        (canonical_fixture.session_dir / "raw" / "late-arrival.txt").write_text(
            "appeared\n", encoding="utf-8"
        )
        with pytest.raises(ArchiveError) as caught:
            found.verify_unchanged(config)
        assert "appeared" in str(caught.value)

    def test_a_removed_file_is_caught(self, canonical_fixture: FixtureTruth) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        found = build_source_set(canonical_fixture.session_dir, config)
        (canonical_fixture.session_dir / found.entries[0].relative_path).unlink()
        with pytest.raises(ArchiveError) as caught:
            found.verify_unchanged(config)
        assert "removed" in str(caught.value)


class TestObjectKeys:
    def test_a_key_names_its_session_its_path_and_its_digest(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        found = source_set(canonical_fixture)
        entry = found.entries[0]
        key = object_key("session-2026-08-15", entry)
        assert key.startswith("sessions/archive-v1/session-2026-08-15/objects/")
        assert key.endswith(f".{entry.sha256}.zst")
        assert decode_component(key.split("/")[-1].rsplit(".", 2)[0]) == entry.relative_path

    def test_the_same_bytes_at_the_same_path_give_the_same_key(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Which is what makes a re-upload idempotent rather than a second copy."""
        found = source_set(canonical_fixture)
        again = source_set(canonical_fixture)
        assert [object_key("s", e) for e in found.entries] == [
            object_key("s", e) for e in again.entries
        ]

    def test_changed_bytes_give_a_different_key_rather_than_overwriting(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        found = source_set(canonical_fixture)
        entry = found.entries[0]
        target = canonical_fixture.session_dir / entry.relative_path
        target.write_bytes(target.read_bytes() + b"\x00")
        changed = source_set(canonical_fixture).entries[0]
        assert object_key("s", changed) != object_key("s", entry)

    def test_a_session_id_with_a_space_becomes_a_key_rather_than_a_refusal(self) -> None:
        """`SessionConfig` accepts any non-empty string, and narrowing it was refused.

        The plan review's finding: constraining `session_id` would move every processing
        cache identity and make an already-inspected `Session 01` unarchivable.
        """
        key = object_key("Session 01", DETACHED_ENTRY)
        assert "Session%2001" in key
        assert decode_component(key.split("/")[2]) == "Session 01"
