"""Filename parsing is fail-soft, and yields hints rather than decisions.

The two tests that matter most here are the ones asserting what parsing *cannot* do:
produce a track identity (INV-11) or a timing source (INV-12). The grammar is measured,
while its counter remains non-authoritative (OQ-003), so the tests prove that being wrong
about it costs a hint rather than a file.
"""

from __future__ import annotations

import datetime as dt

import pytest

from dnd_audio.fixtures.session import dji_filename
from dnd_audio.inspection.naming import (
    FilenameHints,
    parse_filename,
    sequence_discontinuities,
)


class TestTheAssumedGrammar:
    def test_a_canonical_name_yields_every_hint(self) -> None:
        hints = parse_filename("TX01_MIC002_20260815_190000_orig.wav")
        assert hints.recognized
        assert hints.tx_label == "TX01"
        assert hints.sequence == 2
        assert hints.variant == "orig"
        assert hints.named_date == dt.date(2026, 8, 15)
        assert hints.named_time == dt.time(19, 0, 0)

    def test_it_reads_what_the_fixture_generator_writes(self) -> None:
        """The writer and the parser were written independently and share no table.

        A round-trip through one shared grammar definition would prove only that the
        definition is self-consistent. This proves the two agree.
        """
        written = dji_filename("TX02", 7, dt.datetime(2026, 8, 15, 19, 4, 5, tzinfo=dt.UTC), "orig")
        hints = parse_filename(written)
        assert hints.recognized
        assert (hints.tx_label, hints.sequence, hints.variant) == ("TX02", 7, "orig")
        assert hints.named_time == dt.time(19, 4, 5)

    def test_an_edit_variant_is_recognized(self) -> None:
        assert parse_filename("TX01_MIC002_20260815_190000_edit.wav").variant == "edit"

    def test_a_longer_counter_still_parses(self) -> None:
        """Nothing establishes that the counter is exactly three digits (OQ-003)."""
        hints = parse_filename("TX01_MIC1024_20260815_190000_orig.wav")
        assert hints.recognized
        assert hints.sequence == 1024


class TestFailSoft:
    @pytest.mark.parametrize(
        "filename",
        [
            "recording.wav",
            "TX1_MIC002_20260815_190000_orig.wav",
            "TX01-MIC002-20260815-190000-orig.wav",
            "DJI_20260815_190000.wav",
            "",
        ],
    )
    def test_an_unrecognized_name_is_still_a_file(self, filename: str) -> None:
        """The failure this prevents: a firmware update renames files, the grammar stops
        matching, and a whole session is silently invisible to a tool whose entire job is
        noticing what is there."""
        hints = parse_filename(filename)
        assert not hints.recognized
        assert hints.tx_label is None
        assert hints.sequence is None

    def test_the_variant_survives_a_name_the_grammar_rejects(self) -> None:
        """Consuming a processed file as if it were the original loses the 32-bit float
        master, so the variant is worth recovering even from a name nothing else parses.
        """
        assert parse_filename("weird-name_edit.wav").variant == "edit"
        assert parse_filename("weird-name_orig.wav").variant == "orig"

    def test_a_name_with_no_variant_marker_is_unknown_not_processed(self) -> None:
        """Single-file mode may write no suffix at all (OQ-007). Guessing `edit` would
        make the only recording unusable; `unknown` lets selection decide with the
        evidence it has."""
        assert parse_filename("recording.wav").variant == "unknown"

    def test_an_impossible_date_does_not_raise(self) -> None:
        hints = parse_filename("TX01_MIC002_20261345_190000_orig.wav")
        assert hints.recognized
        assert hints.named_date is None
        assert hints.named_time == dt.time(19, 0, 0)

    def test_an_impossible_time_does_not_raise(self) -> None:
        hints = parse_filename("TX01_MIC002_20260815_996100_orig.wav")
        assert hints.named_time is None


class TestItCannotBecomeIdentityOrTiming:
    def test_hints_carry_no_track_id(self) -> None:
        """INV-11 structurally: there is no field here that could become one."""
        assert "track_id" not in set(FilenameHints.__slots__)
        assert "speaker_id" not in set(FilenameHints.__slots__)

    def test_the_same_label_appears_on_different_transmitters(self) -> None:
        """OQ-002: identical labels from two kits parse identically, which is exactly
        why the label cannot be the identity."""
        first = parse_filename("TX01_MIC001_20260815_190000_orig.wav")
        second = parse_filename("TX01_MIC001_20260815_190500_orig.wav")
        assert first.tx_label == second.tx_label == "TX01"

    def test_the_named_time_is_not_offered_as_a_start_position(self) -> None:
        """It is a `dt.time`, not samples or a timecode, and nothing converts it.

        The strategy chain in `starttime.py` never reads these fields; the proof that
        matters is over there, where a file with a plausible name and no embedded timing
        is still fatal (INV-12).
        """
        hints = parse_filename("TX01_MIC002_20260815_190000_orig.wav")
        assert isinstance(hints.named_time, dt.time)


class TestSequenceHints:
    def test_a_contiguous_run_has_no_discontinuity(self) -> None:
        hints = [_with_sequence(n) for n in (1, 2, 3)]
        assert sequence_discontinuities(hints) == ()

    def test_a_missing_counter_is_reported_as_a_pair(self) -> None:
        hints = [_with_sequence(n) for n in (1, 2, 5)]
        assert sequence_discontinuities(hints) == ((2, 5),)

    def test_order_of_discovery_does_not_matter(self) -> None:
        """Directory iteration order must not reach a diagnostic (INV-02)."""
        assert sequence_discontinuities([_with_sequence(n) for n in (5, 1, 2)]) == ((2, 5),)

    def test_unrecognized_names_do_not_invent_a_discontinuity(self) -> None:
        hints = [_with_sequence(1), parse_filename("mystery.wav"), _with_sequence(2)]
        assert sequence_discontinuities(hints) == ()


def _with_sequence(sequence: int) -> FilenameHints:
    return parse_filename(f"TX01_MIC{sequence:03d}_20260815_190000_orig.wav")
