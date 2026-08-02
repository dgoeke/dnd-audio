"""INV-04: rate labels map to exact rationals, and bad drop-frame syntax is rejected.

Every expectation here is written as a :class:`~fractions.Fraction`. Comparing against
``29.97`` would assert the very thing the invariant forbids.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from dnd_audio.config import FrameRateLabel
from dnd_audio.errors import TimecodeError
from dnd_audio.timecode import (
    FRAME_RATE_LABELS,
    FRAME_RATES,
    parse_frame_rate,
    parse_timecode,
)

# The spec's table, transcribed. If DJI adds a rate, it is added here first.
_EXPECTED: dict[str, tuple[Fraction, bool]] = {
    "23.98F": (Fraction(24000, 1001), False),
    "24F": (Fraction(24, 1), False),
    "25F": (Fraction(25, 1), False),
    "29.97F": (Fraction(30000, 1001), False),
    "29.97DF": (Fraction(30000, 1001), True),
    "30F": (Fraction(30, 1), False),
    "50F": (Fraction(50, 1), False),
    "60F": (Fraction(60, 1), False),
}


class TestFrameRates:
    @pytest.mark.parametrize(("label", "expected"), sorted(_EXPECTED.items()))
    def test_label_maps_to_the_exact_rate(
        self, label: str, expected: tuple[Fraction, bool]
    ) -> None:
        rate, drop_frame = expected
        parsed = parse_frame_rate(label)
        assert parsed.rate == rate
        assert parsed.drop_frame is drop_frame

    def test_fractional_rates_are_not_floats(self) -> None:
        """A binary float cannot represent 30000/1001, and INV-04 forbids trying."""
        rate = parse_frame_rate("29.97F").rate
        assert isinstance(rate, Fraction)
        assert rate != Fraction(2997, 100)
        assert rate * 1001 == 30000

    def test_drop_frame_and_non_drop_share_a_rate(self) -> None:
        """29.97DF is a labelling scheme, not a different speed."""
        assert parse_frame_rate("29.97DF").rate == parse_frame_rate("29.97F").rate

    @pytest.mark.parametrize(
        ("label", "expected"),
        [("23.98F", 24), ("29.97F", 30), ("29.97DF", 30), ("30F", 30), ("60F", 60)],
    )
    def test_frame_index_modulus_is_the_ceiling(self, label: str, expected: int) -> None:
        """Timecode counts 30 frames in a 29.97 second. That gap is why drop-frame exists."""
        assert parse_frame_rate(label).frames_per_timecode_second == expected

    def test_every_label_is_covered_by_this_test(self) -> None:
        assert set(FRAME_RATES) == set(_EXPECTED)

    def test_config_literal_matches_the_rate_table(self) -> None:
        """The configuration's allowed labels and the rate table cannot drift apart."""
        from typing import get_args

        assert set(get_args(FrameRateLabel)) == set(FRAME_RATE_LABELS)

    def test_unknown_label_is_fatal(self) -> None:
        with pytest.raises(TimecodeError, match="unknown frame rate"):
            parse_frame_rate("29.97")


class TestTimecodeSyntax:
    def test_non_drop_timecode_parses(self) -> None:
        parsed = parse_timecode("19:00:00:00", parse_frame_rate("30F"))
        assert (parsed.hours, parsed.minutes, parsed.seconds, parsed.frames) == (19, 0, 0, 0)

    @pytest.mark.parametrize("separator", [";", "."])
    def test_drop_frame_separator_at_a_non_drop_rate_is_rejected(self, separator: str) -> None:
        """The spec's named case: the notation and the rate contradict each other."""
        with pytest.raises(TimecodeError, match="drop-frame notation"):
            parse_timecode(f"19:00:00{separator}00", parse_frame_rate("30F"))

    def test_drop_frame_separator_at_a_drop_frame_rate_is_accepted(self) -> None:
        assert parse_timecode("19:10:00;00", parse_frame_rate("29.97DF")).frames == 0

    def test_colon_at_a_drop_frame_rate_is_accepted(self) -> None:
        """Common spelling, and unambiguous — unlike ';' at 30F."""
        assert parse_timecode("19:10:00:00", parse_frame_rate("29.97DF")).frames == 0

    @pytest.mark.parametrize(
        "text",
        ["19:00:00", "1:00:00:00", "19:00:00:0", "19-00-00-00", "", "19:00:00:00:00"],
    )
    def test_malformed_syntax_is_rejected(self, text: str) -> None:
        with pytest.raises(TimecodeError, match="malformed timecode"):
            parse_timecode(text, parse_frame_rate("30F"))

    @pytest.mark.parametrize(
        ("text", "field"),
        [
            ("24:00:00:00", "hours"),
            ("19:60:00:00", "minutes"),
            ("19:00:60:00", "seconds"),
            ("19:00:00:30", "frames"),
        ],
    )
    def test_out_of_range_fields_are_rejected(self, text: str, field: str) -> None:
        with pytest.raises(TimecodeError, match=field):
            parse_timecode(text, parse_frame_rate("30F"))

    def test_frame_index_is_bounded_by_the_rate(self) -> None:
        """24 frames per second means 00..23, and 23.98F is still 24 labels."""
        rate = parse_frame_rate("23.98F")
        assert parse_timecode("19:00:00:23", rate).frames == 23
        with pytest.raises(TimecodeError, match="frames=24"):
            parse_timecode("19:00:00:24", rate)

    @pytest.mark.parametrize("text", ["19:01:00;00", "19:01:00;01", "19:59:00;00"])
    def test_drop_frame_skipped_labels_are_rejected(self, text: str) -> None:
        """Frames 00 and 01 do not exist at the start of most minutes at 29.97DF."""
        with pytest.raises(TimecodeError, match="skips"):
            parse_timecode(text, parse_frame_rate("29.97DF"))

    @pytest.mark.parametrize(
        ("text", "minute"), [("19:00:00;00", 0), ("19:10:00;01", 10), ("19:20:00;00", 20)]
    )
    def test_drop_frame_keeps_every_tenth_minute(self, text: str, minute: int) -> None:
        assert parse_timecode(text, parse_frame_rate("29.97DF")).minutes == minute

    def test_drop_frame_label_exists_later_in_the_minute(self) -> None:
        """Only the first second of a minute drops labels."""
        assert parse_timecode("19:01:01;00", parse_frame_rate("29.97DF")).frames == 0

    def test_those_labels_are_fine_at_a_non_drop_rate(self) -> None:
        assert parse_timecode("19:01:00:00", parse_frame_rate("30F")).frames == 0

    def test_round_trips_through_str(self) -> None:
        rate = parse_frame_rate("29.97DF")
        assert str(parse_timecode("19:10:00;05", rate)) == "19:10:00;05"
