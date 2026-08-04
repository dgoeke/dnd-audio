"""Exact sample positions, at every rate the hardware can be set to.

This file is the proof for spec acceptance criterion 2 and for INV-04, so its expectations
are computed **independently** — as `Fraction` arithmetic written out longhand in the test
— and never by calling the function under test. A test whose expected value comes from the
implementation is a change detector wearing a correctness test's clothes.

The cases that actually bite are the fractional ones. At 30000/1001 fps a frame is 8008/5
samples at 48 kHz, so a frame boundary lands on a whole sample only every fifth frame; at
24000/1001 it is 10010/5 = 2002 samples, which always does. Drop-frame changes which
*labels* exist without changing the rate, and rollover at a non-drop fractional rate adds
86 486.4 real seconds rather than 86 400 — the error a plausible implementation makes and
the one nothing downstream would attribute to arithmetic.
"""

from __future__ import annotations

import datetime as dt
from fractions import Fraction

import pytest

from dnd_audio.artifacts.manifest import (
    BwfSampleReferenceRecord,
    RationalRate,
    SessionOffsetRecord,
    StartEvidenceRecord,
    TimecodeRecord,
)
from dnd_audio.determinism import to_samples
from dnd_audio.timecode import FRAME_RATES, frame_index, frames_per_day, parse_timecode
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.rasterize import (
    SECONDS_PER_DAY,
    absolute_seconds,
    cycle_units,
    evidence_quantum_samples,
    has_mixed_absolute_domains,
    is_absolute,
    quantization_tolerance_samples,
    relative_seconds,
    session_position,
    timecode_day_discrepancy_seconds,
)

RATE = CANONICAL_SAMPLE_RATE


def timecode_record(text: str, label: str) -> TimecodeRecord:
    """A manifest timecode record, built the way M1 builds one."""
    frame_rate = FRAME_RATES[label]
    parsed = parse_timecode(text, frame_rate)
    return TimecodeRecord(
        text=text,
        frames=frame_index(parsed),
        frame_rate_label=label,
        frame_rate=RationalRate(
            numerator=frame_rate.rate.numerator, denominator=frame_rate.rate.denominator
        ),
        drop_frame=frame_rate.drop_frame,
    )


def bwf(samples: int, sample_rate: int = RATE) -> BwfSampleReferenceRecord:
    return BwfSampleReferenceRecord(samples=samples, sample_rate=sample_rate)


class TestExactSecondsFromEvidence:
    """Each evidence kind converts to exact rational seconds from its own day origin."""

    def test_a_bwf_reference_is_samples_over_its_own_rate(self) -> None:
        # 19:00:00 at 48 kHz. Stated as the arithmetic, not as a magic constant.
        samples = 19 * 3600 * 48000
        assert absolute_seconds(bwf(samples), FRAME_RATES["30F"]) == Fraction(19 * 3600)

    def test_a_bwf_reference_uses_the_files_rate_not_the_sessions(self) -> None:
        """A 44.1 kHz file counts 44100ths of a second.

        Reading it as 48000ths would misplace the file by 8.75% — nine minutes into a
        two-hour session. M2 refuses such a source before it builds a timeline, but the
        conversion has to be right regardless: the refusal is a policy and this is
        arithmetic.
        """
        assert absolute_seconds(bwf(44100, 44100), FRAME_RATES["30F"]) == Fraction(1)
        assert absolute_seconds(bwf(44100, 48000), FRAME_RATES["30F"]) == Fraction(44100, 48000)

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("24F", Fraction(1, 24)),
            ("25F", Fraction(1, 25)),
            ("30F", Fraction(1, 30)),
            ("50F", Fraction(1, 50)),
            ("60F", Fraction(1, 60)),
            ("23.98F", Fraction(1001, 24000)),
            ("29.97F", Fraction(1001, 30000)),
            ("29.97DF", Fraction(1001, 30000)),
        ],
    )
    def test_one_frame_is_the_reciprocal_of_the_exact_rate(
        self, label: str, expected: Fraction
    ) -> None:
        """Frame 1 sits exactly one frame period after frame 0, at every rate.

        29.97DF appears alongside 29.97F because drop-frame changes which labels exist,
        never the rate: a dropped label is a label, not a frame of audio.
        """
        one_frame = timecode_record("00:00:00:01", label)
        assert absolute_seconds(one_frame, FRAME_RATES[label]) == expected

    def test_a_session_offset_has_no_time_of_day(self) -> None:
        """Asking for one is a caller confusing the two coordinate systems (ADR-0006)."""
        offset = SessionOffsetRecord(samples=48000, sample_rate=RATE)
        with pytest.raises(ValueError, match="no time of day"):
            absolute_seconds(offset, FRAME_RATES["30F"])
        assert relative_seconds(offset) == Fraction(1)

    def test_a_session_offset_may_be_negative(self) -> None:
        """The spec permits a signed offset, so the conversion must carry the sign."""
        assert relative_seconds(SessionOffsetRecord(samples=-24000, sample_rate=RATE)) == Fraction(
            -1, 2
        )


class TestSamplePositions:
    """Criterion 2: evidence maps to the expected integer sample position."""

    def test_non_drop_integer_rate_lands_on_whole_samples(self) -> None:
        """At 30 fps a frame is exactly 1600 samples, so nothing rounds."""
        zero = absolute_seconds(timecode_record("19:00:00:00", "30F"), FRAME_RATES["30F"])
        start = absolute_seconds(timecode_record("19:00:02:15", "30F"), FRAME_RATES["30F"])
        # 2 seconds and 15 frames = 2 * 48000 + 15 * 1600.
        assert session_position(start, zero, RATE) == 2 * 48000 + 15 * 1600

    def test_a_fractional_rate_frame_is_not_a_whole_number_of_samples(self) -> None:
        """29.97 fps: one frame is 8008/5 = 1601.6 samples at 48 kHz.

        This is the case ADR-0008 exists for. The exact position of frame 1 is 8008/5, and
        the documented rule rounds halves away from zero, giving 1602.
        """
        rate = FRAME_RATES["29.97F"]
        zero = absolute_seconds(timecode_record("00:00:00:00", "29.97F"), rate)
        one = absolute_seconds(timecode_record("00:00:00:01", "29.97F"), rate)
        assert one - zero == Fraction(1001, 30000)
        assert (one - zero) * RATE == Fraction(8008, 5)
        assert session_position(one, zero, RATE) == 1602

    @pytest.mark.parametrize("frames", [0, 5, 10, 2000, 12345])
    def test_every_fifth_frame_at_29_97_is_exact(self, frames: int) -> None:
        """8008/5 samples per frame means five frames is exactly 8008 samples.

        Chosen because it is the property the drop-frame fixture depends on: a chunk can
        only start on a whole sample *and* a frame boundary when its frame index divides
        by five.
        """
        rate = FRAME_RATES["29.97F"]
        zero = absolute_seconds(timecode_record("00:00:00:00", "29.97F"), rate)
        exact = Fraction(frames * 5 * 1001, 30000)
        assert session_position(zero + exact, zero, RATE) == frames * 8008

    def test_23_98_frames_are_exactly_2002_samples(self) -> None:
        """24000/1001 fps: 48000 * 1001/24000 = 2002, a whole number every frame."""
        rate = FRAME_RATES["23.98F"]
        zero = absolute_seconds(timecode_record("00:00:00:00", "23.98F"), rate)
        ten = absolute_seconds(timecode_record("00:00:00:10", "23.98F"), rate)
        assert session_position(ten, zero, RATE) == 10 * 2002

    def test_a_drop_frame_label_counts_the_frames_it_does_not_skip(self) -> None:
        """`00:01:00:02` is frame 1800, not 1802 and not 1798.

        Drop-frame never assigns the labels `00:01:00:00` and `00:01:00:01`, so the first
        frame of minute one carries the label `:02` while being the 1800th frame — exactly
        60 nominal seconds of 30 labels. Skipping labels is what lets the label track real
        time; no audio is discarded, which is why the position is 1800 frame periods and
        not 1802.
        """
        rate = FRAME_RATES["29.97DF"]
        zero = absolute_seconds(timecode_record("00:00:00:00", "29.97DF"), rate)
        minute = timecode_record("00:01:00:02", "29.97DF")
        assert minute.frames == 60 * 30
        # 1800 frames at 1001/30000 s. 1800 divides by five, so this is a whole number of
        # samples: 1800 * 8008/5 = 2 882 880, and the rounding rule has nothing to do.
        assert session_position(absolute_seconds(minute, rate), zero, RATE) == 1800 * 8008 // 5

    def test_drop_frame_converges_on_real_time_over_ten_minutes(self) -> None:
        """Drop-frame's actual guarantee, and it is a ten-minute one.

        Two labels are skipped every minute except every tenth, which is 18 per ten
        minutes against the 18.018 that exact correction would need. So at one minute the
        label is 60 ms ahead of real time and at ten minutes it is 0.6 ms behind — well
        inside a single frame. Asserting the one-minute case as "within a frame of a
        minute" would have been false; this is the claim drop-frame actually makes.
        """
        rate = FRAME_RATES["29.97DF"]
        zero = absolute_seconds(timecode_record("00:00:00:00", "29.97DF"), rate)
        ten_minutes = timecode_record("00:10:00:00", "29.97DF")
        assert ten_minutes.frames == 600 * 30 - 18

        position = session_position(absolute_seconds(ten_minutes, rate), zero, RATE)
        assert position == to_samples(Fraction(17982 * 1001, 30000), RATE)
        one_frame = 1602
        assert abs(position - 600 * RATE) < one_frame

        # Non-drop at the same rate is the comparison that makes the point: after ten
        # minutes of real time its label has fallen 36 frames behind.
        non_drop = timecode_record("00:10:00:00", "29.97F")
        assert non_drop.frames - ten_minutes.frames == 18

    def test_a_session_offset_is_already_a_position(self) -> None:
        """No conversion, no rounding: it is signed 48 kHz samples from session zero."""
        offset = SessionOffsetRecord(samples=-12345, sample_rate=RATE)
        assert session_position(relative_seconds(offset), Fraction(0), RATE) == -12345

    def test_the_subtraction_happens_before_the_rounding(self) -> None:
        """ADR-0008's central rule, demonstrated on a case where the two differ.

        Session zero at 29.97 frame 1 (8008/5 samples) and a source at frame 2 (16016/5).
        Rounding each and subtracting gives 3203 - 1602 = 1601. Subtracting exactly and
        rounding once gives 8008/5 = 1601.6 -> 1602. The one-rounding answer is the
        correct distance between two frames; the other is a sample short.
        """
        rate = FRAME_RATES["29.97F"]
        zero = absolute_seconds(timecode_record("00:00:00:01", "29.97F"), rate)
        source = absolute_seconds(timecode_record("00:00:00:02", "29.97F"), rate)

        rounded_then_subtracted = to_samples(source, RATE) - to_samples(zero, RATE)
        assert rounded_then_subtracted == 1601
        assert session_position(source, zero, RATE) == 1602


class TestTheDayCycleIsCountedInItsOwnUnits:
    """ADR-0009's table, asserted rather than described."""

    @pytest.mark.parametrize(
        ("label", "expected_frames", "expected_seconds"),
        [
            ("24F", 2073600, Fraction(86400)),
            ("25F", 2160000, Fraction(86400)),
            ("30F", 2592000, Fraction(86400)),
            ("50F", 4320000, Fraction(86400)),
            ("60F", 5184000, Fraction(86400)),
            ("23.98F", 2073600, Fraction(864864, 10)),
            ("29.97F", 2592000, Fraction(864864, 10)),
            ("29.97DF", 2589408, Fraction(2589408 * 1001, 30000)),
        ],
    )
    def test_a_timecode_day_is_a_number_of_frames_not_a_number_of_seconds(
        self, label: str, expected_frames: int, expected_seconds: Fraction
    ) -> None:
        rate = FRAME_RATES[label]
        assert frames_per_day(rate) == expected_frames
        assert Fraction(expected_frames) / rate.rate == expected_seconds

    def test_a_non_drop_fractional_day_is_longer_than_a_calendar_day(self) -> None:
        """86.4 seconds longer, which is the whole finding.

        Unwrapping a 29.97F rollover by 86 400 seconds would place the wrapped chunk
        86.4 seconds *before* the frame that preceded it.
        """
        assert timecode_day_discrepancy_seconds(FRAME_RATES["29.97F"]) == Fraction(864, 10)
        assert timecode_day_discrepancy_seconds(FRAME_RATES["23.98F"]) == Fraction(864, 10)

    def test_drop_frame_nearly_but_not_exactly_tracks_a_calendar_day(self) -> None:
        """-86.4 ms: drop-frame's residual. The same kind of assumption, tiny."""
        discrepancy = timecode_day_discrepancy_seconds(FRAME_RATES["29.97DF"])
        assert discrepancy == Fraction(2589408 * 1001, 30000) - 86400
        assert Fraction(-1, 10) < discrepancy < 0

    @pytest.mark.parametrize("label", ["24F", "25F", "30F", "50F", "60F"])
    def test_an_integer_rate_day_is_a_calendar_day(self, label: str) -> None:
        assert timecode_day_discrepancy_seconds(FRAME_RATES[label]) == 0

    def test_a_bwf_cycle_is_samples_at_the_files_rate(self) -> None:
        assert cycle_units(bwf(0, 48000), FRAME_RATES["30F"]) == SECONDS_PER_DAY * 48000
        assert cycle_units(bwf(0, 44100), FRAME_RATES["30F"]) == SECONDS_PER_DAY * 44100

    def test_a_session_offset_has_no_cycle(self) -> None:
        offset = SessionOffsetRecord(samples=0, sample_rate=RATE)
        assert cycle_units(offset, FRAME_RATES["30F"]) is None


class TestRollover:
    """Unwrapping, in each domain, with the wrong answer written out for contrast."""

    def test_a_bwf_rollover_adds_exactly_one_calendar_day(self) -> None:
        just_after_midnight = bwf(5 * 48000)
        unwrapped = absolute_seconds(just_after_midnight, FRAME_RATES["30F"], cycles=1)
        assert unwrapped == Fraction(SECONDS_PER_DAY + 5)

    def test_a_29_97_rollover_adds_a_code_cycle_not_a_calendar_day(self) -> None:
        """The finding, as a test.

        A chunk at 00:00:05:00 that really belongs to the next timecode day sits
        2 592 150 frames after 00:00:00:00 of the previous one. In seconds that is
        86 491.4, not 86 405 — and a naive implementation that added 86 400 seconds would
        put it 86.4 seconds too early.
        """
        rate = FRAME_RATES["29.97F"]
        wrapped = timecode_record("00:00:05:00", "29.97F")
        unwrapped = absolute_seconds(wrapped, rate, cycles=1)

        expected_frames = wrapped.frames + 2592000
        assert unwrapped == Fraction(expected_frames * 1001, 30000)

        naive = absolute_seconds(wrapped, rate) + SECONDS_PER_DAY
        assert unwrapped - naive == Fraction(864, 10)

    def test_a_drop_frame_rollover_adds_the_drop_frame_cycle(self) -> None:
        rate = FRAME_RATES["29.97DF"]
        wrapped = timecode_record("00:00:05:00", "29.97DF")
        unwrapped = absolute_seconds(wrapped, rate, cycles=1)
        assert unwrapped == Fraction((wrapped.frames + 2589408) * 1001, 30000)

    def test_a_rollover_position_is_exact_at_48_khz(self) -> None:
        """End to end: a wrapped 30F chunk lands where a calendar day says it should."""
        rate = FRAME_RATES["30F"]
        zero = absolute_seconds(timecode_record("23:59:00:00", "30F"), rate)
        wrapped = absolute_seconds(timecode_record("00:01:00:00", "30F"), rate, cycles=1)
        assert session_position(wrapped, zero, RATE) == 2 * 60 * 48000

    def test_backward_rollover_is_not_offered(self) -> None:
        """The spec permits a single *forward* rollover and nothing else."""
        with pytest.raises(ValueError, match="not be negative"):
            absolute_seconds(bwf(0), FRAME_RATES["30F"], cycles=-1)


class TestQuantizationTolerance:
    """How large an overlap is explainable by rounding (ADR-0008, ADR-0010)."""

    def test_two_bwf_starts_tolerate_the_hardware_they_came_from(self) -> None:
        """A sample count is not automatically sample-*exact* (OQ-004).

        DJI's `time_reference` moves in steps of 1600 samples, so two chunks of one track
        can overlap by up to that much from rounding alone. A recorder that really is
        exact says so, and gets the old one-sample tolerance back.
        """
        coarse = quantization_tolerance_samples(
            bwf(0), bwf(48000), FRAME_RATES["29.97F"], RATE, bwf_quantum_samples=1600
        )
        assert coarse == 1600
        exact = quantization_tolerance_samples(
            bwf(0), bwf(48000), FRAME_RATES["29.97F"], RATE, bwf_quantum_samples=1
        )
        assert exact == 1

    def test_a_timecode_start_tolerates_a_whole_frame(self) -> None:
        """1602 samples at 29.97, because the recorder's own rounding dominates ours."""
        tolerance = quantization_tolerance_samples(
            timecode_record("00:00:00:00", "29.97F"),
            timecode_record("00:00:01:00", "29.97F"),
            FRAME_RATES["29.97F"],
            RATE,
            bwf_quantum_samples=1,
        )
        assert tolerance == 1602

    def test_one_timecode_start_is_enough_to_widen_the_tolerance(self) -> None:
        """The tolerance is the pair's worst resolution, not an average of the two."""
        tolerance = quantization_tolerance_samples(
            bwf(0),
            timecode_record("00:00:01:00", "30F"),
            FRAME_RATES["30F"],
            RATE,
            bwf_quantum_samples=1,
        )
        assert tolerance == 1600

    def test_the_tolerance_is_a_property_of_the_pair_not_of_the_session(self) -> None:
        """Two exact chunks keep a one-sample tolerance even at a fractional session rate.

        Widening it because some *other* track uses timecodes would hide a real
        1602-sample overlap between two sample-exact chunks.
        """
        assert (
            quantization_tolerance_samples(
                bwf(0), bwf(1), FRAME_RATES["29.97DF"], RATE, bwf_quantum_samples=1
            )
            == 1
        )


class TestDomainClassification:
    def test_only_offsets_are_relative(self) -> None:
        assert is_absolute(bwf(0))
        assert is_absolute(timecode_record("00:00:00:00", "30F"))
        assert not is_absolute(SessionOffsetRecord(samples=0, sample_rate=RATE))

    def test_each_kind_of_evidence_states_its_own_resolution(self) -> None:
        """A timecode's from its rate, a BWF reference's from configuration, an
        operator's offset from the fact that they typed it."""
        quantum = 1600

        def measure(evidence: StartEvidenceRecord) -> int:
            return evidence_quantum_samples(
                evidence, FRAME_RATES["29.97F"], RATE, bwf_quantum_samples=quantum
            )

        assert measure(timecode_record("00:00:00:00", "29.97F")) == 1602
        assert measure(bwf(0)) == quantum
        assert measure(SessionOffsetRecord(samples=0, sample_rate=RATE)) == 1

    def test_a_bwf_quantum_is_stated_at_the_files_rate_and_applied_at_the_sessions(
        self,
    ) -> None:
        """1600 samples of a 44.1 kHz file is more than 1600 samples of session time.

        Such a file is refused before layout, so nothing depends on this today — but the
        function converts rather than assuming, and a silent unit mismatch here would move
        every chunk of an affected track.
        """
        assert (
            evidence_quantum_samples(
                bwf(0, sample_rate=44100), FRAME_RATES["30F"], RATE, bwf_quantum_samples=1600
            )
            == 1742
        )

    def test_mixed_domains_needs_both_absolute_kinds(self) -> None:
        timecode = timecode_record("00:00:00:00", "30F")
        offset = SessionOffsetRecord(samples=0, sample_rate=RATE)
        assert has_mixed_absolute_domains([bwf(0), timecode])
        assert not has_mixed_absolute_domains([bwf(0), bwf(1)])
        assert not has_mixed_absolute_domains([timecode, offset])

    def test_the_canonical_fixtures_shape_is_free_at_30_fps(self) -> None:
        """Five BWF tracks plus one ISMP timecode, at 30F: mixed, and costs nothing.

        The mixing is real, and the discrepancy that would make it an assumption worth
        warning about is exactly zero at an integer rate.
        """
        evidence: list[StartEvidenceRecord] = [
            bwf(0),
            bwf(1),
            timecode_record("19:00:00:00", "30F"),
        ]
        assert has_mixed_absolute_domains(evidence)
        assert timecode_day_discrepancy_seconds(FRAME_RATES["30F"]) == 0


class TestOriginationDatesSurvive:
    """The evidence records carry their dates; rasterization does not consume them."""

    def test_a_dated_reference_keeps_its_date(self) -> None:
        record = BwfSampleReferenceRecord(
            samples=0, sample_rate=RATE, origination_date=dt.date(2026, 8, 15)
        )
        assert record.origination_date == dt.date(2026, 8, 15)
        assert absolute_seconds(record, FRAME_RATES["30F"]) == 0
