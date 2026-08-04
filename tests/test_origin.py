"""Session zero, the 24-hour wrap, and what happens when neither is decidable.

Everything in M2 hangs off this: a chunk's position is its evidence minus session zero, so
an error here moves every sample of every track by the same amount and nothing downstream
can see it. The tests are correspondingly blunt about stating expected positions as
arithmetic.

The rollover cases are crossed with the frame rate on purpose. A wrapped chunk at 30F, at
29.97F, and at 29.97DF must move by three *different* numbers of real seconds — 86 400,
86 486.4, and 86 399.9136 — so an implementation that adds a calendar day passes the first
and fails the other two. One rate would not have caught it.
"""

from __future__ import annotations

import datetime as dt
from fractions import Fraction

import pytest

from dnd_audio.determinism import to_samples
from dnd_audio.errors import TimecodeError
from dnd_audio.timecode import FRAME_RATES, frame_index, frames_per_day, parse_timecode
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.origin import (
    PLAUSIBLE_SPAN_SECONDS,
    SessionOrigin,
    determine_origin,
)
from dnd_audio.timeline.rasterize import (
    SECONDS_PER_DAY,
    timecode_day_discrepancy_seconds,
)
from tests.manifests import asserted, bwf, config, config_for, manifest, offset, source, timecode

RATE = CANONICAL_SAMPLE_RATE
HOUR = 3600 * RATE


def placed(origin: SessionOrigin, path: str) -> int:
    """Where one source ended up, by path."""
    return {start.relative_path: start.session_start_sample for start in origin.starts}[path]


class TestSessionZero:
    """Where zero comes from, and what is never allowed to decide it."""

    def test_a_configured_origin_is_session_zero(self) -> None:
        """`19:00:00:00` on a stated date, and a chunk two hours later lands at two hours."""
        found = determine_origin(
            manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))]}),
            config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        )
        assert found.zero.source == "configured_origin"
        assert found.zero.since_domain_origin_samples == 19 * HOUR
        assert placed(found, "raw/tx-a/one.wav") == 0

    def test_without_an_origin_zero_is_the_earliest_source(self) -> None:
        """The spec's rule, and the offsets between sources survive it exactly."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR + 5 * RATE))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(19 * HOUR))],
                }
            ),
            config(),
        )
        assert found.zero.source == "earliest_source"
        assert placed(found, "raw/tx-b/one.wav") == 0
        assert placed(found, "raw/tx-a/one.wav") == 5 * RATE

    def test_origin_date_is_never_inferred_from_a_date_shaped_session_id(self) -> None:
        """The spec forbids it, and a session id that looks like a date is the trap.

        With no `origin_date` and no `origin_timecode`, zero must come from the sources —
        not from parsing "2026-08-15" out of the session's name. If it were inferred, the
        first source would land at 19 hours rather than at zero.
        """
        found = determine_origin(
            manifest(
                {"tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))]},
                session_id="2026-08-15",
            ),
            config(),
        )
        assert found.zero.source == "earliest_source"
        assert found.zero.origin_date is None
        assert placed(found, "raw/tx-a/one.wav") == 0

    def test_a_session_with_only_offsets_has_a_relative_origin(self) -> None:
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", offset(0))],
                    "tx-b": [source("raw/tx-b/one.wav", offset(24000))],
                }
            ),
            config(),
        )
        assert found.zero.domain == "relative"
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == 24000

    def test_zeros_domain_is_read_from_the_evidence_that_produced_it(self) -> None:
        """A derived zero inherits the earliest source's coordinate system.

        It matters at a fractional non-drop rate, where the two domains' day origins are
        86.4 seconds apart (OQ-015): recording which one zero belongs to is the difference
        between a reader being able to check the assumption and having to guess.
        """
        from_timecode = determine_origin(
            manifest({"tx-a": [source("raw/tx-a/one.wav", timecode("19:00:00:00"))]}),
            config(),
        )
        assert from_timecode.zero.domain == "timecode"

        from_bwf = determine_origin(
            manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))]}), config()
        )
        assert from_bwf.zero.domain == "recorder_epoch"


class TestSignedRecoveryOffsets:
    """The spec permits a signed offset; a blanket "before zero is fatal" would delete it."""

    def test_a_negative_offset_shifts_the_whole_timeline(self) -> None:
        """Zero is *defined* as the earliest start when no origin is configured.

        So an offset reaching below the other sources moves the origin rather than being
        refused, and every distance between sources is unchanged — which is all the
        evidence actually fixes.
        """
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", offset(0))],
                    "tx-b": [source("raw/tx-b/one.wav", offset(-24000))],
                }
            ),
            config(),
        )
        assert placed(found, "raw/tx-b/one.wav") == 0
        assert placed(found, "raw/tx-a/one.wav") == 24000
        assert any(d.code == "timeline_shifted_to_earliest_source" for d in found.decisions)

    def test_a_negative_offset_against_a_configured_origin_is_fatal(self) -> None:
        """The operator asserted where zero is; moving it silently would contradict them."""
        with pytest.raises(TimecodeError, match="before the configured session origin") as caught:
            determine_origin(
                manifest(
                    {
                        "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                        "tx-b": [source("raw/tx-b/one.wav", offset(-24000))],
                    }
                ),
                config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
            )
        assert caught.value.code == "audio_before_session_zero"

    def test_a_shifted_timeline_records_the_origin_it_actually_has(self) -> None:
        """The declared zero and sample 0 must be the same instant.

        An absolute source at 19:00 alongside an offset reaching a second below it moves
        the whole timeline, so sample 0 is 18:59:59 — and recording 19:00 would make every
        mapping from a session sample back to wall clock wrong by exactly the shift, in a
        way nothing downstream could detect.
        """
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", offset(-RATE))],
                }
            ),
            config(),
        )
        assert placed(found, "raw/tx-b/one.wav") == 0
        assert placed(found, "raw/tx-a/one.wav") == RATE

        recorded = found.zero.since_domain_origin_samples
        assert recorded is not None
        # Session zero, plus where the absolute source sits on the timeline, is that
        # source's own time of day. Nothing else is a consistent reading.
        assert recorded + placed(found, "raw/tx-a/one.wav") == 19 * HOUR
        assert recorded == 19 * HOUR - RATE

    def test_absolute_evidence_fixes_zero_and_offsets_are_placed_against_it(self) -> None:
        """Mixed absolute and relative: the absolute set decides, the offset follows."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", offset(5 * RATE))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        )
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == 5 * RATE


class TestRollover:
    """A wrapped chunk moves by its own domain's cycle, not by a calendar day."""

    def test_a_bwf_rollover_moves_by_a_calendar_day(self) -> None:
        """23:30 then 00:30: the second is an hour later, not 23 hours earlier."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(23 * HOUR + 30 * 60 * RATE))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(30 * 60 * RATE))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="23:30:00:00"),
        )
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == HOUR
        assert any(d.code == "rollover_inferred" for d in found.decisions)

    @pytest.mark.parametrize(
        ("label", "origin_text", "wrapped_text"),
        [
            # A one-hour *label* difference across midnight, at three rates. The excess
            # over a real hour is the day's excess scaled by 1/24: 3.6 s at 29.97
            # non-drop, essentially nothing at drop-frame.
            ("30F", "23:30:00:00", "00:30:00:00"),
            ("29.97F", "23:30:00:00", "00:30:00:00"),
            ("29.97DF", "23:30:00:00", "00:30:00:02"),
        ],
    )
    def test_a_timecode_rollover_moves_by_a_code_cycle(
        self, label: str, origin_text: str, wrapped_text: str
    ) -> None:
        """The wrapped chunk lands one code cycle on, converted at the exact rate.

        The expectation is rebuilt here from frame indices and the rational rate, never
        from the code under test: a wrapped chunk sits `(its frames + one cycle) - (the
        origin's frames)` frame periods after zero.
        """
        rate = FRAME_RATES[label]
        origin_frames = frame_index(parse_timecode(origin_text, rate))
        wrapped_frames = frame_index(parse_timecode(wrapped_text, rate))
        elapsed_frames = wrapped_frames + frames_per_day(rate) - origin_frames
        expected = to_samples(Fraction(elapsed_frames) / rate.rate, RATE)

        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", timecode(origin_text, label))],
                    "tx-b": [source("raw/tx-b/one.wav", timecode(wrapped_text, label))],
                }
            ),
            config_for(
                ("tx-a", "tx-b"),
                frame_rate=label,
                origin_date="2026-08-15",
                origin_timecode=origin_text,
            ),
        )
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == expected

    @pytest.mark.parametrize("label", ["23.98F", "29.97F"])
    def test_the_naive_calendar_day_answer_is_a_long_way_off(self, label: str) -> None:
        """The specific wrong implementation, computed alongside the right one.

        A plausible first draft converts the wrapped timecode to seconds and adds 86 400.
        At a non-drop fractional rate that is not the cycle, and the two answers differ by
        exactly the day discrepancy — 86.4 seconds. Not a rounding artefact: an audible
        misplacement of a whole chunk, in a direction nothing downstream would attribute
        to arithmetic.

        This is the assertion no single-rate test could make: at 30F the naive and correct
        answers are identical, which is exactly why the bug survives a 30F test suite.
        """
        rate = FRAME_RATES[label]
        origin_frames = frame_index(parse_timecode("23:30:00:00", rate))
        wrapped_frames = frame_index(parse_timecode("00:30:00:00", rate))

        correct = to_samples(
            Fraction(wrapped_frames + frames_per_day(rate) - origin_frames) / rate.rate, RATE
        )
        naive = to_samples(
            Fraction(wrapped_frames - origin_frames) / rate.rate + SECONDS_PER_DAY, RATE
        )

        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", timecode("23:30:00:00", label))],
                    "tx-b": [source("raw/tx-b/one.wav", timecode("00:30:00:00", label))],
                }
            ),
            config_for(
                ("tx-a", "tx-b"),
                frame_rate=label,
                origin_date="2026-08-15",
                origin_timecode="23:30:00:00",
            ),
        )
        assert placed(found, "raw/tx-b/one.wav") == correct
        assert correct - naive == to_samples(timecode_day_discrepancy_seconds(rate), RATE)
        assert correct - naive == 864 * RATE // 10

    def test_at_an_integer_rate_the_naive_answer_is_indistinguishable(self) -> None:
        """Why the bug is invisible until someone sets the receivers to 29.97.

        At 30F a timecode day *is* 86 400 seconds, so both implementations agree exactly.
        A suite that only ever tested 30F would have shipped the fractional bug.
        """
        rate = FRAME_RATES["30F"]
        origin_frames = frame_index(parse_timecode("23:30:00:00", rate))
        wrapped_frames = frame_index(parse_timecode("00:30:00:00", rate))
        correct = to_samples(
            Fraction(wrapped_frames + frames_per_day(rate) - origin_frames) / rate.rate, RATE
        )
        naive = to_samples(
            Fraction(wrapped_frames - origin_frames) / rate.rate + SECONDS_PER_DAY, RATE
        )
        assert correct == naive == HOUR

    def test_reject_refuses_to_infer(self) -> None:
        """The operator who knows their session did not cross midnight gets an error."""
        with pytest.raises(TimecodeError, match="rollover_policy is 'reject'") as caught:
            determine_origin(
                manifest(
                    {
                        "tx-a": [source("raw/tx-a/one.wav", bwf(23 * HOUR))],
                        "tx-b": [source("raw/tx-b/one.wav", bwf(30 * 60 * RATE))],
                    }
                ),
                config(
                    origin_date="2026-08-15",
                    origin_timecode="23:00:00:00",
                    rollover_policy="reject",
                ),
            )
        assert caught.value.code == "rollover_rejected"

    def test_no_rollover_is_inferred_when_nothing_wrapped(self) -> None:
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(20 * HOUR))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        )
        assert all(start.cycles == 0 for start in found.starts)
        assert not [d for d in found.decisions if d.code == "rollover_inferred"]


class TestRolloverWithoutAConfiguredOrigin:
    """With no anchor, the day boundary is inferred from where the sources are *not*."""

    def test_the_widest_quiet_stretch_is_where_midnight_falls(self) -> None:
        """Sources at 23:00, 23:30, and 00:30 are a 1.5-hour session, not a 23-hour one."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(23 * HOUR))],
                    "tx-b": [
                        source("raw/tx-b/one.wav", bwf(23 * HOUR + 30 * 60 * RATE)),
                        source("raw/tx-b/two.wav", bwf(30 * 60 * RATE)),
                    ],
                }
            ),
            config(),
        )
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == 30 * 60 * RATE
        assert placed(found, "raw/tx-b/two.wav") == 90 * 60 * RATE

    def test_nothing_wraps_when_the_quiet_stretch_already_contains_midnight(self) -> None:
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(20 * HOUR))],
                }
            ),
            config(),
        )
        assert all(start.cycles == 0 for start in found.starts)
        assert placed(found, "raw/tx-b/one.wav") == HOUR

    def test_an_ambiguous_spread_is_fatal_rather_than_a_coin_flip(self) -> None:
        """Three sources spaced exactly eight hours apart have no widest gap.

        Every reading is as good as every other, and INV-12 says a time that cannot be
        established is an error with a diagnostic.
        """
        with pytest.raises(TimecodeError, match="ambiguous") as caught:
            determine_origin(
                manifest(
                    {
                        "tx-a": [
                            source("raw/tx-a/one.wav", bwf(0)),
                            source("raw/tx-a/two.wav", bwf(8 * HOUR)),
                            source("raw/tx-a/three.wav", bwf(16 * HOUR)),
                        ]
                    }
                ),
                config_for(("tx-a",)),
            )
        assert caught.value.code == "rollover_ambiguous"
        assert "origin_timecode" in str(caught.value)

    def test_a_single_source_never_needs_a_rollover(self) -> None:
        found = determine_origin(
            manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(3 * HOUR))]}), config_for(("tx-a",))
        )
        assert placed(found, "raw/tx-a/one.wav") == 0

    def test_mixed_domains_with_differing_cycles_refuse_to_infer(self) -> None:
        """At 29.97F a timecode day and a calendar day are 86.4 s apart.

        Unwrapping a BWF reference and a timecode against each other then needs a rule for
        which day is which, and there is no evidence for one — so it asks for a dated
        origin instead of inventing an answer.
        """
        with pytest.raises(TimecodeError, match="24-hour cycles differ") as caught:
            determine_origin(
                manifest(
                    {
                        "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                        "tx-b": [source("raw/tx-b/one.wav", timecode("19:00:00:00", "29.97F"))],
                    }
                ),
                config(frame_rate="29.97F"),
            )
        assert caught.value.code == "rollover_ambiguous"

    def test_mixed_domains_are_fine_at_an_integer_rate(self) -> None:
        """The canonical fixture's shape: five BWF tracks and one ISMP timecode at 30F."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", timecode("19:00:01:00"))],
                }
            ),
            config(),
        )
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == RATE


class TestOnlyAnOperatorsDateAssignsACycle:
    """ADR-0031. A date the operator asserted is evidence; a date read from a file is not.

    The asymmetry is not fastidiousness. `bext.origination_date`/`origination_time` carry
    the receiver's real-time clock, and on 2026-08-03 two receivers were measured **48.7 s
    apart** while their timecode agreed to under one frame. This code applies day
    differences in the coarsest unit there is — whole 24-hour cycles — so two clocks
    straddling midnight would place their tracks a *day* apart on evidence known to be a
    minute wrong.
    """

    def test_matching_asserted_dates_mean_no_rollover(self) -> None:
        day = dt.date(2026, 8, 15)
        found = determine_origin(
            manifest(
                {
                    "tx-a": [asserted("raw/tx-a/one.wav", timecode("23:00:00:00", date=day))],
                    "tx-b": [asserted("raw/tx-b/one.wav", timecode("01:00:00:00", date=day))],
                }
            ),
            config(),
        )
        assert all(start.cycles == 0 for start in found.starts)
        # 01:00 really is 22 hours before 23:00 on the same day, however odd that looks.
        assert placed(found, "raw/tx-b/one.wav") == 0
        assert placed(found, "raw/tx-a/one.wav") == 22 * HOUR

    def test_differing_asserted_dates_are_applied_as_whole_cycles(self) -> None:
        found = determine_origin(
            manifest(
                {
                    "tx-a": [
                        asserted(
                            "raw/tx-a/one.wav",
                            timecode("23:00:00:00", date=dt.date(2026, 8, 15)),
                        )
                    ],
                    "tx-b": [
                        asserted(
                            "raw/tx-b/one.wav",
                            timecode("01:00:00:00", date=dt.date(2026, 8, 16)),
                        )
                    ],
                }
            ),
            config(),
        )
        assert placed(found, "raw/tx-a/one.wav") == 0
        assert placed(found, "raw/tx-b/one.wav") == 2 * HOUR
        assert any(d.code == "rollover_from_recorded_dates" for d in found.decisions)

    def test_the_same_dates_read_from_the_files_assign_nothing(self) -> None:
        """The mutation that matters: identical evidence, from the file rather than the
        operator, must not produce a day of shift.

        Inference takes over instead — it reads the counters themselves, involves no wall
        clock, and here reaches the same answer by a route that cannot be a minute wrong.
        """
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(23 * HOUR, date=dt.date(2026, 8, 15)))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(1 * HOUR, date=dt.date(2026, 8, 16)))],
                }
            ),
            config(),
        )
        assert not [d for d in found.decisions if d.code == "rollover_from_recorded_dates"]
        assert any(note.code == "midnight_rollover_inferred" for note in found.warnings)

    def test_a_partially_asserted_session_falls_back_to_inference(self) -> None:
        """M1 produces exactly this: `bext` carries a date, an `ISMP` timecode does not."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [
                        asserted(
                            "raw/tx-a/one.wav",
                            timecode("19:00:00:00", date=dt.date(2026, 8, 15)),
                        )
                    ],
                    "tx-b": [source("raw/tx-b/one.wav", timecode("20:00:00:00"))],
                }
            ),
            config(),
        )
        assert placed(found, "raw/tx-b/one.wav") == HOUR
        assert not [d for d in found.decisions if d.code == "rollover_from_recorded_dates"]

    def test_an_origin_date_after_the_earliest_recording_is_fatal(self) -> None:
        with pytest.raises(TimecodeError, match="cannot begin after its own") as caught:
            determine_origin(
                manifest(
                    {
                        "tx-a": [
                            asserted(
                                "raw/tx-a/one.wav",
                                timecode("01:00:00:00", date=dt.date(2026, 8, 15)),
                            )
                        ]
                    }
                ),
                config_for(("tx-a",), origin_date="2026-08-16"),
            )
        assert caught.value.code == "origin_after_earliest_source"


class TestWarnings:
    def test_an_implausible_span_warns_without_failing(self) -> None:
        """Arithmetically unambiguous, humanly unlikely — a warning, not an error (OQ-014)."""
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(0))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(20 * HOUR))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="00:00:00:00"),
        )
        assert any(note.code == "implausible_session_span" for note in found.warnings)
        assert placed(found, "raw/tx-b/one.wav") == 20 * HOUR

    def test_an_ordinary_session_does_not_warn(self) -> None:
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(22 * HOUR))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        )
        assert not found.warnings
        assert 3 * HOUR < PLAUSIBLE_SPAN_SECONDS * RATE

    def test_mixed_domains_warn_only_where_the_day_lengths_differ(self) -> None:
        """At 30F the two origins coincide; at 29.97F they are 86.4 seconds apart."""
        at_30 = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", timecode("19:00:00:00"))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        )
        assert not [n for n in at_30.warnings if n.code == "mixed_time_domains"]

        at_29_97 = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
                    "tx-b": [source("raw/tx-b/one.wav", timecode("19:00:00:00", "29.97F"))],
                }
            ),
            config(frame_rate="29.97F", origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        )
        assert [n for n in at_29_97.warnings if n.code == "mixed_time_domains"]
