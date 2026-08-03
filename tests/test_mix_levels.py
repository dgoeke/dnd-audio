"""Per-track voice-level correction: the first of M5's gate criteria.

> Conservative per-track voice-level correction estimated from high-confidence speech
> attributed to that track, clamped to a safe range.

The failure this file exists to make impossible is the quiet one. A track with no speech
reference is `None` in the graph — M3 chose that over a default *precisely* so a consumer could
not read "no measurement" as "0 dBFS" — and a correction computed against zero would lift that
track by the full clamp, amplifying whatever it did record. Nothing downstream would notice: the
mix would simply have a hiss in it.
"""

from __future__ import annotations

import math

import pytest

from dnd_audio.config import EnvelopeConfig
from dnd_audio.mix.levels import MILLIBELS_PER_DB, level_corrections
from tests.graphs import a_graph, a_track


def _corrections(*references: int | None, clamp_db: float = 6.0) -> dict[str, int]:
    """`{track_id: correction_mb}` for a session whose tracks measure ``references``."""
    tracks = [
        a_track(f"tx-{chr(ord('a') + index)}", speech_reference_mbfs=reference)
        for index, reference in enumerate(references)
    ]
    found = level_corrections(
        a_graph(tracks=tracks), settings=EnvelopeConfig(max_level_correction_db=clamp_db)
    )
    return {item.track_id: item.correction_mb for item in found.corrections}


class TestVoiceLevelCorrection:
    def test_the_target_is_the_tracks_own_median(self) -> None:
        """Six wearers end up level with each other; where that lands absolutely is the
        two-pass loudness normalization's question, not this one (ADR-0023)."""
        found = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-3000),
                    a_track("tx-b", speech_reference_mbfs=-2600),
                    a_track("tx-c", speech_reference_mbfs=-2800),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        assert found.target_mbfs == -2800

    def test_a_quiet_track_is_lifted_and_a_loud_one_is_cut(self) -> None:
        corrections = _corrections(-3000, -2800, -2600)
        assert corrections["tx-a"] == 200  # 2 dB up toward the median
        assert corrections["tx-b"] == 0
        assert corrections["tx-c"] == -200  # 2 dB down

    def test_the_correction_is_clamped_in_both_directions(self) -> None:
        """The spec's "clamp correction to a safe range", exercised on both signs at once."""
        corrections = _corrections(-6000, -2800, -400, clamp_db=6.0)
        assert corrections["tx-a"] == 600
        assert corrections["tx-c"] == -600

    def test_a_clamped_track_says_so_and_says_by_how_much(self) -> None:
        found = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-6000),
                    a_track("tx-b", speech_reference_mbfs=-2800),
                    a_track("tx-c", speech_reference_mbfs=-2700),
                ]
            ),
            settings=EnvelopeConfig(max_level_correction_db=6.0),
        )
        clamped = [item for item in found.corrections if item.clamped]
        assert [item.track_id for item in clamped] == ["tx-a"]
        codes = {note.code for note in found.warnings}
        assert codes == {"mix_level_correction_clamped"}
        message = next(note.message for note in found.warnings)
        assert "-60.00 dBFS" in message
        assert "+6.00 dB" in message

    def test_a_track_with_no_reference_is_corrected_by_zero_and_warned_about(self) -> None:
        """`None` is "unknown", never zero (ADR-0014, and M3's closeout says so twice).

        Treating it as 0 dBFS would compute a +32 dB correction, clamp it to the maximum, and
        lift a track whose level nobody measured — quietly, because a hiss in a mix looks like
        a room.
        """
        found = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-2800),
                    a_track("tx-b", speech_reference_mbfs=None),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        by_id = {item.track_id: item for item in found.corrections}
        assert by_id["tx-b"].correction_mb == 0
        assert by_id["tx-b"].gain == 1.0
        assert by_id["tx-b"].reference_mbfs is None
        assert [note.code for note in found.warnings] == ["mix_level_uncorrected"]
        assert [note.path for note in found.warnings] == ["tx-b"]

    def test_an_unmeasured_track_does_not_move_the_target_for_the_others(self) -> None:
        """The median is over the references that exist. Counting a `None` as anything at all
        would drag every other track's correction with it."""
        with_gap = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-3000),
                    a_track("tx-b", speech_reference_mbfs=-2800),
                    a_track("tx-c", speech_reference_mbfs=None),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        without = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-3000),
                    a_track("tx-b", speech_reference_mbfs=-2800),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        assert with_gap.target_mbfs == without.target_mbfs

    def test_a_session_where_nothing_was_measured_corrects_nothing(self) -> None:
        """Six tracks of silence, or a detector that found no speech anywhere. Every gain is
        unity and every track is warned about, rather than a target invented from no data."""
        found = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=None),
                    a_track("tx-b", speech_reference_mbfs=None),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        assert found.target_mbfs is None
        assert all(item.correction_mb == 0 for item in found.corrections)
        assert len(found.warnings) == 2

    def test_a_correction_of_zero_is_a_gain_of_exactly_one(self) -> None:
        """Not approximately: an uncorrected track's samples must be untouched, so that a
        session with matched levels produces the same mix as one with the feature disabled."""
        found = level_corrections(a_graph(tracks=[a_track("tx-a")]), settings=EnvelopeConfig())
        assert found.corrections[0].gain == 1.0

    @pytest.mark.parametrize(("millibels", "expected_db"), [(600, 6.0), (-600, -6.0), (150, 1.5)])
    def test_millibels_become_the_linear_factor_they_name(
        self, millibels: int, expected_db: float
    ) -> None:
        """One conversion, in one place. A second spelling of `10 ** (db / 20)` is how the
        unit stops meaning what the graph says it means."""
        found = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-2800 - millibels),
                    a_track("tx-b", speech_reference_mbfs=-2800),
                    a_track("tx-c", speech_reference_mbfs=-2800),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        gain = next(item.gain for item in found.corrections if item.track_id == "tx-a")
        assert gain == pytest.approx(10.0 ** (expected_db / 20.0))
        assert MILLIBELS_PER_DB * expected_db == pytest.approx(millibels)

    def test_gains_come_back_in_the_order_asked_for(self) -> None:
        """The renderer multiplies a `(samples, tracks)` matrix by these, so a mismatched
        order applies one wearer's correction to another's audio — inaudibly wrong."""
        found = level_corrections(
            a_graph(
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-3000),
                    a_track("tx-b", speech_reference_mbfs=-2800),
                    a_track("tx-c", speech_reference_mbfs=-2600),
                ]
            ),
            settings=EnvelopeConfig(),
        )
        forwards = found.gains(("tx-a", "tx-b", "tx-c"))
        backwards = found.gains(("tx-c", "tx-b", "tx-a"))
        assert list(forwards) == list(reversed(list(backwards)))
        assert forwards[0] > 1.0 > forwards[2]

    def test_asking_for_a_track_this_session_does_not_have_raises(self) -> None:
        found = level_corrections(a_graph(tracks=[a_track("tx-a")]), settings=EnvelopeConfig())
        with pytest.raises(KeyError):
            found.gains(("tx-a", "tx-z"))

    def test_the_clamp_bounds_every_gain_by_construction(self) -> None:
        """What ADR-0022's achievability validator relies on: the correction can erode the
        dominance margin by at most twice the clamp, because no single gain exceeds it."""
        clamp_db = 6.0
        corrections = _corrections(-7000, -2800, -100, None, clamp_db=clamp_db)
        limit = round(clamp_db * MILLIBELS_PER_DB)
        assert all(abs(value) <= limit for value in corrections.values())
        assert max(corrections.values()) == limit
        assert min(corrections.values()) == -limit
        assert math.isclose(10.0 ** (limit / (20.0 * MILLIBELS_PER_DB)), 10.0 ** (clamp_db / 20.0))
