"""The gain envelope — the assertions M5's charter calls the real gate.

> Decoded loudness alone is *not* evidence of correct channel selection. If the only tests are
> loudness tests, a mix that picks the wrong speaker will pass.

So every criterion here is asserted against the **applied coefficient** — the share times the
track's level correction, which is what actually multiplies a sample — and at both extremes of
the correction clamp where that differs most from the share. Asserting the share alone was the
first finding of M5's plan review, and it is the difference between a bound on something
audible and a bound on an intermediate.

The tolerances are configuration (`mix.envelope`), not constants in this file, because the gate
asks for "explicit configurable tolerances" and because a validator refuses a configuration
whose margin the two weight floors cannot deliver.
"""

from __future__ import annotations

import numpy as np
import pytest

from dnd_audio.artifacts.activity import ActivityGraph, ActivityTrack
from dnd_audio.config import EnvelopeConfig
from dnd_audio.mix.envelope import (
    EnvelopeError,
    EnvelopeStream,
    active_spans,
    expand,
    frame_interval,
)
from dnd_audio.mix.levels import LevelCorrections, TrackCorrection, level_corrections
from tests.graphs import a_candidate, a_graph, a_track

SECOND = 48_000

#: Six tracks, the roster the spec describes. Envelope behaviour depends on the count — the
#: share of a silent channel is 1/N — so the fixtures here use the real number.
TRACKS = ("tx-a", "tx-b", "tx-c", "tx-d", "tx-e", "tx-f")


def _tracks(*, reference: int | None = -2800) -> list[ActivityTrack]:
    return [a_track(track_id, speech_reference_mbfs=reference) for track_id in TRACKS]


def _uncorrected(track_ids: tuple[str, ...] = TRACKS) -> LevelCorrections:
    """Every gain exactly 1.0, so a test can isolate the share from the correction."""
    return LevelCorrections(
        target_mbfs=-2800,
        corrections=tuple(
            TrackCorrection(track_id=t, reference_mbfs=-2800, correction_mb=0, clamped=False)
            for t in track_ids
        ),
        warnings=(),
    )


def _corrected(millibels: dict[str, int]) -> LevelCorrections:
    """Corrections stated directly, so a test can put two tracks at opposite clamp extremes."""
    return LevelCorrections(
        target_mbfs=-2800,
        corrections=tuple(
            TrackCorrection(
                track_id=t,
                reference_mbfs=-2800,
                correction_mb=millibels.get(t, 0),
                clamped=False,
            )
            for t in TRACKS
        ),
        warnings=(),
    )


def envelope(
    graph: ActivityGraph,
    *,
    settings: EnvelopeConfig | None = None,
    corrections: LevelCorrections | None = None,
    track_ids: tuple[str, ...] = TRACKS,
    chunk_frames: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """The whole session's `(shares, applied)`, concatenated.

    Only ever used on fixtures a few seconds long. The production path never does this — see
    `TestTheEnvelopeIsBounded` and `tests/test_memory.py`.
    """
    stream = EnvelopeStream(
        graph,
        settings=settings or EnvelopeConfig(),
        corrections=corrections or _uncorrected(track_ids),
        track_ids=track_ids,
    )
    chunks = list(stream.chunks(chunk_frames=chunk_frames))
    return (
        np.concatenate([chunk.shares for chunk in chunks]),
        np.concatenate([chunk.applied for chunk in chunks]),
    )


def _db(value: float) -> float:
    return float(20.0 * np.log10(value))


def _solo_graph(
    *, start: int = 2 * SECOND, end: int = 4 * SECOND, score: int = 800
) -> ActivityGraph:
    """One speaker, five silent lavs, with silence before and after."""
    return a_graph(
        candidates=[a_candidate("tx-a", start, end, score_permille=score)],
        tracks=_tracks(),
        duration_samples=6 * SECOND,
    )


class TestSolo:
    """ "After the attack interval, a solo speaker's channel gain dominates every inactive
    channel by at least the configured attenuation margin."" """

    @pytest.mark.parametrize("score", [1000, 800, 300, 0])
    def test_a_solo_speaker_dominates_every_silent_channel(self, score: int) -> None:
        """Including at score zero, which is the point of the `min_active_share` floor.

        With a single floor the dominance would scale with the score and this criterion would
        hold on the fixture and fail on a session where a genuine speaker was recorded badly —
        which is exactly the person the level correction exists for.
        """
        settings = EnvelopeConfig()
        _, applied = envelope(_solo_graph(score=score), settings=settings)
        frames_per_second = settings.control_rate_hz
        settled = applied[2 * frames_per_second + settings.attack_ms : 4 * frames_per_second]

        margin = _db(settled[:, 0].min()) - _db(settled[:, 1:].max())
        assert margin >= settings.solo_attenuation_margin_db

    def test_the_margin_holds_at_both_extremes_of_the_correction_clamp(self) -> None:
        """The worst case the achievability validator budgets for: the speaker cut by the full
        clamp while a silent neighbour is lifted by it.

        Asserting the *share* instead would miss this entirely — the share is identical in
        every one of these three cases.
        """
        settings = EnvelopeConfig()
        clamp = round(settings.max_level_correction_db * 100)
        _, applied = envelope(
            _solo_graph(),
            settings=settings,
            corrections=_corrected({"tx-a": -clamp, "tx-b": clamp}),
        )
        settled = applied[2 * settings.control_rate_hz + settings.attack_ms :][
            : 2 * settings.control_rate_hz - settings.attack_ms
        ]
        margin = _db(settled[:, 0].min()) - _db(settled[:, 1:].max())
        assert margin >= settings.solo_attenuation_margin_db

    def test_the_configured_margin_is_achievable_from_the_floors_alone(self) -> None:
        """The bound the validator computes, verified against the envelope that must meet it.

        This is the difference between a criterion that holds and one that cannot fail to: the
        arithmetic below uses no fixture at all.
        """
        settings = EnvelopeConfig()
        separation = _db(settings.min_active_share / settings.room_tone_share)
        guaranteed = separation - 2.0 * settings.max_level_correction_db
        assert guaranteed >= settings.solo_attenuation_margin_db

        _, applied = envelope(_solo_graph(score=0), settings=settings)
        settled = applied[3 * settings.control_rate_hz]
        assert _db(settled[0]) - _db(settled[1:].max()) == pytest.approx(separation, abs=0.01)

    def test_there_is_no_dominance_before_the_speaker_starts(self) -> None:
        """ "After the attack interval" is a claim about *when*, so the frame before the
        candidate opens is asserted too.

        Deliberately not "and not before the attack finishes": with the shipped floors the
        margin is cleared within a frame or two of the onset, which is the *good* direction —
        the criterion is a lower bound on dominance, not a schedule. Writing it the other way
        would pin an accident of the floor ratio and fail the moment OQ-019 moves one.
        """
        settings = EnvelopeConfig()
        _, applied = envelope(_solo_graph(), settings=settings)
        onset = 2 * settings.control_rate_hz

        before = applied[onset - 1]
        assert _db(before[0]) - _db(before[1:].max()) == pytest.approx(0.0)

        settled = applied[onset + settings.attack_ms]
        assert _db(settled[0]) - _db(settled[1:].max()) >= settings.solo_attenuation_margin_db


class TestGenuineOverlap:
    """ "During genuine two-person overlap, both active source channels retain nontrivial
    audible gain."" """

    @staticmethod
    def _two_speakers(scores: tuple[int, int]) -> ActivityGraph:
        return a_graph(
            candidates=[
                a_candidate("tx-d", 2 * SECOND, 4 * SECOND, score_permille=scores[0]),
                a_candidate("tx-e", 2 * SECOND, 4 * SECOND, score_permille=scores[1]),
            ],
            tracks=_tracks(),
            duration_samples=6 * SECOND,
        )

    @pytest.mark.parametrize("scores", [(900, 900), (900, 100), (0, 0)])
    def test_both_speakers_keep_nontrivial_gain(self, scores: tuple[int, int]) -> None:
        """Including the badly-scored pair. The quieter of two genuine speakers is the person
        M3's veto exists to protect, and dropping them here would undo that at the last step."""
        settings = EnvelopeConfig()
        _, applied = envelope(self._two_speakers(scores), settings=settings)
        settled = applied[3 * settings.control_rate_hz]
        assert _db(settled[3]) >= settings.overlap_min_gain_db
        assert _db(settled[4]) >= settings.overlap_min_gain_db

    def test_both_keep_nontrivial_gain_at_the_worst_correction(self) -> None:
        settings = EnvelopeConfig()
        clamp = round(settings.max_level_correction_db * 100)
        _, applied = envelope(
            self._two_speakers((0, 0)),
            settings=settings,
            corrections=_corrected({"tx-d": -clamp, "tx-e": -clamp}),
        )
        settled = applied[3 * settings.control_rate_hz]
        assert _db(settled[3]) >= settings.overlap_min_gain_db
        assert _db(settled[4]) >= settings.overlap_min_gain_db

    def test_an_ambiguous_candidate_is_as_eligible_as_any_other(self) -> None:
        """ADR-0014: `ambiguous` marks what the *track-level veto* kept, which is the least
        obvious bleed case there is, not the most.

        M5's charter said the opposite before its plan review; this is the sentence that stops
        it coming back. The two graphs differ in one boolean and must mix identically.
        """
        plain = self._two_speakers((900, 400))
        flagged = a_graph(
            candidates=[
                a_candidate("tx-d", 2 * SECOND, 4 * SECOND, score_permille=900),
                a_candidate("tx-e", 2 * SECOND, 4 * SECOND, score_permille=400, ambiguous=True),
            ],
            tracks=_tracks(),
            duration_samples=6 * SECOND,
        )
        assert np.array_equal(envelope(plain)[1], envelope(flagged)[1])

    def test_the_worst_admissible_pair_still_clears_the_floor(self) -> None:
        """The cross-product the other tests here miss, and the one that failed.

        Asymmetric scores were tested without an adverse correction, and an adverse correction
        only with equal scores. Both at once — a speaker scoring zero beside one scoring 1000,
        cut by the full clamp — is the worst case the rule admits, and it lands at -15.66 dB.
        The shipped `overlap_min_gain_db` was -15, so the criterion held on the combinations
        the tests happened to use rather than on the rule. Found by M5's code review; the
        default is now derived from `guaranteed_overlap_gain_db` instead of estimated.
        """
        settings = EnvelopeConfig()
        clamp = round(settings.max_level_correction_db * 100)
        _, applied = envelope(
            self._two_speakers((1000, 0)),
            settings=settings,
            corrections=_corrected({"tx-e": -clamp}),
        )
        settled = applied[3 * settings.control_rate_hz]
        assert _db(settled[4]) >= settings.overlap_min_gain_db

    def test_the_measured_worst_case_is_the_bound_the_validator_computes(self) -> None:
        """The arithmetic and the envelope agree, so neither can drift without the other."""
        settings = EnvelopeConfig()
        clamp = round(settings.max_level_correction_db * 100)
        _, applied = envelope(
            self._two_speakers((1000, 0)),
            settings=settings,
            corrections=_corrected({"tx-e": -clamp}),
        )
        settled = applied[3 * settings.control_rate_hz]
        assert _db(settled[4]) == pytest.approx(
            settings.guaranteed_overlap_gain_db(len(TRACKS)), abs=0.01
        )

    def test_the_louder_speaker_still_leads(self) -> None:
        """Nontrivial is a floor, not equality: the score still orders the two."""
        _, applied = envelope(self._two_speakers((900, 200)))
        settled = applied[3 * EnvelopeConfig().control_rate_hz]
        assert settled[3] > settled[4]


class TestSilence:
    """ "During silence or uncertainty, blend low-level room tone without allowing six tracks
    of noise to add coherently."" """

    def test_every_channel_holds_an_equal_share_when_nobody_is_speaking(self) -> None:
        shares, _ = envelope(a_graph(tracks=_tracks(), duration_samples=2 * SECOND))
        assert shares == pytest.approx(1.0 / len(TRACKS))

    def test_six_independent_noise_floors_sum_to_one_over_root_n(self) -> None:
        """The arithmetic the plan first got wrong, kept as a test on constructed inputs.

        N equal gains of 1/N over *independent* equal-power noise give 1/sqrt(N) of one
        track's RMS — not 1/N. Asserting 1/N would have pushed an extra 4 dB of attenuation
        into the mixer to make a wrong expectation pass.
        """
        shares, _ = envelope(a_graph(tracks=_tracks(), duration_samples=SECOND))
        gains = shares[0]
        rng = np.random.default_rng(20260802)
        noise = rng.standard_normal((len(TRACKS), 200_000))
        one_track = float(np.sqrt(np.mean(noise[0] ** 2)))
        mixed = float(np.sqrt(np.mean((gains @ noise) ** 2)))
        assert mixed / one_track == pytest.approx(1.0 / np.sqrt(len(TRACKS)), rel=0.02)

    def test_six_perfectly_correlated_floors_cannot_exceed_one_track(self) -> None:
        """The other half, and the one the normalized share actually guarantees: the gains sum
        to 1, so identical inputs sum to exactly one of them and never more."""
        shares, _ = envelope(a_graph(tracks=_tracks(), duration_samples=SECOND))
        gains = shares[0]
        rng = np.random.default_rng(20260802)
        one = rng.standard_normal(200_000)
        identical = np.tile(one, (len(TRACKS), 1))
        mixed = gains @ identical
        assert float(np.sqrt(np.mean(mixed**2))) == pytest.approx(float(np.sqrt(np.mean(one**2))))


class TestBleedIsNotPromoted:
    """ "Obvious correlated bleed is not promoted on two channels simultaneously."" """

    @staticmethod
    def _with_bleed(decision: str) -> ActivityGraph:
        """tx-a speaking, with tx-b carrying a copy that M3 either suppressed or did not."""
        speaker = a_candidate("tx-a", 2 * SECOND, 4 * SECOND, score_permille=900)
        bleed_kwargs: dict[str, object] = {"score_permille": 200}
        if decision == "suppressed":
            bleed_kwargs |= {
                "decision": "suppressed",
                "suppressed_by_candidate_id": speaker.candidate_id,
                "evidence": [
                    {
                        "other_candidate_id": speaker.candidate_id,
                        "other_track_id": "tx-a",
                        "overlap_start_sample": 2 * SECOND,
                        "overlap_end_sample": 4 * SECOND,
                        "compared_derivative_samples": 32_000,
                        "correlation_permille": 900,
                        "lag_derivative_samples": 48,
                        "score_margin_permille": 700,
                        "level_delta_mb": 2000,
                        "outcome": "suppresses",
                    }
                ],
            }
        return a_graph(
            candidates=[
                speaker,
                a_candidate("tx-b", 2 * SECOND + 144, 4 * SECOND + 144, **bleed_kwargs),
            ],
            tracks=_tracks(),
            duration_samples=6 * SECOND,
        )

    def test_a_suppressed_copy_stays_at_the_room_tone_share(self) -> None:
        settings = EnvelopeConfig()
        shares, _ = envelope(self._with_bleed("suppressed"), settings=settings)
        settled = shares[3 * settings.control_rate_hz]
        silent = settled[2:]
        assert settled[1] == pytest.approx(silent.max())
        assert _db(settled[0]) - _db(settled[1]) >= settings.solo_attenuation_margin_db

    def test_the_same_graph_with_the_copy_retained_does_promote_it(self) -> None:
        """The contrast, so the assertion above is about M3's *decision* rather than about
        these particular numbers happening to work out.

        Without it, a mixer that ignored the graph entirely and gated on level would pass.
        """
        settings = EnvelopeConfig()
        shares, _ = envelope(self._with_bleed("retained"), settings=settings)
        settled = shares[3 * settings.control_rate_hz]
        assert settled[1] > settled[2] * 10
        assert _db(settled[0]) - _db(settled[1]) < settings.solo_attenuation_margin_db

    def test_a_suppressed_candidate_produces_no_span_at_all(self) -> None:
        spans = active_spans(
            self._with_bleed("suppressed"), settings=EnvelopeConfig(), track_ids=TRACKS
        )
        assert [span.track_index for span in spans] == [0]


class TestSlew:
    """ "Gain envelopes contain no discontinuities and do not exceed configured attack,
    release, or maximum-slew limits."" """

    @staticmethod
    def _busy() -> ActivityGraph:
        """Overlapping onsets and offsets, so the slew is exercised in both directions at
        once and across a chunk boundary rather than only at one clean edge."""
        return a_graph(
            candidates=[
                a_candidate("tx-a", 1 * SECOND, 2 * SECOND),
                a_candidate("tx-b", 2 * SECOND, 2 * SECOND + 4800),
                a_candidate("tx-c", SECOND // 2, 5 * SECOND),
            ],
            tracks=_tracks(),
            duration_samples=6 * SECOND,
        )

    def test_no_frame_moves_faster_than_the_configured_attack_or_release(self) -> None:
        """Over every frame of every track, not a sampled one. A slew violation is a click,
        and a click is one frame.

        Asserted on the *presence* — the smoothed control signal — because that is where the
        limit is a per-frame bound. A track's normalized share can legitimately move faster
        than its own ramp when another track's presence collapses beneath it, so a bound on
        the share would either be false or would have to be loosened until it proved nothing.
        The share's own guarantee is continuity, which the two tests below cover.
        """
        settings = EnvelopeConfig()
        stream = EnvelopeStream(
            self._busy(),
            settings=settings,
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        presence = np.concatenate(
            [np.zeros((1, len(TRACKS)))]
            + [chunk.presence for chunk in stream.chunks(chunk_frames=700)]
        )
        steps = np.diff(presence, axis=0)
        rise = 1.0 / (settings.attack_ms * settings.control_rate_hz / 1000)
        fall = 1.0 / (settings.release_ms * settings.control_rate_hz / 1000)
        assert steps.max() <= rise + 1e-9
        assert steps.min() >= -fall - 1e-9
        assert steps.max() == pytest.approx(rise)
        assert steps.min() == pytest.approx(-fall)

    def test_the_attack_reaches_full_weight_in_exactly_the_configured_frames(self) -> None:
        settings = EnvelopeConfig(attack_ms=20)
        _, applied = envelope(_solo_graph(score=1000), settings=settings)
        onset = 2 * settings.control_rate_hz
        frames = settings.attack_ms * settings.control_rate_hz // 1000
        rising = applied[onset : onset + frames, 0]
        assert np.all(np.diff(rising) > 0)
        assert applied[onset + frames, 0] == pytest.approx(applied[onset + frames + 50, 0])

    def test_the_release_is_slower_than_the_attack_by_their_configured_ratio(self) -> None:
        settings = EnvelopeConfig(attack_ms=10, release_ms=300)
        _, applied = envelope(_solo_graph(score=1000), settings=settings)
        onset, offset = 2 * settings.control_rate_hz, 4 * settings.control_rate_hz
        rise = applied[onset + 5, 0] - applied[onset, 0]
        fall = applied[offset, 0] - applied[offset + 5, 0]
        assert rise > fall * 10

    def test_the_per_sample_gain_is_continuous_across_frame_boundaries(self) -> None:
        """Interpolation makes this structural; the test is that the structure is used.

        A renderer that held each control frame's value for its whole 48 samples would produce
        a staircase — inaudible in isolation and a buzz once six of them are moving.
        """
        settings = EnvelopeConfig()
        stream = EnvelopeStream(
            _solo_graph(),
            settings=settings,
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        chunk = next(iter(stream.chunks(chunk_frames=3000)))
        gains = expand(chunk, samples_per_frame=stream.samples_per_frame, n_samples=3000 * 48)
        steps = np.abs(np.diff(gains, axis=0))
        per_frame = np.abs(np.diff(chunk.applied, axis=0)).max()
        assert steps.max() <= per_frame / stream.samples_per_frame + 1e-12

    def test_expansion_lands_exactly_on_each_frames_value_at_its_last_sample(self) -> None:
        """The ramp reaches the frame's value on the frame's last sample, which is what makes
        consecutive frames join without a step."""
        settings = EnvelopeConfig()
        stream = EnvelopeStream(
            _solo_graph(),
            settings=settings,
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        chunk = next(iter(stream.chunks(chunk_frames=100)))
        spf = stream.samples_per_frame
        gains = expand(chunk, samples_per_frame=spf, n_samples=100 * spf)
        for frame in (0, 37, 99):
            assert gains[(frame + 1) * spf - 1] == pytest.approx(chunk.applied[frame])


class TestTheBoundedGainInvariant:
    """ "The chosen normalized/equal-power gain invariant remains bounded at every sample or
    control frame, including silence and transitions."" """

    @pytest.mark.parametrize(
        "graph_factory",
        [
            pytest.param(
                lambda: a_graph(tracks=_tracks(), duration_samples=2 * SECOND), id="silence"
            ),
            pytest.param(_solo_graph, id="solo"),
            pytest.param(
                lambda: a_graph(
                    candidates=[
                        a_candidate(t, 2 * SECOND, 4 * SECOND)
                        for t in ("tx-a", "tx-b", "tx-c", "tx-d", "tx-e", "tx-f")
                    ],
                    tracks=_tracks(),
                    duration_samples=6 * SECOND,
                ),
                id="everyone-at-once",
            ),
        ],
    )
    def test_the_share_sums_to_one_at_every_frame(self, graph_factory: object) -> None:
        shares, _ = envelope(graph_factory())  # type: ignore[operator]
        assert np.sum(shares, axis=-1) == pytest.approx(1.0, abs=1e-12)

    def test_the_applied_coefficients_stay_inside_the_correction_clamp(self) -> None:
        """The bound that means something audible. `sum(share) == 1` says nothing about it:
        six tracks each corrected by +6 dB sum to 2.0 while the share still reports 1.0."""
        settings = EnvelopeConfig()
        clamp = round(settings.max_level_correction_db * 100)
        _, applied = envelope(
            _solo_graph(),
            settings=settings,
            corrections=_corrected(dict.fromkeys(TRACKS, clamp)),
        )
        limit = 10.0 ** (settings.max_level_correction_db / 20.0)
        totals = np.sum(applied, axis=-1)
        assert totals.max() == pytest.approx(limit)
        assert totals.min() >= 1.0 / limit

    def test_the_runtime_check_fires_when_the_share_does_not_normalize(self) -> None:
        """A check that cannot fail is decoration.

        Driven by breaking the sharing law from outside, which is the only thing that can
        break it — every other input is bounded by the configuration validator.
        """
        stream = EnvelopeStream(
            _solo_graph(),
            settings=EnvelopeConfig(),
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        stream._share = lambda presence: np.maximum(presence, 0.01)  # type: ignore[method-assign]
        with pytest.raises(EnvelopeError, match="sums to"):
            list(stream.chunks(chunk_frames=100))

    @pytest.mark.parametrize("clamp_db", [0.015, 0.025, 6.006, 6.015])
    def test_a_fractional_clamp_does_not_make_the_check_fire_on_its_own_corrections(
        self, clamp_db: float
    ) -> None:
        """The clamp is one number, and it used to be spelled two ways.

        `levels` rounds it to whole millibels, the way every level in this project is carried;
        the checker converted the raw dB instead. For any clamp whose hundredths round *up* —
        `round(0.015 * 100) == 2` — a track's own permitted correction then exceeded the bound
        the checker enforces, and an ordinary two-speaker session failed the mix stage with an
        invariant violation. Found by M5's verify phase.
        """
        settings = EnvelopeConfig(max_level_correction_db=clamp_db)
        corrections = level_corrections(
            a_graph(
                candidates=[a_candidate("tx-a", 0, 2 * SECOND)],
                tracks=[
                    a_track("tx-a", speech_reference_mbfs=-4000),
                    a_track("tx-b", speech_reference_mbfs=-2000),
                    a_track("tx-c", speech_reference_mbfs=-2000),
                ],
                duration_samples=3 * SECOND,
            ),
            settings=settings,
        )
        graph = a_graph(
            candidates=[a_candidate("tx-a", 0, 2 * SECOND)],
            tracks=[
                a_track("tx-a", speech_reference_mbfs=-4000),
                a_track("tx-b", speech_reference_mbfs=-2000),
                a_track("tx-c", speech_reference_mbfs=-2000),
            ],
            duration_samples=3 * SECOND,
        )
        ids = ("tx-a", "tx-b", "tx-c")
        stream = EnvelopeStream(graph, settings=settings, corrections=corrections, track_ids=ids)
        applied = np.concatenate([chunk.applied for chunk in stream.chunks(chunk_frames=500)])
        limit = 10.0 ** (round(clamp_db * 100) / 2000.0)
        assert np.sum(applied, axis=-1).max() <= limit + 1e-9

    def test_the_runtime_check_fires_when_a_correction_escapes_the_clamp(self) -> None:
        stream = EnvelopeStream(
            _solo_graph(),
            settings=EnvelopeConfig(),
            corrections=_corrected(dict.fromkeys(TRACKS, 0)),
            track_ids=TRACKS,
        )
        stream._corrections = np.full(len(TRACKS), 100.0)
        with pytest.raises(EnvelopeError, match="bounds nothing audible"):
            list(stream.chunks(chunk_frames=100))


class TestTheEnvelopeIsBounded:
    """INV-07 at the unit level. `tests/test_memory.py` proves it over the composed path."""

    def test_the_same_envelope_comes_out_however_it_is_partitioned(self) -> None:
        """Slew state crosses chunk boundaries, exactly as the decimator's filter state does.

        Without this the envelope would depend on the caller's window size, and a mix would
        stop being reproducible from its own inputs.
        """
        graph = _solo_graph()
        whole = envelope(graph, chunk_frames=100_000)[1]
        for chunk_frames in (1, 7, 48, 1000, 4999):
            assert np.array_equal(envelope(graph, chunk_frames=chunk_frames)[1], whole)

    def test_overlapping_spans_on_one_track_partition_identically_too(self) -> None:
        """The case the test above cannot see, because a solo graph has one span per track.

        `_targets` used to stop scanning a track at the first span running past the chunk end,
        so a second, overlapping span was skipped in that chunk and applied in the next: the
        envelope depended on the caller's window size, which the mix's cache identity does not
        carry. M3's merge keeps a track's retained candidates disjoint, so this is unreachable
        through the pipeline — which is the reason to assert it structurally rather than to
        rely on an upstream promise the artifact does not make. Found by M5's verify phase.
        """
        graph = a_graph(
            candidates=[
                a_candidate("tx-a", 0, 2 * SECOND, score_permille=0),
                a_candidate("tx-a", SECOND // 2, 3 * SECOND // 2, score_permille=1000),
            ],
            tracks=_tracks(),
            duration_samples=4 * SECOND,
        )
        whole = envelope(graph, chunk_frames=100_000)[1]
        for chunk_frames in (1, 7, 48, 1000, 4999):
            assert np.array_equal(envelope(graph, chunk_frames=chunk_frames)[1], whole), (
                chunk_frames
            )

    def test_no_chunk_exceeds_the_size_asked_for(self) -> None:
        stream = EnvelopeStream(
            _solo_graph(),
            settings=EnvelopeConfig(),
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        assert all(chunk.n_frames <= 250 for chunk in stream.chunks(chunk_frames=250))

    def test_the_chunks_tile_the_session_exactly(self) -> None:
        settings = EnvelopeConfig()
        graph = a_graph(tracks=_tracks(), duration_samples=6 * SECOND + 17)
        stream = EnvelopeStream(
            graph, settings=settings, corrections=_uncorrected(), track_ids=TRACKS
        )
        chunks = list(stream.chunks(chunk_frames=700))
        assert chunks[0].start_frame == 0
        assert sum(chunk.n_frames for chunk in chunks) == stream.total_frames
        assert stream.total_frames == -(-(6 * SECOND + 17) // stream.samples_per_frame)

    def test_a_zero_length_session_produces_nothing(self) -> None:
        stream = EnvelopeStream(
            a_graph(tracks=_tracks(), duration_samples=0),
            settings=EnvelopeConfig(),
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        assert list(stream.chunks(chunk_frames=100)) == []

    def test_a_nonpositive_chunk_size_is_refused(self) -> None:
        stream = EnvelopeStream(
            _solo_graph(),
            settings=EnvelopeConfig(),
            corrections=_uncorrected(),
            track_ids=TRACKS,
        )
        with pytest.raises(ValueError, match="chunk_frames must be positive"):
            list(stream.chunks(chunk_frames=0))

    def test_a_session_with_no_tracks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one track"):
            EnvelopeStream(
                a_graph(tracks=_tracks()),
                settings=EnvelopeConfig(),
                corrections=_uncorrected(()),
                track_ids=(),
            )


class TestTheControlGrid:
    def test_a_candidates_frames_cover_its_samples(self) -> None:
        """Start floors, end ceils — the same covering rule as the 48/16 kHz mapping.

        Rounding both ends alike would clip a few milliseconds off every utterance, and the
        slew limit would then turn the clipped onset into a fade-in over the first word.
        """
        assert frame_interval(0, 1, 48) == (0, 1)
        assert frame_interval(47, 49, 48) == (0, 2)
        assert frame_interval(48, 96, 48) == (1, 2)
        assert frame_interval(1, 47, 48) == (0, 1)

    def test_a_candidate_ending_on_a_boundary_does_not_open_the_next_frame(self) -> None:
        assert frame_interval(0, 96, 48) == (0, 2)

    def test_the_span_weight_is_the_score_lifted_onto_the_active_floor(self) -> None:
        settings = EnvelopeConfig()
        spans = active_spans(
            a_graph(
                candidates=[a_candidate("tx-a", 0, SECOND, score_permille=500)],
                tracks=_tracks(),
            ),
            settings=settings,
            track_ids=TRACKS,
        )
        expected = settings.min_active_share + (1.0 - settings.min_active_share) * 0.5
        assert spans[0].weight == pytest.approx(expected)

    def test_a_candidate_on_a_track_the_mix_does_not_carry_is_ignored(self) -> None:
        """A track with no working audio has no reader, so its candidates cannot be mixed.
        Silently dropping the *span* is right; silently dropping the track from the share
        would change every other track's gain, so the caller decides the track list."""
        spans = active_spans(
            a_graph(
                candidates=[a_candidate("tx-a", 0, SECOND)],
                tracks=_tracks(),
            ),
            settings=EnvelopeConfig(),
            track_ids=("tx-b", "tx-c"),
        )
        assert spans == ()

    def test_two_overlapping_spans_on_one_track_take_the_louder(self) -> None:
        """M3 merges a track's candidates so this should not arise; the artifact does not
        promise it, and arriving second is not a reason to win."""
        settings = EnvelopeConfig()
        graph = a_graph(
            candidates=[
                a_candidate("tx-a", 0, 2 * SECOND, score_permille=1000),
                a_candidate("tx-a", SECOND, 3 * SECOND, score_permille=0),
            ],
            tracks=_tracks(),
            duration_samples=4 * SECOND,
        )
        _, applied = envelope(graph, settings=settings)
        held = applied[int(1.5 * settings.control_rate_hz), 0]
        assert held == pytest.approx(applied[int(0.9 * settings.control_rate_hz), 0])


class TestAgainstARealGraph:
    """The same criteria, on a graph this milestone did not hand-build."""

    def test_the_canonical_fixtures_graph_mixes_within_every_bound(
        self, canonical_activity_graph: ActivityGraph
    ) -> None:
        settings = EnvelopeConfig()
        track_ids = tuple(track.track_id for track in canonical_activity_graph.tracks)
        corrections = level_corrections(canonical_activity_graph, settings=settings)
        shares, applied = envelope(
            canonical_activity_graph,
            settings=settings,
            corrections=corrections,
            track_ids=track_ids,
        )
        assert np.sum(shares, axis=-1) == pytest.approx(1.0, abs=1e-12)
        limit = 10.0 ** (settings.max_level_correction_db / 20.0)
        assert np.sum(applied, axis=-1).max() <= limit + 1e-12

    def test_alices_utterance_dominates_the_four_tracks_it_bled_into(
        self, canonical_activity_graph: ActivityGraph
    ) -> None:
        """The fixture's own scenario, end to end through the envelope: tx-a speaks at
        249600 and four other lavs hear her. M3 suppressed all four copies, so the mix must
        show one channel up and five down."""
        settings = EnvelopeConfig()
        track_ids = tuple(track.track_id for track in canonical_activity_graph.tracks)
        alice = track_ids.index("tx-a")
        _, applied = envelope(
            canonical_activity_graph,
            settings=settings,
            corrections=level_corrections(canonical_activity_graph, settings=settings),
            track_ids=track_ids,
        )
        frame = (249_600 + 24_000) * settings.control_rate_hz // 48_000
        others = np.delete(applied[frame], alice)
        assert _db(applied[frame][alice]) - _db(others.max()) >= (
            settings.solo_attenuation_margin_db
        )

    def test_dan_and_erin_both_survive_their_simultaneous_utterance(
        self, canonical_activity_graph: ActivityGraph
    ) -> None:
        settings = EnvelopeConfig()
        track_ids = tuple(track.track_id for track in canonical_activity_graph.tracks)
        _, applied = envelope(
            canonical_activity_graph,
            settings=settings,
            corrections=level_corrections(canonical_activity_graph, settings=settings),
            track_ids=track_ids,
        )
        frame = (326_400 + 24_000) * settings.control_rate_hz // 48_000
        for track_id in ("tx-d", "tx-e"):
            assert _db(applied[frame][track_ids.index(track_id)]) >= (settings.overlap_min_gain_db)
