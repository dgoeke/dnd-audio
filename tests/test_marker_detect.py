"""The detector, against channels that do to the marker what a room and a phone do.

**Ground truth here is independent of the detector.** Every fixture places the marker at a
sample this file chose, and the assertion is that exact integer — never a value the detector
produced, and never "close enough". That is the difference between a regression test and a
change detector, and it is why the channel models below are all built from operations that
provably do not move the direct arrival: zero-phase filtering, additive noise, gain scaling,
and reverberation as *delayed copies added after* the direct sound.

That last point cost a debugging session and is worth stating. An ordinary IIR band-pass
(`sosfilt`) has group delay, so filtering shifts the whole signal by a few samples and every
"exact sample" assertion silently measures the filter instead. `filtfilt` runs it forwards and
backwards, which is genuinely zero-phase.

Every test runs against **all three candidates**, so whichever the bench selects is already
covered and a different winner is a parameter rather than a re-baseline (ADR-0042).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from scipy.signal import filtfilt, firwin, resample_poly

from dnd_audio.marker import MARKER_SAMPLE_RATE
from dnd_audio.marker.detect import (
    DetectorThresholds,
    OccurrenceCeilingError,
    _assemble,
    _is_locally_ambiguous,
    detect_occurrences,
    to_permille,
    to_permille_array,
)
from dnd_audio.marker.spec import MARKER_SPECS, MarkerSpec
from dnd_audio.marker.synth import marker_samples

ALL_SPECS = pytest.mark.parametrize("spec", MARKER_SPECS.values(), ids=list(MARKER_SPECS))

#: Where the fixtures put the marker's first sample. Arbitrary, prime-ish, and deliberately
#: not a multiple of any block size, so a passing test cannot be an artefact of alignment.
PLACEMENT = 71_317

#: Silence after the marker, so the searched interval is never exactly the marker's length.
TAIL = 30_000


@dataclass
class ArrayReader:
    """A `WindowReader` over one array, recording what was asked of it.

    Reads outside the array return silence rather than raising, matching
    `TrackReader.read`'s contract: a track that stopped early is still readable to the
    session's aligned duration.
    """

    samples: npt.NDArray[np.float32]
    largest_read: int = 0
    reads: int = 0

    def read(self, start_sample: int, n_samples: int, /) -> npt.NDArray[np.float32]:
        self.largest_read = max(self.largest_read, n_samples)
        self.reads += 1
        window = np.zeros(n_samples, dtype=np.float32)
        low = max(0, start_sample)
        high = min(self.samples.size, start_sample + n_samples)
        if high > low:
            window[low - start_sample : high - start_sample] = self.samples[low:high]
        return window


def place(spec: MarkerSpec, *, at: int = PLACEMENT, gain: float = 1.0) -> npt.NDArray[np.float32]:
    """The marker at a known sample, in silence. The baseline every channel starts from."""
    marker = marker_samples(spec).astype(np.float64) / 32768.0
    track = np.zeros(at + marker.size + TAIL, dtype=np.float64)
    track[at : at + marker.size] = marker * gain
    return track.astype(np.float32)


def anchor_of(spec: MarkerSpec, *, at: int = PLACEMENT) -> int:
    """Where the anchor lands for a marker placed at ``at`` — computed, not detected."""
    return at + spec.anchor_sample


def found(track: npt.NDArray[np.float32], spec: MarkerSpec, **kwargs: object) -> list[int]:
    """Anchor samples of every accepted occurrence, in order."""
    reader = ArrayReader(track)
    occurrences = detect_occurrences(reader, spec, interval=(0, track.size), **kwargs)  # type: ignore[arg-type]
    return [occurrence.anchor_sample for occurrence in occurrences]


def band_limit(track: npt.NDArray[np.float32], low: int, high: int) -> npt.NDArray[np.float32]:
    """Zero-phase band-pass. A lav capsule and a phone speaker, crudely and honestly."""
    taps = firwin(301, [low, high], pass_zero=False, fs=MARKER_SAMPLE_RATE)
    return np.asarray(filtfilt(taps, [1.0], track.astype(np.float64)), dtype=np.float32)


def reverberate(track: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Three attenuated reflections. Added *after* the direct sound, so the anchor cannot move."""
    out = track.astype(np.float64).copy()
    for delay, gain in ((1_200, 0.40), (2_900, 0.25), (5_100, 0.15)):
        out[delay:] += gain * track[:-delay]
    return out.astype(np.float32)


class TestAnExactDelayIsRecoveredExactly:
    """The core claim: an integer-sample anchor, from an independently declared position."""

    @ALL_SPECS
    def test_a_clean_marker_is_found_at_the_sample_it_was_placed(self, spec: MarkerSpec) -> None:
        assert found(place(spec), spec) == [anchor_of(spec)]

    @ALL_SPECS
    @pytest.mark.parametrize("offset", [0, 1, 7, 4_799, 4_801, 65_535, 65_536, 65_537])
    def test_every_placement_including_block_boundaries(
        self, spec: MarkerSpec, offset: int
    ) -> None:
        """Stream seams. The default block is 65536, so those three offsets straddle one.

        An occurrence lost only at certain search offsets is the worst shape this bug can
        take: it passes every test whose fixture happens to sit elsewhere.
        """
        assert found(place(spec, at=offset), spec) == [anchor_of(spec, at=offset)]

    @ALL_SPECS
    def test_the_answer_does_not_depend_on_the_block_size(self, spec: MarkerSpec) -> None:
        longest = max(chirp.duration_samples for chirp in spec.chirps)
        track = place(spec)
        answers = {
            block: found(track, spec, block_samples=block)
            for block in (longest, longest + 1, 1 << 14, 1 << 16, 1 << 18)
        }
        assert len(set(map(tuple, answers.values()))) == 1, answers
        assert answers[1 << 16] == [anchor_of(spec)]

    @ALL_SPECS
    def test_a_block_shorter_than_a_template_is_refused_rather_than_silently_empty(
        self, spec: MarkerSpec
    ) -> None:
        """Found while building this: it returned no detections at all.

        Which on a real session reads exactly like "nobody played the marker".
        """
        longest = max(chirp.duration_samples for chirp in spec.chirps)
        with pytest.raises(ValueError, match="below the longest chirp template"):
            found(place(spec), spec, block_samples=longest - 1)


class TestTheChannelsARoomApplies:
    """Everything between the phone's speaker and the recorded file."""

    @ALL_SPECS
    @pytest.mark.parametrize("gain", [1.0, 0.3, 0.05, 0.01])
    def test_level_does_not_move_the_anchor(self, spec: MarkerSpec, gain: float) -> None:
        """Normalized correlation is why: a far lav and a near one must score alike."""
        assert found(place(spec, gain=gain), spec) == [anchor_of(spec)]

    @ALL_SPECS
    def test_band_limiting_a_lav_would_apply(self, spec: MarkerSpec) -> None:
        assert found(band_limit(place(spec, gain=0.5), 300, 7_000), spec) == [anchor_of(spec)]

    @ALL_SPECS
    def test_reverberation(self, spec: MarkerSpec) -> None:
        assert found(reverberate(place(spec, gain=0.4)), spec) == [anchor_of(spec)]

    @ALL_SPECS
    def test_a_quiet_marker_under_noise(self, spec: MarkerSpec) -> None:
        rng = np.random.default_rng(20260805)
        track = place(spec, gain=0.05).astype(np.float64)
        track += rng.normal(0.0, 0.004, track.size)
        assert found(track.astype(np.float32), spec) == [anchor_of(spec)]

    @ALL_SPECS
    def test_the_whole_chain_at_once(self, spec: MarkerSpec) -> None:
        """Band limit, reverberate, attenuate and add noise — the realistic case."""
        rng = np.random.default_rng(4)
        track = reverberate(band_limit(place(spec, gain=0.25), 300, 7_000)).astype(np.float64)
        track += rng.normal(0.0, 0.003, track.size)
        assert found(track.astype(np.float32), spec) == [anchor_of(spec)]

    @ALL_SPECS
    def test_moderate_clipping_still_locates_it_and_says_so(self, spec: MarkerSpec) -> None:
        """The nearest lav. Position stays usable; the score stops being trustworthy."""
        track = np.clip(place(spec, gain=4.0).astype(np.float64), -1.0, 1.0).astype(np.float32)
        reader = ArrayReader(track)
        occurrences = detect_occurrences(reader, spec, interval=(0, track.size))
        assert [item.anchor_sample for item in occurrences] == [anchor_of(spec)]
        assert occurrences[0].clipped is True

    @ALL_SPECS
    def test_a_clean_marker_is_not_reported_as_clipped(self, spec: MarkerSpec) -> None:
        """The contrast that makes the previous test mean something."""
        reader = ArrayReader(place(spec))
        occurrences = detect_occurrences(reader, spec, interval=(0, reader.samples.size))
        assert occurrences[0].clipped is False


class TestWhatMustNotBeAccepted:
    """Every fixture here contains something marker-shaped, and none of it is the marker."""

    @ALL_SPECS
    def test_a_reversed_marker_is_rejected(self, spec: MarkerSpec) -> None:
        """The asymmetric gaps arrive in the wrong order. This is what they are for."""
        track = place(spec).astype(np.float64)
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        track[PLACEMENT : PLACEMENT + marker.size] = marker[::-1]
        assert found(track.astype(np.float32), spec) == []

    @ALL_SPECS
    def test_a_truncated_marker_is_rejected(self, spec: MarkerSpec) -> None:
        """Two chirps of three. A partial sequence is not a detection."""
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        cut = spec.chirp_intervals()[-1][0]
        track = np.zeros(PLACEMENT + marker.size + TAIL, dtype=np.float64)
        track[PLACEMENT : PLACEMENT + cut] = marker[:cut]
        assert found(track.astype(np.float32), spec) == []

    @ALL_SPECS
    def test_one_chirp_alone_is_never_a_detection(self, spec: MarkerSpec) -> None:
        """However strong. The charter is explicit and the gaps are the discriminator."""
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        start, end = spec.chirp_intervals()[0]
        track = np.zeros(PLACEMENT + marker.size + TAIL, dtype=np.float64)
        track[PLACEMENT : PLACEMENT + (end - start)] = marker[start:end]
        assert found(track.astype(np.float32), spec) == []

    @ALL_SPECS
    def test_the_right_chirps_with_the_wrong_gaps_are_rejected(self, spec: MarkerSpec) -> None:
        """The strongest negative available: identical energy, identical spectrum, wrong code."""
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        track = np.zeros(PLACEMENT + 4 * marker.size, dtype=np.float64)
        position = PLACEMENT
        wrong_gap = max(spec.gaps_samples) + 4 * DetectorThresholds().gap_tolerance_samples
        for index, (start, end) in enumerate(spec.chirp_intervals()):
            track[position : position + (end - start)] = marker[start:end]
            position += (end - start) + wrong_gap * (index + 1)
        assert found(track.astype(np.float32), spec) == []

    @ALL_SPECS
    def test_speech_shaped_noise_produces_nothing(self, spec: MarkerSpec) -> None:
        """Deterministic, and shaped where a voice lives rather than flat."""
        rng = np.random.default_rng(1_607)
        noise = rng.normal(0.0, 0.25, 8 * MARKER_SAMPLE_RATE)
        track = band_limit(noise.astype(np.float32), 120, 4_000)
        assert found(track, spec) == []

    @ALL_SPECS
    def test_a_music_like_sweep_produces_nothing(self, spec: MarkerSpec) -> None:
        """One long glissando covering the same band — a synthesizer, not the marker."""
        length = 6 * MARKER_SAMPLE_RATE
        time = np.arange(length) / MARKER_SAMPLE_RATE
        sweep = np.sin(
            2 * np.pi * (200 * time + 3_000 * time * time / (length / MARKER_SAMPLE_RATE))
        )
        assert found((sweep * 0.5).astype(np.float32), spec) == []

    @ALL_SPECS
    def test_silence_produces_nothing(self, spec: MarkerSpec) -> None:
        assert found(np.zeros(5 * MARKER_SAMPLE_RATE, dtype=np.float32), spec) == []

    @ALL_SPECS
    def test_a_different_candidate_is_not_this_one(self, spec: MarkerSpec) -> None:
        """Playing cand-b at a bench looking for cand-a must find nothing.

        The bench plays all three; an analyzer that matched any of them would attribute the
        wrong waveform's arrival to the one it was asked about.
        """
        others = [other for other in MARKER_SPECS.values() if other.name != spec.name]
        for other in others:
            if other.chirps[0].duration_samples == spec.chirps[0].duration_samples:
                continue
            assert found(place(other), spec) == []


class TestRepeatsAndAmbiguity:
    """Several plays, and what happens when two are close together."""

    @ALL_SPECS
    def test_three_separated_plays_are_three_occurrences(self, spec: MarkerSpec) -> None:
        """The bench plays each candidate three times. All three must be enumerated."""
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stride = marker.size + 5 * MARKER_SAMPLE_RATE
        track = np.zeros(PLACEMENT + 3 * stride, dtype=np.float64)
        expected = []
        for index in range(3):
            at = PLACEMENT + index * stride
            track[at : at + marker.size] = marker
            expected.append(at + spec.anchor_sample)
        assert found(track.astype(np.float32), spec) == expected

    @ALL_SPECS
    def test_a_reflection_close_behind_does_not_become_a_second_occurrence(
        self, spec: MarkerSpec
    ) -> None:
        """Non-maximum suppression's job: one arrival, not one per reflection."""
        assert found(reverberate(place(spec, gain=0.5)), spec) == [anchor_of(spec)]

    @ALL_SPECS
    def test_the_occurrence_ceiling_fails_rather_than_truncating(self, spec: MarkerSpec) -> None:
        """ADR-0041. A shortened list would look exactly like a session that had that many."""
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stride = marker.size + 20_000
        count = 6
        track = np.zeros(count * stride + TAIL, dtype=np.float64)
        for index in range(count):
            track[index * stride : index * stride + marker.size] = marker

        with pytest.raises(OccurrenceCeilingError) as caught:
            found(
                track.astype(np.float32),
                spec,
                thresholds=DetectorThresholds(max_occurrences_per_track=3),
            )
        assert caught.value.code == "marker_occurrence_ceiling"
        assert "truncated" in str(caught.value)
        # And the same input under the default ceiling really does yield all of them, so the
        # test above is about the ceiling rather than about the detector missing some.
        assert len(found(track.astype(np.float32), spec)) == count

    @ALL_SPECS
    def test_repeated_valid_plays_are_excluded_from_the_local_runner_up(
        self, spec: MarkerSpec
    ) -> None:
        """ADR-0041: other accepted occurrences are not local ambiguity."""
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stride = marker.size + 5 * MARKER_SAMPLE_RATE
        track = np.zeros(PLACEMENT + 2 * stride, dtype=np.float64)
        for index in range(2):
            at = PLACEMENT + index * stride
            track[at : at + marker.size] = marker
        reader = ArrayReader(track.astype(np.float32))
        occurrences = detect_occurrences(reader, spec, interval=(0, reader.samples.size))
        assert len(occurrences) == 2
        assert all(item.runner_up_permille == 0 for item in occurrences)
        assert all(not item.ambiguous for item in occurrences)

    def test_an_unclaimed_local_reflection_reaches_the_ambiguity_outcome(self) -> None:
        """The detector, not only the integer helper, must carry a competing peak through.

        Cand-c's short chirps leave a measurable 2,900-sample reflection outside per-chirp
        NMS but inside sequence NMS. Raising only the separation requirement makes that known
        alternative ambiguous without changing whether the sequence itself is accepted.
        """
        spec = MARKER_SPECS["cand-c"]
        track = reverberate(place(spec, gain=0.5))
        reader = ArrayReader(track)
        occurrences = detect_occurrences(
            reader,
            spec,
            interval=(0, track.size),
            thresholds=DetectorThresholds(min_runner_up_separation_permille=1000),
        )
        assert len(occurrences) == 1
        assert occurrences[0].runner_up_permille > 0
        assert occurrences[0].ambiguous is True


class TestTimingPerturbation:
    """What OQ-029 asks about: playback that is not quite 48 kHz."""

    @staticmethod
    def _stretched(spec: MarkerSpec, ppm: int) -> tuple[npt.NDArray[np.float32], int]:
        """The marker time-stretched by ``ppm``, and where its anchor now sits.

        The expected anchor is computed from the stretch ratio, never read back from the
        detector.
        """
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stretched = np.asarray(resample_poly(marker, 1_000_000 + ppm, 1_000_000), dtype=np.float64)
        track = np.zeros(PLACEMENT + stretched.size + TAIL, dtype=np.float64)
        track[PLACEMENT : PLACEMENT + stretched.size] = stretched
        expected = PLACEMENT + round(spec.anchor_sample * (1_000_000 + ppm) / 1_000_000)
        return track.astype(np.float32), expected

    @ALL_SPECS
    @pytest.mark.parametrize("ppm", [50, 100, 200])
    def test_a_real_clock_difference_costs_no_accuracy(self, spec: MarkerSpec, ppm: int) -> None:
        """The claim that matters: at physical clock offsets the anchor stays sample-exact.

        A phone's DAC and a DJI's ADC are independent crystals, tens of ppm apart; OQ-006
        measured the transmitters' own at about 1 ppm and bounded them at 3. 200 ppm is
        already generous for consumer parts.

        Note what is deliberately *not* modelled here: a browser resampling 48 kHz content
        for a 44.1 kHz device is a rate conversion, not a speed change — the sound still
        lasts as long as it did. Only a genuine clock disagreement stretches the marker in
        time (**OQ-029**).
        """
        track, expected = self._stretched(spec, ppm)
        reader = ArrayReader(track)
        occurrences = detect_occurrences(reader, spec, interval=(0, reader.samples.size))
        assert len(occurrences) == 1
        assert abs(occurrences[0].anchor_sample - expected) <= 2

    @ALL_SPECS
    def test_far_beyond_any_real_clock_it_still_detects_within_a_millisecond(
        self, spec: MarkerSpec
    ) -> None:
        """1000 ppm — five times any plausible offset — degrades gracefully rather than failing.

        The anchor drifts by about ten samples, because a stretched chirp matches its
        unstretched template best somewhere inside rather than at its first sample. Asserted
        against one millisecond rather than against the measured value: 48 samples is an
        order of magnitude below the 1.5-9 ms of acoustic propagation spread that OQ-025
        records as this instrument's floor, so drift of this size cannot change a conclusion.
        Fitting the assertion to the observed ten would be measuring the implementation.
        """
        track, expected = self._stretched(spec, 1_000)
        reader = ArrayReader(track)
        occurrences = detect_occurrences(reader, spec, interval=(0, reader.samples.size))
        assert len(occurrences) == 1
        assert abs(occurrences[0].anchor_sample - expected) <= MARKER_SAMPLE_RATE // 1_000

    @ALL_SPECS
    def test_the_gap_residual_is_measured_rather_than_merely_tolerated(
        self, spec: MarkerSpec
    ) -> None:
        """The quantity that makes a timing problem diagnosable at the bench.

        A stretch large enough to move a gap by whole samples must show up in
        ``gap_errors_samples``. Without that the analyzer could survive a device with a bad
        clock and report nothing about it, which is how OQ-029 would stay open forever.
        """
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stretched = np.asarray(resample_poly(marker, 1_001, 1_000), dtype=np.float64)
        track = np.zeros(PLACEMENT + stretched.size + TAIL, dtype=np.float64)
        track[PLACEMENT : PLACEMENT + stretched.size] = stretched

        reader = ArrayReader(track.astype(np.float32))
        occurrences = detect_occurrences(reader, spec, interval=(0, reader.samples.size))
        assert len(occurrences) == 1
        assert any(error != 0 for error in occurrences[0].gap_errors_samples)

    @ALL_SPECS
    def test_a_stretch_far_beyond_any_clock_is_rejected(self, spec: MarkerSpec) -> None:
        """5% is not a clock; it is a different sound, and accepting it would be wrong.

        The rejection comes from per-chirp correlation loss, **not** from the gap tolerance:
        5% of even the shortest gap is far inside ``gap_tolerance_samples``. Stated because
        it is the opposite of what the parameter's name suggests — see that field's note.
        """
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stretched = np.asarray(resample_poly(marker, 105, 100), dtype=np.float64)
        track = np.zeros(PLACEMENT + stretched.size + TAIL, dtype=np.float64)
        track[PLACEMENT : PLACEMENT + stretched.size] = stretched
        assert found(track.astype(np.float32), spec) == []


class TestScoresAreIntegers:
    """No decision anywhere compares two floats (ADR-0041)."""

    def test_permille_rounds_halves_away_from_zero(self) -> None:
        assert to_permille(0.0005) == 1
        assert to_permille(-0.0005) == -1
        assert to_permille(0.0004) == 0
        assert to_permille(1.0) == 1_000

    def test_a_non_finite_score_becomes_zero_rather_than_propagating(self) -> None:
        assert to_permille(float("nan")) == 0
        assert to_permille(float("inf")) == 0

    def test_the_array_quantizer_agrees_with_the_scalar_everywhere(self) -> None:
        """The detector quantizes per array; the scalar above is still what the rule *means*.

        `_peaks` cannot afford a Python call per start position, so it uses
        :func:`to_permille_array`. That is only legitimate while the two agree exactly — a
        single disagreeing element would move a peak, and every threshold downstream is an
        integer comparison against this output. Asserted on the cases that distinguish the
        two plausible roundings, then swept.
        """
        exact = np.array(
            [
                *[0.0, -0.0, 1.0, -1.0, 0.0005, -0.0005, 0.0015, -0.0015, 0.0004, 0.9995],
                *[float("nan"), float("inf"), float("-inf")],
            ],
            dtype=np.float64,
        )
        assert to_permille_array(exact).tolist() == [to_permille(float(v)) for v in exact]

        # Halves land on exact binary fractions only by construction, so sweep them directly
        # rather than hoping a uniform draw produces one.
        halves = (np.arange(-2_000, 2_001, dtype=np.float64) + 0.5) / 1_000
        assert to_permille_array(halves).tolist() == [to_permille(float(v)) for v in halves]

        swept = np.random.default_rng(20260805).uniform(-1.0, 1.0, 50_000)
        assert to_permille_array(swept).tolist() == [to_permille(float(v)) for v in swept]

    def test_the_array_quantizer_does_not_alias_the_scores_it_was_given(self) -> None:
        """`_peaks` writes its suppression sentinel into the returned array."""
        scores = np.full(8, 0.5, dtype=np.float64)
        quantized = to_permille_array(scores)
        quantized[:] = -1
        assert scores.tolist() == [0.5] * 8

    @ALL_SPECS
    def test_every_reported_score_is_an_integer_in_range(self, spec: MarkerSpec) -> None:
        reader = ArrayReader(place(spec, gain=0.3))
        occurrences = detect_occurrences(reader, spec, interval=(0, reader.samples.size))
        for occurrence in occurrences:
            assert isinstance(occurrence.score_permille, int)
            assert 0 <= occurrence.score_permille <= 1_000
            for hit in occurrence.hits:
                assert isinstance(hit.score_permille, int)
                assert 0 <= hit.score_permille <= 1_000

    def test_the_sequence_threshold_cannot_sit_below_the_chirp_threshold(self) -> None:
        with pytest.raises(ValueError, match="could never reject"):
            DetectorThresholds(min_chirp_score_permille=800, min_sequence_score_permille=700)

    def test_each_consecutive_gap_must_itself_be_inside_tolerance(self) -> None:
        """Opposite anchor-relative errors must not double the actual second gap."""
        spec = MARKER_SPECS["v1"]
        settings = DetectorThresholds()
        starts = [start for start, _ in spec.chirp_intervals()]
        anchor = 100_000
        peaks = [
            [(anchor, 600)],
            [(anchor + starts[1] - starts[0] - settings.gap_tolerance_samples, 600)],
            [(anchor + starts[2] - starts[0] + settings.gap_tolerance_samples, 600)],
        ]
        assert _assemble(spec, peaks, settings) == []

    @pytest.mark.parametrize(
        ("runner_up", "ambiguous"),
        [(0, False), (550, False), (551, True)],
    )
    def test_runner_up_separation_uses_the_integer_boundary(
        self, runner_up: int, ambiguous: bool
    ) -> None:
        """Exactly 50 permille is decisive; one permille closer is inconclusive."""
        assert _is_locally_ambiguous(600, runner_up, 50) is ambiguous

    def test_the_thresholds_identity_names_every_field(self) -> None:
        """The `derivative_identity_document` property: assert *which* components are there.

        A key that changes for the right reason can still be missing the component that
        matters later (M2's closeout), so this compares names rather than a hash.
        """
        settings = DetectorThresholds()
        assert set(settings.identity()) == {
            "association_lag_samples",
            "clipping_ratio_permille",
            "gap_tolerance_samples",
            "max_occurrences_per_track",
            "max_peak_candidates_per_chirp",
            "material_arrival_change_samples",
            "min_chirp_score_permille",
            "min_runner_up_separation_permille",
            "min_sequence_score_permille",
            "nms_radius_samples",
            "sequence_nms_radius_samples",
            "weak_signal_rms_permille",
        }


class TestBoundedReads:
    """INV-07 at the detector's own boundary; the composed proof lives in test_memory.py."""

    @ALL_SPECS
    def test_no_single_read_exceeds_the_block_plus_the_template(self, spec: MarkerSpec) -> None:
        marker = marker_samples(spec).astype(np.float64) / 32768.0
        stride = marker.size + 5 * MARKER_SAMPLE_RATE
        track = np.zeros(8 * stride, dtype=np.float64)
        for index in range(8):
            track[index * stride : index * stride + marker.size] = marker

        longest = max(chirp.duration_samples for chirp in spec.chirps)
        block = 1 << 16
        reader = ArrayReader(track.astype(np.float32))
        detect_occurrences(reader, spec, interval=(0, reader.samples.size), block_samples=block)
        # The diagnostic pass reads one marker length at each accepted anchor, which is why
        # the bound is a max rather than the block alone. Both terms are constants; neither
        # is a function of how much audio was searched.
        assert reader.largest_read <= max(block + longest, spec.total_samples)

    @ALL_SPECS
    def test_the_largest_read_does_not_grow_with_the_searched_range(self, spec: MarkerSpec) -> None:
        """The bound is a property of the block, not of the question asked."""
        short = ArrayReader(place(spec))
        detect_occurrences(short, spec, interval=(0, short.samples.size))

        long_track = np.zeros(30 * MARKER_SAMPLE_RATE, dtype=np.float32)
        marker = marker_samples(spec).astype(np.float32) / 32768.0
        long_track[PLACEMENT : PLACEMENT + marker.size] = marker
        long_reader = ArrayReader(long_track)
        detect_occurrences(long_reader, spec, interval=(0, long_reader.samples.size))

        assert long_reader.reads > short.reads
        assert long_reader.largest_read == short.largest_read
