"""The checked-in speech-band filter, held to what it claims to do.

Four things are proved here, and they fail differently.

:class:`TestDesignIsReproducible` re-runs the design and compares byte for byte, so a
hand-edited coefficient is caught. It is a change detector, and change detectors prove
nothing about correctness — running the design and comparing it to itself would pass for
any design at all.

:class:`TestFrequencyResponse` measures the committed array against the passband, stopband,
ripple, and attenuation the file declares. Without it, "the speech band filter" degrades
into an arbitrary array of the right length, and every level ADR-0014's gate compares would
be measured through something nobody checked.

:class:`TestBandLimited` proves the delay compensation **by position**. Amplitude cannot
prove it: a filter that passes 1 kHz at unity gain does so whether or not its output is
shifted 175 samples to the right, and a level measured over a candidate's span with an
11 ms offset is a level of partly the wrong audio — plausible in the artifact, and wrong.

:class:`TestLevelMillibels` pins the measurement to arithmetic that can be checked by hand:
a full-scale sine is -3.01 dB RMS, halving an amplitude is -6.02 dB, and a 60 Hz rumble at
the same amplitude as a 1 kHz tone must read far below it — which is the entire reason the
band-limiting exists.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy.signal import freqz

from dnd_audio.activity.band import (
    SILENCE_FLOOR_MB,
    SPEECH_BAND_PATH,
    SpeechBandError,
    SpeechBandFilter,
    _millibels,
    band_limited,
    level_millibels,
)
from dnd_audio.errors import DndAudioError
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

#: Frequency resolution of the response measurement. Dense enough that a ripple or a
#: stopband spike between grid points cannot hide: 65536 points over 8 kHz is one sample
#: every 0.12 Hz.
_RESPONSE_POINTS = 1 << 16

#: Length of the signals the time-domain tests filter. Three seconds at 16 kHz, so the
#: filter's 175-sample ramp at each end is a thousandth of the energy rather than a
#: correction the assertions would have to be loosened for.
_SIGNAL_SAMPLES = 48000

#: Samples skipped at each end before a steady-state amplitude is read. Comfortably more
#: than the group delay, which is the exact extent of the zero-padded region.
_EDGE_GUARD = 400


@pytest.fixture(scope="module")
def speech_band() -> SpeechBandFilter:
    from dnd_audio.activity.band import load_speech_band_filter

    return load_speech_band_filter()


@pytest.fixture(scope="module")
def response(speech_band: SpeechBandFilter) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response of the committed coefficients, on a dense linear grid."""
    frequencies, complex_response = freqz(
        speech_band.coefficients, worN=_RESPONSE_POINTS, fs=speech_band.sample_rate
    )
    return np.asarray(frequencies), np.abs(np.asarray(complex_response))


def _tone(hz: float, *, amplitude: float = 1.0, samples: int = _SIGNAL_SAMPLES) -> np.ndarray:
    """A sine at ``hz``, as float32, starting and ending abruptly."""
    time = np.arange(samples, dtype=np.float64) / DERIVATIVE_SAMPLE_RATE
    return np.asarray(amplitude * np.sin(2 * np.pi * hz * time), dtype=np.float32)


def _faded_tone(hz: float, *, samples: int = _SIGNAL_SAMPLES) -> np.ndarray:
    """A full-scale sine that ramps on and off over 100 ms.

    An abruptly-started tone is a step, and a step is broadband: switching a 60 Hz sine on
    at full scale puts a click through the passband that has nothing to do with 60 Hz. When
    the question is "how much of *this frequency* survives", the ramp is what makes the
    answer about the frequency rather than about the edge of the array.
    """
    ramp = np.arange(samples, dtype=np.float64)
    envelope = np.minimum(1.0, np.minimum(ramp, ramp[::-1]) / (DERIVATIVE_SAMPLE_RATE / 10))
    return np.asarray(_tone(hz, samples=samples) * envelope, dtype=np.float32)


def _smoothed_energy(signal: npt.NDArray[np.float32] | np.ndarray) -> np.ndarray:
    """Short-time energy, for locating *where* a burst is rather than how loud it is.

    A 10 ms boxcar over the squared samples: long enough that the answer is the envelope's
    position rather than which individual zero-crossing happened to be largest, short
    enough that it locates a burst to a handful of samples.
    """
    window = np.ones(161, dtype=np.float64) / 161
    return np.convolve(np.asarray(signal, dtype=np.float64) ** 2, window, mode="same")


class TestFrequencyResponse:
    """What the filter must *do*, independent of how it was designed."""

    def test_the_passband_is_the_telephony_speech_band(self, speech_band: SpeechBandFilter) -> None:
        """300–3400 Hz, and stated in the artifact rather than implied by a tap count.

        Below 300 Hz is where a DC offset and the room's rumble live, and above 3400 Hz is
        where hiss dominates. Neither says anything about which voice was louder, so a
        level comparison must not see them.
        """
        assert speech_band.passband_low_hz == 300.0
        assert speech_band.passband_high_hz == 3400.0
        assert speech_band.lower_stopband_edge_hz < speech_band.passband_low_hz
        assert speech_band.upper_stopband_edge_hz > speech_band.passband_high_hz
        assert speech_band.upper_stopband_edge_hz <= speech_band.sample_rate / 2

    def test_passband_ripple_is_within_the_declared_bound(
        self, speech_band: SpeechBandFilter, response: tuple[np.ndarray, np.ndarray]
    ) -> None:
        frequencies, magnitude = response
        inside = (frequencies >= speech_band.passband_low_hz) & (
            frequencies <= speech_band.passband_high_hz
        )
        passband = magnitude[inside]
        ripple_db = 20 * np.log10(passband.max() / passband.min())
        assert ripple_db <= speech_band.max_passband_ripple_db, (
            f"passband ripple {ripple_db:.4f} dB exceeds the declared "
            f"{speech_band.max_passband_ripple_db} dB"
        )

    def test_the_passband_gain_is_unity_within_the_declared_ripple(
        self, speech_band: SpeechBandFilter, response: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Not merely flat — flat *at one*.

        Peak-to-peak ripple says nothing about where the passband sits. An unnormalized
        windowed sinc, or one normalized for unity DC gain the way the decimator is, would
        be equally flat at some other gain and would shift every level in the artifact by a
        constant. Bleed compares levels to each other, so a constant offset is exactly the
        error that would never be noticed.
        """
        frequencies, magnitude = response
        inside = (frequencies >= speech_band.passband_low_hz) & (
            frequencies <= speech_band.passband_high_hz
        )
        deviation_db = np.abs(20 * np.log10(magnitude[inside]))
        assert deviation_db.max() <= speech_band.max_passband_ripple_db, (
            f"passband gain deviates from unity by {deviation_db.max():.4f} dB, more than "
            f"the declared {speech_band.max_passband_ripple_db} dB"
        )

    def test_the_lower_stopband_meets_the_declared_attenuation(
        self, speech_band: SpeechBandFilter, response: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Everything at and below 100 Hz — DC offset, HVAC, table thumps, handling noise."""
        frequencies, magnitude = response
        stopband = magnitude[frequencies <= speech_band.lower_stopband_edge_hz]
        attenuation_db = -20 * np.log10(stopband.max())
        assert attenuation_db >= speech_band.min_stopband_attenuation_db, (
            f"lower stopband attenuation {attenuation_db:.2f} dB is below the declared "
            f"{speech_band.min_stopband_attenuation_db} dB"
        )

    def test_the_upper_stopband_meets_the_declared_attenuation(
        self, speech_band: SpeechBandFilter, response: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Everything at and above 3600 Hz, which includes the 3800 Hz hiss floor."""
        frequencies, magnitude = response
        stopband = magnitude[frequencies >= speech_band.upper_stopband_edge_hz]
        attenuation_db = -20 * np.log10(stopband.max())
        assert attenuation_db >= speech_band.min_stopband_attenuation_db, (
            f"upper stopband attenuation {attenuation_db:.2f} dB is below the declared "
            f"{speech_band.min_stopband_attenuation_db} dB"
        )

    def test_dc_is_rejected(self, speech_band: SpeechBandFilter) -> None:
        """The sum of the taps is the gain at DC, and a band-pass has none.

        A constant offset in a track would otherwise add to every level measured on it, by
        an amount that differs per transmitter — which is a level comparison decided by
        preamp trim rather than by who was speaking.
        """
        assert abs(float(speech_band.coefficients.sum())) < 10 ** (
            -speech_band.min_stopband_attenuation_db / 20
        )

    def test_the_response_test_can_fail(self, speech_band: SpeechBandFilter) -> None:
        """The naive alternative — measure the level of the raw signal — must not pass.

        A test that only ever sees a good filter proves nothing about its own ability to
        reject a bad one. "No filter at all" is a delta, which is the specific mistake this
        module exists to prevent, and it must fail both stopband checks.
        """
        allpass = np.zeros(speech_band.length, dtype=np.float64)
        allpass[speech_band.group_delay] = 1.0
        frequencies, complex_response = freqz(allpass, worN=1 << 12, fs=speech_band.sample_rate)
        magnitude = np.abs(np.asarray(complex_response))
        frequencies = np.asarray(frequencies)
        for band in (
            magnitude[frequencies <= speech_band.lower_stopband_edge_hz],
            magnitude[frequencies >= speech_band.upper_stopband_edge_hz],
        ):
            assert -20 * np.log10(band.max()) < speech_band.min_stopband_attenuation_db


class TestShape:
    """The structural properties the delay compensation depends on."""

    def test_the_filter_is_exactly_symmetric(self, speech_band: SpeechBandFilter) -> None:
        """Bit-exact, not close: symmetry is what makes the phase linear."""
        assert np.array_equal(speech_band.coefficients, speech_band.coefficients[::-1])

    def test_the_length_is_odd_and_the_delay_is_the_whole_number_it_implies(
        self, speech_band: SpeechBandFilter
    ) -> None:
        """An even length would delay by a half sample, which no slice can undo.

        There is no decimation here, so — unlike the 48→16 filter — the delay has nothing
        to divide by. Being a whole sample is the entire requirement.
        """
        assert speech_band.length % 2 == 1
        assert speech_band.group_delay == (speech_band.length - 1) // 2

    def test_it_is_designed_for_the_rate_the_detector_works_at(
        self, speech_band: SpeechBandFilter
    ) -> None:
        """Levels are measured on the 16 kHz derivative, not on the 48 kHz source."""
        assert speech_band.sample_rate == DERIVATIVE_SAMPLE_RATE

    def test_the_identity_is_the_file_hash(self, speech_band: SpeechBandFilter) -> None:
        """Part of the attribution cache key (INV-08): a changed filter invalidates every
        decision it produced, and it can only do that if it hashes the bytes on disk."""
        from dnd_audio.determinism import sha256_bytes

        assert speech_band.identity == sha256_bytes(SPEECH_BAND_PATH.read_bytes())


class TestBandLimited:
    """Filtering, and the alignment that makes a filtered level mean anything."""

    def test_the_output_has_the_same_length_and_dtype(self) -> None:
        signal = _tone(1000.0, samples=5000)
        filtered = band_limited(signal)
        assert filtered.shape == signal.shape
        assert filtered.dtype == np.float32

    def test_a_speech_band_tone_passes_at_near_unity_amplitude(
        self, speech_band: SpeechBandFilter
    ) -> None:
        """Read away from the ends, where the zero padding is still ramping."""
        filtered = band_limited(_tone(1000.0))
        peak = float(np.abs(filtered[_EDGE_GUARD:-_EDGE_GUARD]).max())
        assert peak == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("hz", [60.0, 6000.0])
    def test_out_of_band_tones_are_attenuated_by_the_declared_amount(
        self, speech_band: SpeechBandFilter, hz: float
    ) -> None:
        """60 Hz is room rumble; 6 kHz is hiss. Both are below the declared stopband."""
        filtered = band_limited(_tone(hz))
        peak = float(np.abs(filtered[_EDGE_GUARD:-_EDGE_GUARD]).max())
        assert peak <= 10 ** (-speech_band.min_stopband_attenuation_db / 20)

    def test_an_impulse_comes_out_where_it_went_in(self, speech_band: SpeechBandFilter) -> None:
        """The sharpest possible proof of the delay compensation, and it is exact.

        A symmetric filter's response to a delta is that filter, centred on the delta. So
        the largest output sample is at the input's index — not near it — and any error in
        the compensating slice moves it by a whole sample.
        """
        index = 2000
        impulse = np.zeros(4000, dtype=np.float32)
        impulse[index] = 1.0
        assert int(np.argmax(np.abs(band_limited(impulse)))) == index

    def test_a_speech_burst_keeps_its_position(self) -> None:
        """The realistic version: a windowed tone burst, located by its energy envelope.

        Tolerance is two samples — 0.125 ms — rather than zero, because the filter reshapes
        the burst as well as delaying it, and a 10 ms smoothing window resolves a peak only
        to about that. An uncompensated filter would be off by 175, which is two orders of
        magnitude outside this; see the next test.
        """
        signal, index = _burst()
        filtered = band_limited(signal)
        assert abs(int(np.argmax(_smoothed_energy(filtered))) - index) <= 2

    def test_without_compensation_the_burst_would_move(self, speech_band: SpeechBandFilter) -> None:
        """What the positional test is actually detecting.

        A test that only ever sees the compensated path proves nothing about its ability to
        notice the uncompensated one, so the uncompensated convolution is run here and its
        burst is shown to arrive a group delay late.
        """
        signal, index = _burst()
        raw = np.convolve(np.asarray(signal, dtype=np.float64), speech_band.coefficients)
        moved = int(np.argmax(_smoothed_energy(raw))) - index
        assert moved == pytest.approx(speech_band.group_delay, abs=10)

    def test_an_empty_window_filters_to_an_empty_window(self) -> None:
        filtered = band_limited(np.zeros(0, dtype=np.float32))
        assert filtered.shape == (0,)
        assert filtered.dtype == np.float32

    def test_multichannel_input_is_refused(self) -> None:
        """Silently filtering the first channel, or the interleaved bytes of two, would
        produce a level for a signal that does not exist."""
        with pytest.raises(ValueError, match="one channel"):
            band_limited(np.zeros((2, 100), dtype=np.float32))


def _burst(index: int = 8000, samples: int = 16000) -> tuple[npt.NDArray[np.float32], int]:
    """A 900 Hz tone burst under a Hann window, centred on ``index``.

    Returned with the index its *own* energy envelope peaks at rather than the index it was
    centred on: a windowed sine's envelope peak sits a few samples from the window's centre,
    and the question here is whether filtering moves it, not where it started.
    """
    width = 800
    signal = np.zeros(samples, dtype=np.float64)
    offsets = np.arange(-width, width + 1, dtype=np.float64)
    signal[index - width : index + width + 1] = np.hanning(2 * width + 1) * np.sin(
        2 * np.pi * 900.0 * offsets / DERIVATIVE_SAMPLE_RATE
    )
    array = np.asarray(signal, dtype=np.float32)
    return array, int(np.argmax(_smoothed_energy(array)))


class TestLevelMillibels:
    """Arithmetic a reader can check by hand, on signals whose answer is known."""

    def test_a_full_scale_sine_reads_minus_three_decibels(self) -> None:
        """A unit sine's RMS is 1/√2, which is -3.0103 dB — so -301 millibels.

        The tolerance is 5 mB (0.05 dB) and covers two real effects: the passband is flat
        only to within its declared ripple, and the first and last 175 output samples are
        computed against implicit zeros, which pulls the RMS of a three-second signal down
        by about a hundredth of a decibel.
        """
        assert level_millibels(_tone(1000.0)) == pytest.approx(-301, abs=5)

    def test_halving_the_amplitude_costs_six_decibels(self) -> None:
        """-6.0206 dB, and the edge ramp cancels because it scales with the signal."""
        loud = level_millibels(_tone(1000.0))
        quiet = level_millibels(_tone(1000.0, amplitude=0.5))
        assert loud - quiet == pytest.approx(602, abs=2)

    def test_digital_silence_reads_exactly_the_floor(self) -> None:
        """Zero has no logarithm. The floor is what the artifact's integer field gets."""
        assert level_millibels(np.zeros(4000, dtype=np.float32)) == SILENCE_FLOOR_MB

    def test_an_empty_window_reads_the_floor(self) -> None:
        assert level_millibels(np.zeros(0, dtype=np.float32)) == SILENCE_FLOOR_MB

    def test_a_level_below_the_floor_is_clamped_to_it(self) -> None:
        """-180 dB is not a measurement of anything; it is a denormal pretending to be one."""
        assert level_millibels(_tone(1000.0, amplitude=1e-9)) == SILENCE_FLOOR_MB

    def test_rumble_reads_far_below_speech_at_the_same_amplitude(
        self, speech_band: SpeechBandFilter
    ) -> None:
        """The whole point of band-limiting before comparing levels.

        Two signals, both full scale. Broadband RMS would call them equally loud and hand
        the interval to whichever track happened to have more rumble — the "single global
        loudness comparison" ADR-0014 forbids, arrived at by accident. Band-limited, the
        rumble is at least the declared stopband attenuation quieter.
        """
        speech = level_millibels(_faded_tone(1000.0))
        rumble = level_millibels(_faded_tone(60.0))
        assert speech - rumble >= int(speech_band.min_stopband_attenuation_db * 100)

    def test_the_measurement_is_repeatable(self) -> None:
        """Byte-stability (INV-02) starts with the number being the same number twice."""
        signal = _burst()[0]
        assert level_millibels(signal) == level_millibels(signal.copy())

    @pytest.mark.parametrize(
        ("decibels", "expected"),
        [(0.125, 13), (-0.125, -13), (0.0, 0), (-3.0103, -301), (0.1249, 12), (-1.0, -100)],
    )
    def test_millibels_round_halves_away_from_zero(self, decibels: float, expected: int) -> None:
        """The tie rule stated, not inherited.

        0.125 dB is exactly 12.5 mB — a value a float represents exactly, so the tie is real
        rather than an artefact of binary — and it must go to 13 in both directions.
        Python's `round` is banker's rounding and would answer 12 for one of them, which is
        how a level that differs by one millibel between two runs gets into a byte-stable
        artifact.
        """
        assert _millibels(decibels) == expected


class TestValidationRejectsABrokenFile:
    """The loader's guards, each proven able to fire.

    `load_speech_band_filter` validates on the way in. A guard that has never rejected
    anything is a guard nobody has tested, so each one is driven with the exact malformed
    input it exists to catch.
    """

    def test_an_even_length_filter_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._expect_rejection(monkeypatch, coefficients=[0.25, 0.5, 0.5, 0.25], match="taps")

    def test_an_asymmetric_filter_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._expect_rejection(monkeypatch, coefficients=[0.2, 0.5, 0.3], match="symmetric")

    def test_a_declared_delay_that_contradicts_the_length_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._expect_rejection(
            monkeypatch,
            coefficients=[0.1, 0.2, 0.4, 0.2, 0.1],
            group_delay=3,
            match="group delay",
        )

    def test_unparsable_json_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from dnd_audio.activity.band import load_speech_band_filter

        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            monkeypatch.setattr("dnd_audio.activity.band.SPEECH_BAND_PATH", path)
            load_speech_band_filter.cache_clear()
            with pytest.raises(SpeechBandError, match="valid JSON"):
                load_speech_band_filter()
        load_speech_band_filter.cache_clear()

    def test_a_missing_file_names_the_fix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dnd_audio.activity.band import load_speech_band_filter

        monkeypatch.setattr(
            "dnd_audio.activity.band.SPEECH_BAND_PATH", SPEECH_BAND_PATH.parent / "absent.json"
        )
        load_speech_band_filter.cache_clear()
        with pytest.raises(SpeechBandError, match=r"design_speech_band\.py"):
            load_speech_band_filter()
        load_speech_band_filter.cache_clear()

    def test_the_error_carries_a_structured_code(self) -> None:
        """INV-13 wants a code a caller can branch on, not prose that gets reworded."""
        assert issubclass(SpeechBandError, DndAudioError)
        assert SpeechBandError("x").code == "speech_band_filter_unusable"

    @staticmethod
    def _expect_rejection(
        monkeypatch: pytest.MonkeyPatch,
        *,
        coefficients: list[float],
        group_delay: int | None = None,
        match: str,
    ) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from dnd_audio.activity.band import load_speech_band_filter

        document = {
            "name": "broken",
            "design": {"sample_rate": 16000, "length": len(coefficients)},
            "response": {
                "passband_low_hz": 300.0,
                "passband_high_hz": 3400.0,
                "max_passband_ripple_db": 1.0,
                "lower_stopband_edge_hz": 100.0,
                "upper_stopband_edge_hz": 3600.0,
                "min_stopband_attenuation_db": 60.0,
            },
            "group_delay_samples": (
                group_delay if group_delay is not None else (len(coefficients) - 1) // 2
            ),
            "coefficients": coefficients,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            monkeypatch.setattr("dnd_audio.activity.band.SPEECH_BAND_PATH", path)
            load_speech_band_filter.cache_clear()
            with pytest.raises(SpeechBandError, match=match):
                load_speech_band_filter()
        load_speech_band_filter.cache_clear()


class TestDesignIsReproducible:
    """The committed bytes are what `scripts/design_speech_band.py` produces.

    A change detector, and labelled as one. It catches a coefficient edited by hand or a
    file half-written; it says nothing about whether the design is any good, which is what
    :class:`TestFrequencyResponse` is for.

    A SciPy upgrade that moves a coefficient in its last bit will fail this. That is the
    intended behaviour: regenerate, watch the response test still pass, and commit both —
    which is exactly the review INV-08 wants before the attribution cache's identity moves.
    """

    def test_the_checked_in_file_matches_the_design(self) -> None:
        import importlib.util
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "design_speech_band.py"
        spec = importlib.util.spec_from_file_location("design_speech_band", script)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["design_speech_band"] = module
        spec.loader.exec_module(module)

        assert module.document() == SPEECH_BAND_PATH.read_text(encoding="utf-8")
