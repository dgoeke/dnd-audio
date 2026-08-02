"""The checked-in decimation filter, held to what it claims to do.

Two tests, guarding two different failures.

:class:`TestDesignIsReproducible` re-runs the design and compares byte for byte, so a
hand-edited coefficient is caught. It is a change detector, and change detectors prove
nothing about correctness — running the design and comparing it to itself would pass for
any design at all.

:class:`TestFrequencyResponse` is the one that matters. It measures the committed array
against the passband, stopband, ripple, and attenuation the file declares. Without it,
"one canonical fixed decimator" degrades into an arbitrary set of numbers that happens to
produce the expected sample count — the derivative would have the right length and the
wrong contents, and every VAD and ASR result downstream would be quietly built on aliased
audio.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

from dnd_audio.errors import DndAudioError
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.fir import (
    FIR_PATH,
    DecimationFilter,
    FilterError,
    load_decimation_filter,
)

#: Frequency resolution of the response measurement. Dense enough that a ripple or a
#: stopband spike between grid points cannot hide: 65536 points over 24 kHz is one sample
#: every 0.37 Hz.
_RESPONSE_POINTS = 1 << 16


@pytest.fixture(scope="module")
def decimator() -> DecimationFilter:
    return load_decimation_filter()


@pytest.fixture(scope="module")
def response(decimator: DecimationFilter) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response of the committed coefficients, on a dense linear grid."""
    frequencies, complex_response = freqz(
        decimator.coefficients, worN=_RESPONSE_POINTS, fs=decimator.input_rate
    )
    return np.asarray(frequencies), np.abs(np.asarray(complex_response))


class TestFrequencyResponse:
    """What the filter must *do*, independent of how it was designed."""

    def test_the_stopband_starts_at_or_below_the_derivative_nyquist(
        self, decimator: DecimationFilter
    ) -> None:
        """Everything above 8 kHz folds back into the 16 kHz derivative.

        This is physics rather than preference: content above the output Nyquist does not
        merely get lost, it reappears mirrored inside the speech band, where nothing
        downstream can tell it from signal.
        """
        assert decimator.stopband_edge_hz <= DERIVATIVE_SAMPLE_RATE / 2

    def test_the_passband_covers_the_declared_edge(self, decimator: DecimationFilter) -> None:
        """A passband that stops below the speech band would be a different failure."""
        assert decimator.passband_edge_hz >= 7000.0

    def test_passband_ripple_is_within_the_declared_bound(
        self, decimator: DecimationFilter, response: tuple[np.ndarray, np.ndarray]
    ) -> None:
        frequencies, magnitude = response
        passband = magnitude[frequencies <= decimator.passband_edge_hz]
        ripple_db = 20 * np.log10(passband.max() / passband.min())
        assert ripple_db <= decimator.max_passband_ripple_db, (
            f"passband ripple {ripple_db:.4f} dB exceeds the declared "
            f"{decimator.max_passband_ripple_db} dB"
        )

    def test_stopband_attenuation_meets_the_declared_bound(
        self, decimator: DecimationFilter, response: tuple[np.ndarray, np.ndarray]
    ) -> None:
        frequencies, magnitude = response
        stopband = magnitude[frequencies >= decimator.stopband_edge_hz]
        attenuation_db = -20 * np.log10(stopband.max())
        assert attenuation_db >= decimator.min_stopband_attenuation_db, (
            f"stopband attenuation {attenuation_db:.2f} dB is below the declared "
            f"{decimator.min_stopband_attenuation_db} dB"
        )

    def test_dc_gain_is_unity(self, decimator: DecimationFilter) -> None:
        """A DC gain that is merely close to one shifts every derivative's level.

        The coefficients are normalized by their sum, so this is exact rather than
        approximate — and asserting it exactly is what would catch a renormalization
        being dropped.
        """
        assert decimator.coefficients.sum() == pytest.approx(1.0, abs=1e-15)

    def test_the_response_test_can_fail(self, decimator: DecimationFilter) -> None:
        """The naive alternative — decimate with no filter — must not pass.

        A test that only ever sees a good filter proves nothing about its own ability to
        reject a bad one. An all-pass "filter" is what "take every third sample" is, and
        it is the specific mistake ADR-0011 forbids.
        """
        allpass = np.zeros(decimator.length, dtype=np.float64)
        allpass[decimator.group_delay_input] = 1.0
        frequencies, magnitude = freqz(allpass, worN=1 << 12, fs=decimator.input_rate)
        stopband = np.abs(magnitude)[np.asarray(frequencies) >= decimator.stopband_edge_hz]
        attenuation_db = -20 * np.log10(stopband.max())
        assert attenuation_db < decimator.min_stopband_attenuation_db


class TestShape:
    """The structural properties the mapping arithmetic depends on."""

    def test_the_filter_is_exactly_symmetric(self, decimator: DecimationFilter) -> None:
        """Bit-exact, not close: symmetry is what makes the phase linear."""
        assert np.array_equal(decimator.coefficients, decimator.coefficients[::-1])

    def test_the_group_delay_is_a_whole_number_of_output_samples(
        self, decimator: DecimationFilter
    ) -> None:
        """129 at 48 kHz is exactly 43 at 16 kHz.

        This is the property that makes `sample16 = sample48 // 3` true rather than nearly
        true. A length whose delay did not divide by three would leave a third of a
        sample of error in every word timestamp, in the same direction, forever.
        """
        assert decimator.group_delay_input == (decimator.length - 1) // 2
        assert decimator.group_delay_input % decimator.decimation == 0
        assert decimator.group_delay_output == decimator.group_delay_input // decimator.decimation

    def test_the_rates_are_the_ones_the_pipeline_uses(self, decimator: DecimationFilter) -> None:
        assert decimator.input_rate == decimator.output_rate * decimator.decimation
        assert decimator.output_rate == DERIVATIVE_SAMPLE_RATE


class TestValidationRejectsABrokenFile:
    """The loader's guards, each proven able to fire.

    `load_decimation_filter` validates on the way in. A guard that has never rejected
    anything is a guard nobody has tested, so each one is driven with the exact malformed
    input it exists to catch.
    """

    def test_an_even_length_filter_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._expect_rejection(monkeypatch, coefficients=[0.25, 0.5, 0.25, 0.0], match="taps")

    def test_an_asymmetric_filter_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._expect_rejection(monkeypatch, coefficients=[0.2, 0.5, 0.3], match="symmetric")

    def test_a_delay_that_does_not_divide_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Length 5 delays by 2, which is not a whole number of samples at 1/3 rate.
        self._expect_rejection(
            monkeypatch,
            coefficients=[0.1, 0.2, 0.4, 0.2, 0.1],
            delay_input=2,
            delay_output=0,
            match="whole",
        )

    def test_a_missing_file_names_the_fix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dnd_audio.timeline.fir.FIR_PATH", FIR_PATH.parent / "absent.json")
        load_decimation_filter.cache_clear()
        with pytest.raises(FilterError, match=r"design_fir\.py"):
            load_decimation_filter()
        load_decimation_filter.cache_clear()

    def test_the_error_carries_a_structured_code(self) -> None:
        """INV-13 wants a code a caller can branch on, not prose that gets reworded."""
        assert issubclass(FilterError, DndAudioError)
        assert FilterError("x").code == "decimation_filter_unusable"

    @staticmethod
    def _expect_rejection(
        monkeypatch: pytest.MonkeyPatch,
        *,
        coefficients: list[float],
        delay_input: int | None = None,
        delay_output: int | None = None,
        match: str,
    ) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        document = {
            "name": "broken",
            "design": {"input_rate": 48000, "output_rate": 16000, "decimation": 3},
            "response": {
                "passband_edge_hz": 7000.0,
                "max_passband_ripple_db": 0.1,
                "stopband_edge_hz": 8000.0,
                "min_stopband_attenuation_db": 80.0,
            },
            "group_delay_samples_input": (
                delay_input if delay_input is not None else (len(coefficients) - 1) // 2
            ),
            "group_delay_samples_output": (
                delay_output if delay_output is not None else ((len(coefficients) - 1) // 2) // 3
            ),
            "coefficients": coefficients,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            monkeypatch.setattr("dnd_audio.timeline.fir.FIR_PATH", path)
            load_decimation_filter.cache_clear()
            with pytest.raises(FilterError, match=match):
                load_decimation_filter()
        load_decimation_filter.cache_clear()


class TestDesignIsReproducible:
    """The committed bytes are what `scripts/design_fir.py` produces.

    A change detector, and labelled as one. It catches a coefficient edited by hand or a
    file half-written; it says nothing about whether the design is any good, which is
    what :class:`TestFrequencyResponse` is for.

    A SciPy upgrade that moves a coefficient in its last bit will fail this. That is the
    intended behaviour: regenerate, watch the response test still pass, and commit both —
    which is exactly the review INV-08 wants before a cached derivative's identity moves.
    """

    def test_the_checked_in_file_matches_the_design(self) -> None:
        import importlib.util
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "design_fir.py"
        spec = importlib.util.spec_from_file_location("design_fir", script)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["design_fir"] = module
        spec.loader.exec_module(module)

        assert module.document() == FIR_PATH.read_text(encoding="utf-8")
