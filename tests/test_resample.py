"""The 48→16 kHz derivative: exact length, exact position, no boundary artefacts.

M2's charter names this as "the most likely source of a subtle, late-discovered offset"
and says to test it against known impulse positions rather than durations. A derivative
that is 40 ms early has the right length, decodes fine, and makes every word timestamp
downstream wrong in a way that looks like a bad aligner.

Three properties, each with its own failure mode:

* **Length** — `ceil(n / 3)`. Rounding down drops up to two samples off every track whose
  length is not a multiple of three, always in the same direction.
* **Position** — output *k* corresponds to input *3k*, so a filtered impulse peaks at the
  output sample nearest its input position. Getting the group-delay compensation wrong
  shifts every derivative by a constant nobody would attribute to the resampler. Writing
  this test is what corrected the module's original claim of `n // 3`: that is the right
  rule for the *start* of an interval and the wrong one for where a peak lands.
* **Continuity** — a streamed run must be byte-identical to a one-shot run at every window
  partitioning. Resetting the filter per window, or per chunk, puts a transient at each
  boundary and makes the result depend on how DJI split the recording.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import upfirdn

from dnd_audio.timeline.fir import load_decimation_filter
from dnd_audio.timeline.resample import (
    Decimator,
    decimate_stream,
    output_length,
    to_derivative_interval,
    to_source_sample,
)

DOWN = 3


@pytest.fixture(scope="module")
def taps() -> np.ndarray:
    return load_decimation_filter().coefficients


def one_shot(samples: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """The whole-signal reference, computed without the streaming machinery.

    Deliberately a second implementation rather than a call into the module under test:
    comparing the streamed path against itself would prove only that it is deterministic.
    """
    decimator = load_decimation_filter()
    full = upfirdn(taps, np.asarray(samples, dtype=np.float64), up=1, down=DOWN)
    expected = output_length(int(samples.shape[0]), DOWN)
    return np.asarray(full[decimator.group_delay_output :][:expected], dtype=np.float32)


def streamed(samples: np.ndarray, window: int) -> np.ndarray:
    blocks = [samples[i : i + window] for i in range(0, int(samples.shape[0]), window)]
    produced = list(decimate_stream(blocks, int(samples.shape[0])))
    return np.concatenate(produced) if produced else np.zeros(0, dtype=np.float32)


class TestOutputLength:
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 4, 5, 6, 299, 300, 301, 302, 48000, 100000])
    def test_length_is_ceil_of_the_input_over_three(self, n_input: int) -> None:
        """Every residue mod 3, including the ones a floor rule would truncate."""
        rng = np.random.default_rng(n_input)
        samples = rng.standard_normal(n_input).astype(np.float32)
        assert streamed(samples, 4096).shape[0] == -(-n_input // DOWN)

    def test_the_floor_rule_would_differ_where_it_matters(self) -> None:
        """Naming the alternative, so the choice is visible rather than incidental."""
        assert output_length(301, DOWN) == 101
        assert 301 // DOWN == 100

    def test_a_zero_length_track_produces_nothing(self) -> None:
        assert streamed(np.zeros(0, dtype=np.float32), 1024).shape[0] == 0


class TestImpulsePosition:
    @pytest.mark.parametrize("position", [0, 1, 2, 3, 4, 5, 30000, 30001, 30002])
    def test_an_impulse_peaks_at_the_nearest_output_sample(self, position: int) -> None:
        """Output *k* sits at input *3k*, so the peak lands on the nearest grid point.

        The positions cover all three decimation phases. An impulse two-thirds of the way
        between two output samples peaks at the *later* one — which is a fact about where
        the grid is, not a rounding rule, and is why converting an interval floors its
        start and ceils its end instead of rounding both.
        """
        samples = np.zeros(60000, dtype=np.float32)
        samples[position] = 1.0
        derived = streamed(samples, 8192)
        nearest = round(position / DOWN)
        assert int(np.argmax(np.abs(derived))) == nearest

    def test_the_delay_is_compensated_not_merely_present(self) -> None:
        """Without the 43-sample skip the peak would sit 43 samples late.

        Stated as the contrast, because "an impulse produces a peak somewhere" is true of
        every implementation including the broken one.
        """
        decimator = load_decimation_filter()
        samples = np.zeros(60000, dtype=np.float32)
        samples[30000] = 1.0
        derived = streamed(samples, 8192)
        uncompensated = 30000 // DOWN + decimator.group_delay_output
        assert int(np.argmax(np.abs(derived))) == 30000 // DOWN
        assert int(np.argmax(np.abs(derived))) != uncompensated


class TestIntervalMapping:
    """Converting a 48 kHz interval onto the derivative grid, without losing an end."""

    def test_an_output_sample_maps_back_exactly(self) -> None:
        assert to_source_sample(0, DOWN) == 0
        assert to_source_sample(16000, DOWN) == 48000

    @pytest.mark.parametrize(
        ("start", "end"),
        [(0, 1), (0, 3), (1, 2), (1, 4), (100, 101), (30001, 30002), (47999, 48001)],
    )
    def test_the_mapped_interval_always_covers_the_original(self, start: int, end: int) -> None:
        """The property, rather than a table of expected values.

        A mapping that covers the input can only be wrong by being too generous, which
        costs a sample of context. One that does not cover it drops audio, and the audio
        it drops is the start of a word.
        """
        low, high = to_derivative_interval(start, end, DOWN)
        assert to_source_sample(low, DOWN) <= start
        assert to_source_sample(high, DOWN) >= end

    def test_rounding_both_ends_the_same_way_would_lose_samples(self) -> None:
        """Naming the wrong implementation, so the choice is visible."""
        start, end = 1, 2
        low, high = to_derivative_interval(start, end, DOWN)
        assert (low, high) == (0, 1)
        naive = (start // DOWN, end // DOWN)
        assert naive == (0, 0)
        assert naive[1] - naive[0] == 0

    def test_a_backwards_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="backwards"):
            to_derivative_interval(10, 5, DOWN)


class TestStreamingEqualsOneShot:
    @pytest.mark.parametrize("window", [1, 2, 3, 7, 999, 3000, 49999, 50000])
    def test_every_window_partitioning_gives_identical_bytes(
        self, window: int, taps: np.ndarray
    ) -> None:
        """Byte-identical, not merely close.

        A filter reset per window would still be "close" on a noise signal — the error
        lives at the boundaries — so equality is the assertion that distinguishes them.
        Window sizes 1, 2, and 7 are not multiples of three on purpose: the decimation
        phase has to survive a block that does not align with it.
        """
        rng = np.random.default_rng(11)
        samples = rng.standard_normal(50000).astype(np.float32)
        assert np.array_equal(streamed(samples, window), one_shot(samples, taps))

    def test_a_boundary_inside_a_transient_is_not_visible(self, taps: np.ndarray) -> None:
        """The case a per-window reset would fail loudly.

        A click at sample 24000 with the window boundary landing on it: a reset would
        truncate the filter's view of the transient and leave a step in the output.
        """
        samples = np.zeros(48000, dtype=np.float32)
        samples[24000:24010] = 1.0
        assert np.array_equal(streamed(samples, 24000), one_shot(samples, taps))

    def test_silence_stays_silent(self) -> None:
        """A gap must not acquire filter ringing from the audio around it."""
        samples = np.zeros(48000, dtype=np.float32)
        assert not streamed(samples, 4096).any()


class TestTheFilterIsNotResetAtChunkBoundaries:
    def test_a_track_split_in_two_matches_the_same_track_whole(self, taps: np.ndarray) -> None:
        """The property ADR-0011 states, expressed the way the pipeline would break it.

        Decimating each chunk separately and concatenating is the tempting implementation:
        it is simpler, it parallelizes, and it makes the derivative depend on where DJI
        happened to split the file. Here the two answers are computed and shown to differ.
        """
        rng = np.random.default_rng(3)
        samples = rng.standard_normal(30000).astype(np.float32)

        whole = streamed(samples, 4096)
        per_chunk = np.concatenate(
            [
                streamed(samples[:15000], 4096),
                streamed(samples[15000:], 4096),
            ]
        )
        assert np.array_equal(whole, one_shot(samples, taps))
        assert whole.shape[0] != per_chunk.shape[0] or not np.array_equal(whole, per_chunk)


class TestDecimatorContract:
    def test_it_declares_its_output_length_before_seeing_a_sample(self) -> None:
        """The writer needs the length up front to choose RIFF or RF64 and to size a header."""
        assert Decimator(100000).expected_output == output_length(100000, DOWN)

    def test_flushing_delivers_exactly_the_declared_length(self) -> None:
        decimator = Decimator(48001)
        produced = int(decimator.process(np.zeros(48001, dtype=np.float32)).shape[0])
        produced += int(decimator.flush().shape[0])
        assert produced == decimator.expected_output

    def test_a_dc_signal_passes_at_unity(self) -> None:
        """Unity DC gain, end to end rather than as a property of the coefficients.

        The steady-state middle is checked and the filter's ramp-up at each end is not:
        a windowed sinc necessarily takes its length to reach steady state, and asserting
        otherwise would be asserting something false.
        """
        samples = np.ones(48000, dtype=np.float32)
        derived = streamed(samples, 4096)
        steady = derived[200:-200]
        assert np.allclose(steady, 1.0, atol=1e-6)
