"""The marker waveform: exact, platform-independent, and actually a chirp.

Three layers, guarding three different failures, in the arrangement `test_fir.py`
established for the decimation filter.

:class:`TestTheTableIsReproducible` re-runs the design and compares. It is a change
detector: it catches a hand-edited entry and proves nothing at all about correctness.

:class:`TestTheTableIsASine` is the real acceptance test for the table. It measures the
committed array against ``math.sin`` — an implementation nothing here shares a line of code
with — and against the endpoints, monotonicity and symmetry identities the evaluator relies
on. Without it, "integer sine table" degrades into an arbitrary array that happens to produce
a waveform, and every frozen hash downstream would be pinning noise.

:class:`TestTheChirpSweeps` and :class:`TestTheWaveformIsExact` do the same job one level up
for synthesis: the phase formula is checked by *differencing successive phases* and comparing
against the frequencies the spec declares, which is a property of the arithmetic rather than
of the spectrum it happens to produce.
"""

from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.determinism import sha256_bytes
from dnd_audio.marker import MARKER_SAMPLE_RATE
from dnd_audio.marker.sine import (
    QUARTER_STEPS,
    TABLE_PATH,
    TABLE_SCALE,
    TURN_STEPS,
    SineTable,
    SineTableError,
    load_sine_table,
    round_half_away,
)
from dnd_audio.marker.spec import MARKER_SPECS, ChirpSpec, MarkerSpec, UnknownMarkerError, resolve
from dnd_audio.marker.synth import chirp_phase_numerator, marker_samples, marker_templates
from dnd_audio.marker.wav import marker_wav_bytes

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_SPECS = pytest.mark.parametrize("spec", MARKER_SPECS.values(), ids=list(MARKER_SPECS))


@pytest.fixture(scope="module")
def table() -> SineTable:
    return load_sine_table()


class TestTheTableIsReproducible:
    """A hand-edited entry is caught. This proves nothing else."""

    def test_the_design_script_reproduces_the_committed_file(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/design_sine_table.py", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"the committed sine table differs from what scripts/design_sine_table.py "
            f"produces:\n{completed.stdout}{completed.stderr}"
        )


class TestTheTableIsASine:
    """What the table must *be*, measured against an implementation it shares no code with."""

    def test_every_entry_matches_libm_to_the_unit(self, table: SineTable) -> None:
        """The design is computed in Decimal; this checks it landed where a sine is.

        Unit-exact rather than within a tolerance. The table is scaled by 2**30 and computed
        at fifty digits, so agreeing with a double-precision `math.sin` to the last integer
        is the expected outcome — a tolerance here would let a real error hide.
        """
        worst = max(
            abs(value - round(math.sin(index * math.pi / 2 / QUARTER_STEPS) * TABLE_SCALE))
            for index, value in enumerate(table.quarter)
        )
        assert worst == 0

    def test_the_endpoints_are_exact(self, table: SineTable) -> None:
        """sin(0) and sin(pi/2) are the two values that must not be interpolated."""
        assert table.quarter[0] == 0
        assert table.quarter[-1] == TABLE_SCALE

    def test_the_quarter_wave_rises_strictly(self, table: SineTable) -> None:
        """A repeated or reversed entry would put a discontinuity inside every chirp."""
        values = table.quarter
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_the_symmetry_the_evaluator_relies_on_holds_over_a_full_turn(
        self, table: SineTable
    ) -> None:
        """Quarter-wave folding is where the third quadrant's sign goes wrong.

        Checked against `math.sin` at every one of the 4096 steps rather than at the four
        quadrant boundaries: a sign error inside one quadrant passes a boundary check.
        """
        worst = max(
            abs(table.value(step) / TABLE_SCALE - math.sin(2 * math.pi * step / TURN_STEPS))
            for step in range(TURN_STEPS)
        )
        assert worst < 1e-9

    def test_the_cardinal_points_are_exactly_zero_and_plus_or_minus_one(
        self, table: SineTable
    ) -> None:
        assert table.value(0) == 0
        assert table.value(QUARTER_STEPS) == TABLE_SCALE
        assert table.value(2 * QUARTER_STEPS) == 0
        assert table.value(3 * QUARTER_STEPS) == -TABLE_SCALE

    def test_interpolation_stays_inside_the_declared_error_bound(self, table: SineTable) -> None:
        """The bound the file declares, measured at phases that miss every table point.

        9973 is prime and coprime with 4096, so no sampled phase lands on a table entry and
        every value exercises the interpolation rather than the lookup.
        """
        worst = max(
            abs(
                Fraction(*table.sine_at(numerator, 9973)) / TABLE_SCALE
                - Fraction(math.sin(2 * math.pi * numerator / 9973))
            )
            for numerator in range(9973)
        )
        assert float(worst) < 3.0e-7

    def test_sine_at_returns_an_exact_fraction_rather_than_a_rounded_value(
        self, table: SineTable
    ) -> None:
        """The property the one-rounding promise rests on (ADR-0041).

        A phase that lands between two table entries must produce a numerator that is *not*
        a multiple of its denominator — if it were, the interpolation had already been
        rounded away and composing it with an envelope would round twice.
        """
        numerator, denominator = table.sine_at(1, 9973)
        assert denominator == 9973
        assert numerator % denominator != 0


class TestTheTableIsValidated:
    """A malformed table is refused before anything is built from it."""

    def test_a_missing_file_names_the_script_that_makes_it(self, tmp_path: Path) -> None:
        load_sine_table.cache_clear()
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr("dnd_audio.marker.sine.TABLE_PATH", tmp_path / "absent.json")
                with pytest.raises(SineTableError, match="design_sine_table"):
                    load_sine_table()
        finally:
            load_sine_table.cache_clear()

    @pytest.mark.parametrize(
        ("mutation", "expected"),
        [
            ({"quarter": [0, 1]}, "quarter wave inclusive"),
            ({"quarter_steps": 7}, "[Rr]egenerate"),
            ({"scale": 3}, "[Rr]egenerate"),
        ],
        ids=["truncated", "wrong-steps", "wrong-scale"],
    )
    def test_a_corrupted_table_is_refused(
        self, tmp_path: Path, mutation: dict[str, object], expected: str
    ) -> None:
        import json

        document = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
        document.update(mutation)
        broken = tmp_path / "sine_table.json"
        broken.write_text(json.dumps(document), encoding="utf-8")

        load_sine_table.cache_clear()
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr("dnd_audio.marker.sine.TABLE_PATH", broken)
                with pytest.raises(SineTableError, match=expected):
                    load_sine_table()
        finally:
            load_sine_table.cache_clear()

    def test_the_endpoint_check_would_catch_a_rescaled_wave(self, tmp_path: Path) -> None:
        """Halving every entry keeps the shape and the length; only the endpoints reveal it."""
        import json

        document = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
        document["quarter"] = [value // 2 for value in document["quarter"]]
        broken = tmp_path / "sine_table.json"
        broken.write_text(json.dumps(document), encoding="utf-8")

        load_sine_table.cache_clear()
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr("dnd_audio.marker.sine.TABLE_PATH", broken)
                with pytest.raises(SineTableError, match="runs from"):
                    load_sine_table()
        finally:
            load_sine_table.cache_clear()


class TestRounding:
    """The one amplitude tie rule, stated rather than inherited."""

    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        [(1, 2, 1), (-1, 2, -1), (3, 2, 2), (-3, 2, -2), (1, 3, 0), (2, 3, 1), (-2, 3, -1)],
    )
    def test_halves_go_away_from_zero(
        self, numerator: int, denominator: int, expected: int
    ) -> None:
        assert round_half_away(numerator, denominator) == expected

    def test_it_disagrees_with_bankers_rounding_where_that_matters(self) -> None:
        """`round(0.5)` is 0 and `round(1.5)` is 2. A frozen waveform cannot depend on that."""
        assert round_half_away(1, 2) == 1 != round(0.5)
        assert round_half_away(3, 2) == 2 == round(1.5)

    def test_a_nonpositive_denominator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            round_half_away(1, 0)


class TestTheChirpSweeps:
    """The phase formula, checked as arithmetic rather than through a spectrum."""

    @ALL_SPECS
    def test_instantaneous_frequency_runs_from_start_to_end(self, spec: MarkerSpec) -> None:
        """Differencing successive phases recovers the frequency the spec declares.

        This is the direct statement of what a linear chirp is, and it is independent of the
        sine table, the envelope, and the quantization — so it fails for a wrong formula even
        where the output still looks like a sweep.
        """
        for chirp in spec.chirps:
            span = chirp.duration_samples - 1
            denominator = MARKER_SAMPLE_RATE * span

            def frequency(
                sample: int, chirp: ChirpSpec = chirp, den: int = denominator
            ) -> Fraction:
                step = chirp_phase_numerator(chirp, sample + 1) - chirp_phase_numerator(
                    chirp, sample
                )
                return Fraction(step * MARKER_SAMPLE_RATE, den)

            # Sample indices run 0..span inclusive, and `frequency(k)` is the instantaneous
            # frequency *at* sample k — the left endpoint of the step from k to k+1. So the
            # sweep's far end is at `span`, the last rendered sample, not at `span - 1`.
            assert frequency(0) == chirp.start_hz
            assert frequency(span) == chirp.end_hz
            # And exactly linear in between, not merely correct at the ends: a formula that
            # overshot and came back, or that swept quadratically, would pass an endpoint
            # check and fail here.
            for index in (1, span // 3, span // 2, span - 1):
                expected = chirp.start_hz + Fraction((chirp.end_hz - chirp.start_hz) * index, span)
                assert frequency(index) == expected

    @ALL_SPECS
    def test_the_phase_is_a_closed_form_not_an_accumulation(self, spec: MarkerSpec) -> None:
        """Phase at n equals the sum of the first n frequency steps, exactly.

        The property that makes the closed form legitimate. An accumulator would agree here
        and diverge by a rounding per sample; this asserts they are the *same number*.
        """
        chirp = spec.chirps[0]
        span = chirp.duration_samples - 1
        running = 0
        for sample in range(min(span, 512)):
            assert chirp_phase_numerator(chirp, sample) == running
            running += chirp.start_hz * span + (chirp.end_hz - chirp.start_hz) * sample


class TestTheWaveformIsExact:
    """Structure and determinism of the samples the WAV is built from."""

    @ALL_SPECS
    def test_two_builds_are_identical(self, spec: MarkerSpec) -> None:
        """INV-02 at the source. Rebuilt from a fresh spec so the cache is not what agrees."""
        rebuilt = MarkerSpec(
            name=spec.name,
            chirps=spec.chirps,
            gaps_samples=spec.gaps_samples,
            lead_silence_samples=spec.lead_silence_samples,
            trail_silence_samples=spec.trail_silence_samples,
            peak_amplitude=spec.peak_amplitude,
            rationale=spec.rationale,
        )
        assert sha256_bytes(marker_samples(spec).tobytes()) == sha256_bytes(
            marker_samples(rebuilt).tobytes()
        )

    @ALL_SPECS
    def test_the_silence_is_silent_and_the_gaps_are_gaps(self, spec: MarkerSpec) -> None:
        """The gaps *are* the code; a gap carrying energy would blur the sequence check."""
        samples = marker_samples(spec)
        assert not samples[: spec.lead_silence_samples].any()
        assert not samples[len(samples) - spec.trail_silence_samples :].any()
        for start, end in spec.gap_intervals():
            assert not samples[start:end].any()

    @ALL_SPECS
    def test_every_chirp_starts_and_ends_at_exactly_zero(self, spec: MarkerSpec) -> None:
        """What the raised-cosine fade is for: no discontinuity, no click, no spectral splatter."""
        samples = marker_samples(spec)
        for start, end in spec.chirp_intervals():
            assert samples[start] == 0
            assert samples[end - 1] == 0

    @ALL_SPECS
    def test_the_peak_reaches_the_configured_amplitude_and_never_exceeds_it(
        self, spec: MarkerSpec
    ) -> None:
        """Headroom is the point (OQ-025): the nearest lav must not clip."""
        samples = marker_samples(spec)
        assert int(np.abs(samples).max()) == spec.peak_amplitude
        assert spec.peak_amplitude < 32768

    @ALL_SPECS
    def test_the_anchor_is_the_first_sample_of_the_first_chirp(self, spec: MarkerSpec) -> None:
        """ADR-0041's frozen anchor. Every lag in the project is measured from this."""
        assert spec.anchor_sample == spec.chirp_intervals()[0][0]
        assert spec.anchor_sample == spec.lead_silence_samples

    @ALL_SPECS
    def test_the_templates_are_slices_of_the_waveform_not_a_second_synthesis(
        self, spec: MarkerSpec
    ) -> None:
        """The rule ADR-0041 exists for: the detector matches the bytes that were played."""
        samples = marker_samples(spec)
        for template, (start, end) in zip(
            marker_templates(spec), spec.chirp_intervals(), strict=True
        ):
            assert np.array_equal(template, samples[start:end])

    @ALL_SPECS
    def test_the_energy_lands_inside_the_declared_band(self, spec: MarkerSpec) -> None:
        """A formula error that still sweeps would show up here as energy outside the band."""
        samples = marker_samples(spec).astype(np.float64)
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
        frequencies = np.fft.rfftfreq(samples.size, 1 / MARKER_SAMPLE_RATE)
        low = min(min(chirp.start_hz, chirp.end_hz) for chirp in spec.chirps)
        high = max(max(chirp.start_hz, chirp.end_hz) for chirp in spec.chirps)
        inside = spectrum[(frequencies >= low * 0.9) & (frequencies <= high * 1.1)].sum()
        assert inside / spectrum.sum() > 0.99

    @ALL_SPECS
    def test_the_waveform_is_read_only(self, spec: MarkerSpec) -> None:
        """It is shared and cached; a caller that mutated it would corrupt every later build."""
        with pytest.raises(ValueError, match="read-only"):
            marker_samples(spec)[0] = 1


class TestTheRegistry:
    """The bench-selected v1 is data, while all three candidate names remain as history."""

    def test_v1_copies_the_bench_winner_but_keeps_its_public_name(self) -> None:
        winner = MARKER_SPECS["cand-b"]
        frozen = MARKER_SPECS["v1"]
        assert frozen.name == "v1"
        assert frozen.chirps == winner.chirps
        assert frozen.gaps_samples == winner.gaps_samples
        assert frozen.lead_silence_samples == winner.lead_silence_samples
        assert frozen.trail_silence_samples == winner.trail_silence_samples
        assert frozen.peak_amplitude == winner.peak_amplitude

    def test_building_without_a_name_resolves_v1(self) -> None:
        assert resolve(None) is MARKER_SPECS["v1"]

    def test_v1_wav_bytes_are_frozen_by_adr_0042(self) -> None:
        assert sha256_bytes(marker_wav_bytes(resolve(None))) == (
            "70355baad6bb72b38e0b606cddbbaa3428c11429bec74cd127aa6f8935ecdf6f"
        )

    def test_an_unknown_name_lists_what_this_build_carries(self) -> None:
        with pytest.raises(UnknownMarkerError, match="cand-a"):
            resolve("cand-z")

    @ALL_SPECS
    def test_every_candidate_resolves_by_name(self, spec: MarkerSpec) -> None:
        assert resolve(spec.name) is spec

    @ALL_SPECS
    def test_the_gaps_are_unequal(self, spec: MarkerSpec) -> None:
        """Equal gaps make a reversed sequence indistinguishable from the real one."""
        assert len(set(spec.gaps_samples)) == len(spec.gaps_samples)

    def test_equal_gaps_are_refused_at_construction(self) -> None:
        chirp = ChirpSpec(start_hz=500, end_hz=8000, duration_samples=4800, fade_samples=480)
        with pytest.raises(ValueError, match="time-reversed"):
            MarkerSpec(
                name="bad",
                chirps=(chirp, chirp, chirp),
                gaps_samples=(4800, 4800),
                lead_silence_samples=4800,
                trail_silence_samples=4800,
                peak_amplitude=1 << 14,
                rationale="equal gaps",
            )

    def test_a_tone_is_refused_because_its_peak_is_ambiguous_by_whole_cycles(self) -> None:
        with pytest.raises(ValueError, match="sweeps"):
            ChirpSpec(start_hz=1000, end_hz=1000, duration_samples=4800, fade_samples=480)

    def test_a_frequency_at_or_above_nyquist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Nyquist"):
            ChirpSpec(start_hz=500, end_hz=24000, duration_samples=4800, fade_samples=480)
