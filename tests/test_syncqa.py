"""Clap cross-correlation as QA: it reports, and it never moves anything.

The distinction the spec draws is between a *constant* lag and a *changing* one. A constant
lag means the receivers disagree about what time it is. A lag that changes between the start
and the end of a session means their sample clocks are running at different rates — OQ-006,
the assumption the whole MVP rests on and has no evidence for.

The drift fixture is built so that only correlation can find it. Both tracks carry identical
metadata: the same chunk start, the same `bext` reference, so the timeline places them
exactly together. What differs is the *audio* — `tx-b` hears the end clap 960 samples after
`tx-a` does. A fixture that moved the metadata instead would let this file pass while the
correlator was never exercised at all; it would be measuring a number the fixture had
already declared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureSession, FixtureTruth, build_session, canonical_session
from dnd_audio.fixtures.variants import DRIFT_END_SHIFT_SAMPLES, drift_session
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE, DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.runner import run_ingest
from dnd_audio.timeline.syncqa import measure_lag


def with_qa(session_dir: Path, **settings: object) -> None:
    """Turn QA on in an already-written `session.yaml`."""
    path = session_dir / "session.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["sync_qa"] = {
        "enabled": True,
        "window_s": 3,
        "max_lag_ms": 100,
        "drift_warn_ms": 5,
        "min_correlation": 0.3,
        **settings,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")


def measurements(report_path: Path) -> dict[str, dict[str, str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        decision["subject"]: decision["details"]
        for decision in report["decisions"]
        if decision["code"] == "sync_qa_measured"
    }


@pytest.fixture
def drift(tmp_path: Path) -> FixtureTruth:
    truth = build_session(drift_session(), tmp_path / "drift")
    with_qa(truth.session_dir)
    return truth


class TestMeasureLag:
    """The correlation itself, on signals whose answer is arithmetic."""

    def test_an_identical_signal_has_zero_lag_and_unit_correlation(self) -> None:
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(4000).astype(np.float32)
        lag, peak = measure_lag(signal, signal, max_lag_samples=100)
        assert lag == 0
        assert peak == pytest.approx(1.0)

    @pytest.mark.parametrize("shift", [-50, -1, 1, 37, 100])
    def test_a_shifted_copy_reports_its_shift(self, shift: int) -> None:
        """Sign included: positive means this track's audio arrives later."""
        rng = np.random.default_rng(2)
        signal = rng.standard_normal(4000).astype(np.float32)
        shifted = np.roll(signal, shift).astype(np.float32)
        lag, peak = measure_lag(signal, shifted, max_lag_samples=200)
        assert lag == shift
        assert peak > 0.9

    def test_unrelated_noise_correlates_weakly(self) -> None:
        """Which is the whole reason for a confidence threshold.

        Two independent noise floors always produce *some* peak. Without a floor under it,
        QA would report a confident lag for a clap that was never recorded.
        """
        rng = np.random.default_rng(3)
        first = rng.standard_normal(8000).astype(np.float32)
        second = rng.standard_normal(8000).astype(np.float32)
        _, peak = measure_lag(first, second, max_lag_samples=200)
        assert peak < 0.2

    def test_normalization_makes_a_quiet_track_comparable(self) -> None:
        """An unnormalized correlation would rank tracks by volume."""
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(4000).astype(np.float32)
        quiet = (signal * 0.001).astype(np.float32)
        lag, peak = measure_lag(signal, quiet, max_lag_samples=100)
        assert lag == 0
        assert peak == pytest.approx(1.0, abs=1e-6)

    def test_silence_is_not_a_measurement(self) -> None:
        signal = np.zeros(1000, dtype=np.float32)
        assert measure_lag(signal, signal, max_lag_samples=100) == (0, 0.0)


class TestDriftDetection:
    def test_the_end_lag_is_found_in_the_audio(self, drift: FixtureTruth) -> None:
        """20 ms, injected into the samples and nowhere else.

        The metadata says both tracks start together, so this number exists only in the
        waveform. Finding it is the correlator working; a fixture with shifted metadata
        could not distinguish that from the correlator reading its own input.
        """
        result = run_ingest(drift.session_dir)
        assert result.exit_code is ExitCode.OK

        found = measurements(result.report_path)
        # 960 samples at 48 kHz is 320 at 16 kHz, so the answer lands exactly on the
        # measurement grid. A tolerance here would be hiding up to three samples of error
        # in a quantity that has none.
        expected_ms = DRIFT_END_SHIFT_SAMPLES * 1000 / CANONICAL_SAMPLE_RATE
        assert expected_ms == 20.0
        assert float(found["tx-b:start"]["lag_ms"]) == 0.0
        assert float(found["tx-b:end"]["lag_ms"]) == expected_ms

    def test_a_changing_lag_warns_about_drift(self, drift: FixtureTruth) -> None:
        result = run_ingest(drift.session_dir)
        assert result.timeline is not None
        codes = {note.code for note in result.timeline.warnings}
        assert "clock_drift_suspected" in codes

    def test_the_timeline_is_not_adjusted(self, drift: FixtureTruth) -> None:
        """The spec's boundary: QA reports disagreement, it does not override timecode.

        Compared byte for byte against the same session with QA switched off, so a
        correction of any size — including one that only moved a derivative — would show.
        """
        with_qa(drift.session_dir, enabled=True)
        with_correction = run_ingest(drift.session_dir)
        assert with_correction.timeline is not None
        placed_with_qa = {
            track.track_id: track.start_sample for track in with_correction.timeline.tracks
        }

        document = yaml.safe_load((drift.session_dir / "session.yaml").read_text())
        document["sync_qa"]["enabled"] = False
        (drift.session_dir / "session.yaml").write_text(yaml.safe_dump(document, sort_keys=True))

        without = run_ingest(drift.session_dir)
        assert without.timeline is not None
        assert {
            track.track_id: track.start_sample for track in without.timeline.tracks
        } == placed_with_qa
        assert placed_with_qa == {"tx-a": 0, "tx-b": 0}

    def test_qa_never_fails_a_run(self, drift: FixtureTruth) -> None:
        """A drift warning is evidence for OQ-006, not a reason to refuse the session."""
        assert run_ingest(drift.session_dir).exit_code is ExitCode.OK


class TestConfidence:
    def test_a_high_threshold_reports_the_measurement_as_inconclusive(
        self, drift: FixtureTruth
    ) -> None:
        """Above the achievable correlation, QA says so rather than reporting a lag."""
        with_qa(drift.session_dir, min_correlation=1.0)
        result = run_ingest(drift.session_dir)
        assert result.timeline is not None
        codes = {note.code for note in result.timeline.warnings}
        assert "sync_qa_inconclusive" in codes
        assert "clock_drift_suspected" not in codes

    def test_a_session_with_no_shared_transient_is_inconclusive(self, tmp_path: Path) -> None:
        """The canonical fixture's second half has no clap in it.

        Its only shared transient is at 4 s, so a window at the end correlates two
        unrelated noise floors — and the honest answer is that nothing was found, not a
        lag derived from whichever noise sample happened to line up.
        """
        truth = build_session(canonical_session(), tmp_path / "canonical")
        with_qa(truth.session_dir, window_s=2, min_correlation=0.5)
        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        assert any(note.code == "sync_qa_inconclusive" for note in result.timeline.warnings)


class TestWhenQaDoesNotRun:
    def test_it_is_off_by_default(self, canonical_fixture: FixtureTruth) -> None:
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        assert not [note for note in result.timeline.warnings if note.code.startswith("sync_qa")]
        assert not measurements(result.report_path)

    def test_a_session_too_short_to_correlate_says_so(self, tmp_path: Path) -> None:
        """Rather than correlating a window too small for a peak to mean anything."""
        spec = FixtureSession(
            session_id="tiny",
            title="Tiny",
            tracks=drift_session().tracks[:2],
        )
        # Rebuild with very short chunks by trimming the spec's own tracks.
        from dataclasses import replace

        from dnd_audio.fixtures.session import FixtureChunk

        short = tuple(
            replace(track, chunks=(FixtureChunk(start_sample=0, n_samples=2000, sequence=1),))
            for track in spec.tracks
        )
        truth = build_session(replace(spec, tracks=short, claps=(), speech=()), tmp_path / "tiny")
        with_qa(truth.session_dir, window_s=30)

        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        assert any(note.code == "sync_qa_skipped" for note in result.timeline.warnings)


class TestTheMeasurementIsAtTheDerivativeRate:
    def test_lag_resolution_is_one_sixteen_kilohertz_sample(self, drift: FixtureTruth) -> None:
        """62.5 microseconds, which is far finer than any threshold expressed in ms.

        Recorded because it is the assumption behind measuring on the derivative rather
        than the 48 kHz path: three times cheaper, and still an order of magnitude finer
        than the smallest drift anyone would act on.
        """
        result = run_ingest(drift.session_dir)
        found = measurements(result.report_path)
        step = 1000 / DERIVATIVE_SAMPLE_RATE
        for details in found.values():
            remainder = abs(float(details["lag_ms"])) % step
            assert remainder < 1e-9 or abs(remainder - step) < 1e-9
