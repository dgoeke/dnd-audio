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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureSession, FixtureTruth, build_session, canonical_session
from dnd_audio.fixtures.variants import (
    DRIFT_END_SHIFT_SAMPLES,
    constant_offset_session,
    drift_session,
)
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE, DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.pcm import open_pcm
from dnd_audio.timeline.runner import run_ingest
from dnd_audio.timeline.syncqa import measure_lag, offset_floor_samples
from tests.manifests import bwf, config_for, timecode

RATE = CANONICAL_SAMPLE_RATE


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


def decisions(report_path: Path, code: str) -> dict[str, dict[str, str]]:
    """Every QA decision of one kind, by `track:position`."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        decision["subject"]: decision["details"]
        for decision in report["decisions"]
        if decision["code"] == code
    }


def measurements(report_path: Path) -> dict[str, dict[str, str]]:
    return decisions(report_path, "sync_qa_measured")


def _silence_one_track(truth: FixtureTruth) -> None:
    """Overwrite `tx-b`'s audio with digital silence, before the run reads anything.

    A dead channel is what `sync_qa_no_signal` exists to name, and the fixture generator
    always writes a noise floor — deliberately, because a real recording has one. Writing
    the zeros here rather than teaching the generator to produce silence keeps that default
    honest, and it happens before `run_ingest` takes its INV-01 snapshot.
    """
    for chunk in truth.for_track("tx-b"):
        path = truth.session_dir / chunk.relative_path
        source = open_pcm(path)
        with path.open("r+b") as handle:
            handle.seek(source.data_offset)
            handle.write(bytes(source.data_bytes))


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


class TestThreeOutcomes:
    """A weak peak, a silent window, and a good measurement are three different facts.

    Before M8 the first two shared one code that said "no shared transient found", which
    cost the 2026-08-03 capture six correct measurements: ordinary speech does not
    correlate like a clap, and the lags it reported matched an independent hand
    measurement. The instrument had the answer and threw it away.
    """

    def test_a_weak_peak_keeps_its_lag_and_raises_nothing(self, drift: FixtureTruth) -> None:
        """Above the achievable correlation, the measurement is kept and marked."""
        with_qa(drift.session_dir, min_correlation=1.0)
        result = run_ingest(drift.session_dir)
        assert result.timeline is not None
        codes = {note.code for note in result.timeline.warnings}
        assert "sync_qa_low_confidence" in codes
        assert "clock_drift_suspected" not in codes
        assert "timecode_disagreement" not in codes

        # The evidence survives: the lag it found is in the report, with its correlation.
        weak = decisions(result.report_path, "sync_qa_low_confidence")
        assert weak, "a low-confidence measurement must still be recorded"
        assert float(weak["tx-b:end"]["lag_ms"]) == DRIFT_END_SHIFT_SAMPLES * 1000 / RATE
        assert 0.0 < float(weak["tx-b:end"]["correlation"]) < 1.0

    def test_a_silent_window_is_not_a_weak_measurement(self, tmp_path: Path) -> None:
        """ "Nobody clapped" and "the jam failed" must not read the same way."""
        spec = drift_session()
        truth = build_session(replace(spec, claps=(), speech=()), tmp_path / "silent")
        _silence_one_track(truth)
        with_qa(truth.session_dir, window_s=2)

        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        codes = {note.code for note in result.timeline.warnings}
        assert "sync_qa_no_signal" in codes
        assert "sync_qa_low_confidence" not in codes
        assert not decisions(result.report_path, "sync_qa_measured")

    def test_a_session_with_no_shared_transient_reports_low_confidence(
        self, tmp_path: Path
    ) -> None:
        """The canonical fixture's second half has no clap in it.

        Its only shared transient is at 4 s, so a window at the end correlates two
        unrelated noise floors. That is a *weak* peak rather than nothing at all — there is
        audio, it simply does not agree — and it must not raise a disagreement.
        """
        truth = build_session(canonical_session(), tmp_path / "canonical")
        with_qa(truth.session_dir, window_s=2, min_correlation=0.5)
        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        codes = {note.code for note in result.timeline.warnings}
        assert "sync_qa_low_confidence" in codes
        assert "timecode_disagreement" not in codes


class TestTheConstantOffsetThreshold:
    """Defect 5a. A constant offset cannot be finer than one tick of a timecode counter.

    The jam-verification run raised `timecode_disagreement` at **+11.31 ms** — well inside
    the 33.3 ms quantum OQ-024 established as this hardware's floor — because a single
    5 ms `drift_warn_ms` governed both the constant offset and the start-to-end change. A
    threshold below the quantization floor fires on every healthy session, which trains an
    operator to ignore the one warning that matters.
    """

    def test_an_offset_inside_one_frame_raises_nothing(self, tmp_path: Path) -> None:
        """543 samples is 11.31 ms: the number the real capture warned about."""
        truth = build_session(constant_offset_session(543), tmp_path / "inside")
        with_qa(truth.session_dir)
        result = run_ingest(truth.session_dir)
        assert result.timeline is not None

        found = decisions(result.report_path, "sync_qa_measured")
        assert float(found["tx-b:start"]["lag_ms"]) == pytest.approx(11.3125, abs=0.07)
        assert "timecode_disagreement" not in {n.code for n in result.timeline.warnings}

    def test_an_offset_far_beyond_one_frame_still_warns(self, tmp_path: Path) -> None:
        """The threshold widened to the hardware's floor, not into silence."""
        truth = build_session(constant_offset_session(120 * RATE // 1000), tmp_path / "beyond")
        with_qa(truth.session_dir, max_lag_ms=200)
        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        assert "timecode_disagreement" in {n.code for n in result.timeline.warnings}

    def test_the_floor_comes_from_the_evidence_not_the_configured_frame_rate(self) -> None:
        """OQ-024: a receiver set to 60 fps wrote 30/1 references anyway.

        So a 60F session's `bext` evidence still moves in 1600-sample steps, and deriving
        the floor from `timecode.frame_rate` would give it a 16.7 ms threshold against
        source timing that has not changed — reinstating the false alarm.
        """
        settings = config_for(("tx-a",), frame_rate="60F")
        from_bwf = offset_floor_samples([bwf(0)], settings, rate=DERIVATIVE_SAMPLE_RATE)
        assert from_bwf == 534  # ceil(1600 * 16000 / 48000): 33.375 ms

        # A timecode tag really is finer at 60 fps, and a session carrying only those says so.
        from_timecode = offset_floor_samples(
            [timecode("00:00:00:00", "60F")], settings, rate=DERIVATIVE_SAMPLE_RATE
        )
        assert from_timecode == 267  # ceil(16000 / 60): 16.7 ms

        # The pair takes the coarser, which is the only safe reading.
        both = offset_floor_samples(
            [bwf(0), timecode("00:00:00:00", "60F")], settings, rate=DERIVATIVE_SAMPLE_RATE
        )
        assert both == 534

    def test_an_unmeasurable_threshold_is_refused_rather_than_widened(self) -> None:
        """Silently raising 5 ms to 33 leaves the operator's belief intact and wrong."""
        with pytest.raises(Exception, match="finer than one frame"):
            config_for(("tx-a",), frame_rate="30F", sync_qa={"enabled": True, "offset_warn_ms": 5})

    def test_a_stated_threshold_is_refused_against_the_quantum_not_the_frame_rate(self) -> None:
        """The gap between the two halves above, which each passed on its own.

        `offset_floor_samples` takes the coarser of the two kinds of evidence, but a stated
        `offset_warn_ms` **replaces** it outright rather than being floored by it — so the
        only thing standing between an operator and a sub-quantum threshold is this
        validator. Deriving its floor from `frame_rate` alone accepted 20 ms at `60F`
        against evidence that still moves in 33.375 ms steps, which is defect 5a reinstated
        at exactly the rate the charter amended its criterion to cover. Both independent
        reviewers found it; `test_an_unmeasurable_threshold_is_refused_rather_than_widened`
        could not, because at 30F the frame floor and the BWF floor are the same number.
        """
        with pytest.raises(Exception, match="BWF reference quantum of 1600 samples"):
            config_for(("tx-a",), frame_rate="60F", sync_qa={"enabled": True, "offset_warn_ms": 20})

        # 34 ms clears the quantum and is accepted — the refusal is a floor, not a ban.
        accepted = config_for(
            ("tx-a",), frame_rate="60F", sync_qa={"enabled": True, "offset_warn_ms": 34}
        )
        assert accepted.sync_qa.offset_warn_ms == 34

        # A recorder that really is sample-exact restores the frame-rate floor exactly.
        exact = config_for(
            ("tx-a",),
            frame_rate="60F",
            bwf_reference_quantum_samples=1,
            sync_qa={"enabled": True, "offset_warn_ms": 17},
        )
        assert exact.sync_qa.offset_warn_ms == 17

    def test_a_healthy_60f_session_inside_the_quantum_raises_nothing(self, tmp_path: Path) -> None:
        """The defect the validator above now prevents, measured end to end.

        25 ms of cross-receiver offset at `60F`: inside the 33.375 ms the hardware can
        actually express, so a healthy session. Before the fix a stated 20 ms was accepted
        here and this run raised `timecode_disagreement` twice.
        """
        truth = build_session(constant_offset_session(25 * RATE // 1000), tmp_path / "sixty")
        path = truth.session_dir / "session.yaml"
        document = yaml.safe_load(path.read_text())
        document["timecode"]["frame_rate"] = "60F"
        document["sync_qa"] = {"enabled": True, "max_lag_ms": 200}
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")

        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        assert "timecode_disagreement" not in {n.code for n in result.timeline.warnings}


class TestThresholdsCompareAsIntegers:
    """INV-04 for defect 5a, in the two halves it actually splits into.

    **The charter's stated proof for this criterion does not work, and saying so is the
    point.** It asks for "a lag exactly one sample either side of the threshold, which no
    float-millisecond comparison resolves correctly". That premise is false: at 16 kHz an
    integer-millisecond threshold converts to a whole number of samples exactly, so
    `lag > threshold` and `lag * 1000 / 16000 > threshold * 1000 / 16000` agree on every
    integer lag, at every threshold, in both signs. Measured before this test was written
    rather than assumed — a boundary test justified that way would have been a test that
    cannot fail for the reason its docstring gives, which is the failure mode the verify
    phase exists to catch.

    So the two claims are separated. The boundary test below proves what it can actually
    observe — the comparison is strict and symmetric in sign — and the structural test
    proves the invariant itself, which at these magnitudes is only observable in the source.
    """

    def _assessed(self, lag: int, *, threshold: int) -> list[str]:
        import datetime as dt

        from dnd_audio.artifacts.report import ReportBuilder
        from dnd_audio.config import SyncQaConfig
        from dnd_audio.timeline.syncqa import LagMeasurement, _assess

        measurements = [
            LagMeasurement(track_id="tx-b", position="start", lag_samples=lag, correlation=0.9)
        ]
        builder = ReportBuilder(
            "boundary",
            config_hash=None,
            started_at=dt.datetime(2026, 8, 15, tzinfo=dt.UTC),
        )
        return [
            note.code
            for note in _assess(
                "tx-a",
                measurements,
                settings=SyncQaConfig(enabled=True),
                builder=builder,
                offset_threshold=threshold,
                drift_threshold=threshold,
            )
        ]

    @pytest.mark.parametrize("sign", [1, -1])
    def test_the_threshold_is_strict_and_symmetric(self, sign: int) -> None:
        """Exactly at the threshold is clean; one sample beyond it warns.

        Both signs, because the comparison takes an absolute value: a dropped `abs` leaves
        every positive case passing and fails only on a track that arrives *early*, which
        is half of them and the half no one-sided test would see.
        """
        threshold = 534  # the 60F/BWF floor, in derivative samples
        assert "timecode_disagreement" not in self._assessed(sign * threshold, threshold=threshold)
        assert "timecode_disagreement" in self._assessed(
            sign * (threshold + 1), threshold=threshold
        )

    def test_no_lag_in_milliseconds_is_ever_an_operand_of_a_comparison(self) -> None:
        """The invariant itself, and the only form in which it is observable here.

        Parsed rather than grepped: `lag_ms` appears in five f-strings in this function, so
        a substring search for `<` or `>` would trip over format specifiers and prove
        nothing. Walking the AST asks the precise question — is a millisecond value an
        operand of a `Compare`? — and it stays true as the function grows, which is what
        this needs to be. The pre-M8 code compared `found.lag_ms` against a float threshold
        and defect 5a was about to add a second one.
        """
        import ast
        import inspect
        import textwrap

        from dnd_audio.timeline import syncqa

        tree = ast.parse(textwrap.dedent(inspect.getsource(syncqa._assess)))
        offenders = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            for operand in [node.left, *node.comparators]
            if any(
                isinstance(inner, ast.Attribute) and inner.attr.endswith("_ms")
                for inner in ast.walk(operand)
            )
        ]
        assert offenders == [], offenders


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
