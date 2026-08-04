"""`dnd-audio ingest`, end to end, on real session directories.

`tests/test_origin.py` and `tests/test_layout.py` test the rules in isolation against
hand-built manifests. This file runs the actual command against actual audio, which is the
only thing that can show the pieces agreeing: that FFprobe reports what the strategy chain
expects, that the refusals happen *before* the first placement rather than after, and that
a failure still leaves a report behind.

Two habits from M1's closeout are followed deliberately. Every failure test **starts from a
stale artifact already on disk**, because a test that starts from an empty directory cannot
see the difference between "removed the stale one" and "never wrote one". And every claim
about ordering — "fails before timeline construction" — is checked by watching whether the
later stage was *entered*, not by observing that its output is absent.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
import yaml

from dnd_audio.artifacts.manifest import Manifest
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureSession, FixtureTruth, build_session
from dnd_audio.fixtures.variants import (
    BWF_REFERENCE_QUANTUM_SAMPLES,
    QUANTIZED_BACKWARD_SAMPLES,
    WALL_CLOCK_SKEW,
    drop_frame_session,
    inconsistent_rate_session,
    mixed_format_session,
    no_origin_session,
    nonconforming_rate_session,
    overlapping_session,
    quantized_reference_session,
    rollover_session,
    wall_clock_skew_session,
)
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE, TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.runner import IngestResult, run_ingest

RATE = CANONICAL_SAMPLE_RATE


@pytest.fixture
def a_session(tmp_path: Path) -> Callable[[FixtureSession], FixtureTruth]:
    """Build any fixture variant into its own directory."""

    def build(spec: FixtureSession) -> FixtureTruth:
        return build_session(spec, tmp_path / spec.session_id)

    return build


def stale_timeline(session_dir: Path) -> Path:
    """Plant a timeline from an imaginary earlier run.

    Every failure test starts from one. M1's closeout records the pattern this exists to
    avoid: a test that begins with an empty directory and then asserts a file is absent
    passes whether the code removed a stale artifact or simply never wrote one.
    """
    path = session_dir / TIMELINE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"stale": true}\n', encoding="utf-8")
    return path


def report_of(result: IngestResult) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(result.report_path.read_text(encoding="utf-8"))
    return document


def _stages(result: IngestResult) -> dict[str, Any]:
    return {stage["stage"]: stage for stage in report_of(result)["stages"]}


def _error_codes(result: IngestResult) -> set[str]:
    return {error["code"] for stage in report_of(result)["stages"] for error in stage["errors"]}


class TestTheCanonicalSession:
    def test_it_reconstructs_six_tracks(self, canonical_fixture: FixtureTruth) -> None:
        result = run_ingest(canonical_fixture.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None
        assert len(result.timeline.tracks) == 6

    def test_aligned_duration_matches_the_latest_source_end(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Criterion 4, within one 48 kHz sample, against the generator's own truth.

        The comparison is with the fixture's declared chunk table, not with any track: the
        latest end belongs to `tx-c` *after* its gap, and taking the shortest track, the
        first track, or the longest single chunk would each give a different wrong answer.
        """
        expected = max(chunk.start_sample + chunk.n_samples for chunk in canonical_fixture.chunks)
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        assert abs(result.timeline.duration_samples - expected) <= 1
        assert result.timeline.duration_samples == expected

    def test_post_gap_speech_stays_where_the_fixture_put_it(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """`tx-c` resumes at 384000, not at 240000 where concatenation would put it."""
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        track = next(t for t in result.timeline.tracks if t.track_id == "tx-c")
        audio = [segment for segment in track.segments if segment.kind == "audio"]
        declared = sorted(chunk.start_sample for chunk in canonical_fixture.for_track("tx-c"))
        assert [segment.session_start_sample for segment in audio] == declared

    def test_a_rerun_is_byte_identical(self, canonical_fixture: FixtureTruth) -> None:
        """INV-02, with a different clock and a warm cache on the second run."""
        first = run_ingest(
            canonical_fixture.session_dir, now=dt.datetime(2026, 8, 15, 19, tzinfo=dt.UTC)
        )
        before = first.timeline_path.read_bytes()
        second = run_ingest(
            canonical_fixture.session_dir, now=dt.datetime(2027, 1, 1, 3, tzinfo=dt.UTC)
        )
        assert second.timeline_path.read_bytes() == before

    def test_the_derivative_is_byte_identical_on_a_rebuild(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """A cached artifact that changed under a rebuild would make the cache a lie."""
        run_ingest(canonical_fixture.session_dir)
        derived = sorted((canonical_fixture.session_dir / "work/cache/audio/16000").glob("*.wav"))
        before = {path.name: path.read_bytes() for path in derived}
        assert before

        run_ingest(canonical_fixture.session_dir, use_cache=False)
        after = {
            path.name: path.read_bytes()
            for path in sorted(
                (canonical_fixture.session_dir / "work/cache/audio/16000").glob("*.wav")
            )
        }
        assert after == before

    def test_the_warm_run_reuses_inspection(self, canonical_fixture: FixtureTruth) -> None:
        """Inspection runs every time; when everything hits, the stage says `reused`.

        `skipped` would be false — the stage completed and its manifest is current — and a
        caller checking whether the manifest reflects what is on disk would read the wrong
        answer from it.
        """
        run_ingest(canonical_fixture.session_dir)
        warm = run_ingest(canonical_fixture.session_dir)
        stages = _stages(warm)
        assert stages["inspect"]["status"] == "complete"
        assert stages["inspect"]["origin"] == "reused"
        assert stages["reconstruct"]["status"] == "complete"

    def test_the_manifest_is_rewritten_rather_than_trusted(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """A manifest whose config hash matches is not evidence that it is current.

        Corrupting it and re-running must produce a valid manifest again, because
        inspection ran — not because the corrupt one was accepted.
        """
        run_ingest(canonical_fixture.session_dir)
        manifest_path = canonical_fixture.session_dir / "work/manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")

        result = run_ingest(canonical_fixture.session_dir)
        assert result.exit_code is ExitCode.OK
        assert json.loads(manifest_path.read_text())["tracks"]

    def test_the_report_carries_both_deliverables(self, canonical_fixture: FixtureTruth) -> None:
        result = run_ingest(canonical_fixture.session_dir)
        paths = {item["relative_path"] for item in report_of(result)["provenance"]["deliverables"]}
        assert paths == {"work/manifest.json", "work/timeline.json"}


class TestRefusalsHappenBeforeConstruction:
    """Spec criterion 13, proven by what was *entered* rather than by what is absent.

    These runs exit **4, not 1**, and that is correct rather than a compromise. Inspection
    completed and produced a real, current `work/manifest.json` — the artifact that
    *explains* the refusal, since it carries the offending source's container facts. So the
    run is `partial` by ADR-0005's definition: at least one stage failed and at least one
    produced a deliverable. INV-13's requirement is that partial success never exits zero,
    and a caller that needs the detail reads the per-stage status rather than the code.
    """

    @pytest.mark.parametrize(
        ("spec", "code"),
        [
            (nonconforming_rate_session, "unsupported_sample_rate"),
            (inconsistent_rate_session, "inconsistent_sample_rate"),
        ],
        ids=["44100Hz", "mixed-rates"],
    )
    def test_a_bad_rate_fails_without_building_anything(
        self,
        a_session: Callable[[FixtureSession], FixtureTruth],
        monkeypatch: pytest.MonkeyPatch,
        spec: Callable[[], FixtureSession],
        code: str,
    ) -> None:
        truth = a_session(spec())
        stale = stale_timeline(truth.session_dir)

        entered: list[str] = []
        _spy(monkeypatch, entered)

        result = run_ingest(truth.session_dir)
        assert result.exit_code is not ExitCode.OK
        assert entered == [], f"these ran despite the refusal: {entered}"
        assert not stale.exists()
        assert result.timeline is None

        stages = _stages(result)
        assert stages["reconstruct"]["status"] == "failed"
        assert stages["inspect"]["status"] == "complete"
        assert report_of(result)["overall_status"] == "partial"
        assert code in _error_codes(result)

    def test_a_material_overlap_is_fatal_by_default(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        truth = a_session(overlapping_session())
        stale = stale_timeline(truth.session_dir)
        result = run_ingest(truth.session_dir)
        assert result.exit_code is not ExitCode.OK
        assert not stale.exists()
        assert "chunk_overlap" in _error_codes(result)

    def test_a_failure_still_writes_a_report(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """INV-13. The report is how an operator finds out what went wrong."""
        truth = a_session(nonconforming_rate_session())
        result = run_ingest(truth.session_dir)
        assert result.report_path.exists()
        assert report_of(result)["overall_status"] == "partial"
        assert result.exit_code == ExitCode.PARTIAL

    def test_a_session_with_no_config_reports_rather_than_raising(self, tmp_path: Path) -> None:
        """The failure that happens before inspection can even start.

        `build()` refuses a report with a stage unaccounted for, so an early failure used
        to produce no report at all — the exact outcome INV-13 exists to prevent.
        """
        session_dir = tmp_path / "empty"
        (session_dir / "raw").mkdir(parents=True)
        result = run_ingest(session_dir)
        assert result.exit_code is ExitCode.FATAL
        assert result.report_path.exists()
        stages = _stages(result)
        assert stages["inspect"]["status"] == "failed"
        assert stages["reconstruct"]["status"] == "failed"


def _placement_only(document: dict[str, Any]) -> dict[str, Any]:
    """A timeline with exactly the fields that hash the source bytes removed.

    Two sessions that differ only in their `bext` date tags are still two different sets of
    bytes, so `manifest_sha256` and each segment's `source_sha256` must differ — those are
    doing their job. Everything else is placement, and none of it may move. Listing the
    exclusions here rather than comparing a hand-picked subset keeps the claim maximal: any
    field added to this artifact in future is compared by default.
    """
    content_addressed = {"manifest_sha256", "source_sha256", "cache_key", "relative_path"}

    def prune(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: prune(value)
                for key, value in node.items()
                if key not in content_addressed
                # A source's own path is placement, not addressing; only the derivative
                # cache's path is derived from a hash.
                or (key == "relative_path" and str(value).startswith("raw/"))
            }
        if isinstance(node, list):
            return [prune(item) for item in node]
        return node

    pruned: dict[str, Any] = prune(document)
    return pruned


class TestWallClockNeverAnchorsPlacement:
    """ADR-0031, measured: two receivers 48.7 s apart with timecode agreeing to a frame.

    `bext.origination_date`/`origination_time` carry the receiver's real-time clock, and on
    2026-08-03 two of them disagreed by nearly a minute while their jammed timecode agreed
    to under one frame. `_cycles_from_dates` applied recorded dates as whole 24-hour cycles,
    so a pair straddling midnight would have been placed a *day* apart on evidence known to
    be a minute wrong.
    """

    def test_a_48_second_wall_clock_disagreement_across_midnight_changes_no_placement(
        self, tmp_path: Path
    ) -> None:
        agreed = build_session(wall_clock_skew_session(), tmp_path / "agreed")
        skewed = build_session(wall_clock_skew_session(WALL_CLOCK_SKEW), tmp_path / "skewed")

        assert run_ingest(agreed.session_dir).exit_code is ExitCode.OK
        assert run_ingest(skewed.session_dir).exit_code is ExitCode.OK

        # The fixture really did write the disagreement — otherwise the comparison below
        # would be two identical sessions agreeing about nothing in particular.
        manifests = [
            Manifest.model_validate_json(
                (session / "work/manifest.json").read_text(encoding="utf-8")
            )
            for session in (agreed.session_dir, skewed.session_dir)
        ]
        recorded = [
            {
                track.track_id: getattr(source.start_time.evidence, "origination_date", None)
                for track in built.tracks
                for source in track.sources
                if source.start_time is not None
            }
            for built in manifests
        ]
        assert len(set(recorded[0].values())) == 1, "the control session must agree with itself"
        assert len(set(recorded[1].values())) == 2, "the skewed session must straddle midnight"

        first = json.loads(
            (agreed.session_dir / TIMELINE_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        second = json.loads(
            (skewed.session_dir / TIMELINE_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        assert first["manifest_sha256"] != second["manifest_sha256"]
        assert json.dumps(_placement_only(first), sort_keys=True) == json.dumps(
            _placement_only(second), sort_keys=True
        )

    def test_nothing_in_the_pipeline_reads_the_creation_time_tag(self) -> None:
        """FFprobe surfaces the same wall clock twice, as `date` and as `creation_time`.

        The second has never been read, which is why no rule was needed to stop it. A rule
        is needed now that the first is known untrustworthy — otherwise the obvious fix for
        a missing date is to reach for its twin.
        """
        root = Path(__file__).resolve().parent.parent / "src"
        offenders = [
            path
            for path in sorted(root.rglob("*.py"))
            if "creation_time" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []


class TestAQuantizedReferenceCounter:
    """Defect 2b, end to end: a real recorder's `bext` counter ticks once a frame.

    Unit coverage lives in `test_layout.py`; this is here because the claim is about what
    `ingest` does to a session directory, and because the previous behaviour was a *failed
    run* rather than a misplaced sample.
    """

    def test_a_second_chunk_rounded_backward_places_where_its_audio_is(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        truth = a_session(quantized_reference_session())
        second = sorted(truth.for_track("tx-a"), key=lambda c: c.start_sample)[1]
        # The fixture really did write a reference that under-reports its own start.
        assert second.time_reference % BWF_REFERENCE_QUANTUM_SAMPLES == 0

        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK, _error_codes(result)
        assert result.timeline is not None

        track = next(t for t in result.timeline.tracks if t.track_id == "tx-a")
        audio = [segment for segment in track.segments if segment.kind == "audio"]
        assert [segment.session_start_sample for segment in audio] == [
            chunk.start_sample
            for chunk in sorted(truth.for_track("tx-a"), key=lambda c: c.start_sample)
        ]
        assert audio[1].shift_samples == QUANTIZED_BACKWARD_SAMPLES

    def test_calling_the_counter_sample_exact_fails_the_session(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """The behaviour before M8, reachable by configuration — and the proof that the
        fixture exercises the tolerance rather than passing for some other reason."""
        truth = a_session(quantized_reference_session())
        document = yaml.safe_load((truth.session_dir / "session.yaml").read_text())
        document["timecode"]["bwf_reference_quantum_samples"] = 1
        (truth.session_dir / "session.yaml").write_text(
            yaml.safe_dump(document, sort_keys=True), encoding="utf-8"
        )

        result = run_ingest(truth.session_dir)
        assert result.exit_code is not ExitCode.OK
        assert "chunk_overlap" in _error_codes(result)


class TestAMixedFormatSession:
    """ADR-0030, end to end: a session half float and half 24-bit integer ingests.

    This is the only test here that runs a session the previous release *refused*, which
    is why it asserts the whole shape rather than one code — an exit status, a manifest
    that records the two widths, and audio that actually came from the 24-bit file.
    """

    def test_it_ingests_rather_than_refusing_half_the_session(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        truth = a_session(mixed_format_session())
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK, _error_codes(result)
        assert result.timeline is not None

        manifest = Manifest.model_validate_json(
            (truth.session_dir / "work/manifest.json").read_text(encoding="utf-8")
        )
        widths = {
            track.track_id: source.container.codec_name
            for track in manifest.tracks
            for source in track.sources
            if source.container is not None
        }
        assert widths == {"tx-a": "pcm_f32le", "tx-b": "pcm_s24le"}

    def test_the_24_bit_track_reaches_the_working_path_with_its_own_samples(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """Exit zero would also be satisfied by a track of silence."""
        from dnd_audio.timeline.pcm import PcmReader, open_pcm

        truth = a_session(mixed_format_session())
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK, _error_codes(result)

        chunk = next(c for c in truth.chunks if c.track_id == "tx-b")
        with PcmReader(open_pcm(truth.session_dir / chunk.relative_path)) as reader:
            source = reader.read(0, chunk.n_samples)
        assert reader.source.sample_format.codec_name == "pcm_s24le"
        # Speech, not a silent placeholder — the fixture puts a second of it on tx-b.
        assert float(np.abs(source).max()) > 0.05

    def test_the_operator_is_still_told_about_the_mismatch(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """Readable is not the same as intended. The setting was still wrong."""
        truth = a_session(mixed_format_session())
        run_ingest(truth.session_dir)
        manifest = Manifest.model_validate_json(
            (truth.session_dir / "work/manifest.json").read_text(encoding="utf-8")
        )
        codes = {
            note.code
            for track in manifest.tracks
            for source in track.sources
            for note in source.warnings
        }
        assert "unexpected_codec" in codes


class TestSessionZeroVariants:
    def test_a_session_with_no_configured_origin_starts_at_its_earliest_source(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        truth = a_session(no_origin_session())
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None
        assert result.timeline.session_zero.source == "earliest_source"

        starts = {track.track_id: track.start_sample for track in result.timeline.tracks}
        declared = {chunk.track_id: chunk.start_sample for chunk in truth.chunks}
        assert starts == declared
        assert min(starts.values()) == 0

    def test_a_session_across_midnight_is_unwrapped(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """`tx-b` is two seconds after `tx-a`, not 86 398 seconds before it."""
        truth = a_session(rollover_session())
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None

        starts = {track.track_id: track.start_sample for track in result.timeline.tracks}
        assert starts["tx-b"] - starts["tx-a"] == 2 * RATE
        assert result.timeline.duration_samples < 10 * RATE


class TestFractionalRatesEndToEnd:
    """Criterion 6's fractional cases, through the whole pipeline rather than in unit form.

    `tests/test_rasterize.py` proves the arithmetic against longhand `Fraction`s. This
    proves the arithmetic survives being written into a file, read by FFprobe, parsed by
    the strategy chain, and placed — which is where a rate that is exact on paper stops
    being exact in practice.
    """

    def test_drop_frame_chunks_land_on_their_declared_samples(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """29.97DF, with `INFO`/`ISMP` tags and no `bext` reference anywhere.

        At 30000/1001 fps a frame is 8008/5 samples, so the fixture's offsets are the only
        ones expressible exactly — and the generator refuses to write any other, which is
        what makes its declared truth worth comparing against.
        """
        truth = a_session(drop_frame_session())
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None
        assert result.timeline.frame_rate_label == "29.97DF"
        assert result.timeline.frame_rate.numerator == 30000
        assert result.timeline.frame_rate.denominator == 1001

        declared = {chunk.relative_path: chunk.start_sample for chunk in truth.chunks}
        placed = {
            segment.source_relative_path: segment.session_start_sample
            for track in result.timeline.tracks
            for segment in track.segments
            if segment.kind == "audio"
        }
        assert placed == declared

    def test_the_gap_between_them_is_exactly_fifty_frames(
        self, a_session: Callable[[FixtureSession], FixtureTruth]
    ) -> None:
        """50 frames at 8008/5 samples each is 80 080 — a whole number, and asserted as one.

        An implementation that rounded per frame rather than once would land at 80 100:
        fifty separate roundings of 1601.6 up to 1602.
        """
        truth = a_session(drop_frame_session())
        result = run_ingest(truth.session_dir)
        assert result.timeline is not None
        starts = {track.track_id: track.start_sample for track in result.timeline.tracks}
        assert starts["tx-b"] - starts["tx-a"] == 50 * 8008 // 5
        assert starts["tx-b"] - starts["tx-a"] != 50 * 1602


class TestExplicitOverrides:
    """Criterion 6's override case, through a real `session.yaml` rather than in unit form.

    `tests/test_origin.py` places a `SessionOffsetRecord` correctly, but at that boundary an
    override is indistinguishable from any other evidence. Only a session on disk proves
    the whole chain: the YAML is parsed, M1's strategy chain prefers the override over the
    file's own `bext` reference, and M2 places the result where the operator said.
    """

    def override_session(self, tmp_path: Path, **override: object) -> FixtureTruth:
        spec = no_origin_session()
        truth = build_session(spec, tmp_path / "override")
        target = sorted(truth.for_track("tx-b"), key=lambda c: c.start_sample)[0]
        document = yaml.safe_load((truth.session_dir / "session.yaml").read_text())
        document["recovery"]["source_time_overrides"] = {
            target.relative_path: {
                "sha256": target.sha256,
                "reason": "verification fixture: the field log disagrees with the file",
                **override,
            }
        }
        (truth.session_dir / "session.yaml").write_text(
            yaml.safe_dump(document, sort_keys=True), encoding="utf-8"
        )
        return truth

    def test_a_signed_offset_override_places_the_source(self, tmp_path: Path) -> None:
        """`tx-b`'s file says 2 s; the override says 5 s, and the override wins."""
        truth = self.override_session(tmp_path, start_offset_samples=5 * RATE)
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None

        starts = {track.track_id: track.start_sample for track in result.timeline.tracks}
        assert starts == {"tx-a": 0, "tx-b": 5 * RATE}

        # The fixture wrote `tx-b`'s own metadata saying 2 s. The override says 5 s, and
        # the placement follows the override — so this is the override winning, not the
        # file happening to agree with it.
        declared = {chunk.track_id: chunk.start_sample for chunk in truth.chunks}
        assert declared["tx-b"] == 2 * RATE
        assert starts["tx-b"] != declared["tx-b"]

    def test_a_negative_offset_override_shifts_the_timeline(self, tmp_path: Path) -> None:
        """The signed half of the field the spec permits, end to end.

        `tx-b` is placed a second *before* `tx-a`, and with no configured origin session
        zero is redefined as that earlier start — so `tx-a` moves to 1 s and every distance
        between the two is unchanged.
        """
        truth = self.override_session(tmp_path, start_offset_samples=-RATE)
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None
        starts = {track.track_id: track.start_sample for track in result.timeline.tracks}
        assert starts == {"tx-a": RATE, "tx-b": 0}

    def test_a_timecode_override_places_the_source(self, tmp_path: Path) -> None:
        """The other override form: a time of day copied out of a field log.

        `tx-a` starts at 19:00:00 by its own `bext`; the override puts `tx-b` at 19:00:03,
        which is three seconds later regardless of what `tx-b`'s own metadata says.
        """
        truth = self.override_session(tmp_path, start_timecode="19:00:03:00")
        result = run_ingest(truth.session_dir)
        assert result.exit_code is ExitCode.OK
        assert result.timeline is not None
        starts = {track.track_id: track.start_sample for track in result.timeline.tracks}
        assert starts == {"tx-a": 0, "tx-b": 3 * RATE}

    def test_the_override_is_recorded_as_a_decision(self, tmp_path: Path) -> None:
        """The spec requires overrides recorded prominently; a placement is not enough."""
        truth = self.override_session(tmp_path, start_offset_samples=5 * RATE)
        result = run_ingest(truth.session_dir)
        codes = {decision["code"] for decision in report_of(result)["decisions"]}
        assert "recovery_override_applied" in codes


class TestMaterializing48k:
    def test_it_is_off_by_default(self, canonical_fixture: FixtureTruth) -> None:
        """The segment map is the working path; 16 GB per session is opt-in."""
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        rates = {
            derivative.sample_rate
            for track in result.timeline.tracks
            for derivative in track.derivatives
        }
        assert rates == {16000}
        assert not (canonical_fixture.session_dir / "work/cache/audio/48000").exists()

    def test_it_writes_one_file_per_track_when_asked(self, canonical_fixture: FixtureTruth) -> None:
        result = run_ingest(canonical_fixture.session_dir, materialize_48k=True)
        assert result.timeline is not None
        for track in result.timeline.tracks:
            rates = {derivative.sample_rate for derivative in track.derivatives}
            assert rates == {16000, 48000}
            for derivative in track.derivatives:
                audio = canonical_fixture.session_dir / derivative.relative_path
                assert audio.stat().st_size == derivative.size_bytes

    def test_the_materialized_file_covers_the_whole_session(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Every track is the session's aligned duration, padded with silence at its ends."""
        result = run_ingest(canonical_fixture.session_dir, materialize_48k=True)
        assert result.timeline is not None
        duration = result.timeline.duration_samples
        for track in result.timeline.tracks:
            full = next(d for d in track.derivatives if d.sample_rate == 48000)
            assert full.output_samples == duration


class TestRawSourcesAreUntouched:
    def test_every_source_hash_is_unchanged(self, canonical_fixture: FixtureTruth) -> None:
        """INV-01, over the whole run rather than over the files it happened to read."""
        from dnd_audio.determinism import sha256_file

        before = {
            path: sha256_file(path)
            for path in sorted((canonical_fixture.session_dir / "raw").rglob("*"))
            if path.is_file()
        }
        assert run_ingest(canonical_fixture.session_dir).exit_code is ExitCode.OK
        after = {
            path: sha256_file(path)
            for path in sorted((canonical_fixture.session_dir / "raw").rglob("*"))
            if path.is_file()
        }
        assert after == before

    def test_a_source_changed_mid_run_is_caught(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check has to be able to fail, or it proves nothing.

        A source is corrupted from inside the run, after the snapshot is taken and before
        it is verified — which is the only window the invariant is about.
        """
        from dnd_audio.timeline import layout

        victim = canonical_fixture.session_dir / canonical_fixture.chunks[0].relative_path
        original = layout.reject_unusable_sources

        def corrupting(manifest: Manifest) -> None:
            original(manifest)
            with victim.open("r+b") as handle:
                handle.seek(0, 2)
                handle.write(b"\x00" * 16)

        monkeypatch.setattr("dnd_audio.timeline.runner.reject_unusable_sources", corrupting)

        result = run_ingest(canonical_fixture.session_dir)
        assert result.exit_code is not ExitCode.OK
        assert result.timeline is None
        assert "raw_sources_modified" in _error_codes(result)

    def test_a_failed_run_leaves_no_usable_derivative_behind(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sequence that poisoned the cache before the sidecars were staged.

        Run once cleanly. Corrupt a source mid-run so INV-01 fails — but note the
        derivative for that run was built from the *corrupted* bytes while its cache key
        was computed from the manifest, which describes the *original* bytes. Restore the
        file and the key matches again. Before staging, that served corrupted audio as a
        valid hit, permanently and silently.
        """
        from dnd_audio.determinism import sha256_file
        from dnd_audio.timeline import layout

        session_dir = canonical_fixture.session_dir
        victim = session_dir / canonical_fixture.chunks[0].relative_path
        pristine = victim.read_bytes()

        clean = run_ingest(session_dir)
        assert clean.timeline is not None
        good = {
            derivative.relative_path: sha256_file(session_dir / derivative.relative_path)
            for track in clean.timeline.tracks
            for derivative in track.derivatives
        }

        original = layout.reject_unusable_sources

        def corrupting(manifest: Manifest) -> None:
            original(manifest)
            victim.write_bytes(pristine[:-2000] + b"\x40" * 2000)

        monkeypatch.setattr("dnd_audio.timeline.runner.reject_unusable_sources", corrupting)
        failed = run_ingest(session_dir)
        assert failed.exit_code is not ExitCode.OK
        monkeypatch.undo()

        victim.write_bytes(pristine)
        again = run_ingest(session_dir)
        assert again.exit_code is ExitCode.OK
        assert again.timeline is not None
        for track in again.timeline.tracks:
            for derivative in track.derivatives:
                served = sha256_file(session_dir / derivative.relative_path)
                assert served == good[derivative.relative_path], (
                    f"{derivative.relative_path} was served from a run that failed INV-01"
                )

    def test_an_output_that_would_land_inside_raw_writes_nothing(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """INV-01 outranks INV-13 here: the report's own location is the violation."""
        output = canonical_fixture.session_dir / "output"
        output.mkdir(exist_ok=True)
        output.rmdir()
        output.symlink_to(canonical_fixture.session_dir / "raw" / "tx-a")

        result = run_ingest(canonical_fixture.session_dir)
        assert result.exit_code is ExitCode.FATAL
        assert not result.report_written
        assert not (canonical_fixture.session_dir / "raw/tx-a/ingest-report.json").exists()


#: Everything that must not be entered once a source has been refused: placement, layout,
#: reading a source, and writing a derivative.
_MUST_NOT_RUN: Final = ("determine_origin", "build_layout", "TrackReader", "WavWriter")


def _spy(monkeypatch: pytest.MonkeyPatch, entered: list[str]) -> None:
    """Record whether anything past the refusals was reached.

    The absence of `work/timeline.json` does not prove a source failed *before* timeline
    construction — it is equally consistent with building the whole thing and failing to
    write it. So the four steps that must not run are instrumented instead.
    """
    from dnd_audio.timeline import runner

    def watch(name: str) -> Callable[..., Any]:
        original = getattr(runner, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            entered.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in _MUST_NOT_RUN:
        monkeypatch.setattr(runner, name, watch(name))
