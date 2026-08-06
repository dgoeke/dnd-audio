"""`marker analyze` end to end, against sessions whose marker positions we chose.

**Ground truth is independent of the analyzer.** Every fixture below injects the canonical
marker into a session's raw audio at a sample this file picked, with a per-track delay this
file picked, before `ingest` runs. Every assertion is against those integers — never against
a value the detector produced.

The injection happens **before** the session is inspected, so nothing here modifies a source
during a run: the recordings simply contain a marker, exactly as they would after a bench.
INV-01 is about what a pipeline stage does, and these fixtures are the recording.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.cli import app
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth, build_session, canonical_session
from dnd_audio.marker import ANALYSIS_RELATIVE_PATH, MARKER_REPORT_RELATIVE_PATH
from dnd_audio.marker.analysis import (
    ArrivalOutcome,
    DetectionOutcome,
    GroupMember,
    OccurrenceGroup,
    SyncMarkerAnalysis,
)
from dnd_audio.marker.detect import DetectorThresholds, MarkerOccurrence
from dnd_audio.marker.eventlog import EventLogError, load_event_log
from dnd_audio.marker.inputs import read_session_artifacts
from dnd_audio.marker.report import (
    AnalysisStatus,
    MarkerReport,
    MarkerReportError,
    OverallStatus,
    ReportDeliverable,
)
from dnd_audio.marker.runner import (
    _Accumulator,
    _associate,
    _compare_arrival,
    _usable,
    marker_analyze_outputs,
    run_marker_analyze,
)
from dnd_audio.marker.spec import MARKER_SPECS
from dnd_audio.marker.synth import marker_samples
from dnd_audio.timeline.pcm import open_pcm
from dnd_audio.timeline.runner import run_ingest

runner = CliRunner()

SPEC: Final = MARKER_SPECS["cand-a"]

#: Session samples the marker is injected at. Both sit inside every track's recorded extent
#: on the canonical fixture, and neither is a multiple of the detector's block size.
FIRST: Final = 175_000
SECOND: Final = 400_000

#: Per-track acoustic delay, in samples. Deliberately unequal and deliberately small — a
#: table is 0.5-3 m across, which is 24 to 430 samples at 48 kHz.
DELAYS: Final[dict[str, int]] = {
    "tx-a": 0,
    "tx-b": 48,
    "tx-c": 96,
    "tx-d": 144,
    "tx-e": 192,
    "tx-f": 240,
}


def inject(truth: FixtureTruth, placements: list[int], *, gain: float = 0.5) -> None:
    """Add the marker to every chunk that fully contains it, before anything inspects it."""
    marker = marker_samples(SPEC).astype(np.float32) / 32768.0
    for chunk in truth.chunks:
        path = truth.session_dir / chunk.relative_path
        source = open_pcm(path)
        raw = path.read_bytes()
        end = source.data_offset + source.n_samples * 4
        audio = np.frombuffer(raw[source.data_offset : end], dtype="<f4").copy()
        for at in placements:
            local = at - chunk.start_sample + DELAYS[chunk.track_id]
            if local >= 0 and local + marker.size <= audio.size:
                audio[local : local + marker.size] += marker * gain
        path.write_bytes(raw[: source.data_offset] + audio.astype("<f4").tobytes() + raw[end:])


@pytest.fixture(scope="module")
def _ingested_templates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each session shape built and ingested **once**, to be copied per test.

    Building the canonical fixture, injecting, and ingesting it costs about 0.45 s, and this
    module asks for it in more than thirty tests. Nothing about that work varies per test —
    it is the same six transmitters, the same injected samples, the same ingest — so it is
    done here and each test below takes a copy.

    A *copy*, not a share: `marker analyze` writes its artifacts into the session directory,
    and tests here assert on what a run left behind. Sharing one directory would make each
    test's answer depend on which tests ran before it. The copy restores the isolation the
    old per-test fixture got by rebuilding, at roughly a fortieth of the cost.

    Copying is sound because an ingested session is relocatable: no artifact under it records
    an absolute path, only paths relative to the session root. :func:`_copy_of` asserts that
    the copy is byte-identical, so a future artifact that *did* embed one would fail here
    rather than silently make these fixtures wrong.
    """
    root = tmp_path_factory.mktemp("marker-templates")
    templates: dict[str, Path] = {}
    for name, placements in (("marked", [FIRST]), ("unmarked", []), ("two", [FIRST, SECOND])):
        truth = build_session(canonical_session(), root / name)
        if placements:
            inject(truth, placements)
        assert run_ingest(truth.session_dir).exit_code is ExitCode.OK
        templates[name] = truth.session_dir
    return templates


def _copy_of(template: Path, destination: Path) -> Path:
    """A private, byte-identical copy of an ingested session template."""
    shutil.copytree(template, destination)
    originals = sorted(path for path in template.rglob("*") if path.is_file())
    copies = sorted(path for path in destination.rglob("*") if path.is_file())
    assert [path.relative_to(template) for path in originals] == [
        path.relative_to(destination) for path in copies
    ]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(originals, copies, strict=True))
    return destination


@pytest.fixture
def marked_session(_ingested_templates: dict[str, Path], tmp_path: Path) -> Path:
    """A canonical session carrying the marker at :data:`FIRST`, ingested and ready."""
    return _copy_of(_ingested_templates["marked"], tmp_path / "session")


@pytest.fixture
def unmarked_session(_ingested_templates: dict[str, Path], tmp_path: Path) -> Path:
    """The canonical session, with no marker in it at all."""
    return _copy_of(_ingested_templates["unmarked"], tmp_path / "session")


def analysis_of(session_dir: Path) -> SyncMarkerAnalysis:
    document = (session_dir / ANALYSIS_RELATIVE_PATH).read_text(encoding="utf-8")
    return SyncMarkerAnalysis.model_validate_json(document)


class TestItFindsWhatWasPlaced:
    """Exact anchors, exact lags, from positions this file chose."""

    def test_the_anchor_is_the_sample_the_marker_was_placed_at(self, marked_session: Path) -> None:
        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert result.exit_code is ExitCode.OK
        analysis = analysis_of(marked_session)
        anchors = {item.track_id: item.anchor_sample for item in analysis.occurrences}
        assert set(anchors) == set(DELAYS)
        for track_id, delay in DELAYS.items():
            assert anchors[track_id] == FIRST + SPEC.anchor_sample + delay

    def test_every_relative_lag_is_the_delay_that_was_injected(self, marked_session: Path) -> None:
        """The measurement the whole instrument exists for, against known ground truth."""
        run_marker_analyze(marked_session, marker=SPEC.name)
        analysis = analysis_of(marked_session)
        assert analysis.groups
        group = analysis.groups[0]
        reference = analysis.identity.reference_track
        for member in group.members:
            if member.outcome is not DetectionOutcome.DETECTED:
                continue
            expected = DELAYS[member.track_id] - DELAYS[reference]
            assert member.relative_lag_samples == expected, member.track_id

    def test_the_anchor_maps_to_a_real_position_in_a_real_source_file(
        self, marked_session: Path
    ) -> None:
        run_marker_analyze(marked_session, marker=SPEC.name)
        analysis = analysis_of(marked_session)
        timeline = Timeline.model_validate_json(
            (marked_session / "work" / "timeline.json").read_text(encoding="utf-8")
        )
        by_track = {track.track_id: track for track in timeline.tracks}

        for item in analysis.occurrences:
            assert item.source_relative_path is not None
            assert item.source_sample is not None
            # Recomputed from the segment map independently of the runner's own mapping.
            segment = next(
                seg
                for seg in by_track[item.track_id].segments
                if seg.kind == "audio"
                and seg.session_start_sample <= item.anchor_sample < seg.session_end_sample
            )
            offset = item.anchor_sample - segment.session_start_sample
            assert item.source_relative_path == segment.source_relative_path
            assert item.source_sample == (segment.source_start_sample or 0) + offset

    def test_a_track_that_did_not_hear_it_is_missing_rather_than_absent(
        self, tmp_path: Path
    ) -> None:
        """A silent track is a fact about the capture, and must be visible as one."""
        truth = build_session(canonical_session(), tmp_path / "session")
        marker = marker_samples(SPEC).astype(np.float32) / 32768.0
        for chunk in truth.chunks:
            if chunk.track_id in ("tx-e", "tx-f"):
                continue
            path = truth.session_dir / chunk.relative_path
            source = open_pcm(path)
            raw = path.read_bytes()
            end = source.data_offset + source.n_samples * 4
            audio = np.frombuffer(raw[source.data_offset : end], dtype="<f4").copy()
            local = FIRST - chunk.start_sample + DELAYS[chunk.track_id]
            if local >= 0 and local + marker.size <= audio.size:
                audio[local : local + marker.size] += marker * 0.5
            path.write_bytes(raw[: source.data_offset] + audio.astype("<f4").tobytes() + raw[end:])
        run_ingest(truth.session_dir)

        result = run_marker_analyze(truth.session_dir, marker=SPEC.name)
        analysis = analysis_of(truth.session_dir)
        outcomes = {m.track_id: m.outcome for m in analysis.groups[0].members}
        assert outcomes["tx-e"] is DetectionOutcome.MISSING
        assert outcomes["tx-f"] is DetectionOutcome.MISSING
        assert result.report.inconclusive is True

    def test_weak_occurrences_are_reported_but_do_not_make_a_conclusive_group(
        self, marked_session: Path
    ) -> None:
        result = run_marker_analyze(
            marked_session,
            marker=SPEC.name,
            thresholds=DetectorThresholds(weak_signal_rms_permille=1000),
        )
        analysis = analysis_of(marked_session)
        assert analysis.occurrences
        assert all(item.weak for item in analysis.occurrences)
        assert analysis.groups == []
        assert result.report.inconclusive is True

    def test_clipped_occurrences_are_kept_distinct_and_inconclusive(self, tmp_path: Path) -> None:
        truth = build_session(canonical_session(), tmp_path / "session")
        inject(truth, [FIRST], gain=4.0)
        run_ingest(truth.session_dir)
        result = run_marker_analyze(truth.session_dir, marker=SPEC.name)
        analysis = analysis_of(truth.session_dir)
        assert analysis.occurrences
        assert all(item.clipped for item in analysis.occurrences)
        assert analysis.groups == []
        assert result.report.inconclusive is True

    def test_within_timecode_quantum_disagreement_stays_healthy(self, marked_session: Path) -> None:
        """M8's quantization floor still outranks the marker's sample precision."""
        run_marker_analyze(marked_session, marker=SPEC.name)
        comparisons = analysis_of(marked_session).timecode
        assert comparisons
        assert all(not item.beyond_quantum for item in comparisons)

    def test_a_marker_at_the_very_end_of_the_recording_is_still_found(self, tmp_path: Path) -> None:
        """The bench's closing block lands here, so it is not a hypothetical edge.

        An operator who stops recording promptly after the last play leaves the marker's
        100 ms of trailing silence running past the end of the session. The detector then
        reads a window that is partly beyond what exists, and the diagnostics pass reads a
        further whole marker length from the anchor. `TrackReader` answers those with
        silence rather than raising — a raw `PcmReader` would raise — and this is what
        proves the composed path uses the one that does.
        """
        truth = build_session(canonical_session(), tmp_path / "session")
        marker = marker_samples(SPEC).astype(np.float32) / 32768.0
        # Every placement, per track. A set rather than one value because the fixture may
        # chunk a track, and asserting against only the last chunk's position would let an
        # unexpected extra detection pass unnoticed.
        placed: dict[str, set[int]] = {}
        truncated = 0
        for chunk in truth.chunks:
            path = truth.session_dir / chunk.relative_path
            source = open_pcm(path)
            raw = path.read_bytes()
            end = source.data_offset + source.n_samples * 4
            audio = np.frombuffer(raw[source.data_offset : end], dtype="<f4").copy()
            # The last chirp ends exactly at the last recorded sample, so all three chirps
            # are present and only the trailing silence runs past the end. `chirp_intervals`
            # is absolute within the marker and already includes the lead silence.
            local = audio.size - SPEC.chirp_intervals()[-1][1]
            if local < 0:
                continue
            usable = min(marker.size, audio.size - local)
            truncated += marker.size - usable
            audio[local : local + usable] += marker[:usable] * 0.5
            path.write_bytes(raw[: source.data_offset] + audio.astype("<f4").tobytes() + raw[end:])
            placed.setdefault(chunk.track_id, set()).add(
                chunk.start_sample + local + SPEC.anchor_sample
            )
        assert placed, "the fixture placed no marker at all"
        assert truncated, "nothing ran past the end, so this is testing the ordinary case"
        run_ingest(truth.session_dir)

        result = run_marker_analyze(truth.session_dir, marker=SPEC.name)
        assert result.exit_code is ExitCode.OK
        found = analysis_of(truth.session_dir).occurrences
        assert found, "a marker at the end of the recording was not found at all"
        for item in found:
            assert item.anchor_sample in placed[item.track_id], item.track_id

    def test_an_unmatched_detection_is_kept_rather_than_dropped(self, tmp_path: Path) -> None:
        """An arrival no group claimed is evidence about the reference, not noise."""
        truth = build_session(canonical_session(), tmp_path / "session")
        inject(truth, [FIRST, SECOND])
        run_ingest(truth.session_dir)

        run_marker_analyze(truth.session_dir, marker=SPEC.name)
        analysis = analysis_of(truth.session_dir)
        claimed = {
            (member.track_id, member.anchor_sample)
            for group in analysis.groups
            for member in group.members
            if member.anchor_sample is not None
        }
        for item in analysis.unmatched:
            assert (item.track_id, item.anchor_sample) not in claimed


class TestAQuietRoomIsNotAFailure:
    """The distinction the report exists to keep."""

    def test_no_marker_completes_inconclusively_and_exits_zero(
        self, unmarked_session: Path
    ) -> None:
        result = run_marker_analyze(unmarked_session, marker=SPEC.name)
        assert result.exit_code is ExitCode.OK
        assert result.report.analysis_status.value == "complete"
        assert result.report.inconclusive is True
        assert any(w.code == "marker_not_found" for w in result.report.warnings)

    def test_and_still_writes_both_artifacts(self, unmarked_session: Path) -> None:
        run_marker_analyze(unmarked_session, marker=SPEC.name)
        assert (unmarked_session / ANALYSIS_RELATIVE_PATH).is_file()
        assert (unmarked_session / MARKER_REPORT_RELATIVE_PATH).is_file()

    def test_a_stale_timeline_is_a_failure_rather_than_an_empty_result(
        self, marked_session: Path
    ) -> None:
        """The contrast: a quiet room and a broken input must not share an exit code."""
        (marked_session / "work" / "timeline.json").unlink()
        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert result.exit_code is ExitCode.FATAL
        assert result.report.inconclusive is False
        assert [error.code for error in result.report.errors] == ["timeline_missing"]
        assert (marked_session / MARKER_REPORT_RELATIVE_PATH).is_file()


class TestStaleInputsAreRefusedComponentByComponent:
    """Every identity component, each with its own code (second plan review finding 3)."""

    def test_a_missing_manifest_names_itself(self, marked_session: Path) -> None:
        (marked_session / "work" / "manifest.json").unlink()
        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert [error.code for error in result.report.errors] == ["manifest_missing"]

    @pytest.mark.parametrize(
        ("field", "value", "code"),
        [
            ("config_hash", "0" * 64, "timeline_stale_config"),
            ("manifest_sha256", "0" * 64, "timeline_stale_manifest"),
        ],
    )
    def test_a_changed_top_level_identity_is_refused(
        self, marked_session: Path, field: str, value: str, code: str
    ) -> None:
        path = marked_session / "work" / "timeline.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document[field] = value
        path.write_text(json.dumps(document), encoding="utf-8")

        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert [error.code for error in result.report.errors] == [code]

    @pytest.mark.parametrize(
        ("field", "value", "code"),
        [
            ("timeline_semantics_version", 99, "timeline_stale_semantics"),
            ("inspection_semantics_version", 99, "timeline_stale_semantics"),
            ("numpy_version", "0.0.0", "timeline_stale_numerics"),
            ("scipy_version", "0.0.0", "timeline_stale_numerics"),
        ],
    )
    def test_a_changed_provenance_component_is_refused(
        self, marked_session: Path, field: str, value: object, code: str
    ) -> None:
        """A timeline can agree with its manifest and still be built by obsolete logic."""
        path = marked_session / "work" / "timeline.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["provenance"][field] = value
        path.write_text(json.dumps(document), encoding="utf-8")

        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert [error.code for error in result.report.errors] == [code]

    def test_an_unreadable_timeline_says_so(self, marked_session: Path) -> None:
        (marked_session / "work" / "timeline.json").write_text("{}", encoding="utf-8")
        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert [error.code for error in result.report.errors] == ["timeline_unreadable"]

    def test_a_source_changed_after_ingest_is_stale_even_when_stable_during_analysis(
        self, marked_session: Path
    ) -> None:
        """The entry/exit guard alone cannot catch a pre-existing raw mutation."""
        source = next((marked_session / "raw").rglob("*.wav"))
        payload = bytearray(source.read_bytes())
        payload[-1] ^= 1
        source.write_bytes(payload)
        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert [error.code for error in result.report.errors] == ["manifest_stale_source"]

    def test_validation_reads_and_writes_nothing(self, marked_session: Path) -> None:
        """It must not run inspection, which writes `work/ffprobe/` on a cold sidecar."""
        from dnd_audio.config import load_session_config
        from dnd_audio.raw_guard import raw_roots, snapshot

        config = load_session_config(marked_session / "session.yaml")
        before = {
            path: path.read_bytes() for path in sorted(marked_session.rglob("*")) if path.is_file()
        }
        read_session_artifacts(
            marked_session,
            config,
            current_sources=snapshot(marked_session, raw_roots(config)),
        )
        after = {
            path: path.read_bytes() for path in sorted(marked_session.rglob("*")) if path.is_file()
        }
        assert after == before


class TestNothingElseMoves:
    """The gate's "invalidates only its own artifacts" criterion."""

    def test_every_pre_existing_artifact_is_byte_identical_afterwards(
        self, marked_session: Path
    ) -> None:
        before = {
            path.relative_to(marked_session): path.read_bytes()
            for path in sorted(marked_session.rglob("*"))
            if path.is_file()
        }
        run_marker_analyze(marked_session, marker=SPEC.name)
        after = {
            path.relative_to(marked_session): path.read_bytes()
            for path in sorted(marked_session.rglob("*"))
            if path.is_file()
        }

        added = set(after) - set(before)
        assert added == {
            Path(ANALYSIS_RELATIVE_PATH),
            Path(MARKER_REPORT_RELATIVE_PATH),
        }
        for name, payload in before.items():
            assert after[name] == payload, f"{name} changed"

    def test_the_analysis_is_byte_stable_across_two_runs(self, marked_session: Path) -> None:
        run_marker_analyze(marked_session, marker=SPEC.name)
        first = (marked_session / ANALYSIS_RELATIVE_PATH).read_bytes()
        run_marker_analyze(marked_session, marker=SPEC.name)
        assert (marked_session / ANALYSIS_RELATIVE_PATH).read_bytes() == first

    def test_the_analysis_contains_no_floats(self, marked_session: Path) -> None:
        """INV-02 the way `timeline.json` holds it: a float is how a threshold sneaks back."""
        run_marker_analyze(marked_session, marker=SPEC.name)
        document = json.loads((marked_session / ANALYSIS_RELATIVE_PATH).read_text("utf-8"))

        def walk(value: object, path: str = "") -> None:
            if isinstance(value, float):
                raise AssertionError(f"float at {path}: {value!r}")
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        walk(document)

    def test_the_identity_names_every_component(self, marked_session: Path) -> None:
        """Asserted by name, not by hash: a key can move for the right reason and still
        be missing the component that matters later (M2's closeout)."""
        run_marker_analyze(marked_session, marker=SPEC.name)
        identity = analysis_of(marked_session).identity
        assert set(identity.model_dump()) == {
            "config_hash",
            "detector_semantics_version",
            "event_log_schema_version",
            "event_log_sha256",
            "manifest_schema_version",
            "manifest_sha256",
            "marker_analysis_semantics_version",
            "marker_name",
            "marker_wav_sha256",
            "marker_semantics_version",
            "numpy_version",
            "reference_track",
            "scipy_version",
            "searched_intervals",
            "thresholds",
            "timeline_config_hash",
            "timeline_schema_version",
        }

    @pytest.mark.parametrize(
        "attribute",
        [
            "marker_semantics_version",
            "detector_semantics_version",
            "marker_analysis_semantics_version",
        ],
    )
    def test_each_semantic_version_moves_the_identity_independently(
        self, marked_session: Path, attribute: str
    ) -> None:
        """Three versions, three independent effects — the point of having three."""
        run_marker_analyze(marked_session, marker=SPEC.name)
        identity = analysis_of(marked_session).identity
        bumped = identity.model_copy(update={attribute: getattr(identity, attribute) + 1})
        assert bumped.digest() != identity.digest()

    @pytest.mark.parametrize("attribute", ["manifest_schema_version", "timeline_schema_version"])
    def test_each_consumed_artifact_schema_moves_the_identity(
        self, marked_session: Path, attribute: str
    ) -> None:
        run_marker_analyze(marked_session, marker=SPEC.name)
        identity = analysis_of(marked_session).identity
        bumped = identity.model_copy(update={attribute: getattr(identity, attribute) + 1})
        assert bumped.digest() != identity.digest()


def test_a_locally_ambiguous_occurrence_is_not_usable_for_grouping() -> None:
    """The detector's ambiguity bit must prevent a precise-looking lag downstream."""
    occurrence = MarkerOccurrence(
        anchor_sample=100_000,
        score_permille=600,
        hits=(),
        gap_errors_samples=(),
        runner_up_permille=551,
        ambiguous=True,
    )
    assert _usable(occurrence) is False


def test_a_locally_ambiguous_occurrence_becomes_an_ambiguous_group_member() -> None:
    """No precise-looking lag may survive the detector-to-analyzer boundary."""
    clean = MarkerOccurrence(100_000, 600, (), ())
    ambiguous = MarkerOccurrence(100_048, 600, (), (), runner_up_permille=551, ambiguous=True)
    members = _associate(
        100_000,
        {"tx-a": [clean], "tx-b": [ambiguous]},
        "tx-a",
        settings=DetectorThresholds(),
        used={},
    )
    by_track = {member.track_id: member for member in members}
    assert by_track["tx-b"].outcome is DetectionOutcome.AMBIGUOUS
    assert by_track["tx-b"].relative_lag_samples is None


class TestTheEventLog:
    """Roles, geometry, and the claim geometry licenses."""

    def write_log(self, path: Path, *, geometry: str | None, session_id: str) -> Path:
        document = {
            "schema_version": 1,
            "session_id": session_id,
            "events": [
                {
                    "role": "start",
                    "marker_name": SPEC.name,
                    "start_ms": 3_000,
                    "end_ms": 5_000,
                    "playback_order": 0,
                    "geometry_id": geometry,
                },
                {
                    "role": "end",
                    "marker_name": SPEC.name,
                    "start_ms": 8_000,
                    "end_ms": 9_500,
                    "playback_order": 1,
                    "geometry_id": geometry,
                },
            ],
        }
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return path

    @pytest.fixture
    def two_marker_session(self, _ingested_templates: dict[str, Path], tmp_path: Path) -> Path:
        return _copy_of(_ingested_templates["two"], tmp_path / "session")

    def test_roles_come_from_the_log(self, two_marker_session: Path, tmp_path: Path) -> None:
        log = self.write_log(tmp_path / "events.yaml", geometry="table-1", session_id="2026-08-15")
        run_marker_analyze(two_marker_session, marker=SPEC.name, event_log=log)
        analysis = analysis_of(two_marker_session)
        roles = {group.role for group in analysis.groups if group.role}
        assert roles <= {"start", "end"}
        assert all(group.role_source == "event_log" for group in analysis.groups if group.role)

    def test_event_log_schema_and_digest_are_identity_components(
        self, two_marker_session: Path, tmp_path: Path
    ) -> None:
        log = self.write_log(tmp_path / "events.yaml", geometry="table-1", session_id="2026-08-15")
        run_marker_analyze(two_marker_session, marker=SPEC.name, event_log=log)
        identity = analysis_of(two_marker_session).identity
        assert identity.event_log_schema_version == 1
        assert identity.event_log_sha256 is not None
        assert (
            identity.model_copy(update={"event_log_schema_version": 2}).digest()
            != identity.digest()
        )
        assert (
            identity.model_copy(update={"event_log_sha256": "0" * 64}).digest() != identity.digest()
        )

    def test_without_a_geometry_id_a_change_is_differential_arrival_not_drift(
        self, two_marker_session: Path, tmp_path: Path
    ) -> None:
        """ADR-0040's central rule. A moved wearer is not a drifting clock."""
        log = self.write_log(tmp_path / "events.yaml", geometry=None, session_id="2026-08-15")
        run_marker_analyze(two_marker_session, marker=SPEC.name, event_log=log)
        analysis = analysis_of(two_marker_session)
        for comparison in analysis.arrival:
            assert comparison.outcome is not ArrivalOutcome.CLOCK_DRIFT_EVIDENCE

    @pytest.mark.parametrize(("change", "warns"), [(47, False), (48, True), (-48, True)])
    def test_only_a_material_fixed_geometry_change_warns(self, change: int, warns: bool) -> None:
        """ADR-0042's one-millisecond boundary, including its negative direction."""
        groups = [
            OccurrenceGroup(
                group_index=index,
                reference_anchor_sample=100_000 + index * 200_000,
                role=role,
                role_source="event_log",
                geometry_id="fixed-table",
                members=[
                    GroupMember(
                        track_id="tx-a",
                        outcome=DetectionOutcome.DETECTED,
                        anchor_sample=100_000 + index * 200_000,
                        relative_lag_samples=lag,
                        score_permille=600,
                    )
                ],
            )
            for index, (role, lag) in enumerate((("start", 100), ("end", 100 + change)))
        ]
        accumulator = _Accumulator()
        comparisons = _compare_arrival(
            groups, settings=DetectorThresholds(), accumulator=accumulator
        )
        assert comparisons[0].change_samples == change
        assert comparisons[0].outcome is ArrivalOutcome.CLOCK_DRIFT_EVIDENCE
        assert bool(accumulator.warnings) is warns

    def test_an_event_log_naming_another_marker_labels_nothing(
        self, two_marker_session: Path, tmp_path: Path
    ) -> None:
        """The bench plays three candidates; scoring a take against the wrong one must warn."""
        path = tmp_path / "events.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "session_id": "2026-08-15",
                    "events": [
                        {
                            "role": "start",
                            "marker_name": "cand-c",
                            "start_ms": 3_000,
                            "end_ms": 5_000,
                            "playback_order": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        result = run_marker_analyze(two_marker_session, marker=SPEC.name, event_log=path)
        assert any(w.code == "marker_event_log_names_no_occurrence" for w in result.report.warnings)

    def test_an_event_log_for_another_session_is_refused(
        self, two_marker_session: Path, tmp_path: Path
    ) -> None:
        log = self.write_log(tmp_path / "events.yaml", geometry="table-1", session_id="other")
        result = run_marker_analyze(two_marker_session, marker=SPEC.name, event_log=log)
        assert result.exit_code is ExitCode.FATAL
        assert [error.code for error in result.report.errors] == ["event_log_session_mismatch"]

    def test_overlapping_logged_roles_leave_the_occurrence_unassigned(
        self, marked_session: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "events.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "session_id": "2026-08-15",
                    "events": [
                        {
                            "role": role,
                            "marker_name": SPEC.name,
                            "start_ms": 3_000,
                            "end_ms": 5_000,
                            "playback_order": order,
                            "geometry_id": "fixed",
                        }
                        for order, role in enumerate(("start", "end"))
                    ],
                }
            ),
            encoding="utf-8",
        )
        result = run_marker_analyze(marked_session, marker=SPEC.name, event_log=path)
        analysis = analysis_of(marked_session)
        assert all(group.role is None for group in analysis.groups)
        assert analysis.arrival == []
        assert any(w.code == "marker_event_overlap" for w in result.report.warnings)

    def test_multiple_same_role_groups_do_not_silently_choose_a_pair(self) -> None:
        def group(index: int, role: str) -> OccurrenceGroup:
            return OccurrenceGroup(
                group_index=index,
                reference_anchor_sample=100_000 + 100_000 * index,
                role=role,
                role_source="event_log",
                geometry_id="fixed",
                members=[
                    GroupMember(
                        track_id="tx-a",
                        outcome=DetectionOutcome.DETECTED,
                        anchor_sample=100_000 + 100_000 * index,
                        relative_lag_samples=index,
                        score_permille=600,
                    )
                ],
            )

        accumulator = _Accumulator()
        comparisons = _compare_arrival(
            [group(0, "start"), group(1, "start"), group(2, "end")],
            settings=DetectorThresholds(),
            accumulator=accumulator,
        )
        assert comparisons == []
        assert [warning.code for warning in accumulator.warnings] == [
            "marker_arrival_pair_ambiguous"
        ]

    def test_two_starts_with_different_geometry_are_refused_at_load(self, tmp_path: Path) -> None:
        path = tmp_path / "events.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "session_id": "s",
                    "events": [
                        {
                            "role": "start",
                            "marker_name": SPEC.name,
                            "start_ms": 0,
                            "end_ms": 100,
                            "playback_order": 0,
                            "geometry_id": "a",
                        },
                        {
                            "role": "start",
                            "marker_name": SPEC.name,
                            "start_ms": 200,
                            "end_ms": 300,
                            "playback_order": 1,
                            "geometry_id": "b",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(EventLogError, match="geometry_id"):
            load_event_log(path)

    def test_intervals_are_quantized_through_the_one_quantizer(self, tmp_path: Path) -> None:
        """INV-04: milliseconds in, samples out, once — never two independent roundings."""
        path = tmp_path / "events.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "session_id": "s",
                    "events": [
                        {
                            "role": "start",
                            "marker_name": SPEC.name,
                            "start_ms": 1_001,
                            "end_ms": 2_003,
                            "playback_order": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        event = load_event_log(path).events[0]
        assert event.interval_samples(48_000) == (48_048, 96_144)

    def test_the_digest_ignores_formatting_and_notices_a_number(self, tmp_path: Path) -> None:
        """Reformatting the YAML must not invalidate an analysis; editing a time must."""
        base = {
            "schema_version": 1,
            "session_id": "s",
            "events": [
                {
                    "role": "start",
                    "marker_name": SPEC.name,
                    "start_ms": 100,
                    "end_ms": 200,
                    "playback_order": 0,
                }
            ],
        }
        one = tmp_path / "a.yaml"
        one.write_text(yaml.safe_dump(base, default_flow_style=False), encoding="utf-8")
        two = tmp_path / "b.yaml"
        two.write_text(yaml.safe_dump(base, default_flow_style=True), encoding="utf-8")
        assert load_event_log(one).digest() == load_event_log(two).digest()

        changed = tmp_path / "c.yaml"
        base["events"][0]["end_ms"] = 201  # type: ignore[index]
        changed.write_text(yaml.safe_dump(base), encoding="utf-8")
        assert load_event_log(changed).digest() != load_event_log(one).digest()


class TestTheCommandAndItsReport:
    """Through the CLI, and INV-13 at the boundary."""

    def test_the_command_runs_and_reports(self, marked_session: Path) -> None:
        result = runner.invoke(
            app, ["marker", "analyze", str(marked_session), "--marker", "cand-a"]
        )
        assert result.exit_code == 0, result.output
        assert "occurrence(s)" in result.output

    def test_the_report_validates_against_its_checked_in_schema(self, marked_session: Path) -> None:
        import jsonschema

        run_marker_analyze(marked_session, marker=SPEC.name)
        for relative, schema_name in (
            (ANALYSIS_RELATIVE_PATH, "marker-analysis.schema.json"),
            (MARKER_REPORT_RELATIVE_PATH, "marker-report.schema.json"),
        ):
            document = json.loads((marked_session / relative).read_text(encoding="utf-8"))
            schema = json.loads(Path("schemas", schema_name).read_text(encoding="utf-8"))
            jsonschema.validate(document, schema)

    def test_the_report_does_not_hash_itself(self, marked_session: Path) -> None:
        run_marker_analyze(marked_session, marker=SPEC.name)
        report = json.loads(
            (marked_session / MARKER_REPORT_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        assert [item["relative_path"] for item in report["deliverables"]] == [
            ANALYSIS_RELATIVE_PATH
        ]

    def test_the_report_model_represents_an_explained_skip(self) -> None:
        now = dt.datetime(2026, 8, 5, tzinfo=dt.UTC)
        report = MarkerReport(
            session_id="session-01",
            marker_name="v1",
            overall_status=OverallStatus.COMPLETE,
            analysis_status=AnalysisStatus.SKIPPED,
            skip_reason="caller explicitly declined marker analysis",
            started_at=now,
            finished_at=now,
        )
        assert report.exit_code() is ExitCode.OK

    def test_a_failed_stage_with_a_published_deliverable_is_partial_and_nonzero(self) -> None:
        now = dt.datetime(2026, 8, 5, tzinfo=dt.UTC)
        report = MarkerReport(
            session_id="session-01",
            marker_name="v1",
            overall_status=OverallStatus.PARTIAL,
            analysis_status=AnalysisStatus.FAILED,
            errors=[MarkerReportError(code="late_failure", message="failed after publication")],
            deliverables=[
                ReportDeliverable(
                    relative_path=ANALYSIS_RELATIVE_PATH,
                    sha256="0" * 64,
                    size_bytes=1,
                )
            ],
            started_at=now,
            finished_at=now,
        )
        assert report.exit_code() is ExitCode.PARTIAL

    def test_an_unknown_reference_track_fails_with_a_report(self, marked_session: Path) -> None:
        result = run_marker_analyze(marked_session, marker=SPEC.name, reference_track="tx-zzz")
        assert result.exit_code is ExitCode.FATAL
        assert [error.code for error in result.report.errors] == ["unknown_reference_track"]
        assert (marked_session / MARKER_REPORT_RELATIVE_PATH).is_file()

    def test_both_outputs_are_declared_before_anything_is_written(
        self, marked_session: Path
    ) -> None:
        """The signature `reject_outputs_inside_raw` takes; an undeclared output is the bug."""
        declared = set(marker_analyze_outputs(marked_session).values())
        assert declared == {
            marked_session / ANALYSIS_RELATIVE_PATH,
            marked_session / MARKER_REPORT_RELATIVE_PATH,
        }

    def test_a_report_path_inside_the_sources_writes_nothing(self, marked_session: Path) -> None:
        """INV-01 outranks INV-13: no report rather than a report inside `raw/`."""
        output = marked_session / "output"
        output.mkdir(exist_ok=True)
        if output.is_symlink():  # pragma: no cover - defensive
            output.unlink()
        recording = next((marked_session / "raw" / "tx-a").glob("*.wav"))
        before = recording.read_bytes()

        # `output -> raw/tx-a` is M1's exact defeat: lexically the report is under `output/`.
        import shutil

        shutil.rmtree(output)
        output.symlink_to(marked_session / "raw" / "tx-a")

        result = run_marker_analyze(marked_session, marker=SPEC.name)
        assert result.exit_code is ExitCode.FATAL
        assert result.report_written is False
        assert recording.read_bytes() == before


class TestSearchedIntervals:
    """Overlapping windows cannot detect one occurrence twice."""

    def test_two_overlapping_windows_are_merged(self, marked_session: Path, tmp_path: Path) -> None:
        path = tmp_path / "events.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "session_id": "2026-08-15",
                    "events": [
                        {
                            "role": "start",
                            "marker_name": SPEC.name,
                            "start_ms": 3_000,
                            "end_ms": 5_000,
                            "playback_order": 0,
                        },
                        {
                            "role": "diagnostic",
                            "marker_name": SPEC.name,
                            "start_ms": 3_500,
                            "end_ms": 6_000,
                            "playback_order": 1,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_marker_analyze(marked_session, marker=SPEC.name, event_log=path)
        analysis = analysis_of(marked_session)
        assert len(analysis.identity.searched_intervals) == 1

        anchors = [(item.track_id, item.anchor_sample) for item in analysis.occurrences]
        assert len(anchors) == len(set(anchors)), "an occurrence was reported twice"


class TestTheDefaultWindowsAreBounded:
    """`--start-window-s`/`--end-window-s` decide what is read when no event log does.

    The canonical fixture is 10.5 s long and the marker sits at :data:`FIRST`, 3.96 s in.
    A one-second window at each end therefore cannot reach it and a five-second opening
    can, which is what makes these assertions about the *search* rather than about the
    detector.
    """

    def test_a_window_too_narrow_to_reach_the_marker_finds_nothing(
        self, marked_session: Path
    ) -> None:
        result = run_marker_analyze(
            marked_session, marker=SPEC.name, start_window_seconds=1, end_window_seconds=1
        )
        assert result.exit_code is ExitCode.OK, "an unsearched marker is not a failure"
        assert analysis_of(marked_session).occurrences == []

    def test_a_wide_enough_opening_reaches_it(self, marked_session: Path) -> None:
        """And finds exactly what the unbounded default does — no more, no fewer."""
        run_marker_analyze(marked_session, marker=SPEC.name)
        whole = {
            (item.track_id, item.anchor_sample) for item in analysis_of(marked_session).occurrences
        }
        assert whole, "the fixture itself is broken"

        run_marker_analyze(
            marked_session, marker=SPEC.name, start_window_seconds=5, end_window_seconds=1
        )
        found = {
            (item.track_id, item.anchor_sample) for item in analysis_of(marked_session).occurrences
        }
        assert found == whole
        assert all(anchor == FIRST + SPEC.anchor_sample + DELAYS[track] for track, anchor in found)

    def test_the_two_windows_stay_disjoint_rather_than_scanning_everything(
        self, marked_session: Path
    ) -> None:
        run_marker_analyze(
            marked_session, marker=SPEC.name, start_window_seconds=1, end_window_seconds=1
        )
        intervals = analysis_of(marked_session).identity.searched_intervals
        assert len(intervals) == 2
        assert intervals[0][1] < intervals[1][0], "the ends met in the middle"

    def test_an_event_log_is_not_widened_by_them(
        self, marked_session: Path, tmp_path: Path
    ) -> None:
        """A log states its own intervals; the window flags must not reach past them."""
        path = tmp_path / "events.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "session_id": "2026-08-15",
                    "events": [
                        {
                            "role": "start",
                            "marker_name": SPEC.name,
                            "start_ms": 3_500,
                            "end_ms": 4_500,
                            "playback_order": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        narrow = run_marker_analyze(
            marked_session,
            marker=SPEC.name,
            event_log=path,
            start_window_seconds=1,
            end_window_seconds=1,
        )
        logged = analysis_of(marked_session).identity.searched_intervals
        assert narrow.exit_code is ExitCode.OK

        run_marker_analyze(
            marked_session,
            marker=SPEC.name,
            event_log=path,
            start_window_seconds=600,
            end_window_seconds=600,
        )
        assert analysis_of(marked_session).identity.searched_intervals == logged

    def test_a_zero_length_window_is_refused_rather_than_searched(
        self, marked_session: Path
    ) -> None:
        """Otherwise it would report `marker_not_found` without ever having looked."""
        result = run_marker_analyze(marked_session, marker=SPEC.name, start_window_seconds=0)
        assert result.exit_code is ExitCode.FATAL
        assert [error.code for error in result.report.errors] == ["invalid_search_window"]

    def test_the_command_rejects_a_zero_before_the_session_is_opened(
        self, marked_session: Path
    ) -> None:
        invoked = runner.invoke(
            app,
            [
                "marker",
                "analyze",
                str(marked_session),
                "--marker",
                SPEC.name,
                "--end-window-s",
                "0",
            ],
        )
        assert invoked.exit_code != 0
        assert not (marked_session / MARKER_REPORT_RELATIVE_PATH).exists()
