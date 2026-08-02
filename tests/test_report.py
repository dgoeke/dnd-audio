"""INV-13, INV-03, INV-02: the report tells the truth about a partial run.

Four properties, and every one of them is something a later milestone could quietly
break: every stage has a status, provenance carries no wall-clock, the output does not
depend on which branch finished first, and the report is never half-written.
"""

from __future__ import annotations

import datetime as dt
import json
import typing
from pathlib import Path

import pytest

from dnd_audio.artifacts.report import (
    REPORT_FILENAME,
    STAGE_ORDER,
    Decision,
    IngestReport,
    OverallStatus,
    Provenance,
    ReportBuilder,
    ReportWarning,
    StageName,
    StageReport,
    StageStatus,
    StructuredError,
    Telemetry,
    roll_up,
)
from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.errors import ExitCode

_CONFIG_HASH = "c" * 64


def _builder(instant: dt.datetime) -> ReportBuilder:
    return ReportBuilder("2026-08-15", config_hash=_CONFIG_HASH, started_at=instant)


def _error(code: str = "asr_failed") -> StructuredError:
    return StructuredError(code=code, message="the transcriber raised")


def _fill_remaining(builder: ReportBuilder, *, recorded: set[StageName]) -> None:
    """Skip whatever the test did not set, so the report is complete (INV-13)."""
    for stage in STAGE_ORDER:
        if stage not in recorded:
            builder.stage_skipped(stage, "not exercised by this test")


def _stages(*reports: StageReport) -> list[StageReport]:
    """A full stage list, with anything unspecified marked skipped."""
    given = {report.stage: report for report in reports}
    return [
        given.get(
            stage,
            StageReport(
                stage=stage, status=StageStatus.SKIPPED, skip_reason="not exercised by this test"
            ),
        )
        for stage in STAGE_ORDER
    ]


class TestStageStatuses:
    def test_all_three_statuses_survive_serialization(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_complete(StageName.INSPECT)
        builder.stage_failed(StageName.TRANSCRIBE, [_error()])
        builder.stage_skipped(StageName.RENDER, "no transcript records to render")
        _fill_remaining(
            builder, recorded={StageName.INSPECT, StageName.TRANSCRIBE, StageName.RENDER}
        )

        report = builder.build(instant)
        statuses = {stage.stage: stage.status for stage in report.stages}

        assert statuses[StageName.INSPECT] is StageStatus.COMPLETE
        assert statuses[StageName.TRANSCRIBE] is StageStatus.FAILED
        assert statuses[StageName.RENDER] is StageStatus.SKIPPED

    def test_a_skipped_stage_must_say_why(self) -> None:
        """ "Skipped" with no reason cannot be told apart from "nobody ran it"."""
        with pytest.raises(ValueError, match="skipped without a reason"):
            StageReport(stage=StageName.MIX, status=StageStatus.SKIPPED)

    def test_a_failed_stage_must_carry_a_structured_error(self) -> None:
        with pytest.raises(ValueError, match="without a structured error"):
            StageReport(stage=StageName.MIX, status=StageStatus.FAILED)

    def test_structured_errors_round_trip(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_failed(
            StageName.MIX,
            [
                StructuredError(
                    code="loudness_out_of_tolerance",
                    message="decoded MP3 measured -14.2 LUFS",
                    path="output/session.mp3",
                    details={"measured_lufs": "-14.2", "target_lufs": "-16.0"},
                )
            ],
        )
        _fill_remaining(builder, recorded={StageName.MIX})
        payload = json.loads(canonical_json(builder.build(instant).model_dump(mode="json")))
        error = next(stage["errors"][0] for stage in payload["stages"] if stage["stage"] == "mix")
        assert error["code"] == "loudness_out_of_tolerance"
        assert error["details"]["measured_lufs"] == "-14.2"

    def test_a_stage_cannot_be_recorded_twice(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_complete(StageName.MIX)
        with pytest.raises(ValueError, match="already has an outcome"):
            builder.stage_failed(StageName.MIX, [_error()])


class TestEveryStageIsAccountedFor:
    """INV-13: "every stage reports complete, failed, or skipped" — every stage.

    An absent stage reads as an oversight, and a report with no stages at all used to
    roll up to `complete` and exit zero, which is the exact outcome the invariant
    exists to prevent.
    """

    def test_building_with_a_stage_unaccounted_for_is_an_error(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_complete(StageName.INSPECT)
        with pytest.raises(ValueError, match="no outcome recorded for"):
            builder.build(instant)

    def test_the_error_names_the_missing_stages(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_complete(StageName.INSPECT)
        with pytest.raises(ValueError, match="no outcome recorded for") as caught:
            builder.build(instant)
        for stage in STAGE_ORDER:
            if stage is not StageName.INSPECT:
                assert stage.value in str(caught.value)

    def test_an_empty_report_cannot_be_constructed(self, instant: dt.datetime) -> None:
        with pytest.raises(ValueError, match="accounts for no outcome"):
            IngestReport(
                session_id="s",
                overall_status=OverallStatus.COMPLETE,
                stages=[],
                provenance=Provenance(config_hash=_CONFIG_HASH),
                telemetry=Telemetry(started_at=instant, finished_at=instant),
            )

    def test_a_report_cannot_claim_a_status_its_stages_contradict(
        self, instant: dt.datetime
    ) -> None:
        """Otherwise a caller could hand-build a `complete` report over a failed stage."""
        with pytest.raises(ValueError, match="roll up to"):
            IngestReport(
                session_id="s",
                overall_status=OverallStatus.COMPLETE,
                stages=_stages(
                    StageReport(stage=StageName.MIX, status=StageStatus.FAILED, errors=[_error()])
                ),
                provenance=Provenance(config_hash=_CONFIG_HASH),
                telemetry=Telemetry(started_at=instant, finished_at=instant),
            )


class TestDecisions:
    """The spec's decision subsection: deterministic even though the report is not."""

    def test_decisions_are_ordered_independently_of_when_they_were_recorded(
        self, instant: dt.datetime
    ) -> None:
        made = [
            Decision(code="orig_selected", subject="tx-a", detail="ignored the edit variant"),
            Decision(code="gap_preserved", subject="tx-c", detail="41.2 s of silence kept"),
            Decision(code="orig_selected", subject="tx-b", detail="ignored the edit variant"),
        ]

        def build(order: list[Decision]) -> str:
            builder = _builder(instant)
            for decision in order:
                builder.record_decision(decision)
            _fill_remaining(builder, recorded=set())
            return canonical_json(
                [item.model_dump(mode="json") for item in builder.build(instant).decisions]
            )

        assert build(made) == build(list(reversed(made)))

    def test_decisions_sort_by_code_then_subject(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        for subject in ("tx-c", "tx-a", "tx-b"):
            builder.record_decision(
                Decision(code="orig_selected", subject=subject, detail="ignored the edit")
            )
        _fill_remaining(builder, recorded=set())
        assert [d.subject for d in builder.build(instant).decisions] == ["tx-a", "tx-b", "tx-c"]

    def test_decisions_carry_no_time_typed_field(self) -> None:
        """INV-03 applies here as much as to provenance."""
        forbidden = {dt.datetime, dt.date, dt.time, dt.timedelta}
        annotations = {field.annotation for field in Decision.model_fields.values()}
        assert not forbidden & annotations


class TestRollup:
    def test_all_complete_is_complete(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_complete(StageName.INSPECT)
        builder.stage_complete(StageName.MIX)
        _fill_remaining(builder, recorded={StageName.INSPECT, StageName.MIX})
        assert builder.build(instant).overall_status is OverallStatus.COMPLETE

    def test_a_skipped_stage_is_not_a_failure(self, instant: dt.datetime) -> None:
        """Running only `mix` skips `transcribe` deliberately."""
        builder = _builder(instant)
        builder.stage_complete(StageName.MIX)
        builder.stage_skipped(StageName.TRANSCRIBE, "not requested")
        _fill_remaining(builder, recorded={StageName.MIX, StageName.TRANSCRIBE})
        assert builder.build(instant).overall_status is OverallStatus.COMPLETE

    def test_failed_transcript_with_a_good_mix_is_partial(self, instant: dt.datetime) -> None:
        """The spec's named case, and the reason INV-09 exists."""
        builder = _builder(instant)
        builder.stage_complete(StageName.MIX)
        builder.stage_failed(StageName.TRANSCRIBE, [_error()])
        _fill_remaining(builder, recorded={StageName.MIX, StageName.TRANSCRIBE})
        assert builder.build(instant).overall_status is OverallStatus.PARTIAL

    def test_nothing_survived_is_failed(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_failed(StageName.INSPECT, [_error("no_usable_original")])
        _fill_remaining(builder, recorded={StageName.INSPECT})
        assert builder.build(instant).overall_status is OverallStatus.FAILED

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (OverallStatus.COMPLETE, ExitCode.OK),
            (OverallStatus.PARTIAL, ExitCode.PARTIAL),
            (OverallStatus.FAILED, ExitCode.FATAL),
        ],
    )
    def test_partial_success_never_exits_zero(
        self, instant: dt.datetime, status: OverallStatus, expected: ExitCode
    ) -> None:
        """INV-13, stated as an exit code so automation cannot misread it."""
        outcomes = {
            OverallStatus.COMPLETE: [StageReport(stage=StageName.MIX, status=StageStatus.COMPLETE)],
            OverallStatus.PARTIAL: [
                StageReport(stage=StageName.MIX, status=StageStatus.COMPLETE),
                StageReport(
                    stage=StageName.TRANSCRIBE, status=StageStatus.FAILED, errors=[_error()]
                ),
            ],
            OverallStatus.FAILED: [
                StageReport(stage=StageName.INSPECT, status=StageStatus.FAILED, errors=[_error()])
            ],
        }[status]
        report = IngestReport(
            session_id="s",
            overall_status=status,
            stages=_stages(*outcomes),
            provenance=Provenance(config_hash=_CONFIG_HASH),
            telemetry=Telemetry(started_at=instant, finished_at=instant),
        )
        assert report.exit_code() == expected
        assert (report.exit_code() == 0) is (status is OverallStatus.COMPLETE)

    def test_rolling_up_nothing_is_an_error(self) -> None:
        """A run that recorded no stage is not a complete run (INV-13)."""
        with pytest.raises(ValueError, match="empty stage list"):
            roll_up([])


class TestDeterminism:
    def test_stage_order_is_independent_of_completion_order(self, instant: dt.datetime) -> None:
        """INV-02: the mix and transcript branches race by design (INV-09)."""
        mix_first = _builder(instant)
        mix_first.stage_complete(StageName.MIX)
        mix_first.stage_failed(StageName.TRANSCRIBE, [_error()])
        mix_first.stage_complete(StageName.ACTIVITY)
        _fill_remaining(
            mix_first, recorded={StageName.MIX, StageName.TRANSCRIBE, StageName.ACTIVITY}
        )

        transcript_first = _builder(instant)
        transcript_first.stage_complete(StageName.ACTIVITY)
        transcript_first.stage_failed(StageName.TRANSCRIBE, [_error()])
        transcript_first.stage_complete(StageName.MIX)
        _fill_remaining(
            transcript_first,
            recorded={StageName.MIX, StageName.TRANSCRIBE, StageName.ACTIVITY},
        )

        assert mix_first.build(instant).stages == transcript_first.build(instant).stages

    def test_provenance_is_identical_across_completion_orders(self, instant: dt.datetime) -> None:
        def build(reverse: bool) -> str:
            builder = _builder(instant)
            entries = [("ffmpeg", "8.0"), ("ffprobe", "8.0"), ("sox", "14.4.2")]
            for name, version in reversed(entries) if reverse else entries:
                builder.record_tool_version(name, version)
            _fill_remaining(builder, recorded=set())
            return canonical_json(builder.build(instant).provenance.model_dump(mode="json"))

        assert build(reverse=False) == build(reverse=True)

    def test_stages_serialize_in_dag_order(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        for stage in reversed(STAGE_ORDER):
            builder.stage_complete(stage)
        assert [stage.stage for stage in builder.build(instant).stages] == list(STAGE_ORDER)

    def test_deliverables_sort_by_path(self, tmp_path: Path, instant: dt.datetime) -> None:
        for name in ("transcript.md", "session.mp3", "transcript.json"):
            (tmp_path / name).write_text(name, encoding="utf-8")

        builder = _builder(instant)
        for name in ("transcript.md", "session.mp3", "transcript.json"):
            builder.add_deliverable(tmp_path / name, relative_to=tmp_path)
        _fill_remaining(builder, recorded=set())

        paths = [d.relative_path for d in builder.build(instant).provenance.deliverables]
        assert paths == sorted(paths)


class TestProvenanceTelemetrySplit:
    def test_provenance_has_no_time_typed_field(self) -> None:
        """INV-03, as a structural fact rather than a review habit.

        Walks the annotations so a field added in M1..M6 cannot slip a timestamp in.
        """
        forbidden = {dt.datetime, dt.date, dt.time, dt.timedelta}

        def annotations_of(model: type) -> list[object]:
            found: list[object] = []
            for field in model.model_fields.values():  # type: ignore[attr-defined]
                annotation = field.annotation
                found.append(annotation)
                found.extend(typing.get_args(annotation))
                if hasattr(annotation, "model_fields"):
                    found.extend(annotations_of(annotation))
            return found

        assert not forbidden & set(annotations_of(Provenance))

    def test_telemetry_is_where_the_clock_lives(self, instant: dt.datetime) -> None:
        assert {"started_at", "finished_at"} <= set(Telemetry.model_fields)

    def test_naive_timestamps_are_rejected(self) -> None:
        naive = dt.datetime(2026, 8, 15, 19, 0, 0)  # noqa: DTZ001 - the point of the test
        with pytest.raises(ValueError, match="timezone-aware"):
            Telemetry(started_at=naive, finished_at=naive)

    def test_finishing_before_starting_is_rejected(self, instant: dt.datetime) -> None:
        with pytest.raises(ValueError, match="precedes"):
            Telemetry(started_at=instant, finished_at=instant - dt.timedelta(seconds=1))

    def test_cache_counters_are_telemetry_not_provenance(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.record_cache(hits=3, misses=1)
        _fill_remaining(builder, recorded=set())
        report = builder.build(instant)
        assert report.telemetry.cache_hits == 3
        assert "cache_hits" not in report.provenance.model_dump()


class TestDeliverables:
    def test_hashes_what_was_produced(self, tmp_path: Path, instant: dt.datetime) -> None:
        deliverable = tmp_path / "output" / "transcript.json"
        deliverable.parent.mkdir()
        deliverable.write_bytes(b"{}")

        builder = _builder(instant)
        recorded = builder.add_deliverable(deliverable, relative_to=tmp_path)

        assert recorded.relative_path == "output/transcript.json"
        assert recorded.sha256 == sha256_bytes(b"{}")
        assert recorded.size_bytes == 2

    def test_the_report_is_never_one_of_its_own_deliverables(
        self, tmp_path: Path, instant: dt.datetime
    ) -> None:
        """ADR-0003: writing the hash would change the bytes the hash describes."""
        report_path = tmp_path / REPORT_FILENAME
        report_path.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="ADR-0003"):
            _builder(instant).add_deliverable(report_path, relative_to=tmp_path)


class TestWriting:
    def test_written_report_validates_and_is_readable(
        self, tmp_path: Path, instant: dt.datetime
    ) -> None:
        builder = _builder(instant)
        builder.stage_complete(StageName.MIX)
        builder.stage_failed(StageName.TRANSCRIBE, [_error()])
        _fill_remaining(builder, recorded={StageName.MIX, StageName.TRANSCRIBE})

        target = tmp_path / REPORT_FILENAME
        written = builder.write(target, instant)

        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["overall_status"] == "partial"
        assert payload["schema_version"] == 1
        assert IngestReport.model_validate(payload) == written

    def test_written_even_when_everything_failed(
        self, tmp_path: Path, instant: dt.datetime
    ) -> None:
        """INV-13: the report is the thing that survives a bad run."""
        builder = _builder(instant)
        builder.stage_failed(StageName.INSPECT, [_error("no_usable_original")])
        _fill_remaining(builder, recorded={StageName.INSPECT})

        target = tmp_path / REPORT_FILENAME
        report = builder.write(target, instant)

        assert target.is_file()
        assert report.exit_code() == ExitCode.FATAL

    def test_a_failed_write_leaves_the_previous_report_intact(
        self, tmp_path: Path, instant: dt.datetime
    ) -> None:
        target = tmp_path / REPORT_FILENAME
        first = _builder(instant)
        first.stage_complete(StageName.INSPECT)
        _fill_remaining(first, recorded={StageName.INSPECT})
        first.write(target, instant)
        original = target.read_bytes()

        class Unwritable(ReportBuilder):
            def build(self, finished_at: dt.datetime) -> IngestReport:
                message = "stage accounting blew up while finalizing"
                raise RuntimeError(message)

        with pytest.raises(RuntimeError):
            Unwritable("2026-08-15", config_hash=_CONFIG_HASH, started_at=instant).write(
                target, instant
            )

        assert target.read_bytes() == original
        assert [p.name for p in tmp_path.iterdir()] == [REPORT_FILENAME]

    def test_warnings_are_carried_without_failing_the_stage(self, instant: dt.datetime) -> None:
        builder = _builder(instant)
        builder.stage_complete(
            StageName.RECONSTRUCT,
            warnings=[ReportWarning(code="real_gap", message="tx-c was off for 41 s")],
        )
        _fill_remaining(builder, recorded={StageName.RECONSTRUCT})
        report = builder.build(instant)
        assert report.overall_status is OverallStatus.COMPLETE
        reconstruct = next(s for s in report.stages if s.stage is StageName.RECONSTRUCT)
        assert reconstruct.warnings[0].code == "real_gap"
