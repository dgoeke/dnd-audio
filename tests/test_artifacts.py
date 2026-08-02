"""Artifacts validate against the **checked-in** schemas, not against their own models.

The spec is specific about this: "Tests must validate real outputs against those
artifacts, not merely round-trip them through the same Pydantic class that created
them." A round-trip only proves pydantic is self-consistent; it would pass even if the
committed schema described something else entirely.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from dnd_audio.artifacts.manifest import Manifest, ManifestSource, ManifestTrack
from dnd_audio.artifacts.report import (
    IngestReport,
    OverallStatus,
    Provenance,
    StageName,
    StageReport,
    StageStatus,
    Telemetry,
)
from dnd_audio.artifacts.transcript import (
    SegmentProvenance,
    Transcript,
    TranscriptSegment,
    TranscriptSpeaker,
    TranscriptWord,
)
from dnd_audio.config import load_session_config
from dnd_audio.determinism import public_seconds
from dnd_audio.schema_export import SCHEMA_DIRNAME


@pytest.fixture
def schema_dir(repo_root: Path) -> Path:
    return repo_root / SCHEMA_DIRNAME


def _validator(schema_dir: Path, name: str) -> Draft202012Validator:
    schema = json.loads((schema_dir / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _sample_manifest() -> Manifest:
    return Manifest(
        session_id="2026-08-15",
        config_hash="a" * 64,
        tracks=[
            ManifestTrack(
                track_id="tx-b",
                speaker_id="bob",
                active=False,
            ),
            ManifestTrack(
                track_id="tx-a",
                speaker_id="alice",
                active=True,
                sources=[
                    ManifestSource(
                        relative_path="raw/tx-a/TX01_MIC002_20260815_190500_orig.wav",
                        sha256="b" * 64,
                        size_bytes=1024,
                    ),
                    ManifestSource(
                        relative_path="raw/tx-a/TX01_MIC002_20260815_190000_orig.wav",
                        sha256="c" * 64,
                        size_bytes=2048,
                    ),
                ],
            ),
        ],
        warnings=["tx-b has no usable original recording"],
    )


def _sample_transcript() -> Transcript:
    provenance = SegmentProvenance(
        asr_model="Qwen/Qwen3-ASR-1.7B",
        asr_model_revision="0123456789abcdef",
        alignment_status="aligned",
        source_candidate_id="candidate_000456",
    )
    return Transcript(
        session_id="2026-08-15",
        title="Session 01",
        duration_s=public_seconds(14432),
        speakers=[
            TranscriptSpeaker(speaker_id="bob", speaker_name="Bob", track_id="tx-b"),
            TranscriptSpeaker(speaker_id="alice", speaker_name="Alice", track_id="tx-a"),
        ],
        segments=[
            TranscriptSegment(
                segment_id="seg_000124",
                start_s=4824.91,
                end_s=4826.10,
                speaker_id="bob",
                speaker_name="Bob",
                track_id="tx-b",
                text="Absolutely not.",
                overlap=True,
                provenance=provenance,
            ),
            TranscriptSegment(
                segment_id="seg_000123",
                start_s=4821.44,
                end_s=4824.91,
                speaker_id="alice",
                speaker_name="Alice",
                track_id="tx-a",
                text="We should go back to Zephyrine.",
                overlap=True,
                words=[TranscriptWord(start_s=4821.44, end_s=4821.68, text="We")],
                provenance=provenance,
            ),
        ],
    )


def _sample_report(instant: dt.datetime) -> IngestReport:
    return IngestReport(
        session_id="2026-08-15",
        overall_status=OverallStatus.PARTIAL,
        stages=[
            StageReport(stage=StageName.INSPECT, status=StageStatus.COMPLETE),
            StageReport(
                stage=StageName.RENDER,
                status=StageStatus.SKIPPED,
                skip_reason="transcription failed, so there is nothing to render",
            ),
        ],
        provenance=Provenance(config_hash="d" * 64, tool_versions={"ffmpeg": "8.0"}),
        telemetry=Telemetry(started_at=instant, finished_at=instant),
    )


class TestAgainstCheckedInSchemas:
    def test_manifest(self, schema_dir: Path) -> None:
        payload = _sample_manifest().model_dump(mode="json")
        _validator(schema_dir, "manifest.schema.json").validate(payload)

    def test_transcript(self, schema_dir: Path) -> None:
        payload = _sample_transcript().model_dump(mode="json")
        _validator(schema_dir, "transcript.schema.json").validate(payload)

    def test_report(self, schema_dir: Path, instant: dt.datetime) -> None:
        payload = _sample_report(instant).model_dump(mode="json")
        _validator(schema_dir, "ingest-report.schema.json").validate(payload)

    def test_session_config(self, schema_dir: Path, valid_session_yaml: Path) -> None:
        payload = load_session_config(valid_session_yaml).model_dump(mode="json")
        _validator(schema_dir, "session-config.schema.json").validate(payload)

    def test_the_schemas_actually_reject_something(self, schema_dir: Path) -> None:
        """A schema that accepts everything would make the tests above meaningless."""
        from jsonschema.exceptions import ValidationError

        validator = _validator(schema_dir, "manifest.schema.json")
        broken: dict[str, Any] = _sample_manifest().model_dump(mode="json")
        broken["tracks"][0]["sha256_of_nothing"] = "surprise"
        with pytest.raises(ValidationError):
            validator.validate(broken)


class TestOrdering:
    def test_manifest_tracks_and_sources_sort(self) -> None:
        """INV-02: directory iteration order must not reach an artifact."""
        manifest = _sample_manifest()
        assert [track.track_id for track in manifest.tracks] == ["tx-a", "tx-b"]
        paths = [source.relative_path for source in manifest.tracks[0].sources]
        assert paths == sorted(paths)

    def test_transcript_sorts_by_start_then_id(self) -> None:
        transcript = _sample_transcript()
        assert [segment.segment_id for segment in transcript.segments] == [
            "seg_000123",
            "seg_000124",
        ]
        assert [speaker.speaker_id for speaker in transcript.speakers] == ["alice", "bob"]

    def test_overlapping_turns_stay_separate_entries(self) -> None:
        transcript = _sample_transcript()
        assert all(segment.overlap for segment in transcript.segments)
        assert len(transcript.segments) == 2


class TestValidation:
    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs"):
            TranscriptSpeaker(speaker_id="a", speaker_name="A", track_id="tx-a", confidence=0.9)  # type: ignore[call-arg]

    def test_backwards_segment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="before it starts"):
            TranscriptSegment(
                segment_id="seg_000001",
                start_s=10.0,
                end_s=9.0,
                speaker_id="a",
                speaker_name="A",
                track_id="tx-a",
                text="",
                provenance=SegmentProvenance(
                    asr_model="m", alignment_status="aligned", source_candidate_id="c"
                ),
            )

    def test_segment_id_format_is_enforced(self) -> None:
        """IDs are derived, not free text; a stray format would break stable ordering."""
        with pytest.raises(ValueError, match="should match pattern"):
            TranscriptSegment(
                segment_id="123",
                start_s=0.0,
                end_s=1.0,
                speaker_id="a",
                speaker_name="A",
                track_id="tx-a",
                text="",
                provenance=SegmentProvenance(
                    asr_model="m", alignment_status="aligned", source_candidate_id="c"
                ),
            )

    def test_manifest_hash_format_is_enforced(self) -> None:
        with pytest.raises(ValueError, match="should match pattern"):
            ManifestSource(relative_path="raw/a.wav", sha256="short", size_bytes=1)

    def test_no_confidence_field_exists_on_a_segment(self) -> None:
        """The spec forbids manufacturing one the model does not provide."""
        assert "confidence" not in TranscriptSegment.model_fields
        assert "confidence" not in TranscriptWord.model_fields
