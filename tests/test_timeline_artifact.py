"""`timeline.json`: valid against its checked-in schema, and free of floats.

The float check is the one that matters. INV-04 forbids a fractional rate becoming a binary
float anywhere in timestamp arithmetic, and the way that rule dies is not a deliberate
decision — it is one "just for display" seconds field, added because a reader wanted it,
which some later milestone then computes from. M1 walks its manifest for the same reason;
this walks the timeline.

The rest of the file is about the artifact's validators refusing shapes a consumer could not
interpret. A segment map with a hole in it reads as either silence or a forgetful builder,
and M3 and M5 both index into these — so the ambiguous shape is rejected at the boundary
rather than documented as "should not happen".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from dnd_audio.artifacts.timeline import (
    DerivativeRecord,
    SessionZero,
    Timeline,
    TimelineSegment,
    TimelineTrack,
)
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.schema_export import SCHEMA_DIRNAME
from dnd_audio.timeline.runner import run_ingest


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    """The committed artifact, not the model that produced it."""
    root = Path(__file__).resolve().parent.parent
    document = json.loads((root / SCHEMA_DIRNAME / "timeline.schema.json").read_text())
    assert isinstance(document, dict)
    return document


def audio_segment(start: int = 0, n: int = 100, **overrides: Any) -> TimelineSegment:
    fields: dict[str, Any] = {
        "kind": "audio",
        "session_start_sample": start,
        "n_samples": n,
        "source_relative_path": "raw/tx-a/one.wav",
        "source_sha256": "a" * 64,
        "source_start_sample": 0,
        "evidence_start_sample": start,
    }
    fields.update(overrides)
    return TimelineSegment(**fields)


def a_track(segments: list[TimelineSegment], start: int, end: int) -> TimelineTrack:
    return TimelineTrack(
        track_id="tx-a",
        speaker_id="alice",
        speaker_name="Alice",
        start_sample=start,
        end_sample=end,
        segments=segments,
    )


class TestNoFloatsAnywhere:
    def test_a_real_timeline_contains_no_float(self, canonical_fixture: FixtureTruth) -> None:
        """Walked as serialized JSON, so nothing depends on how the models are typed."""
        result = run_ingest(canonical_fixture.session_dir)
        document = json.loads(result.timeline_path.read_text(encoding="utf-8"))
        assert list(_floats(document)) == []

    def test_a_timeline_with_a_derivative_still_has_none(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The derivative records are where a "duration in seconds" would be tempting."""
        result = run_ingest(canonical_fixture.session_dir, materialize_48k=True)
        document = json.loads(result.timeline_path.read_text(encoding="utf-8"))
        assert list(_floats(document)) == []
        assert any(track["derivatives"] for track in document["tracks"])

    def test_the_check_can_fail(self, canonical_fixture: FixtureTruth) -> None:
        """A walk that found nothing because it looks nowhere would pass silently."""
        result = run_ingest(canonical_fixture.session_dir)
        document = json.loads(result.timeline_path.read_text(encoding="utf-8"))
        document["tracks"][0]["duration_seconds"] = 10.5
        assert list(_floats(document)) == ["$.tracks[0].duration_seconds"]

    def test_the_frame_rate_is_two_integers(self, canonical_fixture: FixtureTruth) -> None:
        """29.97 is 30000/1001 here, as it is everywhere else (INV-04)."""
        result = run_ingest(canonical_fixture.session_dir)
        document = json.loads(result.timeline_path.read_text(encoding="utf-8"))
        assert document["frame_rate"] == {"numerator": 30, "denominator": 1}
        assert isinstance(document["frame_rate"]["numerator"], int)


class TestSchema:
    def test_a_real_timeline_validates(
        self, canonical_fixture: FixtureTruth, schema: dict[str, Any]
    ) -> None:
        """Against the committed file, not against the model that produced it.

        The spec asks for exactly this: tests must validate real outputs against the
        checked-in artifacts rather than round-tripping through the same class.
        """
        result = run_ingest(canonical_fixture.session_dir)
        document = json.loads(result.timeline_path.read_text(encoding="utf-8"))
        jsonschema.validate(document, schema)

    def test_the_schema_rejects_a_malformed_document(self, schema: dict[str, Any]) -> None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"schema_version": 1}, schema)


class TestTheMapTilesItsExtent:
    """A hole has two readings and a consumer cannot tell them apart."""

    def test_a_contiguous_map_is_accepted(self) -> None:
        a_track([audio_segment(0, 100), audio_segment(100, 50)], 0, 150)

    def test_a_gap_must_be_an_explicit_silence_segment(self) -> None:
        with pytest.raises(ValueError, match="hole"):
            a_track([audio_segment(0, 100), audio_segment(200, 50)], 0, 250)

    def test_a_silence_segment_closes_the_same_gap(self) -> None:
        a_track(
            [
                audio_segment(0, 100),
                TimelineSegment(kind="silence", session_start_sample=100, n_samples=100),
                audio_segment(200, 50),
            ],
            0,
            250,
        )

    def test_overlapping_segments_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            a_track([audio_segment(0, 100), audio_segment(50, 50)], 0, 100)

    def test_the_extent_must_match_the_segments(self) -> None:
        with pytest.raises(ValueError, match="ends at"):
            a_track([audio_segment(0, 100)], 0, 200)

    def test_a_track_may_not_begin_or_end_with_silence(self) -> None:
        """Leading and trailing silence is the reader's job; putting it in the map would
        make a track's extent a matter of opinion."""
        with pytest.raises(ValueError, match="begins or ends with silence"):
            a_track(
                [
                    TimelineSegment(kind="silence", session_start_sample=0, n_samples=100),
                    audio_segment(100, 50),
                ],
                0,
                150,
            )

    def test_an_empty_track_must_have_an_empty_extent(self) -> None:
        a_track([], 0, 0)
        with pytest.raises(ValueError, match="no segments"):
            a_track([], 0, 100)


class TestSegmentsCarryWhatTheyClaim:
    def test_an_audio_segment_must_name_its_source(self) -> None:
        with pytest.raises(ValueError, match="must name its source"):
            TimelineSegment(kind="audio", session_start_sample=0, n_samples=10)

    def test_a_silence_segment_must_not(self) -> None:
        with pytest.raises(ValueError, match="not a file"):
            TimelineSegment(
                kind="silence",
                session_start_sample=0,
                n_samples=10,
                source_relative_path="raw/tx-a/one.wav",
            )

    def test_the_shift_must_reconcile_the_two_starts(self) -> None:
        """So a shift can never be absorbed into the placed position and lost."""
        with pytest.raises(ValueError, match="do not sum to it"):
            audio_segment(100, 10, evidence_start_sample=50, shift_samples=10)

    def test_a_nudged_segment_records_both_positions(self) -> None:
        segment = audio_segment(100, 10, evidence_start_sample=90, shift_samples=10)
        assert segment.evidence_start_sample == 90
        assert segment.session_start_sample == 100


class TestDerivativeRecords:
    def test_the_length_must_follow_the_ceil_rule(self) -> None:
        with pytest.raises(ValueError, match="under the ceil rule"):
            _derivative(input_samples=100, output_samples=33)

    def test_the_ceil_length_is_accepted(self) -> None:
        assert _derivative(input_samples=100, output_samples=34).output_samples == 34

    def test_the_group_delay_must_divide(self) -> None:
        with pytest.raises(ValueError, match="whole output samples"):
            _derivative(input_samples=99, output_samples=33, delay_in=129, delay_out=42)


class TestDurationIsTheLatestTrackEnd:
    def test_it_must_match_the_tracks(self) -> None:
        with pytest.raises(ValueError, match="latest track ends at"):
            _timeline(duration=999, tracks=[a_track([audio_segment(0, 100)], 0, 100)])

    def test_the_shortest_track_does_not_decide_it(self) -> None:
        """Named explicitly because the shortest track is the tempting wrong answer."""
        short = a_track([audio_segment(0, 100)], 0, 100)
        long = TimelineTrack(
            track_id="tx-b",
            speaker_id="bob",
            speaker_name="Bob",
            start_sample=0,
            end_sample=500,
            segments=[audio_segment(0, 500)],
        )
        timeline = _timeline(duration=500, tracks=[short, long])
        assert timeline.duration_samples == 500


def _derivative(
    *, input_samples: int, output_samples: int, delay_in: int = 129, delay_out: int = 43
) -> DerivativeRecord:
    return DerivativeRecord(
        sample_rate=16000,
        relative_path="work/cache/audio/16000/x.wav",
        cache_key="a" * 64,
        size_bytes=0,
        input_samples=input_samples,
        output_samples=output_samples,
        decimation=3,
        filter_name="fir",
        filter_identity="b" * 64,
        group_delay_input_samples=delay_in,
        group_delay_output_samples=delay_out,
    )


def _timeline(*, duration: int, tracks: list[TimelineTrack]) -> Timeline:
    from dnd_audio.artifacts.manifest import RationalRate
    from dnd_audio.artifacts.timeline import TimelineProvenance

    return Timeline(
        session_id="test",
        config_hash="a" * 64,
        manifest_sha256="b" * 64,
        provenance=TimelineProvenance(
            timeline_semantics_version=1,
            inspection_semantics_version=1,
            numpy_version="2",
            scipy_version="1",
        ),
        sample_rate=48000,
        duration_samples=duration,
        session_zero=SessionZero(source="earliest_source", domain="real_time", detail="test"),
        frame_rate_label="30F",
        frame_rate=RationalRate(numerator=30, denominator=1),
        tracks=tracks,
    )


def _floats(node: Any, path: str = "$") -> Iterator[str]:
    """Every float in a serialized document, by path."""
    if isinstance(node, bool):
        return
    if isinstance(node, float):
        yield path
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _floats(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _floats(value, f"{path}[{index}]")
