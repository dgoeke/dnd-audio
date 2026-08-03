"""`transcript.json` and `transcript.md`, and the properties both have to hold.

The gate criteria here: the JSON validates against the **checked-in** schema rather than
round-tripping through the model that produced it, public times are millisecond-precise with
stable tie-breakers, ids derive from sorted source identity and time, `overlap` means what the
spec says it means, and the Markdown is the spec's format with user and model text escaped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from dnd_audio.artifacts.records import (
    SegmentRecord,
    TranscriberIdentity,
    TranscriptRecords,
    TranscriptRecordsProvenance,
    WordRecord,
    segment_id,
)
from dnd_audio.artifacts.transcript import Transcript, TranscriptSpeaker
from dnd_audio.transcript.render import build_transcript, render_markdown, timestamp

HASH = "f" * 64
RATE = 48_000


def a_speaker(speaker_id: str = "alice", track_id: str = "tx-a") -> TranscriptSpeaker:
    return TranscriptSpeaker(
        speaker_id=speaker_id, speaker_name=speaker_id.title(), track_id=track_id
    )


def a_segment(index: int = 0, start: int = RATE, end: int = RATE * 2, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "segment_id": segment_id(index),
        "track_id": "tx-a",
        "speaker_id": "alice",
        "speaker_name": "Alice",
        "start_sample": start,
        "end_sample": end,
        "ownership_start_sample": start,
        "ownership_end_sample": end,
        "text": "We should go back to Zephyrine.",
        "alignment_status": "not_attempted",
        "decision": "retained",
        "source_candidate_ids": [f"cand_tx-a_{start:012d}"],
        "request_ids": [f"req_tx-a_{start:012d}"],
    }
    return SegmentRecord(**{**fields, **overrides})


def records(**overrides: Any) -> TranscriptRecords:
    fields: dict[str, Any] = {
        "session_id": "2026-08-15",
        "title": "Session 01",
        "language": "English",
        "config_hash": HASH,
        "timeline_sha256": HASH,
        "activity_cache_key": HASH,
        "sample_rate": RATE,
        "duration_samples": RATE * 600,
        "speakers": [a_speaker()],
        "segments": [a_segment()],
        "provenance": TranscriptRecordsProvenance(
            transcript_semantics_version=1,
            activity_semantics_version=1,
            timeline_semantics_version=1,
            inspection_semantics_version=1,
            numpy_version="2.3.4",
            transcriber=TranscriberIdentity(
                name="scripted", max_new_tokens=1024, language="English", variant_digest=HASH
            ),
        ),
    }
    return TranscriptRecords(**{**fields, **overrides})


class TestTheJsonDocument:
    def test_it_validates_against_the_checked_in_schema(self, repo_root: Path) -> None:
        """Against the artifact, not through the pydantic class that produced it."""
        schema = json.loads(
            (repo_root / "schemas" / "transcript.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(build_transcript(records()).model_dump(mode="json"), schema)

    def test_the_specs_own_example_still_validates(self, repo_root: Path) -> None:
        """Independent ground truth: if a change makes it stop validating, the change is
        wrong (the charter says so in as many words)."""
        schema = json.loads(
            (repo_root / "schemas" / "transcript.schema.json").read_text(encoding="utf-8")
        )
        example = json.loads(
            (repo_root / "tests" / "data" / "transcript-spec-example.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.validate(example, schema)
        assert Transcript.model_validate(example).segments[0].segment_id == "seg_000123"

    def test_only_retained_segments_are_rendered(self) -> None:
        document = build_transcript(
            records(
                segments=[
                    a_segment(0),
                    a_segment(
                        1,
                        RATE,
                        RATE * 2,
                        track_id="tx-b",
                        decision="duplicate",
                        duplicate_of_segment_id=segment_id(0),
                    ),
                ],
                speakers=[a_speaker()],
            )
        )
        assert [segment.segment_id for segment in document.segments] == [segment_id(0)]

    def test_a_gap_in_the_numbering_is_what_a_collapse_looks_like(self) -> None:
        document = build_transcript(
            records(
                segments=[
                    a_segment(0),
                    a_segment(
                        1,
                        RATE * 2,
                        RATE * 3,
                        decision="duplicate",
                        duplicate_of_segment_id=segment_id(0),
                    ),
                    a_segment(2, RATE * 4, RATE * 5),
                ]
            )
        )
        assert [segment.segment_id for segment in document.segments] == [
            segment_id(0),
            segment_id(2),
        ]

    def test_provenance_names_the_transcriber_and_the_source_candidate(self) -> None:
        (segment,) = build_transcript(records()).segments
        assert segment.provenance.asr_model == "scripted"
        assert segment.provenance.source_candidate_id == f"cand_tx-a_{RATE:012d}"
        assert segment.provenance.alignment_status == "not_attempted"

    def test_a_real_model_name_and_revision_reach_the_transcript(self) -> None:
        document = records(
            provenance=TranscriptRecordsProvenance(
                transcript_semantics_version=1,
                activity_semantics_version=1,
                timeline_semantics_version=1,
                inspection_semantics_version=1,
                numpy_version="2.3.4",
                transcriber=TranscriberIdentity(
                    name="qwen",
                    model="Qwen/Qwen3-ASR-1.7B",
                    model_revision="abc123",
                    max_new_tokens=1024,
                    language="English",
                ),
            )
        )
        (segment,) = build_transcript(document).segments
        assert segment.provenance.asr_model == "Qwen/Qwen3-ASR-1.7B"
        assert segment.provenance.asr_model_revision == "abc123"

    def test_there_is_no_confidence_field_anywhere(self) -> None:
        """The spec forbids manufacturing one the model does not expose."""
        serialized = json.dumps(build_transcript(records()).model_dump(mode="json"))
        assert "confidence" not in serialized


class TestPublicTimes:
    def test_a_sample_position_becomes_its_exact_millisecond(self) -> None:
        """Not "the value has three decimals" — the value `public_seconds` produces, on a
        position that is deliberately not millisecond aligned."""
        document = build_transcript(records(segments=[a_segment(0, 1, 49)]))
        (segment,) = document.segments
        assert (segment.start_s, segment.end_s) == (0.0, 0.001)

    def test_a_half_millisecond_rounds_away_from_zero(self) -> None:
        """24 samples at 48 kHz is exactly half a millisecond; Python's own round() would
        give zero here, which is the trap `determinism` exists to avoid."""
        document = build_transcript(records(segments=[a_segment(0, 24, RATE)]))
        assert document.segments[0].start_s == 0.001

    def test_word_times_are_millisecond_precise_too(self) -> None:
        segment = a_segment(
            0,
            words=[WordRecord(start_sample=RATE + 1, end_sample=RATE + 49, text="We")],
            alignment_status="aligned",
        )
        (rendered,) = build_transcript(records(segments=[segment])).segments
        assert [(word.start_s, word.end_s) for word in rendered.words] == [(1.0, 1.001)]

    def test_the_duration_is_the_sessions(self) -> None:
        assert build_transcript(records()).duration_s == 600.0


class TestOrderingAndIds:
    def test_segments_sort_by_start_then_id(self) -> None:
        document = build_transcript(
            records(
                segments=[
                    a_segment(2, RATE * 4, RATE * 5),
                    a_segment(1, RATE, RATE * 2, track_id="tx-b"),
                    a_segment(0, RATE, RATE * 2),
                ],
                speakers=[a_speaker()],
            )
        )
        assert [segment.segment_id for segment in document.segments] == [
            segment_id(0),
            segment_id(1),
            segment_id(2),
        ]

    def test_two_segments_starting_on_one_sample_are_ordered_by_id(self) -> None:
        """Start time alone is not a total order once two people talk at once."""
        document = build_transcript(
            records(segments=[a_segment(1, RATE, RATE * 2), a_segment(0, RATE, RATE * 2)])
        )
        assert [segment.start_s for segment in document.segments] == [1.0, 1.0]
        assert [segment.segment_id for segment in document.segments] == [
            segment_id(0),
            segment_id(1),
        ]

    def test_the_document_does_not_depend_on_the_order_segments_arrive_in(self) -> None:
        first = a_segment(0, RATE, RATE * 2)
        second = a_segment(1, RATE * 4, RATE * 5)
        assert build_transcript(records(segments=[first, second])) == build_transcript(
            records(segments=[second, first])
        )

    def test_speakers_sort_by_id(self) -> None:
        document = build_transcript(
            records(speakers=[a_speaker("bob", "tx-b"), a_speaker("alice", "tx-a")])
        )
        assert [speaker.speaker_id for speaker in document.speakers] == ["alice", "bob"]


class TestMarkdown:
    def test_it_renders_the_specs_format(self) -> None:
        rendered = render_markdown(
            records(
                segments=[a_segment(0, 231_429_120, 231_600_000)],
                duration_samples=RATE * 20_000,
            )
        )
        assert rendered.startswith("# Session 01\n\n")
        assert "**[01:20:21.440] Alice:** We should go back to Zephyrine.\n" in rendered

    def test_an_overlapping_turn_is_marked(self) -> None:
        rendered = render_markdown(records(segments=[a_segment(0, overlap=True)]))
        assert "**[00:00:01.000] Alice [overlap]:** We should go back to Zephyrine." in rendered

    def test_overlapping_turns_stay_separate_entries(self) -> None:
        """Merging two people talking at once would invent a turn nobody took."""
        rendered = render_markdown(
            records(
                segments=[
                    a_segment(0, overlap=True),
                    a_segment(
                        1,
                        RATE,
                        RATE * 2,
                        track_id="tx-b",
                        speaker_id="bob",
                        speaker_name="Bob",
                        text="Absolutely not.",
                        overlap=True,
                    ),
                ],
                speakers=[a_speaker(), a_speaker("bob", "tx-b")],
            )
        )
        assert rendered.count("**[00:00:01.000]") == 2
        assert "Alice [overlap]" in rendered
        assert "Bob [overlap]" in rendered

    def test_turns_are_in_start_order(self) -> None:
        rendered = render_markdown(
            records(
                segments=[
                    a_segment(1, RATE * 4, RATE * 5, text="second"),
                    a_segment(0, RATE, RATE * 2, text="first"),
                ]
            )
        )
        assert rendered.index("first") < rendered.index("second")

    def test_collapsed_duplicates_are_absent(self) -> None:
        rendered = render_markdown(
            records(
                segments=[
                    a_segment(0),
                    a_segment(
                        1,
                        RATE,
                        RATE * 2,
                        text="the collapsed copy",
                        decision="duplicate",
                        duplicate_of_segment_id=segment_id(0),
                    ),
                ]
            )
        )
        assert "the collapsed copy" not in rendered

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("*emphasis*", r"\*emphasis\*"),
            ("_under_", r"\_under\_"),
            ("`code`", r"\`code\`"),
            ("[link](x)", r"\[link\](x)"),
            ("<tag>", r"\<tag\>"),
            ("back\\slash", "back\\\\slash"),
        ],
    )
    def test_model_text_is_escaped(self, text: str, expected: str) -> None:
        rendered = render_markdown(records(segments=[a_segment(0, text=text)]))
        assert expected in rendered

    def test_a_speaker_name_is_escaped_too(self) -> None:
        """It comes from `session.yaml`, which a person wrote."""
        rendered = render_markdown(records(segments=[a_segment(0, speaker_name="*DM*")]))
        assert r"\*DM\*" in rendered

    def test_a_newline_that_survived_normalization_cannot_break_the_line(self) -> None:
        """Defensive: the format is one turn per line, and the renderer does not have to
        trust that everything upstream normalized its text."""
        rendered = render_markdown(records(segments=[a_segment(0, text="one\ntwo")]))
        assert "one two" in rendered
        assert len([line for line in rendered.splitlines() if line.startswith("**[")]) == 1

    def test_an_empty_transcript_still_has_its_title(self) -> None:
        assert render_markdown(records(segments=[])) == "# Session 01\n\n"

    def test_it_ends_with_a_newline(self) -> None:
        assert render_markdown(records()).endswith("\n")


class TestTheTwoDocumentsAgree:
    def test_the_markdown_timestamp_is_the_json_time(self) -> None:
        """Computed from the same integer millisecond count, so a rounding cannot differ."""
        document = records(
            segments=[a_segment(0, 231_429_120, 231_600_000)], duration_samples=RATE * 20_000
        )
        (segment,) = build_transcript(document).segments
        assert segment.start_s == 4821.44
        assert timestamp(231_429_120, RATE) == "01:20:21.440"

    def test_an_hour_past_midnight_is_not_wrapped(self) -> None:
        """A session that ran past midnight is one session."""
        assert timestamp(RATE * 3600 * 25, RATE) == "25:00:00.000"


class TestByteStability:
    def test_rendering_twice_produces_identical_bytes(self) -> None:
        document = records()
        assert render_markdown(document) == render_markdown(document)
        assert build_transcript(document) == build_transcript(document)
