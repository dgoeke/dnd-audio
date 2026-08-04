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


class TestPresentationTurns:
    def test_json_and_markdown_join_the_same_adjacent_records_with_plural_lineage(self) -> None:
        shared = "req_tx-a_shared"
        first = a_segment(
            0,
            RATE,
            RATE + 1000,
            text="Finally",
            request_ids=[shared],
            source_candidate_ids=["cand-a"],
        )
        second = a_segment(
            1,
            RATE + 17_800,
            RATE + 22_000,
            text="here's the fourth microphone",
            request_ids=[shared],
            source_candidate_ids=["cand-b"],
        )
        granular = records(
            segments=[first, second],
            presentation_join_gap_samples=16_800,
            overlap_min_samples=12_000,
        )

        (public,) = build_transcript(granular).segments
        markdown = render_markdown(granular)
        assert len(granular.retained()) == 2
        assert public.text == "Finally here's the fourth microphone"
        assert public.provenance.source_segment_ids == [segment_id(0), segment_id(1)]
        assert public.provenance.source_candidate_ids == ["cand-a", "cand-b"]
        assert markdown.count("**[") == 1
        assert public.text in markdown

    def test_request_batching_alone_cannot_join_across_the_presentation_gap(self) -> None:
        shared = "req_tx-a_shared"
        document = records(
            segments=[
                a_segment(0, RATE, RATE + 1000, text="one", request_ids=[shared]),
                a_segment(
                    1,
                    RATE + 17_802,
                    RATE + 20_000,
                    text="two",
                    request_ids=[shared],
                ),
            ],
            presentation_join_gap_samples=16_800,
            overlap_min_samples=12_000,
        )
        assert [item.text for item in build_transcript(document).segments] == ["one", "two"]

    def test_a_shared_gap_without_shared_request_lineage_stays_granular(self) -> None:
        document = records(
            segments=[
                a_segment(0, RATE, RATE + 1000, text="one", request_ids=["req-one"]),
                a_segment(
                    1,
                    RATE + 2000,
                    RATE + 3000,
                    text="two",
                    request_ids=["req-two"],
                ),
            ],
            presentation_join_gap_samples=16_800,
            overlap_min_samples=12_000,
        )
        assert len(build_transcript(document).segments) == 2

    @pytest.mark.parametrize(
        "second_change",
        [
            {"alignment_status": "segment_only"},
            {"overlap": True},
        ],
    )
    def test_incompatible_alignment_or_overlap_stays_granular(
        self, second_change: dict[str, Any]
    ) -> None:
        shared = "req_tx-a_shared"
        document = records(
            segments=[
                a_segment(0, RATE, RATE + 1000, text="one", request_ids=[shared]),
                a_segment(
                    1,
                    RATE + 2000,
                    RATE + 3000,
                    text="two",
                    request_ids=[shared],
                    **second_change,
                ),
            ],
            presentation_join_gap_samples=16_800,
            overlap_min_samples=12_000,
        )
        assert len(build_transcript(document).segments) == 2

    def test_an_intervening_speaker_prevents_joining(self) -> None:
        shared = "req_tx-a_shared"
        document = records(
            segments=[
                a_segment(0, RATE, RATE + 1000, text="one", request_ids=[shared]),
                a_segment(
                    1,
                    RATE + 1500,
                    RATE + 1800,
                    track_id="tx-b",
                    speaker_id="bob",
                    speaker_name="Bob",
                    text="interrupting",
                    request_ids=["req-b"],
                    source_candidate_ids=["cand-b"],
                ),
                a_segment(
                    2,
                    RATE + 2000,
                    RATE + 3000,
                    text="two",
                    request_ids=[shared],
                ),
            ],
            speakers=[a_speaker(), a_speaker("bob", "tx-b")],
            presentation_join_gap_samples=16_800,
            overlap_min_samples=12_000,
        )
        assert [item.text for item in build_transcript(document).segments] == [
            "one",
            "interrupting",
            "two",
        ]

    def test_overlap_is_recomputed_when_another_speaker_spans_the_joined_gap(self) -> None:
        bob = a_segment(
            0,
            0,
            1200,
            track_id="tx-b",
            speaker_id="bob",
            speaker_name="Bob",
            text="a long turn",
            request_ids=["req-b"],
            source_candidate_ids=["cand-b"],
        )
        alice_one = a_segment(
            1,
            900,
            950,
            text="one",
            request_ids=["req-a"],
            source_candidate_ids=["cand-a1"],
        )
        alice_two = a_segment(
            2,
            1150,
            1200,
            text="two",
            request_ids=["req-a"],
            source_candidate_ids=["cand-a2"],
        )
        document = records(
            segments=[bob, alice_one, alice_two],
            speakers=[a_speaker(), a_speaker("bob", "tx-b")],
            presentation_join_gap_samples=200,
            overlap_min_samples=250,
        )

        turns = build_transcript(document).segments
        assert [item.text for item in turns] == ["a long turn", "one two"]
        assert [item.overlap for item in turns] == [True, True]
        assert "Bob [overlap]" in render_markdown(document)
        assert "Alice [overlap]" in render_markdown(document)

    def test_legacy_records_without_thresholds_do_not_gain_new_grouping(self) -> None:
        shared = "req_tx-a_shared"
        document = records(
            segments=[
                a_segment(0, RATE, RATE + 1000, text="one", request_ids=[shared]),
                a_segment(1, RATE + 1001, RATE + 2000, text="two", request_ids=[shared]),
            ]
        )
        assert [item.text for item in build_transcript(document).segments] == ["one", "two"]


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


class TestTheMarkdownTimestampIsExactToo:
    """INV-04 at the *other* public boundary.

    `transcript.json` goes through `public_seconds` and is tested on a position that is not
    millisecond aligned. `timestamp()` had no such test, so the whole suite passed with its
    exact rational arithmetic replaced by `int(sample / rate * 1000)` — and the two documents
    would then disagree about a rounding, which the renderer's own docstring says cannot
    happen (M4's verify phase).
    """

    def test_a_half_millisecond_rounds_the_way_the_json_does(self) -> None:
        """24 samples at 48 kHz is exactly half a millisecond. Truncation says `.000`;
        `to_milliseconds` rounds half away from zero and says `.001`, as the JSON does."""
        assert timestamp(24, RATE) == "00:00:00.001"
        assert (
            build_transcript(records(segments=[a_segment(0, 24, RATE)])).segments[0].start_s
            == 0.001
        )

    def test_an_unaligned_position_agrees_with_the_json_to_the_millisecond(self) -> None:
        """Driven over positions that are deliberately not multiples of 48."""
        for sample in (1, 23, 24, 25, 49, 71, 72, 1_000_001):
            document = records(
                segments=[a_segment(0, sample, sample + RATE)], duration_samples=RATE * 20_000
            )
            (rendered,) = build_transcript(document).segments
            stamp = timestamp(sample, RATE)
            milliseconds = round(rendered.start_s * 1000)
            expected = (
                f"{milliseconds // 3_600_000:02d}:"
                f"{milliseconds // 60_000 % 60:02d}:"
                f"{milliseconds // 1000 % 60:02d}."
                f"{milliseconds % 1000:03d}"
            )
            assert stamp == expected, f"sample {sample}: markdown {stamp}, json {rendered.start_s}"
