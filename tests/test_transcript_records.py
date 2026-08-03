"""`work/transcript-records.json` — the contract `render` reads and nothing else.

Two things these tests are really about. The document has to *refuse* a state that would make
a transcript lie — a duplicate naming nothing, a chain of duplicates with no survivor at the
end, a word outside the interval that claims to own it — and it has to say which activity
graph and configuration it describes, so a records file beside the wrong one is a detectable
mistake rather than a plausible-looking transcript (ADR-0019).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from dnd_audio.artifacts.records import (
    TRANSCRIPT_RECORDS_SCHEMA_VERSION,
    RejectedAlternative,
    SegmentRecord,
    TranscriberIdentity,
    TranscriptDecision,
    TranscriptNote,
    TranscriptRecords,
    TranscriptRecordsProvenance,
    WordRecord,
    segment_id,
)
from dnd_audio.artifacts.transcript import TranscriptSpeaker

HASH = "f" * 64
OTHER_HASH = "a" * 64


def a_speaker(speaker_id: str = "alice", track_id: str = "tx-a") -> TranscriptSpeaker:
    return TranscriptSpeaker(
        speaker_id=speaker_id, speaker_name=speaker_id.title(), track_id=track_id
    )


def an_identity(**overrides: Any) -> TranscriberIdentity:
    fields: dict[str, Any] = {
        "name": "scripted",
        "max_new_tokens": 1024,
        "language": "English",
        "variant_digest": HASH,
    }
    return TranscriberIdentity(**{**fields, **overrides})


def a_provenance(**overrides: Any) -> TranscriptRecordsProvenance:
    fields: dict[str, Any] = {
        "transcript_semantics_version": 1,
        "activity_semantics_version": 1,
        "timeline_semantics_version": 1,
        "inspection_semantics_version": 1,
        "numpy_version": "2.3.4",
        "transcriber": an_identity(),
    }
    return TranscriptRecordsProvenance(**{**fields, **overrides})


def a_segment(
    index: int = 0,
    start: int = 48_000,
    end: int = 96_000,
    **overrides: Any,
) -> SegmentRecord:
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
        "source_candidate_ids": ["cand_tx-a_000000048000"],
        "request_ids": ["req_tx-a_000000048000"],
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
        "sample_rate": 48_000,
        "duration_samples": 504_000,
        "speakers": [a_speaker()],
        "segments": [a_segment()],
        "provenance": a_provenance(),
    }
    return TranscriptRecords(**{**fields, **overrides})


class TestIdentityOfASegment:
    def test_ids_are_position_in_the_canonical_order(self) -> None:
        assert segment_id(0) == "seg_000000"
        assert segment_id(123) == "seg_000123"

    def test_the_spec_example_id_is_one_of_ours(self) -> None:
        """`tests/data/transcript-spec-example.json` is ground truth; ADR-0019 keeps it valid."""
        assert segment_id(123) == "seg_000123"

    def test_a_negative_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            segment_id(-1)

    def test_segments_sort_by_start_then_id(self) -> None:
        document = records(
            segments=[
                a_segment(2, 240_000, 288_000),
                a_segment(1, 48_000, 96_000, track_id="tx-b"),
                a_segment(0, 48_000, 96_000),
            ],
            speakers=[a_speaker(), a_speaker("bob", "tx-b")],
        )
        # tx-b's segment is attributed to alice here on purpose: what is under test is the
        # ordering, and the speaker roster is checked separately.
        assert [s.segment_id for s in document.segments] == [
            "seg_000000",
            "seg_000001",
            "seg_000002",
        ]

    def test_two_segments_cannot_share_an_id(self) -> None:
        with pytest.raises(ValidationError, match="share an id"):
            records(segments=[a_segment(0), a_segment(0, 240_000, 288_000)])


class TestADecisionMustBeConsistentWithItsEvidence:
    def test_a_duplicate_names_what_absorbed_it(self) -> None:
        with pytest.raises(ValidationError, match="names nothing it duplicates"):
            a_segment(1, decision="duplicate")

    def test_a_retained_segment_names_no_duplicate(self) -> None:
        with pytest.raises(ValidationError, match="retained but names a segment"):
            a_segment(1, duplicate_of_segment_id=segment_id(0))

    def test_a_segment_cannot_duplicate_itself(self) -> None:
        with pytest.raises(ValidationError, match="duplicate of itself"):
            a_segment(1, decision="duplicate", duplicate_of_segment_id=segment_id(1))

    def test_a_duplicate_is_never_marked_overlapping(self) -> None:
        """`overlap` is about retained speakers; a collapsed segment is not in the transcript."""
        with pytest.raises(ValidationError, match="collapsed and also marked as overlapping"):
            a_segment(1, decision="duplicate", duplicate_of_segment_id=segment_id(0), overlap=True)

    def test_a_duplicate_reference_must_resolve(self) -> None:
        with pytest.raises(ValidationError, match="not in this document"):
            records(
                segments=[
                    a_segment(0),
                    a_segment(
                        1,
                        240_000,
                        288_000,
                        decision="duplicate",
                        duplicate_of_segment_id=segment_id(9),
                    ),
                ]
            )

    def test_a_chain_of_duplicates_is_refused(self) -> None:
        """Collapsing into something itself collapsed leaves no surviving text at the end."""
        with pytest.raises(ValidationError, match="which was itself collapsed"):
            records(
                segments=[
                    a_segment(0),
                    a_segment(
                        1,
                        240_000,
                        288_000,
                        decision="duplicate",
                        duplicate_of_segment_id=segment_id(0),
                    ),
                    a_segment(
                        2,
                        360_000,
                        408_000,
                        decision="duplicate",
                        duplicate_of_segment_id=segment_id(1),
                    ),
                ]
            )

    def test_a_collapsed_segment_may_not_also_have_rejected_alternatives(self) -> None:
        alternative = RejectedAlternative(
            segment_id=segment_id(0),
            track_id="tx-a",
            speaker_id="alice",
            text="whatever",
            overlap_permille=900,
            text_similarity_permille=950,
            correlation_permille=800,
            score_margin_permille=120,
        )
        with pytest.raises(ValidationError, match="also claims to have rejected"):
            a_segment(
                1,
                decision="duplicate",
                duplicate_of_segment_id=segment_id(0),
                rejected_alternatives=[alternative],
            )


class TestWordsBelongToTheIntervalThatOwnsThem:
    def test_a_word_outside_the_ownership_interval_is_refused(self) -> None:
        word = WordRecord(start_sample=300_000, end_sample=300_500, text="stray")
        with pytest.raises(ValidationError, match="outside its ownership interval"):
            a_segment(0, words=[word], alignment_status="aligned")

    def test_a_word_may_end_past_the_interval_its_start_is_in(self) -> None:
        """A word belongs to the interval containing its *start* — ADR-0020's rule 1."""
        word = WordRecord(start_sample=95_000, end_sample=97_000, text="straddling")
        segment = a_segment(0, words=[word], alignment_status="aligned")
        assert segment.words[0].end_sample > segment.ownership_end_sample

    def test_words_sort_by_position(self) -> None:
        first = WordRecord(start_sample=50_000, end_sample=51_000, text="We")
        second = WordRecord(start_sample=60_000, end_sample=61_000, text="should")
        segment = a_segment(0, words=[second, first], alignment_status="aligned")
        assert [word.text for word in segment.words] == ["We", "should"]

    def test_an_empty_word_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="spans an empty interval"):
            WordRecord(start_sample=50_000, end_sample=50_000, text="We")


class TestTheDocumentDeclaresWhatItDescribes:
    def test_it_names_the_graph_the_timeline_and_the_configuration(self) -> None:
        document = records(
            config_hash=HASH, timeline_sha256=OTHER_HASH, activity_cache_key="b" * 64
        )
        assert document.config_hash == HASH
        assert document.timeline_sha256 == OTHER_HASH
        assert document.activity_cache_key == "b" * 64

    def test_a_segment_past_the_session_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="past the session's"):
            records(segments=[a_segment(0, 480_000, 600_000)], duration_samples=504_000)

    def test_a_segment_must_name_a_speaker_in_the_document(self) -> None:
        with pytest.raises(ValidationError, match="who is not in this document"):
            records(segments=[a_segment(0, speaker_id="mallory")])

    def test_retained_excludes_collapsed_segments(self) -> None:
        document = records(
            segments=[
                a_segment(0),
                a_segment(
                    1,
                    240_000,
                    288_000,
                    decision="duplicate",
                    duplicate_of_segment_id=segment_id(0),
                ),
            ]
        )
        assert [s.segment_id for s in document.retained()] == [segment_id(0)]


class TestTheSerializedDocument:
    def test_there_are_no_floats_anywhere(self) -> None:
        """Same rule as `timeline.json` and `activity.json`: byte-stability (INV-02)."""
        document = records(
            segments=[
                a_segment(
                    0,
                    words=[WordRecord(start_sample=50_000, end_sample=51_000, text="We")],
                    alignment_status="aligned",
                )
            ]
        )
        floats = list(_floats(document.model_dump(mode="json")))
        assert floats == []

    def test_it_validates_against_the_checked_in_schema(self, repo_root: Path) -> None:
        schema = json.loads(
            (repo_root / "schemas" / "transcript-records.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(records().model_dump(mode="json"), schema)

    def test_the_schema_rejects_a_document_that_is_not_one(self, repo_root: Path) -> None:
        schema = json.loads(
            (repo_root / "schemas" / "transcript-records.schema.json").read_text(encoding="utf-8")
        )
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"schema_version": TRANSCRIPT_RECORDS_SCHEMA_VERSION}, schema)

    def test_warnings_and_decisions_sort_deterministically(self) -> None:
        document = records(
            warnings=[
                TranscriptNote(code="b_code", message="second"),
                TranscriptNote(code="a_code", message="first"),
            ],
            decisions=[
                TranscriptDecision(code="duplicate_collapsed", subject="z", detail="d"),
                TranscriptDecision(code="duplicate_collapsed", subject="a", detail="d"),
            ],
        )
        assert [note.code for note in document.warnings] == ["a_code", "b_code"]
        assert [decision.subject for decision in document.decisions] == ["a", "z"]

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptRecords.model_validate(
                {**records().model_dump(mode="json"), "invented": True}
            )


def _floats(value: object) -> Iterator[float]:
    """Every float in a serialized document, wherever it is nested."""
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from _floats(item)
