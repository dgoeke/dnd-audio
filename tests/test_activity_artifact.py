"""`activity.json` is a frozen contract, and this is what freezing it means.

M4 and M5 both index into this document, so the tests here are less about "does the model
work" than about "can the model change without anyone noticing". Three of them are load-bearing
in that sense and would be worth keeping even if everything else here were deleted:

* the **field allowlist**, which is how INV-09 is actually enforced;
* the **no-floats walk**, which is how INV-02 stays true across a NumPy upgrade;
* the **consumer reads**, which are the only place M4's and M5's access patterns are
  exercised before those milestones exist to exercise them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import jsonschema
import pytest
from pydantic import ValidationError

from dnd_audio.artifacts.activity import (
    ACTIVITY_SCHEMA_VERSION,
    ActivityCandidate,
    ActivityDecision,
    ActivityGraph,
    ActivityNote,
    ActivityProvenance,
    ActivityTrack,
    CandidateEvidence,
    DetectorIdentity,
    DetectorInterface,
    candidate_id,
)

#: **Every** property name the frozen document may contain (ADR-0012).
#:
#: This list is INV-09's enforcement. "The activity package imports nothing from the
#: transcript layer" does not stop a later milestone adding a local `normalized_text: str`
#: to a model in this file — the import graph would be unchanged and the mix would start
#: depending on what an ASR model said. A name that is not here fails a test, so adding one
#: is a deliberate edit to a frozen contract rather than a quiet widening of it.
#:
#: If you are here because this test failed: adding a field to a frozen artifact is an
#: ADR-0005 decision. Establish that it is model-independent, add it as optional, and add
#: the name below in the same commit.
FROZEN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        # ActivityGraph
        "schema_version",
        "session_id",
        "config_hash",
        "timeline_sha256",
        "attribution_cache_key",
        "provenance",
        "sample_rate",
        "derivative_sample_rate",
        "duration_samples",
        "tracks",
        "candidates",
        "warnings",
        "decisions",
        # ActivityProvenance
        "activity_semantics_version",
        "timeline_semantics_version",
        "inspection_semantics_version",
        "numpy_version",
        "scipy_version",
        "onnxruntime_version",
        "detector",
        "speech_band_filter_name",
        "speech_band_filter_identity",
        # DetectorIdentity and DetectorInterface
        "name",
        "release",
        "commit",
        "model_sha256",
        "runtime",
        "runtime_version",
        "execution_provider",
        "interface",
        "variant_digest",
        "frame_samples",
        "context_samples",
        "state_shape",
        "input_names",
        # ActivityTrack
        "track_id",
        "speaker_id",
        "speaker_name",
        "detection_cache_key",
        "probability_relative_path",
        "probability_frames",
        "speech_reference_mbfs",
        # M8's diagnostic 8. Four counts, deliberately admitted one at a time rather than by
        # relaxing this list: each is an integer tally of decisions the activity stage already
        # made, so none can carry anything text-derived (INV-09), and together they make the
        # speech reference auditable from the artifact. That mattered enough to be worth four
        # entries — ADR-0029's defect was found by measuring audio by hand, and since that fix
        # the reference's population is a subset of the candidates and cannot be recomputed
        # from the document at all.
        "candidate_count",
        "reference_candidate_count",
        "retained_candidate_count",
        "suppressed_candidate_count",
        # ActivityCandidate
        "candidate_id",
        "start_sample",
        "end_sample",
        "derivative_start_sample",
        "derivative_end_sample",
        "probability_permille",
        "peak_probability_permille",
        "band_level_mbfs",
        "relative_level_mb",
        "score_permille",
        "score_level_permille",
        "score_confidence_permille",
        "score_dominance_permille",
        "score_correlation_permille",
        "decision",
        "ambiguous",
        "suppressed_by_candidate_id",
        "evidence",
        # CandidateEvidence
        "other_candidate_id",
        "other_track_id",
        "overlap_start_sample",
        "overlap_end_sample",
        "compared_derivative_samples",
        "correlation_permille",
        "lag_derivative_samples",
        "score_margin_permille",
        "level_delta_mb",
        "outcome",
        # ActivityNote and ActivityDecision
        "code",
        "message",
        "path",
        "subject",
        "detail",
    }
)

DIGEST: Final = "a" * 64
OTHER_DIGEST: Final = "b" * 64


def a_track(track_id: str = "tx-a", **overrides: Any) -> ActivityTrack:
    settings: dict[str, Any] = {
        "track_id": track_id,
        "speaker_id": "alice",
        "speaker_name": "Alice",
        "detection_cache_key": DIGEST,
        "probability_relative_path": f"work/cache/activity/detect/{DIGEST}.probs",
        "probability_frames": 100,
        "frame_samples": 512,
        "speech_reference_mbfs": -2800,
    }
    return ActivityTrack(**{**settings, **overrides})


def a_candidate(
    track_id: str = "tx-a", start: int = 48000, end: int = 96000, **overrides: Any
) -> ActivityCandidate:
    settings: dict[str, Any] = {
        "candidate_id": candidate_id(track_id, start),
        "track_id": track_id,
        "start_sample": start,
        "end_sample": end,
        "derivative_start_sample": start // 3,
        "derivative_end_sample": -(-end // 3),
        "probability_permille": 900,
        "peak_probability_permille": 980,
        "band_level_mbfs": -2800,
        "relative_level_mb": 0,
        "score_permille": 800,
        "score_level_permille": 1000,
        "score_confidence_permille": 900,
        "score_dominance_permille": 700,
        "score_correlation_permille": 600,
        "decision": "retained",
    }
    return ActivityCandidate(**{**settings, **overrides})


def a_graph(**overrides: Any) -> ActivityGraph:
    settings: dict[str, Any] = {
        "session_id": "2026-08-15",
        "config_hash": DIGEST,
        "timeline_sha256": OTHER_DIGEST,
        "attribution_cache_key": DIGEST,
        "provenance": ActivityProvenance(
            activity_semantics_version=1,
            timeline_semantics_version=1,
            inspection_semantics_version=1,
            numpy_version="2.1.0",
            scipy_version="1.18.0",
            detector=DetectorIdentity(name="scripted", variant_digest=DIGEST),
            speech_band_filter_name="fir_speechband_16k",
            speech_band_filter_identity=OTHER_DIGEST,
        ),
        "sample_rate": 48000,
        "derivative_sample_rate": 16000,
        "duration_samples": 504000,
        "tracks": [a_track()],
        "candidates": [a_candidate()],
    }
    return ActivityGraph(**{**settings, **overrides})


def property_names(node: Any, found: set[str]) -> set[str]:
    """Every key under any `properties` object anywhere in a JSON Schema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                found.update(value)
            property_names(value, found)
    elif isinstance(node, list):
        for item in node:
            property_names(item, found)
    return found


class TestTheFrozenFieldSet:
    """INV-09, enforced by name rather than by import graph."""

    def test_the_schema_contains_no_field_outside_the_allowlist(self, repo_root: Path) -> None:
        """The test a text-derived field has to walk past, and cannot.

        An import test cannot catch `normalized_text: str` added to a model in this package;
        this does. Every property name in the checked-in schema must be one someone wrote
        down as belonging to a model-independent document.
        """
        schema = json.loads((repo_root / "schemas" / "activity.schema.json").read_text())
        unexpected = sorted(property_names(schema, set()) - FROZEN_FIELDS)
        assert unexpected == [], (
            f"the frozen activity graph gained {unexpected}. Adding a field to it is an "
            f"ADR-0005 decision, and a text-derived one is an INV-09 violation."
        )

    def test_the_allowlist_names_nothing_the_schema_does_not_have(self, repo_root: Path) -> None:
        """Otherwise the list rots into a wish list and stops constraining anything."""
        schema = json.loads((repo_root / "schemas" / "activity.schema.json").read_text())
        assert sorted(FROZEN_FIELDS - property_names(schema, set())) == []

    def test_the_activity_package_does_not_import_the_transcript_layer(
        self, repo_root: Path
    ) -> None:
        """The other half of INV-09, structural rather than by name.

        Weaker than the allowlist and it fails for a different reason, which is why both are
        here: this one catches a dependency, that one catches a field.
        """
        forbidden = ("artifacts.transcript", "activity.silero" + "", "Transcriber")
        offenders = []
        for path in sorted((repo_root / "src" / "dnd_audio" / "activity").rglob("*.py")):
            text = path.read_text()
            for name in ("artifacts.transcript", "Transcriber", "TranscriptionResult"):
                if name in text and name != forbidden[1]:
                    offenders.append(f"{path.name}: {name}")
        assert offenders == [], (
            f"the activity package reaches into the transcript layer: {offenders}. The mix "
            f"consumes this graph and nothing text-derived may reach it (INV-09)."
        )


class TestAgainstTheCheckedInSchema:
    def test_a_real_graph_validates(self, repo_root: Path) -> None:
        """Against the committed file, not against the class that produced it."""
        schema = json.loads((repo_root / "schemas" / "activity.schema.json").read_text())
        jsonschema.validate(a_graph().model_dump(mode="json"), schema)

    def test_the_schema_version_is_frozen_at_one(self) -> None:
        assert ACTIVITY_SCHEMA_VERSION == 1
        assert a_graph().schema_version == 1

    def test_there_are_no_floats_anywhere_in_the_document(self) -> None:
        """Walked, not assumed. Every measured quantity is an integer in a named unit.

        A float here would be the quotient of two NumPy reductions, and INV-02 requires this
        document to be byte-identical on an unchanged rerun.
        """
        graph = a_graph(
            candidates=[
                a_candidate(),
                a_candidate(
                    track_id="tx-b",
                    start=48000,
                    end=96000,
                    decision="suppressed",
                    suppressed_by_candidate_id=candidate_id("tx-a", 48000),
                    evidence=[an_evidence()],
                ),
            ],
            tracks=[a_track(), a_track("tx-b", speaker_id="bob", speaker_name="Bob")],
        )
        floats = list(_floats_in(graph.model_dump(mode="json"), ""))
        assert floats == []


def an_evidence(**overrides: Any) -> CandidateEvidence:
    settings: dict[str, Any] = {
        "other_candidate_id": candidate_id("tx-a", 48000),
        "other_track_id": "tx-a",
        "overlap_start_sample": 48000,
        "overlap_end_sample": 96000,
        "compared_derivative_samples": 16000,
        "correlation_permille": 900,
        "lag_derivative_samples": 48,
        "score_margin_permille": 300,
        "level_delta_mb": 2600,
        "outcome": "suppresses",
    }
    return CandidateEvidence(**{**settings, **overrides})


def _floats_in(value: Any, path: str) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [f"{path} = {value!r}"]
    if isinstance(value, dict):
        return [f for key, item in value.items() for f in _floats_in(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [f for i, item in enumerate(value) for f in _floats_in(item, f"{path}[{i}]")]
    return []


class TestDeterministicIdentity:
    def test_an_id_derives_from_the_track_and_the_start(self) -> None:
        assert candidate_id("tx-a", 249600) == "cand_tx-a_000000249600"

    def test_ids_sort_lexically_in_time_order(self) -> None:
        """Zero-padded to twelve digits, so a text sort and a numeric sort agree.

        A consumer that sorts candidate ids as strings must not get a different order from
        one that sorts by `start_sample`, or two renderings of the same session disagree.
        """
        ids = [candidate_id("tx-a", position) for position in (9, 100, 48000, 691200000)]
        assert ids == sorted(ids)

    def test_a_candidate_whose_id_does_not_match_its_position_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="does not derive from its track"):
            a_candidate(candidate_id="cand_tx-a_000000000001")

    def test_two_candidates_may_not_share_an_id(self) -> None:
        with pytest.raises(ValidationError, match="share an id"):
            a_graph(candidates=[a_candidate(), a_candidate()])


class TestOrderingIsStatedNotIncidental:
    def test_candidates_sort_by_start_then_track(self) -> None:
        graph = a_graph(
            tracks=[a_track(), a_track("tx-b", speaker_id="bob", speaker_name="Bob")],
            candidates=[
                a_candidate("tx-b", 96000, 144000),
                a_candidate("tx-b", 48000, 96000),
                a_candidate("tx-a", 48000, 96000),
            ],
        )
        assert [(c.start_sample, c.track_id) for c in graph.candidates] == [
            (48000, "tx-a"),
            (48000, "tx-b"),
            (96000, "tx-b"),
        ]

    def test_tracks_sort_by_id(self) -> None:
        graph = a_graph(
            tracks=[a_track("tx-c", speaker_id="carol", speaker_name="Carol"), a_track("tx-a")],
            candidates=[],
        )
        assert [track.track_id for track in graph.tracks] == ["tx-a", "tx-c"]

    def test_evidence_sorts_by_the_competitor_it_names(self) -> None:
        candidate = a_candidate(
            evidence=[
                an_evidence(other_candidate_id="cand_tx-c_000000048000", other_track_id="tx-c"),
                an_evidence(other_candidate_id="cand_tx-b_000000048000", other_track_id="tx-b"),
            ],
            decision="suppressed",
            suppressed_by_candidate_id="cand_tx-b_000000048000",
        )
        assert [item.other_candidate_id for item in candidate.evidence] == [
            "cand_tx-b_000000048000",
            "cand_tx-c_000000048000",
        ]


class TestTheGridsCannotDisagree:
    def test_the_derivative_interval_must_cover_the_session_one(self) -> None:
        """Floor the start, ceil the end. Rounding both alike loses a word's first phoneme."""
        candidate = a_candidate(start=48001, end=96001)
        assert candidate.derivative_start_sample == 16000
        assert candidate.derivative_end_sample == 32001

    def test_a_derivative_interval_that_shrinks_the_session_one_is_refused(self) -> None:
        """Checked on the graph rather than the candidate, because only the graph knows both
        rates. A candidate carrying a shrunk interval is well formed in isolation and wrong
        in context, which is exactly where an off-by-one hides."""
        with pytest.raises(ValidationError, match="floors and"):
            a_graph(candidates=[a_candidate(start=48001, end=96001, derivative_end_sample=32000)])

    def test_a_candidate_past_the_session_duration_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="past the session"):
            a_graph(candidates=[a_candidate(start=480000, end=600000)])

    def test_the_two_rates_must_divide(self) -> None:
        with pytest.raises(ValidationError, match="not a whole multiple"):
            a_graph(derivative_sample_rate=44100, candidates=[])


class TestADecisionMustBeSupportedByItsEvidence:
    def test_a_suppressed_candidate_must_name_its_suppressor(self) -> None:
        with pytest.raises(ValidationError, match="names no suppressor"):
            a_candidate(decision="suppressed")

    def test_the_named_suppressor_must_have_a_suppressing_record(self) -> None:
        """Otherwise the document asserts a suppression nothing in it supports."""
        with pytest.raises(ValidationError, match="no evidence record"):
            a_candidate(
                decision="suppressed",
                suppressed_by_candidate_id=candidate_id("tx-b", 48000),
                evidence=[
                    an_evidence(
                        other_candidate_id=candidate_id("tx-b", 48000),
                        other_track_id="tx-b",
                        outcome="insufficient_margin",
                    )
                ],
            )

    def test_a_retained_candidate_may_not_name_a_suppressor(self) -> None:
        with pytest.raises(ValidationError, match="retained but names a suppressor"):
            a_candidate(suppressed_by_candidate_id=candidate_id("tx-b", 48000))

    def test_a_retained_candidate_may_not_carry_a_suppressing_record(self) -> None:
        with pytest.raises(ValidationError, match="claims to suppress it"):
            a_candidate(
                evidence=[
                    an_evidence(
                        other_candidate_id=candidate_id("tx-b", 48000), other_track_id="tx-b"
                    )
                ]
            )

    def test_suppressed_and_ambiguous_cannot_both_be_true(self) -> None:
        """Ambiguity is a reason to keep a candidate, never a reason to drop one."""
        with pytest.raises(ValidationError, match="both suppressed and ambiguous"):
            a_candidate(
                decision="suppressed",
                ambiguous=True,
                suppressed_by_candidate_id=candidate_id("tx-b", 48000),
                evidence=[
                    an_evidence(
                        other_candidate_id=candidate_id("tx-b", 48000), other_track_id="tx-b"
                    )
                ],
            )

    def test_evidence_must_name_a_candidate_that_exists(self) -> None:
        with pytest.raises(ValidationError, match="not in this graph"):
            a_graph(
                candidates=[
                    a_candidate(
                        evidence=[
                            an_evidence(
                                other_candidate_id=candidate_id("tx-z", 1000),
                                other_track_id="tx-z",
                                outcome="insufficient_margin",
                            )
                        ]
                    )
                ]
            )

    def test_a_candidate_is_never_compared_against_its_own_track(self) -> None:
        """Two candidates on one track are two utterances, not one duplicated."""
        with pytest.raises(ValidationError, match="its own track"):
            a_graph(
                candidates=[
                    a_candidate(
                        evidence=[
                            an_evidence(
                                other_candidate_id=candidate_id("tx-a", 96000),
                                other_track_id="tx-a",
                                outcome="insufficient_margin",
                            )
                        ]
                    ),
                    a_candidate(start=96000, end=144000),
                ]
            )

    def test_a_candidate_on_an_unknown_track_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not in this graph"):
            a_graph(candidates=[a_candidate("tx-z")])

    def test_a_peak_below_the_mean_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="peak probability below its mean"):
            a_candidate(probability_permille=900, peak_probability_permille=800)


class TestTheConsumerReads:
    """The access patterns M4 and M5 will use, exercised before they exist.

    ADR-0012 names both explicitly, and a contract nobody has read from is a contract nobody
    has checked. These are deliberately written the way a consumer would write them rather
    than the way the producer thinks about it.
    """

    def graph(self) -> ActivityGraph:
        tracks = [a_track(), a_track("tx-b", speaker_id="bob", speaker_name="Bob")]
        candidates = [
            a_candidate("tx-a", 48000, 96000),
            a_candidate(
                "tx-b",
                48000,
                96000,
                decision="suppressed",
                suppressed_by_candidate_id=candidate_id("tx-a", 48000),
                evidence=[an_evidence()],
            ),
            a_candidate("tx-a", 240000, 288000),
        ]
        return a_graph(tracks=tracks, candidates=candidates)

    def test_m4_takes_retained_candidates_in_time_order(self) -> None:
        """What it must transcribe, and — by omission — what it must not."""
        retained = self.graph().retained()
        assert [c.candidate_id for c in retained] == [
            candidate_id("tx-a", 48000),
            candidate_id("tx-a", 240000),
        ]
        assert all(c.decision == "retained" for c in retained)

    def test_m5_takes_one_tracks_active_intervals_with_a_confidence(self) -> None:
        """Plus the per-track voice-level correction it was asked to estimate."""
        graph = self.graph()
        intervals = [
            (c.start_sample, c.end_sample, c.score_permille) for c in graph.retained("tx-a")
        ]
        assert intervals == [(48000, 96000, 800), (240000, 288000, 800)]
        assert graph.retained("tx-b") == []
        reference = {t.track_id: t.speech_reference_mbfs for t in graph.tracks}
        assert reference == {"tx-a": -2800, "tx-b": -2800}

    def test_a_suppressed_candidate_can_be_traced_to_what_beat_it(self) -> None:
        """The candidate, not merely the track — "tx-a won" does not say which utterance."""
        graph = self.graph()
        suppressed = next(c for c in graph.candidates if c.decision == "suppressed")
        winner = next(
            c for c in graph.candidates if c.candidate_id == suppressed.suppressed_by_candidate_id
        )
        assert winner.track_id == "tx-a"
        assert winner.start_sample == 48000
        record = next(
            item
            for item in suppressed.evidence
            if item.other_candidate_id == suppressed.suppressed_by_candidate_id
        )
        assert record.outcome == "suppresses"
        assert record.correlation_permille == 900
        assert record.lag_derivative_samples == 48


class TestTheDetectorIdentity:
    def test_a_fake_carries_a_digest_and_no_model(self) -> None:
        identity = DetectorIdentity(name="scripted", variant_digest=DIGEST)
        assert identity.model_sha256 is None
        assert identity.interface is None

    def test_an_interface_records_how_the_model_was_called(self) -> None:
        """Not merely which weights it holds — a changed frame protocol is a changed answer."""
        interface = DetectorInterface(
            frame_samples=512,
            context_samples=64,
            state_shape=[2, 1, 128],
            input_names=["input", "state", "sr"],
            sample_rate=16000,
        )
        assert interface.frame_samples == 512
        assert interface.state_shape == [2, 1, 128]


class TestNotesAndDecisions:
    def test_warnings_and_decisions_sort_deterministically(self) -> None:
        graph = a_graph(
            warnings=[
                ActivityNote(code="b_code", message="second"),
                ActivityNote(code="a_code", message="first"),
            ],
            decisions=[
                ActivityDecision(code="bleed_suppressed", subject="z", detail="d"),
                ActivityDecision(code="bleed_suppressed", subject="a", detail="d"),
            ],
        )
        assert [note.code for note in graph.warnings] == ["a_code", "b_code"]
        assert [decision.subject for decision in graph.decisions] == ["a", "z"]

    def test_an_empty_overlap_in_evidence_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="which is empty"):
            an_evidence(overlap_start_sample=96000, overlap_end_sample=96000)
