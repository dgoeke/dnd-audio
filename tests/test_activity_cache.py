"""Two identities that invalidate for different reasons, and no half-entry that reads as a hit.

The failure available to a detection cache is the same one M2's derivative cache has: not a
crash, but a stale answer that looks exactly like a fresh one. A graph built from five
current detections and one from before a threshold moved has the right shape, the right
track ids, and candidates in the wrong places — and nothing downstream can notice, because
"where the speech is" is precisely what it was asking.

So this file follows `test_derivatives.py`, and for its stated reason: each component of
each identity is varied **independently**, and the *document* is asserted on as well as the
hash. A key that happened to change for the right reason in one test can still be missing a
component, and the missing one is always the one that matters later.

The other half is ADR-0016's actual claim — that the two identities are separate, not
merely two names for the same inputs. Tuning `activity.bleed` must move the attribution key
and leave every detection key alone, or the tuning loop OQ-017 guarantees gets walked
re-runs six tracks of inference to discover that a score weight cannot change a per-frame
probability. Both projections are computed from real :class:`SessionConfig` objects here, so
these tests exercise the projection table rather than two arbitrary strings.

INV-08's last clause gets its own class: an entry is committed only after INV-01 has been
re-verified, never at publish time. M2's closeout records shipping that wrong once — a run
that correctly fails on a changed source leaves behind an entry keyed on the bytes it read,
restoring the file makes that key match again, and the cache is poisoned permanently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy

from dnd_audio.activity import ACTIVITY_SEMANTICS_VERSION, DETECTOR_FRAME_SAMPLES
from dnd_audio.activity.cache import (
    CACHE_RECORD_VERSION,
    PROBABILITY_DTYPE,
    AttributionCache,
    DetectionCache,
    attribution_identity,
    attribution_identity_document,
    detection_identity,
    detection_identity_document,
    probability_relative_path,
)
from dnd_audio.activity.detect import DetectionResult, SpeechRegion
from dnd_audio.artifacts.activity import (
    ActivityCandidate,
    ActivityGraph,
    ActivityProvenance,
    ActivityTrack,
    DetectorIdentity,
    candidate_id,
)
from dnd_audio.config import (
    ActivityConfig,
    BleedConfig,
    ScoringConfig,
    SessionConfig,
    TrackConfig,
    VadConfig,
    stage_config_hash,
)
from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.timeline import TIMELINE_SEMANTICS_VERSION

KEY = "a1" * 32
OTHER_KEY = "b2" * 32
DERIVATIVE_KEY = "c3" * 32
TIMELINE_SHA = "d4" * 32
FILTER_ID = "e5" * 32

#: Written out rather than imported, so a test that moves the cache layout has to say so.
#: Pinned against the module's own path builder by `test_the_probabilities_live_beside...`.
DETECTION_DIR = "work/cache/activity/detect"
ATTRIBUTION_DIR = "work/cache/activity/graph"


def a_detector(*, name: str = "scripted", variant_digest: str = "1" * 64) -> DetectorIdentity:
    return DetectorIdentity(name=name, variant_digest=variant_digest)


def a_config(*, activity: ActivityConfig | None = None) -> SessionConfig:
    """A minimal real session, so the projections under test are the real ones."""
    return SessionConfig(
        session_id="2026-08-15",
        title="Session 01",
        tracks=[
            TrackConfig(
                track_id="tx-a",
                receiver_id="rx-1",
                receiver_channel=1,
                speaker_id="alice",
                speaker_name="Alice",
                input="raw/tx-a",
            )
        ],
        activity=activity if activity is not None else ActivityConfig(),
    )


def detection_key_for(**overrides: object) -> str:
    settings: dict[str, object] = {
        "track_id": "tx-a",
        "derivative_cache_key": DERIVATIVE_KEY,
        "detector": a_detector(),
        "stage_config_hash": stage_config_hash(a_config(), "detection"),
    }
    settings.update(overrides)
    return detection_identity(**settings)  # type: ignore[arg-type]


def detection_document_for(**overrides: object) -> dict[str, Any]:
    settings: dict[str, object] = {
        "track_id": "tx-a",
        "derivative_cache_key": DERIVATIVE_KEY,
        "detector": a_detector(),
        "stage_config_hash": stage_config_hash(a_config(), "detection"),
    }
    settings.update(overrides)
    return detection_identity_document(**settings)  # type: ignore[arg-type]


def attribution_key_for(**overrides: object) -> str:
    settings: dict[str, object] = {
        "detection_keys": [KEY, OTHER_KEY],
        "timeline_sha256": TIMELINE_SHA,
        "speech_band_identity": FILTER_ID,
        "stage_config_hash": stage_config_hash(a_config(), "attribution"),
    }
    settings.update(overrides)
    return attribution_identity(**settings)  # type: ignore[arg-type]


def attribution_document_for(**overrides: object) -> dict[str, Any]:
    settings: dict[str, object] = {
        "detection_keys": [KEY, OTHER_KEY],
        "timeline_sha256": TIMELINE_SHA,
        "speech_band_identity": FILTER_ID,
        "stage_config_hash": stage_config_hash(a_config(), "attribution"),
    }
    settings.update(overrides)
    return attribution_identity_document(**settings)  # type: ignore[arg-type]


def a_result(
    *,
    track_id: str = "tx-a",
    probabilities: tuple[int, ...] = (0, 500, 1000, 250),
    from_detector: bool = True,
) -> DetectionResult:
    """One track's detection pass, with regions that a round-trip can be checked against."""
    return DetectionResult(
        track_id=track_id,
        regions=(
            SpeechRegion(
                start_sample=512,
                end_sample=2048,
                probability_permille=583,
                peak_probability_permille=1000,
            ),
        ),
        frame_probabilities=np.array(probabilities, dtype=np.uint16),
        from_detector=from_detector,
    )


def a_graph(key: str = KEY) -> ActivityGraph:
    """A minimal graph that satisfies every validator the artifact carries.

    One track and one candidate rather than none: an empty graph would round-trip through
    a cache that dropped both lists, and the candidate is where the model's cross-field
    checks live.
    """
    return ActivityGraph(
        session_id="2026-08-15",
        config_hash="c" * 64,
        timeline_sha256=TIMELINE_SHA,
        attribution_cache_key=key,
        provenance=ActivityProvenance(
            activity_semantics_version=ACTIVITY_SEMANTICS_VERSION,
            timeline_semantics_version=TIMELINE_SEMANTICS_VERSION,
            inspection_semantics_version=1,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            detector=a_detector(),
            speech_band_filter_name="speech-band-fir",
            speech_band_filter_identity=FILTER_ID,
        ),
        sample_rate=48_000,
        derivative_sample_rate=16_000,
        duration_samples=480_000,
        tracks=[
            ActivityTrack(
                track_id="tx-a",
                speaker_id="alice",
                speaker_name="Alice",
                detection_cache_key=DERIVATIVE_KEY,
                probability_relative_path=probability_relative_path(DERIVATIVE_KEY),
                probability_frames=4,
                frame_samples=DETECTOR_FRAME_SAMPLES,
                speech_reference_mbfs=-1800,
            )
        ],
        candidates=[
            ActivityCandidate(
                candidate_id=candidate_id("tx-a", 48_000),
                track_id="tx-a",
                start_sample=48_000,
                end_sample=96_000,
                derivative_start_sample=16_000,
                derivative_end_sample=32_000,
                probability_permille=800,
                peak_probability_permille=900,
                band_level_mbfs=-2000,
                score_permille=700,
                score_level_permille=600,
                score_confidence_permille=800,
                score_dominance_permille=500,
                score_correlation_permille=400,
                decision="retained",
            )
        ],
    )


class TestDetectionIdentityCoversEverythingThatChangesADetection:
    def test_the_same_inputs_give_the_same_key(self) -> None:
        assert detection_key_for() == detection_key_for()

    def test_the_track_id_changes_the_key(self) -> None:
        """Two tracks detected with one detector are still two answers."""
        assert detection_key_for() != detection_key_for(track_id="tx-b")

    def test_a_different_derivative_changes_the_key(self) -> None:
        """The load-bearing component: it already carries the sources and the segment map.

        A placement fix that moves a chunk changes no source byte and no configuration
        value, and only the derivative's own key records it. Without this, detection would
        keep serving candidates aligned to a timeline that has since moved.
        """
        assert detection_key_for() != detection_key_for(derivative_cache_key="e" * 64)

    def test_a_different_detector_changes_the_key(self) -> None:
        assert detection_key_for() != detection_key_for(detector=a_detector(name="silero"))

    def test_two_scripted_detectors_with_different_scripts_are_different_detectors(self) -> None:
        """The fake's `variant_digest` is why one test's answers cannot be served to another."""
        other = a_detector(variant_digest="2" * 64)
        assert detection_key_for() != detection_key_for(detector=other)

    def test_the_detection_projection_changes_the_key(self) -> None:
        tuned = a_config(activity=ActivityConfig(vad=VadConfig(min_speech_ms=300)))
        assert detection_key_for() != detection_key_for(
            stage_config_hash=stage_config_hash(tuned, "detection")
        )

    @pytest.mark.parametrize(
        "component",
        [
            "activity_semantics_version",
            "cache_record_version",
            "derivative_cache_key",
            "detector",
            "frame_samples",
            "numpy_version",
            "stage_config_hash",
            "timeline_semantics_version",
            "track_id",
        ],
    )
    def test_the_identity_carries_every_declared_component(self, component: str) -> None:
        """Asserted on the document, not inferred from hashes changing.

        The semantics and NumPy versions cannot be varied in-process, and a component that
        is simply absent would leave every "different inputs, different key" test above
        passing. The only way to know they are in there is to look.
        """
        document = detection_document_for()
        assert component in document
        assert document[component]

    def test_the_document_is_what_gets_hashed(self) -> None:
        """So the two cannot drift apart and leave the assertions above meaningless."""
        expected = sha256_bytes(canonical_json(detection_document_for()).encode("utf-8"))
        assert detection_key_for() == expected


class TestAttributionIdentityCoversEverythingThatChangesAGraph:
    def test_the_same_inputs_give_the_same_key(self) -> None:
        assert attribution_key_for() == attribution_key_for()

    def test_a_different_detection_changes_the_key(self) -> None:
        """Carried rather than re-derived, so this inherits everything each key covers."""
        assert attribution_key_for() != attribution_key_for(detection_keys=[KEY, "f" * 64])

    def test_dropping_a_detection_changes_the_key(self) -> None:
        """A graph built from five fresh detections and one missing one is a different graph."""
        assert attribution_key_for() != attribution_key_for(detection_keys=[KEY])

    def test_reordering_the_detections_does_not_change_the_key(self) -> None:
        """The keys are a set of inputs; the order they arrive in is not an input (INV-02)."""
        assert attribution_key_for(detection_keys=[KEY, OTHER_KEY]) == attribution_key_for(
            detection_keys=[OTHER_KEY, KEY]
        )

    def test_a_different_timeline_changes_the_key(self) -> None:
        """A graph read beside a timeline that no longer describes it misplaces every word."""
        assert attribution_key_for() != attribution_key_for(timeline_sha256="8" * 64)

    def test_a_different_speech_band_filter_changes_the_key(self) -> None:
        """Every level comparison in the graph was measured through it."""
        assert attribution_key_for() != attribution_key_for(speech_band_identity="a" * 64)

    def test_the_attribution_projection_changes_the_key(self) -> None:
        tuned = a_config(activity=ActivityConfig(bleed=BleedConfig(min_correlation=0.51)))
        assert attribution_key_for() != attribution_key_for(
            stage_config_hash=stage_config_hash(tuned, "attribution")
        )

    @pytest.mark.parametrize(
        "component",
        [
            "activity_semantics_version",
            "cache_record_version",
            "detection_keys",
            "numpy_version",
            "scipy_version",
            "speech_band_identity",
            "stage_config_hash",
            "timeline_semantics_version",
            "timeline_sha256",
        ],
    )
    def test_the_identity_carries_every_declared_component(self, component: str) -> None:
        document = attribution_document_for()
        assert component in document
        assert document[component]

    def test_the_document_is_what_gets_hashed(self) -> None:
        expected = sha256_bytes(canonical_json(attribution_document_for()).encode("utf-8"))
        assert attribution_key_for() == expected


class TestTheTwoIdentitiesInvalidateSeparately:
    """ADR-0016's whole claim, stated in both directions.

    A too-narrow key serves a stale artifact as current, which is silent; a too-broad one
    recomputes, which is merely slow. So each of these asserts both that the right key moved
    and that the other one did not.
    """

    @staticmethod
    def keys_for(config: SessionConfig) -> tuple[str, str]:
        detection = detection_key_for(stage_config_hash=stage_config_hash(config, "detection"))
        attribution = attribution_key_for(
            stage_config_hash=stage_config_hash(config, "attribution")
        )
        return detection, attribution

    def test_tuning_a_vad_threshold_invalidates_both(self) -> None:
        """Detection because it is inference; attribution because it consumes detections."""
        base_detection, base_attribution = self.keys_for(a_config())
        tuned = a_config(activity=ActivityConfig(vad=VadConfig(speech_threshold=0.6)))
        detection, attribution = self.keys_for(tuned)

        assert detection != base_detection
        assert attribution != base_attribution

    def test_tuning_a_bleed_threshold_leaves_every_detection_alone(self) -> None:
        """Raising `min_correlation` by a hundredth cannot change a per-frame probability.

        Under whole-configuration hashing it re-ran six tracks of inference to find that
        out, which is the tuning loop this project expects to walk repeatedly (OQ-017).
        """
        base_detection, base_attribution = self.keys_for(a_config())
        tuned = a_config(activity=ActivityConfig(bleed=BleedConfig(min_correlation=0.51)))
        detection, attribution = self.keys_for(tuned)

        assert attribution != base_attribution
        assert detection == base_detection

    def test_tuning_a_score_weight_leaves_every_detection_alone(self) -> None:
        base_detection, base_attribution = self.keys_for(a_config())
        tuned = a_config(activity=ActivityConfig(scoring=ScoringConfig(level_weight=0.36)))
        detection, attribution = self.keys_for(tuned)

        assert attribution != base_attribution
        assert detection == base_detection

    def test_a_different_detector_changes_the_detection_key(self) -> None:
        assert detection_key_for(detector=a_detector(name="silero")) != detection_key_for()

    def test_a_different_derivative_changes_the_detection_key(self) -> None:
        assert detection_key_for(derivative_cache_key="e" * 64) != detection_key_for()

    def test_a_different_speech_band_filter_never_reaches_detection(self) -> None:
        """The filter band-limits a level comparison; it cannot touch a 16 kHz probability.

        Asserted on the document as well as through the key, because "the detection key did
        not change" is trivially true of an argument the function does not take — and would
        stay true if the filter later became something detection *did* depend on.
        """
        assert attribution_key_for(speech_band_identity="a" * 64) != attribution_key_for()
        assert "speech_band_identity" not in detection_document_for()


class TestAnIncompleteDetectionEntryIsNeverAHit:
    """INV-08 states it outright, so each way of being incomplete is driven directly."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> DetectionCache:
        return DetectionCache(session_dir=tmp_path)

    def a_published_entry(self, cache: DetectionCache, key: str = KEY) -> str:
        cache.publish(key, a_result())
        cache.commit()
        return key

    @staticmethod
    def sidecar(cache: DetectionCache, key: str) -> Path:
        return cache.session_dir / f"{DETECTION_DIR}/{key}.json"

    @staticmethod
    def probabilities(cache: DetectionCache, key: str) -> Path:
        return cache.session_dir / f"{DETECTION_DIR}/{key}.probs"

    def edit_sidecar(self, cache: DetectionCache, entry_key: str, changes: dict[str, Any]) -> None:
        """Rewrite one field of a committed sidecar. ``changes`` is a dict rather than
        keyword arguments because one of the fields a test edits is called ``key``."""
        path = self.sidecar(cache, entry_key)
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(changes)
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_the_probabilities_live_beside_the_sidecar_under_the_key(self) -> None:
        """Content-addressed, so two runs of one session can coexist while one rebuilds."""
        assert probability_relative_path(KEY) == f"{DETECTION_DIR}/{KEY}.probs"

    def test_a_complete_entry_is_a_hit(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        found = cache.get(key)

        assert found is not None
        assert found.track_id == "tx-a"
        assert found.frame_count == 4
        assert found.from_detector is True
        assert found.regions == a_result().regions
        assert found.probabilities(cache.session_dir).tolist() == [0, 500, 1000, 250]
        assert (cache.hits, cache.misses) == (1, 0)

    def test_hits_and_misses_are_counted_separately(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        assert cache.get(OTHER_KEY) is None
        assert cache.get(key) is not None
        assert cache.get(key) is not None
        assert (cache.hits, cache.misses) == (2, 1)

    def test_no_sidecar_at_all_is_a_miss(self, cache: DetectionCache) -> None:
        """Probabilities written, the process died before the sidecar. The safe order."""
        cache.publish(KEY, a_result())
        assert self.probabilities(cache, KEY).exists()
        assert cache.get(KEY) is None
        assert cache.misses == 1

    def test_an_unparsable_sidecar_costs_time_not_a_session(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        self.sidecar(cache, key).write_text("{not json", encoding="utf-8")
        assert cache.get(key) is None

    def test_a_sidecar_that_is_not_an_object_is_a_miss(self, cache: DetectionCache) -> None:
        """Valid JSON, wrong shape: `.get` on a list is an AttributeError, not a miss."""
        key = self.a_published_entry(cache)
        self.sidecar(cache, key).write_text("[1, 2, 3]", encoding="utf-8")
        assert cache.get(key) is None

    def test_a_sidecar_naming_a_different_key_is_a_miss(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        self.edit_sidecar(cache, key, {"key": OTHER_KEY})
        assert cache.get(key) is None

    def test_a_record_shape_this_code_never_wrote_is_a_miss(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        self.edit_sidecar(cache, key, {"cache_record_version": CACHE_RECORD_VERSION + 998})
        assert cache.get(key) is None

    def test_a_sidecar_naming_another_file_is_a_miss(self, cache: DetectionCache) -> None:
        """It would grant a hit on the strength of a file nothing goes on to read.

        The decoy is the right size and full of the wrong numbers, which is the case a
        presence-and-size check alone would accept.
        """
        key = self.a_published_entry(cache)
        decoy = self.probabilities(cache, OTHER_KEY)
        decoy.write_bytes(b"\xff" * 8)
        self.edit_sidecar(
            cache, key, {"probability_relative_path": f"{DETECTION_DIR}/{OTHER_KEY}.probs"}
        )
        assert cache.get(key) is None

    def test_a_sidecar_recording_another_frame_size_is_a_miss(self, cache: DetectionCache) -> None:
        """The frame is the resolution every probability in the file is quoted at."""
        key = self.a_published_entry(cache)
        self.edit_sidecar(cache, key, {"frame_samples": DETECTOR_FRAME_SAMPLES // 2})
        assert cache.get(key) is None

    def test_a_sidecar_recording_another_dtype_is_a_miss(self, cache: DetectionCache) -> None:
        """Little-endian is spelled out, so a cache written on one machine is not misread."""
        key = self.a_published_entry(cache)
        self.edit_sidecar(cache, key, {"probability_dtype": ">u2"})
        assert cache.get(key) is None

    def test_a_sidecar_missing_a_field_is_a_miss(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        path = self.sidecar(cache, key)
        document = json.loads(path.read_text(encoding="utf-8"))
        del document["frame_count"]
        path.write_text(json.dumps(document), encoding="utf-8")
        assert cache.get(key) is None

    def test_a_missing_probability_file_is_a_miss(self, cache: DetectionCache) -> None:
        key = self.a_published_entry(cache)
        self.probabilities(cache, key).unlink()
        assert cache.get(key) is None

    def test_a_truncated_probability_file_is_a_miss(self, cache: DetectionCache) -> None:
        """The check that makes "incomplete is never a hit" true rather than intended.

        Two bytes per frame and no header, so a short file is a perfectly readable array of
        fewer frames. Without comparing its size against what the sidecar recorded, the
        cache would serve a track whose last frames are simply gone — silence, in the place
        where a run was interrupted.
        """
        key = self.a_published_entry(cache)
        path = self.probabilities(cache, key)
        path.write_bytes(path.read_bytes()[:6])
        assert cache.get(key) is None

    def test_a_probability_file_longer_than_recorded_is_a_miss(self, cache: DetectionCache) -> None:
        """The other direction: an entry rebuilt over a longer track, sidecar not replaced."""
        key = self.a_published_entry(cache)
        path = self.probabilities(cache, key)
        path.write_bytes(path.read_bytes() + b"\x00\x00")
        assert cache.get(key) is None

    def test_reading_can_be_disabled_without_disabling_writing(self, tmp_path: Path) -> None:
        """`--no-cache` distrusts what is stored; refusing to store too would make every
        run slow rather than one."""
        cache = DetectionCache(session_dir=tmp_path, read_enabled=False)
        key = self.a_published_entry(cache)
        assert cache.get(key) is None
        assert cache.misses == 1
        assert self.probabilities(cache, key).exists()


class TestNothingIsFindableUntilCommit:
    """INV-08's ordering clause, which M2's closeout records shipping wrong once.

    A run that correctly fails on a changed source must not leave behind an entry keyed on
    the bytes it read: restoring the original file makes that key match again, forever. The
    probabilities landing on disk is therefore not what makes an entry findable — the
    sidecar is, and it is staged in memory until the caller has re-verified INV-01.
    """

    def test_publishing_alone_leaves_the_entry_invisible(self, tmp_path: Path) -> None:
        cache = DetectionCache(session_dir=tmp_path)
        cache.publish(KEY, a_result())

        assert cache.get(KEY) is None
        # A second reader of the same session, in case the miss above were only bookkeeping
        # inside the instance that staged it.
        assert DetectionCache(session_dir=tmp_path).get(KEY) is None

    def test_committing_makes_it_findable(self, tmp_path: Path) -> None:
        cache = DetectionCache(session_dir=tmp_path)
        cache.publish(KEY, a_result())
        assert cache.commit() == 1
        assert DetectionCache(session_dir=tmp_path).get(KEY) is not None

    def test_discarding_leaves_the_probabilities_inert(self, tmp_path: Path) -> None:
        """The data stays — it is content-addressed and harmless — and nothing names it."""
        cache = DetectionCache(session_dir=tmp_path)
        cache.publish(KEY, a_result())
        cache.discard()

        assert cache.commit() == 0
        assert cache.get(KEY) is None
        assert (tmp_path / f"{DETECTION_DIR}/{KEY}.probs").exists()
        assert not (tmp_path / f"{DETECTION_DIR}/{KEY}.json").exists()

    def test_a_staged_graph_is_not_findable_either(self, tmp_path: Path) -> None:
        cache = AttributionCache(session_dir=tmp_path)
        cache.publish(KEY, a_graph())

        assert AttributionCache(session_dir=tmp_path).get(KEY) is None
        assert cache.commit() == 1
        assert AttributionCache(session_dir=tmp_path).get(KEY) is not None

    def test_a_discarded_graph_never_reaches_disk(self, tmp_path: Path) -> None:
        cache = AttributionCache(session_dir=tmp_path)
        cache.publish(KEY, a_graph())
        cache.discard()

        assert cache.commit() == 0
        assert cache.get(KEY) is None
        assert not (tmp_path / ATTRIBUTION_DIR).exists()


class TestTheAttributionCache:
    @pytest.fixture
    def cache(self, tmp_path: Path) -> AttributionCache:
        return AttributionCache(session_dir=tmp_path)

    @staticmethod
    def entry(cache: AttributionCache, key: str) -> Path:
        return cache.session_dir / f"{ATTRIBUTION_DIR}/{key}.json"

    def test_a_stored_graph_round_trips(self, cache: AttributionCache) -> None:
        """Every field, not just the ones a summary would compare."""
        graph = a_graph()
        cache.publish(KEY, graph)
        cache.commit()

        found = cache.get(KEY)
        assert found == graph
        assert (cache.hits, cache.misses) == (1, 0)

    def test_a_graph_stored_under_another_key_is_a_miss(self, cache: AttributionCache) -> None:
        """The graph carries its own identity so a consumer can check it without recomputing.

        A document whose `attribution_cache_key` disagrees with the path it was found at is
        one of the two that got renamed, and neither is the one being asked for.
        """
        cache.publish(OTHER_KEY, a_graph(KEY))
        cache.commit()

        assert cache.get(OTHER_KEY) is None
        assert cache.misses == 1

    def test_a_document_that_no_longer_validates_is_a_miss(self, cache: AttributionCache) -> None:
        """A schema change must cost a rebuild, not a session.

        `schema_version: 2` is what a future artifact revision looks like from here, and the
        artifact is regenerable — refusing to rebuild it would turn a version bump into an
        unrecoverable session.
        """
        cache.publish(KEY, a_graph())
        cache.commit()
        path = self.entry(cache, KEY)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["schema_version"] = 2
        path.write_text(json.dumps(document), encoding="utf-8")

        assert cache.get(KEY) is None
        assert cache.misses == 1

    def test_a_document_that_breaks_a_cross_field_rule_is_a_miss(
        self, cache: AttributionCache
    ) -> None:
        """The validators that check a candidate against its own id and track (INV-02)."""
        cache.publish(KEY, a_graph())
        cache.commit()
        path = self.entry(cache, KEY)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["candidates"][0]["track_id"] = "tx-nobody"
        path.write_text(json.dumps(document), encoding="utf-8")

        assert cache.get(KEY) is None

    def test_an_unparsable_entry_is_a_miss(self, cache: AttributionCache) -> None:
        cache.publish(KEY, a_graph())
        cache.commit()
        self.entry(cache, KEY).write_text("{not json", encoding="utf-8")
        assert cache.get(KEY) is None

    def test_a_missing_entry_is_a_miss(self, cache: AttributionCache) -> None:
        assert cache.get(KEY) is None
        assert cache.misses == 1

    def test_reading_can_be_disabled_without_disabling_writing(self, tmp_path: Path) -> None:
        cache = AttributionCache(session_dir=tmp_path, read_enabled=False)
        cache.publish(KEY, a_graph())
        cache.commit()

        assert cache.get(KEY) is None
        assert cache.misses == 1
        assert (tmp_path / f"{ATTRIBUTION_DIR}/{KEY}.json").exists()


class TestTheProbabilityFileIsWrittenAsDeclared:
    """The dtype the sidecar records is the dtype on disk, or a reader sees other numbers."""

    def test_two_little_endian_bytes_per_frame(self, tmp_path: Path) -> None:
        cache = DetectionCache(session_dir=tmp_path)
        cache.publish(KEY, a_result(probabilities=(0, 1, 256, 1000)))
        cache.commit()

        raw = (tmp_path / f"{DETECTION_DIR}/{KEY}.probs").read_bytes()
        assert PROBABILITY_DTYPE == "<u2"
        assert raw == bytes([0, 0, 1, 0, 0, 1, 232, 3])

    def test_the_recorded_frame_count_is_what_the_detector_produced(self, tmp_path: Path) -> None:
        cache = DetectionCache(session_dir=tmp_path)
        entry = cache.publish(KEY, a_result(probabilities=(0,) * 31))
        cache.commit()

        assert entry.frame_count == 31
        found = cache.get(KEY)
        assert found is not None
        assert found.frame_count == 31
