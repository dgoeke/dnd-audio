"""The mix intermediate's cache identity, and the half of INV-08 that is easy to skip.

Two halves, and only the first is obvious.

**Identity.** Every component is asserted to be *present* rather than only that some change
produced some different hash, because — as M2's closeout puts it — a key that changes for the
right reason in one test can still be missing a component, and the missing one is always the
one that matters later. That is what `mix_identity_document` is separate from
`mix_identity` for.

**Completeness.** "An incomplete entry is never a hit" needs a *size* check, not merely the
presence of the file the sidecar names. A truncated float32 WAV reads as a shorter mix, which
is silence at the end of a session and nothing an operator would attribute to a cache. Four
ways an entry can be incomplete are exercised here; M5's plan review pointed out that the
first draft of the plan tested none of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.config import EnvelopeConfig, SessionConfig, stage_config_hash
from dnd_audio.mix import MIX_CACHE_DIRNAME
from dnd_audio.mix.cache import (
    CACHE_RECORD_VERSION,
    MixCache,
    mix_identity,
    mix_identity_document,
    mix_relative_path,
)
from dnd_audio.mix.levels import LevelCorrections, TrackCorrection, level_corrections
from tests.graphs import a_graph, a_track

TRACKS = ("tx-a", "tx-b", "tx-c")


@pytest.fixture
def graph() -> ActivityGraph:
    return a_graph(tracks=[a_track(track_id) for track_id in TRACKS])


@pytest.fixture
def corrections(graph: ActivityGraph) -> LevelCorrections:
    return level_corrections(graph, settings=EnvelopeConfig())


@pytest.fixture
def scope(valid_session_yaml: Path) -> str:
    import yaml

    raw = yaml.safe_load(valid_session_yaml.read_text(encoding="utf-8"))
    return stage_config_hash(SessionConfig.model_validate(raw), "mix")


def _identity(graph: ActivityGraph, corrections: LevelCorrections, scope: str, **kw: Any) -> str:
    return mix_identity(
        kw.pop("graph_override", graph),
        stage_config_hash=kw.pop("scope_override", scope),
        corrections=kw.pop("corrections_override", corrections),
        track_ids=kw.pop("track_ids", TRACKS),
    )


def _write_audio(path: Path, n_samples: int) -> None:
    from dnd_audio.timeline.wavwrite import WavWriter

    with WavWriter(path, sample_rate=48_000, n_samples=n_samples) as writer:
        writer.write(np.zeros(n_samples, dtype=np.float32))


class TestTheIdentityCarriesEverythingThatCanChangeASample:
    def test_the_document_names_every_component(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        """Asserted by name. A hash that changes for the right reason can still be missing a
        component, and the missing one is always the one that matters later."""
        document = mix_identity_document(
            graph, stage_config_hash=scope, corrections=corrections, track_ids=TRACKS
        )
        assert set(document) == {
            "attribution_cache_key",
            "cache_record_version",
            "config_hash",
            "corrections_mb",
            "duration_samples",
            "mix_semantics_version",
            "numpy_version",
            "sample_rate",
            "timeline_sha256",
            "track_ids",
        }

    def test_a_different_graph_is_a_different_mix(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        """The graph's own key covers every activity setting and every candidate decision."""
        moved = graph.model_copy(update={"attribution_cache_key": "d" * 64})
        assert _identity(graph, corrections, scope) != _identity(
            graph, corrections, scope, graph_override=moved
        )

    def test_a_moved_timeline_is_a_different_mix(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        """A placement fix moves a chunk without changing a source byte. A mix aligned to a
        timeline that has since moved is not obviously wrong when you listen to it."""
        moved = graph.model_copy(update={"timeline_sha256": "e" * 64})
        assert _identity(graph, corrections, scope) != _identity(
            graph, corrections, scope, graph_override=moved
        )

    def test_a_different_envelope_configuration_is_a_different_mix(
        self, graph: ActivityGraph, corrections: LevelCorrections, valid_session_yaml: Path
    ) -> None:
        import yaml

        raw = yaml.safe_load(valid_session_yaml.read_text(encoding="utf-8"))
        base = stage_config_hash(SessionConfig.model_validate(raw), "mix")
        raw.setdefault("mix", {})["envelope"] = {"release_ms": 250}
        changed = stage_config_hash(SessionConfig.model_validate(raw), "mix")
        assert base != changed

    def test_the_encode_settings_are_not_in_it(
        self, graph: ActivityGraph, corrections: LevelCorrections, valid_session_yaml: Path
    ) -> None:
        """ADR-0023's render boundary, as a property rather than a comment.

        The intermediate is unity master gain, so a loudness target, a bitrate or a retry
        budget cannot reach a single sample of it — and re-mixing six four-hour tracks to
        change a bitrate is the cost this split exists to avoid.
        """
        import yaml

        raw = yaml.safe_load(valid_session_yaml.read_text(encoding="utf-8"))
        base = stage_config_hash(SessionConfig.model_validate(raw), "mix")
        for change in (
            {"integrated_lufs": -20.0},
            {"true_peak_dbtp": -2.0},
            {"mp3_bitrate_kbps": 192},
            {"encode": {"max_retries": 1}},
        ):
            raw["mix"] = {**raw.get("mix", {}), **change}
            assert stage_config_hash(SessionConfig.model_validate(raw), "mix") == base, change

    def test_a_different_level_correction_is_a_different_mix(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        """The correction multiplies the audio here rather than being an encode parameter,
        so it belongs in the key that addresses the audio."""
        lifted = LevelCorrections(
            target_mbfs=corrections.target_mbfs,
            corrections=tuple(
                TrackCorrection(
                    track_id=item.track_id,
                    reference_mbfs=item.reference_mbfs,
                    correction_mb=item.correction_mb + 100,
                    clamped=False,
                )
                for item in corrections.corrections
            ),
            warnings=(),
        )
        assert _identity(graph, corrections, scope) != _identity(
            graph, corrections, scope, corrections_override=lifted
        )

    def test_a_different_track_set_is_a_different_mix(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        """The share divides between however many tracks there are, so dropping one changes
        every remaining track's gain."""
        assert _identity(graph, corrections, scope) != _identity(
            graph, corrections, scope, track_ids=("tx-a", "tx-b")
        )

    def test_a_different_track_order_is_a_different_mix(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        assert _identity(graph, corrections, scope) != _identity(
            graph, corrections, scope, track_ids=("tx-c", "tx-b", "tx-a")
        )

    def test_the_same_inputs_hash_the_same_way_twice(
        self, graph: ActivityGraph, corrections: LevelCorrections, scope: str
    ) -> None:
        assert _identity(graph, corrections, scope) == _identity(graph, corrections, scope)


class TestAnIncompleteEntryIsNeverAHit:
    """INV-08's other half. Every one of these leaves a *plausible-looking* cache."""

    @pytest.fixture
    def populated(self, tmp_path: Path) -> tuple[MixCache, str]:
        cache = MixCache(session_dir=tmp_path)
        key = "a" * 64
        audio = tmp_path / mix_relative_path(key)
        _write_audio(audio, 1000)
        cache.publish(key, sample_rate=48_000, n_samples=1000)
        cache.commit()
        return cache, key

    def test_a_complete_entry_is_a_hit(self, populated: tuple[MixCache, str]) -> None:
        """The contrast that makes every test below mean something."""
        cache, key = populated
        found = cache.get(key, expected_samples=1000)
        assert found is not None
        assert found.n_samples == 1000
        assert cache.hits == 1
        assert cache.misses == 0

    def test_a_truncated_intermediate_is_a_miss(
        self, populated: tuple[MixCache, str], tmp_path: Path
    ) -> None:
        """The one that needs a *size* check. A short float32 WAV reads as a mix that fades
        to silence at the end, which nobody would attribute to a cache."""
        cache, key = populated
        audio = tmp_path / mix_relative_path(key)
        audio.write_bytes(audio.read_bytes()[: 44 + 400])
        assert cache.get(key) is None
        assert cache.misses == 1

    def test_a_sidecar_whose_audio_is_gone_is_a_miss(
        self, populated: tuple[MixCache, str], tmp_path: Path
    ) -> None:
        cache, key = populated
        (tmp_path / mix_relative_path(key)).unlink()
        assert cache.get(key) is None

    def test_audio_with_no_sidecar_is_a_miss(self, tmp_path: Path) -> None:
        """The publication order's whole point: audio arrives first and is inert until the
        sidecar commits, so a run that failed INV-01 leaves nothing usable behind."""
        cache = MixCache(session_dir=tmp_path)
        key = "b" * 64
        _write_audio(tmp_path / mix_relative_path(key), 1000)
        cache.publish(key, sample_rate=48_000, n_samples=1000)
        cache.discard()
        assert cache.get(key) is None
        assert not (tmp_path / f"{MIX_CACHE_DIRNAME}/{key}.json").exists()

    def test_a_sidecar_naming_another_file_is_a_miss(
        self, populated: tuple[MixCache, str], tmp_path: Path
    ) -> None:
        """A record that disagrees with itself would grant a hit on the strength of a file
        nothing goes on to read: the caller reads the canonical path."""
        cache, key = populated
        sidecar = tmp_path / f"{MIX_CACHE_DIRNAME}/{key}.json"
        document = json.loads(sidecar.read_text())
        document["relative_path"] = "work/cache/mix/elsewhere.wav"
        sidecar.write_text(json.dumps(document))
        assert cache.get(key) is None

    def test_a_sidecar_from_an_older_record_shape_is_a_miss(
        self, populated: tuple[MixCache, str], tmp_path: Path
    ) -> None:
        cache, key = populated
        sidecar = tmp_path / f"{MIX_CACHE_DIRNAME}/{key}.json"
        document = json.loads(sidecar.read_text())
        document["cache_record_version"] = CACHE_RECORD_VERSION + 1
        sidecar.write_text(json.dumps(document))
        assert cache.get(key) is None

    def test_an_unparsable_sidecar_is_a_miss_rather_than_an_error(
        self, populated: tuple[MixCache, str], tmp_path: Path
    ) -> None:
        """A corrupted cache should cost time, not a session."""
        cache, key = populated
        (tmp_path / f"{MIX_CACHE_DIRNAME}/{key}.json").write_text("{ not json")
        assert cache.get(key) is None

    def test_a_length_the_caller_did_not_expect_is_a_miss(
        self, populated: tuple[MixCache, str]
    ) -> None:
        """The session's aligned duration is known before the cache is consulted, so a
        recorded length that disagrees with it describes a different session."""
        cache, key = populated
        assert cache.get(key, expected_samples=999) is None

    def test_reading_is_skipped_entirely_when_the_cache_is_disabled(
        self, populated: tuple[MixCache, str]
    ) -> None:
        """`--no-cache` distrusts what is stored. It still writes, so it costs one slow run
        rather than every run slow."""
        cache, key = populated
        cache.read_enabled = False
        assert cache.get(key) is None
        assert cache.misses == 1


class TestPublicationOrder:
    def test_nothing_reaches_disk_until_commit(self, tmp_path: Path) -> None:
        """An entry may only be published once INV-01 has been re-verified, and the mix is
        the one stage after inspection that reads *source* audio."""
        cache = MixCache(session_dir=tmp_path)
        key = "c" * 64
        _write_audio(tmp_path / mix_relative_path(key), 500)
        cache.publish(key, sample_rate=48_000, n_samples=500)

        assert not (tmp_path / f"{MIX_CACHE_DIRNAME}/{key}.json").exists()
        assert cache.commit() == 1
        assert (tmp_path / f"{MIX_CACHE_DIRNAME}/{key}.json").exists()

    def test_the_published_record_describes_the_file_on_disk(self, tmp_path: Path) -> None:
        cache = MixCache(session_dir=tmp_path)
        key = "d" * 64
        audio = tmp_path / mix_relative_path(key)
        _write_audio(audio, 777)
        record = cache.publish(key, sample_rate=48_000, n_samples=777)
        assert record.size_bytes == audio.stat().st_size
        assert record.relative_path == mix_relative_path(key)
        assert cache.audio_path(key) == audio

    def test_the_intermediate_lives_under_the_session_cache(self, tmp_path: Path) -> None:
        """The spec asks for it in `work/`, "not as a required user-facing deliverable"."""
        assert mix_relative_path("f" * 64).startswith(f"{MIX_CACHE_DIRNAME}/")
        assert MIX_CACHE_DIRNAME.startswith("work/cache/")
