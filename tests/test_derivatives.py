"""The derivative cache: complete identity, and no half-entry ever reads as a hit (INV-08).

A derivative is regenerable, so the only failure available is serving a stale one — and a
stale 16 kHz track is not obviously wrong. It has the right length and the right speech; it
is aligned to a timeline that has since moved. Every VAD span and word timestamp built on it
would be off by a constant nobody would attribute to a cache.

So each component of the identity is varied **independently** here. A key that happened to
change for the right reasons in one combined test would still be missing a component, and
the missing one is always the one that matters later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.artifacts.timeline import TimelineSegment, TimelineTrack
from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.derivatives import (
    DerivativeCache,
    derivative_identity,
    derivative_identity_document,
)
from dnd_audio.timeline.runner import run_ingest

CONFIG_HASH = "a" * 64
FILTER_ID = "b" * 64


def a_track(
    *,
    track_id: str = "tx-a",
    start: int = 0,
    n_samples: int = 48000,
    source: str = "raw/tx-a/one.wav",
    digest: str = "c" * 64,
) -> TimelineTrack:
    return TimelineTrack(
        track_id=track_id,
        speaker_id="alice",
        speaker_name="Alice",
        start_sample=start,
        end_sample=start + n_samples,
        segments=[
            TimelineSegment(
                kind="audio",
                session_start_sample=start,
                n_samples=n_samples,
                source_relative_path=source,
                source_sha256=digest,
                source_start_sample=0,
                evidence_start_sample=start,
            )
        ],
    )


def key_for(track: TimelineTrack, **overrides: object) -> str:
    settings: dict[str, object] = {
        "stage_config_hash": CONFIG_HASH,
        "target_rate": DERIVATIVE_SAMPLE_RATE,
        "filter_identity": FILTER_ID,
    }
    settings.update(overrides)
    return derivative_identity(track, **settings)  # type: ignore[arg-type]


class TestIdentityCoversEverythingThatChangesTheAudio:
    def test_the_same_inputs_give_the_same_key(self) -> None:
        assert key_for(a_track()) == key_for(a_track())

    def test_a_moved_chunk_changes_the_key(self) -> None:
        """The component the first draft omitted, and the one that matters most.

        A parser fix in M1 moves a chunk without changing a single source byte, a single
        byte of configuration, or any version number this project controls. Only the
        segment map records it.
        """
        assert key_for(a_track(start=0)) != key_for(a_track(start=48000))

    def test_a_different_source_file_changes_the_key(self) -> None:
        assert key_for(a_track()) != key_for(a_track(source="raw/tx-a/two.wav"))

    def test_changed_source_bytes_change_the_key(self) -> None:
        assert key_for(a_track()) != key_for(a_track(digest="d" * 64))

    def test_a_different_length_changes_the_key(self) -> None:
        assert key_for(a_track()) != key_for(a_track(n_samples=48001))

    def test_the_track_id_changes_the_key(self) -> None:
        """Two tracks with byte-identical audio are still two tracks."""
        assert key_for(a_track()) != key_for(a_track(track_id="tx-b"))

    def test_the_configuration_changes_the_key(self) -> None:
        assert key_for(a_track()) != key_for(a_track(), stage_config_hash="e" * 64)

    def test_the_filter_changes_the_key(self) -> None:
        """A redesigned filter must rebuild every derivative it ever produced."""
        assert key_for(a_track()) != key_for(a_track(), filter_identity="f" * 64)

    def test_the_target_rate_changes_the_key(self) -> None:
        """Or a track's 16 kHz and 48 kHz artifacts would collide at one path."""
        assert key_for(a_track()) != key_for(a_track(), target_rate=48000)

    def test_a_passthrough_and_a_resampled_artifact_differ(self) -> None:
        """The 48 kHz copy passes through no filter, so it carries no filter identity."""
        assert key_for(a_track(), target_rate=48000, filter_identity=None) != key_for(
            a_track(), target_rate=48000
        )

    @pytest.mark.parametrize(
        "component",
        [
            "numpy_version",
            "scipy_version",
            "inspection_semantics_version",
            "timeline_semantics_version",
            "cache_record_version",
            "segments",
            "track_extent",
        ],
    )
    def test_the_identity_carries_every_declared_component(self, component: str) -> None:
        """Asserted on the document, not inferred from hashes changing.

        NumPy and SciPy cannot be varied in-process, and a version that is simply absent
        would make every "different inputs, different key" test above pass anyway. The
        only way to know they are in there is to look.
        """
        document = derivative_identity_document(
            a_track(),
            stage_config_hash=CONFIG_HASH,
            target_rate=DERIVATIVE_SAMPLE_RATE,
            filter_identity=FILTER_ID,
        )
        assert component in document
        assert document[component]

    def test_the_document_is_what_gets_hashed(self) -> None:
        """So the two cannot drift apart and leave the assertions above meaningless."""
        document = derivative_identity_document(
            a_track(),
            stage_config_hash=CONFIG_HASH,
            target_rate=DERIVATIVE_SAMPLE_RATE,
            filter_identity=FILTER_ID,
        )
        assert key_for(a_track()) == sha256_bytes(canonical_json(document).encode("utf-8"))


class TestAnIncompleteEntryIsNeverAHit:
    """INV-08 states it outright, so each way of being incomplete is driven directly."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> DerivativeCache:
        return DerivativeCache(session_dir=tmp_path)

    def a_published_entry(self, cache: DerivativeCache, key: str = "k" * 64) -> str:
        audio = cache.audio_path(key, DERIVATIVE_SAMPLE_RATE)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00" * 128)
        cache.publish(key, target_rate=DERIVATIVE_SAMPLE_RATE, n_samples=32)
        cache.commit()
        return key

    def test_nothing_is_findable_until_commit(self, cache: DerivativeCache) -> None:
        """Publication stages; the caller commits after INV-01 has been re-verified.

        Until then the audio sits on disk with no sidecar naming it, which reads as a miss
        — so a run that discovers a source changed under it cannot leave behind an entry
        keyed on the bytes it *read* but built from the bytes that replaced them.
        """
        key = "k" * 64
        audio = cache.audio_path(key, DERIVATIVE_SAMPLE_RATE)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00" * 128)
        cache.publish(key, target_rate=DERIVATIVE_SAMPLE_RATE, n_samples=32)

        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None
        assert cache.commit() == 1
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is not None

    def test_discarding_leaves_the_audio_inert(self, cache: DerivativeCache) -> None:
        key = "k" * 64
        audio = cache.audio_path(key, DERIVATIVE_SAMPLE_RATE)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00" * 128)
        cache.publish(key, target_rate=DERIVATIVE_SAMPLE_RATE, n_samples=32)
        cache.discard()
        assert cache.commit() == 0
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None
        assert audio.exists()

    def test_a_sidecar_naming_another_file_is_a_miss(self, cache: DerivativeCache) -> None:
        """It would grant a hit on the strength of a file nothing goes on to read.

        The runner reads the *canonical* path, so a record pointing elsewhere — even at a
        real file of the right size — is a hit for the wrong reasons.
        """
        key = self.a_published_entry(cache)
        decoy = cache.audio_path("d" * 64, DERIVATIVE_SAMPLE_RATE)
        decoy.write_bytes(b"\xff" * 128)
        self._edit_sidecar(cache, key, relative_path=str(decoy.relative_to(cache.session_dir)))
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_a_sidecar_recording_another_rate_is_a_miss(self, cache: DerivativeCache) -> None:
        key = self.a_published_entry(cache)
        self._edit_sidecar(cache, key, sample_rate=44100)
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_a_record_shape_this_code_never_wrote_is_a_miss(self, cache: DerivativeCache) -> None:
        key = self.a_published_entry(cache)
        self._edit_sidecar(cache, key, cache_record_version=999)
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_a_length_the_caller_did_not_expect_is_a_miss(self, cache: DerivativeCache) -> None:
        """The caller knows how many samples this track should decimate to."""
        key = self.a_published_entry(cache)
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE, expected_samples=32) is not None
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE, expected_samples=33) is None

    @staticmethod
    def _edit_sidecar(cache: DerivativeCache, key: str, **changes: object) -> None:
        path = cache.session_dir / f"work/cache/audio/{DERIVATIVE_SAMPLE_RATE}/{key}.json"
        document = json.loads(path.read_text())
        document.update(changes)
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_a_complete_entry_is_a_hit(self, cache: DerivativeCache) -> None:
        key = self.a_published_entry(cache)
        found = cache.get(key, DERIVATIVE_SAMPLE_RATE)
        assert found is not None
        assert found.size_bytes == 128
        assert cache.hits == 1

    def test_a_missing_sidecar_is_a_miss(self, cache: DerivativeCache) -> None:
        """Audio written, the process died before the sidecar. The order that makes it safe."""
        key = "k" * 64
        audio = cache.audio_path(key, DERIVATIVE_SAMPLE_RATE)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00" * 128)
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None
        assert cache.misses == 1

    def test_a_sidecar_whose_audio_is_gone_is_a_miss(self, cache: DerivativeCache) -> None:
        key = self.a_published_entry(cache)
        cache.audio_path(key, DERIVATIVE_SAMPLE_RATE).unlink()
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_a_truncated_file_is_a_miss(self, cache: DerivativeCache) -> None:
        """The check that makes "incomplete is never a hit" true rather than intended.

        A short WAV reads as valid audio with silence at the end. Without comparing the
        size against what the sidecar recorded, the cache would serve it forever.
        """
        key = self.a_published_entry(cache)
        audio = cache.audio_path(key, DERIVATIVE_SAMPLE_RATE)
        audio.write_bytes(audio.read_bytes()[:64])
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_an_unparsable_sidecar_costs_time_not_a_session(self, cache: DerivativeCache) -> None:
        key = self.a_published_entry(cache)
        (cache.session_dir / f"work/cache/audio/{DERIVATIVE_SAMPLE_RATE}/{key}.json").write_text(
            "{not json", encoding="utf-8"
        )
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_a_sidecar_naming_a_different_key_is_a_miss(self, cache: DerivativeCache) -> None:
        key = self.a_published_entry(cache)
        sidecar = cache.session_dir / f"work/cache/audio/{DERIVATIVE_SAMPLE_RATE}/{key}.json"
        document = json.loads(sidecar.read_text())
        document["key"] = "z" * 64
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None

    def test_reading_can_be_disabled_without_disabling_writing(self, tmp_path: Path) -> None:
        """`--no-cache` distrusts what is stored; making it refuse to store too would
        turn "one slow run" into "every run slow"."""
        cache = DerivativeCache(session_dir=tmp_path, read_enabled=False)
        key = self.a_published_entry(cache)
        assert cache.get(key, DERIVATIVE_SAMPLE_RATE) is None
        assert cache.audio_path(key, DERIVATIVE_SAMPLE_RATE).exists()


class TestTheCacheIsActuallyConsulted:
    """A perfect key that nothing reads would pass every test above."""

    def test_a_second_run_reuses_the_derivatives(self, canonical_fixture: FixtureTruth) -> None:
        run_ingest(canonical_fixture.session_dir)
        derived = canonical_fixture.session_dir / "work/cache/audio/16000"
        before = {path: path.stat().st_mtime_ns for path in sorted(derived.glob("*.wav"))}

        # Overwrite the cached audio with something recognisably wrong. A run that
        # consults the cache serves these bytes back; one that ignores it rewrites them.
        for path in before:
            size = path.stat().st_size
            path.write_bytes(b"\x7f" * size)

        result = run_ingest(canonical_fixture.session_dir)
        assert result.exit_code is ExitCode.OK
        assert all(path.read_bytes() == b"\x7f" * path.stat().st_size for path in before)

    def test_a_changed_source_rebuilds_them(self, canonical_fixture: FixtureTruth) -> None:
        """Not a cache test so much as proof the identity reaches the audio.

        `--no-cache` rebuilds, and the rebuilt bytes must be the real ones again.
        """
        run_ingest(canonical_fixture.session_dir)
        derived = canonical_fixture.session_dir / "work/cache/audio/16000"
        poisoned = sorted(derived.glob("*.wav"))
        for path in poisoned:
            path.write_bytes(b"\x7f" * path.stat().st_size)

        run_ingest(canonical_fixture.session_dir, use_cache=False)
        for path in poisoned:
            assert path.read_bytes() != b"\x7f" * path.stat().st_size

    def test_the_recorded_key_matches_the_file_it_names(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The timeline's `cache_key` and the artifact's filename cannot drift apart."""
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        for track in result.timeline.tracks:
            for derivative in track.derivatives:
                assert derivative.relative_path.endswith(f"{derivative.cache_key}.wav")
                audio = canonical_fixture.session_dir / derivative.relative_path
                assert audio.stat().st_size == derivative.size_bytes


class TestPublishedAudioIsWhatWasRead:
    def test_the_derivative_holds_the_tracks_audio(self, canonical_fixture: FixtureTruth) -> None:
        """Decimated independently and compared, so the cache cannot hide a wrong file."""
        from dnd_audio.timeline.pcm import PcmReader, open_pcm
        from dnd_audio.timeline.reader import TrackReader
        from dnd_audio.timeline.resample import decimate_stream

        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        track = next(t for t in result.timeline.tracks if t.track_id == "tx-a")
        record = next(d for d in track.derivatives if d.sample_rate == DERIVATIVE_SAMPLE_RATE)

        duration = result.timeline.duration_samples
        with TrackReader(canonical_fixture.session_dir, track, duration) as reader:
            blocks = (block for _, block in reader.windows(window_samples=8192))
            expected = np.concatenate(list(decimate_stream(blocks, duration)))

        with PcmReader(open_pcm(canonical_fixture.session_dir / record.relative_path)) as stored:
            assert np.array_equal(stored.read(0, record.output_samples), expected)

    def test_two_tracks_never_share_a_file(self, canonical_fixture: FixtureTruth) -> None:
        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        paths = [
            derivative.relative_path
            for track in result.timeline.tracks
            for derivative in track.derivatives
        ]
        assert len(paths) == len(set(paths))
        assert len(paths) == sum(1 for track in result.timeline.tracks if track.segments)
