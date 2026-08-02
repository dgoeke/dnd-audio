"""The timeline, checked against the fixture generator's independent ground truth.

`tests/test_origin.py` and `tests/test_layout.py` build manifests by hand, which makes
each rule testable in isolation and means every expectation was written by the same person
who wrote the rule. This file closes that loop: the fixture generator states, in samples
and before any file exists, where it is about to put every chunk, gap, and speech interval
(:class:`~dnd_audio.fixtures.FixtureTruth`). The pipeline then reads the *audio*, and the
two are compared.

Nothing here derives an expectation from the manifest or from the timeline. If the
generator and the timeline agree, twelve independent chains — WAV assembly, the `bext`
chunk, FFprobe, the RIFF walk, the strategy chain, rasterization, rollover, and layout —
all agreed about the same integer.
"""

from __future__ import annotations

import pytest

from dnd_audio.artifacts.manifest import Manifest
from dnd_audio.config import SessionConfig, load_session_config
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.runner import run_inspect
from dnd_audio.timeline.layout import TrackLayout, build_layout, reject_unusable_sources
from dnd_audio.timeline.origin import SessionOrigin, determine_origin


@pytest.fixture(scope="module")
def _built(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[FixtureTruth, Manifest, SessionConfig]:
    """Inspect the canonical fixture once; every test here reads the same manifest."""
    from dnd_audio.fixtures import build_session, canonical_session

    directory = tmp_path_factory.mktemp("truth")
    truth = build_session(canonical_session(), directory)
    result = run_inspect(directory)
    assert result.exit_code is ExitCode.OK
    return truth, result.manifest, load_session_config(directory / "session.yaml")


@pytest.fixture(scope="module")
def truth(_built: tuple[FixtureTruth, Manifest, SessionConfig]) -> FixtureTruth:
    return _built[0]


@pytest.fixture(scope="module")
def origin(_built: tuple[FixtureTruth, Manifest, SessionConfig]) -> SessionOrigin:
    _, manifest, config = _built
    reject_unusable_sources(manifest)
    return determine_origin(manifest, config)


@pytest.fixture(scope="module")
def layouts(
    _built: tuple[FixtureTruth, Manifest, SessionConfig], origin: SessionOrigin
) -> tuple[TrackLayout, ...]:
    _, manifest, config = _built
    built, _, _ = build_layout(manifest, config, origin)
    return built


class TestPlacementMatchesTheGeneratorsTruth:
    def test_session_zero_is_where_the_generator_put_it(
        self, truth: FixtureTruth, origin: SessionOrigin
    ) -> None:
        """19:00:00:00 on 2026-08-15, stated by the fixture before any audio was written."""
        assert origin.zero.source == "configured_origin"
        assert origin.zero.since_day_origin_samples == truth.session_zero_since_midnight

    def test_every_chunk_lands_on_its_declared_sample(
        self, truth: FixtureTruth, origin: SessionOrigin
    ) -> None:
        """Twelve chunks, six start offsets, two evidence strategies, exact agreement."""
        declared = {chunk.relative_path: chunk.start_sample for chunk in truth.chunks}
        placed = {start.relative_path: start.session_start_sample for start in origin.starts}
        assert placed == declared

    def test_the_track_carrying_an_ismp_timecode_places_like_the_rest(
        self, truth: FixtureTruth, origin: SessionOrigin
    ) -> None:
        """`tx-f` has no `bext` chunk at all — its timing is an INFO/ISMP tag.

        A different evidence kind, a different strategy, a different coordinate system on
        the way in, and the same integer on the way out.
        """
        declared = {chunk.relative_path: chunk.start_sample for chunk in truth.for_track("tx-f")}
        placed = {
            start.relative_path: start.session_start_sample for start in origin.by_track("tx-f")
        }
        assert placed == declared
        assert all(start.evidence.kind == "timecode" for start in origin.by_track("tx-f"))

    def test_no_rollover_is_inferred_for_an_evening_session(self, origin: SessionOrigin) -> None:
        assert all(start.cycles == 0 for start in origin.starts)
        assert not [note for note in origin.warnings if note.code == "midnight_rollover_inferred"]


class TestGapsMatchTheGeneratorsTruth:
    def test_the_only_gap_is_the_one_the_fixture_declared(
        self, truth: FixtureTruth, layouts: tuple[TrackLayout, ...]
    ) -> None:
        """`tx-c` stops at 5.0 s and resumes at 8.0 s, and nothing else has a hole."""
        found = tuple(
            (track.track_id, segment.session_start_sample, segment.session_end_sample)
            for track in layouts
            for segment in track.segments
            if segment.kind == "silence"
        )
        assert found == truth.gaps()

    def test_the_post_gap_chunk_did_not_slide_earlier(
        self, truth: FixtureTruth, layouts: tuple[TrackLayout, ...]
    ) -> None:
        """The spec's sentence, against ground truth.

        `tx-c`'s second chunk starts at 384000. An implementation that concatenated
        chunks would put it at 240000 — a three-second misalignment against five other
        tracks, and the fixture's post-gap speech is there precisely so a bug has
        something to get wrong.
        """
        declared = {chunk.relative_path: chunk.start_sample for chunk in truth.for_track("tx-c")}
        audio = [
            segment
            for track in layouts
            if track.track_id == "tx-c"
            for segment in track.segments
            if segment.kind == "audio"
        ]
        assert [segment.session_start_sample for segment in audio] == sorted(declared.values())

        concatenated = audio[0].n_samples
        assert audio[1].session_start_sample != concatenated


class TestAlignedDuration:
    def test_duration_is_the_latest_track_end_not_the_shortest(
        self, truth: FixtureTruth, layouts: tuple[TrackLayout, ...]
    ) -> None:
        """Criterion 4, within one 48 kHz sample — and it is exact here.

        The latest end is `tx-c`'s, *after* its gap; the shortest track ends 192000
        samples earlier. Taking the shortest, or the first, or the mean would each give a
        different wrong answer, so the assertion names the value from the generator's own
        chunk table rather than from any track in particular.
        """
        expected = max(chunk.start_sample + chunk.n_samples for chunk in truth.chunks)
        found = max(track.end_sample for track in layouts)
        assert abs(found - expected) <= 1
        assert found == expected

        shortest = min(track.end_sample for track in layouts if track.segments)
        assert shortest < found

    def test_every_tracks_extent_matches_its_own_chunks(
        self, truth: FixtureTruth, layouts: tuple[TrackLayout, ...]
    ) -> None:
        """No track is padded to the session duration; padding would invent audio."""
        for track in layouts:
            chunks = truth.for_track(track.track_id)
            assert track.start_sample == min(chunk.start_sample for chunk in chunks)
            assert track.end_sample == max(chunk.start_sample + chunk.n_samples for chunk in chunks)

    def test_the_session_spans_ten_and_a_half_seconds(
        self, layouts: tuple[TrackLayout, ...]
    ) -> None:
        """A sanity check on the fixture itself, so a silently empty run cannot pass.

        The second assertion is about the *tracks*, not about arithmetic: comparing
        `504000 / 48000` to `10.5` would be true whatever the code did.
        """
        assert max(track.end_sample for track in layouts) == 504000
        assert sum(len(track.segments) for track in layouts) == 13


class TestSegmentsPointAtRealAudio:
    def test_every_audio_segment_names_a_source_and_its_hash(
        self, truth: FixtureTruth, layouts: tuple[TrackLayout, ...]
    ) -> None:
        """The map has to be followable: a path, a hash, and an offset into the file."""
        hashes = {chunk.relative_path: chunk.sha256 for chunk in truth.chunks}
        for track in layouts:
            for segment in track.segments:
                if segment.kind != "audio":
                    continue
                assert segment.source_relative_path in hashes
                assert segment.source_sha256 == hashes[segment.source_relative_path]
                assert segment.source_start_sample == 0

    def test_segment_lengths_match_the_files_sample_counts(
        self, truth: FixtureTruth, layouts: tuple[TrackLayout, ...]
    ) -> None:
        lengths = {chunk.relative_path: chunk.n_samples for chunk in truth.chunks}
        for track in layouts:
            for segment in track.segments:
                if segment.kind != "audio":
                    continue
                assert segment.source_relative_path is not None
                assert segment.n_samples == lengths[segment.source_relative_path]

    def test_nothing_was_shifted(self, layouts: tuple[TrackLayout, ...]) -> None:
        """The canonical fixture has no overlaps, so every placed start is its evidence."""
        for track in layouts:
            for segment in track.segments:
                if segment.kind == "audio":
                    assert segment.shift_samples == 0
                    assert segment.evidence_start_sample == segment.session_start_sample
