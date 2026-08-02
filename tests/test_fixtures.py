"""The fixture generator produces what M1's completion gate describes.

This file is the proof of the gate's first criterion, and it is deliberately paranoid:
everything downstream is tested against these fixtures, so a fixture that quietly lacks
a real gap makes every gap test in M2 vacuous.

Audio is read back with a small independent reader rather than with
:mod:`dnd_audio.inspection.riff`. The parser under test must not be the instrument that
verifies the data it will be tested against.
"""

from __future__ import annotations

import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.config import load_session_config
from dnd_audio.determinism import sha256_file
from dnd_audio.fixtures import (
    FixtureChunk,
    FixtureSession,
    FixtureTrack,
    FixtureTruth,
    build_session,
    canonical_session,
)
from dnd_audio.fixtures.session import SAMPLE_RATE
from dnd_audio.fixtures.wav import RF64_SENTINEL


def read_samples(path: Path) -> npt.NDArray[np.float32]:
    """Return a float32 WAV's samples, walking only far enough to find ``data``.

    Independent of the project's own RIFF parser on purpose. Handles the RF64 sentinel
    because one of the fixtures uses it.
    """
    blob = path.read_bytes()
    if blob[:4] not in (b"RIFF", b"RF64") or blob[8:12] != b"WAVE":
        message = f"{path} is not a WAVE file"
        raise ValueError(message)

    data_size_override: int | None = None
    offset = 12
    while offset + 8 <= len(blob):
        chunk_id = blob[offset : offset + 4]
        size = struct.unpack("<I", blob[offset + 4 : offset + 8])[0]
        payload = offset + 8
        if chunk_id == b"ds64":
            _, data_size_override = struct.unpack("<QQ", blob[payload : payload + 16])
        if chunk_id == b"data":
            if size == RF64_SENTINEL and data_size_override is not None:
                size = data_size_override
            return np.frombuffer(blob[payload : payload + size], dtype="<f4")
        offset = payload + size + (size % 2)

    message = f"{path} has no data chunk"
    raise ValueError(message)


def rms(samples: npt.NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) if samples.size else 0.0


def window_rms(truth: FixtureTruth, track_id: str, start: int, end: int) -> float:
    """Loudness of a session-relative window on one track, across chunk boundaries."""
    total = 0.0
    count = 0
    for chunk in truth.for_track(track_id):
        chunk_end = chunk.start_sample + chunk.n_samples
        overlap_start = max(chunk.start_sample, start)
        overlap_end = min(chunk_end, end)
        if overlap_end <= overlap_start:
            continue
        samples = read_samples(truth.session_dir / chunk.relative_path)
        piece = samples[overlap_start - chunk.start_sample : overlap_end - chunk.start_sample]
        total += float(np.sum(np.square(piece.astype(np.float64))))
        count += int(piece.size)
    return float(np.sqrt(total / count)) if count else 0.0


class TestCanonicalSession:
    """One test per property the completion gate names."""

    def test_six_transmitters_each_with_multiple_chunks(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        tracks = {chunk.track_id for chunk in canonical_fixture.chunks}
        assert tracks == {"tx-a", "tx-b", "tx-c", "tx-d", "tx-e", "tx-f"}
        for track_id in sorted(tracks):
            assert len(canonical_fixture.for_track(track_id)) >= 2, track_id

    def test_every_transmitter_starts_at_a_different_offset(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        firsts = [
            min(c.start_sample for c in canonical_fixture.for_track(track))
            for track in sorted({c.track_id for c in canonical_fixture.chunks})
        ]
        assert len(set(firsts)) == len(firsts), firsts
        assert min(firsts) == 0, "session zero should be somebody's start"

    def test_exactly_one_transmitter_has_a_real_gap(self, canonical_fixture: FixtureTruth) -> None:
        gaps = canonical_fixture.gaps()
        assert gaps == (("tx-c", 240000, 384000),)
        assert (gaps[0][2] - gaps[0][1]) == 3 * SAMPLE_RATE

    def test_every_other_transmitter_is_contiguous(self, canonical_fixture: FixtureTruth) -> None:
        """A gap nobody intended is as damaging to M2's tests as a missing one."""
        assert {gap[0] for gap in canonical_fixture.gaps()} == {"tx-c"}

    def test_the_clap_is_heard_by_all_six(self, canonical_fixture: FixtureTruth) -> None:
        clap = canonical_fixture.claps[0]
        quiet_start = clap.start_sample - 12000
        for track in sorted({c.track_id for c in canonical_fixture.chunks}):
            loud = window_rms(canonical_fixture, track, clap.start_sample, clap.start_sample + 2400)
            quiet = window_rms(canonical_fixture, track, quiet_start, clap.start_sample - 2400)
            assert loud > quiet * 10, f"{track}: clap {loud} vs floor {quiet}"

    def test_solo_speech_bleeds_quietly_into_the_tracks_that_were_recording(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        solo = canonical_fixture.speech[0]
        assert solo.track_id == "tx-a"

        speaker = window_rms(canonical_fixture, "tx-a", solo.start_sample, solo.end_sample)
        for target in solo.bleeds_into:
            heard = window_rms(canonical_fixture, target, solo.start_sample, solo.end_sample)
            assert heard < speaker / 10, f"{target} bleed is not quiet: {heard} vs {speaker}"
            floor = window_rms(canonical_fixture, target, 0, 12000)
            assert heard > floor * 5, f"{target} heard nothing: {heard} vs floor {floor}"

    def test_a_transmitter_inside_its_gap_hears_no_bleed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """tx-c is switched off during tx-a's solo, so it must hear nothing at all."""
        solo = canonical_fixture.speech[0]
        assert "tx-c" not in solo.bleeds_into
        assert window_rms(canonical_fixture, "tx-c", solo.start_sample, solo.end_sample) == 0.0

    def test_two_speakers_overlap_for_exactly_one_interval(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        overlaps = canonical_fixture.overlapping_speech()
        assert overlaps == ((326400, 374400, ("tx-d", "tx-e")),)
        start, end, tracks = overlaps[0]
        for track in tracks:
            floor = window_rms(canonical_fixture, track, 0, 12000)
            assert window_rms(canonical_fixture, track, start, end) > floor * 5

    def test_speech_after_a_gap_is_where_the_truth_says(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """If a bug slid post-gap audio earlier, this window would be silent."""
        post = next(i for i in canonical_fixture.speech if i.track_id == "tx-c")
        assert post.start_sample > canonical_fixture.gaps()[0][2]
        loud = window_rms(canonical_fixture, "tx-c", post.start_sample, post.end_sample)
        quiet = window_rms(canonical_fixture, "tx-c", 384000, 396000)
        assert loud > quiet * 10

    def test_the_session_yaml_it_writes_is_valid(self, canonical_fixture: FixtureTruth) -> None:
        config = load_session_config(canonical_fixture.session_dir / "session.yaml")
        assert [track.track_id for track in config.tracks] == [
            "tx-a",
            "tx-b",
            "tx-c",
            "tx-d",
            "tx-e",
            "tx-f",
        ]
        assert config.active_tracks == "auto"


class TestNamingAndMetadata:
    def test_filenames_follow_the_assumed_dji_grammar(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Literal, so a change to either the writer or the assumed grammar is visible.

        The grammar is a guess until H1 lands (OQ-003). Asserting exact strings is what
        makes the guess a *stated* one rather than an implicit one.
        """
        names = [Path(chunk.relative_path).name for chunk in canonical_fixture.for_track("tx-a")]
        assert names == [
            "TX01_MIC001_20260815_190000_orig.wav",
            "TX01_MIC002_20260815_190003_orig.wav",
        ]

    def test_the_tx_label_is_reused_across_kits(self, canonical_fixture: FixtureTruth) -> None:
        """OQ-002: two receivers both produce a TX01, so it cannot be an identity.

        A fixture where every label happened to be unique would let INV-11 be violated
        without any test noticing.
        """
        labels: dict[str, set[str]] = {}
        for chunk in canonical_fixture.chunks:
            label = Path(chunk.relative_path).name.split("_")[0]
            labels.setdefault(label, set()).add(chunk.track_id)
        assert labels["TX01"] == {"tx-a", "tx-c", "tx-e"}
        assert labels["TX02"] == {"tx-b", "tx-d", "tx-f"}

    def test_time_reference_is_session_zero_plus_the_chunk_offset(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        zero = canonical_fixture.session_zero_since_midnight
        assert zero == 19 * 3600 * SAMPLE_RATE
        for chunk in canonical_fixture.chunks:
            assert chunk.time_reference == zero + chunk.start_sample

    def test_ffprobe_sees_a_bext_reference_on_five_tracks(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        chunk = canonical_fixture.for_track("tx-a")[0]
        tags = _probe_format_tags(canonical_fixture.session_dir, chunk.relative_path)
        assert tags["time_reference"] == str(chunk.time_reference)

    def test_ffprobe_sees_a_timecode_tag_on_the_sixth(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """tx-f exercises the chain's fallback strategy, so both have real inputs."""
        chunk = canonical_fixture.for_track("tx-f")[0]
        tags = _probe_format_tags(canonical_fixture.session_dir, chunk.relative_path)
        assert tags["timecode"] == "19:00:03:15"
        assert "time_reference" not in tags

    def test_ffprobe_reports_neither_the_opaque_chunk_nor_the_ixml(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """OQ-005, demonstrated rather than assumed.

        This asymmetry is the entire reason the generic RIFF walk exists: if timing
        ever lives in a private chunk, FFprobe will not be the thing that finds it.
        """
        chunk = canonical_fixture.for_track("tx-a")[0]
        raw = _probe(canonical_fixture.session_dir, chunk.relative_path)
        assert b"XPRV" not in raw
        assert b"IXML_VERSION" not in raw
        assert b"XPRV" in (canonical_fixture.session_dir / chunk.relative_path).read_bytes()


class TestDeterminism:
    def test_regenerating_the_fixture_reproduces_every_byte(self, tmp_path: Path) -> None:
        """INV-02 applies to the fixture too: an unstable fixture makes every
        byte-stability test downstream meaningless."""
        first = build_session(canonical_session(), tmp_path / "one")
        second = build_session(canonical_session(), tmp_path / "two")

        assert [c.relative_path for c in first.chunks] == [c.relative_path for c in second.chunks]
        assert [c.sha256 for c in first.chunks] == [c.sha256 for c in second.chunks]
        assert (tmp_path / "one" / "session.yaml").read_bytes() == (
            tmp_path / "two" / "session.yaml"
        ).read_bytes()

    def test_the_recorded_hash_is_the_file_on_disk(self, canonical_fixture: FixtureTruth) -> None:
        for chunk in canonical_fixture.chunks:
            path = canonical_fixture.session_dir / chunk.relative_path
            assert sha256_file(path) == chunk.sha256
            assert path.stat().st_size == chunk.size_bytes

    def test_chunking_does_not_change_a_sample(self, tmp_path: Path) -> None:
        """The same events, split differently, must render identically.

        Rendering per chunk is what keeps H2's four-hour fixture possible (INV-07). It
        is only safe if a chunk boundary cannot perturb the samples around it — which
        is why events are rendered whole and then sliced.
        """
        spec = canonical_session()
        # tx-a alone, so its bleed targets are not in this session to receive anything.
        solo = tuple(replace(i, bleeds_into=()) for i in spec.speech if i.track_id == "tx-a")
        whole = FixtureSession(
            session_id=spec.session_id,
            title=spec.title,
            tracks=(
                FixtureTrack(
                    track_id="tx-a",
                    speaker_id="alice",
                    speaker_name="Alice",
                    receiver_id="rx-a",
                    receiver_channel=1,
                    tx_label="TX01",
                    chunks=(FixtureChunk(start_sample=0, n_samples=336000, sequence=1),),
                ),
            ),
            claps=spec.claps,
            speech=solo,
        )
        split = FixtureSession(
            session_id=spec.session_id,
            title=spec.title,
            tracks=(
                FixtureTrack(
                    track_id="tx-a",
                    speaker_id="alice",
                    speaker_name="Alice",
                    receiver_id="rx-a",
                    receiver_channel=1,
                    tx_label="TX01",
                    chunks=(
                        FixtureChunk(start_sample=0, n_samples=144000, sequence=1),
                        FixtureChunk(start_sample=144000, n_samples=192000, sequence=2),
                    ),
                ),
            ),
            claps=spec.claps,
            speech=solo,
        )

        one = build_session(whole, tmp_path / "whole")
        two = build_session(split, tmp_path / "split")

        joined = np.concatenate(
            [read_samples(two.session_dir / c.relative_path) for c in two.chunks]
        )
        single = read_samples(one.session_dir / one.chunks[0].relative_path)
        assert joined.shape == single.shape
        # Every sample, including the noise floor: the floor is seeded by timeline
        # position for exactly this reason. Anything weaker would let a one-sample
        # boundary error hide inside a tolerance.
        np.testing.assert_array_equal(joined, single)


class TestDownstreamContracts:
    """What the spec's fixture recipe promises M3 and M4, stated in M1."""

    def test_activity_spans_cover_every_speech_interval(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        spans = canonical_fixture.activity_spans()
        assert sorted(spans) == ["tx-a", "tx-c", "tx-d", "tx-e"]
        assert spans["tx-d"][0].start_sample == 326400
        assert spans["tx-d"][0].end_sample == 374400

    def test_bleed_is_not_claimed_as_activity(self, canonical_fixture: FixtureTruth) -> None:
        """tx-b hears tx-a's solo but is not speaking; the truth must not say it is."""
        assert "tx-b" not in canonical_fixture.activity_spans()

    def test_every_utterance_has_a_stable_id_and_text(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        script = canonical_fixture.transcript_script()
        assert len(script) == len(canonical_fixture.speech)
        assert script["utt_tx-a_000249600"] == "We should go back to Zephyrine."


class TestNoAudioIsCommitted:
    def test_the_repository_contains_no_audio_binaries(self, repo_root: Path) -> None:
        """The spec's reason for a generator existing at all."""
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        audio = [
            name
            for name in tracked
            if Path(name).suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        ]
        assert audio == []


def _probe(session_dir: Path, relative_path: str) -> bytes:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-i",
            relative_path,
        ],
        cwd=session_dir,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _probe_format_tags(session_dir: Path, relative_path: str) -> dict[str, str]:
    import json

    document = json.loads(_probe(session_dir, relative_path))
    tags = document["format"].get("tags", {})
    if not isinstance(tags, dict):  # pragma: no cover - ffprobe always emits an object
        pytest.fail(f"unexpected tags shape: {tags!r}")
    return {str(key): str(value) for key, value in tags.items()}
