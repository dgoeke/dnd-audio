"""The streamed mix: mono, exact length, and the samples the gains say it should be.

Two habits from earlier milestones' closeouts are followed deliberately.

**The expected samples are derived independently.** During a stretch where nobody is speaking
every share is `1/N` — a number this test computes from the track count, not from the
renderer — so the mix there is the plain mean of the six tracks, and the test reads those six
tracks itself. A test that re-derived the sum by calling the renderer's own helper could only
prove the sum it re-derived.

**Length and container are checked against the timeline, not against the file's own header.**
A short float32 WAV reads as silence at the end rather than as an error, which is precisely
the failure a cache must never serve (INV-08) and precisely what nobody would notice at the
end of a four-hour session.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.config import EnvelopeConfig
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.mix.envelope import EnvelopeStream
from dnd_audio.mix.levels import TrackCorrection, level_corrections
from dnd_audio.mix.render import render_mix
from dnd_audio.timeline import TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.pcm import PcmReader, open_pcm
from dnd_audio.timeline.reader import TrackReader
from dnd_audio.timeline.runner import run_ingest


@pytest.fixture
def reconstructed(canonical_fixture: FixtureTruth) -> tuple[Path, Timeline]:
    """The canonical session, reconstructed. `mix` composes `ingest`; this is its output."""
    result = run_ingest(canonical_fixture.session_dir)
    assert result.timeline is not None
    return canonical_fixture.session_dir, result.timeline


def _stream(
    graph: ActivityGraph, timeline: Timeline, settings: EnvelopeConfig | None = None
) -> tuple[EnvelopeStream, tuple[str, ...]]:
    resolved = settings or EnvelopeConfig()
    track_ids = tuple(track.track_id for track in timeline.tracks)
    return (
        EnvelopeStream(
            graph,
            settings=resolved,
            corrections=level_corrections(graph, settings=resolved),
            track_ids=track_ids,
        ),
        track_ids,
    )


def _read(path: Path) -> np.ndarray:
    with PcmReader(open_pcm(path)) as reader:
        return reader.read(0, reader.source.n_samples)


class TestTheRenderedMix:
    def test_it_is_mono_float32_at_the_session_rate_and_exactly_as_long(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        session_dir, timeline = reconstructed
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        summary = render_mix(
            tmp_path / "mix.wav",
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
        )

        assert summary.n_samples == timeline.duration_samples
        with PcmReader(open_pcm(summary.path)) as reader:
            assert reader.source.n_samples == timeline.duration_samples
            assert reader.source.sample_rate == timeline.sample_rate
        assert _read(summary.path).ndim == 1
        assert summary.path.stat().st_size == timeline.duration_samples * 4 + 44

    def test_silence_mixes_to_the_plain_mean_of_every_track(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """Where nobody is speaking, every share is `1/N` and the mix is the tracks' mean.

        `1/N` is computed here from the track count, and the six tracks are read here through
        `TrackReader` — neither number comes from the mixer. The window chosen is inside the
        fixture's leading silence, before any candidate opens.
        """
        session_dir, timeline = reconstructed
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        summary = render_mix(
            tmp_path / "mix.wav",
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
        )
        mixed = _read(summary.path)

        start, length = 100_000, 4_000
        expected = np.zeros(length, dtype=np.float64)
        for track_id in track_ids:
            track = next(t for t in timeline.tracks if t.track_id == track_id)
            with TrackReader(session_dir, track, timeline.duration_samples) as reader:
                expected += reader.read(start, length)
        expected /= len(track_ids)

        assert mixed[start : start + length] == pytest.approx(expected, abs=1e-6)

    def test_the_speaker_dominates_the_bleed_where_the_graph_says_so(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """The envelope's decision, audible in the samples rather than only in the gains.

        tx-a speaks at 249600 and four other lavs hear her. The mix during that stretch must
        be far closer to tx-a's own audio than to the sum of six.
        """
        session_dir, timeline = reconstructed
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        summary = render_mix(
            tmp_path / "mix.wav",
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
        )
        mixed = _read(summary.path)

        start, length = 260_000, 20_000
        alice = next(t for t in timeline.tracks if t.track_id == "tx-a")
        with TrackReader(session_dir, alice, timeline.duration_samples) as reader:
            own = reader.read(start, length)

        window = mixed[start : start + length]
        residual = window - own
        assert float(np.sqrt(np.mean(residual**2))) < 0.1 * float(np.sqrt(np.mean(own**2)))

    def test_the_reported_peak_is_the_files_own_peak(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """The first encode aims its gain with this number, so it must describe what was
        written rather than what was intended."""
        session_dir, timeline = reconstructed
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        summary = render_mix(
            tmp_path / "mix.wav",
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
        )
        assert summary.peak == pytest.approx(float(np.abs(_read(summary.path)).max()))
        assert 0.0 < summary.peak <= 1.0

    @pytest.mark.parametrize("window", [4_800, 48_000, 96_000, 480_000])
    def test_the_same_mix_comes_out_however_it_is_windowed(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
        window: int,
    ) -> None:
        """The property that makes a mix reproducible from its own inputs.

        The envelope's slew state and the interpolation's carried previous frame both cross
        window boundaries; either one held per-window would make the output depend on how the
        caller chose to loop, and a cached intermediate would then be keyed on something its
        identity does not contain.
        """
        session_dir, timeline = reconstructed
        reference = tmp_path / "reference.wav"
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        render_mix(
            reference,
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
        )

        other = tmp_path / f"window-{window}.wav"
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        render_mix(
            other,
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
            window_samples=window,
        )
        assert other.read_bytes() == reference.read_bytes()

    def test_rendering_twice_produces_identical_bytes(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """INV-02 on the artifact M5 actually caches."""
        session_dir, timeline = reconstructed
        outputs = []
        for name in ("first.wav", "second.wav"):
            envelope, track_ids = _stream(canonical_activity_graph, timeline)
            outputs.append(
                render_mix(
                    tmp_path / name,
                    session_dir=session_dir,
                    timeline=timeline,
                    track_ids=track_ids,
                    envelope=envelope,
                ).path.read_bytes()
            )
        assert outputs[0] == outputs[1]


class TestTheRendererRefusesWhatItCannotDoCorrectly:
    def test_a_window_that_is_not_whole_control_frames_is_refused(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        session_dir, timeline = reconstructed
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        with pytest.raises(ValueError, match="whole number of 48-sample control frames"):
            render_mix(
                tmp_path / "mix.wav",
                session_dir=session_dir,
                timeline=timeline,
                track_ids=track_ids,
                envelope=envelope,
                window_samples=1000,
            )

    def test_a_track_order_that_disagrees_with_the_envelope_is_refused(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """One wearer's gain applied to another's audio is inaudibly wrong, so it is fatal."""
        session_dir, timeline = reconstructed
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        with pytest.raises(ValueError, match="different orders"):
            render_mix(
                tmp_path / "mix.wav",
                session_dir=session_dir,
                timeline=timeline,
                track_ids=tuple(reversed(track_ids)),
                envelope=envelope,
            )

    def test_a_track_the_timeline_does_not_describe_is_refused(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """A graph track with no timeline track. Skipping it would change every other
        track's share with nothing saying so, so it is fatal rather than dropped.

        The envelope is built over the same list, so this reaches the timeline check rather
        than being caught by the order check above.
        """
        session_dir, timeline = reconstructed
        track_ids = (*[t.track_id for t in timeline.tracks], "tx-z")
        settings = EnvelopeConfig()
        measured = level_corrections(canonical_activity_graph, settings=settings)
        corrections = replace(
            measured,
            corrections=(
                *measured.corrections,
                TrackCorrection(
                    track_id="tx-z", reference_mbfs=None, correction_mb=0, clamped=False
                ),
            ),
        )
        envelope = EnvelopeStream(
            canonical_activity_graph,
            settings=settings,
            corrections=corrections,
            track_ids=track_ids,
        )
        with pytest.raises(ValueError, match="timeline does not describe"):
            render_mix(
                tmp_path / "mix.wav",
                session_dir=session_dir,
                timeline=timeline,
                track_ids=track_ids,
                envelope=envelope,
            )

    def test_an_interrupted_render_publishes_nothing(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        """`WavWriter` refuses to publish a short file, and the temp file does not survive.

        A truncated intermediate would be a cache entry that reads as silence at the end of a
        session, which is the exact shape INV-08's size check exists for.
        """
        session_dir, timeline = reconstructed
        settings = EnvelopeConfig()
        track_ids = tuple(track.track_id for track in timeline.tracks)
        corrections = level_corrections(canonical_activity_graph, settings=settings)

        class Truncated(EnvelopeStream):
            def chunks(self, *, chunk_frames: int):  # type: ignore[no-untyped-def]
                for index, chunk in enumerate(super().chunks(chunk_frames=chunk_frames)):
                    if index >= 2:
                        return
                    yield chunk

        destination = tmp_path / "mix.wav"
        with pytest.raises(Exception, match="declared"):
            render_mix(
                destination,
                session_dir=session_dir,
                timeline=timeline,
                track_ids=track_ids,
                envelope=Truncated(
                    canonical_activity_graph,
                    settings=settings,
                    corrections=corrections,
                    track_ids=track_ids,
                ),
            )
        assert not destination.exists()
        assert not list(tmp_path.glob(".*tmp"))


class TestTheTimelineIsUnchanged:
    def test_the_mix_reads_the_timeline_and_writes_nothing_to_it(
        self,
        reconstructed: tuple[Path, Timeline],
        canonical_activity_graph: ActivityGraph,
        tmp_path: Path,
    ) -> None:
        session_dir, timeline = reconstructed
        before = (session_dir / TIMELINE_RELATIVE_PATH).read_bytes()
        envelope, track_ids = _stream(canonical_activity_graph, timeline)
        render_mix(
            tmp_path / "mix.wav",
            session_dir=session_dir,
            timeline=timeline,
            track_ids=track_ids,
            envelope=envelope,
        )
        assert (session_dir / TIMELINE_RELATIVE_PATH).read_bytes() == before
