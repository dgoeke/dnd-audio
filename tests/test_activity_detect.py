"""Frames, coverage, and the five reshaping steps — each varied on its own.

`activity/detect.py` states the properties this file holds it to. Two of them are where a
subtle bug would survive long enough to reach a transcript:

* **The answer does not depend on the window.** Audio is read in bounded windows (INV-07)
  and a span crossing a boundary is clipped by the detector and rasterized in two pieces.
  If those pieces did not add up to what one pass produces, every region's edges would
  depend on a number chosen for memory reasons, and nothing downstream would attribute the
  drift to the reader. Proved the way `test_resample.py` proves the resampler's continuity:
  several partitionings, identical results.
* **The five reshaping steps are separate.** A test that switched them all on at once would
  pass while two of them did nothing, so each step below is varied alone against a baseline
  with the other four off, and is shown to change the output.

Expected values are stated as literals worked out from the frame arithmetic — 512 samples
at 16 kHz is 32 ms, so a millisecond setting becomes ``ceil(ms * 16 / 512)`` frames — rather
than recomputed with the code under test.

The audio detected over is silence. Every detector here is scripted or stubbed (INV-10), so
no assertion can depend on a sample value; only a file's length and its declared rate reach
one. Writing speech-shaped noise would imply otherwise, and INV-10 exists because that
implication is false.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.activity import DETECTOR_FRAME_SAMPLES
from dnd_audio.activity.detect import (
    PERMILLE,
    FrameProbabilities,
    SpeechRegion,
    assemble_regions,
    detect_track,
    frame_count,
    rasterize_spans,
)
from dnd_audio.config import VadConfig
from dnd_audio.fakes import ScriptedActivityDetector
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.fixtures.wav import write_wav
from dnd_audio.interfaces import AudioWindow, SpeechSpan
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

#: Written out rather than imported, so every expectation below is arithmetic a reader can
#: check by hand. Both are pinned against the package's own constants by
#: `test_the_frame_grid_is_the_one_these_expectations_assume`.
FRAME = 512
RATE = 16_000


def a_derivative(path: Path, n_samples: int, *, sample_rate: int = RATE) -> Path:
    """One mono float32 WAV of exactly ``n_samples``, at the derivative rate.

    The content is silence. The detectors in this file are scripted, so no sample value can
    reach an assertion — only the length and the declared rate can, and those are the two
    things `detect_track` reads the file for.
    """
    write_wav(path, np.zeros(n_samples, dtype=np.float32), sample_rate=sample_rate)
    return path


def vad(
    *,
    speech_threshold: float = 0.5,
    silence_threshold: float = 0.35,
    min_speech_ms: int = 0,
    min_silence_ms: int = 0,
    merge_gap_ms: int = 0,
    pad_ms: int = 0,
) -> VadConfig:
    """VAD settings with every reshaping step switched off, so a test can switch on one.

    Zero disables each of the two merges, the gap join, and the length filter: `_merge`
    joins neighbours strictly closer than the gap and nothing is closer than zero, and a
    region survives when it is at least zero frames long, which every region is.
    """
    return VadConfig(
        speech_threshold=speech_threshold,
        silence_threshold=silence_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        merge_gap_ms=merge_gap_ms,
        pad_ms=pad_ms,
    )


def regions_of(
    values: Sequence[int], settings: VadConfig, *, n_samples: int | None = None
) -> tuple[SpeechRegion, ...]:
    """Assemble from per-frame per-mille values. The track is whole frames unless stated."""
    probabilities = np.array(values, dtype=np.uint16)
    total = len(values) * FRAME if n_samples is None else n_samples
    return assemble_regions(probabilities, settings=settings, n_samples=total)


def bounds(regions: Sequence[SpeechRegion]) -> list[tuple[int, int]]:
    return [(region.start_sample, region.end_sample) for region in regions]


def assert_sorted_and_disjoint(regions: Sequence[SpeechRegion], *, n_samples: int) -> None:
    """A track's candidates are ordered, non-overlapping, and inside the track."""
    for earlier, later in itertools.pairwise(regions):
        assert earlier.end_sample <= later.start_sample
    for region in regions:
        assert 0 <= region.start_sample < region.end_sample <= n_samples


class StubFrameDetector:
    """A detector that reports its own per-frame probabilities.

    Written here rather than imported from `activity.silero`, which needs an ONNX runtime
    and a model file the default suite has neither of (INV-05). What it exists to prove is
    structural: a detector satisfying :class:`FrameProbabilities` has *its* array kept, and
    the rasterized reconstruction is not used.
    """

    def __init__(self, probabilities: Sequence[int], spans: Sequence[SpeechSpan] = ()) -> None:
        self._probabilities = np.array(probabilities, dtype=np.uint16)
        self._spans = tuple(spans)
        self.windows: list[tuple[int, int]] = []

    def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
        self.windows.append((window.start_sample, len(window)))
        return self._spans

    def frame_probabilities(self) -> npt.NDArray[np.uint16]:
        return self._probabilities


def test_the_frame_grid_is_the_one_these_expectations_assume() -> None:
    """Pins the constants every literal in this file was worked out from."""
    assert DETECTOR_FRAME_SAMPLES == FRAME
    assert DERIVATIVE_SAMPLE_RATE == RATE
    assert PERMILLE == 1000


class TestFrameCount:
    @pytest.mark.parametrize(
        ("n_samples", "expected"),
        [(0, 0), (1, 1), (511, 1), (512, 1), (513, 2), (1024, 2), (1025, 3), (16000, 32)],
    )
    def test_frames_cover_the_track_by_ceiling(self, n_samples: int, expected: int) -> None:
        """16 000 is there because one second of derivative audio is 31.25 frames, so 32."""
        assert frame_count(n_samples) == expected

    def test_a_partial_final_frame_is_kept_rather_than_dropped(self) -> None:
        """Naming the alternative, so the choice is visible rather than incidental.

        A floor rule silently loses the last 32 ms of every track whose length is not a
        multiple of the frame — always at the end, always in the same direction.
        """
        assert frame_count(513) == 2
        assert 513 // FRAME == 1

    def test_a_zero_length_track_has_no_frames(self) -> None:
        assert frame_count(0) == 0


class TestRasterizeSpansCreditsCoverage:
    """Coverage, not containment: a boundary frame reads what it is actually worth."""

    def test_a_fully_covered_frame_reads_full_scale(self) -> None:
        assert rasterize_spans((SpeechSpan(0, FRAME),), n_frames=1).tolist() == [1000]

    def test_a_frame_covered_exactly_half_reads_the_thresholds_tie_point(self) -> None:
        """500 is where the default `speech_threshold` sits, which is the point of it.

        The boundary frame lands on the value that decides it rather than being rounded
        outward to a whole frame, which is what makes a scripted fixture's chosen sample
        positions recoverable at all.
        """
        assert rasterize_spans((SpeechSpan(0, FRAME // 2),), n_frames=1).tolist() == [500]

    def test_containment_would_lose_the_boundary_frames_value(self) -> None:
        """Both wrong answers named: a span straddling two frames half-covers each."""
        raster = rasterize_spans((SpeechSpan(FRAME // 2, FRAME + FRAME // 2),), n_frames=2)
        assert raster.tolist() == [500, 500]
        # Containment would have read [0, 0] (neither frame is inside the span) or
        # [1000, 1000] (the span touches both), and neither says where the edges were.

    @pytest.mark.parametrize(
        ("end", "probability", "expected"),
        [(FRAME, 1.0, 1000), (FRAME, 0.25, 250), (FRAME // 2, 0.5, 250), (FRAME, 0.0, 0)],
    )
    def test_the_spans_probability_scales_its_coverage(
        self, end: int, probability: float, expected: int
    ) -> None:
        span = SpeechSpan(0, end, probability=probability)
        assert rasterize_spans((span,), n_frames=1).tolist() == [expected]

    def test_a_span_past_the_end_of_the_array_contributes_nothing(self) -> None:
        assert rasterize_spans((SpeechSpan(FRAME * 2, FRAME * 3),), n_frames=1).tolist() == [0]

    def test_a_span_before_the_offset_contributes_nothing(self) -> None:
        raster = rasterize_spans((SpeechSpan(0, FRAME),), n_frames=1, offset_samples=FRAME * 2)
        assert raster.tolist() == [0]

    def test_the_offset_places_frame_zero(self) -> None:
        """A window's chunk is rasterized on its own array, so frame zero moves with it."""
        spans = (SpeechSpan(FRAME * 2, FRAME * 3),)
        assert rasterize_spans(spans, n_frames=2, offset_samples=FRAME * 2).tolist() == [1000, 0]
        assert rasterize_spans(spans, n_frames=2, offset_samples=FRAME * 3).tolist() == [0, 0]

    def test_a_span_running_past_the_array_is_clipped_not_wrapped(self) -> None:
        raster = rasterize_spans((SpeechSpan(FRAME // 2, FRAME * 8),), n_frames=1)
        assert raster.tolist() == [500]

    def test_overlapping_spans_saturate_at_full_scale(self) -> None:
        """Two detectors' worth of certainty is still certainty, not 2000 per mille."""
        spans = (SpeechSpan(0, FRAME), SpeechSpan(0, FRAME))
        assert rasterize_spans(spans, n_frames=1).tolist() == [1000]

    def test_the_array_is_unsigned_16_bit(self) -> None:
        """Two bytes per 32 ms is what keeps the probability artifact bounded (INV-07)."""
        raster = rasterize_spans((SpeechSpan(0, FRAME),), n_frames=4)
        assert raster.dtype == np.uint16
        assert raster.shape == (4,)

    def test_no_spans_is_an_array_of_zeros(self) -> None:
        assert rasterize_spans((), n_frames=3).tolist() == [0, 0, 0]


class TestHysteresisKeepsAWordWhole:
    """Speech opens above `speech_threshold` and closes only below `silence_threshold`."""

    def test_a_dip_between_the_two_thresholds_does_not_split_a_word(self) -> None:
        """The wobble mid-syllable a single threshold would cut the word in half on.

        400 per mille is below the 500 that opens a region and above the 350 that closes
        one, so the region continues through it.
        """
        regions = regions_of([800, 800, 400, 800, 800], vad())
        assert bounds(regions) == [(0, 5 * FRAME)]
        assert regions[0].probability_permille == 720  # (800 + 800 + 400 + 800 + 800) / 5
        assert regions[0].peak_probability_permille == 800

    def test_narrowing_the_gap_splits_the_same_word(self) -> None:
        """The same probabilities, one setting moved: the dip now falls below silence."""
        regions = regions_of([800, 800, 400, 800, 800], vad(silence_threshold=0.45))
        assert bounds(regions) == [(0, 2 * FRAME), (3 * FRAME, 5 * FRAME)]

    def test_a_frame_below_the_speech_threshold_does_not_open_a_region(self) -> None:
        """The other half of the hysteresis: 400 continues speech but never starts it."""
        assert bounds(regions_of([400, 800], vad())) == [(FRAME, 2 * FRAME)]

    def test_a_frame_exactly_at_the_speech_threshold_opens_a_region(self) -> None:
        """500 is what a half-covered frame rasterizes to, so the tie has to be decided."""
        assert bounds(regions_of([500, 500], vad())) == [(0, 2 * FRAME)]

    def test_a_region_still_open_at_the_last_frame_is_closed_at_the_track_end(self) -> None:
        assert bounds(regions_of([0, 800, 800], vad())) == [(FRAME, 3 * FRAME)]


class TestMinSilenceMerges:
    """Step 1: a dip shorter than `min_silence_ms` is a stop consonant, not a turn."""

    #: 64 ms is 1024 derivative samples, which is 2 frames.
    TWO_FRAMES_MS = 64

    def test_a_dip_shorter_than_min_silence_does_not_split_a_word(self) -> None:
        regions = regions_of([800, 800, 0, 800, 800], vad(min_silence_ms=self.TWO_FRAMES_MS))
        assert bounds(regions) == [(0, 5 * FRAME)]
        assert regions[0].probability_permille == 640  # (800 * 4 + 0) / 5

    def test_the_same_dip_splits_the_word_with_the_step_switched_off(self) -> None:
        """The step varied on its own, so the merge above cannot be some other step's."""
        regions = regions_of([800, 800, 0, 800, 800], vad(min_silence_ms=0))
        assert bounds(regions) == [(0, 2 * FRAME), (3 * FRAME, 5 * FRAME)]

    def test_a_dip_as_long_as_min_silence_is_not_merged(self) -> None:
        """Two frames of silence against a two-frame threshold: the merge is strict."""
        regions = regions_of([800, 800, 0, 0, 800, 800], vad(min_silence_ms=self.TWO_FRAMES_MS))
        assert bounds(regions) == [(0, 2 * FRAME), (4 * FRAME, 6 * FRAME)]


class TestMinSpeechDrops:
    """Step 2: what is left shorter than `min_speech_ms` is a cough, not a word."""

    TWO_FRAMES_MS = 64

    def test_a_region_shorter_than_min_speech_is_dropped(self) -> None:
        regions = regions_of([800, 0, 800, 800, 0], vad(min_speech_ms=self.TWO_FRAMES_MS))
        assert bounds(regions) == [(2 * FRAME, 4 * FRAME)]

    def test_both_regions_survive_with_the_step_switched_off(self) -> None:
        regions = regions_of([800, 0, 800, 800, 0], vad(min_speech_ms=0))
        assert bounds(regions) == [(0, FRAME), (2 * FRAME, 4 * FRAME)]

    def test_the_threshold_is_a_whole_frame_coarser_than_the_millisecond(self) -> None:
        """A millisecond longer costs a whole frame, because the filter counts frames.

        32 ms is 512 samples, exactly one frame; 33 ms is 528, which rounds up to two. So a
        one-frame region survives the first and not the second — worth stating, because a
        reader tuning `min_speech_ms` in a session file is choosing frames whether they
        meant to or not (OQ-017).
        """
        assert bounds(regions_of([800, 0], vad(min_speech_ms=32))) == [(0, FRAME)]
        assert bounds(regions_of([800, 0], vad(min_speech_ms=33))) == []


class TestMergeGapJoins:
    """Step 3: one sentence, not eight fragments."""

    #: 96 ms is 1536 derivative samples, which is 3 frames.
    THREE_FRAMES_MS = 96

    def test_a_gap_shorter_than_merge_gap_becomes_one_candidate(self) -> None:
        regions = regions_of([800, 0, 0, 800], vad(merge_gap_ms=self.THREE_FRAMES_MS))
        assert bounds(regions) == [(0, 4 * FRAME)]
        assert regions[0].probability_permille == 400  # (800 + 0 + 0 + 800) / 4

    def test_the_previous_steps_leave_them_separate(self) -> None:
        """Two frames of gap outlives `min_silence_ms`, and is joined only by this step."""
        settings = vad(min_silence_ms=self.THREE_FRAMES_MS - 64, merge_gap_ms=0)
        assert bounds(regions_of([800, 0, 0, 800], settings)) == [
            (0, FRAME),
            (3 * FRAME, 4 * FRAME),
        ]

    def test_a_gap_as_long_as_merge_gap_is_left_alone(self) -> None:
        regions = regions_of([800, 0, 0, 0, 800], vad(merge_gap_ms=self.THREE_FRAMES_MS))
        assert bounds(regions) == [(0, FRAME), (4 * FRAME, 5 * FRAME)]


class TestPadding:
    """Steps 4 and 5: extend both ends, then re-merge whatever that pushed together."""

    #: 32 ms is 512 derivative samples, which is one frame's worth of padding.
    ONE_FRAME_MS = 32

    def test_padding_extends_both_ends(self) -> None:
        regions = regions_of([0, 0, 800, 0, 0], vad(pad_ms=self.ONE_FRAME_MS))
        assert bounds(regions) == [(FRAME, 4 * FRAME)]

    def test_padding_is_clamped_at_the_start_of_the_track(self) -> None:
        """A region opening on frame zero cannot be padded to a negative sample."""
        regions = regions_of([800, 0, 0], vad(pad_ms=self.ONE_FRAME_MS))
        assert bounds(regions) == [(0, 2 * FRAME)]

    def test_padding_is_clamped_at_the_end_of_the_track(self) -> None:
        """1400 samples is two and three-quarter frames, so the last frame is partial."""
        regions = regions_of([0, 0, 800], vad(pad_ms=self.ONE_FRAME_MS), n_samples=1400)
        assert bounds(regions) == [(FRAME, 1400)]

    def test_the_quoted_confidence_covers_the_padded_region(self) -> None:
        """Padding reaches into frames the detector called silence, and says so.

        Quoting the unpadded mean would make a candidate look more confident than the audio
        it actually contains, which is exactly the number M5 smooths a gain envelope from.
        """
        regions = regions_of([0, 0, 800, 0, 0], vad(pad_ms=self.ONE_FRAME_MS))
        assert regions[0].probability_permille == 267  # (0 + 800 + 0) / 3, half away from zero
        assert regions[0].peak_probability_permille == 800

    def test_padding_that_makes_two_regions_overlap_merges_them(self) -> None:
        """Otherwise a track's candidates would overlap, and M5 would mix one twice."""
        values = [800, 0, 800]
        assert bounds(regions_of(values, vad(pad_ms=0))) == [(0, FRAME), (2 * FRAME, 3 * FRAME)]
        assert bounds(regions_of(values, vad(pad_ms=self.ONE_FRAME_MS))) == [(0, 3 * FRAME)]

    @pytest.mark.parametrize("pad_ms", [0, 1, 30, 32, 100, 1000])
    def test_candidates_stay_disjoint_at_every_padding(self, pad_ms: int) -> None:
        """The property, rather than another table: disjointness is what M4 and M5 assume."""
        rng = np.random.default_rng(17)
        values = rng.integers(0, 1001, size=64).tolist()
        settings = vad(min_silence_ms=100, min_speech_ms=60, merge_gap_ms=200, pad_ms=pad_ms)
        n_samples = 64 * FRAME
        regions = assemble_regions(
            np.array(values, dtype=np.uint16), settings=settings, n_samples=n_samples
        )
        assert regions
        assert_sorted_and_disjoint(regions, n_samples=n_samples)


class TestAssembleRegionsEdges:
    def test_no_frame_above_the_threshold_is_no_candidates(self) -> None:
        assert regions_of([0, 100, 400], vad()) == ()

    def test_an_empty_probability_array_is_no_candidates(self) -> None:
        assert assemble_regions(np.zeros(0, dtype=np.uint16), settings=vad(), n_samples=0) == ()

    def test_a_region_reports_its_own_length(self) -> None:
        regions = regions_of([800, 800], vad())
        assert regions[0].n_samples == 2 * FRAME


class TestDetectTrackDoesNotDependOnTheWindow:
    """The partition-invariance proof for the span-based path (INV-07).

    The window bounds memory and nothing else. A detector's spans are clipped to whatever
    window they arrive in, so a candidate that straddles a boundary is rasterized in two
    pieces; if those did not sum to the one-pass answer, every region's edges would depend
    on a memory setting.
    """

    N_SAMPLES = 5000
    SPANS = (SpeechSpan(300, 1500), SpeechSpan(3000, 4800, probability=0.75))

    def run(self, path: Path, window_samples: int) -> tuple[tuple[SpeechRegion, ...], list[int]]:
        detector = ScriptedActivityDetector({"tx-a": self.SPANS})
        result = detect_track(
            path,
            track_id="tx-a",
            detector=detector,
            settings=vad(min_silence_ms=64, min_speech_ms=64, merge_gap_ms=96, pad_ms=32),
            window_samples=window_samples,
        )
        return result.regions, result.frame_probabilities.tolist()

    @pytest.mark.parametrize("window_samples", [1, 300, 512, 513, 700, 1024, 4096, 100_000])
    def test_every_window_gives_identical_regions_and_probabilities(
        self, window_samples: int, tmp_path: Path
    ) -> None:
        """100 000 is larger than the whole track; 300, 513 and 700 are not whole frames."""
        path = a_derivative(tmp_path / "tx-a.wav", self.N_SAMPLES)
        expected_regions, expected_probabilities = self.run(path, self.N_SAMPLES)
        assert expected_regions  # otherwise every partitioning agrees on nothing

        regions, probabilities = self.run(path, window_samples)
        assert regions == expected_regions
        assert probabilities == expected_probabilities

    def test_the_regions_are_where_the_scripted_spans_were(self, tmp_path: Path) -> None:
        """Stated independently, so "identical across windows" is not identically wrong.

        Span [300, 1500) covers frame 0 from 300 (212/512 = 414 per mille, below the 500
        that opens a region), frame 1 whole, and frame 2 up to 1500 (476/512 = 930). So the
        candidate runs from frame 1 to frame 3, padded by one frame either side.
        """
        path = a_derivative(tmp_path / "tx-a.wav", self.N_SAMPLES)
        regions, probabilities = self.run(path, 4096)
        assert probabilities[:4] == [414, 1000, 930, 0]
        assert bounds(regions) == [(0, 4 * FRAME), (5 * FRAME, self.N_SAMPLES)]
        assert_sorted_and_disjoint(regions, n_samples=self.N_SAMPLES)

    def test_a_window_is_rounded_up_to_whole_frames(self, tmp_path: Path) -> None:
        """A window that split a frame would make the frame grid depend on the window.

        Asserted on what the detector was actually handed: 300 samples was asked for, 512
        arrived, and the last window is short only because the track ends.
        """
        path = a_derivative(tmp_path / "tx-a.wav", 1200)
        detector = StubFrameDetector([0, 0, 0])
        detect_track(
            path,
            track_id="tx-a",
            detector=detector,
            settings=vad(),
            window_samples=300,
        )
        assert detector.windows == [(0, 512), (512, 512), (1024, 176)]


class TestDetectTrackContract:
    def test_a_derivative_at_the_wrong_rate_is_refused(self, tmp_path: Path) -> None:
        """48 kHz working audio has three times the samples per frame the detector expects.

        Detecting over it would place every candidate at a third of its real position, with
        the right length and no other symptom.
        """
        path = a_derivative(tmp_path / "tx-a.wav", 4800, sample_rate=48_000)
        with pytest.raises(ValueError, match="48000 Hz"):
            detect_track(
                path,
                track_id="tx-a",
                detector=ScriptedActivityDetector({}),
                settings=vad(),
                window_samples=4096,
            )

    @pytest.mark.parametrize(
        ("n_samples", "expected_frames"), [(0, 0), (1, 1), (512, 1), (513, 2), (5000, 10)]
    )
    def test_the_probability_array_covers_the_whole_track(
        self, n_samples: int, expected_frames: int, tmp_path: Path
    ) -> None:
        """One value per frame, always — a short array would silence what it is short by."""
        path = a_derivative(tmp_path / "tx-a.wav", n_samples)
        result = detect_track(
            path,
            track_id="tx-a",
            detector=ScriptedActivityDetector({"tx-a": (SpeechSpan(0, max(n_samples, 1)),)}),
            settings=vad(),
            window_samples=1024,
        )
        assert result.frame_probabilities.shape == (expected_frames,)
        assert result.frame_probabilities.dtype == np.uint16
        assert result.frame_samples == FRAME
        assert result.track_id == "tx-a"

    def test_a_detector_reporting_the_wrong_number_of_frames_is_an_error(
        self, tmp_path: Path
    ) -> None:
        """Not a warning: the missing frames would read as silence for the rest of the run."""
        path = a_derivative(tmp_path / "tx-a.wav", 5000)
        with pytest.raises(ValueError, match="does not cover the track"):
            detect_track(
                path,
                track_id="tx-a",
                detector=StubFrameDetector([1000] * 9),
                settings=vad(),
                window_samples=1024,
            )


class TestWhereTheProbabilitiesCameFrom:
    """A reader of the probability file has to know measurement from reconstruction."""

    def test_a_detector_with_frames_has_its_own_array_kept(self, tmp_path: Path) -> None:
        """The stub returns no spans at all, so a rasterized array would be all zeros.

        That is the whole difference: the regions below exist only because the detector's
        own probabilities were used rather than reconstructed from what it reported.
        """
        path = a_derivative(tmp_path / "tx-a.wav", 3 * FRAME)
        detector = StubFrameDetector([0, 900, 0])
        assert isinstance(detector, FrameProbabilities)

        result = detect_track(
            path, track_id="tx-a", detector=detector, settings=vad(), window_samples=1024
        )
        assert result.from_detector is True
        assert result.frame_probabilities.tolist() == [0, 900, 0]
        assert bounds(result.regions) == [(FRAME, 2 * FRAME)]

    def test_a_detector_without_frames_gets_the_rasterized_reconstruction(
        self, tmp_path: Path
    ) -> None:
        """Half of frame zero is covered, so it reads 500 rather than 0 or 1000."""
        path = a_derivative(tmp_path / "tx-a.wav", 2 * FRAME)
        detector = ScriptedActivityDetector({"tx-a": (SpeechSpan(0, FRAME // 2),)})
        assert not isinstance(detector, FrameProbabilities)

        result = detect_track(
            path, track_id="tx-a", detector=detector, settings=vad(), window_samples=1024
        )
        assert result.from_detector is False
        assert result.frame_probabilities.tolist() == [500, 0]


class TestAgainstTheFixturesGroundTruth:
    """The scripted detector driven from what the generator actually wrote.

    `leaky_activity_spans` rather than `activity_spans`: a real VAD fires on bleed as well
    as on speech, which is what gives a single track more than one candidate to be disjoint
    about. The spans are requested at the derivative rate — 48 kHz spans would land past the
    end of every window, the detector would return nothing, and every assertion here would
    pass over an empty result.
    """

    def test_each_tracks_candidates_are_sorted_disjoint_and_where_the_speech_was(
        self, canonical_fixture: FixtureTruth, tmp_path: Path
    ) -> None:
        spans = canonical_fixture.leaky_activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE)
        assert spans
        n_samples = max(span.end_sample for items in spans.values() for span in items) + RATE

        for track_id, track_spans in sorted(spans.items()):
            path = a_derivative(tmp_path / f"{track_id}.wav", n_samples)
            result = detect_track(
                path,
                track_id=track_id,
                detector=ScriptedActivityDetector(spans),
                settings=VadConfig(),
                window_samples=8192,
            )
            assert result.regions
            assert_sorted_and_disjoint(result.regions, n_samples=n_samples)
            for region in result.regions:
                assert any(
                    span.start_sample < region.end_sample and region.start_sample < span.end_sample
                    for span in track_spans
                ), f"{track_id} has a candidate at {region} where the fixture wrote no audio"

    def test_a_track_that_hears_bleed_gets_more_than_one_candidate(
        self, canonical_fixture: FixtureTruth, tmp_path: Path
    ) -> None:
        """Otherwise the disjointness above would be a property of a single region."""
        spans = canonical_fixture.leaky_activity_spans(sample_rate=DERIVATIVE_SAMPLE_RATE)
        n_samples = max(span.end_sample for items in spans.values() for span in items) + RATE
        crowded = max(spans, key=lambda track: len(spans[track]))
        assert len(spans[crowded]) > 1

        path = a_derivative(tmp_path / f"{crowded}.wav", n_samples)
        result = detect_track(
            path,
            track_id=crowded,
            detector=ScriptedActivityDetector(spans),
            settings=VadConfig(),
            window_samples=8192,
        )
        assert len(result.regions) > 1
        assert_sorted_and_disjoint(result.regions, n_samples=n_samples)
