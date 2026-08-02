"""Chunk ordering, gap preservation, and the overlap policy (ADR-0010).

The property under test throughout is that **nothing is placed relative to the previous
chunk**. A transmitter switched off and back on must not pull later audio earlier, and the
way to be sure is to construct a track where a relative implementation and an absolute one
give different answers — a gap with audio after it — and assert the absolute answer.

The refusals are here too, because "fails *before* timeline construction" is a claim about
ordering that only a separate function can make good on.
"""

from __future__ import annotations

from typing import Any

import pytest

from dnd_audio.artifacts.manifest import ManifestSource
from dnd_audio.artifacts.timeline import TimelineDecision, TimelineNote, TimelineSegment
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.layout import (
    LayoutError,
    TrackLayout,
    build_layout,
    reject_unusable_sources,
)
from dnd_audio.timeline.origin import determine_origin
from tests.manifests import bwf, config, config_for, manifest, source, timecode

RATE = CANONICAL_SAMPLE_RATE
SECOND = RATE

Laid = tuple[tuple[TrackLayout, ...], tuple[TimelineDecision, ...], tuple[TimelineNote, ...]]


def lay_out(tracks: dict[str, list[ManifestSource]], **timecode_fields: Any) -> Laid:
    """Build a layout from hand-made sources, in the order the runner does it."""
    built = manifest(tracks)
    session = config_for(tuple(sorted(tracks)), **timecode_fields)
    reject_unusable_sources(built)
    origin = determine_origin(built, session)
    return build_layout(built, session, origin)


def segments_of(layouts: tuple[TrackLayout, ...], track_id: str) -> list[tuple[str, int, int]]:
    return [
        (s.kind, s.session_start_sample, s.n_samples) for s in _track(layouts, track_id).segments
    ]


def _audio(layouts: tuple[TrackLayout, ...], track_id: str) -> list[TimelineSegment]:
    return [s for s in _track(layouts, track_id).segments if s.kind == "audio"]


def _track(layouts: tuple[TrackLayout, ...], track_id: str) -> TrackLayout:
    for track in layouts:
        if track.track_id == track_id:
            return track
    raise AssertionError(f"no track {track_id}")


class TestOrdering:
    def test_chunks_are_sorted_by_parsed_start_not_by_filename(self) -> None:
        """The later chunk is named first, so filename order contradicts timecode order.

        Sorting by name would put `aaa` at zero and `zzz` five seconds later, reversing
        the recording. INV-12 forbids deriving timing from a filename, and this is what
        that forbids in practice.
        """
        layouts, _, _ = lay_out(
            {
                "tx-a": [
                    source(
                        "raw/tx-a/aaa.wav", bwf(19 * 3600 * RATE + 5 * SECOND), n_samples=SECOND
                    ),
                    source("raw/tx-a/zzz.wav", bwf(19 * 3600 * RATE), n_samples=SECOND),
                ]
            }
        )
        placed = segments_of(layouts, "tx-a")
        assert placed[0] == ("audio", 0, SECOND)
        assert [s.source_relative_path for s in _audio(layouts, "tx-a")] == [
            "raw/tx-a/zzz.wav",
            "raw/tx-a/aaa.wav",
        ]

    def test_a_chunks_expected_end_is_validated_against_the_next_start(self) -> None:
        """Contiguous chunks leave no silence and no overlap between them."""
        layouts, decisions, warnings = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=2 * SECOND),
                    source(
                        "raw/tx-a/two.wav", bwf(19 * 3600 * RATE + 2 * SECOND), n_samples=SECOND
                    ),
                ]
            }
        )
        assert segments_of(layouts, "tx-a") == [
            ("audio", 0, 2 * SECOND),
            ("audio", 2 * SECOND, SECOND),
        ]
        assert not warnings
        assert not [d for d in decisions if d.code.startswith("chunk_overlap")]


class TestGaps:
    def test_a_real_gap_becomes_an_explicit_silence_segment(self) -> None:
        layouts, decisions, _ = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=SECOND),
                    source(
                        "raw/tx-a/two.wav", bwf(19 * 3600 * RATE + 4 * SECOND), n_samples=SECOND
                    ),
                ]
            }
        )
        assert segments_of(layouts, "tx-a") == [
            ("audio", 0, SECOND),
            ("silence", SECOND, 3 * SECOND),
            ("audio", 4 * SECOND, SECOND),
        ]
        assert [d.code for d in decisions if d.code == "chunk_gap_preserved"] == [
            "chunk_gap_preserved"
        ]

    def test_later_audio_does_not_slide_earlier(self) -> None:
        """The spec's sentence, as an assertion.

        An implementation that concatenated chunks would put the second at 1 s. Its own
        evidence says 4 s, and that is where it goes — which is why the difference between
        the two implementations is 3 seconds of misalignment against every other track.
        """
        layouts, _, _ = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=SECOND),
                    source(
                        "raw/tx-a/two.wav", bwf(19 * 3600 * RATE + 4 * SECOND), n_samples=SECOND
                    ),
                ]
            }
        )
        second = _audio(layouts, "tx-a")[1]
        assert second.session_start_sample == 4 * SECOND
        concatenated = SECOND
        assert second.session_start_sample != concatenated

    def test_the_map_tiles_the_tracks_whole_extent(self) -> None:
        """No holes: a hole reads as either silence or a forgetful builder."""
        layouts, _, _ = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=SECOND),
                    source(
                        "raw/tx-a/two.wav", bwf(19 * 3600 * RATE + 4 * SECOND), n_samples=SECOND
                    ),
                ]
            }
        )
        track = _track(layouts, "tx-a")
        position = track.start_sample
        for segment in track.segments:
            assert segment.session_start_sample == position
            position += segment.n_samples
        assert position == track.end_sample


class TestOverlaps:
    def test_an_overlap_within_the_frame_tolerance_moves_the_later_chunk(self) -> None:
        layouts, decisions, warnings = lay_out(
            {
                "tx-a": [
                    # 1 s long, so it ends at 1 s; the next chunk's timecode says 0.96 s,
                    # a 1920-sample overlap — but both starts are frame-quantized at
                    # 25 fps, where a frame is 1920 samples, so it is exactly explainable.
                    source("raw/tx-a/one.wav", timecode("19:00:00:00", "25F"), n_samples=SECOND),
                    source("raw/tx-a/two.wav", timecode("19:00:00:24", "25F"), n_samples=SECOND),
                ]
            },
            frame_rate="25F",
        )
        assert segments_of(layouts, "tx-a") == [
            ("audio", 0, SECOND),
            ("audio", SECOND, SECOND),
        ]
        nudged = _audio(layouts, "tx-a")[1]
        assert nudged.evidence_start_sample == 24 * 1920
        assert nudged.shift_samples == SECOND - 24 * 1920
        assert not warnings
        assert [d.code for d in decisions if d.code == "chunk_overlap_quantization"]

    def test_a_material_overlap_is_fatal_by_default(self) -> None:
        """`reject` is the default, and it names both the overlap and the tolerance."""
        with pytest.raises(LayoutError, match="exceeds the") as caught:
            lay_out(
                {
                    "tx-a": [
                        source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=2 * SECOND),
                        source(
                            "raw/tx-a/two.wav",
                            bwf(19 * 3600 * RATE + SECOND),
                            n_samples=SECOND,
                        ),
                    ]
                }
            )
        assert caught.value.code == "chunk_overlap"
        assert "nudge_later" in str(caught.value)

    def test_nudge_later_keeps_every_sample(self) -> None:
        """The policy exists so no rule in this project ever discards audio."""
        layouts, _, warnings = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=2 * SECOND),
                    source("raw/tx-a/two.wav", bwf(19 * 3600 * RATE + SECOND), n_samples=SECOND),
                ]
            },
            chunk_overlap_policy="nudge_later",
        )
        audio = _audio(layouts, "tx-a")
        assert sum(segment.n_samples for segment in audio) == 3 * SECOND
        assert [s.session_start_sample for s in audio] == [0, 2 * SECOND]
        assert audio[1].evidence_start_sample == SECOND
        assert audio[1].shift_samples == SECOND
        assert [w.code for w in warnings] == ["chunk_overlap_nudged"]

    def test_a_later_chunk_keeps_its_own_evidence_position(self) -> None:
        """Nudging moves the overlapping chunk, not everything after it.

        A: [0,2), B's evidence [1,3), C's evidence [4,5). B is nudged to [2,4). C is *not*
        shifted — it lands at 4, where its own timecode says, because cross-track alignment
        matters more than this track's internal gap durations: a chunk whose timecode is
        good belongs where its timecode says, and moving it would misalign it against the
        five other transmitters.

        The visible consequence is that B's tail now occupies what used to be a gap. That
        is inherent to nudging — the alternatives are trimming B (forbidden) or shifting C
        (misaligning it) — and it is why ADR-0010's default is `reject`.
        """
        layouts, decisions, warnings = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/a.wav", bwf(19 * 3600 * RATE), n_samples=2 * SECOND),
                    source("raw/tx-a/b.wav", bwf(19 * 3600 * RATE + SECOND), n_samples=2 * SECOND),
                    source("raw/tx-a/c.wav", bwf(19 * 3600 * RATE + 4 * SECOND), n_samples=SECOND),
                ]
            },
            chunk_overlap_policy="nudge_later",
        )
        audio = _audio(layouts, "tx-a")
        assert [s.session_start_sample for s in audio] == [0, 2 * SECOND, 4 * SECOND]
        assert [s.shift_samples for s in audio] == [0, SECOND, 0]
        assert audio[2].evidence_start_sample == 4 * SECOND

        # Every sample survives, which is the property no policy may break.
        assert sum(s.n_samples for s in audio) == 5 * SECOND
        # Exactly one nudge is recorded, and it names the chunk that moved.
        nudged = [d for d in decisions if d.code == "chunk_overlap_nudged"]
        assert [d.subject for d in nudged] == ["raw/tx-a/b.wav"]
        assert [w.path for w in warnings] == ["raw/tx-a/b.wav"]

    def test_a_nudge_that_still_overlaps_cascades(self) -> None:
        """B is long enough that moving it collides with C, so C moves too."""
        layouts, _, warnings = lay_out(
            {
                "tx-a": [
                    source("raw/tx-a/a.wav", bwf(19 * 3600 * RATE), n_samples=2 * SECOND),
                    source("raw/tx-a/b.wav", bwf(19 * 3600 * RATE + SECOND), n_samples=3 * SECOND),
                    source("raw/tx-a/c.wav", bwf(19 * 3600 * RATE + 4 * SECOND), n_samples=SECOND),
                ]
            },
            chunk_overlap_policy="nudge_later",
        )
        audio = _audio(layouts, "tx-a")
        assert [s.session_start_sample for s in audio] == [0, 2 * SECOND, 5 * SECOND]
        assert sum(s.n_samples for s in audio) == 6 * SECOND
        assert len(warnings) == 2

    def test_the_tolerance_follows_the_evidence_not_the_session_rate(self) -> None:
        """Two sample-exact chunks overlapping by 1000 samples is a real overlap.

        The session is at 29.97, where a frame is 1602 samples, so a tolerance read from
        the *session* would wave this through. Read from the *pair*, it is refused.
        """
        with pytest.raises(LayoutError, match="1-sample tolerance"):
            lay_out(
                {
                    "tx-a": [
                        source("raw/tx-a/one.wav", bwf(19 * 3600 * RATE), n_samples=2 * SECOND),
                        source(
                            "raw/tx-a/two.wav",
                            bwf(19 * 3600 * RATE + 2 * SECOND - 1000),
                            n_samples=SECOND,
                        ),
                    ]
                },
                frame_rate="29.97F",
            )


class TestRefusalsHappenFirst:
    """Spec criterion 13: a bad source fails *before* timeline construction."""

    def test_a_44_1_khz_selected_source_is_fatal(self) -> None:
        built = manifest(
            {"tx-a": [source("raw/tx-a/one.wav", bwf(0, sample_rate=44100), sample_rate=44100)]}
        )
        with pytest.raises(LayoutError, match="44100 Hz") as caught:
            reject_unusable_sources(built)
        assert caught.value.code == "unsupported_sample_rate"

    def test_chunks_disagreeing_about_their_rate_are_fatal(self) -> None:
        """A capture-procedure problem, and it gets its own message.

        Checked before the flat 48 kHz rule so the operator is told the two chunks
        disagree rather than being told one of them is the wrong rate.
        """
        built = manifest(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(0)),
                    source("raw/tx-a/two.wav", bwf(0, sample_rate=44100), sample_rate=44100),
                ]
            }
        )
        with pytest.raises(LayoutError, match="disagree about their sample rate") as caught:
            reject_unusable_sources(built)
        assert caught.value.code == "inconsistent_sample_rate"

    def test_an_integer_pcm_source_is_refused_rather_than_rounded(self) -> None:
        """s32 cannot become float32 exactly, so it is named rather than silently lost."""
        built = manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(0), codec="pcm_s32le")]})
        with pytest.raises(LayoutError, match="pcm_f32le") as caught:
            reject_unusable_sources(built)
        assert caught.value.code == "undecodable_source"

    def test_a_stereo_source_is_refused(self) -> None:
        built = manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(0), channels=2)]})
        with pytest.raises(LayoutError, match="2-channel"):
            reject_unusable_sources(built)

    def test_a_source_with_no_sample_count_is_fatal(self) -> None:
        """Without a length its end is unknown, and timing is never invented (INV-12)."""
        built = manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(0), with_container=False)]})
        with pytest.raises(LayoutError, match="never successfully inspected") as caught:
            reject_unusable_sources(built)
        assert caught.value.code == "source_not_inspected"

    def test_an_unselected_bad_source_does_not_fail_the_session(self) -> None:
        """A duplicate or a stray at the wrong rate is described, not fatal.

        M1's closeout records the same asymmetry: making every candidate inspectable once
        turned one corrupt stray into a failed session. The refusals here are about the
        sources the timeline will actually read.
        """
        built = manifest(
            {
                "tx-a": [
                    source("raw/tx-a/one.wav", bwf(0)),
                    source(
                        "raw/tx-a/dup.wav",
                        bwf(0, sample_rate=44100),
                        sample_rate=44100,
                        role="duplicate",
                    ),
                ]
            }
        )
        reject_unusable_sources(built)


class TestInactiveTracks:
    def test_a_track_with_no_selected_sources_is_present_and_empty(self) -> None:
        """The roster is durable; an absent player is reported, not dropped."""
        built = manifest({"tx-a": [source("raw/tx-a/one.wav", bwf(0))], "tx-b": []})
        session = config()
        reject_unusable_sources(built)
        layouts, _, _ = build_layout(built, session, determine_origin(built, session))
        empty = next(track for track in layouts if track.track_id == "tx-b")
        assert empty.segments == ()
        assert empty.start_sample == empty.end_sample == 0
