"""Sessions built to break one rule each.

:func:`~dnd_audio.fixtures.canonical_session` is the well-formed six-transmitter session
everything from M2 onward is tested against. These are the shapes that are *not* well
formed — one deviation apiece, so that when a test fails it is obvious which rule it was
about.

Each one is a real session directory with real audio, driven through the real CLI. That
matters more than it sounds: `tests/manifests.py` can construct any evidence a rule needs
without writing a byte, and is the right tool for the arithmetic, but it cannot prove that
`ffprobe` reports what the strategy chain expects, or that a refusal happens before the
first placement rather than after. Only a session on disk does that.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

from dnd_audio.fixtures.session import (
    ClapInterval,
    FixtureChunk,
    FixtureSession,
    FixtureTrack,
    SpeechInterval,
)

__all__ = [
    "DRIFT_END_SHIFT_SAMPLES",
    "drift_session",
    "drop_frame_session",
    "inconsistent_rate_session",
    "no_origin_session",
    "nonconforming_rate_session",
    "overlapping_session",
    "rollover_session",
]

_SECOND: Final = 48000

#: How far the end transient is moved on one track in :func:`drift_session`, in samples at
#: 48 kHz — 20 ms. Chosen to be far outside any plausible quantization and far inside the
#: default correlation lag window, so a detector that finds it is finding the transient
#: rather than the edge of its search.
DRIFT_END_SHIFT_SAMPLES: Final = 960


def _track(
    track_id: str,
    speaker: str,
    chunks: tuple[FixtureChunk, ...],
    *,
    tx_label: str = "TX01",
    receiver: str = "rx-a",
    channel: int = 1,
) -> FixtureTrack:
    return FixtureTrack(
        track_id=track_id,
        speaker_id=speaker,
        speaker_name=speaker.title(),
        receiver_id=receiver,
        receiver_channel=channel,
        tx_label=tx_label,
        chunks=chunks,
    )


def _two_tracks(
    first: tuple[FixtureChunk, ...], second: tuple[FixtureChunk, ...]
) -> tuple[FixtureTrack, ...]:
    return (
        _track("tx-a", "alice", first),
        _track("tx-b", "bob", second, tx_label="TX02", channel=2),
    )


def nonconforming_rate_session() -> FixtureSession:
    """One selected source at 44.1 kHz. Fatal before the timeline is built.

    The spec lists a non-48 kHz selected track among the fatal errors and M1 recorded it
    as a warning on purpose — refusing to *describe* a readable file would have lost the
    diagnostic that explains this failure.
    """
    return FixtureSession(
        session_id="nonconforming",
        title="Nonconforming rate",
        tracks=_two_tracks(
            (FixtureChunk(start_sample=0, n_samples=_SECOND, sequence=1),),
            (FixtureChunk(start_sample=0, n_samples=_SECOND, sequence=1, sample_rate=44100),),
        ),
    )


def inconsistent_rate_session() -> FixtureSession:
    """One track whose two chunks disagree about their sample rate.

    A different failure from the one above, and it gets a different message: a track that
    is entirely 44.1 kHz is a settings problem, and a track that is half of each is a
    capture-procedure problem.
    """
    return FixtureSession(
        session_id="inconsistent",
        title="Inconsistent rate",
        tracks=(
            _track(
                "tx-a",
                "alice",
                (
                    FixtureChunk(start_sample=0, n_samples=_SECOND, sequence=1),
                    FixtureChunk(
                        start_sample=_SECOND,
                        n_samples=_SECOND,
                        sequence=2,
                        sample_rate=44100,
                    ),
                ),
            ),
        ),
    )


def overlapping_session() -> FixtureSession:
    """Two chunks of one track claiming the same half-second.

    Far beyond any quantization tolerance — both starts come from `bext`, where the
    tolerance is a single sample — so it exercises `chunk_overlap_policy` rather than the
    automatic nudge.
    """
    return FixtureSession(
        session_id="overlapping",
        title="Overlapping chunks",
        tracks=(
            _track(
                "tx-a",
                "alice",
                (
                    FixtureChunk(start_sample=0, n_samples=2 * _SECOND, sequence=1),
                    FixtureChunk(start_sample=3 * _SECOND // 2, n_samples=_SECOND, sequence=2),
                ),
            ),
        ),
    )


def no_origin_session() -> FixtureSession:
    """No configured origin, so session zero is the earliest valid source start.

    The canonical fixture always states an origin, so this branch of ADR-0009 has no
    coverage without a fixture of its own. `tx-b` starts two seconds after `tx-a`, and
    `tx-a` must land at zero however far into the day the recording actually was.
    """
    return FixtureSession(
        session_id="no-origin",
        title="Derived origin",
        session_zero_timecode="19:00:00:00",
        origin_timecode=None,
        tracks=_two_tracks(
            (FixtureChunk(start_sample=0, n_samples=2 * _SECOND, sequence=1),),
            (FixtureChunk(start_sample=2 * _SECOND, n_samples=2 * _SECOND, sequence=1),),
        ),
    )


def rollover_session() -> FixtureSession:
    """A session that starts before midnight and continues after it.

    `tx-a` starts at 23:59:58 and `tx-b` two seconds later, which is 00:00:00 the next
    day. An implementation that did not unwrap would place `tx-b` 86 398 seconds *before*
    `tx-a`, which is not a subtle failure.
    """
    return FixtureSession(
        session_id="rollover",
        title="Across midnight",
        session_zero_timecode="23:59:58:00",
        tracks=_two_tracks(
            (FixtureChunk(start_sample=0, n_samples=_SECOND, sequence=1),),
            (FixtureChunk(start_sample=2 * _SECOND, n_samples=_SECOND, sequence=1),),
        ),
    )


def drop_frame_session() -> FixtureSession:
    """29.97 drop-frame, with every offset on an expressible frame boundary.

    At 30000/1001 fps a frame is 8008/5 samples at 48 kHz, so only frame indices that
    divide by five land on a whole sample. `19:00:00;00` is frame 2 049 948, which does
    not — hence `;02`, two frames later at 2 049 950, which does. The generator refuses
    anything else rather than rounding, and that refusal is what keeps the fixture's
    declared truth exact rather than approximately exact.

    Both tracks carry `INFO`/`ISMP` timecode tags, because a `bext` sample reference would
    not exercise the fractional-rate arithmetic at all.
    """
    step = 8008 * 5
    return FixtureSession(
        session_id="drop-frame",
        title="Drop frame",
        frame_rate="29.97DF",
        session_zero_timecode="19:00:00;02",
        tracks=_two_tracks(
            (
                FixtureChunk(
                    start_sample=0,
                    n_samples=step * 10,
                    sequence=1,
                    timecode_source="info_ismp",
                ),
            ),
            (
                FixtureChunk(
                    start_sample=step * 2,
                    n_samples=step * 10,
                    sequence=1,
                    timecode_source="info_ismp",
                ),
            ),
        ),
    )


def drift_session() -> FixtureSession:
    """Two tracks with shared transients at both ends, one of which arrives late.

    **The metadata is identical on both tracks.** Every chunk starts at the same sample and
    carries the same `bext` reference, so the timeline places them exactly together. What
    differs is the audio: `tx-b` hears the *end* clap 960 samples after `tx-a` does, while
    the start clap is simultaneous.

    That distinction is the whole point. Moving the metadata instead would let a drift test
    pass without the correlator ever detecting an acoustic lag change — it would be
    measuring the number the fixture already declared. Here the only way to find the drift
    is to correlate the samples.
    """
    length = 12 * _SECOND
    tracks = (
        _track("tx-a", "alice", (FixtureChunk(start_sample=0, n_samples=length, sequence=1),)),
        _track(
            "tx-b",
            "bob",
            (FixtureChunk(start_sample=0, n_samples=length, sequence=1),),
            tx_label="TX02",
            channel=2,
        ),
    )
    return FixtureSession(
        session_id="drift",
        title="Clock drift",
        tracks=tracks,
        claps=(
            # Heard by both at the same instant: the session starts in sync.
            ClapInterval(start_sample=_SECOND),
            # Heard by `tx-a` here...
            ClapInterval(start_sample=10 * _SECOND, tracks=("tx-a",)),
            # ...and by `tx-b` 20 ms later. Same event, two clocks.
            ClapInterval(start_sample=10 * _SECOND + DRIFT_END_SHIFT_SAMPLES, tracks=("tx-b",)),
        ),
        speech=(
            SpeechInterval(
                track_id="tx-a",
                start_sample=4 * _SECOND,
                n_samples=_SECOND,
                utterance_id="utt_drift_a",
                text="Testing, one two.",
            ),
        ),
    )


def dated_session(day: dt.date) -> FixtureSession:
    """A short two-track session on a stated calendar day, for date-driven placement."""
    return FixtureSession(
        session_id="dated",
        title="Dated",
        origin_date=day,
        tracks=_two_tracks(
            (FixtureChunk(start_sample=0, n_samples=_SECOND, sequence=1),),
            (FixtureChunk(start_sample=_SECOND, n_samples=_SECOND, sequence=1),),
        ),
    )
