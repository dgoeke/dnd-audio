"""The mono mix itself: six readers, one envelope, one streamed float32 file.

The spec forbids both easy answers — "do not concatenate speakers end-to-end, and do not sum
six full-volume channels" — and INV-07 forbids the third: this is the largest read in the
project, four hours of 48 kHz float32 across six transmitters, on a host where memory
pressure kills processes.

So the shape is one loop over windows, and everything inside it is bounded:

* every track's samples come from `TrackReader.read`, which assembles a window from whichever
  chunks it touches and returns silence for the rest (ADR-0011);
* the gains come from :class:`~dnd_audio.mix.envelope.EnvelopeStream` a chunk at a time, with
  its slew state carried across boundaries — never materialized, because 1 kHz of control
  frames over six tracks and four hours is 690 MB;
* the result goes straight into `timeline.wavwrite.WavWriter`, which streams and chooses RF64
  from the declared length rather than discovering the 4 GiB limit partway through.

`determinism.write_atomic` is deliberately unreachable from here: it holds its whole payload
in memory, which is right for JSON and a direct INV-07 violation for a waveform.

**The window is a whole number of control frames**, checked rather than assumed. A window that
ended mid-frame would make the interpolation's starting value depend on the caller's window
size, so the mix would stop being reproducible from its own inputs.

**The intermediate is unity master gain** (ADR-0023). Loudness normalization is an encode
parameter, so a true-peak retry costs one encode rather than one re-mix, and changing the
loudness target reuses this file outright.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.mix.envelope import EnvelopeStream, expand
from dnd_audio.timeline.reader import TrackReader
from dnd_audio.timeline.wavwrite import WavWriter

__all__ = ["DEFAULT_MIX_WINDOW", "MixSummary", "render_mix"]

#: One second at 48 kHz, matching the reader's own default. Six tracks of it is 1.2 MB of
#: float32, against 16 GB for a materialized four-hour session.
DEFAULT_MIX_WINDOW: Final = 48_000


@dataclass(frozen=True, slots=True)
class MixSummary:
    """What one render produced. Everything here is exact rather than measured."""

    path: Path
    sample_rate: int
    n_samples: int
    #: The largest absolute sample value written. Not a true peak — that needs oversampling
    #: and belongs to the decoded MP3 (ADR-0023) — but it is what the first encode's gain is
    #: aimed with, and it costs one reduction per window.
    peak: float


def render_mix(
    destination: Path,
    *,
    session_dir: Path,
    timeline: Timeline,
    track_ids: Sequence[str],
    envelope: EnvelopeStream,
    window_samples: int = DEFAULT_MIX_WINDOW,
) -> MixSummary:
    """Mix ``track_ids`` into one mono float32 file at unity master gain.

    Args:
        track_ids: The tracks to mix, in the order the envelope's gains are in. The
            envelope's own order is authoritative and is checked against this.
        envelope: A fresh stream. One instance produces one pass, so a caller re-rendering
            must build a new one.

    Raises:
        ValueError: if the window is not a whole number of control frames, or if the track
            order does not match the envelope's.
    """
    samples_per_frame = envelope.samples_per_frame
    if window_samples <= 0 or window_samples % samples_per_frame:
        message = (
            f"a mix window of {window_samples} samples is not a whole number of "
            f"{samples_per_frame}-sample control frames. A window ending mid-frame would make "
            f"the gain interpolation depend on the caller's window size."
        )
        raise ValueError(message)
    if tuple(track_ids) != envelope.track_ids:
        message = (
            f"the renderer was given tracks {tuple(track_ids)!r} and the envelope carries "
            f"{envelope.track_ids!r}. Mixing them in different orders applies one wearer's "
            f"gain to another's audio, which is inaudibly wrong."
        )
        raise ValueError(message)

    duration = timeline.duration_samples
    chunk_frames = window_samples // samples_per_frame
    peak = 0.0

    with (
        _TrackReaders(session_dir, timeline, track_ids, duration) as readers,
        WavWriter(destination, sample_rate=timeline.sample_rate, n_samples=duration) as out,
    ):
        position = 0
        for chunk in envelope.chunks(chunk_frames=chunk_frames):
            length = min(window_samples, duration - position)
            gains = expand(chunk, samples_per_frame=samples_per_frame, n_samples=length)
            mixed = np.zeros(length, dtype=np.float64)
            for index, reader in enumerate(readers):
                mixed += reader.read(position, length) * gains[:, index]
            block = mixed.astype(np.float32)
            peak = max(peak, float(np.abs(block).max(initial=0.0)))
            out.write(block)
            position += length

    return MixSummary(
        path=destination, sample_rate=timeline.sample_rate, n_samples=duration, peak=peak
    )


class _TrackReaders:
    """Every track's reader, opened together and closed together.

    A context manager rather than `ExitStack` so the failure mode is stated: a track named by
    the mix but absent from the timeline is a caller bug, not a track to skip, because
    dropping it would change every other track's share without anything saying so.
    """

    def __init__(
        self, session_dir: Path, timeline: Timeline, track_ids: Sequence[str], duration: int
    ) -> None:
        by_id = {track.track_id: track for track in timeline.tracks}
        missing = [track_id for track_id in track_ids if track_id not in by_id]
        if missing:
            message = (
                f"the mix names {', '.join(missing)}, which the timeline does not describe. "
                f"Dropping a track silently would change every other track's gain share."
            )
            raise ValueError(message)
        self._readers = [
            TrackReader(session_dir, by_id[track_id], duration) for track_id in track_ids
        ]

    def __enter__(self) -> list[TrackReader]:
        return self._readers

    def __exit__(self, *_: object) -> None:
        for reader in self._readers:
            reader.close()
