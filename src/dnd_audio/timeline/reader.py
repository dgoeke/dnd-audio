"""The virtual track: bounded windows over a segment map (INV-07, ADR-0011).

This is what "a lossless 48 kHz floating-point working path" *is* in this pipeline. There
is no session-length array and, by default, no session-length file: a caller asks for a
window and gets one, assembled from whichever source chunks that window touches, with
silence wherever the transmitter was not recording.

Two properties the rest of the milestone leans on:

**A read outside a track's own extent is silence, not an error.** The session's aligned
duration is set by the latest track end, so a track that stopped early is still readable
up to that duration — and returning silence there is what lets every track answer to one
duration without the map having to contain invented audio.

**Memory is bounded by the window, not by the session.** :meth:`TrackReader.windows` is a
generator, so a consumer that streams stays bounded and a consumer that accumulates has to
do so visibly. `tests/test_memory.py` asserts the whole reader → resampler → writer path
writes before its last read, which nothing that buffers a session can satisfy.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import numpy as np
import numpy.typing as npt

from dnd_audio.artifacts.timeline import TimelineSegment, TimelineTrack
from dnd_audio.timeline.pcm import PcmReader, open_pcm

__all__ = ["DEFAULT_WINDOW_SAMPLES", "DerivativeReader", "TrackReader"]

#: One second at 48 kHz. Small enough that six open tracks cost megabytes rather than
#: gigabytes, large enough that per-window overhead is irrelevant against the read itself.
DEFAULT_WINDOW_SAMPLES: Final = 48000


class TrackReader:
    """Reads one reconstructed track, in bounded windows.

    Source files are opened lazily and kept open: a track with two chunks holds two
    handles, not one per window. Silence segments open nothing at all, which is why a
    four-hour gap costs nothing to read.
    """

    def __init__(self, session_dir: Path, track: TimelineTrack, duration_samples: int) -> None:
        self._session_dir = session_dir
        self._track = track
        self._duration = duration_samples
        self._audio: tuple[TimelineSegment, ...] = tuple(
            segment for segment in track.segments if segment.kind == "audio"
        )
        self._open: dict[str, PcmReader] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for reader in self._open.values():
            reader.close()
        self._open.clear()

    @property
    def track_id(self) -> str:
        return self._track.track_id

    @property
    def duration_samples(self) -> int:
        """The *session's* aligned duration, which every track answers to."""
        return self._duration

    def read(self, start_sample: int, n_samples: int) -> npt.NDArray[np.float32]:
        """``[start_sample, start_sample + n_samples)`` of this track, silence included.

        Silence covers three cases and they are deliberately indistinguishable to a
        caller: before the track started, inside a real gap, and after it stopped. All
        three mean "this transmitter recorded nothing here", which is the only thing a
        consumer can act on.
        """
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        window = np.zeros(n_samples, dtype=np.float32)
        end = start_sample + n_samples
        for segment in self._audio:
            overlap_start = max(segment.session_start_sample, start_sample)
            overlap_end = min(segment.session_end_sample, end)
            if overlap_end <= overlap_start:
                continue
            reader = self._reader_for(segment)
            offset = (segment.source_start_sample or 0) + (
                overlap_start - segment.session_start_sample
            )
            window[overlap_start - start_sample : overlap_end - start_sample] = reader.read(
                offset, overlap_end - overlap_start
            )
        return window

    def windows(
        self, *, window_samples: int = DEFAULT_WINDOW_SAMPLES, until: int | None = None
    ) -> Iterator[tuple[int, npt.NDArray[np.float32]]]:
        """Yield ``(start_sample, samples)`` over the whole track, in order.

        A generator rather than a list: the difference between streaming and materializing
        the session is exactly this keyword, and a list here would make every caller's
        memory bound the session's length no matter how carefully they wrote the rest.
        """
        if window_samples <= 0:
            message = f"window_samples must be positive, got {window_samples}"
            raise ValueError(message)
        total = self._duration if until is None else until
        position = 0
        while position < total:
            length = min(window_samples, total - position)
            yield position, self.read(position, length)
            position += length

    def _reader_for(self, segment: TimelineSegment) -> PcmReader:
        path = segment.source_relative_path
        if path is None:  # pragma: no cover - the artifact validator forbids it
            message = "an audio segment with no source path reached the reader"
            raise ValueError(message)
        reader = self._open.get(path)
        if reader is None:
            reader = PcmReader(open_pcm(self._session_dir / path))
            reader.__enter__()
            self._open[path] = reader
        return reader


class DerivativeReader:
    """Bounded reads of every track's cached 16 kHz derivative, one open handle per track.

    The grid the detector decided on (M3) and the grid ASR consumes (M4, ADR-0017), so both
    stages read it the same way and neither has its own copy of "find this track's derivative
    and open it". A context manager because both step over the same tracks repeatedly:
    reopening per read would be a syscall per comparison for no benefit, and leaking handles
    across six tracks and hundreds of candidates is its own failure.

    A track with no derivative is simply absent, and asking for one raises — that a track has
    16 kHz audio at all is a fact the caller established when it decided the track was usable.
    """

    def __init__(self, session_dir: Path, track_paths: Mapping[str, str]) -> None:
        self._session_dir = session_dir
        self._paths = dict(track_paths)
        self._open: dict[str, PcmReader] = {}

    @property
    def track_ids(self) -> frozenset[str]:
        return frozenset(self._paths)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for reader in self._open.values():
            reader.close()
        self._open.clear()

    def read(self, track_id: str, start: int, n_samples: int) -> npt.NDArray[np.float32]:
        """``[start, start + n)`` of one track's derivative, clamped to what exists.

        Clamped rather than trusted: the intervals handed here are already bounded by the
        session duration, so a read past the end means an assumption broke, and returning
        silence for the tail is a better failure than a traceback in the middle of a stage.
        """
        reader = self._open.get(track_id)
        if reader is None:
            reader = PcmReader(open_pcm(self._session_dir / self._paths[track_id]))
            reader.__enter__()
            self._open[track_id] = reader
        available = max(0, min(n_samples, reader.source.n_samples - start))
        if available <= 0:
            return np.zeros(0, dtype=np.float32)
        return reader.read(start, available)
