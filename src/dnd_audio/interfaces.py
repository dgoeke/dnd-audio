"""The seams models sit behind (INV-10).

``Transcriber`` and ``ActivityDetector`` are protocols so every stage above them is
testable without a model, a GPU, or a network. The production implementations land in
M6b and M3; :mod:`dnd_audio.fakes` provides the test ones.

Two decisions here that later milestones inherit:

**Audio is passed as a bounded window, never a session.** :class:`AudioWindow` carries a
start sample and the samples for that window only. A protocol that accepted "the track"
would invite an implementation to materialize four hours of float32 per track, which is
six full waveforms in RAM on a machine where memory pressure kills processes (INV-07).

**Times are integer samples, not seconds.** Everything internal stays exact (INV-04);
floats appear only when an artifact is serialized. A request therefore names its core
interval in samples, and results come back in samples for the caller to place on the
session timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "ActivityDetector",
    "AudioWindow",
    "SpeechSpan",
    "TranscribedWord",
    "Transcriber",
    "TranscriptionRequest",
    "TranscriptionResult",
]


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """A bounded slice of one track's working audio.

    Attributes:
        track_id: The authoritative track identity — the configured directory, never a
            filename component (INV-11).
        sample_rate: Samples per second of ``samples``. The working path is 48 kHz and
            the ASR/VAD derivative is 16 kHz, so this is never assumed.
        start_sample: Position of ``samples[0]`` on that track's timeline, at
            ``sample_rate``.
        samples: Mono float32. Mono because the mix is mono and each transmitter
            records one channel; a stereo array here means something upstream is wrong.
    """

    track_id: str
    sample_rate: int
    start_sample: int
    samples: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            message = f"sample_rate must be positive, got {self.sample_rate}"
            raise ValueError(message)
        if self.start_sample < 0:
            message = f"start_sample must not be negative, got {self.start_sample}"
            raise ValueError(message)
        if self.samples.ndim != 1:
            message = f"samples must be mono (1-D), got shape {self.samples.shape}"
            raise ValueError(message)

    @property
    def end_sample(self) -> int:
        """One past the last sample, on the same timeline as ``start_sample``."""
        return self.start_sample + int(self.samples.shape[0])

    def __len__(self) -> int:
        return int(self.samples.shape[0])


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """One segment submitted for transcription.

    ``audio`` includes padding for word recovery; ``core_start_sample`` and
    ``core_end_sample`` bound the unpadded interval this request actually owns. Words
    outside the core belong to a neighbouring request, which is how M4 stitches padded
    requests without duplicating words.
    """

    request_id: str
    audio: AudioWindow
    core_start_sample: int
    core_end_sample: int
    language: str = "English"
    #: Glossary text, when the session has one. Absence must not block a run.
    context: str | None = None
    max_new_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.core_end_sample <= self.core_start_sample:
            message = (
                f"request {self.request_id}: core interval "
                f"[{self.core_start_sample}, {self.core_end_sample}) is empty"
            )
            raise ValueError(message)
        if self.core_start_sample < self.audio.start_sample:
            message = (
                f"request {self.request_id}: core starts at {self.core_start_sample}, "
                f"before the padded window at {self.audio.start_sample}"
            )
            raise ValueError(message)
        if self.core_end_sample > self.audio.end_sample:
            message = (
                f"request {self.request_id}: core ends at {self.core_end_sample}, "
                f"past the padded window at {self.audio.end_sample}"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class TranscribedWord:
    """One word with sample-exact bounds on the requesting track's timeline."""

    start_sample: int
    end_sample: int
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """What a transcriber returned. No confidence field — see the transcript schema."""

    request_id: str
    text: str
    words: tuple[TranscribedWord, ...] = ()
    language: str = "English"
    #: The backend reported a length stop, or the text looks cut off at the generation
    #: ceiling. M4 responds by splitting the core and retrying, within a bounded count.
    truncated: bool = False


@runtime_checkable
class Transcriber(Protocol):
    """Turns a bounded audio request into text and word times.

    Implementations must not send audio anywhere (INV-06): local paths and in-memory
    arrays only.
    """

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...


@dataclass(frozen=True, slots=True)
class SpeechSpan:
    """A candidate speech region on one track, in samples.

    ``probability`` is the detector's own confidence, kept because a bad attribution is
    much easier to debug with the numbers that produced it than without.
    """

    start_sample: int
    end_sample: int
    probability: float = 1.0
    #: Detector-specific diagnostics, retained for the report.
    details: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_sample <= self.start_sample:
            message = f"span [{self.start_sample}, {self.end_sample}) is empty"
            raise ValueError(message)
        if not 0.0 <= self.probability <= 1.0:
            message = f"probability must be within [0, 1], got {self.probability}"
            raise ValueError(message)


@runtime_checkable
class ActivityDetector(Protocol):
    """Finds speech candidates in a bounded window of one track.

    Returns candidates, not decisions: attribution and bleed rejection happen above
    this seam, in the model-independent activity graph the mix consumes (INV-09).
    """

    def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]: ...
