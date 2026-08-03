"""Deterministic test implementations of the model seams (INV-10).

Both fakes are *scripted*: they return what the test told them to return. That is a
deliberate choice over a clever fake that derives plausible output from the audio.
M4 needs to exercise truncation, alignment failure, overlapping utterances, and
matching short utterances on two tracks — cases that only exist if the test can state
them. A fake that invents its own answers cannot be asked for a specific one, and its
"determinism" only ever proves that a hash function is a function.

Neither fake imports a model, touches the network, or needs a GPU. The spec's warning
applies to the real detector rather than these: synthetic speech-shaped noise must
never be expected to trigger a particular learned Silero release.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from dnd_audio.artifacts.activity import DetectorIdentity
from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.interfaces import (
    AudioWindow,
    SpeechSpan,
    TranscribedWord,
    TranscriptionRequest,
    TranscriptionResult,
)

__all__ = ["ScriptedActivityDetector", "ScriptedTranscriber", "SessionScriptTranscriber"]


class ScriptedTranscriber:
    """A :class:`~dnd_audio.interfaces.Transcriber` that returns scripted results.

    Args:
        responses: Keyed by ``request_id``. A request with no scripted response is an
            error rather than a silent empty result — a test that transcribes something
            it did not plan for has a bug, and returning "" would hide it.

    The requests it received are kept in :attr:`requests`, in call order, so a test can
    assert on what was asked as well as what came back — that no padded waveform
    exceeded ``max_segment_s``, for instance.
    """

    def __init__(self, responses: Mapping[str, TranscriptionResult]) -> None:
        self._responses = dict(responses)
        self.requests: list[TranscriptionRequest] = []

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.requests.append(request)
        try:
            return self._responses[request.request_id]
        except KeyError:
            known = ", ".join(sorted(self._responses)) or "(none)"
            message = f"no scripted response for request {request.request_id!r}; scripted: {known}"
            raise KeyError(message) from None


@dataclass(frozen=True, slots=True)
class ScriptedUtterance:
    """One thing a session's declared script says a track can be heard saying.

    Positions are on the canonical 48 kHz session grid, because that is what the fixture
    declared; the transcriber converts them to whatever grid the request it is answering is on.
    """

    track_id: str
    start_sample: int
    end_sample: int
    text: str
    words: tuple[tuple[int, int, str], ...] = ()


class SessionScriptTranscriber:
    """A :class:`~dnd_audio.interfaces.Transcriber` driven by a session's declared script.

    Still scripted, and still not derived from audio: it answers a request with the utterances
    the *fixture* declared for that track over that window, not with anything it heard. What it
    adds over :class:`ScriptedTranscriber` is that it can be asked about a request whose id
    nobody knew in advance, which is what a whole-session run needs (ADR-0018).

    It returns everything overlapping the **padded** window, exactly as a real model would —
    the model has no idea which part of what it heard the pipeline owns. Dropping the words
    outside the ownership interval is the pipeline's job, and handing this fake a
    pre-trimmed answer would quietly stop testing that.
    """

    def __init__(self, utterances: Sequence[ScriptedUtterance], *, sample_rate: int) -> None:
        self._utterances = tuple(utterances)
        self._sample_rate = sample_rate
        self.requests: list[TranscriptionRequest] = []

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.requests.append(request)
        scale = self._sample_rate // request.audio.sample_rate
        found = [
            utterance
            for utterance in self._utterances
            if utterance.track_id == request.audio.track_id
            and utterance.start_sample // scale < request.audio.end_sample
            and -(-utterance.end_sample // scale) > request.audio.start_sample
        ]
        words = tuple(
            TranscribedWord(
                start_sample=start // scale,
                end_sample=max(-(-end // scale), start // scale + 1),
                text=text,
            )
            for utterance in found
            for start, end, text in utterance.words
        )
        return TranscriptionResult(
            request_id=request.request_id,
            text=" ".join(utterance.text for utterance in found),
            words=words,
            language=request.language,
            alignment_status="aligned" if words else "not_attempted",
        )


class ScriptedActivityDetector:
    """An :class:`~dnd_audio.interfaces.ActivityDetector` with a ground-truth mask.

    Args:
        spans: Ground-truth speech regions per ``track_id``, in that track's samples.

    Spans are clipped to the requested window rather than returned whole, because a
    real detector only sees the window it was given. A caller that reads a session in
    bounded windows (INV-07) must get the same answer as one that reads it at once, and
    clipping is what makes that true.
    """

    def __init__(self, spans: Mapping[str, Sequence[SpeechSpan]]) -> None:
        self._spans = {track: tuple(track_spans) for track, track_spans in spans.items()}

    def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
        return tuple(self._clip(self._spans.get(window.track_id, ()), window))

    def identity(self) -> DetectorIdentity:
        """What this detector is, for a cache key that has to tell two of them apart.

        The digest covers the whole script. Two scripted detectors with different spans are
        different detectors, and a cache that could not distinguish them would serve one
        test's answers to another — which is the shape of bug that makes a suite pass while
        the thing it tests is broken.

        No model hash, no runtime, and no interface: this one runs no model, and a fabricated
        interface would make the identity claim something untrue about how it was called.
        """
        script = canonical_json(
            {
                track: [
                    [span.start_sample, span.end_sample, span.probability]
                    for span in sorted(spans, key=lambda s: (s.start_sample, s.end_sample))
                ]
                for track, spans in sorted(self._spans.items())
            }
        )
        return DetectorIdentity(
            name="scripted", variant_digest=sha256_bytes(script.encode("utf-8"))
        )

    @staticmethod
    def _clip(spans: Iterable[SpeechSpan], window: AudioWindow) -> list[SpeechSpan]:
        clipped: list[SpeechSpan] = []
        for span in spans:
            start = max(span.start_sample, window.start_sample)
            end = min(span.end_sample, window.end_sample)
            if end > start:
                clipped.append(
                    SpeechSpan(
                        start_sample=start,
                        end_sample=end,
                        probability=span.probability,
                        details=dict(span.details),
                    )
                )
        return sorted(clipped, key=lambda span: (span.start_sample, span.end_sample))
