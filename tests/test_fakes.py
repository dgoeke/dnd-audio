"""INV-10: the model seams exist and have deterministic implementations.

The fakes are scripted rather than clever. These tests check that the seam is real
(protocol conformance), that the fakes are deterministic, and that the interface types
enforce the bounded-window contract INV-07 depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from dnd_audio.fakes import ScriptedActivityDetector, ScriptedTranscriber
from dnd_audio.interfaces import (
    ActivityDetector,
    AudioWindow,
    SpeechSpan,
    TranscribedWord,
    Transcriber,
    TranscriptionRequest,
    TranscriptionResult,
)


def _window(track_id: str = "tx-a", start: int = 0, length: int = 16_000) -> AudioWindow:
    return AudioWindow(
        track_id=track_id,
        sample_rate=16_000,
        start_sample=start,
        samples=np.zeros(length, dtype=np.float32),
    )


def _request(request_id: str = "req-1") -> TranscriptionRequest:
    window = _window(length=48_000)
    return TranscriptionRequest(
        request_id=request_id,
        audio=window,
        core_start_sample=8_000,
        core_end_sample=40_000,
    )


class TestProtocolConformance:
    def test_scripted_transcriber_satisfies_the_protocol(self) -> None:
        assert isinstance(ScriptedTranscriber({}), Transcriber)

    def test_scripted_detector_satisfies_the_protocol(self) -> None:
        assert isinstance(ScriptedActivityDetector({}), ActivityDetector)

    def test_conformance_is_more_than_a_name_check(self) -> None:
        """`runtime_checkable` only checks method presence, so exercise the call too."""
        transcriber: Transcriber = ScriptedTranscriber(
            {"req-1": TranscriptionResult(request_id="req-1", text="hello")}
        )
        assert transcriber.transcribe(_request()).text == "hello"


class TestScriptedTranscriber:
    def test_returns_the_scripted_result(self) -> None:
        expected = TranscriptionResult(
            request_id="req-1",
            text="We should go back to Zephyrine.",
            words=(TranscribedWord(start_sample=8_000, end_sample=9_000, text="We"),),
            alignment_status="aligned",
        )
        transcriber = ScriptedTranscriber({"req-1": expected})
        assert transcriber.transcribe(_request()) == expected

    def test_is_deterministic(self) -> None:
        transcriber = ScriptedTranscriber(
            {"req-1": TranscriptionResult(request_id="req-1", text="same")}
        )
        assert transcriber.transcribe(_request()) == transcriber.transcribe(_request())

    def test_an_unscripted_request_is_an_error(self) -> None:
        """Returning "" would hide a test transcribing something it did not plan for."""
        with pytest.raises(KeyError, match="no scripted response"):
            ScriptedTranscriber({}).transcribe(_request())

    def test_records_what_it_was_asked(self) -> None:
        """M4 asserts on requests too — that no padded waveform exceeded max_segment_s."""
        transcriber = ScriptedTranscriber(
            {
                "req-1": TranscriptionResult(request_id="req-1", text="a"),
                "req-2": TranscriptionResult(request_id="req-2", text="b"),
            }
        )
        transcriber.transcribe(_request("req-1"))
        transcriber.transcribe(_request("req-2"))
        assert [r.request_id for r in transcriber.requests] == ["req-1", "req-2"]

    def test_can_script_a_truncated_response(self) -> None:
        """The case M4 needs and a content-derived fake could not be asked for."""
        transcriber = ScriptedTranscriber(
            {"req-1": TranscriptionResult(request_id="req-1", text="cut off mid", truncated=True)}
        )
        assert transcriber.transcribe(_request()).truncated is True


class TestAlignmentStatusIsStatedNotInferred:
    """A result says whether its word times are aligned; nothing guesses from `words`.

    ADR-0005 named three states and only the adapter can tell `segment_only` — the aligner
    ran and failed — from `not_attempted`. The two consistency rules exist so a mismatch is
    an error at the seam rather than a wrong `alignment_status` in a transcript.
    """

    def test_aligned_requires_words(self) -> None:
        with pytest.raises(ValueError, match="carries no words"):
            TranscriptionResult(request_id="req-1", text="hello", alignment_status="aligned")

    def test_words_require_aligned(self) -> None:
        word = TranscribedWord(start_sample=8_000, end_sample=9_000, text="hello")
        with pytest.raises(ValueError, match="alignment_status='segment_only'"):
            TranscriptionResult(
                request_id="req-1",
                text="hello",
                words=(word,),
                alignment_status="segment_only",
            )

    def test_a_wordless_result_defaults_to_not_attempted(self) -> None:
        result = TranscriptionResult(request_id="req-1", text="hello")
        assert result.alignment_status == "not_attempted"
        assert result.words == ()

    def test_a_public_document_is_optional_and_is_not_the_result_itself(self) -> None:
        """`None` means "this object is already its own public form" — M6b fills it."""
        assert TranscriptionResult(request_id="req-1", text="hi").public_document is None
        carried = TranscriptionResult(
            request_id="req-1", text="hi", public_document={"language": "English", "text": "hi"}
        )
        assert carried.public_document == {"language": "English", "text": "hi"}


class TestScriptedActivityDetector:
    def test_returns_the_ground_truth_mask(self) -> None:
        spans = (SpeechSpan(start_sample=100, end_sample=900),)
        detector = ScriptedActivityDetector({"tx-a": spans})
        assert detector.detect(_window()) == spans

    def test_is_deterministic(self) -> None:
        detector = ScriptedActivityDetector({"tx-a": [SpeechSpan(100, 900)]})
        assert detector.detect(_window()) == detector.detect(_window())

    def test_a_track_with_no_script_is_silent(self) -> None:
        detector = ScriptedActivityDetector({"tx-a": [SpeechSpan(100, 900)]})
        assert detector.detect(_window(track_id="tx-b")) == ()

    def test_spans_are_clipped_to_the_window(self) -> None:
        """A windowed read must agree with a whole-track read (INV-07)."""
        detector = ScriptedActivityDetector({"tx-a": [SpeechSpan(500, 20_000)]})
        clipped = detector.detect(_window(start=0, length=16_000))
        assert clipped == (SpeechSpan(start_sample=500, end_sample=16_000),)

    def test_spans_outside_the_window_are_dropped(self) -> None:
        detector = ScriptedActivityDetector({"tx-a": [SpeechSpan(50_000, 60_000)]})
        assert detector.detect(_window(start=0, length=16_000)) == ()

    def test_results_are_sorted(self) -> None:
        detector = ScriptedActivityDetector(
            {"tx-a": [SpeechSpan(9_000, 10_000), SpeechSpan(100, 900)]}
        )
        starts = [span.start_sample for span in detector.detect(_window())]
        assert starts == sorted(starts)


class TestInterfaceContracts:
    def test_window_reports_its_end(self) -> None:
        window = _window(start=48_000, length=16_000)
        assert window.end_sample == 64_000
        assert len(window) == 16_000

    def test_stereo_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mono"):
            AudioWindow(
                track_id="tx-a",
                sample_rate=48_000,
                start_sample=0,
                samples=np.zeros((2, 100), dtype=np.float32),
            )

    def test_negative_start_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not be negative"):
            AudioWindow(
                track_id="tx-a",
                sample_rate=48_000,
                start_sample=-1,
                samples=np.zeros(10, dtype=np.float32),
            )

    def test_core_must_lie_inside_the_padded_window(self) -> None:
        """Padding exists for word recovery; a core outside it is a stitching bug."""
        window = _window(start=1_000, length=1_000)
        with pytest.raises(ValueError, match="before the padded window"):
            TranscriptionRequest(
                request_id="r",
                audio=window,
                core_start_sample=500,
                core_end_sample=1_500,
            )
        with pytest.raises(ValueError, match="past the padded window"):
            TranscriptionRequest(
                request_id="r",
                audio=window,
                core_start_sample=1_500,
                core_end_sample=9_000,
            )

    def test_empty_core_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is empty"):
            TranscriptionRequest(
                request_id="r",
                audio=_window(),
                core_start_sample=100,
                core_end_sample=100,
            )

    def test_empty_span_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is empty"):
            SpeechSpan(start_sample=10, end_sample=10)

    def test_probability_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="within"):
            SpeechSpan(start_sample=0, end_sample=10, probability=1.5)

    def test_times_are_integers_not_seconds(self) -> None:
        """INV-04: floats belong at the serialization boundary, not in the interfaces."""
        span = SpeechSpan(start_sample=48_000, end_sample=96_000)
        assert isinstance(span.start_sample, int)
        word = TranscribedWord(start_sample=0, end_sample=1, text="x")
        assert isinstance(word.start_sample, int)
