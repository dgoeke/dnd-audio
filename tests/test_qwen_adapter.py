"""The Qwen adapter, driven through its real body against a fake backend (ADR-0028).

The seam under test is `QwenBackend`, one level below `Transcriber`. That placement is the
whole point: a fake `Transcriber` — which M4 already has, and which every other transcript
test uses — would replace exactly the code these tests exist to exercise. A fake *backend*
leaves `QwenTranscriber` running, so the timestamp decode, the alignment recovery, the
truncation heuristic and INV-06 are asserted against the code that ships.

Nothing here imports torch, `transformers` or `qwen_asr`, and nothing here touches a model
(INV-05). The one property that cannot be observed in-process — that offline mode is set
*before* those libraries are imported — is proved in a subprocess at the bottom of the file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.interfaces import AudioWindow, TranscriptionRequest
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.transcript.qwen import (
    ATTENTION_IMPLEMENTATION,
    OFFLINE_ENV_VARS,
    AlignedItem,
    QwenBackend,
    QwenError,
    QwenText,
    QwenTranscriber,
    decode_alignment,
    enable_offline_mode,
)

MAX_NEW_TOKENS = 1024
MARGIN = 16

#: Far enough into a session that a decoder which forgot to rebase would place every word
#: near sample zero — which is the failure the plan review found, and it is invisible in a
#: test whose request starts at 0.
WINDOW_START = 1_600_000


class FakeBackend:
    """A scripted `QwenBackend`. Records what it was handed; returns what it was told to.

    It also *enforces* INV-06 rather than merely not violating it: `transcribe_text` and
    `align` fail the test if handed anything that is not a mono float32 array. A backend
    that accepted a path would make the protocol's promise unfalsifiable.
    """

    def __init__(
        self,
        *,
        text: str = "we should go back to Zephyrine",
        language: str = "English",
        items: tuple[AlignedItem, ...] = (),
        tokens: int = 8,
        align_error: Exception | None = None,
        document: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.language = language
        self.items = items
        self.tokens = tokens
        self.align_error = align_error
        self.document = document
        self.transcribe_calls: list[dict[str, Any]] = []
        self.align_calls: list[dict[str, Any]] = []
        self.counted: list[str] = []

    @staticmethod
    def _check_audio(audio: object) -> npt.NDArray[np.float32]:
        assert isinstance(audio, np.ndarray), f"audio reached the model as {type(audio)!r}"
        assert audio.dtype == np.float32, audio.dtype
        assert audio.ndim == 1, audio.shape
        return audio

    def transcribe_text(
        self, audio: npt.NDArray[np.float32], *, context: str, language: str
    ) -> QwenText:
        self.transcribe_calls.append(
            {"audio": self._check_audio(audio), "context": context, "language": language}
        )
        return QwenText(
            language=self.language,
            text=self.text,
            document=self.document
            if self.document is not None
            else {"language": self.language, "text": self.text, "time_stamps": None},
        )

    def align(
        self, audio: npt.NDArray[np.float32], *, text: str, language: str
    ) -> tuple[AlignedItem, ...]:
        self.align_calls.append(
            {"audio": self._check_audio(audio), "text": text, "language": language}
        )
        if self.align_error is not None:
            raise self.align_error
        return self.items

    def count_tokens(self, text: str) -> int:
        self.counted.append(text)
        return self.tokens


def a_request(
    *,
    start: int = WINDOW_START,
    samples: int = DERIVATIVE_SAMPLE_RATE * 4,
    context: str | None = None,
    language: str = "English",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> TranscriptionRequest:
    rng = np.random.default_rng(0)
    return TranscriptionRequest(
        request_id="tx-a-0001",
        audio=AudioWindow(
            track_id="tx-a",
            sample_rate=DERIVATIVE_SAMPLE_RATE,
            start_sample=start,
            samples=rng.standard_normal(samples).astype(np.float32) * 0.01,
        ),
        core_start_sample=start + DERIVATIVE_SAMPLE_RATE // 2,
        core_end_sample=start + samples - DERIVATIVE_SAMPLE_RATE // 2,
        language=language,
        context=context,
        max_new_tokens=max_new_tokens,
    )


def a_transcriber(backend: QwenBackend, *, margin: int = MARGIN) -> QwenTranscriber:
    return QwenTranscriber(backend, max_new_tokens=MAX_NEW_TOKENS, truncation_margin_tokens=margin)


class TestTheProtocolIsSatisfied:
    def test_the_fake_is_a_qwen_backend(self) -> None:
        assert isinstance(FakeBackend(), QwenBackend)

    def test_the_adapter_is_a_transcriber(self) -> None:
        from dnd_audio.interfaces import Transcriber

        assert isinstance(a_transcriber(FakeBackend()), Transcriber)


class TestAudioNeverLeavesAsAPathOrUrl:
    """INV-06, at the point where it is the adapter's to enforce."""

    def test_audio_reaches_the_backend_as_an_array(self) -> None:
        backend = FakeBackend(items=(AlignedItem("hello", 0.5, 0.9),))
        request = a_request()

        a_transcriber(backend).transcribe(request)

        submitted = backend.transcribe_calls[0]["audio"]
        assert isinstance(submitted, np.ndarray)
        np.testing.assert_array_equal(submitted, request.audio.samples)

    def test_the_aligner_is_handed_the_same_audio_not_a_path(self) -> None:
        """The second call is a second chance to get this wrong."""
        backend = FakeBackend(items=(AlignedItem("hello", 0.5, 0.9),))
        request = a_request()

        a_transcriber(backend).transcribe(request)

        np.testing.assert_array_equal(backend.align_calls[0]["audio"], request.audio.samples)

    def test_the_protocol_offers_no_way_to_pass_a_path(self) -> None:
        """Structural rather than behavioural: `QwenBackend`'s audio parameter is an array,
        so an implementation cannot send audio anywhere without changing the protocol."""
        from inspect import signature

        for method in (QwenBackend.transcribe_text, QwenBackend.align):
            annotation = signature(method).parameters["audio"].annotation
            assert "NDArray" in str(annotation), (method, annotation)


class TestLanguageAndContext:
    def test_the_configured_language_is_forced(self) -> None:
        backend = FakeBackend()
        a_transcriber(backend).transcribe(a_request(language="German"))
        assert backend.transcribe_calls[0]["language"] == "German"

    def test_a_glossary_is_passed_as_the_context(self) -> None:
        backend = FakeBackend()
        a_transcriber(backend).transcribe(a_request(context="Zephyrine, Vhalor"))
        assert backend.transcribe_calls[0]["context"] == "Zephyrine, Vhalor"

    def test_an_absent_glossary_never_blocks_a_run(self) -> None:
        """The spec: "its absence never blocks a run". Empty string, not None — the
        package's own signature defaults to `""` and treats it as no context."""
        backend = FakeBackend()
        result = a_transcriber(backend).transcribe(a_request(context=None))
        assert backend.transcribe_calls[0]["context"] == ""
        assert result.text

    def test_the_models_own_language_wins_when_it_reports_one(self) -> None:
        """It is what the aligner must be told, and what the record should carry."""
        backend = FakeBackend(language="French", items=(AlignedItem("bonjour", 0.1, 0.4),))
        result = a_transcriber(backend).transcribe(a_request(language="English"))

        assert backend.align_calls[0]["language"] == "French"
        assert result.language == "French"

    def test_the_requested_language_is_the_fallback_when_the_model_reports_none(self) -> None:
        backend = FakeBackend(language="", items=(AlignedItem("hello", 0.1, 0.4),))
        result = a_transcriber(backend).transcribe(a_request(language="English"))

        assert backend.align_calls[0]["language"] == "English"
        assert result.language == "English"


class TestTimestampDecoding:
    """Where seconds become samples. One conversion, rebased, and strict about bad data."""

    def test_word_times_are_session_absolute_on_the_derivative_grid(self) -> None:
        backend = FakeBackend(items=(AlignedItem("hello", 0.5, 0.75),))
        result = a_transcriber(backend).transcribe(a_request())

        assert result.alignment_status == "aligned"
        word = result.words[0]
        assert word.start_sample == WINDOW_START + 8000
        assert word.end_sample == WINDOW_START + 12000

    def test_a_decoder_that_forgot_to_rebase_would_be_caught_here(self) -> None:
        """The plan review's finding, kept as its own assertion.

        Without the rebase a word at 0.5 s lands at sample 8000 — near session zero, an
        hour before the audio it came from — and M4's ownership rule then correctly drops
        every word in the request. Nothing would raise; the transcript would be empty.
        """
        backend = FakeBackend(items=(AlignedItem("hello", 0.5, 0.75),))
        result = a_transcriber(backend).transcribe(a_request(start=WINDOW_START))

        assert result.words[0].start_sample > WINDOW_START
        assert result.words[0].start_sample != 8000

    def test_the_conversion_is_exact_for_a_decimal_the_aligner_can_produce(self) -> None:
        """`Fraction(str(x))` rather than `Fraction(float)` (INV-04).

        0.001 s at 16 kHz is exactly 16 samples. Through the binary double the value is
        0.001000000000000000020816681711721685... — which still rounds to 16 here, so the
        difference is not visible in the answer for this one input. What this pins is that
        the *decimal* is what gets quantized, and `test_a_third_decimal_place_is_honoured`
        below is the case where the two roads visibly diverge.
        """
        backend = FakeBackend(items=(AlignedItem("a", 0.001, 0.002),))
        result = a_transcriber(backend).transcribe(a_request())

        assert result.words[0].start_sample == WINDOW_START + 16
        assert result.words[0].end_sample == WINDOW_START + 32

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0.1, 1600), (0.123, 1968), (1.999, 31984), (2.5, 40000), (0.0005, 8)],
    )
    def test_a_third_decimal_place_is_honoured(self, seconds: float, expected: int) -> None:
        """Three decimal places is the aligner's resolution; every one of them must land."""
        backend = FakeBackend(items=(AlignedItem("a", seconds, seconds + 0.05),))
        result = a_transcriber(backend).transcribe(a_request())
        assert result.words[0].start_sample == WINDOW_START + expected

    def test_punctuation_only_tokens_are_dropped_rather_than_refused(self) -> None:
        """The aligner's tokenizer strips punctuation, so a token can clean to nothing."""
        backend = FakeBackend(
            items=(
                AlignedItem("hello", 0.1, 0.2),
                AlignedItem("  ", 0.2, 0.3),
                AlignedItem("there", 0.3, 0.4),
            )
        )
        result = a_transcriber(backend).transcribe(a_request())

        assert [word.text for word in result.words] == ["hello", "there"]
        assert result.alignment_status == "aligned"

    def test_adjacent_words_may_share_a_boundary(self) -> None:
        backend = FakeBackend(items=(AlignedItem("one", 0.1, 0.2), AlignedItem("two", 0.2, 0.3)))
        result = a_transcriber(backend).transcribe(a_request())
        assert len(result.words) == 2


class TestMalformedAlignmentIsRecoverable:
    """One bad aligner item must cost one segment's word times, never a session.

    Every case here is a real output of `fix_timestamp`, which interpolates over anomalous
    positions and can extrapolate past the audio it was given. Unchecked, each becomes a
    `ValidationError` four stages downstream with nothing in it naming the aligner.
    """

    @pytest.mark.parametrize(
        ("items", "why"),
        [
            ((AlignedItem("a", float("nan"), 0.5),), "not a number"),
            ((AlignedItem("a", 0.1, float("inf")),), "infinite"),
            ((AlignedItem("a", -0.5, 0.5),), "negative"),
            ((AlignedItem("a", 0.6, 0.4),), "end before start"),
            ((AlignedItem("a", 0.1, 0.2), AlignedItem("b", 0.05, 0.15)), "goes backwards"),
            ((AlignedItem("a", 0.1, 0.2), AlignedItem("b", 99.0, 99.5)), "past the window"),
        ],
    )
    def test_it_degrades_to_segment_only_and_keeps_the_text(
        self, items: tuple[AlignedItem, ...], why: str
    ) -> None:
        backend = FakeBackend(items=items)
        result = a_transcriber(backend).transcribe(a_request())

        assert result.alignment_status == "segment_only", why
        assert result.words == ()
        assert result.text == "we should go back to Zephyrine", why

    def test_a_zero_length_item_is_quantization_and_not_corruption(self) -> None:
        """Written from measurement, not from reasoning — see `decode_alignment`.

        The aligner quantizes to 80 ms, so any word shorter than one step comes back with
        `end == start`. The first real utterance this project ever transcribed contained
        one ("a", at 10.800 -> 10.800), and the first draft of the decoder called it
        corruption and threw away all fifteen word times in the segment. The exact items
        below are that measurement, kept verbatim.
        """
        items = (
            AlignedItem("Testing", 10.480, 10.720),
            AlignedItem("a", 10.800, 10.800),
            AlignedItem("first", 10.880, 11.120),
            AlignedItem("transmitter", 11.120, 11.680),
        )
        result = a_transcriber(FakeBackend(items=items)).transcribe(a_request(samples=16000 * 20))

        assert result.alignment_status == "aligned"
        assert [word.text for word in result.words] == [
            "Testing",
            "a",
            "first",
            "transmitter",
        ]

    def test_a_zero_length_item_becomes_the_smallest_interval_that_can_exist(self) -> None:
        """One sample: the word is there and its start is known; its extent is below the
        aligner's resolution and inventing one would be worse than saying so."""
        items = (AlignedItem("a", 10.800, 10.800),)
        result = a_transcriber(FakeBackend(items=items)).transcribe(a_request(samples=16000 * 20))

        word = result.words[0]
        assert word.end_sample == word.start_sample + 1

    def test_adjacent_words_sharing_a_boundary_are_not_a_backwards_list(self) -> None:
        """The aligner emits this constantly — `first` ends where `transmitter` starts —
        so a monotonicity rule stated over intervals rather than starts would reject almost
        every real segment."""
        items = (
            AlignedItem("first", 10.880, 11.120),
            AlignedItem("transmitter", 11.120, 11.680),
        )
        result = a_transcriber(FakeBackend(items=items)).transcribe(a_request(samples=16000 * 20))
        assert len(result.words) == 2

    def test_an_empty_alignment_is_a_failure_of_alignment_not_an_absence_of_it(self) -> None:
        """`segment_only` and `not_attempted` mean different things to an operator reading
        a transcript with no word times in it (ADR-0005). The aligner ran here."""
        backend = FakeBackend(items=())
        result = a_transcriber(backend).transcribe(a_request())

        assert result.alignment_status == "segment_only"
        assert backend.align_calls, "the aligner must actually have been asked"

    def test_decode_alignment_never_raises_on_bad_data(self) -> None:
        """Asserted directly, because the caller relies on it and a raise would be a
        session-level failure rather than a segment-level one."""
        request = a_request()
        assert decode_alignment((AlignedItem("a", float("nan"), 1.0),), request=request) == ()


class TestAlignmentFailureKeepsTheText:
    """The gate criterion the package's combined call cannot satisfy (ADR-0028)."""

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("HIP out of memory"),
            ValueError("Batch size mismatch: audio=1, text=1, language=0"),
            IndexError("list index out of range"),
            OSError("libamdhip64.so: cannot open shared object file"),
        ],
    )
    def test_an_aligner_exception_keeps_the_text_as_segment_only(self, error: Exception) -> None:
        """What an aligner can raise is not enumerable, so the recovery cannot be either.

        `fix_timestamp` alone can raise `IndexError` on a degenerate output, and a GPU can
        raise anything at all. Narrowing this to the failures seen so far would mean the
        next one costs a whole session.
        """
        backend = FakeBackend(align_error=error)
        result = a_transcriber(backend).transcribe(a_request())

        assert result.alignment_status == "segment_only"
        assert result.text == "we should go back to Zephyrine"
        assert result.words == ()

    def test_the_asr_call_still_happened_and_is_what_was_kept(self) -> None:
        backend = FakeBackend(align_error=RuntimeError("boom"))
        result = a_transcriber(backend).transcribe(a_request())

        assert len(backend.transcribe_calls) == 1
        assert result.text == backend.text

    def test_an_asr_failure_is_not_recovered_from(self) -> None:
        """Only *alignment* is recoverable. A model that cannot transcribe at all is a
        failed stage, not a segment with no word times."""

        class Broken(FakeBackend):
            def transcribe_text(self, audio: Any, **kwargs: Any) -> QwenText:
                message = "HIP error"
                raise RuntimeError(message)

        with pytest.raises(RuntimeError):
            a_transcriber(Broken()).transcribe(a_request())


class TestEmptyText:
    def test_nothing_said_means_no_aligner_ran(self) -> None:
        """A VAD fires on coughs and door closes. `not_attempted` is truthful here."""
        backend = FakeBackend(text="   ")
        result = a_transcriber(backend).transcribe(a_request())

        assert result.alignment_status == "not_attempted"
        assert result.text == ""
        assert result.words == ()
        assert backend.align_calls == []

    def test_empty_text_is_never_reported_as_truncated(self) -> None:
        backend = FakeBackend(text="", tokens=MAX_NEW_TOKENS)
        assert a_transcriber(backend).transcribe(a_request()).truncated is False


class TestTruncationHeuristic:
    """0.0.6 exposes no finish reason, so length is the whole of the evidence."""

    @pytest.mark.parametrize(
        ("tokens", "truncated"),
        [
            (8, False),
            (MAX_NEW_TOKENS - MARGIN - 1, False),
            (MAX_NEW_TOKENS - MARGIN, True),
            (MAX_NEW_TOKENS, True),
            (MAX_NEW_TOKENS + 40, True),
        ],
    )
    def test_the_margin_is_where_the_verdict_changes(self, tokens: int, truncated: bool) -> None:
        backend = FakeBackend(tokens=tokens, items=(AlignedItem("a", 0.1, 0.2),))
        assert a_transcriber(backend).transcribe(a_request()).truncated is truncated

    def test_it_retokenizes_the_returned_text_and_nothing_else(self) -> None:
        backend = FakeBackend(text="hello there")
        a_transcriber(backend).transcribe(a_request())
        assert backend.counted == ["hello there"]

    def test_a_zero_margin_means_only_reaching_the_ceiling_counts(self) -> None:
        backend = FakeBackend(tokens=MAX_NEW_TOKENS - 1, items=(AlignedItem("a", 0.1, 0.2),))
        assert a_transcriber(backend, margin=0).transcribe(a_request()).truncated is False

    def test_truncation_survives_an_alignment_failure(self) -> None:
        """The two are independent: a response can be both cut off and unalignable, and
        M4's split-and-retry needs the flag either way."""
        backend = FakeBackend(tokens=MAX_NEW_TOKENS, align_error=RuntimeError("boom"))
        result = a_transcriber(backend).transcribe(a_request())

        assert result.truncated is True
        assert result.alignment_status == "segment_only"

    def test_no_private_finish_reason_or_generation_path_is_used(self) -> None:
        """The specific boundary the spec draws, rather than "no private attribute at all".

        The prohibition is on depending on a private finish-reason or lower-level response
        path that the public wrapper does not provide — reaching into `model.generate`,
        `sequences`, or `_infer_asr_*`. The protocol has three public operations and the
        adapter calls nothing else.
        """
        source = Path("src/dnd_audio/transcript/qwen.py").read_text(encoding="utf-8")
        body = source.split("class QwenTranscriber")[1].split("def load_qwen_backend")[0]
        for forbidden in ("generate(", ".sequences", "_infer_asr", "finish_reason", "scores"):
            assert forbidden not in body, forbidden


class TestTheBoundCeiling:
    def test_a_request_disagreeing_with_the_bound_ceiling_raises(self) -> None:
        """The package takes `max_new_tokens` at construction; M4 puts it on each request.

        A bundle whose identity says 512 over a backend still generating 1024 would key a
        different cache entry for identical model behaviour — a stale answer served under a
        fresh-looking key, which is INV-08's whole subject.
        """
        with pytest.raises(QwenError) as caught:
            a_transcriber(FakeBackend()).transcribe(a_request(max_new_tokens=512))
        assert caught.value.code == "asr_adapter_misused"

    def test_a_matching_request_is_accepted(self) -> None:
        result = a_transcriber(FakeBackend()).transcribe(a_request(max_new_tokens=MAX_NEW_TOKENS))
        assert result.request_id == "tx-a-0001"

    @pytest.mark.parametrize(("ceiling", "margin"), [(0, 16), (-1, 16), (1024, -1)])
    def test_a_nonsensical_construction_is_refused(self, ceiling: int, margin: int) -> None:
        with pytest.raises(QwenError):
            QwenTranscriber(FakeBackend(), max_new_tokens=ceiling, truncation_margin_tokens=margin)


class TestTheGrid:
    def test_a_request_at_the_wrong_sample_rate_is_refused(self) -> None:
        """ASR consumes the cached 16 kHz derivative (ADR-0017). Resampling here would be a
        second resampler under a cache key, which is the failure INV-04 names for time."""
        request = TranscriptionRequest(
            request_id="tx-a-0001",
            audio=AudioWindow(
                track_id="tx-a",
                sample_rate=48_000,
                start_sample=0,
                samples=np.zeros(48_000, dtype=np.float32),
            ),
            core_start_sample=0,
            core_end_sample=48_000,
        )
        with pytest.raises(QwenError) as caught:
            a_transcriber(FakeBackend()).transcribe(request)
        assert caught.value.code == "asr_adapter_misused"


class TestThePublicDocument:
    """The spec's lossless raw artifact — the half M4 froze the envelope for."""

    def test_it_carries_the_backends_own_result(self) -> None:
        document = {"language": "English", "text": "hello", "time_stamps": None, "extra": 1}
        backend = FakeBackend(
            text="hello", document=document, items=(AlignedItem("hello", 0.1, 0.2),)
        )

        result = a_transcriber(backend).transcribe(a_request())

        assert result.public_document is not None
        assert result.public_document["asr_transcription"] == document

    def test_it_is_never_none_for_the_real_adapter(self) -> None:
        """`None` means "this result already is its public form", which is true of a fake
        and must not be true of the adapter (M4's charter note)."""
        assert a_transcriber(FakeBackend()).transcribe(a_request()).public_document is not None

    def test_it_records_which_calls_were_made(self) -> None:
        """A reader must be able to tell "the aligner was never asked" from "the aligner
        was asked and this is what came back"."""
        aligned = a_transcriber(FakeBackend(items=(AlignedItem("a", 0.1, 0.2),))).transcribe(
            a_request()
        )
        failed = a_transcriber(FakeBackend(align_error=RuntimeError("x"))).transcribe(a_request())
        silent = a_transcriber(FakeBackend(text="")).transcribe(a_request())

        assert aligned.public_document is not None
        assert failed.public_document is not None
        assert silent.public_document is not None
        assert aligned.public_document["calls"] == ["transcribe", "align"]
        assert failed.public_document["calls"] == ["transcribe"]
        assert silent.public_document["calls"] == ["transcribe"]

    def test_alignment_items_are_serialized_in_the_aligners_own_units(self) -> None:
        """Seconds, as returned. The samples are the pipeline's reading of them; the raw
        artifact records what the model said, not what this project made of it."""
        items = (AlignedItem("hello", 0.5, 0.75),)
        result = a_transcriber(FakeBackend(items=items)).transcribe(a_request())

        assert result.public_document is not None
        assert result.public_document["forced_alignment"]["items"] == [
            {"text": "hello", "start_time": 0.5, "end_time": 0.75}
        ]

    def test_it_survives_json_serialization(self) -> None:
        """It is written to `work/asr/<key>.raw.json` through `canonical_json`, so anything
        that is not JSON in it fails the run rather than the test."""
        from dnd_audio.determinism import canonical_json

        result = a_transcriber(FakeBackend(items=(AlignedItem("hello", 0.5, 0.75),))).transcribe(
            a_request()
        )
        assert canonical_json(result.public_document)


class TestOfflineMode:
    def test_enable_offline_mode_sets_every_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, _ in OFFLINE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        enable_offline_mode()

        for name, value in OFFLINE_ENV_VARS:
            assert os.environ[name] == value

    def test_it_overrides_an_operator_who_asked_to_be_online(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that quietly honoured `HF_HUB_OFFLINE=0` would reach the network from a
        stage INV-06 forbids to."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        enable_offline_mode()
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    def test_telemetry_is_disabled_too(self) -> None:
        """INV-06 is about audio, and a usage ping naming a model is not audio — but it is
        outbound traffic from a stage that must not make any."""
        assert ("HF_HUB_DISABLE_TELEMETRY", "1") in OFFLINE_ENV_VARS


class TestImportingThisModuleIsFree:
    """INV-05, and the boundary M6a documented: a subprocess is where the autouse fixtures
    cannot look, so what is asserted here is asserted in one."""

    def test_importing_the_adapter_does_not_import_torch(self, tmp_path: Path) -> None:
        program = textwrap.dedent(
            """
            import sys
            import dnd_audio.transcript.qwen  # noqa: F401
            leaked = sorted(n for n in sys.modules if n.split(".")[0] in
                            {"torch", "transformers", "qwen_asr"})
            print(",".join(leaked))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        assert completed.stdout.strip() == ""

    def test_offline_mode_is_set_before_the_backend_is_imported(self, tmp_path: Path) -> None:
        """The one ordering that cannot be observed in-process.

        `transformers` and `huggingface_hub` read `HF_HUB_OFFLINE` at *import* time, so
        setting it around `from_pretrained` would be too late — and would look like it
        worked, because both models load from local directories and would not reach the
        network anyway on a healthy machine. The point is that they cannot on an unhealthy
        one.

        A shim shadows `torch` and `qwen_asr` on the child's `PYTHONPATH` and records the
        environment as it saw it at import, which is exactly `tests/test_runtime.py`'s
        technique for the same reason (M6a).
        """
        shim = tmp_path / "shim"
        shim.mkdir()
        (shim / "torch.py").write_text(
            "bfloat16 = 'bfloat16'\nfloat32 = 'float32'\n", encoding="utf-8"
        )
        (shim / "qwen_asr.py").write_text(
            textwrap.dedent(
                """
                import os
                SEEN = {
                    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                }
                class Qwen3ASRModel:
                    @classmethod
                    def from_pretrained(cls, *a, **k):
                        raise RuntimeError("shim: " + repr(SEEN))
                class Qwen3ForcedAligner:
                    @classmethod
                    def from_pretrained(cls, *a, **k):
                        raise RuntimeError("shim")
                """
            ),
            encoding="utf-8",
        )
        program = textwrap.dedent(
            """
            import os
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            from pathlib import Path
            from dnd_audio.transcript.qwen import QwenError, load_qwen_backend
            try:
                load_qwen_backend(asr_dir=Path("."), aligner_dir=Path("."),
                                  device="cpu", dtype="float32", max_new_tokens=8)
            except QwenError as exc:
                print(str(exc))
            """
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(shim), *([environment["PYTHONPATH"]] if "PYTHONPATH" in environment else [])]
        )
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)

        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            env=environment,
        )
        assert "'HF_HUB_OFFLINE': '1'" in completed.stdout, completed.stdout
        assert "'TRANSFORMERS_OFFLINE': '1'" in completed.stdout, completed.stdout


class TestAttentionIsFixed:
    def test_sdpa_is_what_the_spec_asks_for(self) -> None:
        assert ATTENTION_IMPLEMENTATION == "sdpa"

    def test_it_is_not_configurable(self) -> None:
        """Recorded in provenance and the cache key, but not a knob: the spec names one
        value and a knob with no second value and no consumer is interface nobody asked
        for (ADR-0028)."""
        from dnd_audio.config import AsrConfig

        assert "attention" not in AsrConfig.model_fields


class TestBoundedMemory:
    """INV-07 at this seam. `asr.py` bounds submission; this bounds the adapter.

    The composed proof is `tests/test_memory.py`'s — an ordered event log showing a write
    before the last read, which nothing accumulating a session-length array can satisfy.
    What is left to show here is the narrower claim the module docstring makes: that the
    adapter itself keeps nothing after it answers. A transcriber that stashed each window
    "for debugging" would satisfy every other test in this file and turn a four-hour session
    into six full waveforms in RAM on a machine where that gets the process killed.
    """

    def test_nothing_survives_the_request_that_carried_it(self) -> None:
        """Asserted through the garbage collector rather than by reading the code.

        A `weakref` is the only way to state "nothing holds this any more" without
        enumerating the places that might. The *request* is dropped as well as the backend's
        recording, and that is not a weakening of the claim — it is the claim. The adapter
        does not copy the window (see below), so while a caller holds the request the array
        is alive because the caller wants it. What must not happen is the adapter outliving
        the caller's interest, which is what a transcriber stashing windows "for debugging"
        would do: six of those at session length is exactly what INV-07 exists to prevent.
        """
        import gc
        import weakref

        backend = FakeBackend(items=(AlignedItem("hello", 0.1, 0.2),))
        transcriber = a_transcriber(backend)
        request = a_request()

        transcriber.transcribe(request)
        submitted = weakref.ref(backend.transcribe_calls[0]["audio"])

        backend.transcribe_calls.clear()
        backend.align_calls.clear()
        del request
        gc.collect()

        assert submitted() is None, "the adapter is still holding a submitted window"

    def test_the_window_is_submitted_without_being_copied(self) -> None:
        """`np.ascontiguousarray` is a no-op on an array that is already contiguous float32,
        which every window from `DerivativeReader` is. So the normalization the adapter does
        for safety costs nothing in the ordinary case, and a request's audio is not briefly
        resident twice. Worth pinning: switching to `np.array(...)` or `.astype(np.float32)`
        would look equivalent and would double peak memory per request.
        """
        backend = FakeBackend(items=(AlignedItem("hello", 0.1, 0.2),))
        request = a_request()

        a_transcriber(backend).transcribe(request)

        assert backend.transcribe_calls[0]["audio"] is request.audio.samples

    def test_it_submits_exactly_one_window_per_request(self) -> None:
        """Two calls, one array. The adapter transcribes and aligns the *same* window rather
        than making a second copy to align — which would double peak memory for no gain."""
        backend = FakeBackend(items=(AlignedItem("hello", 0.1, 0.2),))
        a_transcriber(backend).transcribe(a_request())

        assert len(backend.transcribe_calls) == 1
        assert len(backend.align_calls) == 1
        assert backend.align_calls[0]["audio"] is backend.transcribe_calls[0]["audio"]
