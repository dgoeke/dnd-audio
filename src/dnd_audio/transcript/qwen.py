"""The real `Transcriber`: Qwen3-ASR for text, Qwen3-ForcedAligner for word times.

M4 built everything above this seam and left one implementation behind it. This is that
implementation, and ADR-0028 records the four decisions it rests on. Three of them are only
obvious once you have read `qwen-asr` 0.0.6's source, which is why they are restated here.

**Two public calls, not one.** The package's `transcribe(return_time_stamps=True)` runs ASR,
*then* alignment, and constructs its `ASRTranscription` only after alignment returns — so an
aligner exception escapes carrying the already-generated text with it. The gate requires the
opposite: *"if alignment fails for one segment, retain the segment-level transcript and emit
a warning rather than failing the entire session."* That is unimplementable through the
combined call, so this module drives `transcribe` and `align` separately and puts alignment
in its own `try`. Both are public API.

**Truncation is a retokenized-length heuristic, and not because it was preferred.**
`_infer_asr_transformers` decodes `text_ids.sequences` to strings and keeps nothing else;
there is no finish reason to read. The spec offers "public backend metadata **or**
retokenized-length heuristics" and 0.0.6 supplies only the second. Reaching into
`model.generate` for a finish reason is what the spec prohibits in as many words.

**Timestamps are request-relative decimal seconds**, rounded to three places by the aligner.
:func:`decode_alignment` is the one place they cross to samples, and it does two things that
look like fussiness and are not: it parses through `Fraction(str(value))` rather than
`Fraction(float)`, because the package rounded to a decimal and the binary double is an
approximation of it (INV-04); and it rebases onto `request.audio.start_sample`, without
which a request beginning an hour into a session returns its words near session zero and
M4's ownership rule then correctly drops every one of them.

**The seam is the backend, one level below `Transcriber`.** Same placement, and the same
reason, as `activity/silero.py`'s `OnnxSession`: everything worth asserting here — the
timestamp decode, the alignment recovery, the truncation heuristic, that audio reaches the
model as an array and never as a path (INV-06) — is behaviour of *this* module, and a fake
`Transcriber` would replace exactly the code under test. A fake backend leaves it running.

Torch, `transformers` and `qwen_asr` are imported lazily inside :func:`load_qwen_backend`
and nowhere else, so importing this module stays free for the default suite (INV-05).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from dnd_audio.determinism import to_samples
from dnd_audio.errors import DndAudioError
from dnd_audio.interfaces import (
    TranscribedWord,
    TranscriptionRequest,
    TranscriptionResult,
)
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "OFFLINE_ENV_VARS",
    "QWEN_BACKEND_NAME",
    "AlignedItem",
    "QwenBackend",
    "QwenError",
    "QwenText",
    "QwenTranscriber",
    "decode_alignment",
    "enable_offline_mode",
    "load_qwen_backend",
    "qwen_asr_version",
]

#: The attention implementation, fixed rather than configured. The spec asks for PyTorch
#: SDPA as the baseline and names no alternative; a knob with one value and no consumer is
#: interface nobody asked for. The *resolved* value still reaches the report and the cache
#: key, which is what makes a future change visible (ADR-0028).
ATTENTION_IMPLEMENTATION: Final = "sdpa"

#: What this transcriber is called in the report and in `TranscriberIdentity.name`.
QWEN_BACKEND_NAME: Final = "qwen-asr"

#: Set **before** `transformers` or `qwen_asr` is imported, because both read the
#: environment at import time and cache what they find — setting them around
#: `from_pretrained` would be too late and would look like it worked, since the models are
#: loaded from local directories and would not reach the network anyway on a healthy
#: machine. The point is that they cannot reach it on an unhealthy one.
#:
#: Telemetry is disabled for INV-06's sake rather than for offline mode's: nothing about a
#: local session may leave this machine, including a usage ping naming a model.
OFFLINE_ENV_VARS: Final[tuple[tuple[str, str], ...]] = (
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
)


class QwenError(DndAudioError):
    """The adapter cannot run, or was driven in a way it cannot be.

    Fatal in every case (INV-13). Distinct from an *alignment* failure, which is
    recoverable by design and produces `segment_only` rather than an exception.
    """

    default_code = "asr_adapter_unusable"


@dataclass(frozen=True, slots=True)
class QwenText:
    """What the ASR model said, before anything is aligned.

    ``document`` is the backend's own public result serialized losslessly — the spec's
    "save the unmodified public `ASRTranscription`". Built by the backend because only the
    backend has seen the package's object; passed through unread by this module, which is
    what "unmodified" has to mean.
    """

    language: str
    text: str
    document: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AlignedItem:
    """One aligned unit, in the aligner's own units: seconds, rounded to three places.

    Kept as `float` rather than converted at the seam so that the decimal-to-exact
    conversion happens in exactly one place (:func:`decode_alignment`) and a second backend
    could not quietly introduce a different one (INV-04).

    ``text`` is the aligner's own token, which is **not** a substring partition of the ASR
    text: `tokenize_space_lang` strips punctuation. M4's `comparison_key` normalizes before
    comparing, so nothing downstream breaks, but a reader comparing a word list against its
    segment text will see the difference and it is not a bug.
    """

    text: str
    start_seconds: float
    end_seconds: float


@runtime_checkable
class QwenBackend(Protocol):
    """The three model operations this adapter performs, and the seam every test drives.

    Audio crosses this boundary as a mono float32 array at 16 kHz and in no other form.
    That is INV-06 made structural rather than promised: there is no parameter here that
    could carry a path or a URL, so an implementation cannot send audio anywhere without
    changing this protocol first.
    """

    def transcribe_text(
        self, audio: npt.NDArray[np.float32], *, context: str, language: str
    ) -> QwenText: ...

    def align(
        self, audio: npt.NDArray[np.float32], *, text: str, language: str
    ) -> tuple[AlignedItem, ...]: ...

    def count_tokens(self, text: str) -> int: ...


def enable_offline_mode() -> None:
    """Put Hugging Face into offline mode. Must run before either library is imported.

    Unconditional rather than "if not already set": an operator who exported
    ``HF_HUB_OFFLINE=0`` has expressed a preference this pipeline does not offer, and a
    production run that quietly honoured it would reach the network from a stage INV-06
    forbids to.
    """
    for name, value in OFFLINE_ENV_VARS:
        os.environ[name] = value


def qwen_asr_version() -> str:
    """The installed `qwen-asr` version, for the report and the cache key (INV-08).

    Its own metadata rather than a `__version__` attribute: `qwen_asr/__init__.py` declares
    ``__all__ = ["__version__"]`` and then never defines it, so the obvious read raises.
    """
    from importlib.metadata import version

    return version("qwen-asr")


class QwenTranscriber:
    """One `Transcriber` over one loaded backend.

    Args:
        backend: The model pair. Shared across tracks and requests — unlike Silero's
            detector this holds no state between calls, because a transformer conditioned
            only on its input is a function of that input.
        max_new_tokens: The ceiling the backend was **constructed** with. Every request is
            checked against it: the package takes this at construction while M4 puts it on
            each request, so a bundle whose identity says 512 over a backend still
            generating 1024 would key a different cache entry for identical behaviour.
        truncation_margin_tokens: How close to the ceiling a retokenized response must land
            to be treated as cut off (**OQ-018**).
    """

    def __init__(
        self,
        backend: QwenBackend,
        *,
        max_new_tokens: int,
        truncation_margin_tokens: int,
    ) -> None:
        if max_new_tokens <= 0:
            message = f"max_new_tokens must be positive, got {max_new_tokens}"
            raise QwenError(message, code="asr_adapter_misused")
        if truncation_margin_tokens < 0:
            message = (
                f"truncation_margin_tokens must not be negative, got {truncation_margin_tokens}"
            )
            raise QwenError(message, code="asr_adapter_misused")

        self._backend = backend
        self._max_new_tokens = max_new_tokens
        self._margin = truncation_margin_tokens

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Transcribe one padded window, then align it. Alignment may fail on its own.

        Raises:
            QwenError: if the request is on the wrong grid or asks for a generation ceiling
                this backend was not built with. Both are programming errors rather than
                operator ones, which is why they are loud instead of repaired.
        """
        self._check(request)
        audio = np.ascontiguousarray(request.audio.samples, dtype=np.float32)

        spoken = self._backend.transcribe_text(
            audio, context=request.context or "", language=request.language
        )
        truncated = self._looks_truncated(spoken.text)
        text = spoken.text.strip()

        if not text:
            # Nothing to align, and nothing failed. A VAD fires on coughs and door closes;
            # a retained candidate the model found no words in is ordinary (M4 counts them
            # into one warning). `not_attempted` is the truthful status: no aligner ran.
            return self._result(
                request,
                spoken,
                text="",
                words=(),
                status="not_attempted",
                truncated=truncated,
                aligned=None,
            )

        language = spoken.language or request.language
        try:
            items = self._backend.align(audio, text=text, language=language)
        except Exception:
            # Deliberately `Exception`. This is the recovery path the gate names, and what
            # an aligner can raise is not enumerable: an out-of-memory error, a shape
            # mismatch from a text its tokenizer split differently than the model expected,
            # an `IndexError` out of `fix_timestamp` on a degenerate output. Narrowing this
            # to the failures seen so far would mean the *next* one costs a whole session
            # instead of one segment's word times, which is what this exists to prevent.
            return self._result(
                request,
                spoken,
                text=text,
                words=(),
                status="segment_only",
                truncated=truncated,
                aligned=None,
            )

        words = decode_alignment(items, request=request)
        if not words:
            # The aligner ran and produced nothing usable — no items, or none that survived
            # validation. That is a failure *of* alignment rather than an absence of it, and
            # the two mean different things to an operator reading a transcript (ADR-0005).
            return self._result(
                request,
                spoken,
                text=text,
                words=(),
                status="segment_only",
                truncated=truncated,
                aligned=items,
            )

        return self._result(
            request,
            spoken,
            text=text,
            words=words,
            status="aligned",
            truncated=truncated,
            aligned=items,
        )

    def _check(self, request: TranscriptionRequest) -> None:
        if request.audio.sample_rate != DERIVATIVE_SAMPLE_RATE:
            message = (
                f"request {request.request_id} is at {request.audio.sample_rate} Hz and this "
                f"adapter is called at {DERIVATIVE_SAMPLE_RATE} Hz. ASR consumes the cached "
                f"derivative, never the 48 kHz working path — resampling here would be a "
                f"second resampler under a cache key (ADR-0017)."
            )
            raise QwenError(message, code="asr_adapter_misused")

        if request.max_new_tokens != self._max_new_tokens:
            message = (
                f"request {request.request_id} asks for max_new_tokens="
                f"{request.max_new_tokens} and this backend was built with "
                f"{self._max_new_tokens}. The package takes the ceiling at construction, so "
                f"honouring the request would key a cache entry under a number the model "
                f"never used (INV-08)."
            )
            raise QwenError(message, code="asr_adapter_misused")

    def _looks_truncated(self, text: str) -> bool:
        """Did generation stop at the ceiling rather than at an end-of-sequence token?

        0.0.6 exposes no finish reason, so the evidence is length: a response whose own
        tokens reach within `truncation_margin_tokens` of the ceiling is one the model was
        probably still in the middle of. The margin exists because the decode is not
        reversible token-for-token — `skip_special_tokens=True` drops the terminator, and
        retokenizing text is not guaranteed to reproduce the sequence that produced it.
        """
        if not text.strip():
            return False
        return self._backend.count_tokens(text) >= self._max_new_tokens - self._margin

    def _result(
        self,
        request: TranscriptionRequest,
        spoken: QwenText,
        *,
        text: str,
        words: tuple[TranscribedWord, ...],
        status: Literal["aligned", "segment_only", "not_attempted"],
        truncated: bool,
        aligned: tuple[AlignedItem, ...] | None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            request_id=request.request_id,
            text=text,
            words=words,
            language=spoken.language or request.language,
            truncated=truncated,
            alignment_status=status,
            public_document=_public_document(spoken, aligned),
        )


def _public_document(spoken: QwenText, aligned: tuple[AlignedItem, ...] | None) -> dict[str, Any]:
    """The spec's lossless raw artifact, in the shape the two calls actually produced.

    The spec asks for "the unmodified public `ASRTranscription`... losslessly serialize all
    public fields, including language, text, and timestamp items when present". Because
    this adapter makes the two public calls *separately* (ADR-0028), no single returned
    object holds all of it — so the envelope names the calls it made rather than
    reassembling them into an `ASRTranscription` that no call returned. `calls` is the
    honest part: a reader can tell "the aligner was never asked" from "the aligner was
    asked and this is what came back".
    """
    document: dict[str, Any] = {
        "asr_transcription": spoken.document,
        "calls": ["transcribe"] if aligned is None else ["transcribe", "align"],
        "package": QWEN_BACKEND_NAME,
    }
    if aligned is not None:
        document["forced_alignment"] = {
            "items": [
                {"text": item.text, "start_time": item.start_seconds, "end_time": item.end_seconds}
                for item in aligned
            ]
        }
    return document


def decode_alignment(
    items: tuple[AlignedItem, ...], *, request: TranscriptionRequest
) -> tuple[TranscribedWord, ...]:
    """Aligner seconds to session-absolute samples on the request's own grid.

    Returns ``()`` when the alignment is unusable, which the caller turns into
    `segment_only`. **Never raises for bad data**: one malformed item must cost one
    segment's word times, not a session's transcript. The artifact layer would refuse a
    word outside its ownership interval, so an unchecked item here becomes a
    `ValidationError` four stages downstream with nothing in it naming the aligner.

    Refused, and why each one is a real output of a model rather than a hypothetical:

    * non-finite or negative times — `fix_timestamp` interpolates over anomalous positions
      and can extrapolate outside the audio it was given;
    * ``end < start`` — two timestamp tokens decoding out of order;
    * starts that go backwards — the same interpolation collapsing or reordering a run;
    * anything outside the submitted window — the request is what the model heard, and a
      word cannot be somewhere it was not played.

    **A zero-length item is not malformed**, and this is the one rule here that was written
    from measurement rather than from reasoning. The aligner quantizes to
    `timestamp_segment_time` — 80 ms on this model — so any word shorter than one step comes
    back with ``end == start``. On the very first real utterance transcribed by this project
    that was the word "a", and the first draft of this function treated it as corruption and
    threw away all fifteen word times in the segment. It would have done that to most
    segments in most sessions, and the only visible symptom would have been a transcript
    with no word times and a warning saying alignment failed.

    Such an item is widened to one sample: the word *is* there and its start is known; what
    is unknown is a duration below the aligner's resolution, and one sample is the smallest
    half-open interval that says so without inventing an extent. Dropping it instead would
    silently delete a word from the transcript.

    Empty text is dropped rather than refused: the aligner's tokenizer strips punctuation,
    so a token that was entirely punctuation cleans to nothing. That is ordinary.
    """
    words: list[TranscribedWord] = []
    previous_start = request.audio.start_sample
    for item in items:
        text = item.text.strip()
        if not text:
            continue
        start = _to_sample(item.start_seconds, request)
        end = _to_sample(item.end_seconds, request)
        if start is None or end is None or end < start:
            return ()
        if start < previous_start and words:
            # Starts must not go backwards. Deliberately about starts rather than about
            # overlap: two adjacent words routinely share a boundary and the aligner emits
            # exactly that, but a list that runs backwards is one whose times cannot all be
            # right, with no way to tell which half to believe.
            return ()
        words.append(TranscribedWord(start_sample=start, end_sample=max(end, start + 1), text=text))
        previous_start = start
    return tuple(words)


def _to_sample(seconds: float, request: TranscriptionRequest) -> int | None:
    """One decimal-seconds timestamp as an absolute sample, or ``None`` if it is not one.

    `Fraction(str(seconds))` and not `Fraction(seconds)`. The aligner rounds to three
    decimal places, so the short decimal is the value it meant; `Fraction(float)` would
    preserve the binary approximation of that decimal instead, which is exactly the
    "never accumulate floating point durations" failure INV-04 names. `to_samples` is the
    project's single quantizer.
    """
    if not np.isfinite(seconds) or seconds < 0:
        return None
    offset = to_samples(Fraction(str(seconds)), DERIVATIVE_SAMPLE_RATE)
    absolute = request.audio.start_sample + offset
    if absolute > request.audio.end_sample:
        return None
    return absolute


def load_qwen_backend(
    *,
    asr_dir: Path,
    aligner_dir: Path,
    device: str,
    dtype: str,
    max_new_tokens: int,
) -> QwenBackend:
    """Load both models from verified local directories, offline, onto ``device``.

    Everything heavyweight is imported here and nowhere else (INV-05). Offline mode is
    established *first*, before either library is imported — `tests/test_qwen_adapter.py`
    proves the ordering in a subprocess, because that is the only place it is observable.

    Args:
        asr_dir: A verified `QWEN3_ASR` snapshot directory. Never a repository name: a name
            would be resolved, and this pipeline resolves nothing at run time (ADR-0027).
        aligner_dir: A verified `QWEN3_ALIGNER` snapshot directory.
        device: ``cuda:0`` or ``cpu``, as `resolve_runtime` decided.
        dtype: ``bfloat16`` or ``float32``, likewise.
        max_new_tokens: Bound into the model here, and asserted per request above.
    """
    enable_offline_mode()

    try:
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    except Exception as exc:
        # The import is inside a `try` for the same reason `probe_runtime` catches
        # `Exception` rather than `ImportError` (M6a): a ROCm build with a mismatched
        # shared library raises `OSError` from the dynamic loader, not `ImportError`. And
        # the ordinary case matters more — on the project environment there is no torch at
        # all, deliberately (INV-05), so this is the path every `transcribe` without
        # `--fake-models` takes on a machine that was never set up for ASR. It must produce
        # the actionable diagnostic, not a traceback.
        #
        # Note this is reached whatever `asr.device` resolved to: CPU inference needs torch
        # just as much as GPU inference does, so a `device: auto` fallback to CPU does not
        # rescue a machine without the runtime.
        message = (
            f"the ASR runtime is not usable here: {type(exc).__name__}: {exc}. It lives in "
            f"the opt-in `asr-qwen` group, which the project environment deliberately does "
            f"not carry — install it with `nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT="
            f".venv-rocm uv sync --group asr-qwen'`, or transcribe a synthetic session from "
            f"its own declared script with `--fake-models`."
        )
        raise QwenError(message, code="asr_runtime_unavailable") from exc

    torch_dtype = getattr(torch, dtype)
    try:
        model = Qwen3ASRModel.from_pretrained(
            str(asr_dir),
            dtype=torch_dtype,
            attn_implementation=ATTENTION_IMPLEMENTATION,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=1,
        )
        aligner = Qwen3ForcedAligner.from_pretrained(
            str(aligner_dir), dtype=torch_dtype, attn_implementation=ATTENTION_IMPLEMENTATION
        )
    except Exception as exc:
        # `Exception`, not a narrower type, and M6a learned this the hard way: a ROCm build
        # with a mismatched shared library raises `OSError` from the dynamic loader, and a
        # model that fails to load is the failure an operator most needs a diagnostic for.
        message = (
            f"loading the Qwen models failed: {type(exc).__name__}: {exc}. Both snapshots "
            f"verified, so this is the runtime rather than the weights — "
            f"`dnd-audio doctor --device cuda --dtype {dtype}` checks the same stack."
        )
        raise QwenError(message, code="asr_model_load_failed") from exc

    model.model.to(device).eval()
    aligner.model.to(device).eval()
    _force_greedy(model.model)
    return _TransformersBackend(model=model, aligner=aligner)


def _force_greedy(model: Any) -> None:
    """Disable sampling explicitly rather than inheriting the snapshot's preference.

    `transcript.json` and `transcript.md` are byte-stable deterministic artifacts (INV-02),
    and a sampled decode would make that false the first time a cache was cleared. The
    snapshot ships a `generation_config.json` whose contents are upstream's business, so
    the setting is stated here rather than assumed. Whether *greedy* decoding is itself
    reproducible on this stack is a separate question and is **OQ-022**.
    """
    config = model.generation_config
    config.do_sample = False
    config.num_beams = 1
    config.temperature = None
    config.top_p = None
    config.top_k = None


@dataclass(frozen=True, slots=True)
class _TransformersBackend:
    """The production `QwenBackend`: the official wrapper's two public entry points.

    Thin on purpose. Everything this class does is call a public method and reshape its
    result into the protocol's types, so that the logic worth testing lives above it in
    code the default suite can reach.
    """

    model: Any
    aligner: Any

    def transcribe_text(
        self, audio: npt.NDArray[np.float32], *, context: str, language: str
    ) -> QwenText:
        """ASR only. ``return_time_stamps`` is deliberately false — see the module docstring.

        The audio goes in as ``(array, rate)``, which is an in-memory array and not a path
        or a URL (INV-06). At 16 kHz the package's normalizer neither resamples nor reads a
        file; it clips to [-1, 1] and hands the samples to the processor.
        """
        results = self.model.transcribe(
            audio=[(audio, DERIVATIVE_SAMPLE_RATE)],
            context=[context],
            language=[language],
            return_time_stamps=False,
        )
        first = results[0]
        return QwenText(
            language=str(first.language or ""),
            text=str(first.text or ""),
            document={
                "language": first.language,
                "text": first.text,
                "time_stamps": first.time_stamps,
            },
        )

    def align(
        self, audio: npt.NDArray[np.float32], *, text: str, language: str
    ) -> tuple[AlignedItem, ...]:
        results = self.aligner.align(
            audio=[(audio, DERIVATIVE_SAMPLE_RATE)], text=[text], language=[language]
        )
        return tuple(
            AlignedItem(
                text=str(item.text),
                start_seconds=float(item.start_time),
                end_seconds=float(item.end_time),
            )
            for item in results[0].items
        )

    def count_tokens(self, text: str) -> int:
        """The processor's own public tokenizer, never a private generation path."""
        return len(self.model.processor.tokenizer.encode(text))
