"""Real weights, real device: the completion gate's central criterion, plus measurements.

Run from the ROCm environment on the target host:

    nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync \\
        pytest -m host_smoke'

Everything here is marked `host_smoke` and excluded from `./scripts/gate.sh` (INV-05). The
gate asks for one thing — a short real transcription and alignment that passes — and most of
this file is the other thing this milestone owes: the numbers behind **OQ-018**, **OQ-009**
and **OQ-022**, which are guesses about *this* model that nothing before now could measure.

**Real speech is required and is not synthesized.** INV-10 forbids expecting a learned model
to behave a particular way on audio no human made, and every question below is about what
Qwen does with a voice. Whatever recordings are in `samples/` are used — they are gitignored
because they are session audio, and they are *discovered* rather than named so that replacing
them re-runs these measurements instead of silently skipping them. Without any, the
measurements skip with a reason naming what they cannot be measured against.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.interfaces import AudioWindow, TranscriptionRequest
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.transcript.qwen import (
    ATTENTION_IMPLEMENTATION,
    QwenBackend,
    QwenTranscriber,
)

pytestmark = pytest.mark.host_smoke

MAX_NEW_TOKENS = 1024
MARGIN = 16

#: Session-absolute, and deliberately not zero: a decoder that forgot to rebase word times
#: onto the request's own grid passes every test that starts at sample zero.
WINDOW_START = 1_600_000


#: Whatever real speech is on this machine, lowest filename first.
#:
#: **Discovered rather than pinned**, and that is deliberate. The recordings this milestone
#: measured against were an early smoke-test capture, and the owner intends to replace them
#: with a properly synced set in the right formats. A pinned filename would turn that
#: replacement into a silent skip — every measurement here quietly not running, with a green
#: suite and no signal — which is the worst of the three possible outcomes.
#:
#: Any of the four decodes: `ffmpeg` reads `pcm_s24le` as happily as `pcm_f32le`, and the
#: 24-bit refusal is `ingest`'s working-path guard (**OQ-007**), not a limit on ASR.
#:
#: One file, not several. Nothing here needs two tracks, and the sample probe's four are
#: **not** mutually aligned — within a receiver they are timecode-synced, but the jam between
#: receivers is reported not to have taken (**OQ-012**). A test that concatenated across the
#: pairs would be measuring that rather than the model.
def _sample() -> Path | None:
    found = sorted(Path("samples").glob("*.wav"))
    return found[0] if found else None


SAMPLE = _sample()

_no_speech = pytest.mark.skipif(
    SAMPLE is None,
    reason=(
        "needs real speech at samples/ — OQ-018 and OQ-022 are questions about what this "
        "model does with a voice, and INV-10 forbids answering them against synthetic noise"
    ),
)


def decode(path: Path, *, seconds: float, start: float = 0.0) -> npt.NDArray[np.float32]:
    """``seconds`` of ``path`` as mono 16 kHz float32 — the grid ASR consumes (ADR-0017)."""
    argv = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-i", str(path)]
    argv += ["-ac", "1", "-ar", str(DERIVATIVE_SAMPLE_RATE), "-f", "f32le"]
    argv += ["-t", f"{seconds}", "-"]
    raw = subprocess.run(
        argv,
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def _speech_path() -> Path:
    """The discovered recording, narrowed. Every caller is behind `_no_speech`, so this
    raising would mean a test lost its marker rather than that a machine lacks samples."""
    assert SAMPLE is not None, "guarded by `_no_speech`"
    return SAMPLE


def a_request(
    audio: npt.NDArray[np.float32],
    *,
    start: int = WINDOW_START,
    request_id: str = "smoke-0001",
    pad: int = DERIVATIVE_SAMPLE_RATE // 2,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> TranscriptionRequest:
    return TranscriptionRequest(
        request_id=request_id,
        audio=AudioWindow(
            track_id="tx-a",
            sample_rate=DERIVATIVE_SAMPLE_RATE,
            start_sample=start,
            samples=audio,
        ),
        core_start_sample=start + pad,
        core_end_sample=start + int(audio.size) - pad,
        language="English",
        max_new_tokens=max_new_tokens,
    )


@dataclass(frozen=True, slots=True)
class Loaded:
    """Both models, loaded once. Loading costs about eight seconds and six gigabytes."""

    backend: QwenBackend
    device: str
    dtype: str
    gfx_target: str | None
    #: The two `torch.nn.Module`s underneath the wrappers. Reached through the concrete
    #: backend rather than the protocol on purpose: the gate asks for bfloat16, `cuda:0` and
    #: SDPA *in effect*, and the only object that knows whether Transformers honoured those
    #: requests is the loaded model itself.
    asr_module: Any
    aligner_module: Any


@pytest.fixture(scope="session")
def loaded() -> Loaded:
    """The real pair, on whatever this host resolves to.

    Session-scoped because loading is slow and the models are stateless between calls — a
    transformer conditioned only on its input is a function of that input, which is exactly
    what distinguishes this from Silero's per-track recurrent detector (ADR-0013).
    """
    from dnd_audio.models import QWEN3_ALIGNER, QWEN3_ASR, require_snapshot
    from dnd_audio.runtime import probe_runtime, resolve_runtime
    from dnd_audio.transcript.qwen import load_qwen_backend

    probe = probe_runtime()
    assert probe.installed, (
        "run this from the ROCm environment: nix run .#fhs -- -c "
        "'UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync pytest -m host_smoke'"
    )
    resolution = resolve_runtime(device="auto", dtype="auto", probe=probe)
    backend = load_qwen_backend(
        asr_dir=require_snapshot(QWEN3_ASR),
        aligner_dir=require_snapshot(QWEN3_ALIGNER),
        device=resolution.device,
        dtype=resolution.dtype,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return Loaded(
        backend=backend,
        device=resolution.device,
        dtype=resolution.dtype,
        gfx_target=probe.gfx_target,
        asr_module=backend.model.model,  # type: ignore[attr-defined]
        aligner_module=backend.aligner.model,  # type: ignore[attr-defined]
    )


@pytest.fixture(scope="session")
def transcriber(loaded: Loaded) -> QwenTranscriber:
    return QwenTranscriber(
        loaded.backend, max_new_tokens=MAX_NEW_TOKENS, truncation_margin_tokens=MARGIN
    )


@pytest.fixture(scope="session")
def speech() -> npt.NDArray[np.float32]:
    """Twenty seconds of the real recording, which holds four short utterances."""
    if SAMPLE is None:
        # Belt and braces: every class here already carries `_no_speech`, and this catches
        # a future test that requests the fixture without the marker.
        pytest.skip(
            "needs real speech at samples/ — OQ-018 and OQ-022 are questions about this "
            "model's behaviour on a voice, which INV-10 forbids answering with noise"
        )
    return decode(_speech_path(), seconds=20.0)


@_no_speech
class TestTheGateCriterion:
    """*"A `host_smoke` test performs a short real transcription and alignment."*"""

    def test_it_transcribes_and_aligns_real_speech(
        self, transcriber: QwenTranscriber, speech: npt.NDArray[np.float32], loaded: Loaded
    ) -> None:
        result = transcriber.transcribe(a_request(speech))

        assert loaded.device == "cuda:0", f"expected the GPU, resolved {loaded.device}"
        assert loaded.dtype == "bfloat16", loaded.dtype
        assert loaded.gfx_target == "gfx1151", loaded.gfx_target
        assert result.alignment_status == "aligned", result.text
        assert result.text.strip(), "the model transcribed nothing from real speech"
        assert result.words, "alignment produced no word times"
        assert result.language == "English"

    def test_every_word_time_is_inside_the_submitted_window(
        self, transcriber: QwenTranscriber, speech: npt.NDArray[np.float32]
    ) -> None:
        """The rebase, against a request that does not start at sample zero.

        A decoder that returned the aligner's own request-relative seconds would put every
        word an hour before the audio it came from, and M4's ownership rule would then drop
        all of them — with nothing raising anywhere.
        """
        request = a_request(speech)
        result = transcriber.transcribe(request)

        for word in result.words:
            assert request.audio.start_sample <= word.start_sample <= request.audio.end_sample
            assert word.end_sample > word.start_sample

    def test_word_times_advance(
        self, transcriber: QwenTranscriber, speech: npt.NDArray[np.float32]
    ) -> None:
        starts = [word.start_sample for word in transcriber.transcribe(a_request(speech)).words]
        assert starts == sorted(starts)

    def test_the_raw_artifact_carries_the_backends_own_result(
        self, transcriber: QwenTranscriber, speech: npt.NDArray[np.float32]
    ) -> None:
        """M4 froze the envelope and left this half to M6b. Checked against the real object
        rather than against a fake's stand-in for it."""
        result = transcriber.transcribe(a_request(speech))

        assert result.public_document is not None
        assert result.public_document["calls"] == ["transcribe", "align"]
        assert result.public_document["asr_transcription"]["text"] == result.text
        assert result.public_document["forced_alignment"]["items"]

    def test_the_loaded_models_really_are_bf16_sdpa_on_the_gpu(self, loaded: Loaded) -> None:
        """The completion gate's *"Transformers backend, `torch.bfloat16`, `cuda:0`, SDPA
        attention"* — read off the loaded models rather than off this project's constants.

        The first version of this test asserted `ATTENTION_IMPLEMENTATION == "sdpa"`, which
        is a module constant already pinned by an offline test: a `host_smoke` test that
        needed a GPU and six gigabytes of weights in order to compare a string to itself.
        What it never checked is the thing that can actually be wrong — that
        `load_qwen_backend` passes those choices *through* to `from_pretrained` and that
        Transformers honoured them. A silently-ignored `attn_implementation`, a dtype
        overridden by a checkpoint's own config, or a model left on the CPU while the
        resolver reported `cuda:0` would all have passed. Found by M6b's verify phase.
        """
        import torch

        for name, module in (("asr", loaded.asr_module), ("aligner", loaded.aligner_module)):
            assert module.config._attn_implementation == ATTENTION_IMPLEMENTATION, (
                f"{name} loaded with {module.config._attn_implementation}, "
                f"not {ATTENTION_IMPLEMENTATION}"
            )
            parameter = next(module.parameters())
            assert parameter.dtype is torch.bfloat16, f"{name} is {parameter.dtype}"
            assert parameter.device.type == "cuda", f"{name} is on {parameter.device}"
            assert not module.training, f"{name} was left in training mode"


@_no_speech
class TestOq018Padding:
    """**OQ-018 (1)** — is `transcript.pad_ms` enough context to recover an edge word?

    M4 had to choose 500 ms before any model existed to check it against. The measurement:
    take a window whose first and last words are known from a generously padded request,
    then re-submit the same speech clipped hard at those words and compare. If the padded
    request recovers edge words the clipped one loses, the padding is doing its job.
    """

    def test_it_measures_what_a_hard_clip_costs_against_a_padded_request(
        self, transcriber: QwenTranscriber
    ) -> None:
        generous = transcriber.transcribe(
            a_request(decode(_speech_path(), seconds=14.0, start=9.0), request_id="pad-generous")
        )
        assert generous.words, "the reference request produced no words to clip against"

        first, last = generous.words[0], generous.words[-1]
        offset_s = (first.start_sample - WINDOW_START) / DERIVATIVE_SAMPLE_RATE
        length_s = (last.end_sample - first.start_sample) / DERIVATIVE_SAMPLE_RATE
        clipped = transcriber.transcribe(
            a_request(
                decode(_speech_path(), seconds=length_s, start=9.0 + offset_s),
                request_id="pad-clipped",
                pad=0,
            )
        )

        # Recorded rather than asserted as a threshold: what this run *measures* is how the
        # two texts differ, and a hard assertion on word counts would be pinning one
        # recording's outcome as a property of the model.
        #
        # The name of this test used to be `test_padding_recovers_words_a_hard_clip_loses`,
        # which is a claim about the result rather than a description of the instrument —
        # and the measured result is the opposite: the two texts came back *identical*, so
        # the clip lost nothing for the padding to recover. Both of M6b's reviewers flagged
        # the mismatch. The finding is real and recorded under OQ-018(1); what was wrong was
        # a test asserting non-emptiness under a name promising a recovery.
        print(f"\nOQ-018(1) padded:  {generous.text!r}")
        print(f"OQ-018(1) clipped: {clipped.text!r}")
        print(f"OQ-018(1) identical: {generous.text.strip() == clipped.text.strip()}")
        assert generous.text.strip()
        assert clipped.text.strip()


@_no_speech
class TestOq018TimestampStability:
    """**OQ-018 (2)** — do two overlapping requests agree about the words they share?

    M4's stitch rule recognizes a duplicate at a boundary by text equality plus interval
    overlap. If Qwen's times wander by more than a word's length between requests, that rule
    stops recognizing the duplicate and the word is emitted twice.
    """

    def test_a_word_in_the_overlap_lands_at_close_to_the_same_time_in_both(
        self, transcriber: QwenTranscriber
    ) -> None:
        early = transcriber.transcribe(
            a_request(decode(_speech_path(), seconds=16.0, start=8.0), request_id="overlap-early")
        )
        # Same audio, four seconds later, so the two windows share twelve seconds. The
        # session-absolute start is shifted by the same four seconds, so a shared word must
        # land at the same *absolute* position in both if the rebase is right.
        shift = int(4.0 * DERIVATIVE_SAMPLE_RATE)
        late = transcriber.transcribe(
            a_request(
                decode(_speech_path(), seconds=16.0, start=12.0),
                start=WINDOW_START + shift,
                request_id="overlap-late",
            )
        )

        # Paired the way M4's stitch rule pairs: same comparison key **and** overlapping in
        # time. Matching on text alone is what a first draft of this test did, and it
        # produced five outliers of 2–9 seconds — because this recording says "testing",
        # "a" and "transmitter" twice, so a text-only key cheerfully paired the first
        # occurrence in one window with the second in the other. Those were measurements of
        # the test's own matching, not of the model.
        from dnd_audio.transcript.normalize import comparison_key

        # **Two passes, and the denominator is the point.** Pairing on "same key *and*
        # overlapping" is how the stitch rule pairs, but using it alone to *measure* the
        # rule silently excludes its own failures: a word whose two placements drifted far
        # enough to stop overlapping simply vanishes from the sample, so the surviving
        # deltas are small by construction and "worst 0 ms" would be true of a model that
        # got half of them badly wrong. Codex's code review caught the selection bias.
        #
        # So: the first pass counts every word of the shared span by text alone — the
        # candidates the rule *ought* to recognize — and the second reports how many of them
        # it actually did. The gap between the two numbers is the measurement.
        shared_start = max(early.words[0].start_sample, late.words[0].start_sample)
        shared_end = min(early.words[-1].end_sample, late.words[-1].end_sample)

        # **Three populations, because the third turned out to be a different phenomenon.**
        # `deltas_ms` is every pair; `interior_ms` drops pairs involving either window's
        # *first* word. That split is not a convenience — see the block below the loop.
        candidates = 0
        deltas_ms: list[int] = []
        interior_ms: list[int] = []
        for late_word in late.words:
            if not (shared_start <= late_word.start_sample < shared_end):
                continue
            matches = [
                early_word
                for early_word in early.words
                if comparison_key(early_word.text) == comparison_key(late_word.text)
            ]
            if not matches:
                continue
            candidates += 1
            paired = [
                early_word
                for early_word in matches
                if early_word.start_sample < late_word.end_sample
                and late_word.start_sample < early_word.end_sample
            ]
            if paired:
                delta = abs(paired[0].start_sample - late_word.start_sample)
                delta_ms = delta * 1000 // DERIVATIVE_SAMPLE_RATE
                deltas_ms.append(delta_ms)
                if paired[0] is not early.words[0] and late_word is not late.words[0]:
                    interior_ms.append(delta_ms)

        assert candidates, "the two windows shared no word for the stitch rule to pair"
        print(f"\nOQ-018(2) {len(deltas_ms)}/{candidates} shared word(s) paired by the rule")
        print(f"OQ-018(2) |delta| ms: {sorted(deltas_ms)}")
        print(f"OQ-018(2) worst {max(deltas_ms) if deltas_ms else 'n/a'} ms")
        print(f"OQ-018(2) interior worst {max(interior_ms) if interior_ms else 'n/a'} ms")

        # The **ratio** is the assertion that matters and the half that could not fail
        # before: it is the stitch rule's own hit rate, and a word the rule fails to pair is
        # one it emits twice. Re-measured 2026-08-03 over all four of the jam capture's
        # recordings, after `samples/` was replaced during M8:
        #
        #     TX01_MIC002  16/16 paired, worst 6160 ms, interior worst 80 ms
        #     TX01_MIC005  15/16 paired, worst   80 ms, interior worst 80 ms
        #     TX02_MIC002  15/16 paired, worst   80 ms, interior worst 80 ms
        #     TX02_MIC003  16/16 paired, worst    0 ms, interior worst  0 ms
        assert len(deltas_ms) >= candidates * 3 // 4, (
            f"only {len(deltas_ms)} of {candidates} shared words still overlap between two "
            f"requests — M4's stitch rule would emit the rest twice"
        )

        # **The delta bound applies to interior words, and the reason is a measurement, not
        # a convenience (OQ-027).** One recording produced a 6160 ms outlier and it was a
        # single pair: the *first* word of a window that opens on silence, whose start the
        # aligner pins to the window start rather than to the speech. On TX01_MIC002 the
        # early window's "Hello" came back spanning 6560 ms against the late window's 400 ms
        # — and both agreed on its *end* to the sample. Nothing had wandered; a word had been
        # stretched backwards over the lead-in.
        #
        # Excluding those pairs silently would be exactly the weakening this bound exists to
        # prevent, so the behaviour is asserted in its own right below instead. What remains
        # here is the original claim, over the population it was always about: times do not
        # wander by more than a word's length, and a word is a few hundred milliseconds.
        assert max(interior_ms) <= 1000, (
            f"interior word starts moved by {max(interior_ms)} ms between two requests over "
            f"the same audio; the stitch rule pairs on overlap and cannot survive that"
        )

    def test_a_words_start_depends_on_how_much_lead_in_its_window_had(
        self, transcriber: QwenTranscriber
    ) -> None:
        """**OQ-027**, and the reason the bound above is scoped to interior words.

        The same word, in the same audio, gets a different *start* depending on how much
        non-speech the window opened with — and the same *end* to the sample. On
        TX01_MIC002 the window opening 6 s earlier returned "Hello" spanning 6560 ms where
        the later window returned 400 ms, both ending at 106560 ms. The aligner had absorbed
        the lead-in into the word.

        It is lead-in rather than silence, which is worth knowing for a real table: this
        window's first second is *louder* than the region where the word actually is (rms
        0.0077 against 0.0020, with a 0.13 transient at 2 s). Handling noise, a chair, a
        cough before someone speaks — that is what gets swallowed, and a real session has
        more of it than this one, not less.

        M4's stitch rule survives it, because the stretched word still overlaps the short one
        and the duplicate is recognized. What does not survive is ownership:
        `transcript/segments.py::_owned_words` assigns a word to the interval containing its
        **start**, so a start dragged seconds early falls outside every interval and is
        dropped. That is M8's diagnostic 9 counting a word the model never misplaced, and it
        is why the counter had to exist before `vad.pad_ms` moves (OQ-017).

        Stated as a comparison between two requests rather than as a millisecond bound: how
        much lead-in precedes the first word is a property of the recording, and asserting a
        number would pin this test to `samples/` again — the failure that produced it.
        """
        early = transcriber.transcribe(
            a_request(decode(_speech_path(), seconds=16.0, start=8.0), request_id="lead-in-long")
        )
        shift = int(4.0 * DERIVATIVE_SAMPLE_RATE)
        late = transcriber.transcribe(
            a_request(
                decode(_speech_path(), seconds=16.0, start=12.0),
                start=WINDOW_START + shift,
                request_id="lead-in-short",
            )
        )
        from dnd_audio.transcript.normalize import comparison_key

        first_early, first_late = early.words[0], late.words[0]
        assert comparison_key(first_early.text) == comparison_key(first_late.text), (
            "the two windows must open on the same word for this comparison to mean anything"
        )

        print(
            f"\nOQ-027 {first_early.text!r} long lead-in: {first_early.start_sample}"
            f"..{first_early.end_sample}"
        )
        print(
            f"OQ-027 {first_late.text!r} short lead-in: {first_late.start_sample}"
            f"..{first_late.end_sample}"
        )

        # The end is trustworthy. If it were not, the stitch rule would stop pairing and the
        # ratio asserted above would already have caught it — so this is the half that holds.
        assert first_early.end_sample == first_late.end_sample

        # The start is not, and it is dragged toward whichever window opened earlier.
        assert first_early.start_sample < first_late.start_sample, (
            "the longer lead-in is supposed to produce the earlier start — if it no longer "
            "does, OQ-027 has been answered and the interior-only bound above can be widened "
            "back to every pair"
        )


@_no_speech
class TestOq018Truncation:
    """**OQ-018 (3)** — does a low ceiling truncate, and does the heuristic see it?

    The retokenized-length heuristic is the whole of truncation detection, because 0.0.6
    exposes no finish reason (ADR-0028). What this measures is that it fires on a response
    that really was cut off, and does not on the same audio with a workable ceiling.
    """

    def test_a_low_ceiling_is_detected_as_truncation(self, loaded: Loaded) -> None:
        from dnd_audio.models import QWEN3_ALIGNER, QWEN3_ASR, require_snapshot
        from dnd_audio.transcript.qwen import load_qwen_backend

        # A *separate* backend, because the ceiling is bound at construction and the adapter
        # refuses a request that disagrees with the one it was built with. That refusal is
        # the point — it is what stops a cache entry being keyed under a number the model
        # never used — so the test honours it rather than working around it.
        ceiling = 8
        small = load_qwen_backend(
            asr_dir=require_snapshot(QWEN3_ASR),
            aligner_dir=require_snapshot(QWEN3_ALIGNER),
            device=loaded.device,
            dtype=loaded.dtype,
            max_new_tokens=ceiling,
        )
        cut = QwenTranscriber(small, max_new_tokens=ceiling, truncation_margin_tokens=0).transcribe(
            a_request(decode(_speech_path(), seconds=20.0), max_new_tokens=ceiling)
        )

        print(f"\nOQ-018(3) ceiling={ceiling}: truncated={cut.truncated} text={cut.text!r}")
        assert cut.truncated, (
            "a response generated under an eight-token ceiling was not detected as "
            "truncated; the retokenized-length heuristic is the only signal there is"
        )

    def test_a_workable_ceiling_is_not_detected_as_truncation(
        self, transcriber: QwenTranscriber, speech: npt.NDArray[np.float32]
    ) -> None:
        """The other direction, which is what stops the heuristic from being trivially
        satisfiable by always answering yes."""
        assert transcriber.transcribe(a_request(speech)).truncated is False


@_no_speech
class TestOq022Determinism:
    """**OQ-022** — is greedy decoding on this stack reproducible?

    `transcript.json` and `transcript.md` are byte-stable deterministic artifacts (INV-02),
    and from M6b their bytes come out of a GPU kernel. The ASR cache hides this completely:
    a warm run replays stored text, so every byte-stability test this project has ever run
    would pass on a model that answered differently every time.

    Compared **exactly** rather than within a tolerance. A tolerance would pass on precisely
    the wobble this exists to detect.
    """

    def test_the_same_request_twice_returns_the_same_text_and_the_same_word_times(
        self, transcriber: QwenTranscriber, speech: npt.NDArray[np.float32]
    ) -> None:
        first = transcriber.transcribe(a_request(speech))
        second = transcriber.transcribe(a_request(speech))

        assert first.text == second.text
        assert first.words == second.words
        assert first.language == second.language
        assert first.truncated == second.truncated

    def test_it_holds_for_a_second_piece_of_audio_too(self, transcriber: QwenTranscriber) -> None:
        """One utterance agreeing with itself could be a short-output coincidence."""
        audio = decode(_speech_path(), seconds=30.0)
        assert transcriber.transcribe(a_request(audio, request_id="det-a")).words == (
            transcriber.transcribe(a_request(audio, request_id="det-b")).words
        )


@_no_speech
class TestOq009WhereThePackageChunks:
    """**OQ-009** — where does `qwen-asr`'s timestamp path chunk, and does it bind on us?

    The question's own evidence requirement is "reading the installed package plus a
    long-segment experiment". The reading is in ADR-0028: `MAX_FORCE_ALIGN_INPUT_SECONDS`
    is 180 and `MAX_ASR_INPUT_SECONDS` is 1200. The experiment is here — and it turns up
    something the reading alone does not, which is that neither limit is on this project's
    route at all, because the adapter never calls the combined `transcribe(
    return_time_stamps=True)` that does the chunking (ADR-0028).
    """

    def test_the_constants_are_what_adr_0028_quotes(self) -> None:
        from qwen_asr.inference.utils import (
            MAX_ASR_INPUT_SECONDS,
            MAX_FORCE_ALIGN_INPUT_SECONDS,
        )

        assert MAX_FORCE_ALIGN_INPUT_SECONDS == 180
        assert MAX_ASR_INPUT_SECONDS == 1200

    def test_the_chunking_limit_is_reached_only_through_a_call_this_project_never_makes(
        self, loaded: Loaded
    ) -> None:
        """`split_audio_into_chunks` is called from `Qwen3ASRModel.transcribe`, and the
        limit it is given depends on `return_time_stamps`. The adapter passes `False` and
        aligns separately, so the 180 s path is unreachable from here — and `max_segment_s`
        caps a padded window at 120 s regardless, which is under both.
        """
        import inspect

        source = inspect.getsource(type(loaded.backend).align)
        assert "split_audio_into_chunks" not in source
        assert "return_time_stamps" not in source
