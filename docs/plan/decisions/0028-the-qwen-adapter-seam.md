# ADR-0028 — The Qwen adapter: three operations, a strict decoder, and truncation by retokenization

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M6b

## Context

M4 finished the `Transcriber` seam and everything above it: request planning, padded
submission, bounded truncation retry, word ownership, duplicate collapse, and both
deliverables. What it could not build is the one implementation behind the seam, and it left
three obligations for this milestone in as many words — fill `public_document` from the
model's own result, state `alignment_status` rather than infer it, and supply `truncated`
from public backend metadata or a retokenized-length heuristic rather than a private
finish-reason API.

The official `qwen-asr` 0.0.6 package shapes what is possible. Read from the published wheel
rather than from its README:

- `Qwen3ASRModel.transcribe(audio, context, language, return_time_stamps=True)` returns
  `[ASRTranscription(language, text, time_stamps)]`, where `time_stamps` is a
  `ForcedAlignResult` whose items carry `text`, `start_time`, `end_time` **in seconds,
  rounded to three decimal places**.
- `(np.ndarray, sr)` is a supported input. At `sr == 16000` nothing resamples.
- `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`, against `MAX_ASR_INPUT_SECONDS = 1200`. Alignment
  is what this pipeline always asks for, so 180 s is the limit that binds (**OQ-009**).
- `_infer_asr_transformers` decodes `text_ids.sequences` and returns strings. **Nothing
  survives that says generation stopped at the ceiling.** There is no public finish reason
  to prefer over the heuristic; the spec's "or" is not a choice here.
- `transcribe(return_time_stamps=True)` runs ASR, *then* alignment, and constructs
  `ASRTranscription` only afterwards. **An aligner exception destroys text that was already
  generated.**

That last point is the one that decides the design, and it was found by M6b's plan review
rather than by reading the plan.

## Decision

**The seam is `QwenBackend`, one level below `Transcriber`, with three operations:
`transcribe_text`, `align`, and `count_tokens`.**

Placing the seam below `Transcriber` is `activity/silero.py::OnnxSession`'s pattern and it is
here for the same reason. INV-10 is satisfied by a fake `Transcriber`, but the properties
that matter in *this* module — that word times are rebased onto the request's own grid, that
a malformed alignment degrades rather than aborts, that audio reaches the model as an array,
that `public_document` is the backend's own result — are properties of the code a fake
`Transcriber` would replace. A fake backend leaves the production adapter running.

**Three operations rather than one, because the gate requires text to survive alignment
failure.** The package's combined call cannot deliver that: the exception escapes before
`ASRTranscription` exists. So the adapter transcribes text first, then aligns in its own
`try`, and an alignment failure yields `segment_only` plus a warning with the text intact —
which is what "retain the segment-level transcript and emit a warning rather than failing the
entire session" asks for. Both operations are public.

**One strict timestamp decoder, and it is the only place seconds become samples:**

```
request.audio.start_sample + to_samples(Fraction(str(item.start_time)), 16_000)
```

`Fraction(str(...))` and not `Fraction(float)`: the package rounds to three decimals, so the
decimal string is the value it meant and the binary double is an approximation of it —
exactly the substitution INV-04 exists to forbid. `to_samples` is the project's single
quantizer.

Rebasing on `request.audio.start_sample` is not incidental. Qwen's times are relative to the
waveform submitted; a request beginning at sample 1 600 000 whose first word came back at
0.5 s would otherwise be placed near session zero, and M4's ownership rule would then
correctly drop it as outside its own core. The word would vanish with no error anywhere.

**A malformed alignment is a recoverable per-segment failure, never an exception.**
Non-finite values, negative times, `end <= start`, non-monotonic items, or items outside the
submitted window degrade the whole result to `segment_only` with a warning. The alternative
is one bad aligner item failing `WordRecord` validation and aborting a four-hour session's
transcript, on the very criterion that asks for the opposite.

**Truncation is the retokenized-length heuristic**: the returned text retokenized with the
processor's own public tokenizer, compared against `max_new_tokens` less
`asr.truncation_margin_tokens`. Not a preference — see the context; there is no public
finish reason in 0.0.6. M4's split-and-retry machinery consumes the resulting `truncated`
flag unchanged.

**`max_new_tokens` is bound to the constructed backend and every request is asserted against
it.** The package takes the ceiling at construction while M4 puts it on each
`TranscriptionRequest`. A bundle whose identity says 512 over a backend still generating 1024
would key a different cache entry for identical model behaviour, so the disagreement raises
rather than being reconciled silently.

**Attention is hard-coded to SDPA and recorded, not configured.** The spec asks for SDPA and
names no second value; a knob with no alternative and no consumer is interface this
milestone's non-goals do not ask for. The *resolved* value still reaches the report and the
cache key, because that is what makes a later change visible.

**Everything that decides what the model would say reaches `TranscriberIdentity`, and the
runtime half does so as a nested `RuntimeProvenance`** rather than as a parallel row of
scalar fields. M6a defined those fields once precisely so M6b would not build a second
vocabulary to drift from them.

**Torch, `transformers` and `qwen_asr` are imported lazily inside functions**, and
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set *before* those imports rather than
around `from_pretrained` — both libraries read the environment at import time.

## Alternatives considered

**Use `transcribe(return_time_stamps=True)` and accept whatever it does on alignment
failure.** Simplest, and it fails the gate. It also fails it invisibly: alignment succeeds on
every clean utterance, so the defect would surface for the first time on a real session, on
the one segment that mattered.

**Infer `alignment_status` from whether words came back.** Rejected by M4 already, and the
reasoning holds: only the adapter can tell "the aligner ran and failed" from "no aligner
ran", and those two mean different things to an operator reading a transcript with no word
times in it.

**Put the seam at `Transcriber` and test with a fake transcriber.** That is the obvious
place, and it would leave every line of this module untested. The seam belongs below the
code whose behaviour is in question.

**Detect truncation by asking the model to regenerate with `return_dict_in_generate`.** That
means reaching past the high-level API into `model.generate`, which the spec prohibits in as
many words, and it doubles inference cost to answer a question a tokenizer answers for free.

**Clamp malformed alignment items into range instead of degrading to `segment_only`.**
Rejected for the reason `silero.py` refuses to clamp an out-of-range probability: it turns a
wrong answer into a plausible one, and the transcript is where that becomes invisible.

## Consequences

The adapter runs two model invocations per request where the package's combined call runs
one batched pass. On a session-length recording that is a real cost, and it buys the gate
criterion the combined call cannot satisfy. If it proves expensive enough to matter, the
optimization is to keep the split and batch within it — not to go back to the combined call.

Aligned word texts come from the aligner's own tokenizer, which strips punctuation
(`tokenize_space_lang`). Word texts are therefore not a substring partition of the segment
text. M4's `comparison_key` already normalizes for comparison, so nothing downstream breaks,
but a reader comparing `transcript.json`'s words against its segment text will see the
difference and should not conclude something is wrong.

`transcript.json` and `transcript.md` are byte-stable deterministic artifacts (INV-02). That
claim now rests on a GPU kernel being reproducible across cold runs, which is an assumption
about the world rather than about this code. It is registered as **OQ-022** and cited from
the adapter, and the host smoke test measures it.

**OQ-018** items 1–3 — padding sufficient for word recovery, timestamp stability across
overlapping requests, and whether a low-energy split beats the midpoint — become measurable
for the first time and are answered by this milestone's smoke test. Item 4, the
text-similarity thresholds, still needs a real session.
