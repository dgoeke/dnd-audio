# M6b — Qwen ASR adapter and forced aligner

**Status:** not started
**Depends on:** M4, M6a
**Spec sections:** Milestone 4 (Qwen specifics); Target-host runtime; Tests and
acceptance criteria 12, 14

## Goal

A real `Transcriber` implementation backed by `Qwen/Qwen3-ASR-1.7B` and
`Qwen/Qwen3-ForcedAligner-0.6B`, with revision-locked offline model loading and a
complete cache identity — dropped in behind the interface M4 already exercises.

## Completion gate

- [ ] `models fetch` is the only command that touches the network for models. It
      resolves mutable Hugging Face names to exact snapshot commit revisions and
      writes a local model lock. Caches live outside session directories and out of
      version control.
- [ ] `process` uses that lock rather than re-resolving a moving branch, and
      production processing runs under Hugging Face offline mode.
- [ ] Explicit model/aligner revisions may be set in configuration.
- [ ] Transformers backend, `torch.bfloat16`, `cuda:0`, SDPA attention.
- [ ] Audio reaches the model only as local paths or arrays — never a URL or API
      (INV-06).
- [ ] English forced by default, language configurable. `glossary.txt` passed via
      Qwen's context parameter when present; its absence never blocks a run.
- [ ] Forced aligner produces word times; per-segment alignment failure warns and
      retains segment-level text.
- [ ] Cache identity: exact segment-audio hash, model and aligner identifiers **and
      resolved revisions**, context hash, language, backend, dtype, attention
      implementation, and every output-affecting generation/alignment parameter
      including `max_new_tokens` (INV-08). Atomic writes; incomplete entries never hit.
- [ ] `max_new_tokens` defaults to 1024, not the upstream wrapper's 512. Changing
      it invalidates the cache — tested.
- [ ] Truncation detection uses public backend metadata or retokenized-length
      heuristics, never a private Qwen finish-reason API. The split/retry machinery
      from M4 is reused unchanged.
- [ ] Processing runs in bounded windows; documented that it must not run alongside
      a heavy ComfyUI or large-LLM workload on this UMA host (INV-07).
- [ ] Report records Python, `qwen-asr`, Transformers, Torch, HIP runtime, device,
      dtype, attention implementation, and resolved model revisions.
- [ ] A `host_smoke` test performs a short **real** transcription and alignment on
      the target host and passes.
- [ ] The default suite still passes with none of this installed (INV-05).

## Explicitly not in this milestone

- Replacing or reworking M4's normalization, collapse, or rendering logic. If the
  real model forces a change there, that is a finding worth an ADR.
- Any LLM prose cleanup pass.

## What M4 already provides (read before starting)

- **The seam is finished and exercised; one implementation behind it is not.** A
  `TranscriberBundle` carries a `Transcriber` plus everything about it that reaches a cache
  key and the report — model, both revisions, aligner, `variant_digest`. Build one and
  `run_transcribe` works unchanged. `transcript/runner.py::_default_transcriber` is the
  `DEFERRED: M6b` raise site to replace, and `--fake-models` keeps working afterwards, which
  makes it the regression harness for the adapter's first run (ADR-0018).
- **ASR consumes the cached 16 kHz derivative, not the 48 kHz path** (ADR-0017). Do not
  resample: a second resampler under a cache key is the failure INV-04 names for time. Word
  times come back on that grid and are converted once, by M2's helper.
- **`TranscriptionResult.alignment_status` is stated, never inferred.** `aligned` requires
  words and words require `aligned` — the seam enforces both. Only the adapter can tell
  "the aligner ran and failed" (`segment_only`, warned about) from "no aligner ran"
  (`not_attempted`), and the transcript records whichever it says.
- **`TranscriptionResult.public_document` is the spec's lossless raw artifact.** M4 froze the
  envelope and proved the preservation contract; **this milestone owes the other half** —
  filling it from every public field of Qwen's `ASRTranscription`, tested against the real
  object. A `None` there means "this result already is its public form", which is true of a
  fake and must not be true of the adapter.
- **The ASR cache key already carries the request's identity** alongside the audio hash, the
  transcriber identity, the context hash, the language and `max_new_tokens` (ADR-0019). Add
  the backend, dtype, attention implementation and resolved revisions to `TranscriberIdentity`
  and they reach the key without a second place to disagree.
- **The truncation machinery is reused unchanged and is budget-bounded, not depth-bounded**
  (ADR-0020). What this milestone supplies is `truncated` on the result, from public backend
  metadata or a retokenized-length heuristic — never a private finish-reason API.
- **`transcript.json` and `work/transcript-records.json` are frozen at M4's close**: additive
  optional fields only (ADR-0005).
- **OQ-018 is yours to answer.** Padding for word recovery, timestamp stability across
  overlapping requests, whether a low-energy split beats the midpoint, the retry budget, and
  the text-similarity thresholds are all guesses about *this* model. The smoke test can settle
  the first four directly.

## Known risks and open questions

- Depends on **OQ-008, OQ-009, OQ-018**.
- `qwen-asr` pins Transformers and pulls Accelerate, Gradio, Flask, and Python SoX.
  All of it stays inside the `asr-qwen` group.
- If real Qwen output differs structurally from the fake, the fake was wrong.
  Fix the fake and keep both in the default suite rather than weakening assertions.
