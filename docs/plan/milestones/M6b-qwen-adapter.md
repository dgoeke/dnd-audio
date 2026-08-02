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

## Known risks and open questions

- Depends on **OQ-008, OQ-009**.
- `qwen-asr` pins Transformers and pulls Accelerate, Gradio, Flask, and Python SoX.
  All of it stays inside the `asr-qwen` group.
- If real Qwen output differs structurally from the fake, the fake was wrong.
  Fix the fake and keep both in the default suite rather than weakening assertions.
