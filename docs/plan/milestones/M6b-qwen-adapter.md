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

- **One collapse case is deferred to you, with its reproducing scores.** Three mutually
  duplicate segments scoring A=800, B=700, C=900 in canonical order leave A and C both
  retained, because a segment that has absorbed another may not itself be absorbed. It fails
  safe — both kept, both marked overlapping — so M4 left it. Decide it with real output: if
  three lavs never agree closely enough for the shape to occur, delete `collapse.py`'s claim
  that the survivor is the best source score rather than writing the resolution pass. See
  M4's closeout.

## What M5 already provides (read before starting)

- **`process` composes the transcript branch rather than reimplementing it.** M4's
  `_transcribe`, `_models` and `_write_deliverables` became public `perform_transcript`,
  `resolve_models` and `write_transcript_deliverables`, and `orchestrate.py` calls them. So
  replacing `_default_transcriber`'s `DEFERRED: M6b` raise reaches **both** `transcribe` and
  `process` through one seam — there is no second construction site to keep in step.
- **Both commands raise before any work, not partway.** `resolve_models` runs before the
  snapshot is acted on, so a host without the adapter gets ADR-0005's "this pipeline has not
  built that yet" rather than a half-finished run. Keep that ordering when the adapter lands:
  a model that fails to *load* should still fail before the first cache is written.
- **The audio branch no longer depends on you at all.** `dnd-audio mix` runs the whole
  right-hand branch with no ASR adapter and no `--fake-models`, so a host where M6b is broken
  still produces `session.mp3`. That is INV-09 enforced rather than intended (M5's closeout),
  and it means an adapter regression can never cost a session its audio deliverable.
- **`process --fake-models` is a second regression harness**, alongside `transcribe
  --fake-models`. It exercises the adapter seam with both branches running and one shared
  snapshot, which is where a model that holds a file descriptor or mutates a shared path
  shows up and a single-branch run does not.

## What M6a already provides (read before starting)

- **The environment works and OQ-008 is answered.** `torch 2.9.1+rocm7.13.0` (HIP
  `7.13.99004-3309c6114a`) resolves from AMD's gfx1151 index, installs into `.venv-rocm`
  from inside the FHS shell, and computes both bfloat16 and float32 exactly right on
  `Radeon 8060S Graphics` / `gfx1151`. `transformers==4.57.6` and `accelerate==1.12.0` are
  already locked at the versions `qwen-asr` 0.0.6 pins, **so adding `qwen-asr` should not
  relock or redownload the stack** — if it wants to, something moved and that is worth
  understanding before accepting it.
- **There are two environments.** `.venv` is the project one and never carries torch;
  `.venv-rocm` is the ROCm one. Anything touching torch runs as
  `nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm …'`. Do not sync the group into
  `.venv`: the everyday gate runs `--no-sync` against it, and that is what keeps INV-05's
  group-absent case continuously proved rather than proved once (ADR-0025).
- **`[tool.uv.sources]` only routes packages that are also *direct* members of a
  dependency list.** A transitive-only requirement resolves from PyPI regardless —
  silently, with the wrong registry simply recorded in the lock. `qwen-asr` pulls Gradio,
  Flask, `nagisa`, `soynlp` and Python SoX; if any of them brings an AMD-only requirement,
  the fix is to add it to the group *and* the sources table.
  `test_packaging.py::test_every_routed_package_is_also_a_direct_dependency` is the guard,
  and `test_everything_else_comes_from_pypi` is what catches the silent case.
- **`dnd_audio.runtime` is the seam, and it is finished.** `probe_runtime()` returns one
  frozen `RuntimeProbe` — device nodes, torch/HIP identity, gfx target, and which dtypes
  computed correctly on which device. `resolve_runtime(device=…, dtype=…, probe=…)` is a
  pure function of it and already implements every rule the spec states. The adapter calls
  both once and then honours the answer; nothing in the resolution logic should need to
  change (ADR-0026).
- **The report's `runtime` subsection exists and is empty, and filling it is yours.**
  `Provenance.runtime` is a `RuntimeProvenance` carrying python, torch, hip, device,
  device name and dtype; `ReportBuilder.record_runtime()` puts it there;
  `RuntimeResolution.provenance()` builds it. Nothing in M6a resolves a runtime during a
  run, so it is `None` everywhere today. **Add the same fields to `TranscriberIdentity`**
  so they reach the ASR cache key (INV-08) — they are defined in one place precisely so
  there is no second vocabulary to drift from.
- **Attention implementation is not in `RuntimeProvenance`.** M6b's gate asks for it and
  M6a had nothing to put there. Add it alongside the rest rather than in a second
  structure.
- **`doctor --device cuda --dtype bfloat16` is the pre-flight**, and it exercises the
  same resolver the adapter will. If it reports a healthy GPU and the adapter still fails,
  the problem is the adapter, which is a genuinely useful thing to be able to say.
- **The two gfx1151 environment variables are applied by both shells** and re-checked by
  `doctor`. Promoting them to host defaults waits for *your* smoke test — that is the
  event M6a deferred it to.
- **The `host_smoke` marker now means "needs the ROCm environment" as well as "needs the
  GPU".** A `host_smoke` test run from `.venv` reports "torch is not installed" rather
  than a GPU failure, and the assertion messages say which environment to use.

## Known risks and open questions

- Depends on **OQ-008** (answered), **OQ-009, OQ-018**.
- `qwen-asr` pins Transformers and pulls Accelerate, Gradio, Flask, and Python SoX.
  All of it stays inside the `asr-qwen` group.
- If real Qwen output differs structurally from the fake, the fake was wrong.
  Fix the fake and keep both in the default suite rather than weakening assertions.
