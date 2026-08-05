# M6b — Qwen ASR adapter and forced aligner

**Status:** closed
**Depends on:** M4, M6a
**Spec sections:** Milestone 4 (Qwen specifics); Target-host runtime; Tests and
acceptance criteria 12, 14

## Goal

A real `Transcriber` implementation backed by `Qwen/Qwen3-ASR-1.7B` and
`Qwen/Qwen3-ForcedAligner-0.6B`, with revision-locked offline model loading and a
complete cache identity — dropped in behind the interface M4 already exercises.

## Completion gate

- [x] `models fetch` is the only command that touches the network for models. It
      resolves mutable Hugging Face names to exact snapshot commit revisions and
      writes a local model lock. Caches live outside session directories and out of
      version control.
- [x] `process` uses that lock rather than re-resolving a moving branch, and
      production processing runs under Hugging Face offline mode.
- [x] Explicit model/aligner revisions may be set in configuration.
- [x] Transformers backend, `torch.bfloat16`, `cuda:0`, SDPA attention.
- [x] Audio reaches the model only as local paths or arrays — never a URL or API
      (INV-06).
- [x] English forced by default, language configurable. `glossary.txt` passed via
      Qwen's context parameter when present; its absence never blocks a run.
- [x] Forced aligner produces word times; per-segment alignment failure warns and
      retains segment-level text.
- [x] Cache identity: exact segment-audio hash, model and aligner identifiers **and
      resolved revisions**, context hash, language, backend, dtype, attention
      implementation, and every output-affecting generation/alignment parameter
      including `max_new_tokens` (INV-08). Atomic writes; incomplete entries never hit.
- [x] `max_new_tokens` defaults to 1024, not the upstream wrapper's 512. Changing
      it invalidates the cache — tested.
- [x] Truncation detection uses public backend metadata or retokenized-length
      heuristics, never a private Qwen finish-reason API. The split/retry machinery
      from M4 is reused unchanged.
- [x] Processing runs in bounded windows; documented that it must not run alongside
      a heavy ComfyUI or large-LLM workload on this UMA host (INV-07).
- [x] Report records Python, `qwen-asr`, Transformers, Torch, HIP runtime, device,
      dtype, attention implementation, and resolved model revisions.
- [x] A `host_smoke` test performs a short **real** transcription and alignment on
      the target host and passes.
- [x] The default suite still passes with none of this installed (INV-05).

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
- **Three things M6a's verify phase learned the hard way, all of which you will meet.**
  A probe must catch `Exception`, not `ImportError` — a ROCm build with a missing shared
  library raises `OSError` from the loader, and the adapter loading a model has the same
  exposure. A subprocess is where `conftest.py`'s Torch guard cannot look, so a subprocess
  test touching the adapter must shadow `torch` on `PYTHONPATH`
  (`tests/test_runtime.py::shadow`). And after fixing a review finding, **revert the fix
  and watch a test fail** — one of M6a's fixes shipped with no test at all and only a
  mutation run caught it.
- **The `host_smoke` marker now means "needs the ROCm environment" as well as "needs the
  GPU".** A `host_smoke` test run from `.venv` reports "torch is not installed" rather
  than a GPU failure, and the assertion messages say which environment to use.

## Known risks and open questions

- Depends on **OQ-008** (answered), **OQ-009, OQ-018**.
- `qwen-asr` pins Transformers and pulls Accelerate, Gradio, Flask, and Python SoX.
  All of it stays inside the `asr-qwen` group.
- If real Qwen output differs structurally from the fake, the fake was wrong.
  Fix the fake and keep both in the default suite rather than weakening assertions.
## Closeout

### What works end to end

**`dnd-audio transcribe <session>` produces a real transcript from real speech, with no
`--fake-models`.** The last `DEFERRED: M6b` raise in the project is gone. On the target host:

```
  11 segment(s) across 2 speaker(s), 3 collapsed as duplicates, 8 marked as overlap
  warn  candidate_transcribed_to_nothing: 1 retained activity candidate(s) produced no text

**[00:00:10.750] Operator (mic 2):** a first transmitter Hello One two three Here we go
```

and the report carries every component the gate's provenance list names, measured rather
than declared:

```
runtime         sdpa · cuda:0 · bfloat16 · Radeon 8060S Graphics
                torch 2.9.1+rocm7.13.0 · HIP 7.13.99004-3309c6114a · python 3.12.13
asr             Qwen/Qwen3-ASR-1.7B @ 7278e1e70fe206f11671096ffdd38061171dd6e5
aligner         Qwen/Qwen3-ForcedAligner-0.6B @ c7cbfc2048c462b0d63a45797104fc9db3ad62b7
asr_package_version 0.0.6 · transformers_version 4.57.6 · truncation_margin 16
```

Every one of those reaches the ASR cache key through one `TranscriberIdentity`, so a Torch
upgrade, a HIP upgrade, a `transformers` bump or a dtype change re-runs the work instead of
serving a transcript produced under different arithmetic (INV-08).

**Weights arrive once, by commit, and are verified in both directions.**
`./scripts/fetch-models.sh` enters the FHS shell and runs `dnd-audio models fetch --qwen`,
which drives the `hf` CLI. `dnd-audio models plan` prints the pin without touching anything.
Installation is keyed by `(repository, resolved commit)`, staged outside the target, and the
tree is checked to be **exactly** the manifest — every pinned file at its pinned size and
sha256, and no unpinned file anywhere in it, because Transformers loads a *directory* and
anything inside one is a file a model may read.

**`process` still produces `session.mp3` when none of this is installed.** That stopped being
hypothetical the moment the adapter existed: until M6b a missing ASR model meant "not built
yet" and correctly stopped the run; now it means "this machine has no ASR", which is an
ordinary transcription failure and exactly what INV-09 exists for.

**The default suite runs with none of it present**, on a machine with no GPU, no weights and
no Torch — which is what keeps INV-05 continuously proved rather than proved once.

### Tests and commands run, with results

```
$ ./scripts/gate.sh
  pass  system dependencies      pass  pytest (offline, cpu)    2294 tests
  pass  ruff check               pass  lock is current
  pass  ruff format              pass  placeholder scan         148 files
  pass  type check               pass  plan consistency         11 milestones, 13 INV,
GATE PASSED                                                     22 OQ, 29 ADR
```

```
$ nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync \
    pytest -m "not host_smoke and not allow_network" -q'
2294 passed in 36.08s

$ nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync \
    pytest -m host_smoke -n 0 -q -s'
16 passed, 2296 deselected in 47.04s

  OQ-018(1) identical: True
  OQ-018(2) 20/22 shared word(s) paired by the rule; worst 0 ms
  OQ-018(3) ceiling=8: truncated=True text='Testing a first transmitter. Hello, one'
```

**Running the default suite from `.venv-rocm` is not optional and no gate does it.** It is
where six tests that asserted a property of the *machine* rather than of the code were found,
and where the adapter was caught starting HIP before checking whether the weights existed.

**Mutation-tested rather than assumed.** Every fix from the verify phase was reverted and the
suite re-run: the zero-length widening, the INV-09 soft failure, the lock repair, the symlink
guard, the repository guard, `re.fullmatch`, the `align_failure` flag, `verify_tree`'s size
check and unpinned-file scan, and `_to_sample`'s window clamp. All were caught. One was not
on the first attempt — see *Notes*.

New test files: `tests/test_snapshots.py` (63), `tests/test_qwen_adapter.py` (71),
`tests/test_qwen_smoke.py` (16, `host_smoke`).

### Decisions made (→ ADRs)

- **[ADR-0027](../decisions/0027-pinning-hugging-face-snapshots.md) — pinning Hugging Face
  snapshots.** `models fetch --qwen` drives `hf`; installation is keyed by
  `(repository, resolved commit)`; only 40-hex revisions are accepted anywhere; the lock is
  authoritative for an overridden revision while the checked-in manifest is authoritative for
  the pinned one; the installed tree is an exact allowlist in both directions.
- **[ADR-0028](../decisions/0028-the-qwen-adapter-seam.md) — the Qwen adapter seam.** A
  three-operation backend protocol below `Transcriber`; one strict timestamp decoder through
  `determinism.to_samples`; truncation by retokenization because 0.0.6 exposes no finish
  reason; attention hard-coded to SDPA; `RuntimeProvenance` nested in the identity rather than
  flattened beside it. Amended during the verify phase — see *Notes*.

### Assumptions made and open questions raised

- **OQ-009 answered.** `MAX_FORCE_ALIGN_INPUT_SECONDS = 180` in the package's own source, and
  **it is not on this project's route at all**: the chunking limit is reached only through
  `transcribe(return_time_stamps=True)`, and this adapter makes the two public calls
  separately. `max_segment_s ≤ 120` was the right cap for a reason that turned out to be the
  wrong one.
- **OQ-018 items 1–3 answered**, item 4 and half of item 3 still open.
  - *(1) Padding.* `pad_ms = 500` is not the constraint — a hard clip and a padded request
    returned identical text. What the measurement found instead is that **`activity.vad.pad_ms`
    = 30 is costing real words**: the model heard `'Testing a first transmitter…'` and the
    transcript recorded `'a first transmitter…'`, because the aligner places "Testing" 50 ms
    before the VAD candidate's ownership interval begins and M4's rule correctly drops it.
    Five of eleven retained segments lost their opening word. That is M3's number, registered
    under OQ-017, and 47 seconds of one person testing microphones is not evidence to retune a
    detector on.
  - *(2) Timestamp stability.* M4's stitch rule pairs **77 of 80** shared words across four
    recordings, worst in-pair disagreement 400 ms. The three misses are named in the entry;
    one is real and is a very short word one 80 ms quantization step out.
  - *(3) Truncation.* The retokenized-length heuristic fires on a genuinely cut-off response
    and not on the same audio at the default ceiling. Whether a *low-energy split* resolves a
    truncation better than a midpoint is **unmeasured** and stays open: an eight-token ceiling
    truncates everything, and a natural truncation needs an utterance long enough to exhaust
    1024 tokens, which this recording does not contain.
- **OQ-022 raised and answered.** Qwen inference on this ROCm stack is reproducible in process
  and across cold processes, so **INV-02 stands as written** and needs no amendment naming a
  model boundary. Sampling is disabled explicitly rather than inherited from the snapshot's
  `generation_config.json`, so the claim is about greedy decoding and not about a temperature
  that happened to be zero.
- **OQ-012 updated** with the operator's report that the LTC jam between receivers did not
  take on the sample capture — which is why the smoke test measures one recording at a time
  and never concatenates across the receiver pairs.

### Notes for future implementors

**The aligner emits zero-length items, and treating them as corruption destroys most word
times.** It quantizes to `timestamp_segment_time` — 80 ms on this model — so *any* word
shorter than one step comes back with `end == start`. On the very first real utterance this
project transcribed, that was the word "a", and the decoder's `end <= start` rule threw away
all fifteen word times in the segment. It would have done that to most segments in most
sessions and the only symptom would have been a transcript with no word times beside a warning
saying alignment failed. Such an item is now widened to one sample. **ADR-0028 and this
charter both said `end <= start` for the length of a milestone after the code stopped
agreeing** — the code review caught the drift, and an implementor following the ADR would have
reintroduced the loss. If you change this rule, change it in all three places.

**Run the default suite from `.venv-rocm` before you believe it.** Six tests here asserted
"the ASR runtime is unavailable" by *running* `transcribe` and expecting failure — true on
`.venv`, false on `.venv-rocm`. The fix is to configure the absence rather than inherit it:
`session_without_asr_models` pins a revision nothing has installed, so the test asserts a
property of the code on every machine. The same run caught `_default_transcriber` probing the
GPU before checking for weights, which tripped the `no_torch_import` guard. **Weights before
hardware** is now load-bearing in that function and the comment says so.

**A fix can ship with no test, and the only way to know is to revert it.** M6a's closeout says
this and it happened again: the `--fake-models` guard in `_resolve_or_defer` was stated twice,
once per exception handler, and reverting one copy failed nothing. Consolidating the rule into
one statement made both halves load-bearing. Mutation-test every fix, not just the feature.

**A measurement of a rule needs the rule's own notion of sameness for its numerator and
something independent of the rule for its denominator.** OQ-018(2) was measured wrongly twice,
in opposite directions. Matching shared words by text alone reported five outliers of 2–9
seconds — the recording says "testing" and "transmitter" more than once, so a text-only key
paired the first occurrence in one window with the second in the other. Fixing that by pairing
the way the stitch rule pairs introduced the opposite error: selecting only words that already
overlap and then reporting how closely they agree measures nothing, because a word that
drifted far enough to stop overlapping — *the failure under investigation* — leaves the sample.
"20 paired, worst 0 ms" was true and would have been equally true of a model getting half of
them badly wrong.

**A bound measured on one sample file is a bound on which file sorts first.** The replacement
delta bound for OQ-018(2) was drafted at 250 ms and passed — because TX01 sorts ahead of TX03,
whose worst is 400 ms. Replacing the jam corpus would have turned the suite red for a reason
nobody could reconstruct. Its WAVs are *discovered* rather than named precisely so new
recordings re-run these measurements instead of silently skipping them; the corollary is that
any threshold set from them has to hold for all of them.

**Configuration fields that reach provenance must be either honoured or refused.**
`asr.model` existed because the spec's `session.yaml` has it, was ignored by
`_default_transcriber`, and was written verbatim into the cache key and the ingest report — so
naming any other repository produced two artifacts asserting that weights which were never
loaded produced the transcript, with nothing downstream able to detect it. It is now refused
at configuration load, pointing at `model_revision` as the thing that *is* adjustable. The
general shape is worth remembering: a field the code does not read is not inert if something
else records it.

**`--fake-models` is an assertion about what ran and is not softenable.** Teaching
`_resolve_or_defer` to turn a model failure into the transcript branch's error is right for a
host with no weights (INV-09) and wrong under `--fake-models`, where one call builds *both*
seams from `fake-models.json`. Softened, a missing file left activity to build the **real**
Silero detector, and an operator who explicitly asked for fake models got a real MP3 off real
detection with only a failed transcript stage as a hint.

**`hf` and symlinks.** `verify_tree` originally skipped anything answering `is_dir()`, and a
symlink *to* a directory answers it truthfully while `rglob` declines to descend into it — so
an unpinned symlinked directory full of weights passed a check whose entire claim is that the
tree is exactly the manifest, while a plain unpinned file was refused. `hf download
--local-dir` is a tool that has created symlinks into a shared cache. A pinned file may still
*be* a symlink, because content is the rule; an unpinned anything may not.

**The package has no finish reason and `transcribe(return_time_stamps=True)` aligns
internally.** Hence three separate backend operations rather than one call: transcribe, align,
count tokens. Calling the timestamp path would chunk at 180 s, discard the finish signal
anyway, and hide the alignment failure the gate requires be reported separately.

**`uv` will backtrack numpy to 2021.** `qwen-asr` → `librosa` → `numba` requires `numpy<2.5`;
the lock had 2.5.1; old numba's metadata declares no ceiling, so the resolver happily selected
numba 0.53.1, whose sdist refuses to build on Python 3.12. The ceiling lives in **base**
dependencies with a comment saying why, because uv resolves one numpy for the whole lock.

**`[tool.uv.sources]` routes only direct dependency-list members** (M6a's lesson, still true).
Nothing `qwen-asr` pulls needed routing, but `test_every_routed_package_is_also_a_direct_
dependency` is what would catch it.

### Deviations from this charter, and why

- **The weights are a one-time setup step driven by `hf`, at the operator's direction** — but
  behind `dnd-audio models fetch --qwen`, not beside it. The first draft of the plan made
  `scripts/fetch-models.sh` a second network-capable entry point and proposed amending the
  gate criterion *and* INV-06. The plan review refused that, correctly: the spec itself says
  `models fetch` is the only command expected to require network access, so a second authority
  would have meant amending three documents to avoid writing one subcommand. The script
  remains as a thin FHS-shell wrapper, because `hf` ships with the `huggingface_hub` that lives
  in `.venv-rocm`. **Nothing in the gate, INV-06 or the spec moved.**
- **`asr.model` and `asr.aligner` accept exactly one value each.** The charter reads them as
  configurable identity; this build carries snapshots for two repositories and no command can
  install a third. Refusing is the only honest option — see *Notes*.
- **`models fetch` and `models plan` grew `--asr-revision` / `--aligner-revision`.** Not in the
  charter, and required by it: "explicit model/aligner revisions may be set in configuration"
  is not met if no command can install one.
- **The charter's proof table named tests that were never written.** Reconciled during verify;
  where a planned proof was dropped rather than renamed, the table now says which and why.
- **M4's deferred three-way collapse case stays deferred.** One operator testing microphones
  one at a time is not evidence about three lavs agreeing closely enough for the shape to
  occur. That is the same real-session evidence OQ-017 waits on.
- **Non-goals held.** No rework of M4's normalization, collapse or rendering; no LLM prose
  cleanup; no vLLM; no FlashAttention.

### Downstream charters updated

- **A real-table capture** — now also the evidence that settles `activity.vad.pad_ms`, which
  OQ-018(1) found is dropping the first word of roughly half the segments. The symptom to look
  for is a transcript quietly missing an utterance's opening word rather than anything that
  raises.
- **M11 (live Session Zero validation)** — owns OQ-018(4), the text-similarity thresholds, and
  the unmeasured half of item 3: whether a low-energy split resolves a truncation better than
  a midpoint needs an utterance long enough to exhaust 1024 tokens.
- **M6a's deferred promotion of the two gfx1151 environment variables to host defaults** was
  waiting on this smoke test. It has now run repeatedly and clean; the decision is the
  operator's to make against the host's NixOS configuration.

### Next smallest step

**Superseded closeout direction.** At M6b close, the next step was a better synchronized
real-table capture covering OQ-003, OQ-007, OQ-012, OQ-017, and OQ-018(4). The later jam,
minimal-acoustic, and marker-bench evidence answered or retired the first three (ADR-0043).
M11 now owns only the ordinary-play calibration in OQ-017 and OQ-018. The retained jam corpus
is at `/data/dnd-audio/2026-08-03-jam-capture/`; `tests/test_qwen_smoke.py` discovers its WAVs
by glob, so replacing its contents re-runs every measurement in this closeout without a code
change.
