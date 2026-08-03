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

## Working plan

_Scratch. Replaced by the Closeout at the end of this milestone._

### How the weights arrive, and why no document needs amending

The owner directed that the ~6 GB download use the `hf` CLI and be a **one-time environment
setup step** the pipeline depends on and cannot run without. The first draft of this plan
made that a second network-capable entry point (`scripts/fetch-models.sh` calling `hf`
directly) and proposed amending the gate criterion and INV-06 to permit it. The plan review
was right to refuse that: the *spec itself* says "`models fetch` is the only command
expected to require network access for model installation", so a second authority would have
needed the spec amended too — three documents changed to avoid writing one subcommand.

**`dnd-audio models fetch --qwen` drives the `hf` CLI instead.** It is still one-time setup,
it still uses `hf`, the pipeline still hard-fails without it, and there is still exactly one
network authority. `scripts/fetch-models.sh` remains, but as a thin wrapper that runs that
command inside the FHS shell — which it must, because `hf` ships with the `huggingface_hub`
that lives in `.venv-rocm` and not in `.venv`. Nothing in the gate, INV-06, or the spec
moves.

**Installation is keyed by `(repository, resolved commit)`, not by a fixed pair of
snapshots.** A checked-in `SnapshotDescriptor` carries the default commit and every file's
size and sha256 — `SILERO_VAD`'s treatment, and the strongest available pin. But
`asr.model_revision` may name a different commit, and a manifest checked into source cannot
describe one, so:

- `asr.model_revision` / `asr.aligner_revision` are **validated as 40-character lowercase
  hex commit SHAs**. A branch name is rejected at configuration load, which is what makes
  "`process` must use the lock rather than re-resolving a moving branch" true by
  construction rather than by convention.
- The default commit verifies against the checked-in manifest. An overridden commit verifies
  against the **lock**, whose per-file digests were recorded when `models fetch` installed
  it. `read_lock`'s current "the lock is a convenience, `find_model` is the authority" holds
  for Silero and is deliberately **not** true for snapshots: for a snapshot there is nothing
  else that knows what an overridden revision should contain. That change of semantics is
  stated in ADR-0027 and tested, rather than left for someone to infer.

`hf download` writes into a staging directory and only the pinned files are moved into
place, so the loadable tree is an exact allowlist. `hf`'s own `.cache/huggingface` metadata
never reaches it, and an unpinned file that Transformers might load is a verification
failure rather than an unnoticed extra.

### What the package actually does, read before planning

From `qwen_asr` 0.0.6's published wheel. Each of these shapes a decision below:

- `Qwen3ASRModel.transcribe(audio, context, language, return_time_stamps)` returns
  `[ASRTranscription(language, text, time_stamps)]`. `time_stamps` is a `ForcedAlignResult`
  whose `.items` carry `text`, `start_time`, `end_time` **in seconds, rounded to 3 dp**.
- `(np.ndarray, sr)` is a supported input, so audio reaches the model as an in-memory array
  and never as a URL (INV-06). At `sr == 16000` no resampling happens, so ADR-0017's "do not
  add a second resampler" holds.
- **`MAX_FORCE_ALIGN_INPUT_SECONDS = 180`** in `inference/utils.py`, against
  `MAX_ASR_INPUT_SECONDS = 1200` for the text-only path. Alignment is what this pipeline
  always asks for, so 180 s is the limit that binds, exactly as **OQ-009** assumed. Source
  inspection is half of OQ-009's stated evidence; the long-segment experiment is the other
  half and is a `host_smoke` test rather than an amendment to the question.
- **`transcribe(return_time_stamps=True)` runs ASR and then alignment, and only builds
  `ASRTranscription` after alignment returns.** An aligner exception therefore destroys text
  that was already generated — which makes the gate criterion "per-segment alignment failure
  warns and retains segment-level text" *unimplementable* through that one call. The adapter
  must drive the two public operations itself. Found by the plan review; it is the single
  most valuable thing that review produced.
- **There is no public finish reason.** `_infer_asr_transformers` decodes
  `text_ids.sequences` and returns strings; nothing survives to say generation stopped at
  the ceiling. Truncation detection is therefore the retokenized-length heuristic the spec
  names as the alternative, not a choice between two available signals.
- The aligner's `tokenize_space_lang` strips punctuation from word text, so aligned word
  texts are not a substring partition of the segment text. M4's `comparison_key` already
  tolerates that; the records must not pretend otherwise.

### Order of work

1. **Packaging** — `pyproject.toml`, `tests/test_packaging.py`. Add `qwen-asr==0.0.6` to the
   `asr-qwen` group; its requirements are the `transformers`/`accelerate` pins M6a already
   locked plus `nagisa`, `soynlp`, `qwen-omni-utils`, `librosa`, `soundfile`, `sox`,
   `gradio`, `flask`, `pytz` — all PyPI, none AMD-only, so no new `[tool.uv.sources]` entry
   is expected. `uv lock` must not move torch, transformers or accelerate; if it wants to,
   stop and understand why. `test_every_routed_package_is_also_a_direct_dependency` is the
   guard M6a left for exactly this moment.
2. **The snapshot store** — `src/dnd_audio/models.py`, `tests/test_models.py`.
   `SnapshotDescriptor`, `snapshot_dir`, `find_snapshot`, `require_snapshot` (fatal, and the
   message names `models fetch --qwen`), and snapshot entries merging into the existing
   lock. `find_snapshot` is exact in **both** directions: every pinned file present at the
   pinned size and digest, and **no unpinned file in the tree** — an extra `config.json` or
   custom-code file is something Transformers would happily load, so its presence is a
   verification failure, not an unnoticed extra. Descriptors `QWEN3_ASR` and
   `QWEN3_ALIGNER`, at the default commits; an overridden commit verifies against the lock.
3. **Fetching** — `dnd-audio models fetch --qwen`, plus `scripts/fetch-models.sh` as the
   FHS-shell wrapper. `models fetch` stays the single network authority and shells out to
   `hf download <repo> --revision <commit> --local-dir <staging>`; only the pinned files
   move into place. `models plan` prints repository, commit and target with no network, so
   the wrapper cannot drift from the pin.
4. **The adapter** — `src/dnd_audio/transcript/qwen.py`, `tests/test_qwen_adapter.py`. The
   `QwenBackend` protocol is the seam, one level below `Transcriber`, for the reason
   `activity/silero.py::OnnxSession` is: the properties that matter here are properties of
   *this module*, and a fake `Transcriber` would replace the code under test. **Three
   operations, not two** — `transcribe_text`, `align`, `count_tokens` — because alignment
   must be able to fail on its own without taking the text with it. Torch and `qwen_asr` are
   imported lazily inside functions (INV-05).

   The timestamp decoder is strict and is the one place seconds cross to samples:
   `request.audio.start_sample + to_samples(Fraction(str(item.start_time)), 16_000)`.
   `Fraction(str(...))` rather than `Fraction(float)` — the package rounds to 3 dp, so the
   decimal string is the value it meant and the binary float is not (INV-04). Rebasing on
   `audio.start_sample` is not a detail: a request starting at sample 1 600 000 whose words
   came back at 0.5 s would otherwise land near session zero and be dropped by M4's
   ownership rule. Non-finite, negative, `end < start`, non-monotonic, or out-of-window
   items are a **recoverable per-segment alignment failure** — `segment_only` plus a warning
   — never an exception that aborts a session's transcript. `end == start` is *not* in that
   list: the aligner quantizes to 80 ms, so a shorter word comes back zero-length and is
   widened to one sample rather than treated as corruption. This plan said `end <= start`
   until the first real transcription proved it wrong; see ADR-0028.
5. **Identity, runtime, report** — `TranscriberIdentity` gains a **nested
   `runtime: RuntimeProvenance | None`** rather than a parallel row of scalars, so python,
   torch, hip, device, device name and dtype all reach the cache key from the one place M6a
   defined them (INV-08); a second flat vocabulary is exactly what M6a's closeout said not
   to build. `RuntimeProvenance` gains `attention`, which that closeout records as having
   had no home. The identity also gains `package_version`, `transformers_version` and
   `truncation_margin_tokens`, which are the transcriber's and not the device's.
   `AsrConfig` gains `truncation_margin_tokens` only — **attention is hard-coded to SDPA**,
   because the spec asks for SDPA and a knob with no second value and no consumer is
   interface the milestone's non-goals do not ask for.

   `max_new_tokens` is **bound to the constructed backend** and every request is asserted
   against it. `Qwen3ASRModel` takes the ceiling at construction while M4 puts it on each
   request; a bundle whose identity says 512 over a backend still generating 1024 would key
   a different cache entry for identical model behaviour.

   `resolve_models` gains `config` (both call sites — `run_transcribe` and `orchestrate.py`)
   and `Models` gains `runtime`, so the resolver's warnings survive into the report. A
   `device: auto` run that fell back to CPU must carry that warning prominently; dropping it
   is how an operator discovers the fallback from a run that took nine hours.
   `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set **before** `qwen_asr` or
   `transformers` is imported, not around `from_pretrained` — the libraries read those at
   import.
6. **Host smoke test** — `tests/test_qwen_smoke.py`, marked `host_smoke`. Real weights, real
   device. Beyond one transcription and alignment it must answer what it claims to:
   - **OQ-018 (1), padding** — an utterance with known speech at both ownership boundaries,
     submitted padded and clipped, compared. A generic short utterance shows nothing here.
   - **OQ-018 (2), timestamp stability** — two overlapping requests over the same audio.
   - **OQ-018 (3), truncation** — a deliberately low `max_new_tokens`, then whether the
     low-energy split resolves it.
   - **OQ-009** — a >180 s waveform through the backend directly, to observe the package
     chunking rather than to quote its constant.
   - **OQ-022 (new), determinism** — the same request twice with the cache bypassed, text
     and word times compared **exactly**. `transcript.json` is a deterministic artifact
     under INV-02; if ROCm SDPA is not reproducible across cold runs, that claim is false
     the moment a cache is cleared. Sampling is disabled explicitly rather than inherited
     from whatever generation config the snapshot ships.
7. **Ledger** — ADR-0027 (snapshot pinning, the exact-tree rule, and why the lock is the
   authority for an overridden revision), ADR-0028 (the adapter, its three-operation seam,
   and truncation by retokenization). OQ-009 answered from source **and** experiment; OQ-018
   items 1–3 answered; **OQ-022 raised** and cited from the adapter. No invariant, charter
   criterion, or spec sentence needs amending. M4's deferred three-way collapse case
   **stays deferred**: one single-track smoke utterance is not evidence about three lavs
   agreeing, and that is the same real-session evidence OQ-017 is waiting on.

### Completion gate → named proof

**Reconciled in the verify phase**, where several of the names below turned out to be tests
this plan had imagined rather than written. The table now names what exists and was executed;
where a planned proof was dropped rather than renamed, the row says so and why.

| Criterion | Proof |
| --- | --- |
| Network only at fetch; snapshot commits; local lock; caches outside sessions | `TestThePinnedDescriptors`, `TestTheDirectoryLayout`, `TestTheLock`, `tests/test_network_blocked.py` |
| `process` uses the lock, not a moving branch | `TestAConfiguredRevisionVerifiesAgainstTheLock`, `test_a_revision_that_is_not_a_full_commit_is_rejected`. **No `test_absent_lock_is_fatal_even_with_valid_bytes`**: ADR-0027 makes the *checked-in* manifest authoritative for the pinned revision, which is strictly stronger than a local lock — see the verify phase's rejected findings |
| Offline execution | `TestOfflineMode`, `test_offline_mode_is_set_before_the_backend_is_imported` |
| Explicit model/aligner revisions configurable | `TestAConfiguredRevisionVerifiesAgainstTheLock`, `test_a_configured_revision_can_actually_be_installed` |
| Transformers backend, bfloat16, `cuda:0`, SDPA | `TestAttentionIsFixed`; `test_the_loaded_models_really_are_bf16_sdpa_on_the_gpu` (`host_smoke`, read off the loaded modules rather than off this project's constants) |
| Audio never a URL or path (INV-06) | `TestAudioNeverLeavesAsAPathOrUrl` |
| English forced, configurable; glossary via context; absence never blocks | `TestLanguageAndContext` |
| Aligner word times; per-segment failure warns and keeps text | `TestTimestampDecoding`, `TestAlignmentFailureKeepsTheText`, `TestMalformedAlignmentIsRecoverable`, `test_the_segment_survives_the_run_warns_and_the_exit_is_clean` |
| Cache identity complete; atomic; incomplete never hits | `test_the_document_names_every_component`, `test_every_part_of_the_runtime_moves_the_key`, `test_the_package_versions_move_the_key`, `test_the_truncation_margin_moves_the_key`; existing `AsrCache` tests |
| `max_new_tokens` 1024; changing it invalidates the cache | `test_max_new_tokens_moves_the_key`, `TestTheBoundCeiling` |
| Truncation from public metadata or retokenization, never a private API | `TestTruncationHeuristic`, `test_no_private_finish_reason_or_generation_path_is_used` |
| Bounded windows; UMA caveat documented | `TestBoundedMemory`, `tests/test_memory.py`; `README.md` |
| Report records python, qwen-asr, transformers, torch, HIP, device, dtype, attention, revisions | `test_report_records_the_whole_asr_stack` — planned, missing from the first pass, written in the verify phase after both reviewers found the hole |
| A real transcription and alignment on the target host | `tests/test_qwen_smoke.py` (`host_smoke`), 16 passed on the device |
| The default suite still passes with none of it installed | `./scripts/gate.sh` from `.venv`, **and** the whole suite from `.venv-rocm` |

### Invariants at risk, and what stops it

- **INV-05** — `qwen.py` must not import torch at module scope, and a subprocess test must
  shadow `torch` on `PYTHONPATH`: neither autouse fixture can see across a process boundary
  (M6a). The suite is run from `.venv-rocm` as well as `.venv`, which is the only place a
  breach shows.
- **INV-06** — the adapter is the enforcement point; the backend fake refuses anything that
  is not an array. `models fetch` stays the only thing that opens a socket.
- **INV-07** — one request's audio in memory at a time already holds in `asr.py`; the
  adapter must not accumulate, and it is proved over the composed path rather than over one
  function.
- **INV-08** — every identity component asserted by *name*, never by "some hash changed".
  The nested runtime is what keeps the component list from drifting from the report's.
- **INV-04** — exactly one seconds-to-samples conversion, through `to_samples`, fed from a
  decimal string rather than a float.
- **INV-02** — real-model reproducibility is an assumption, not a fact, until the smoke test
  measures it. That is OQ-022, and it is raised rather than assumed.

### Deliberately not doing

The charter's non-goals, unchanged: no rework of M4's normalization, collapse or rendering
(if real output forces one, that is a finding worth an ADR); no LLM prose-cleanup pass; no
vLLM and no FlashAttention.
