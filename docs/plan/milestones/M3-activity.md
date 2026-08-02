# M3 — Conservative speech activity and bleed rejection

**Status:** in progress
**Depends on:** M2
**Spec sections:** Milestone 3; Milestone 5 (activity graph definition); Tests and
acceptance criteria 5, 15

## Goal

Per-track VAD, a conservative pre-ASR bleed gate using lag-tolerant normalized
cross-correlation, and a versioned, model-independent activity/attribution graph
that both the transcript branch and the automixer consume.

## Completion gate

- [ ] VAD runs per 16 kHz track behind an `ActivityDetector` protocol, with a
      deterministic fake / ground-truth-mask implementation used by the default
      suite (INV-10). Synthetic noise is never expected to trigger a specific
      learned Silero release.
- [ ] Silero model artifact pinned by upstream release and commit **and by content
      hash**, with the runtime and the calling interface pinned too, and loaded
      locally — no unpinned runtime `torch.hub` fetch. Identity appears in cache keys
      and the report (INV-08). CPU or ONNX is the baseline so it does not contend with
      ASR for unified memory. _(Amended during the start phase, with the spec, after
      independent review: the original wording said "Silero package and model
      artifact/revision". See ADR-0013.)_
- [ ] Nearby speech regions merged and boundaries padded; all thresholds
      configurable; VAD probabilities and decisions persisted for debugging.
- [ ] Cross-channel similarity uses normalized speech-band cross-correlation over a
      configurable bounded lag (default ±30 ms), **not** zero-lag correlation. Both
      the peak correlation and its selected lag are recorded.
- [ ] Bleed suppressed only when another track is convincingly stronger *and* the
      signals are strongly related. Ambiguous candidates are kept by default.
- [ ] Source scoring combines track-relative speech level, VAD confidence,
      cross-track dominance, and correlation evidence — never a single global
      loudness comparison. The scoring function is isolated and its diagnostics
      appear in `ingest-report.json`.
- [ ] Tests: solo attribution, genuine two-person overlap survives, quiet bleed is
      suppressed to the right track, and correlated bleed delayed within the lag
      window is still detected with its peak lag reported.
- [ ] **The activity graph schema is checked in, versioned, and frozen** (INV-09).
      It is model-independent: nothing text-derived may enter it.
- [ ] Every retained candidate has a deterministic ID derived from sorted source
      identity and time, not completion order (INV-02).

## Explicitly not in this milestone

- Post-ASR duplicate collapse. That is text-dependent and belongs to M4.
- Generic speaker diarization or clustering. Attribution is "the person mapped to
  that track" for the MVP baseline.
- Gain envelopes. That is M5 consuming this graph.

## What M2 already provides (read before starting)

- **The 16 kHz audio VAD consumes already exists, cached and byte-stable.** Each track's
  `DerivativeRecord` in `work/timeline.json` names its `relative_path` under
  `work/cache/audio/16000/`. Read it through `timeline.pcm.open_pcm`; do not resample
  anything yourself. `DerivativeCache.get()` takes the expected output length, because a
  cache entry that is the wrong length must be a miss rather than a subtly short track.
- **The 48↔16 kHz mapping is a settled contract — use it, do not re-derive it.**
  `timeline.resample.to_source_sample` and `to_derivative_interval`. Output sample `k`
  corresponds to input sample `3k` exactly (the FIR's group delay divides by the decimation
  factor). The reverse direction lands between grid points, so an interval **floors its
  start and ceils its end**. Rounding both ends the same way shrinks a speech region by up
  to two samples, which is how a word loses its first phoneme. M3 is this contract's first
  real consumer.
- **Silence has three causes and they are deliberately indistinguishable** to a
  `TrackReader` caller: before the track started, inside a real gap, and after it stopped.
  A VAD sees zeros in all three. Every track answers to the session's aligned
  `duration_samples`, so do not special-case a track that ended early.
- **The lag-tolerant normalized cross-correlation this milestone's bleed gate needs already
  exists.** `timeline.syncqa.measure_lag` returns the peak correlation and the lag it
  occurred at, over a bounded lag window, normalized by both signals' energy — an
  unnormalized correlation ranks tracks by volume and calls the loudest one the best match.
  Reuse it rather than writing a second one; if M3 needs a variant, extend it there.
- **INV-08 for the activity cache.** Whatever key M3 builds must carry
  `TIMELINE_SEMANTICS_VERSION` and the derivative's own `cache_key`, not just the source
  hashes: a placement fix moves a chunk without changing a source byte, and a stale
  activity graph aligned to a timeline that has moved is not obviously wrong.
- **Commit a cache entry only after INV-01 has been re-verified**, never at publish time.
  M2 shipped the other ordering and it meant a run that correctly *failed* on a changed
  source still left a poisoned entry keyed on the bytes it read. See M2's closeout.
- **`timeline.json`'s schema is frozen at version 1** — additive optional fields only.
  Every interval in it is half-open and there are no floats anywhere in the document.

## Known risks and open questions

- Depends on **OQ-010**.
- This is the milestone where a "reasonable" simplification does the most damage:
  a global loudest-wins rule passes casual testing and erases quiet speakers during
  real overlap. Losing real overlapped speech is worse than extra ASR compute.
- The graph contract is consumed by two downstream milestones. Changing it later
  means redoing both. Spend the time on it here.

## Working plan

_Scratch section, written during the start phase and replaced by the Closeout at the end._
_Revised after independent review — `../reviews/M3-plan-20260802-1600.md` carries the ten
findings and the response to each. What follows is the plan as amended._

### Feasibility established before planning (OQ-010)

Three probes, because OQ-010 is the only real unknown in this charter:

- The `silero-vad` PyPI package hard-depends on `torch` and `torchaudio`. Unacceptable in
  the environment the default suite runs in (INV-05), and it would pre-empt M6a's AMD wheel
  index and per-package sourcing.
- Its ONNX protocol needs neither. Inputs are `input` (1, 64 context + 512 samples),
  `state` (2, 1, 128), and `sr` (int64); outputs are a probability and the next state.
- The real model runs under `onnxruntime` 1.28.0 on CPU driven by a plain NumPy loop.
  The artifact at tag `v6.2.1`, commit `7e30209a3e901f9842f81b225f3e93d8199902b1`, is
  byte-identical to the one inside the wheel: sha256
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

So the plan is `onnxruntime` plus a commit-pinned model *file*, not the `silero-vad`
package — which required amending the spec as well as this charter (ADR-0013).

### Decisions recorded before any code

- **[ADR-0012](../decisions/0012-the-activity-graph-contract.md)** — the frozen graph: units,
  grids, orderings, per-pair evidence, the suppressing *candidate* named rather than the
  track, and INV-09 enforced by a field allowlist over the checked-in schema.
- **[ADR-0013](../decisions/0013-silero-through-onnx-runtime.md)** — Silero through ONNX
  Runtime, pinned by commit and content hash, no Torch. Amends the spec twice.
- **[ADR-0014](../decisions/0014-the-conservative-bleed-gate.md)** — suppression needs a
  score margin **and** correlation **and** a track-relative level below the veto.
- **[ADR-0015](../decisions/0015-activity-as-a-stage-command.md)** — `activity` is a stage
  command; a composed run writes one report.
- **[ADR-0016](../decisions/0016-stage-scoped-cache-configuration.md)** — cache identity
  carries a stage-scoped projection of the configuration, not the whole of it.
- **OQ-017** is registered *now*, before the first default threshold, and every defaulted
  field cites it. Its evidence is H2 or the first real session — not H1, whose two-minute
  metadata fixture cannot tune a bleed threshold.

### What gets built, in the reviewer's recommended order

**A. Contract and records first** — M4 and M5 both inherit them.

1. `activity/__init__.py` — `ACTIVITY_SEMANTICS_VERSION`, `ACTIVITY_RELATIVE_PATH`
   (`work/activity.json`), cache dirnames, the 512-sample frame constant.
2. `artifacts/activity.py` — `ActivityGraph`, `ActivityTrack`, `ActivityCandidate`,
   `CandidateEvidence`, `DetectorIdentity`, `ActivityProvenance`, notes and decisions,
   exactly as ADR-0012 specifies: no floats anywhere, per-mille and millibel integers,
   half-open intervals on both grids, `lag_derivative_samples` named for the grid it was
   measured on, evidence one record per compared pair sorted by the competitor's id, and
   every ordering stated in a validator rather than left to the builder.
3. `schemas/activity.schema.json` through the existing `schema_export.schema_documents()`.
4. `FixtureTruth.leaky_activity_spans()` — genuine spans and bleed spans **merged per track
   and sorted**, never `activity_spans() | bleed_spans()`, which is a dict union that
   replaces `tx-d`'s and `tx-e`'s genuine overlap with bleed and would have quietly deleted
   the case the gate is about.

**B. Configuration and cache identity.**

5. `config.py` — `ActivityConfig` grows `vad`, `bleed`, and `scoring`; every default cites
   OQ-017. `stage_config` / `stage_config_hash` land here with their projection table
   (ADR-0016), and M2's `derivative_identity` moves onto the `derivative` projection.
6. Two cache identities, not one (INV-08):
   - **detection**, per track: the derivative's cache key, the detector and model identity,
     the `detection` configuration projection;
   - **attribution**, per session: the sorted detection keys, the `attribution` projection,
     the speech-band filter identity, `ACTIVITY_SEMANTICS_VERSION`.
   Both stage in memory and commit only after INV-01 is re-verified. Per-frame probabilities
   are a raw `uint16` per-mille file with its own sidecar carrying frame count, byte order,
   frame size, and size in bytes — so a truncated file is a miss rather than a short track.

**C. Detection.**

7. `activity/detect.py` — drives a detector over a track's 16 kHz derivative in bounded
   windows, then hysteresis, minimum speech and silence durations, merging, and padding, all
   configurable. Regions stitch across window boundaries.
8. `activity/silero.py` — `onnxruntime` on CPU, 512-sample frames with 64 samples of context
   and carried state. **One instance per track, contiguous ordered windows, and a violation
   of either raises** (ADR-0013). The runner builds one per track through a
   `DetectorFactory`. The interface identity — frame size, context size, state shape, input
   names, rate — is part of the detection key.
9. `models.py` and `models fetch`, VAD only: models directory, commit-pinned URL, sha256
   verified before the file is moved into place, provisional lock. `doctor` gains the
   model-availability check the spec asks for.

**D. Attribution.**

10. `activity/band.py` + `activity/data/fir_speechband_16k.json` +
    `scripts/design_speech_band.py` — a checked-in speech-band FIR, data rather than a
    design run at import time, held to a declared frequency response.
11. `activity/scoring.py` — the isolated four-term score, every term persisted.
12. `activity/bleed.py` — pairwise comparison over the capped overlap using
    `timeline.syncqa.measure_lag`, then ADR-0014's rule: score margin **and** correlation
    **and** the track-relative veto, with everything else retained and mixed evidence marked
    `ambiguous`.

**E. Orchestration and proofs.**

13. `timeline/runner.py` — expose the existing `_ingest` body as a reusable stage taking a
    `ReportBuilder`; `run_ingest` becomes a thin wrapper with unchanged behaviour.
14. `activity/runner.py` — `run_activity`: snapshot raw, reject outputs inside raw, rebuild
    the timeline unconditionally, detect, gate, score, verify INV-01, commit both caches,
    write `work/activity.json`, write one report covering three stages. **The failure
    envelope is designed before the DSP**, per the reviewer's ordering: every expected
    failure becomes a failed stage, a structured error, a written report, and a nonzero exit.
15. `cli.py` — `dnd-audio activity <dir>`; `models fetch` implemented.
16. `fixtures/variants.py` — `delayed_bleed` (bleed near the far edge of the lag window) and
    `mutual_bleed_overlap` (two genuine simultaneous speakers at unequal levels, each lav
    carrying the other's bleed — the reviewer's case, which the canonical fixture cannot
    express).

### Every gate criterion, and the proof for it

| Criterion | Proof |
| --- | --- |
| VAD per 16 kHz track behind the protocol, deterministic fake in the default suite | `test_activity_detect.py` — `ScriptedActivityDetector` over the fixture's declared truth; **partition invariance and cross-track isolation proved against a stateful fake ONNX session** driving the production path, not the stateless scripted detector; a structural test that the default suite never constructs a real session |
| Silero pinned by release, commit, content hash, runtime, and interface; loaded locally | `test_silero.py` — offline: the identity document's contents, refusal on a sha256 mismatch, refusal when absent, refusal on a non-contiguous window, refusal on a second track, and that `torch` is never imported. `host_smoke`: the real model on real audio. `test_models_fetch.py` — lock contents, hash mismatch fatal and the file discarded, no re-download when present (injected downloader), plus one `allow_network` real fetch excluded from the gate |
| Merge, padding, configurable thresholds, probabilities persisted | `test_activity_detect.py` — each threshold varied independently and shown to change the output; the `uint16` probability file round-tripped **and** every incomplete shape (truncated, absent, wrong frame count, wrong record version, sidecar naming another file) proved to read as a miss |
| Correlation over a bounded lag, not zero lag; peak and lag recorded | `test_activity_bleed.py` — the canonical 3 ms bleed found at ≈ +48 derivative samples; the `delayed_bleed` variant near the window edge; a contrast case where zero-lag correlation misses what the lag-tolerant one finds |
| Suppress only when convincingly stronger **and** strongly related; ambiguous kept | `test_activity_bleed.py::TestConservatism` — the four quadrants each asserted separately, **plus the veto case**: `mutual_bleed_overlap`, where dominance and correlation are both satisfied and the quieter genuine speaker still survives because its own level sits at its own track's speech reference |
| Scoring combines four terms, never one global loudness; diagnostics in the report | `test_activity_scoring.py` — each term varied alone and shown to move the score; the decision proved to consume the score (a scoring-weight change alone flips an attribution); `test_activity_run.py` asserts the diagnostics reach `ingest-report.json` |
| Solo, genuine overlap, quiet bleed, delayed correlated bleed | `test_activity_run.py` end to end on the canonical fixture with a **leaky** detector: tx-a's solo retained on tx-a and suppressed on b/d/e/f; the tx-d/tx-e overlap at 326400 retained on **both**; tx-c's post-gap speech retained |
| Graph schema checked in, versioned, frozen, model-independent | `test_activity_artifact.py` — real output validated against the **checked-in** schema; the serialized document walked for floats; a **field allowlist** over every property name, so a later text-derived field fails a test rather than changing a frozen contract silently; plus the structural import test |
| Deterministic IDs from sorted identity and time | `test_activity_artifact.py`, plus rerun byte-identity in `test_activity_run.py` including a run that processes tracks in a different order |
| INV-13 across the composed run | `test_activity_run.py::TestFailures` — missing model, detector exception, unreadable derivative, source mutated mid-run, and a report path resolving inside `raw/`, each **starting from a pre-existing report and graph on disk**: nonzero exit, `activity: failed`, every other stage accounted for, and the stale graph removed |

INV-07 is proved by extending `test_memory.py`'s ordered event log across reader → detector →
gate. INV-01 by full-tree hash equality plus a source corrupted mid-flight. INV-08 by varying
each identity component independently, by proving each cache is consulted, and by the
incomplete-entry shapes above. ADR-0016's projections are tested in **both** directions —
the hash moves for every included section and does not move for any excluded one — and
asserted exhaustive over `SessionConfig`.

### Invariants most at risk, and what stops it

- **INV-09** — this milestone freezes the boundary. Enforced by the field allowlist, which an
  import test cannot do.
- **INV-04 and INV-02** — no floats in the artifact; integer per-mille and millibels through
  the one existing quantizer; 48↔16 conversion only through M2's helpers, whose floor/ceil
  asymmetry is the documented trap.
- **INV-07** — the correlation window is configurable and *capped*, so a long candidate
  cannot pull a session-length array into memory.
- **INV-05 and INV-10** — the default suite never loads the model, and no test expects
  speech-shaped noise to trigger a particular Silero release.
- **INV-08** — two identities rather than one, and neither commits before INV-01 is verified.

### Deviations from this charter, flagged before implementation

1. **The Silero gate criterion and the spec's VAD paragraph are amended** — pinned by
   release, commit, content hash, runtime, and interface rather than by installing the
   package. ADR-0013. Second spec amendment in the project.
2. **`activity` becomes a CLI command** and a composed run writes one report. ADR-0015.
3. **`models fetch` lands its VAD half here**, four milestones early, because INV-06 makes it
   the only command permitted to reach the network. Its lock format is provisional until M6b.
4. **`onnxruntime` becomes a runtime dependency** (with `flatbuffers` and `protobuf`), the
   same shape as M2 adding SciPy.
5. **M2's derivative cache identity changes** to a stage-scoped configuration projection.
   ADR-0016. Closed-milestone code, as M2's own `raw_guard` extraction was.

---

## Closeout

_Filled in during the close phase. Leave the headings; they are the checklist._

### What works end to end

### Tests and commands run, with results

### Decisions made (→ ADRs)

### Assumptions made and open questions raised

### Notes for future implementors

### Deviations from this charter, and why

### Downstream charters updated

### Next smallest step
