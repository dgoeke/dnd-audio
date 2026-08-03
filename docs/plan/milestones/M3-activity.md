# M3 — Conservative speech activity and bleed rejection

**Status:** closed
**Depends on:** M2
**Spec sections:** Milestone 3; Milestone 5 (activity graph definition); Tests and
acceptance criteria 5, 15

## Goal

Per-track VAD, a conservative pre-ASR bleed gate using lag-tolerant normalized
cross-correlation, and a versioned, model-independent activity/attribution graph
that both the transcript branch and the automixer consume.

## Completion gate

- [x] VAD runs per 16 kHz track behind an `ActivityDetector` protocol, with a
      deterministic fake / ground-truth-mask implementation used by the default
      suite (INV-10). Synthetic noise is never expected to trigger a specific
      learned Silero release.
- [x] Silero model artifact pinned by upstream release and commit **and by content
      hash**, with the runtime and the calling interface pinned too, and loaded
      locally — no unpinned runtime `torch.hub` fetch. Identity appears in cache keys
      and the report (INV-08). CPU or ONNX is the baseline so it does not contend with
      ASR for unified memory. _(Amended during the start phase, with the spec, after
      independent review: the original wording said "Silero package and model
      artifact/revision". See ADR-0013.)_
- [x] Nearby speech regions merged and boundaries padded; all thresholds
      configurable; VAD probabilities and decisions persisted for debugging.
- [x] Cross-channel similarity uses normalized speech-band cross-correlation over a
      configurable bounded lag (default ±30 ms), **not** zero-lag correlation. Both
      the peak correlation and its selected lag are recorded.
- [x] Bleed suppressed only when another track is convincingly stronger *and* the
      signals are strongly related. Ambiguous candidates are kept by default.
- [x] Source scoring combines track-relative speech level, VAD confidence,
      cross-track dominance, and correlation evidence — never a single global
      loudness comparison. The scoring function is isolated and its diagnostics
      appear in `ingest-report.json`.
- [x] Tests: solo attribution, genuine two-person overlap survives, quiet bleed is
      suppressed to the right track, and correlated bleed delayed within the lag
      window is still detected with its peak lag reported.
- [x] **The activity graph schema is checked in, versioned, and frozen** (INV-09).
      It is model-independent: nothing text-derived may enter it.
- [x] Every retained candidate has a deterministic ID derived from sorted source
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

---

## Closeout

### What works end to end

`uv run dnd-audio activity /path/to/session` — inspection, the timeline, then who was
speaking. It snapshots the raw roots once for the whole composed run, rebuilds the timeline
unconditionally, runs a VAD per track over the cached 16 kHz derivative, measures every
overlapping pair with a lag-tolerant normalized speech-band cross-correlation, scores each
candidate on four terms, applies the conservative bleed gate, verifies INV-01, commits four
caches at one moment, and writes `work/activity.json` plus one `output/ingest-report.json`
covering three stages.

On the canonical fixture with the pinned Silero model: 6/6 tracks, 0 candidates — which is
the *correct* answer and worth stating plainly. The fixture's speech is synthetic
speech-shaped noise, and INV-10 forbids expecting a particular learned release to fire on
audio no human made. Every attribution proof therefore runs against the deterministic
`ScriptedActivityDetector` over the fixture's declared truth. The real model is exercised
once, under `host_smoke`, on claims that are true of *any* release: probabilities are
probabilities, the frame count is the one the track's length predicts, a loud burst scores
above digital silence, and the recurrence survives a change of window partitioning.

`uv run dnd-audio models fetch` downloads the VAD model, pinned by upstream release
(`v6.2.1`), commit (`7e30209a`), and sha256, verified before the file is moved into place,
and records a provisional lock. It is the only command in the project permitted to touch the
network (INV-06). `doctor` now reports model availability alongside tool versions and disk.

`ingest`, `inspect`, `make_fixture.py`, and the seven M2 fixture variants are unchanged, plus
two new variants: `delayed_bleed` (25 ms — far enough out that a zero-lag correlator finds
nothing) and `mutual_bleed_overlap` (two genuine simultaneous speakers at unequal levels,
each lav carrying the other's voice).

### Tests and commands run, with results

```
./scripts/gate.sh
  pass  system dependencies      pass  lock is current
  pass  ruff check               pass  placeholder scan
  pass  ruff format              pass  plan consistency
  pass  type check               pass  pytest (offline, cpu) — 1503 passed, 3 deselected
GATE PASSED
```

The 3 deselected are the only marked tests in the suite: `test_the_pinned_model_runs_on_real_inference`
(`host_smoke`), `test_the_pinned_url_still_serves_the_pinned_bytes` (`allow_network`), and
`test_marker_opts_out`. There are no `skip` or `xfail` marks anywhere.

Per-file, run during the verify phase:

```
test_activity_detect     74 passed      test_activity_artifact   33 passed
test_silero              51 passed      test_activity_cache      69 passed
test_models              32 passed      test_speech_band         40 passed
test_activity_bleed      25 passed      test_memory               7 passed
test_activity_scoring   140 passed      test_activity_run        28 passed
```

**Mutation probes**, because a passing test is not evidence it can fail. Each was applied to
the implementation, the suite was run, and the source restored:

| Mutation | Result |
| --- | --- |
| `vetoed = False` — ADR-0014's veto removed | 6 failures, incl. end-to-end `two_real_speakers_at_unequal_levels_both_survive` |
| Scoring collapsed to dominance alone (the global-loudness rule the spec forbids) | 12 failures |
| `ActivityGraph`'s canonical candidate sort removed | `test_candidates_sort_by_start_then_track` fails |
| Canonical sort removed from the *builder* instead | **nothing failed** — correctly, because the artifact model sorts in a validator. Determinism is enforced at the boundary, not by the caller |

Live, on the canonical fixture: `work/activity.json` byte-identical across two runs
(`e9d80d10…`), and identical again after the 48↔16 kHz refactor. `doctor` reports the model
at its pinned hash.

The two invariant violations found in verify were each **reproduced before being fixed**. The
INV-08 one, from a standalone probe: a run that correctly failed on a source mutated mid-flight
left twelve `work/cache/inspect/*.json` sidecars; after the fix, zero.

### Decisions made (→ ADRs)

Five recorded before any code was written, and one amended after it:

- **[ADR-0012](../decisions/0012-the-activity-graph-contract.md)** — the frozen graph: units,
  grids, orderings, per-pair evidence, the suppressing *candidate* named rather than the
  track, INV-09 enforced by a field allowlist over the checked-in schema.
- **[ADR-0013](../decisions/0013-silero-through-onnx-runtime.md)** — Silero through ONNX
  Runtime, pinned by commit and content hash, no Torch. **Amends the spec twice** (second
  amendment in the project): once to pin the artifact/runtime/interface rather than the
  package, and once because the original justified CPU inference as avoiding contention "for
  unified GPU memory" — on a unified-memory host a CPU tensor and a GPU tensor draw on the
  same pool, so that reasoning was simply wrong. The preference is right for other reasons,
  which the spec now gives.
- **[ADR-0014](../decisions/0014-the-conservative-bleed-gate.md)** — suppression needs a score
  margin **and** correlation **and** a track-relative level below the veto. **Amended during
  the verify phase** to record what was actually built: the speech reference is the 75th
  percentile of *all* of a track's candidates, not the median of its high-confidence ones,
  and `ambiguous` marks the veto case rather than "some but not all conditions". Both
  implementation choices are better than what the ADR first specified; leaving the ADR
  disagreeing with them silently was not an option.
- **[ADR-0015](../decisions/0015-activity-as-a-stage-command.md)** — `activity` is a stage
  command; a composed run writes one report.
- **[ADR-0016](../decisions/0016-stage-scoped-cache-configuration.md)** — cache identity
  carries a stage-scoped projection of the configuration, not the whole of it, so tuning a
  bleed threshold does not rebuild gigabytes of PCM.

### Assumptions made and open questions raised

- **OQ-010 answered.** Silero is pinned by upstream release, commit, and content hash, and
  loaded locally through ONNX Runtime on CPU with no Torch anywhere in the environment. The
  artifact at tag `v6.2.1` / commit `7e30209a` is byte-identical to the copy inside the
  published `silero-vad` 6.2.1 wheel — verified against both sources, which is what makes
  "we did not install the package" a packaging decision rather than a change of artifact.
- **OQ-017 raised**, before the first default threshold rather than after. Every default in
  `activity.vad`, `activity.bleed`, and `activity.scoring` cites it, and so does the speech
  reference estimator. Its evidence is H2 or the first real session — **not H1**, whose
  two-minute metadata fixture cannot tune a bleed threshold. The pipeline already records
  every number needed to answer it (per-candidate levels, per-pair peak correlation and its
  lag, each track's reference), so answering it is reading one real session's graph rather
  than running an experiment.
- The scoring weights, the VAD thresholds, the veto, and the reference percentile are all
  numbers chosen against synthetic audio whose bleed is a delayed attenuated copy of its
  source. That is the *easy* case: real bleed crosses a room, reflects, and arrives filtered,
  so its correlation against the source track is lower. Nothing here was tuned to make a
  fixture pass.

### Notes for future implementors

**The gate's central risk is real and the veto is what answers it.** A rule of "the loudest
track wins" passes every casual test and deletes quiet speakers during genuine overlap. The
`mutual_bleed_overlap` fixture is that case: dominance and correlation *both* say bleed, and
both are wrong because the quieter person is actually talking. What saves them is that their
own lav hears them at the level that lav hears them at normally. If you change one thing in
this milestone, do not change that — and note the contrast test that makes it meaningful:
the same audio with `min_reference_candidates` raised beyond reach has no reference, so the
veto cannot fire, and the identical overlap is suppressed. Retention on its own would have
proven nothing.

**A cache committed inside a helper defeats the verification its caller does.** `_inspect`
carried a docstring promising it returned the cache uncommitted, "published by the caller once
INV-01 has been re-verified", and called `cache.commit()` three lines below it. Both callers
commit again afterwards, so every reading of the code from the outside looked correct. This
shipped in M2 and survived M3's own charter instruction to get the ordering right. The
regression test could not have caught it because it globbed only the cache M3 had added.
**Assert over every sidecar under `work/cache`, not the one you just wrote.** M5 will add
another.

**Test the boundary, not the builder.** Removing the canonical sort from `_candidates` broke
no test — correctly, because `ActivityGraph` sorts in a validator. That is the right place
for it: a determinism rule enforced in the artifact holds no matter which caller assembles
one, and M4 and M5 both will. Do not "fix" a redundant sort by moving the guarantee up into
the caller.

**Silence has three causes and a VAD sees zeros in all of them** — before a track started,
inside a real gap, after it stopped. That is deliberate (M2), and it means you never
special-case a track that ended early; every track answers to the session's aligned
`duration_samples`.

**The 48↔16 kHz mapping floors its start and ceils its end.** Rounding both the same way
shrinks a speech region by up to two samples, which is how a word loses its first phoneme.
Use `timeline.resample.to_source_sample` and `to_derivative_interval` — this milestone wrote
the conversion out by hand at first and it was caught in verify. The values agreed, which is
exactly why a second copy is dangerous: nothing would have failed when one of them changed.

**Correlation contributes to the score as *independence*, not similarity.** A candidate
strongly correlated with another track is more likely a copy of it than a voice of its own,
so high correlation *lowers* the score. The other sign ranks the best-recorded copy of
someone else's voice above the original, and it is an easy sign to get backwards.

**Do not expect synthetic noise to trigger the real model.** The canonical fixture through
real Silero yields zero candidates and that is a pass, not a bug. Every attribution test uses
the scripted detector. When you need a real-model assertion, assert something true of any
release.

**`onnxruntime` is imported lazily, and a test enforces it.** Importing `activity.silero`
must stay free for the default suite (INV-05); the runtime is imported inside the functions
that need it, and a subprocess test proves `torch` is never imported at all.

**The detector is stateful and owns one track.** Silero is recurrent: one instance per track,
contiguous windows in order, and a violation of either *raises* rather than silently
returning a plausible wrong answer. The partition-invariance proofs run against a stateful
fake ONNX *session* driving the production path — a stateless fake detector would have
replaced the code under test.

**`speech_references` is the softest number in the milestone.** p75-of-all-candidates sits
systematically higher than the median-of-high-confidence the ADR first specified, and a
reference set too high weakens the veto for that wearer's quieter speech. Including bleed
candidates pushes it the other way. Which effect dominates is a property of a real room
(OQ-017). Both directions are written down in ADR-0014 so the next person tuning this knows
which error they are trading against.

### Deviations from this charter, and why

All five were flagged before implementation, and a sixth emerged in verify:

1. **The Silero gate criterion and the spec's VAD paragraph are amended** — pinned by
   release, commit, content hash, runtime, and interface rather than by installing the
   package (ADR-0013). The `silero-vad` distribution hard-depends on `torch` and
   `torchaudio`, which is unacceptable in the environment the default suite runs in and
   would pre-empt M6a's AMD wheel index.
2. **`activity` became a CLI command** and a composed run writes one report (ADR-0015).
3. **`models fetch` landed its VAD half here**, four milestones early, because INV-06 makes
   it the only command permitted to reach the network. Its lock format is provisional
   until M6b.
4. **`onnxruntime` became a runtime dependency** (with `flatbuffers` and `protobuf`), the
   same shape as M2 adding SciPy.
5. **M2's derivative cache identity changed** to a stage-scoped configuration projection
   (ADR-0016) — closed-milestone code, as M2's own `raw_guard` extraction was.
6. **An INV-08 violation inherited from M2 was fixed here** (`_inspect` publishing its own
   cache). Found by this milestone's verify phase, in code M3 did not write but did compose
   into a longer run, which is what made the consequence reachable.

### Downstream charters updated

- **M4** — the activity graph's consumer contract and the `ambiguous` flag's actual meaning;
  the INV-09 free-text caveat.
- **M5** — the same INV-09 caveat, stated as a prohibition: the mix may not read
  `ActivityDecision.detail` or `ActivityNote.message`, because the field allowlist freezes
  names and cannot constrain prose. Plus the deferred `compare_pairs` complexity note, since
  M5 is the next milestone to walk the candidate set.

### Next smallest step

Begin M4 — the transcript branch on fake ASR. Read M4's "What M3 already provides" section
first: the graph's consumer access pattern is already exercised by
`test_activity_artifact.py::TestTheConsumerReads`, which is the closest thing to a worked
example of how to index into this document, and `retained` + `ambiguous` together are what
M4's request builder must respect.
