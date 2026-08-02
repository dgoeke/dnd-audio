# M2 — Reconstruct six synchronized virtual tracks

**Status:** not started
**Depends on:** M1
**Spec sections:** Milestone 2; Tests and acceptance criteria 2, 3, 4, 13

## Goal

`dnd-audio ingest` turns the manifest into six synchronized virtual tracks: chunks
ordered by embedded timecode, real gaps preserved as silence, a lossless streamed
48 kHz working path, and cached 16 kHz mono derivatives — with an exact recorded
mapping between source samples, working samples, and session time.

## Completion gate

- [ ] Session time zero from the earliest valid source start unless
      `timecode.origin_timecode` supplies an explicit origin on `origin_date`.
      `origin_date` is never inferred from a date-shaped `session_id`.
- [ ] Chunks sorted by parsed start time, not filename order; each chunk's expected
      end validated against the next chunk's start.
- [ ] Real gaps preserved as silence. A transmitter switched off and back on does
      not slide later audio earlier. Verified against synthetic ground truth.
- [ ] Overlaps detected. Only quantization-scale overlaps resolved automatically;
      anything larger warns and requires explicit policy rather than discarding audio.
- [ ] Midnight rollover: `infer_forward` infers a single forward rollover only when
      chunk sequence and session span make it unambiguous, and records the
      decision. Ambiguity demands a dated origin or an override, never an ad hoc
      interpretation.
- [ ] Exact sample-position tests for non-drop, fractional (24000/1001,
      30000/1001), drop-frame, rollover, and explicit-override cases (INV-04).
- [ ] Lossless 48 kHz float working path is streamed/windowed over a segment map,
      never six session-length arrays in RAM; contiguous intermediates use RF64;
      work-space and disk preflighted (INV-07).
- [ ] 16 kHz mono derivatives cached, with resampler delay and end rounding
      accounted for in the 48↔16 kHz mapping.
- [ ] Aligned output duration is set by the latest track end and matches within one
      48 kHz sample.
- [ ] A selected 44.1 kHz source, or chunks within one track disagreeing on sample
      rate, fails before timeline construction with a clear diagnostic.
- [ ] Optional clap cross-correlation runs as QA only: it reports disagreement with
      timecode and never overrides valid timecode. Lag is measured near both ends
      and a materially changed lag warns (drift evidence, not correction).
- [ ] An interface hook exists for a future affine time warp, unused in the MVP.

## Explicitly not in this milestone

- Any automatic drift correction. Warn only.
- VAD, activity, or anything that interprets the audio's content.
- Phase-coherent multichannel processing of any kind.

## What M1 already provides (read before starting)

- **Per-file timing evidence already exists, in typed form** (ADR-0006). The manifest's
  `start_time.evidence` is a discriminated union, and the three variants do not share a
  coordinate system: `bwf_sample_reference` is unsigned samples since midnight **at the
  file's own rate**; `timecode` is an exact integer frame index plus a rational rate;
  `session_offset_samples` is **signed, at 48 kHz, relative to session zero**. Reconciling
  them is this milestone's job. Do not add a fourth "just give me the number" accessor —
  that is the collapse ADR-0006 exists to prevent.
- **This milestone owes acceptance criterion 2 a documented quantization rule.** A frame
  at 30000/1001 fps is `8008/5` samples at 48 kHz, so "the expected integer sample
  position" is a property of a rounding rule, not of the evidence. Define it, write it
  down, and test it at 24000/1001 and 30000/1001 where it actually bites. M1 deliberately
  stopped short of inventing one.
- **A non-48 kHz source is a warning in M1 and must be fatal here**, before timeline
  construction. The manifest already carries `unexpected_sample_rate` per source and the
  container facts that explain it; the diagnostic exists, the refusal does not.
- **`container.sample_count` is exact and needs no decode** — it comes from the RIFF
  `data` size over the block alignment, cross-checked against `duration_ts`, with their
  agreement recorded as `sample_count_agrees` (OQ-011).
- **Every candidate is in the manifest, not only the selected ones**, including files in
  unconfigured directories under `unassigned`. Filter on `role == "selected"` when
  building the timeline; nothing else belongs on it.
- **`_snapshot`/`_verify_unchanged` in `inspection/runner.py` is the INV-01 machinery.**
  If this milestone writes anywhere new, extend the "output inside raw" check — and note
  it compares *resolved* paths, because a lexical comparison was defeated by one symlink.

## Known risks and open questions

- Depends on **OQ-004, OQ-006, OQ-011**, and **partially answers OQ-013**: the work-space
  preflight this milestone builds replaces `doctor`'s estimated 40 GiB warning threshold
  with a number derived from the session's actual length. It does not *settle* the
  question, which asks for measured full-pipeline disk use — that needs H2 or a real
  session, and two terms of the original estimate now belong to M5 or do not exist
  (ADR-0011). This charter previously claimed to settle it; corrected during the start
  phase after independent review.
- Raises **OQ-014** (when an inferred session span is implausible) and **OQ-015** (where
  the receivers' timecode zero sits relative to real midnight, which only matters at a
  fractional non-drop rate — ADR-0009).
- **`dnd_audio.determinism.write_atomic` is for artifacts, not audio.** It holds the
  whole payload in memory, which is right for JSON and a direct INV-07 violation for a
  session-length waveform. The streamed working-audio path is this milestone's to build.
- Exact-time helpers already exist: rates are `Fraction`, and `public_seconds()` is the
  only float-producing conversion, built on an integer-millisecond quantizer with a
  documented tie rule. Do not add a second float path.
- INV-04 and INV-07 are both at maximum risk here. Resist the convenience of a
  float seconds field and of one big NumPy array; both work fine on a two-minute
  fixture and fail on a four-hour session.
- The 48→16 kHz mapping is the most likely source of a subtle, late-discovered
  offset. Test it against known impulse positions, not just durations.

## Working plan

_Scratch, written during the start phase and revised after independent review
(`../reviews/M2-plan-20260802-1241.md`). Replaced by the Closeout at the end._

### Decisions taken before any code

Four came from the owner during the start phase and are treated as amendments to this
charter:

1. **The segment map is the authoritative 48 kHz representation.** `ingest` never
   materializes contiguous 48 kHz files by default. `--materialize-48k` writes float32
   RF64 files that are *disposable content-addressed cache artifacts*, not pipeline truth.
   Every duration and sample-position test reads through the virtual segment reader.
2. **The report is rewritten atomically, preserving a completed `inspect`.** Inspect
   provenance, warnings, and decisions carry forward only on a full match; telemetry is
   rebuilt for the current run. `inspect` is marked **complete with `origin: reused`**,
   never skipped. *(Revised below: the manifest is now rebuilt every run, so "reused"
   describes a cache-served inspection rather than a skipped one.)*
3. **One canonical fixed 3:1 linear-phase FIR decimator**, coefficients and design
   metadata checked in, driven through `scipy.signal.upfirdn(up=1, down=3)` behind a thin
   streaming wrapper that carries filter state and decimation phase across windows. Never
   "take every third sample". FFmpeg/soxr is QA comparison only — its delay and boundary
   behaviour are less explicit and more version-dependent.
4. **The checked-in FIR is held to a declared frequency response**, so "fixed decimator"
   cannot decay into an arbitrary array that happens to produce the right sample count:
   passband edge 7000 Hz with ripple ≤ 0.1 dB, stopband beginning no later than 8000 Hz
   (16 kHz Nyquist) with ≥ 80 dB attenuation, exact coefficient symmetry, unity DC gain.

The design that meets it, chosen by sweeping length, Kaiser β, and cutoff against the
contract and taking the most margin on both sides: **length 259, β 9.0, cutoff 7450 Hz**
→ 0.02 dB ripple at 7 kHz and 90.4 dB attenuation at 8 kHz, group delay 129 samples at
48 kHz, which divides by 3 and is therefore exactly 43 samples at 16 kHz.

Written before any code, as the working agreement requires:

- **[ADR-0008](../decisions/0008-rasterizing-time-onto-the-sample-grid.md)** — evidence
  becomes an exact `Fraction` of seconds; the session-relative difference is taken **once**
  and only that result is quantized, half away from zero, by the one quantizer
  `determinism.to_samples`. Also defines the overlap tolerance: one sample for
  sample-exact evidence, one whole frame at the configured rate when either start came
  from a timecode.
- **[ADR-0009](../decisions/0009-session-zero-and-the-24-hour-wrap.md)** — session zero,
  and the 24-hour wrap **unwrapped in each evidence domain's own units**. Session zero is
  `origin_date` + `origin_timecode`, else the earliest valid source start with the whole
  timeline shifted so it lands at zero — which is what keeps a *signed*
  `start_offset_samples` meaningful. With an explicit origin, audio before zero is fatal.
  Rollover reads time evidence only, never a filename counter.
- **[ADR-0010](../decisions/0010-chunk-overlap-policy.md)** — `timecode.chunk_overlap_policy`:
  `reject` (default) is fatal on a material overlap; `nudge_later` moves the later chunk.
  Neither discards audio. Sub-tolerance overlaps are nudged under either policy, with both
  the rasterized and the placed start retained.
- **[ADR-0011](../decisions/0011-the-working-audio-path.md)** — the segment map is the
  working path; sources are read directly by seeking into the RIFF `data` chunk, because
  windowed random access is what a mix pass needs and a pipe is not that. Mono float32
  RIFF/RF64 only. The decimator runs across the whole virtual track and never resets at a
  chunk or gap boundary.
- **OQ-014** raised — when an inferred session span is implausible (12 h). **OQ-015**
  raised — where the receivers' timecode zero sits relative to real midnight, which only
  bites at a fractional non-drop rate.

### Amendments after independent review

Nine of ten findings accepted; the distillation records each with its reasoning. The four
that change the architecture:

- **`ingest` always runs discovery and hashing.** Reusing a manifest on a `config_hash`
  match let a replaced or deleted WAV pass unnoticed, because the INV-01 snapshot only
  covers mutations *during* a run. The spec defines `ingest` as "run `inspect` as needed,
  then construct…", so inspection runs every time, served by M1's content cache. A prior
  report's inspect provenance may be reused only after a current manifest exists.
- **Rollover unwraps per domain.** "Add a day" is wrong at 29.97 non-drop by 86.4 seconds,
  because 2 592 000 frames at 30000/1001 fps is 86 486.4 s. ADR-0009 carries the table;
  rollover is tested crossed with 23.98F, 29.97F, and 29.97DF.
- **The memory proof covers the whole path.** Bounding one reader says nothing about a
  builder that collects every window. Source reads and sink writes now go to one ordered
  event log and the test asserts a write happens before the final read — a property
  nothing accumulating a session-length array can satisfy.
- **Integer PCM is out of scope, and s32→float32 is not lossless.** float32 has 24
  mantissa bits; `2147483647` becomes `2147483648.0`. Mono float32 only, failing clearly
  otherwise.

Rejected: that rollover depends on DJI's filename counter and therefore on OQ-003. It
depends on time evidence alone, and must — INV-12 forbids the coupling, and
`SourceContext` is deliberately shaped so a strategy cannot reach a filename.

### The serialized contract, field by field

M3 and M5 both consume this, so it is settled here rather than discovered later.

- **Intervals are half-open**, `[start, start + n_samples)`, everywhere.
- **A segment** carries `session_start_sample`, `n_samples`, `kind` (`audio` | `silence`),
  and for audio: `source_relative_path`, `source_sha256`, `source_start_sample` (an offset
  into that file's PCM frames), `evidence_start_sample` (where rasterization put it) and
  `shift_samples` (what the layout added). Both starts are kept so a consumer can always
  see what the evidence said before the layout adjusted it.
- **Gaps are explicit `silence` segments**, so a track's map is total over its own extent.
- **A track's map covers `[0, track_end_sample)`**; the session-wide
  `duration_samples` is the maximum track end. Tracks are not padded to it in the map —
  that would be inventing audio — but every reader returns silence past a track's end up
  to the session duration, so the aligned duration is one number and every track answers
  to it.
- **The 16 kHz derivative**: output length is `ceil(n48 / 3)` with the tail zero-padded;
  the mapping is `sample16 = sample48 // 3` after delay compensation. `DerivativeRecord`
  carries the filter identity, `decimation`, `group_delay_input_samples`,
  `group_delay_output_samples`, `input_samples`, `output_samples`, the length rule, and
  the cache key.
- **Resampling runs across the whole virtual track**, never reset at a chunk or gap
  boundary.
- **No floats anywhere in `timeline.json`.** Samples are integers; rates are
  `{numerator, denominator}` as in `manifest.RationalRate`. Human-facing seconds live in
  the report. A test walks the serialized document and fails on any float.

### Files, in build order

The review's ordering is adopted: records, then the contract, then unconditional
inspection, cache identity, origin and layout, the reader and decimator, the memory and
equivalence proofs, CLI and report orchestration, and optional 48 kHz materialization last.

**Shared guard (extraction, not new behaviour).** `raw_guard.py` takes `raw_roots`,
`snapshot`, `verify_unchanged`, and `reject_outputs_inside_raw` out of
`inspection/runner.py`, with the protected-output set becoming a parameter so each stage
declares its own. This charter tells M2 to extend the "output inside raw" check; leaving it
inside `inspection` would make ingest's INV-01 proof depend on another stage's private
helpers. `tests/test_raw_guard.py` keeps M1's falsification tests and adds ingest's set.

**Exact arithmetic.** `determinism.to_samples(seconds, rate)` lands beside
`to_milliseconds`, sharing the one stated tie rule.

**Configuration** (additive optional fields; schemas regenerated):
`TimecodeConfig.chunk_overlap_policy`, and a new `SyncQaConfig` as `SessionConfig.sync_qa`
(`enabled=False`, `window_s` with an upper bound, `max_lag_ms`, `drift_warn_ms`).

**Artifacts.** `artifacts/timeline.py` implements the contract above.
`artifacts/report.py` gains `StageReport.origin: "executed" | "reused"`, rejected as
`reused` on any status but `complete`. `schemas/timeline.schema.json` joins the drift test.

**`dnd_audio/timeline/`**, a new package:

| Module | Responsibility |
| --- | --- |
| `rasterize.py` | evidence → exact seconds; per-domain 24-hour cycles; session-relative position; overlap tolerance (ADR-0008, ADR-0009) |
| `origin.py` | session zero, the shift rule, rollover inference (ADR-0009) |
| `layout.py` | ordering, expected-end validation, gaps, overlaps, duration, the sample-rate refusals |
| `warp.py` | `TimeWarp` protocol and `IdentityWarp`; the affine seam, applied for real by the builder |
| `pcm.py` | seekable mono-float32 reader over the RIFF `data` chunk (ADR-0011) |
| `reader.py` | `TrackReader` — bounded windows over the segment map, silence for gaps and past a track's end |
| `wavwrite.py` | streamed float32 writer, temp-then-rename, RF64 when the final size demands it |
| `fir.py` + `data/fir_48k_16k.json` | checked-in coefficients and their declared design spec |
| `resample.py` | streaming 3:1 decimator, state and phase carried across windows |
| `derivatives.py` | content-addressed derivative cache, INV-08 identity, atomic publication |
| `preflight.py` | work-space estimate from the actual session length and the artifacts requested |
| `syncqa.py` | clap cross-correlation at both ends; disagreement and drift warnings, never a correction |
| `runner.py` | orchestration, report merge, `work/timeline.json` |

**CLI and scripts.** `ingest` with `--materialize-48k` and `--no-cache`;
`scripts/design_fir.py` regenerates the coefficient file from its declared spec.

**Fixture generator.** `_samples_since_midnight` / `_timecode_text` generalize from
hardcoded 30 fps to any configured rate including drop-frame, keeping the existing refusal
to write a fixture whose timecode does not land on a whole sample. New variants: no
explicit origin, midnight rollover at each of three rates, a 44.1 kHz selected source,
intra-track rate disagreement, a material chunk overlap, a negative recovery offset, and a
drift variant that holds nominal timecode fixed and **moves the end transient within the
audio samples** — moving the metadata instead would let the drift test pass without
cross-correlation ever detecting an acoustic lag change.

### Every gate criterion, and what proves it

| Criterion | Proof |
| --- | --- |
| Session zero; `origin_date` never inferred from `session_id` | `test_origin.py::TestSessionZero` — explicit origin, earliest-source, a date-shaped `session_id` with no `origin_date`, and negative offsets in both branches |
| Chunks sorted by parsed start; expected end validated against next start | `test_layout.py::TestOrdering`, on a fixture whose filename order contradicts its timecode order |
| Real gaps preserved; later audio does not slide earlier | `test_layout.py::TestGaps` vs `FixtureTruth.gaps()`; `test_reader.py::test_a_gap_reads_as_silence`; end-to-end, tx-c's post-gap speech at 408000 |
| Overlaps detected; only quantization-scale resolved | `test_layout.py::TestOverlaps` — sub-tolerance nudge recorded with both starts retained; material overlap fatal under `reject`; `nudge_later` proven to preserve the total sample count |
| Rollover inferred only when unambiguous, and recorded | `test_origin.py::TestRollover` — **crossed with 23.98F, 29.97F, 29.97DF**, since each wraps by a different number of real seconds; ambiguous-is-fatal; `reject`-is-fatal; end-to-end rollover fixture |
| Exact sample positions: non-drop, 24000/1001, 30000/1001, drop-frame, rollover, override | `test_rasterize.py`, expectations stated as `Fraction`s and never built by calling the code under test; plus an end-to-end 29.97DF fixture |
| Streamed/windowed; RF64; preflight | `test_memory.py::test_a_write_happens_before_the_last_read` over the **whole** reader → resampler → writer path on a timeline far longer than the window; `test_wavwrite.py::TestRf64` with the header round-tripped through `inspection.riff`; `test_preflight.py` |
| 16 kHz derivative cached; resampler delay and end rounding accounted for | `test_fir.py::TestFrequencyResponse` (passband, stopband, ripple, attenuation, symmetry, DC gain) and `TestDesignIsReproducible`; `test_resample.py` — impulse position, **every input length residue mod 3**, first and last samples, across gaps and chunk boundaries, varied window partitioning, streamed-vs-one-shot byte equality; `test_derivatives.py` varying each identity component, proving the cache is consulted, and rejecting a truncated file and an orphaned sidecar |
| Aligned duration set by the latest track end, within one sample | `test_ingest_run.py::test_aligned_duration_matches_the_latest_source_end`, read through the virtual segment reader (504000 samples on the canonical fixture) |
| 44.1 kHz or intra-track disagreement fatal *before* construction | `test_ingest_run.py`, both through the CLI, **starting from a stale `timeline.json` on disk**: exit 1, structured error, the stale artifact removed, and the layout builder, PCM reader, and derivative writer spied on and proven un-entered |
| Clap correlation is QA only; lag at both ends; changed lag warns | `test_syncqa.py` — agreement; a constant offset reported while the timeline stays byte-identical; a drift fixture whose end transient moved in the audio; a no-clap case that reports low confidence rather than a lag |
| An affine time-warp hook exists, unused | `test_warp.py::test_a_non_identity_warp_moves_the_timeline`, with a test-local affine implementation — a seam that cannot fire is decoration |
| Report behaviour (added after review) | `test_ingest_run.py::TestTheCanonicalSession` — `origin: reused` on a warm run, the manifest rewritten rather than trusted, both deliverables hashed — and `::TestRefusalsHappenBeforeConstruction`, where every failure case starts from a stale `timeline.json` **already on disk** and asserts it is gone. *(This row named a `test_ingest_report.py` that was never written; corrected during verify.)* |

### Invariants at risk, and what stops each

- **INV-04** — a test walks the serialized `timeline.json` and fails on any float, as M1
  does for the manifest. `Fraction` throughout; `to_samples` is the only quantizer; no
  running total is ever accumulated.
- **INV-07** — the whole path is proven streaming by the ordered read/write event log, not
  just the reader. `sync_qa.window_s` is bounded by validation. `write_atomic` is never
  reachable from the audio path.
- **INV-02** — `timeline.json` and the 16 kHz derivative byte-identical on rerun with a
  different clock and a warm cache.
- **INV-01** — snapshot and verify around the whole ingest run, with a test that corrupts a
  source mid-run; the extended output check covers `work/timeline.json`, the derivative
  cache, and any materialized 48 kHz file.
- **INV-08** — the derivative identity carries the track's segment map, source hashes,
  config hash, `INSPECTION_SEMANTICS_VERSION`, `TIMELINE_SEMANTICS_VERSION`, the FIR
  identity, SciPy and NumPy versions, and the target rate — each varied independently.
  Publication is temp-then-rename with the sidecar committed last; a truncated file and an
  orphaned sidecar both read as misses.
- **INV-12** — nothing here reads a filename or an mtime; a selected source without
  evidence stays fatal; rollover uses time evidence only.
- **INV-13** — every fatal path writes the report with a structured error and exits
  nonzero. The single exception is the one INV-01 outranks: when the report's own location
  resolves inside a source directory, **nothing is written**.

### Concerns raised at plan time

1. **`scipy` is now a runtime dependency** — added and verified: 1.18.0 installs from PyPI
   into the flake's `.venv` and `upfirdn` runs. No flake change was needed.
2. **The canonical fixture always sets `origin_timecode`**, so the "earliest valid source
   start" branch has no existing fixture. New fixture work this charter implies but does
   not name.
3. **This charter says "16 kHz derivatives cached" without saying where.** Both derivative
   kinds become content-addressed cache artifacts under `work/cache/`, with
   `timeline.json` naming the path.
4. **Moving the INV-01 helpers touches a closed milestone's code.** No behaviour changes;
   M1's falsification tests move with them.
