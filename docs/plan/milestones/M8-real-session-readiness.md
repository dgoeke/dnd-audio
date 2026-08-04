# M8 — Real-session readiness

**Status:** not started
**Depends on:** M6b (closed). Ordered **before** M7 despite the number — M7 waits on a
processed real session, and this milestone is what makes that session worth recording.
**Spec sections:** Bleed and attribution; Timecode strategy; Tests and acceptance criteria

## Goal

Fix the defects that would corrupt or lose the first real session, and add the diagnostics
that make it produce evidence instead of plausible-looking output. Every item here is a
**structural** defect — wrong regardless of what a real table sounds like — so none of it
depends on tuning data this project does not yet have.

At the end, a four-hour six-transmitter session can be recorded and processed without the
known failure modes, and the thresholds that remain unsettled are visible in the artifacts
rather than inferred by hand.

## Why now

The MVP code path is complete and the timing model is settled: a jam propagates into
`bext.time_reference` and places receivers to one frame (**OQ-023**), and sample-clock drift
is ≈1 ppm (**OQ-006**). What has never been true is that the pipeline is safe to point at a
session that cost six people an evening. The 2026-08-03 `samples` jam-capture run — four tracks, two
receivers, real GPU, exit 0 — surfaced the specific reasons, and they are not thresholds.

Full evidence: `docs/fixtures/2026-08-03-jam-verification.md` and the run's findings.

## The defects, with their sites and the evidence

### 1. The bleed veto's reference is computed from bleed *(root cause — do this first)*

`activity/bleed.py::speech_references` sets each wearer's speech reference to
`REFERENCE_PERCENTILE` (75) of **all** that track's candidate levels — own speech and bleed
together. `veto_db` (12.0) then keeps any candidate within 12 dB of it. When most of a track's
candidates are bleed, the percentile lands on bleed, and the veto protects bleed from
suppression.

Measured on the jam capture (`samples/`), where direct-to-bleed is 17.4 dB:

| track | candidates | reference the pipeline computed | what it actually is |
| --- | --- | --- | --- |
| tx-b | 3 | −40.77 dBFS | its own speech (−38.8) |
| tx-d | 4 | −57.80 dBFS | **bleed** (−57.7) |

**One extra bleed candidate moved the reference 17 dB.** `np.percentile(..., 75,
method="nearest")` lands on the largest of three values and the second-largest of four.

This gets worse with roster size, not better: with six people each speaking ~1/6 of the time,
roughly 83% of any track's candidates are bleed, so the 75th percentile sits in bleed territory
**for every participant**. Raising the percentile buys one or two tracks' headroom against a
problem that scales with the roster — it is not a fix.

It is also the single root cause of three separate symptoms: the weak veto, the contaminated
mix gains (defect 6), and truncated bleed fragments surviving into the transcript.

**Shape of the fix:** two passes. Attribute without the veto, compute each track's reference
from the candidates that *won* attribution, then apply the veto. The current single pass is
circular — the veto needs a reference that attribution has not produced yet.

### 2. `ingest` refuses 24-bit `orig` files, and gives a false reason

`timeline/layout.py:139` raises `undecodable_source` for anything that is not `pcm_f32le`,
saying *"An integer format cannot be converted to float32 exactly, so it is refused rather than
quietly rounded (ADR-0011)."*

**That is false for 24-bit specifically.** A 24-bit signed integer's range fits exactly in
float32's 24-bit significand; the conversion is lossless. The reason is correct for 32-bit
integer and wrong for the format DJI actually produced.

Two of four transmitters in the 2026-08-02 probe wrote `pcm_s24le` from a per-transmitter
setting the operator had not matched across kits — the exact mistake H1's recipe warns about,
which means it will be made again. **This is the item that can cost a whole session**, and it
would be discovered after the recording, not during it. **OQ-007.**

### 2b. A real DJI track's second chunk can be refused as a "material overlap"

_Added during the start phase, from the plan review (`../reviews/M8-plan-20260803-1729.md`,
finding 2). Same class as defect 2 — a real recording `ingest` refuses — and invisible on every
fixture in this repository._

`rasterize.is_frame_quantized` recognizes only a `TimecodeRecord`, so two chunks placed from
`bext` references get a **one-sample** overlap tolerance (`rasterize.py:88`). OQ-004 measured
DJI's references as quantized to **1600 samples**. A perfectly ordinary later chunk whose
reference rounds backward by a few hundred samples is then a *material* overlap, and
`timecode.chunk_overlap_policy` defaults to `reject` — so the session fails.

Nothing has caught it because the synthetic fixtures' references are sample-exact by
construction, and the jam capture has one chunk per track. A four-hour session has several.

The quantum is not derivable from the file: FFprobe does not surface the iXML that declares
the rate, and **OQ-024** proved the receiver's configured rate does not reach an `orig` file.
So it is configuration with a measured default — `timecode.bwf_reference_quantum_samples`,
1600 at 48 kHz, citing OQ-004 and OQ-024 — feeding both this tolerance and defect 5a's floor.

### 3. The timing model still encodes the semantic known to be false

`timeline/rasterize.py:72` treats a BWF reference as samples since **midnight at the file's own
sample rate**, and `inspection/starttime.py:226` stamps that claim into every manifest as
provenance. Neither is true on this hardware (**OQ-004**): the reference is a device-local
frame counter, quantized to 1600 samples.

The reframe made this narrow rather than large — a shared origin is all that alignment needs,
and the jam supplies one — but the pipeline currently *asserts something untrue* about every
real file, and the 24-hour-wrap logic in `timeline/origin.py` assumes a real day these files do
not have. `rg 'OQ-004'` finds every site.

### 4. Nothing prevents `bext.origination_time` being used as a cross-receiver anchor

Measured 2026-08-03: two receivers' wall clocks were **48.7 s apart** while their timecode
agreed to under one frame. Wall clock is fine for archival naming and for a human reading a
report; as a cross-receiver anchor it is off by nearly a minute. Nothing in the code or tests
stops that use, and OQ-004 previously recorded it as a promising hint.

### 5. `sync_qa` reports a false alarm on every healthy session, and discards correct measurements

Two independent problems, both structural:

**(a) The constant-offset threshold sits below the hardware quantum.** `sync_qa.drift_warn_ms`
defaults to 5 ms, and the `samples` jam-capture run raised `timecode_disagreement` at **+11.31 ms** — well
inside the 33.3 ms frame quantum that **OQ-024** established as the floor. Any cross-receiver
disagreement under one frame is expected. A threshold below the quantization floor fires on
every healthy session, which trains the operator to ignore the one warning that matters.
Constant-offset and start-to-end *change* need separate thresholds: the first cannot be finer
than a frame, the second can stay at millisecond scale.

**(b) A correct measurement is thrown away rather than reported with low confidence.**
`min_correlation` (0.5) is calibrated for a clap. On ordinary speech the run produced six
measurements whose lags — 10–31 ms — match the independent hand measurement that validated the
jam, and reported five of them as `sync_qa_inconclusive` ("no shared transient found"). The
instrument found the right answer and discarded it. Reporting the lag with its correlation,
marked low-confidence, keeps the evidence; the operator can then distinguish "no transient was
recorded" from "the jam failed", which today look identical.

The threshold *value* is tuning and out of scope (**OQ-025** covers whether a chirp should make
this robust). Preserving the measurement is not.

### 6. The mix's level-correction warning names the wrong cause

`mix_level_correction_clamped` says *"a lav this far out is usually a mounting or gain problem
the mix should not hide."* On the jam capture it fired for tx-d against a reference of −57.80 dBFS —
which is the bleed level, not a mounting problem. The transmitter gains match to **2.7 dB**
(held levels −38.4, −38.5, −38.8, −41.1). The message would send an operator to check hardware
that is fine.

Downstream of defect 1; verify it is resolved by that fix rather than editing the string alone.

### 7. M4's deferred three-way collapse decision

M4's closeout deferred a fix and set the condition for dropping it: *"if three lavs never agree
closely enough for the shape to occur, delete the docstring's claim rather than write the
pass."*

**They agree easily** — three tracks within 32 ms with identical text, and a four-way group
that collapsed cleanly with the best source score winning. So the claim stays and the condition
for deleting it is not met.

But the pathological ordering itself (A absorbs B; C, with a better source score, is then
blocked from absorbing A) **was not observed** — those three never entered collapse, being below
`min_text_words`. So this remains what M4 called it: a tidiness fix whose failure mode is
already the safe one. **Decide explicitly and record the decision.** Note the interaction:
fixing truncation elsewhere would push such groups into collapse for the first time.

## Diagnostics to add (tier 2)

Neither changes behaviour. Both exist so the first real session yields evidence rather than
output that reads plausibly.

### 8. Surface each track's speech reference and candidate count in the activity graph

Defect 1 was invisible: it had to be inferred by measuring the audio by hand and reverse-
engineering a percentile. The graph already records every candidate's level and its track's
reference — make the reference, the candidate count, and how many were bleed-suppressed
legible per track in the artifact and the report. `OQ-017` then becomes reading one session.

### 9. Count and report words dropped at the ownership boundary

`transcript/segments.py::_owned_words` drops a word that falls inside padding but inside no
ownership interval, silently, by design (ADR-0020). On the 2026-08-02 capture five of eleven
segments lost their opening word and **nothing raised or warned** — the transcript read as
plausible prose. Count them per track, report the total, and name the affected segments.

Count alone did not turn `activity.vad.pad_ms` into a measurement: the first end-to-end run
still needed cache archaeology to distinguish a word 20 ms before an interval from one 500 ms
away. The diagnostic therefore also records, in integer derivative samples, the distance to
the nearest half-open ownership edge, before versus after that edge, and leading versus
non-leading position in the request's returned word list. These are geometry, not a loss
function — a weak lav's padding can contain another speaker's words.

## Completion gate

- [ ] `./scripts/gate.sh` passes, no new skips, and the whole suite also passes from
      `.venv-rocm` (no gate does this; M6a and M6b both found real defects there).
- [ ] **Defect 1:** a track's speech reference is computed only from candidates that won
      attribution. A test with one own-speech candidate and *N* bleed candidates asserts the
      reference lands on own speech **for every N from 1 to 8** — the current code passes at
      N=2 and fails at N=3, so the test must span the boundary.
- [ ] **Defect 1:** a regression test over a bleed-dominated fixture asserts the veto suppresses
      bleed at the measured 17.4 dB ratio and still vetoes genuine simultaneous speech.
- [ ] **Defect 2:** a 24-bit `_orig` source ingests and reaches the working path, with a test
      asserting the converted samples are **bit-exact** against the integers. A 32-bit *integer*
      source is still refused, with the reason that is true for it.
- [ ] **Defect 3:** `rasterize.py` and `starttime.py` describe what the hardware does; no
      artifact claims a midnight-relative reference. `rg 'OQ-004'` shows every remaining
      citation is accurate, and OQ-004 records which sites changed.
- [ ] **Defect 4:** a test asserts no cross-receiver placement depends on
      `bext.origination_date`/`origination_time`, and that a session whose two receivers'
      wall clocks disagree by 48.7 s still places correctly.
- [ ] **Defect 2b:** two chunks placed from `bext` references get an overlap tolerance of the
      configured BWF quantum, not one sample. A 1600-sample backward rounding on a second chunk
      places cleanly under the default `reject` policy; a genuinely material overlap still
      fails.
- [ ] **Defect 5a:** constant-offset and drift thresholds are separate settings, and every
      threshold comparison is in **integer samples** (INV-04), never floating-point
      milliseconds. The constant-offset default is the largest **effective quantization among
      the session's own evidence** — not the configured frame rate, which **OQ-024** proved does
      not reach an `orig` file. A session with a cross-receiver offset of 11.31 ms produces
      **no** `timecode_disagreement`; one at 120 ms does; and a session configured `60F`
      behaves like the 30 fps one, because the source quantum is unchanged.
      _Amended during the start phase: the original criterion said "at least one frame at the
      session's rate", which the plan review showed would give a `60F` session a ~17 ms floor
      against source timing that still has a 33.3 ms quantum — reinstating the false alarm this
      defect exists to remove (`../reviews/M8-plan-20260803-1729.md`, finding 4)._
- [ ] **Defect 5b:** a measurement below `min_correlation` is reported with its lag and
      correlation, marked low-confidence, and is distinguishable in the report from "no
      transient found at all".
- [ ] **Defect 6:** with defect 1 fixed, the calibrated fixture no longer clamps level
      correction, or if it does, the warning names the real cause.
- [ ] **Defect 7:** decided and recorded — either the chain-resolution pass exists with the
      A/B/C case as a test, or `collapse.py`'s docstring describes actual behaviour. An ADR
      states which and why.
- [ ] **Diagnostic 8:** the activity artifact and report carry per-track reference level,
      candidate count, and suppression count. Reproducing defect 1 from the artifact alone
      requires no audio measurement.
- [ ] **Defect 7 (identity):** changing collapse changes what a `transcript-records.json`
      *means*, so transcript semantics are versioned separately from ASR semantics — a future
      collapse change costs a re-render, not four hours of re-inference. An ADR states the
      split. _Added during the start phase from the plan review, finding 6._
- [ ] **Diagnostic 9:** dropped words are counted per track, reported, and their segments
      named. A fixture with a word starting 50 ms before its ownership interval reports
      exactly one dropped word. The metric is **defined** — dropped `(request, word)` pairs —
      and tested over a merged outcome owning two candidate groups, a true padding-only word,
      and the same padding word returned by two overlapping requests. The report also carries
      the exact edge-distance histogram in derivative samples, before/after counts, and
      leading/non-leading counts; a composed-run test proves those fields reach the report.
- [ ] A fixture reproducing the jam capture's acoustics is checked in — **synthetic audio
      calibrated from the measurements** (direct ≈ −39 dBFS, bleed ≈ −56, floor ≈ −66,
      17.4 dB rejection), not committed session audio (INV-06, and H1's no-audio rule).
- [ ] Every deliverable stays byte-stable on rerun (INV-02) and `raw/` is untouched (INV-01).

## Explicitly not in this milestone

- **Threshold tuning.** `vad.pad_ms`, `duplicate.min_text_words`, `bleed.veto_db`,
  `bleed.min_score_margin`, `sync_qa.min_correlation`. These need a real table; 47 seconds of
  one operator holding one microphone at a time is not evidence to move a detector on. This
  milestone makes them *measurable*, and H1/H2 move them.
- **The acoustic sync signal** (**OQ-025**). Keeping the jam is the current decision.
- **An activity-side parameter sweep.** Sweeping thresholds against a broken veto measures the
  breakage. If a sweep is wanted, it belongs after defect 1 is fixed.
- **Archival** (M7), which waits on a processed real session.
- Any rework of M5's mixing, M4's rendering, or M6b's adapter beyond what defects 1 and 6
  require.

## Known risks and open questions

- **Defect 1's fix is in the most consequential function in the project.** Bleed suppression
  deleting real speech is the failure this pipeline most needs to avoid, and M4 states the bias
  outright: prefer keeping both and marking overlap. A two-pass attribution changes what gets
  suppressed; every existing bleed test is load-bearing and none should be weakened to
  accommodate it.
  The plan review found the specific case that makes a naive two-pass *worse* than the code it
  replaces: **a quieter person who speaks only during overlap has no solo winners**, so a
  winners-only reference is absent, the veto cannot fire, and they are deleted — where today's
  contaminated reference happens to save them. `mutual_bleed_session` cannot show it, because
  that fixture gives its quiet speaker three solo utterances. Any fix here needs that
  regression before it needs anything else.
- **Answers/advances:** OQ-007 (defect 2), OQ-004 (defects 2b and 3), OQ-017 (diagnostics 8
  and 9 make it measurable), OQ-024 (defects 2b and 5a consume its answer). **Raises OQ-026** —
  whether a DJI receiver's timecode counter wraps at all, and with what period, which
  "device-local counter" no longer implies. **Does not answer** OQ-017's actual thresholds,
  which need H1.
- **The spec is amended by this milestone**, per AGENTS.md's rule that a proved correction
  moves code and spec in one commit: `orig` input is not always 32-bit float, a BWF
  `time_reference` is not samples since midnight, and origins are not calendar-day based.
- **Invariants at risk:** INV-04 (**exact time arithmetic** — `syncqa` already compares
  floating-point milliseconds against a float threshold, and defect 5a adds a second threshold;
  every comparison becomes integer samples), INV-09 (nothing text-derived may flow into the
  activity graph — the two-pass attribution must stay entirely inside the activity package),
  INV-02 (byte-stable artifacts — new report fields must be deterministically ordered, and the
  report's decision subsection must stay semantically stable cold and warm), INV-07 (the
  24-bit reader must stay windowed — NumPy has no packed 24-bit dtype, so unpacking is exactly
  where a whole source gets expanded; proved over the composed path in `test_memory.py`, not by
  a helper assertion), INV-08 (new activity fields change the graph identity, so caches miss
  once, which is correct), INV-12 (never invent timing — defect 3 removes an invented semantic
  rather than adding one).
- **The activity graph schema was frozen at M3's gate.** Diagnostic 8 adds fields to it.
  Additive optional fields only (ADR-0005); if anything more is needed, that is an ADR and a
  schema version bump, not a quiet edit.

## Working plan

_Scratch. Replaced by the Closeout at the end of the milestone._

**Preconditions.** Tree clean; M6b `closed`; `./scripts/gate.sh` green at `6d00b35` — 8 checks,
2294 tests, zero skips.

**A charter correction, made at the start phase.** This document and `STATE.md` call the jam
capture `samples2`. The operator has since deleted the old `samples/` (the 2026-08-02 probe) and
moved `samples2/` into its place, so every reference here is renamed to `samples/`. The
consequence is not cosmetic: `tests/test_qwen_smoke.py` discovers `samples/*.wav` by glob, so
the `host_smoke` suite is now measuring **different audio**, and M6b's closeout warns that its
OQ-018(2) delta bound "is a bound on which file sorts first". Re-running `host_smoke` and
re-recording those numbers is part of verify.

### Order, and the files each item touches

Defect 1 first — it is the root cause of defect 6 and of the contaminated mix gains. Each item
lands as its own commit with its tests.

**1. The two-pass speech reference** (`activity/bleed.py`). `attribute` runs the gate twice:
a **bootstrap** pass scored against today's all-candidates percentile with the veto disabled,
to find out which candidates win attribution; then the reference recomputed as
`REFERENCE_PERCENTILE` of the **winners'** levels; then a rescore and the authoritative gate
with the veto.

The reference falls back in one direction only, and the direction matters:

| winners | reference | why |
| --- | --- | --- |
| ≥ `bleed.min_attributed_reference_candidates` (**new**, default 1) | percentile of the winners | a winner is direct evidence that this is the wearer speaking; one of those beats three of an unclassified mixture, which is why it does not reuse `min_reference_candidates` |
| zero, but ≥ `min_reference_candidates` candidates overall | today's all-candidates percentile | **the plan review's case**: a quieter person who speaks *only* during overlap has no solo winners, and a winners-only rule would delete them. Today's contaminated reference happens to save them, and losing that would be a regression |
| zero, and below `min_reference_candidates` | `None` | `delayed_bleed_session`'s silent listener — a track that only ever *hears* is exactly what the gate must suppress |

`mutual_bleed_session` proves the main path safe: the bootstrap pass suppresses `tx-b`'s
contested overlap, so `tx-b`'s winners are its three solo utterances — precisely the right
reference — and the veto then fires on a cleaner number than today's. Its contrast test keeps
its exact shape by raising **both** floors beyond reach. Bumps `ACTIVITY_SEMANTICS_VERSION`.

**2. Exactly-convertible PCM** (`timeline/pcm.py`, `timeline/layout.py`, `fixtures/wav.py`,
`fixtures/session.py`). The allowlist becomes a principle, stated in two parts: a format is
accepted when it is a **signed little-endian integer or IEEE float** *and* converts to float32
with **zero error**. That is `pcm_f32le`, `pcm_s24le` and `pcm_s16le`. `pcm_s32le` fails the
second half; `u8` fails the first — WAV 8-bit PCM is *unsigned with an offset of 128*, so the
plan's original "signed integer PCM at 8 bits" was simply wrong — and each is refused with the
reason that is true of it. `PcmSource` gains a sample format and `bytes_per_sample`; integer
decode scales by `2**(bits-1)`, a power of two, so the division is exact. Chosen over an
s24-only allowlist because refusing s16 with "an integer format cannot be converted exactly"
would be the same defect one format over.

**2b. The BWF overlap tolerance** (`timeline/rasterize.py`, `config.py`). `is_frame_quantized`
learns about `BwfSampleReferenceRecord`, and `quantization_tolerance_samples` uses
`timecode.bwf_reference_quantum_samples` (default 1600, citing OQ-004/OQ-024) for it. One
number, evidence-derived rather than read off the receiver's menu, and it also supplies item
5's floor.

**3. The recorder's origin, not midnight** (`inspection/starttime.py`, `timeline/rasterize.py`,
`artifacts/manifest.py`, `artifacts/timeline.py`, `timeline/origin.py`). A `bext.time_reference`
is an unsigned sample count from **the recorder's own timecode origin** at the file's own rate.
Three consequences and one deliberate non-change:

- The *arithmetic* stays, and the assumption under it gets registered. "Device-local counter"
  no longer implies a 24-hour period, which the plan review was right to call out — but
  removing the wrap would delete a spec-required, tested capability (`rollover_session`, M2's
  gate) on the strength of a hypothesis about one vendor. So `SECONDS_PER_DAY` and the unwrap
  survive, the name and comment are corrected, and **OQ-026** records the open question, cited
  from `cycle_units`. The existing `midnight_rollover_inferred` warning already refuses ties
  rather than guessing, and a real DJI session never reaches the inference: its counters are a
  few hundred seconds apart.
- Mixing a `bext` reference with a timecode tag stays **permitted**, because on this hardware
  `time_reference` *is* the jammed timecode count (OQ-023) — the two genuinely share the
  recorder's origin. A configured `origin_timecode` is likewise a statement in that domain,
  which is how an operator reading a receiver display uses it. The canonical fixture stays
  valid; `mixed_time_domains` keeps its fractional-rate condition and stops saying "real
  midnight".
- The artifact stops claiming a day origin it does not have. `ZeroDomain`'s `real_time` becomes
  `recorder_epoch` and `since_day_origin_samples` becomes `since_domain_origin_samples`. That is
  not the additive change ADR-0005 permits, so `TIMELINE_SCHEMA_VERSION` goes to **2** with a
  regenerated `schemas/timeline.schema.json` and an ADR. Keeping the *value* rather than nulling
  it preserves M2's consistency check — zero plus a source's placement equals that source's own
  time in its domain.

**4. Wall clock never anchors placement** (`timeline/origin.py::_cycles_from_dates`). This is
where `origination_date` actually reaches placement, and it assigns **whole 24-hour cycles** —
two receivers whose wall clocks straddle midnight would be placed a day apart on evidence
measured 48.7 s wrong. A date read *from a file* becomes descriptive only; only an operator
assertion (`timecode.origin_date`, a `source_time_overrides` entry's `recording_date`) may
assign a cycle. No schema change is needed: `ManifestStartTime.strategy` already says which
strategy produced the evidence, and `recovery_*` is exactly the operator half. Falling through
to the existing inference reads the counters themselves and involves no wall clock.
`creation_time` — FFprobe's name for `bext.origination_time` — is read nowhere, and a test keeps
it that way.

**5. `sync_qa`'s two defects** (`timeline/syncqa.py`, `config.py`). **(a)** `SyncQaConfig` gains
`offset_warn_ms: int | None = None`, defaulting to the **largest effective quantization among
the session's own evidence** — `bwf_reference_quantum_samples` for a BWF reference, one frame at
the configured rate for a timecode tag. Not the frame-rate setting, which OQ-024 proved does not
reach an `orig` file: deriving it from `60F` would give a 17 ms floor against a 33.3 ms source
quantum and reinstate the false alarm. `drift_warn_ms` keeps 5 ms and now governs only the
start-to-end *change*. The achievability check lives on `SessionConfig` — M5's
`overlap_min_gain_db` pattern — and **refuses** a value below the quantum rather than silently
raising it.

**Every comparison becomes integer samples** (INV-04). `_assess` currently compares
`found.lag_ms`, a float, against a float threshold; the plan review caught that the plan was
about to add a second one. Thresholds convert to exact derivative-sample counts once, lags and
lag deltas compare as integers, and milliseconds appear only in rendered report text.

**(b)** Three outcomes replace two: `sync_qa_measured`, `sync_qa_low_confidence` (below
`min_correlation` but a real peak — **the lag and correlation are reported**, marked
low-confidence, and raise no disagreement), and `sync_qa_no_signal` (no energy at all). "No
transient was recorded" and "the jam failed" stop looking identical.

**6. The mix's clamp warning** (`mix/levels.py`). Verify defect 1's fix removes the clamp on the
calibrated fixture *first*, as the charter requires, then reword what remains to name both
readings and carry diagnostic 8's candidate count.

**7. Duplicate-chain resolution** (`transcript/collapse.py`). **Decided: implement**, at the
operator's direction. `_pairs` is resolved in order of descending winner score, tie-broken by
(winner id, loser id), so the best-scoring segment absorbs first and the chain shape cannot
arise. A deterministic sort, not new logic: `_is_duplicate` still gates everything, so only
*which* of two mutual duplicates survives changes, and the segment now deleted is one the
three-condition test already called a duplicate. The alternative — rewriting the docstring to
match a greedy accident — was rejected because the docstring states the rule that is right.

**7b. Transcript semantic identity, split rather than bumped.** After this change two
`transcript-records.json` documents both stamped version 1 could carry different duplicate
survivors, and no consumer could tell them apart (INV-08). Request-shaping and submission
semantics stay in the ASR cache key; **assembly, collapse and rendering get their own version**,
recorded in the records artifact and kept *out* of that key, so a future collapse change costs a
re-render rather than four hours of GPU inference. The plan review is right that this is
cheapest now, before any real-session cache exists — which is this milestone's whole premise.

**8. Per-track diagnostics** (`artifacts/activity.py`, `activity/runner.py`,
`artifacts/report.py`). `ActivityTrack` gains **additive optional** fields (schema stays at 1):
`candidate_count`, `reference_candidate_count` (what the reference was actually measured from),
`retained_candidate_count` and `suppressed_candidate_count`, plus a per-track report decision
carrying the same numbers. The first two are deliberately distinct names — after item 1 the
winners and the final retained set are no longer the same population, and one field called
"attributed" would be unreadable. The hardcoded field allowlist in
`tests/test_activity_artifact.py` moves in the same change.

**9. Dropped-word counting** (`transcript/segments.py`). Computed **per outcome**, in
`draft_segments`, across all of that outcome's ownership intervals — *not* inside `_owned_words`,
which runs once per candidate group and would therefore count group B's legitimately owned words
as drops while processing group A. The metric is stated rather than left to inference: dropped
`(request, word)` pairs, so a padding word that two overlapping requests both return and both
drop counts twice, and the note says so. Emitted per track as a `TranscriptNote`, with affected
candidates as structured `Decision.details` rather than prose. Behaviour unchanged; only
visibility.

**10. The calibrated fixture** (`fixtures/variants.py`). `bleed_dominated_session()` —
**synthetic audio calibrated from the jam capture**, not committed session audio (INV-06):
direct ≈ −39 dBFS, bleed ≈ −56, floor ≈ −66, 17.4 dB rejection. Shaped so the current code
reproduces the defect and the fixed code does not.

### Gate criteria → the proof for each

| Criterion | Proof |
| --- | --- |
| Gate green, no new skips, suite also green from `.venv-rocm` | `./scripts/gate.sh`; the FHS invocation in M6b's closeout |
| **1** reference from attributed candidates only | `test_activity_bleed.py` — one own-speech candidate against *N* bleed candidates, asserted **for every N in 1..8**. Today's code passes at N=2 and fails at N=3, so the test spans the boundary and fails on the current implementation |
| **1** veto suppresses bleed at 17.4 dB and still saves real overlap | The calibrated fixture end to end, plus every existing `mutual_bleed_session` test unweakened — including the contrast test that removes the reference and watches the overlap get suppressed |
| **1** the overlap-only speaker is **not** deleted | The plan review's regression: a quieter track whose every candidate is contested, so it has no solo winners. Retained via the fallback. Written **before** the two-pass code, because it is the case the fix is most likely to get wrong |
| **2** 24-bit ingests bit-exactly; s32 and u8 refused for their own reasons | Round-trip over 2 000 000 random values plus both range edges, parametrized over 16/24 and **failing at 32**; a cross-check that FFmpeg's own decode agrees sample for sample; end-to-end `ingest` on a mixed-format session |
| **2** the 24-bit path stays bounded (INV-07) | A 24-bit case in `test_memory.py`'s ordered event log — every read bounded by the requested window, every expanded array window-sized, a write before the last read. NumPy has no packed 24-bit dtype, so unpacking is exactly where a source gets expanded |
| **2b** a rounded-back second chunk places cleanly | `test_layout.py` — a 1600-sample backward rounding under the default `reject` policy places without a nudge; a genuinely material overlap still fails; `test_rasterize.py` asserts the tolerance comes from the configured quantum |
| **3** nothing claims a midnight-relative reference | `rg 'OQ-004'`; `test_starttime.py` and `test_artifacts.py` on the new assumption text; the schema drift test proves the v2 regeneration |
| **4** no cross-receiver placement depends on wall clock | A session whose every `date`/`creation_time` tag is rewritten — including two receivers **48.7 s apart across midnight** — produces a **byte-identical `timeline.json`**; plus a structural test that `creation_time` is read nowhere |
| **5a** 11.31 ms raises nothing, 120 ms does | `test_syncqa.py`, both directions, **and at `60F`** where the charter's original rule would have reinstated the alarm; plus a configured `offset_warn_ms` below the quantum refused at configuration load |
| **5a** thresholds compare as integers (INV-04) | `test_syncqa.py` — a lag exactly one sample either side of the threshold, which no float-millisecond comparison resolves correctly; plus a structural check that `lag_ms` reaches only report prose |
| **5b** a weak measurement keeps its lag | `test_syncqa.py` — `sync_qa_low_confidence` carries lag and correlation and is distinguishable from `sync_qa_no_signal` |
| **6** the clamp warning names the real cause | The calibrated fixture no longer clamps; a fixture that legitimately does carries the candidate count |
| **7** decided and recorded | The A/B/C case asserting C survives alone, mutation-checked by reverting the sort; ADR |
| **8** defect 1 reproducible from the artifact alone | A test that reads `work/activity.json` for the calibrated fixture and derives the defect with no audio measurement |
| **9** exactly one dropped word, with usable geometry | A fixture with a word starting 50 ms before its ownership interval; zero on the canonical fixture; plus a merged outcome owning two candidate groups (no false drops), a true padding-only word, one padding word returned by two overlapping requests, exact half-open edge distances, and a composed run asserting the report fields |
| Fixture checked in, no session audio | `bleed_dominated_session()`; `.gitignore` already refuses `*.wav` |
| INV-02 / INV-01 | Byte-stability across two runs for all six deterministic artifacts; **plus** the report's provenance and decision subsections asserted semantically identical cold and warm, since the report itself is exempt as a whole and the new decisions live there; the five composed runners already parametrized in `test_raw_guard.py` |
| The spec agrees with the code | `dnd-audio-ingestion-agent-spec.md` amended in the same commits — `orig` is not always 32-bit float, `time_reference` is not samples since midnight, origins are not calendar-day based. Swept on `midnight`, `calendar`, `day origin` and `absolute timecode` as well as `rg 'OQ-004'`, because a false claim carrying no citation is exactly what the id search cannot find. `manifest.schema.json` regenerated alongside `timeline.schema.json` |

### ADRs this milestone will write

- **ADR-0029** — the two-pass speech reference.
- **ADR-0030** — the exactly-convertible PCM rule. **Amends ADR-0011**, whose principle stands
  and whose implementation was too broad.
- **ADR-0031** — the recorder's origin, not midnight: what a BWF reference means, its
  quantization and what that costs the overlap tolerance (defect 2b), what may assign a 24-hour
  cycle, and `timeline.json` schema v2. **Amends the spec** and supersedes the relevant parts of
  ADR-0006/0008/0009.
- **ADR-0032** — duplicate-chain resolution by source score, and the transcript semantic-version
  split that makes it identifiable.

### Invariants at risk, and what stops each

- **INV-04** — every `sync_qa` threshold comparison becomes integer derivative samples;
  milliseconds survive only in rendered prose. A boundary test one sample either side is what
  proves it, because the charter's own 11.31/120 ms cases are far enough from the threshold to
  pass with the violation intact.
- **INV-09** — the two-pass attribution stays entirely inside the activity package. The
  transitive-import test and the prose-rewrite test both still run.
- **INV-02** — new report and artifact fields are deterministically ordered, and the report's
  decision subsection is asserted stable cold and warm.
- **INV-07** — the 24-bit reader stays windowed, proved over the composed path rather than by a
  helper assertion.
- **INV-08** — `ACTIVITY_SEMANTICS_VERSION`, `TIMELINE_SEMANTICS_VERSION` and the new transcript
  assembly version move where behaviour moved, so caches miss once. Correct, not a cost.
- **INV-12** — defect 3 removes an invented semantic rather than adding one; defect 4 removes a
  wall clock from placement entirely.
- **INV-01, INV-05** — no new write path and no new import.

### Amended after the plan review

`./scripts/codex-review.sh plan 8` produced twelve findings
(`../reviews/M8-plan-20260803-1729.md`). **Ten accepted outright, two with a narrower remedy.**
The two that changed this milestone's shape rather than its detail:

- **The planned fix to the bleed veto was a regression** in one case the current code handles
  correctly — an overlap-only speaker with no solo winners. The fallback table in item 1 is the
  remedy, and the regression is written before the code.
- **A defect the charter never listed** — a real DJI track's second chunk refused as a material
  overlap, because a BWF reference gets a one-sample tolerance against a 1600-sample quantum.
  Added as defect 2b.

Rejected in part: removing the 24-hour BWF wrap (registered as OQ-026 instead, because deleting
a spec-required tested capability on a hypothesis is the larger risk), and "no suppression at
all when a track has too few trusted winners" (too strong in the other direction — it makes a
track that only ever hears bleed unsuppressible). Both refusals are recorded in the review with
their reasons.

---

## Closeout

### What works end to end

The 2026-08-03 jam capture now runs through `process` unchanged: 24-bit and float sources
ingest together, BWF references place chunks in the recorder's domain rather than an invented
midnight domain, frame-quantized chunk boundaries tolerate one source quantum, wall-clock tags
cannot move a receiver, and `sync_qa` distinguishes offset, drift, weak evidence and no signal.

Activity computes a two-pass speech reference from attributed winners with the overlap-only
fallback the plan review required. The calibrated 17.4 dB bleed fixture suppresses the copy
without deleting genuine overlap. The mix warning names the measured reference and candidate
population. Transcript duplicate chains resolve best-source-first under a separately versioned
assembly semantic, so changing collapse never invalidates Qwen inference.

The two new diagnostics are usable from artifacts alone. Each track records candidate,
reference, retained and suppressed counts. A dropped `(request, word)` pair records its nearest
candidate, exact half-open edge-distance histogram on the derivative grid, before/after side,
and leading/non-leading position. The 30/50/100 ms real-model A/B proved why all of those are
needed: a count alone combines true direct-source onset loss with words in weaker lav padding.

### Tests and commands run, with results

- `direnv exec . ./scripts/gate.sh` — **8 checks, 2 360 passed, zero skips; gate passed**.
- `nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync pytest -m
  "not host_smoke and not allow_network" -q'` — **2 360 passed** from the ROCm environment.
- Focused transcript suite in-process — **77 passed**; Ruff and mypy clean.
- Three isolated real-model `process` runs at `activity.vad.pad_ms` 30, 50 and 100 — all
  completed with MP3, transcript and report, and each run's raw guard passed.
- Fixed-response counterfactual through 600 ms, reconstructed from the 30 ms ASR cache and
  exact request plans — recorded under OQ-027, explicitly as newly owned pairs rather than
  recovered truth.

The first sandboxed gate attempt failed only in the five tests that deliberately create
sockets: the execution sandbox refused socket creation before the project's network guard
could. Re-running the identical gate in the normal project environment passed all 2 360.

### Decisions made (→ ADRs)

- **ADR-0029** — attributed-winner two-pass speech references, with a conservative fallback
  for a speaker whose every utterance overlaps.
- **ADR-0030** — exactly convertible integer PCM, admitting 24-bit and keeping unsupported
  integer formats refused for their actual reason.
- **ADR-0031** — recorder origin rather than midnight, source-quantum overlap tolerance, and
  descriptive wall-clock tags; timeline schema version 2.
- **ADR-0032** — best-source-first duplicate chain resolution and the ASR/assembly semantic
  version split.

No threshold default changed. In particular `activity.vad.pad_ms` remains 30 ms: 100 ms is a
candidate for the multi-wearer capture, not a value this one-operator recording can establish.

### Assumptions made and open questions raised

- **OQ-026 raised:** whether DJI's recorder counter wraps at all, and with what period. The
  spec-required 24-hour unwrap remains until hardware disproves it rather than being deleted on
  a hypothesis.
- **OQ-027 raised, then narrowed after the end-to-end reconstruction:** the aligner's first
  word can absorb request lead-in, but production damage is bounded by `transcript.pad_ms`.
  All 30 observed drops were within 500 ms; the earlier claim of an unbounded seconds-scale
  consequence was wrong and is corrected in the ledger.
- **OQ-017 remains open.** The A/B gives a candidate range and demonstrates the tradeoff; it
  does not provide multi-speaker ground truth.

### Notes for future implementors

**Do not optimize the dropped-word count.** On the baseline, apparent direct-source openings
all sat in the 20 ms fixed-response bucket, but so did words on weaker copies. The actual 50 ms
rerun retained only four fewer drops because moving the activity edge also moved the request
window and therefore the aligner's timestamps. At 100 ms all four known direct openings were
present, while a second `Okay` and more fragments also survived. Cached geometry is a
sensitivity curve; every candidate value needs one real-model run.

`vad.pad_ms` is not transcript-only. Candidate structure stayed at 18/17/1 across the A/B,
but wider intervals changed candidate statistics and pushed the two quiet-track speech
references down by as much as 1.44 dB, worsening their +6 dB mix correction clamps. Score the
activity artifact, transcript and mix together.

The listening extracts and complete scratch table are beside the isolated sessions under the
operator's `dnd-audio-pad-ab-20260803/` scratch directory; they are intentionally untracked and
reproducible from the source session. H1's recipe now logs unique hard-onset phrases against
intended track ids so the next measurement can label benefit and cost rather than infer them
from source score.

### Deviations from this charter, and why

- The plan review's overlap-only-speaker case required a fallback population in the two-pass
  reference; using winners unconditionally would have deleted the speaker the fix was meant to
  protect.
- The BWF 24-hour wrap was retained under OQ-026 rather than removed. The existing capability
  is spec-required and a real DJI session never reaches the ambiguous inference path.
- Diagnostic 9 grew after verify. Its original count met the written gate but failed its stated
  purpose on the first real run: choosing a pad still required reconstructing edge distances
  from cache. Exact geometry and a composed report test close that usability gap.
- The 30/50/100 study was kept out of the default. M8 explicitly excluded threshold tuning;
  the study bounds the candidate and improves H1/H2, while respecting that exclusion.

### Downstream charters updated

- **H1** carries the third receiver, power-cycle and `orig`/`edit` evidence, plus logged
  hard-onset phrases, short handoffs and known track identity for OQ-017/OQ-027.
- **H2** still owns long-baseline drift, first-session disk use, natural truncation and
  duplicate-text calibration.
- **OPEN-QUESTIONS.md** records OQ-026 and OQ-027, the corrected bound, the real A/B and the
  fixed-response counterfactual.
- The product spec, affected ADRs, schemas and later charter wording were amended alongside
  the structural fixes; no code/spec disagreement remains.

### Next smallest step

Record and process **H1**. The software path is complete and the remaining decisions need
multi-wearer audio or receiver-display observations that cannot be reconstructed later. Keep
`activity.vad.pad_ms` at 30 for the baseline, run 100 ms as the one candidate comparison, and
do not commit either result until direct-source and overlap ground truth are scored.
