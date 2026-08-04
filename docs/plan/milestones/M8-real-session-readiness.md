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
session that cost six people an evening. The 2026-08-03 `samples2` run — four tracks, two
receivers, real GPU, exit 0 — surfaced the specific reasons, and they are not thresholds.

Full evidence: `docs/fixtures/2026-08-03-jam-verification.md` and the run's findings.

## The defects, with their sites and the evidence

### 1. The bleed veto's reference is computed from bleed *(root cause — do this first)*

`activity/bleed.py::speech_references` sets each wearer's speech reference to
`REFERENCE_PERCENTILE` (75) of **all** that track's candidate levels — own speech and bleed
together. `veto_db` (12.0) then keeps any candidate within 12 dB of it. When most of a track's
candidates are bleed, the percentile lands on bleed, and the veto protects bleed from
suppression.

Measured on `samples2`, where direct-to-bleed is 17.4 dB:

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
defaults to 5 ms, and the `samples2` run raised `timecode_disagreement` at **+11.31 ms** — well
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
the mix should not hide."* On `samples2` it fired for tx-d against a reference of −57.80 dBFS —
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

This is what turns `activity.vad.pad_ms` from a guess into a measurement on the first real
session, without changing it now.

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
- [ ] **Defect 5a:** constant-offset and drift thresholds are separate settings; the
      constant-offset default is at least one frame at the session's rate. A session with a
      cross-receiver offset of 11.31 ms produces **no** `timecode_disagreement`; one at 120 ms
      does.
- [ ] **Defect 5b:** a measurement below `min_correlation` is reported with its lag and
      correlation, marked low-confidence, and is distinguishable in the report from "no
      transient found at all".
- [ ] **Defect 6:** with defect 1 fixed, the `samples2` fixture no longer clamps level
      correction, or if it does, the warning names the real cause.
- [ ] **Defect 7:** decided and recorded — either the chain-resolution pass exists with the
      A/B/C case as a test, or `collapse.py`'s docstring describes actual behaviour. An ADR
      states which and why.
- [ ] **Diagnostic 8:** the activity artifact and report carry per-track reference level,
      candidate count, and suppression count. Reproducing defect 1 from the artifact alone
      requires no audio measurement.
- [ ] **Diagnostic 9:** dropped words are counted per track, reported, and their segments
      named. A fixture with a word starting 50 ms before its ownership interval reports
      exactly one dropped word.
- [ ] A fixture reproducing the `samples2` acoustics is checked in — **synthetic audio
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
- **Answers/advances:** OQ-007 (defect 2), OQ-004 (defect 3), OQ-017 (diagnostics 8 and 9
  make it measurable), OQ-024 (defect 5a consumes its answer). **Does not answer** OQ-017's
  actual thresholds, which need H1.
- **Invariants at risk:** INV-09 (nothing text-derived may flow into the activity graph — the
  two-pass attribution must stay entirely inside the activity package), INV-02 (byte-stable
  artifacts — new report fields must be deterministically ordered), INV-08 (new activity
  fields change the graph identity, so caches miss once, which is correct), INV-12 (never
  invent timing — defect 3 removes an invented semantic rather than adding one).
- **The activity graph schema was frozen at M3's gate.** Diagnostic 8 adds fields to it.
  Additive optional fields only (ADR-0005); if anything more is needed, that is an ADR and a
  schema version bump, not a quiet edit.

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
