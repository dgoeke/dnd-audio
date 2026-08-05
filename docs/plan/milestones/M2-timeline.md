# M2 — Reconstruct six synchronized virtual tracks

**Status:** closed
**Depends on:** M1
**Spec sections:** Milestone 2; Tests and acceptance criteria 2, 3, 4, 13

## Goal

`dnd-audio ingest` turns the manifest into six synchronized virtual tracks: chunks
ordered by embedded timecode, real gaps preserved as silence, a lossless streamed
48 kHz working path, and cached 16 kHz mono derivatives — with an exact recorded
mapping between source samples, working samples, and session time.

## Completion gate

- [x] Session time zero from the earliest valid source start unless
      `timecode.origin_timecode` supplies an explicit origin on `origin_date`.
      `origin_date` is never inferred from a date-shaped `session_id`.
- [x] Chunks sorted by parsed start time, not filename order; each chunk's expected
      end validated against the next chunk's start.
- [x] Real gaps preserved as silence. A transmitter switched off and back on does
      not slide later audio earlier. Verified against synthetic ground truth.
- [x] Overlaps detected. Only quantization-scale overlaps resolved automatically;
      anything larger warns and requires explicit policy rather than discarding audio.
- [x] Midnight rollover: `infer_forward` infers a single forward rollover only when
      chunk sequence and session span make it unambiguous, and records the
      decision. Ambiguity demands a dated origin or an override, never an ad hoc
      interpretation.
- [x] Exact sample-position tests for non-drop, fractional (24000/1001,
      30000/1001), drop-frame, rollover, and explicit-override cases (INV-04).
- [x] Lossless 48 kHz float working path is streamed/windowed over a segment map,
      never six session-length arrays in RAM; contiguous intermediates use RF64;
      work-space and disk preflighted (INV-07).
- [x] 16 kHz mono derivatives cached, with resampler delay and end rounding
      accounted for in the 48↔16 kHz mapping.
- [x] Aligned output duration is set by the latest track end and matches within one
      48 kHz sample.
- [x] A selected 44.1 kHz source, or chunks within one track disagreeing on sample
      rate, fails before timeline construction with a clear diagnostic.
- [x] Optional clap cross-correlation runs as QA only: it reports disagreement with
      timecode and never overrides valid timecode. Lag is measured near both ends
      and a materially changed lag warns (drift evidence, not correction).
- [x] An interface hook exists for a future affine time warp, unused in the MVP.

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
  question, which asks for measured full-pipeline disk use — that needs M11's live
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

---

## Closeout

_The working plan that stood here during the start phase is preserved in the commit
history (`183c94b`), together with the plan review it was revised against
(`../reviews/M2-plan-20260802-1241.md`). What follows is what actually happened._

### What works end to end

`uv run dnd-audio ingest /path/to/session` — the stage that turns evidence into a timeline.

It runs inspection every time (served warm from M1's content cache), rasterizes each
source's timing evidence onto the 48 kHz grid, decides where session zero is and which day
each chunk belongs to, lays out each track's chunks with real gaps preserved as explicit
silence, builds the 16 kHz derivatives through one checked-in filter, optionally correlates
the tracks against each other as QA, and writes `work/timeline.json` plus a rewritten
`output/ingest-report.json`.

```
$ uv run python scripts/make_fixture.py <dir>
wrote 6 tracks, 12 chunks, 7.0 MB
  session zero   3283200000 samples since midnight
  gap            tx-c: samples 240000-384000

$ uv run dnd-audio ingest <dir>
  reconstructed 6/6 track(s), 10.500s aligned (504000 samples)
  timeline  <dir>/work/timeline.json
  report    <dir>/output/ingest-report.json
```

A second run is byte-identical (`timeline.json` at `b143fb5e…` both times) and does no
probing and no filtering: **18 cache hits, 0 misses** — twelve inspection captures and
six derivatives. The report records `inspect` as
`complete` with `origin: reused` and `reconstruct` as `complete` with `origin: executed`;
the four stages M2 does not own are `skipped` with a reason naming why.

The one decision the canonical fixture provokes is the one it was built to provoke:

```
chunk_gap_preserved  raw/tx-c/TX01_MIC002_20260815_190008_orig.wav
  144000 samples of silence precede this chunk, because the previous one ended at
  240000 and this one's own evidence starts at 384000. The transmitter was not
  recording in between.
```

`work/cache/audio/16000/` then holds six content-addressed float32 WAVs with their
sidecars — 4.1 MB of working audio for a 10.5-second session, none of it in the repository
and all of it regenerable.

The 48 kHz working path is **not** a set of files. It is the segment map in
`timeline.json`, read through `TrackReader` in bounded windows. `--materialize-48k` will
write contiguous RF64 files, but they are disposable cache artifacts and nothing in the
pipeline is allowed to depend on them (ADR-0011).

### Tests and commands run, with results

`./scripts/gate.sh` — **8 checks, 923 tests, zero skips** (924 collected, one `host_smoke`
deselected):

```
== type check ==      Success: no issues found in 88 source files
== pytest ==          923 passed, 1 deselected in 42.34s
== placeholder scan ==  no unexplained placeholders (83 files scanned)
== plan consistency ==  ledger consistent: 11 milestone(s), 13 invariant(s),
                        16 open question(s), 12 ADR(s)

  pass  system dependencies      pass  pytest (offline, cpu)
  pass  ruff check               pass  lock is current
  pass  ruff format              pass  placeholder scan
  pass  type check               pass  plan consistency

GATE PASSED
```

**362 of those tests are M2's.** By area: rasterize 55, resample 49, derivatives 36,
origin 31, ingest end-to-end 28, timeline artifact 22, reader 21, layout 18, syncqa 18,
fir 15, preflight 15, wavwrite 14, raw guard 14, timeline truth 12, warp 9, memory 5.

Every completion-gate criterion was proved with executed output during the verify phase.
The table below is the one the working plan carried, corrected during verify and kept here
because it is the milestone's evidence:

| Criterion | Proof |
| --- | --- |
| Session zero; `origin_date` never inferred from `session_id` | `test_origin.py::TestSessionZero` — explicit origin, earliest-source, a date-shaped `session_id` with no `origin_date`, and negative offsets in both branches; plus `test_a_shifted_timeline_records_the_origin_it_actually_has` |
| Chunks sorted by parsed start; expected end validated against next start | `test_layout.py::TestOrdering`, on a fixture whose filename order contradicts its timecode order |
| Real gaps preserved; later audio does not slide earlier | `test_timeline_truth.py` against `FixtureTruth`: all twelve chunks land on the samples the generator declared **before writing any audio**, and the only silence segment is the gap it declared. Plus `test_reader.py::test_a_gap_reads_as_silence` and, end to end, tx-c's post-gap speech at 408000 |
| Overlaps detected; only quantization-scale resolved | `test_layout.py::TestOverlaps` — sub-tolerance nudge with both starts retained; material overlap fatal under `reject`; `nudge_later` proven to preserve the total sample count; a three-chunk case asserting C keeps its own evidence position, and a cascade case |
| Rollover inferred only when unambiguous, and recorded | `test_origin.py::TestRollover`, crossed with 23.98F, 29.97F and 29.97DF, because each wraps by a different number of real seconds; ambiguity and `reject` both fatal; two equally wide gaps refused rather than resolved; end-to-end rollover fixture |
| Exact sample positions: non-drop, 24000/1001, 30000/1001, drop-frame, rollover, override | `test_rasterize.py` — every expectation written longhand as `Fraction` arithmetic, never obtained from the code under test. Two are *contrasts*: naive rollover shown to differ from correct rollover by 86.4 s, and round-then-subtract shown to be one sample short of subtract-then-round. Plus an end-to-end 29.97DF fixture and an end-to-end recovery-override run |
| Streamed/windowed; RF64; preflight | `test_memory.py` instruments reader, resampler and writer into **one ordered event log** and asserts a write happens before the last read — nothing that accumulates a session-length array can satisfy that. `test_wavwrite.py::TestRf64` round-trips the header through `inspection.riff`; `test_preflight.py` |
| 16 kHz derivative cached; resampler delay and end rounding accounted for | `test_fir.py` (passband, stopband, ripple, attenuation, symmetry, DC gain, design reproducibility); `test_resample.py` — impulse position, every input-length residue mod 3, first and last samples, across gaps and chunk boundaries, varied window partitioning, streamed output byte-identical to one-shot; `test_derivatives.py` varying each identity component, proving the cache is consulted, and rejecting a truncated file, an orphaned sidecar, and a self-inconsistent one |
| Aligned duration set by the latest track end, within one sample | `test_ingest_run.py::test_aligned_duration_matches_the_latest_source_end`, read through the virtual segment reader — 504000 samples on the canonical fixture |
| 44.1 kHz or intra-track disagreement fatal *before* construction | `test_ingest_run.py::TestRefusalsHappenBeforeConstruction` — through the CLI, each case starting from a **stale `timeline.json` already on disk**: nonzero exit, structured error, the stale artifact removed, and placement, layout, the reader and the writer spied on and proven un-entered |
| Clap correlation is QA only; lag at both ends; changed lag warns | `test_syncqa.py` — agreement; a constant offset reported while the timeline stays byte-identical; a no-clap case reporting low confidence rather than a lag; and the drift fixture, whose two tracks carry **identical metadata** and differ only in audio (tx-b hears the end clap 960 samples late). Measured +0.000 ms at the start, +20.000 ms at the end, correlation 0.9999 and 0.9996 |
| An affine time-warp hook exists, unused | `test_warp.py::test_a_non_identity_warp_moves_the_timeline`, with a test-local affine implementation — a seam that cannot fire is decoration |
| Report behaviour | `test_ingest_run.py::TestTheCanonicalSession` — `origin: reused` on a warm run, the manifest rewritten rather than trusted, both deliverables hashed. No floats anywhere in `timeline.json` (`test_timeline_artifact.py` walks the serialized document) |

Two independent reviews ran during verify. Codex (`../reviews/M2-code-20260802-1508.md`)
said plainly *"I would not close M2 in its current state"* and produced six findings: five
were real, four were reproduced with a concrete failing case and fixed, and one was a
documentation error. A second fresh-context reviewer found nothing above its confidence
threshold but flagged a real file-handle leak in `syncqa`. Eleven deliberate falsifications
of load-bearing code all produced failures, and each of the four fixes was then falsified
in turn — which is how the first regression test for the shifted-origin bug was found to
cover only one of its two branches.

### Decisions made (→ ADRs)

All four were written **before any code**, and two were corrected during verify when the
review showed the record claimed more than the behaviour did.

- **[ADR-0008](../decisions/0008-rasterizing-time-onto-the-sample-grid.md)** — evidence
  becomes an exact `Fraction` of seconds in its own domain; the session-relative difference
  is taken **once** and only that result is quantized, half away from zero, by the one
  quantizer `determinism.to_samples`. `session_position` takes the source time and zero
  *separately* so a caller cannot quantize an absolute position and subtract a quantized
  origin. Also fixes the overlap tolerance as a property of the evidence *pair*: one sample
  when both starts are sample-exact, one whole frame when either came from a timecode.
- **[ADR-0009](../decisions/0009-session-zero-and-the-24-hour-wrap.md)** — session zero,
  and the 24-hour wrap unwrapped **in each evidence domain's own units**. A recorded
  origination date beats inference outright. With no configured origin, zero is *defined*
  as the earliest start and the whole timeline shifts, which is what keeps a signed
  recovery offset meaningful. *Corrected during verify:* the day-assignment rule is the
  **shortest-arc heuristic**, not a proof of unambiguity — now named as one, registered as
  OQ-016, cited in `_cycles_by_largest_gap`, and stated in the operator's own warning.
- **[ADR-0010](../decisions/0010-chunk-overlap-policy.md)** —
  `timecode.chunk_overlap_policy`: `reject` (default) is fatal on a material overlap,
  `nudge_later` moves the later chunk. Neither discards a sample. *Corrected during verify:*
  it claimed `nudge_later` "shifts everything after the overlap". It does not, and should
  not — moving a chunk whose own timecode is good, to preserve one track's internal
  spacing, would misalign it against the other five in a pipeline whose entire purpose is
  cross-track synchronization. The gap-consumption consequence is now stated explicitly and
  pinned by two tests.
- **[ADR-0011](../decisions/0011-the-working-audio-path.md)** — the segment map *is* the
  working path. Sources are read by seeking into the RIFF `data` chunk, because a mix pass
  asks for a window and a pipe is not that. Mono float32 only: s32 → float32 is not
  lossless (24 mantissa bits; `2147483647` becomes `2147483648.0`), so an integer source is
  named and refused rather than quietly rounded.

### Assumptions made and open questions raised

**Raised.** Three, all of which change no placement and no artifact — they exist so the
assumptions are findable when a real session contradicts them:

- **OQ-014** — how long is a real session, and when is an inferred span implausible? A span
  over 12 hours is arithmetically unambiguous but worth a human's attention, so it warns.
- **OQ-015** — where does the receivers' timecode zero sit relative to real midnight? It
  bites only at a fractional non-drop rate, where a timecode day is 86 486.4 s rather than
  86 400. The canonical fixture is at 30F, where the question does not arise; a session
  mixing BWF and timecode evidence at 23.98F or 29.97F is warned.
- **OQ-016** — is a session always the shortest arc through its chunk start times? Raised by
  the code review, which was right that ADR-0009 overstated its case. Starts at 23:00 and
  01:00 admit a two-hour session across midnight *or* a twenty-two-hour session within one
  day; the evidence does not distinguish them and the code picks the first.

**Partially answered.** **OQ-013** — the preflight now sizes a run from the session's
*actual* duration and the artifacts actually requested, rather than assuming four hours.
But two of the three terms in `doctor`'s original 40 GiB estimate turn out not to exist:
the 48 kHz working audio is a segment map rather than 15 GiB of materialized float32, and
the mix intermediate belongs to M5. The full-pipeline number the question asks for still
needs a full live session, so it stays **open** for M11. The charter originally claimed to settle
it; that was corrected during the start phase after review.

**Depended on and still open.** OQ-004 (is `time_reference` present, midnight-relative, at
the file rate?) and OQ-011 (does FFprobe expose an exact PCM sample count?) are load-bearing
for everything here and remain unsettled against real hardware. OQ-006 gains its measuring
instrument: `sync_qa` produces the differential-lag number later capture QA needs.

### Notes for future implementors

**Read `../reviews/M2-code-20260802-1508.md` before trusting anything in this milestone.**
The gate was green, 900-odd tests passed, and the milestone looked finished at the point
where all five bugs were still present. Two of them were silent by construction.

**INV-01 was vacuous for part of every session, and had been since M1.** `raw_guard.snapshot`
excluded any path whose *any* component was named `work` or `output`, so
`raw/tx-a/work/notes.txt` was never hashed and could be mutated without verification
noticing. The intent was to skip the session's own generated directories; the
implementation skipped them anywhere in the tree. This is inherited code — moved verbatim
out of `inspection/runner.py` during M2 — so **M1 shipped with the same hole**, and it is
precisely the shape M1's own closeout warns about: *a check that is present, looks right,
and verifies nothing*. Exclusion is now anchored at the session root.

**A cache can be poisoned by a run that correctly failed.** Derivative sidecars were
committed at publish time rather than after INV-01 was re-verified. A run that read a
source, built a derivative from those bytes, and *then* discovered the file had changed
would fail loudly — and leave behind an entry keyed on the bytes it read. Restore the
original file and the key matches again, forever. Both caches now stage and commit only
after verification. That discipline was M1's and I diverged from it without noticing;
if you add a third cache, the ordering is: write the audio, rename it into place, verify
INV-01, *then* commit the sidecar.

**A wrong timeline can be indistinguishable from a right one.** With absolute evidence at
19:00 and a signed recovery offset reaching a second below it, the whole timeline shifts —
and `SessionZero.since_day_origin_samples` still recorded 19:00, so sample 0 was really
18:59:59 and every mapping back to wall clock was wrong by exactly the shift. Nothing
downstream could have caught it: a uniformly one-second-late transcript looks like a
correct transcript. If you record an origin, assert the only consistent reading of it —
zero plus a source's placement equals that source's own time of day.

_M8 (2026-08-03): the field is now `SessionZero.since_domain_origin_samples` and "time of
day" is "position in its domain", because OQ-004 showed a BWF reference does not count from
midnight (ADR-0031). The lesson above is unchanged and the check it argues for is why the
value is still recorded rather than nulled._

Things that surprised me, where the code was right and I was wrong:

- **`sample16 = sample48 // 3` is false.** A filtered impulse peaks at the *nearest* output
  sample. The exact statement is the other direction: output `k` sits at input `3k`.
  Converting an *interval* therefore floors the start and **ceils** the end
  (`resample.to_derivative_interval`). Rounding both ends the same way shrinks a speech
  region by up to two samples, which is how a word loses its first phoneme. M3 is the first
  real consumer of this; use the helpers, do not re-derive it.
- **A 29.97 non-drop timecode day is 86 486.4 real seconds**, 86.4 s longer than a calendar
  day. "Add a day" to a wrapped chunk is wrong by that much. Hence `timecode.frames_per_day`
  and unwrapping in frames *before* converting to seconds. My own hand-computed expectation
  said 84 s; the code said 86.4 and the code was right.
- **`00:01:00:02` is drop-frame's 1800th frame**, and drop-frame converges on real time over
  ten minutes, not one. The test now asserts the guarantee drop-frame actually makes.
- **The refusal paths exit 4, not 1.** A 44.1 kHz source fails *after* inspection genuinely
  completed and produced the manifest that explains the refusal — that is `partial` by
  ADR-0005's own definition. INV-13 only requires that partial never exits zero.

**The serialized contract lives in `artifacts/timeline.py`'s module docstring, not in this
document.** M3 and M5 both index into `timeline.json`, so the four rules that matter are
stated where a reader of the code will find them: every interval is half-open, there are no
floats anywhere in the document, a track's map tiles its own extent with explicit `silence`
segments (enforced by a validator, because a hole has two readings), and **both**
`evidence_start_sample` and `session_start_sample` are kept so an operator debugging a bad
sync can see that the evidence disagreed. `TIMELINE_SCHEMA_VERSION` was provisional during
M2 and is now **frozen at 1**: additive optional fields only, anything else bumps it
(ADR-0005).

Sharp edges and things that look wrong but are deliberate:

- **`ingest` always re-runs discovery and hashing**, and never trusts a manifest whose
  `config_hash` matches. A replaced or deleted WAV keeps every hash internally consistent,
  and the INV-01 snapshot only covers mutations *during* a run — so a hash match is not
  evidence that the manifest describes what is on disk. M1's content cache makes the warm
  re-inspection cost no FFprobe calls. This also deleted the planned report-merge machinery:
  re-deriving inspect provenance is strictly stronger than carrying it forward, because it
  cannot be stale.
- **Silence covers three cases and they are deliberately indistinguishable** to a
  `TrackReader` caller: before the track started, inside a real gap, and after it stopped.
  All three mean "this transmitter recorded nothing here", which is the only thing a
  consumer can act on — and it is what lets a track that ended early still answer to the
  session's aligned duration.
- **The decimator is never reset at a chunk or gap boundary.** A reset would put a transient
  at every boundary and make the derivative depend on how DJI happened to split the
  recording. Filter state and decimation phase are carried across windows; streamed output
  is byte-identical to one-shot at every partitioning tested, including sizes that are not
  multiples of three.
- **The FIR is data, not a design run at import time.** `data/fir_48k_16k.json` holds the
  coefficients so a SciPy upgrade cannot silently change what every cached derivative was
  built with, and `scripts/design_fir.py` regenerates it from the declared spec. Length 259,
  Kaiser β 9.0, cutoff 7450 Hz: 0.02 dB ripple at 7 kHz against a 0.1 dB budget, 90.4 dB at
  8 kHz against 80. Group delay is 129 at 48 kHz, which divides by three, so it is exactly
  43 at 16 kHz and the two grids align by a slice rather than an interpolation. **Do not
  hand-edit the coefficients**; the frequency-response tests are the contract, not the array.
- **`to_derivative_interval` and `to_source_sample` are called only from tests today.** That
  is not dead code — they are the 48↔16 kHz mapping contract this charter required M2 to
  *define* for M3 and M4. Rejected as a finding for that reason.
- **`nudge_later` consumes the gap after the nudged chunk**, and that is the intended trade.
  See ADR-0010.
- **`write_atomic` is unreachable from the audio path** and must stay that way — it holds
  the whole payload in memory, which is right for JSON and an INV-07 violation for a
  waveform.

Known, deliberately unfixed:

- **`TrackReader` holds one file descriptor per audio segment.** Bounded in practice — DJI
  splits a four-hour session into a handful of chunks — but a session with thousands of
  chunks would exhaust the limit. A pool would add complexity for a case that does not exist
  yet; fix it if a real session ever gets close.

Two smaller traps worth an hour each:

- The streamed writer leaked its temp file whenever a stream came up short, because cleanup
  was keyed on `exc_type` and the short-write error is raised *inside* `__exit__` itself.
- `syncqa._derivative` called `PcmReader.__enter__` itself and returned the reader, and
  every caller then used it in a `with` — entering it a second time, which unconditionally
  reopens the file and drops the first descriptor unclosed. No audio impact, and no test
  would have caught it.

Finally, on fixtures: **the drift fixture is the one thing here that is easy to build
wrong.** Both its tracks carry identical metadata — same chunk start, same `bext` reference
— and differ only in where the end transient sits *in the audio samples*. Moving the
metadata instead would let a drift test pass while the correlator was never exercised at
all. The same principle applies to `derivative_identity_document`, which is separate from
its hash so a test can assert *which* components are present: a key that changes for the
right reason can still be missing a component, and the missing one is always the one that
matters later.

### Deviations from this charter, and why

- **`--materialize-48k` writes a disposable cache artifact, not pipeline truth.** The
  charter's "lossless 48 kHz float working path" is the segment map. Accepted as an
  amendment during the start phase (ADR-0011); every duration and sample-position test
  reads through the virtual reader.
- **The planned report-merge machinery was not built.** Making `ingest` re-run inspection
  unconditionally made it unnecessary, and the replacement is stronger. `StageReport.origin`
  still landed, and is `null` rather than `"executed"` on skipped and failed stages, where
  the value was noise on four stages out of six.
- **The charter's proof table named `tests/test_ingest_report.py`, which never existed.**
  The tests are in `test_ingest_run.py`. Corrected in place during verify rather than left
  to overstate the evidence.
- **OQ-013 is partially, not fully, answered** — the charter's original claim to settle it
  was corrected during the start phase.
- **The INV-01 helpers moved out of a closed milestone's code** into `raw_guard.py`, with
  the protected-output set becoming a parameter. Flagged at plan time as touching M1;
  it was the right call twice over, since the extraction is what exposed the M1-era hole.
- **`scipy` became a runtime dependency** (1.18.0). Verified at plan time to install from
  PyPI into the flake's `.venv` with no flake change.
- **`ReportBuilder.recorded()` was added mid-milestone.** Updating the CLI tests found a
  real INV-13 bug: a failure *before* inspection left that stage unrecorded, `build()`
  refused a report with a gap in it, and the run produced no report at all — exactly what
  INV-13 exists to prevent.

### Downstream charters updated

- **M3** gains a "What M2 already provides" section: where the 16 kHz derivatives live and
  how to ask for them, the interval-mapping contract and its floor/ceil asymmetry, the
  three-way meaning of silence, the normalized-cross-correlation primitive `syncqa` already
  implements, and the cache-identity components an activity cache must carry.
- **M5** gains the same: the segment map is the working path and `--materialize-48k` is not
  it, `wavwrite` is the streamed RF64 writer for the mix intermediate, the preflight needs
  M5's own term added, and `test_memory.py`'s ordered event log is the technique for proving
  INV-07 over a composed path.
- The then-planned short capture added **OQ-015** to the list of questions its recording might settle — the displayed
  timecode after the LTC jam, recorded against wall-clock time.
- The then-planned long capture added **OQ-014** and **OQ-016**, and recorded that the measuring instrument now
  exists: `session.sync_qa` produces the differential lag its gate asks for.
- **INVARIANTS.md** — INV-01 gains the session-root scoping rule, INV-07 gains M2's
  composed-path proof technique, INV-08 gains "commit the cache entry only after
  verification".
- **ROADMAP.md** — no dependency change. M2's gate wording is unchanged and still accurate.

### Next smallest step

Begin M3 — activity. Start with the `ActivityDetector` protocol and the deterministic fake
over the canonical fixture's declared truth, before any Silero pinning: the fixture already
carries the fake-VAD contract, the 16 kHz derivatives it consumes are cached and byte-stable,
and getting the graph's shape right is the part M4 and M5 both inherit. OQ-010 (offline
Silero loading) is the only real unknown and it is orthogonal to the graph.
