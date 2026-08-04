# ADR-0031 — A BWF reference counts from the recorder's origin, not from midnight

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M8
**Amends:** the spec, [ADR-0006](0006-timing-evidence-is-a-tagged-union.md),
[ADR-0008](0008-rasterizing-time-onto-the-sample-grid.md),
[ADR-0009](0009-session-zero-and-the-24-hour-wrap.md)

## Context

Every piece of this project's timing model was built on EBU Tech 3285's definition, which the
spec repeats: a BWF `time_reference` is a sample count **since midnight** at the file's own
rate. `rasterize.py` converts it that way, `starttime.py` stamps the claim into every manifest
as provenance, and `timeline.json` records session zero's position as
`since_day_origin_samples` in a domain called `real_time`.

**OQ-004 disproved it on this hardware, on both halves.** A 19:26:55 file would carry
3 360 720 000 samples and carries 18 628 800 — a factor of 180 out — and the pair created 44
seconds *later* carries the *smaller* value. Every value is an exact multiple of 1600 samples,
one frame at the 30/1 rate the iXML declares. DJI's own documentation describes timecode as "a
frame counter relative to recording duration" that "resets to zero and restarts", with no
facility for a user to set it.

So the pipeline asserts something untrue about every real file it will ever read. The reframe
that followed (OQ-004, 2026-08-03) shrank the *consequence* considerably — placing tracks is
`session_position`, a subtraction, and a common epoch cancels out of it — but a subtraction
being unaffected is not a reason to keep a false claim in an artifact.

**OQ-023 supplied the missing half.** A jam the operator performs on the receivers propagates
into `bext.time_reference`: two receivers started 5.28 s apart wrote references agreeing on
that offset to 17–30 ms, inside one frame. So the reference *is* the receiver's timecode count.
The two absolute domains this project already models — a BWF sample reference and a timecode
tag — are the same clock in different units.

## Decision

### The day origin is the recorder's `00:00:00:00`, not real midnight

A `bwf_sample_reference` is an unsigned sample count from **the recorder's own timecode
origin**, at the file's own rate. Where that origin sits in the day is unknown, is
**OQ-015**, and cancels out of every placement.

Three things follow and one deliberately does not.

**The arithmetic survives; the labels do not.** `absolute_seconds`, the single quantizer, and
the subtract-before-rounding rule of ADR-0008 are unchanged — they never depended on what the
origin *meant*. The docstrings, the assumption strings stamped into every manifest, and the
constant's comment are corrected in place.

**Mixing a BWF reference with a timecode tag stays permitted.** Under the old reading the two
were different clocks related by an assumption; under this one they are the same clock, which
is *weaker* grounds for refusing to relate them, not stronger. `has_mixed_absolute_domains`
keeps its fractional-rate condition — at 23.98F and 29.97F a timecode day is 86.4 seconds from
a calendar day, which is a real arithmetic discrepancy — and its warning stops saying "count
from real midnight". A configured `origin_timecode` is likewise a statement in the recorder's
counter domain, which is how an operator reading a receiver display would use it.

**The artifact stops claiming a day origin it does not have.** `ZeroDomain`'s `real_time`
becomes `recorder_epoch`, and `since_day_origin_samples` becomes
`since_domain_origin_samples` with `domain` saying which origin that is. Renaming a field and
re-valuing an enum is not the additive change ADR-0005 permits, so **`TIMELINE_SCHEMA_VERSION`
goes to 2**, `schemas/timeline.schema.json` is regenerated, and `TIMELINE_SEMANTICS_VERSION`
bumps. The *value* is kept rather than nulled, because M2's hard-won consistency check depends
on it — session zero plus a source's placement equals that source's own position in its
domain, and a wrong timeline is otherwise indistinguishable from a right one.

### The reference is frame-quantized, and the overlap tolerance has to know

`rasterize.is_frame_quantized` recognized only a `TimecodeRecord`, so two chunks placed from
`bext` references got a **one-sample** overlap tolerance — the only error available from one
rounding. Against a reference quantized to 1600 samples that is wrong by three orders of
magnitude, and the consequence is not cosmetic: an ordinary later chunk whose reference rounds
backward is then a *material* overlap, and `timecode.chunk_overlap_policy` defaults to
`reject`, so **the session fails**.

The quantum is not derivable from the file. FFprobe does not surface the iXML that declares the
rate, and **OQ-024** proved the receiver's configured rate does not reach an `orig` file at all
— a receiver set to 60 fps wrote `TIMECODE_RATE 30/1` on 1600-sample boundaries beside a 30 fps
unit. So it is configuration with a measured default: `timecode.bwf_reference_quantum_samples`,
1600 at 48 kHz, citing OQ-004 and OQ-024.

That same number is the floor for `sync_qa`'s constant-offset threshold. Deriving that floor
from the configured frame rate — which the charter originally asked for — would give a `60F`
session a 17 ms threshold against source timing that still has a 33.3 ms quantum, reinstating
the false alarm the change exists to remove.

### Wall clock never anchors placement

`bext.origination_date`/`origination_time` — FFprobe's `date` and `creation_time` — carry the
receiver's real-time clock. Measured 2026-08-03: **two receivers were 48.7 s apart** while
their timecode agreed to under one frame. The timecode was right and the wall clock was wrong.

`timeline/origin.py::_cycles_from_dates` is where that reaches placement, and it does so in the
coarsest available unit: **whole 24-hour cycles**. Two receivers whose clocks straddle midnight
would be placed a day apart on evidence known to be a minute wrong.

So: **a date read from a file is descriptive only and never assigns a cycle.** Only an operator
assertion may — `timecode.origin_date`, or a `recovery.source_time_overrides` entry's
`recording_date`. No schema change is needed to tell them apart: `ManifestStartTime.strategy`
already records which strategy produced the evidence, and `recovery_*` is exactly the operator
half. A session with file dates falls through to the existing inference, which reads the
counters themselves and involves no wall clock.

Wall clock keeps its uses — archival naming, and a human reading a report. There is no blanket
prohibition on reading the tag, because M7 is a legitimate consumer; what is asserted is that
**placement and synchronization are invariant under wall-clock changes**, proved by rewriting
every source's date and time tags and getting a byte-identical `timeline.json`.

### The 24-hour wrap keeps working, and its assumption gets registered

"Device-local counter" no longer implies a 24-hour period, so the wrap arithmetic lost the
reason that used to make it obvious. It stays anyway — it is spec-required and tested
(`rollover_session`, M2's gate), and a recorder whose reference genuinely is midnight-relative
needs it. What was missing was the registration, which is now **OQ-026**, cited from
`cycle_units`. INV-12 keeps it safe meanwhile: the inference warns, refuses a tie rather than
guessing, and a real DJI session never reaches it.

## Alternatives considered

- **A configuration knob choosing between "midnight" and "recorder" semantics.** Honest, since
  the file does not say which — and exactly the kind of setting that silently ruins a session
  when it is wrong. The two readings also produce identical placement for every session whose
  zero is derived, which is the recommended configuration, so the knob would buy a failure mode
  and no capability.
- **Refuse to relate a BWF reference to a timecode tag at all.** Follows from "different
  origins", and OQ-023 shows they are not different origins. It would also invalidate the
  canonical fixture, which mixes both at 30F, for no gain in correctness.
- **Remove the 24-hour wrap for BWF evidence.** M8's plan review argued for it. Rejected above:
  deleting a tested, spec-required capability on a hypothesis about one vendor is the larger
  risk, and registration answers the actual objection.
- **Keep `since_day_origin_samples` and fix only the prose.** The cheap option, and it leaves
  an artifact whose field name asserts a day origin that a DJI session does not have. A
  document that quietly disagrees with its own meaning is what this project spends its ADRs
  avoiding.
- **Null `since_domain_origin_samples` for a recorder epoch.** Loses M2's consistency check for
  a value that is perfectly meaningful — it is just measured from an origin whose place in the
  day is unknown.
- **Derive the BWF quantum by taking the GCD of the session's references.** Evidence-derived
  and self-tuning, and unreliable: two values that happen to share a larger factor would infer
  a quantum ten times too big, and the failure is silent.

## Consequences

- **`timeline.json` is schema version 2.** Additive-only ends here; a reader of version 1 has to
  be updated, and there are two in this repository (M3's activity stage and M5's mixer), both
  of which read the segment map rather than `SessionZero`.
- **Every timeline cache misses once** on the semantics bump, and every derivative with it,
  because the derivative identity carries `TIMELINE_SEMANTICS_VERSION` (ADR-0011). That is the
  invariant working, not a cost: placement semantics moved.
- **A session dated by its files now infers its days** where it previously read them. It also
  warns (`midnight_rollover_inferred`), which it did not before, and an operator who wants
  evidence rather than inference states `origin_date` and `origin_timecode` — the same escape
  ADR-0009 already documents.
- **`bwf_reference_quantum_samples` is a new number chosen from one capture.** It is a
  measurement rather than a guess — every reference in eight real files is an exact multiple of
  1600 — but it is a measurement of one firmware on one model, cited to OQ-004 and OQ-024.
- The spec is amended in the same commit: `time_reference` is not samples since midnight, and
  origins are not calendar-day based.
