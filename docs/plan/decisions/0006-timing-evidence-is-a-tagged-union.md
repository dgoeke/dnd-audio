# ADR-0006 — Timing evidence is a tagged union, and M1 does not rasterize it

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M1

## Context

M1's timecode strategy chain has to produce *something* the manifest can hold, and the
first draft of the working plan said every strategy yields "exact samples since midnight
at the file's own sample rate". Independent review
(`docs/plan/reviews/M1-plan-20260802-1046.md`) showed that this is false for one of the
four strategies, and that the error is not cosmetic.

The three kinds of evidence M1 can find do not live in the same coordinate system:

| Evidence | Unit | Rate | Origin | Signed |
| -------- | ---- | ---- | ------ | ------ |
| BWF `time_reference` | samples | the **file's own** rate | midnight | no |
| Timecode tag / override timecode | frames | the **configured** frame rate | midnight | no |
| `recovery.start_offset_samples` | samples | canonical **48 kHz** | **session zero** | **yes** |

The spec and `src/dnd_audio/config.py` both define the recovery offset as "a signed
integer `start_offset_samples` at the canonical 48 kHz rate, measured relative to session
time zero". Converting it into a since-midnight count requires already knowing where
session zero is — which is M2's output, not M1's input. Doing it anyway would produce a
wrong timestamp for every session that does not start at midnight, and would silently
discard the distinction M2 needs in order to reconcile absolute time-of-day evidence with
session-relative evidence.

There is a second problem in the same area. One frame at 30000/1001 fps is 8008/5 samples
at 48 kHz. "The expected integer sample position" that spec acceptance criterion 2 asks
for is therefore not a property of the evidence at all; it is a property of a quantization
rule that nothing has yet defined. M1's charter claimed that criterion while explicitly
deferring midnight rollover — the half of it that makes the position meaningful.

## Decision

**The manifest records timing evidence as a tagged union and never collapses it.** Each
source carries exactly one of:

- `bwf_sample_reference` — integer `samples`, the `sample_rate` they are counted at, and
  the origination date when the file states one.
- `timecode` — the parsed timecode text, its **exact integer frame index** since
  midnight, and the rational frame rate that index is counted in.
- `session_offset_samples` — a signed integer at 48 kHz, relative to session zero.

plus, in every case, the strategy that produced it, the strategies that declined, and
the assumptions each one rests on.

**M1 extracts evidence. M2 places it on a timeline.** Concretely, M1 owns filename and
container parsing, the RIFF walk, the strategy chain, and exact frame-index arithmetic.
M2 owns normalizing recording dates, inferring midnight rollover, choosing session zero,
reconciling the three evidence kinds against each other, and rasterizing the result onto
the 48 kHz grid under a documented rounding rule.

M1's `Spec sections` line is corrected accordingly: it claims acceptance criteria 1 and
10, and the *evidence* half of criterion 2. Criterion 2 as a whole is M2's, which already
claimed it.

`dnd_audio.timecode` gains `frame_index()` — the exact number of frames since midnight a
timecode denotes, drop-frame skips included. Its docstring previously said that converting
a timecode to a position "is M2's"; that is amended to distinguish arithmetic **on a
timecode** (here) from placement **on a timeline** (M2).

## Alternatives considered

- **Normalize everything to samples-since-midnight at 48 kHz in M1.** The original plan.
  Rejected: it is not computable for a session-relative offset without session zero, and
  it forces a rounding decision at the point of *reading* a file, where there is no
  policy and no way to record which way it went.
- **Normalize everything to an exact `Fraction` of seconds since midnight.** Exact, and
  it does survive fractional rates. Rejected anyway: it still cannot express a signed
  session-relative offset, and it erases the difference between "the device said 8008/5
  samples" and "a human wrote 19:00:00:00 in a field log", which is exactly the
  provenance INV-12 exists to keep visible.
- **Bring rollover and rasterization into M1** so criterion 2 is satisfied where the
  charter claimed it. Rejected: rollover inference needs chunk sequence and session span,
  which are M2's by construction, and splitting the timeline across two milestones is
  how the arithmetic ends up inconsistent.
- **Amend the spec to require "an exact rational position plus a documented integer
  quantization".** The reviewer suggested it. Rejected: unlike ADR-0003, the spec states
  no impossibility here — criterion 2 becomes checkable the moment M2 documents its
  rounding rule. The spec does not change casually, and this is an underspecified rule
  rather than a contradictory one. M2's charter carries the obligation instead.

## Consequences

- The manifest schema has a discriminated union in it. That is more work for a consumer
  than a single integer, and it is the honest shape: three kinds of evidence with three
  different origins really are three things.
- M2 must define and document one quantization rule for rational time reaching the
  48 kHz grid, and test it at 24000/1001 and 30000/1001 where it actually bites. Its
  charter now says so.
- A fixture whose timecode does not land on a whole sample is not expressible as ground
  truth in M1 — the generator refuses to write one rather than rounding silently.
- If H1 shows DJI writes some fourth kind of timing evidence, it becomes a fourth
  variant rather than a lossy conversion into one of these three (OQ-001, OQ-005).
