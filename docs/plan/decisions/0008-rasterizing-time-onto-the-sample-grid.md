# ADR-0008 — Rasterizing rational time onto the 48 kHz grid

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M2

## Context

ADR-0006 left M2 an explicit obligation: spec acceptance criterion 2 asks that timing
evidence "map to the expected integer sample positions", and no rule for that mapping
exists. It cannot be derived, because it is not a property of the evidence. One frame at
30000/1001 fps is `8008/5` samples at 48 kHz — 1601.6 samples. Any integer answer is a
rounding decision, and an undocumented rounding decision is one that changes silently.

Three things needed settling together, because they are the same arithmetic seen from
different sides:

1. **Where a single piece of evidence lands.** Evidence arrives in three coordinate
   systems (ADR-0006) and has to reach one integer sample index.
2. **How many roundings happen on the way.** A position is a *difference*: the source's
   time minus session zero. Rounding the origin and then rounding the position doubles the
   worst-case error and makes the result depend on where the origin happens to sit.
3. **What counts as a "quantization-scale" overlap.** The charter's completion gate says
   only overlaps "explainable by timestamp/frame quantization" may be resolved
   automatically. Without a number, that phrase resolves to whatever the first
   implementation happened to do.

## Decision

**Every conversion is exact until the last step, and there is exactly one last step.**

Evidence is converted to an exact `Fraction` of seconds since the session's day origin:

| Evidence | Exact seconds |
| --- | --- |
| `bwf_sample_reference` | `samples / sample_rate` — the file's own rate, never the session's |
| `timecode` | `frames × 1/rate`, with `rate` the exact `Fraction` (30000/1001, not 29.97) |
| `session_offset_samples` | already session-relative; `samples / 48000` |

The session-relative position is then computed as a single exact subtraction, and **only
that difference is quantized**:

```
position_samples = to_samples(source_seconds - session_zero_seconds, 48000)
```

`determinism.to_samples` is the only quantizer, and it uses the tie rule
`determinism.to_milliseconds` already states: **halves away from zero**. There is one
rounding rule in this project and one module that owns it.

Accumulating a running position by adding successive durations is forbidden outright
(INV-04). Every chunk is placed by rasterizing its *own* evidence against session zero, so
no error can compound along a track.

**The quantization tolerance** — the largest overlap between two adjacent chunks that is
explainable by rounding rather than by the recorder actually overlapping them — is:

- **1 sample** when both chunk starts came from sample-exact evidence (a BWF reference or
  a session offset). The only error available is the single rounding above.
- **one whole frame at the configured rate**, rounded up, when either start came from a
  frame-quantized timecode. A recorder that writes `19:00:00:00` may have started anywhere
  inside that frame, so its own quantization dominates ours by three orders of magnitude
  (1602 samples at 29.97, against 1).

The tolerance is a property of the *pair*, computed from the two pieces of evidence that
produced the two starts, not a global constant.

## Alternatives considered

- **Round each absolute position, then subtract.** The obvious implementation. Rejected:
  two roundings instead of one, and the result depends on the origin's own fractional
  position, so moving `origin_timecode` by one frame could move an unrelated chunk by a
  sample. The error is small and the nondeterminism-by-configuration is not.
- **Truncate instead of rounding half away from zero.** Cheaper to reason about at a
  boundary, and biased: every position lands early by up to a full sample, which
  accumulates into a systematic offset across six tracks rather than cancelling.
- **Banker's rounding**, via Python's `round`. Rejected for the reason M0 already
  rejected it for milliseconds: `round(0.5)` and `round(1.5)` disagree about which way a
  half goes, and a rule nobody can predict is not a documented rule.
- **Keep positions rational all the way into the audio path.** Exact, and unimplementable:
  a sample index into a PCM file is an integer. The rounding has to happen somewhere, and
  the useful choice is to make it happen once, in a named function, where it can be
  tested.
- **One global overlap tolerance** — say, one frame at the configured rate, always.
  Rejected: it makes a 1602-sample overlap between two BWF-timed chunks look like
  rounding, when for sample-exact evidence it is a real overlap the operator should see.
  A tolerance should be as small as the evidence allows.

## Consequences

- The expected sample position for any evidence is computable by hand, in exact rational
  arithmetic, independently of the implementation. That is what makes criterion 2's tests
  real tests rather than change detectors: `tests/test_rasterize.py` states its
  expectations as `Fraction`s and never calls the code under test to build them.
- Two adjacent chunks whose starts came from timecode tags can overlap by up to a frame
  and still be treated as contiguous. That is correct, and it means a *genuine* overlap
  smaller than one frame is invisible at those rates. Nothing can recover it; the evidence
  does not contain it.
- If a future DJI format writes sub-frame timing in a private chunk (OQ-005), the tolerance for
  that evidence kind becomes 1 sample and the rule above does not otherwise change.
- A future affine drift warp (the interface hook this milestone adds) composes with this
  cleanly, because it operates on the exact rational time before quantization.
