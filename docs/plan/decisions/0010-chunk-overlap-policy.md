# ADR-0010 — What happens when two chunks of one track overlap

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M2

## Context

The spec is precise about the shape of the rule and silent about its vocabulary:

> Detect overlaps. Resolve only tiny overlaps explainable by timestamp/frame quantization;
> otherwise retain a warning and require an explicit policy rather than silently
> discarding audio.

"Require an explicit policy" names a configuration field that does not exist, and does not
say what its values are. ADR-0005 already collected the other vocabularies the spec
implied without naming; this is the last one, and it arrives now because M2 is where an
overlap can first be detected.

The neighbouring question is what "resolve" means. Two chunks whose declared start times
overlap by 300 samples can be reconciled by trimming 300 samples off the head of the later
one, or by moving the later one 300 samples later. The first discards audio. The spec's
sentence ends with "rather than silently discarding audio", which settles it for the
material case; applying the same reasoning to the tiny case keeps one rule instead of two.

## Decision

**A sub-tolerance overlap is resolved by moving the later chunk, never by trimming it.**
The later chunk is placed immediately after the earlier one's end, and the shift is
recorded on the segment as `shift_samples` alongside the position its evidence actually
rasterized to. Tolerance is ADR-0008's: one sample for sample-exact evidence, one frame at
the configured rate when either start came from a timecode.

**A material overlap — larger than the tolerance — is governed by
`timecode.chunk_overlap_policy`:**

- **`reject`** (the default) is fatal, with a diagnostic naming both chunks, the overlap in
  samples, the tolerance it exceeded, and the evidence each start came from.
- **`nudge_later`** applies the same shift as the tiny case, at any size, and records a
  warning as well as the decision.

Both values preserve every sample of audio. There is deliberately no policy that trims,
crops, or drops a chunk.

Real gaps are the other side of the same comparison and need no policy: a later chunk that
starts after the earlier one ends is placed where its evidence says, and the hole becomes
an explicit silence segment in the map. A transmitter switched off and back on cannot pull
later audio earlier, because nothing in the placement path is relative to the previous
chunk.

## Alternatives considered

- **Trim the later chunk's head by the overlap.** The tidier-looking option, and the one
  the spec's final clause rules out. It also destroys evidence: the discarded samples are
  the only record that the two chunks disagreed.
- **Trust the later chunk and truncate the earlier one's tail.** Same objection, and worse
  — DJI splits a file at the *start* of the new chunk, so the earlier chunk's tail is the
  part more likely to be real.
- **One policy value only, with tiny overlaps also fatal.** Rejected: at 29.97 fps a
  timecode start is quantized to 1602 samples, so two perfectly ordinary contiguous chunks
  routinely overlap by less than a frame. Failing on that would make the fractional rates
  unusable.
- **A numeric `overlap_tolerance_samples` in the configuration.** Rejected: the tolerance
  is derivable from the evidence that produced each start, and an operator-supplied number
  would let a large real overlap be waved through by editing one line — which is precisely
  the "silently discarding audio" the spec forbids, one level of indirection away.
- **`keep_overlap`, placing both chunks where their evidence says and letting them
  overlap.** Considered seriously. Rejected because the working path is one contiguous
  virtual track per person: two chunks occupying the same samples has no representation
  there, and inventing one (summing? last-writer-wins?) is a mixing decision inside a
  timeline milestone.

## Consequences

- The default is the safe one, and an operator who hits it gets a message with the two
  filenames and a number rather than a silently repaired timeline.
- `nudge_later` shifts everything after the overlap by the overlap amount, so a session
  recovered that way has a known, recorded, monotonically accumulating error against its
  own timecode. The decision record makes that auditable; using it on a session with real
  timecode is a choice, not an accident.
- The segment map carries both the rasterized start and the placed start, so a consumer
  can always see what the evidence said before the layout adjusted it.
- If H1 shows DJI chunk boundaries routinely overlap by a fixed amount, the fixed amount
  becomes evidence for a different default — and it will be visible in the decisions of
  every session inspected before then.
