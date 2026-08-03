# ADR-0020 — Who owns a word at a boundary, and what "bounded retry" counts

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M4

## Context

Two of M4's gate criteria are one-line sentences hiding a decision each.

*"Words are assigned to core intervals and boundaries stitched deterministically so padding
cannot duplicate words or utterances."* Padding exists so the model hears the context around
an utterance and does not clip its first and last word. That means overlapping submissions,
which means the same word can come back twice.

*"Truncation handling: a faked length-stop response triggers a split at a natural low-energy
boundary in the unpadded core, retries both halves with their own padding, and stitches
deterministically. Retries are bounded."* Splitting is recursive, and "bounded" does not say
bounded by what.

The working plan asserted "assign each word to the interval containing its start" and called
the duplication problem solved. Independent review showed it is not: if the same word is
returned at sample 99 by one request and 101 by another, and the ownership boundary is 100,
a start-based rule keeps both copies. It also pointed out that a result with *no* word times —
the alignment-failure path the spec explicitly requires to survive — cannot be placed by any
rule about word positions at all.

## Decision

### Three rules for word ownership, in order

1. **A word belongs to the ownership interval containing its start.** Half-open, so a word
   starting exactly on a boundary belongs to the later interval.
2. **A word inside padding but inside no ownership interval is dropped.** Ownership intervals
   that are not truncation children are separated by real VAD silence, so this discards only a
   word the detector already missed, and the gate is explicit that padding must not become
   content. A word dropped here is *not* lost text elsewhere: the segment's text is built from
   the words it owns.
3. **At a truncation stitch boundary — the only place two ownership intervals are genuinely
   adjacent — a word from the later child is dropped when its normalized text equals the
   earlier child's last kept word and their intervals overlap.** This is where rule 1's
   failure lives, and it is the only place it can occur.

**A response with no words is `segment_only`.** Its text is kept whole against the ownership
interval, with a warning, because trimming it is impossible and dropping it would lose speech
the spec says must survive. That the kept text may include a word from the padding is a
limitation this decision accepts and records; the alternative is discarding a segment because
its word times failed, which is exactly what the spec forbids.

### `max_truncation_retries` is a global submission budget per original request

Not a recursion depth. Depth doubles: at depth *N* a binary split can cost `2^(N+1) - 1`
submissions, so a "bounded" retry configured as 3 could mean fifteen calls to a model that
takes a minute each. The budget counts *additional submissions* spent resolving one original
request, and it is checked before each child is submitted.

Three further rules make the recursion terminate and stay honest:

- **A child core shorter than `min_split_core_ms` is not split again.** Splitting forever
  produces sub-word requests whose transcription means nothing.
- **The split point is strictly interior**, chosen as the lowest-energy frame among the
  interior candidates of the core. The midpoint is always available, so "no split point
  exists" cannot arise; only the minimum-length rule stops the recursion.
- **Every child request obeys `max_segment_s`** exactly like an original.

### The fallback is atomic per original request

If any descendant is still truncated when the budget runs out, the **original** response is
kept, with an `asr_truncation_unresolved` warning naming the request. Not a mixture of
resolved children and one truncated remainder: a partially stitched result looks complete and
is missing an unknown amount of speech in the middle, which is worse than one response that
is visibly truncated and says so. This is what the charter's "the original response plus a
warning is retained when it cannot be resolved" means, made specific.

## Alternatives considered

- **Assign words by midpoint rather than start.** Shifts the boundary case without removing
  it, and makes a word's owner depend on a duration the model may not report accurately.
- **Keep every word from every request and deduplicate globally by text and time.** Rejected:
  a global fuzzy dedup can silently delete a word someone genuinely said twice, which is the
  failure mode this milestone is most exposed to.
- **Drop the segment when alignment fails.** Directly contradicts the spec and the gate.
- **Bound retries by depth.** Rejected above; the exponent is the whole problem.
- **Non-atomic fallback, keeping whatever children resolved.** Rejected above.

## Consequences

- The stitch-boundary rule depends on the model returning the same *text* for a word it heard
  twice, and returning it at times close enough to overlap. Both are assumptions about a model
  M4 does not have, registered as **OQ-018** and cited from the code that depends on them.
- A `segment_only` segment can contain a word from its padding. Recorded here rather than
  discovered later; when M6b makes alignment failures observable on real audio, this is the
  behaviour to re-examine.
- The budget makes the worst case linear and stated, which matters more once each submission
  costs GPU time rather than a dictionary lookup.
