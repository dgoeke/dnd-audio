# ADR-0033 — Transcript-only ownership grace and contained fragments

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M9
**Amends:** the spec, [ADR-0017](0017-the-asr-grid-and-request-ownership.md),
[ADR-0020](0020-word-ownership-and-bounded-retry.md),
[ADR-0032](0032-duplicate-chain-resolution.md)

## Context

The four-file local evaluation exposed two different transcript failures under the production
activity graph. First, Qwen's aligner placed each direct utterance's opening word 20 ms before
the activity candidate that owned the utterance. ADR-0020 then dropped the word correctly under
its strict start-based rule. Increasing activity padding recovered it only by changing the
graph, speech references and mix. Second, the existing similarity collapse retained suffix and
long-fragment bleed because one transcription was a proper word-sequence subset of the better
source rather than a fuzzy match to the whole text.

Neither observation permits a general aggressive deduplicator. In the same corpus two people
appear to say `Okay` at nearly the same time. The copies correlate at 648/1000 but differ by
only 39/1000 in source score. With no genuine simultaneous-speaker example, deleting either
would convert uncertainty into missing speech.

## Decision

### Ownership grace is a post-ASR partition, not activity padding

`transcript.leading_ownership_grace_ms` defaults to **20 ms** (OQ-027). After every ASR
response is fixed, assembly extends each ownership piece's leading edge by at most that amount.
The extension is clipped by:

- session start;
- the start of audio actually submitted for that request;
- the preceding ownership piece's half-open end on the same track.

Ends never move. Original activity ownership in `RequestOutcome.plan.ownership` never moves.
The transform is computed globally per track in deterministic interval order, so no sample can
belong to two effective pieces. Adjacent pieces therefore receive no grace at their shared
boundary; a real gap can donate only its unowned leading portion to the later piece. Long
candidate divisions and truncation/retry results obey the same partition.

A resolved truncation contributes its final leaf submissions, each with its own sliced
ownership, padded bounds and returned words. The discarded truncated parent remains in the
attempted request-id lineage but is not an ownership occurrence. Assembly compares a returned
word only with the pieces belonging to the submission that returned it; one retry child's
padding can never donate a word to another child's candidate. This distinction was added after
M9 code review found that parent-shaped lineage both hid the retry seam and could misattribute
speech when grace exceeded child padding.

Records retain piece-specific original and effective intervals plus their request and submitted
padded bounds. Aggregate activity bounds remain for compatibility, but are not the validator's
evidence for a new aligned record. Every aligned word start must resolve to exactly one effective
piece. The fields are optional additive fields so schema version 1 records remain readable;
newly assembled M9 records always populate them.

This changes `TRANSCRIPT_ASSEMBLY_SEMANTICS_VERSION`, not
`TRANSCRIPT_SEMANTICS_VERSION`. Grace is deliberately absent from request planning, submitted
audio and the ASR identity document. A different grace value reassembles cached answers instead
of spending GPU inference again.

### Contained fragments are a second, more demanding collapse rule

The existing three-condition similarity algorithm runs to completion unchanged. Only its
survivors enter a second global pass named `contained_fragment`, preventing a new edge from
changing which legacy comparison happens first.

A first-pass survivor may already have absorbed a similarity duplicate. If containment then
collapses that survivor, records keep the completed first-pass edge as a **terminating audit
chain**: the similarity loser still names its original winner, and that intermediate winner
names the final contained-fragment survivor. This is the sole exception to ADR-0032's ban on
duplicate chains. It is permitted only when every intermediate node was itself collapsed by
`contained_fragment`, every link is acyclic, and the chain ends at a retained segment. The
intermediate keeps its rejected alternatives, so the first-pass judgment is not rewritten as
though the final survivor made it directly.

The second pass collapses a weaker segment only when all of these hold:

1. the segments substantially overlap under the existing overlap ratio;
2. graph correlation evidence exists for the full Cartesian product of contributing candidate
   pairings — merely mentioning every candidate in diagonal pairs is insufficient;
3. the winner leads by `duplicate.contained_min_score_margin`, default **300/1000** (OQ-018);
4. the winner's normalized words properly contain the loser's as one contiguous sequence;
5. the winner is the acoustically preferred survivor.

"Properly" means strictly more normalized words. Equal `Yes`, `Okay`, or any other exact
utterance cannot enter this path. The existing similarity rule and its serialized decisions
remain unchanged; only new contained-fragment rejections carry the distinct rule name.

## Alternatives considered

- **Raise `activity.vad.pad_ms` to 80 ms.** Rejected: it changed speech references by as much
  as 1.08 dB and produced more public segments. Transcript recovery must not reach the mix.
- **Raise `activity.vad.merge_gap_ms` to 300 ms.** Rejected: one speech reference moved from
  about -40.77 to -59.86 dBFS and changed the automix clamp.
- **Mutate request ownership before ASR.** Rejected: request identity and cache behaviour would
  then depend on an assembly-only setting, and the audit trail would call grace activity.
- **Run containment as a per-pair fallback.** Rejected after M9 plan review: its globally
  higher-scored edge could preempt an existing similarity decision.
- **Collapse exact short text under correlation.** Rejected: the current corpus cannot tell
  two speakers agreeing from bleed, and the spec says losing real overlap is worse.

## Consequences

- All intended openings in the measured corpus are eligible at the smallest useful value,
  while activity, request audio, ASR cache identity and mix remain byte-identical.
- A word can begin before its activity candidate and still be honestly retained because both
  intervals are present in the records.
- Contained suffix/long fragments may disappear only under a stronger source-dominance floor
  than ordinary similarity collapse. Exact short ambiguity remains visible for M11.
- Assembly semantics bump; ASR semantics and the ASR cache do not.
