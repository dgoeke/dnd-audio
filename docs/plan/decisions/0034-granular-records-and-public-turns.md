# ADR-0034 — Granular records and coalesced public turns

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M9
**Amends:** the spec, [ADR-0017](0017-the-asr-grid-and-request-ownership.md),
[ADR-0019](0019-the-transcript-records-artifact.md)

## Context

With transcript-only grace and conservative fragment collapse, the final announced phrase is
complete but remains split across two adjacent activity candidates on the same track. Joining
the records would erase the exact candidate and request audit trail ADR-0017 deliberately
preserves. Leaving the public transcript as two lines makes one turn read as two.

Shared request lineage alone is not enough. Requests merge gaps up to 1.5 seconds as a batching
optimization, not a conversational classifier. Nor can granular `overlap` flags simply be
copied: a coalesced span can overlap another speaker beyond the threshold even when each piece
did not.

## Decision

`work/transcript-records.json` remains granular and is still the sole render input. Rendering
builds deterministic **presentation turns** from retained records. Adjacent records join only
when all of these hold:

- same speaker and track;
- same alignment status;
- neither granular record is marked overlap;
- at least one shared request id;
- no retained record by another speaker intervenes between them in canonical order;
- the exact-sample gap is no greater than `transcript.presentation_join_gap_ms`, default
  **350 ms** (OQ-018). The measured target gap is 320 ms.

The public turn keeps the first record's `segment_id`, concatenates normalized text and words,
and carries additive plural `source_segment_ids` and `source_candidate_ids` in provenance while
retaining the singular first-candidate field for schema-1 compatibility. JSON and Markdown are
driven by the same presentation-turn iterator.

After grouping, public `overlap` is recomputed from exact sample intervals against every other
public turn by a different speaker. The stored granular flags remain truthful about the
records; the public flags remain truthful about the public spans.

This is presentation coalescing, not ownership merging. ADR-0017 remains true in the records,
and request batching is necessary but never sufficient evidence of one turn.

## Alternatives considered

- **Join the records.** Rejected: it destroys candidate/piece auditability and makes collapse
  evidence refer to invented aggregate segments.
- **Use request lineage alone.** Rejected after plan review: it can merge distinct statements
  separated by the 1.5-second batching gap.
- **Copy the first or OR the granular overlap flags.** Rejected: neither computes overlap of
  the public interval described.
- **Render JSON granular and Markdown joined.** Rejected: two authoritative views would then
  disagree on transcript semantics without a useful benefit.

## Consequences

- `render` still needs no graph, timeline, model or mixer; the records contain every input to
  the presentation rule.
- Public segment ids can have plural lineage. Consumers that understand only schema version 1
  still see the first canonical id/candidate; new consumers can audit every contributing piece.
- The provisional 350 ms boundary needs multi-speaker/table evidence. A shared request never
  grants permission to join across a longer pause.
