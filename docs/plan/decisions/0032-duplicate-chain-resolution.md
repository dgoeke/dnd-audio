# ADR-0032 — Duplicate chains resolve by source score, and transcript semantics split from ASR

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M8
**Amends:** [ADR-0019](0019-the-transcript-records-artifact.md)

## Context

M4's closeout deferred a defect and set the condition for dropping it:

> With three mutually-duplicate segments scoring A=800, B=700, C=900 in canonical order, A
> absorbs B first and is then forbidden from being absorbed by C — `collapse.py` refuses to let
> a segment that has already absorbed another be absorbed itself, because a chain of duplicates
> has no surviving text at the end of it. A and C both reach the transcript, and `collapse.py`'s
> own docstring says the survivor is the one with the best source score.
>
> **M6b should revisit it** once real ASR output shows whether three lavs ever agree closely
> enough for the shape to occur at all. If they do not, the right change is to delete the
> docstring's claim rather than to write the pass.

M6b could not settle it — one operator testing microphones one at a time is not evidence about
three lavs. The 2026-08-03 jam capture is: **three tracks within 32 ms carrying identical text,
and a four-way group that collapsed cleanly with the best source score winning.** They agree
easily, so the condition for deleting the claim is not met.

The pathological ordering itself was still not *observed* — those three never entered collapse,
being below `min_text_words` — so this remains what M4 called it: a tidiness fix whose failure
mode is already the safe one. It keeps both segments and marks them overlapping, which is the
bias M4 states outright and which the gate criterion permits in as many words.

## Decision

### Pairs resolve in order of descending winner score

`collapse` sorts its candidate pairs by `(-winner_score, winner_segment_id, loser_segment_id)`
before the greedy walk, instead of taking them in canonical index order. The best-scoring
segment therefore absorbs first, and the shape where a worse survivor blocks a better one
cannot arise: with A=800, B=700, C=900, the pair `(C, A)` is resolved before `(A, B)`, C absorbs
A, and then C absorbs B.

This is a deterministic sort, not new logic. `_is_duplicate` still gates everything — all three
of substantial temporal overlap, strongly similar normalized text, and supporting acoustic
evidence — so the only thing that changes is *which* of two mutual duplicates survives, and the
segment now removed is one those three conditions had already called a duplicate. The tie-break
is on segment id, which is a function of time and track and therefore of the input rather than
of iteration order (INV-02).

**The alternative was to rewrite the docstring**, and the gate permits either. It was rejected
because the docstring states the rule that is *right*: the survivor should be the copy the
model-independent evidence prefers, never whichever one sorted first. Weakening documentation
to match a greedy accident makes the next reader trust the accident.

**Noted, because it is the thing that will change the exposure:** fixing truncation, or moving
`duplicate.min_text_words`, would push groups like the observed one into collapse for the first
time. The safe-direction failure has been cheap so far partly because it has been rare.

### Transcript semantics split from ASR semantics

Changing collapse changes what a `transcript-records.json` *means*. Two documents both stamped
`transcript_semantics_version: 1` could now carry different duplicate survivors for the same
audio, and no consumer could tell them apart — an incomplete semantic identity, and an INV-08
risk. Raised by M8's plan review (`../reviews/M8-plan-20260803-1729.md`, finding 6).

There were two ways to fix it, and the cheap-now one is not the cheap-later one:

- **Bump `TRANSCRIPT_SEMANTICS_VERSION`.** One line, one ASR-cache miss. But the version is in
  the ASR cache key, so *every* future change to assembly, collapse or rendering re-runs GPU
  inference over the whole session.
- **Split it.** Request shaping and submission keep `TRANSCRIPT_SEMANTICS_VERSION` and stay in
  the ASR cache key. Assembly, collapse and rendering get
  `TRANSCRIPT_ASSEMBLY_SEMANTICS_VERSION`, recorded in the records artifact's provenance and
  deliberately **not** in that key.

The split is taken. A collapse change then costs a re-render rather than four hours of
inference, and this is the moment it is cheap — no real session has been processed, so there is
no expensive cache to invalidate. That is the same reasoning M8 exists on.

The two versions bump independently, and both move in this milestone: assembly for the
collapse ordering, and nothing for ASR, because M8 changes nothing about what is submitted.

## Alternatives considered

- **Rewrite the docstring to describe the greedy behaviour.** Permitted by the gate, rejected
  above.
- **A separate chain-resolution pass** that finds connected components of the duplicate graph
  and keeps the best-scoring member of each. Strictly more general, and it is a structural
  change to the function M4's closeout calls "the function to be frightened of" — two of the
  four correctness defects found in M4's verify phase were in it or fed it, and both deleted
  speech. The sort achieves the same outcome for the shape that actually occurs, with a diff a
  reviewer can hold in their head.
- **Allow a segment that has absorbed another to be absorbed itself.** Produces a chain of
  duplicates with no surviving text at the end, which the records artifact refuses — correctly.
- **Leave the ordering and bump `TRANSCRIPT_SEMANTICS_VERSION` only.** Rejected above: it makes
  every future text-side change cost GPU time.

## Consequences

- **One fewer segment reaches the transcript** in the A/B/C shape, and it is the one with the
  worse source score. That is a behaviour change in the direction of deletion, in the milestone's
  most dangerous function, so it is mutation-checked: reverting the sort must fail a named test.
- **`transcript-records.json` gains a provenance field** — additive and optional, so ADR-0019's
  artifact stays at its schema version — and the ASR cache key loses a component it should never
  have carried.
- A warm ASR cache survives this milestone. `work/transcript-records.json`,
  `output/transcript.json` and `output/transcript.md` are regenerated, which is what `render`
  already does from the records alone.
- The observed four-way group collapsed correctly under the old ordering too. This change is
  insurance against a shape that real audio is now known to be capable of producing, not a fix
  for a wrong transcript anyone has seen.
