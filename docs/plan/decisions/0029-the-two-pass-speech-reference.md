# ADR-0029 — The speech reference comes from the candidates that won attribution

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M8
**Amends:** [ADR-0014](0014-the-conservative-bleed-gate.md)

## Context

ADR-0014's veto is the thing that stops the bleed gate deleting the quieter of two people
genuinely talking at once. It fires when a candidate's band-limited level sits within
`bleed.veto_db` of **its own track's speech reference** — "what this wearer sounds like when
this wearer is talking".

That reference was the 75th percentile of **all** of a track's candidate levels. ADR-0014's
own amendment records the tension and calls it OQ-017: including bleed candidates drags the
reference down, taking the upper quartile pushes it up, and which dominates is a property of a
real room.

The 2026-08-03 jam capture answered it, and the answer is that the estimator does not work:

| track | candidates | reference computed | what it actually is |
| --- | --- | --- | --- |
| tx-b | 3 | −40.77 dBFS | its own speech (−38.8) |
| tx-d | 4 | −57.80 dBFS | **bleed** (−57.7) |

One extra bleed candidate moved the reference 17 dB, because `np.percentile(..., 75,
method="nearest")` lands on the largest of three values and the second-largest of four. A
reference anchored at bleed sets the veto at bleed, which protects bleed from suppression —
the exact inverse of the protection it exists to provide.

**This gets worse with roster size.** With six people each speaking about a sixth of the time,
roughly 83% of any track's candidates are bleed, so the upper quartile sits in bleed territory
*for every participant*. Raising the percentile buys one or two tracks' headroom against a
problem that scales with the roster; it is not a fix.

It is also the single root cause of three symptoms: the weak veto, the mix's contaminated
level corrections (`mix_level_correction_clamped` firing against a bleed level), and truncated
bleed fragments surviving into the transcript.

## Decision

### The reference is estimated from the candidates that won attribution

`bleed.attribute` runs the gate twice:

1. **Bootstrap.** Levels and pair measurements as before; a provisional reference from the
   all-candidates percentile; scores; the gate with the **veto disabled**.
2. **Estimate.** Each track's reference becomes `REFERENCE_PERCENTILE` of the levels of that
   track's candidates that were **retained** by the bootstrap pass.
3. **Authoritative.** Relative levels and scores recomputed against that reference, then the
   gate with the veto.

The single pass was circular: the veto needs a reference that attribution has not produced
yet. Two passes break the circle in the only direction available — decide who is speaking
using everything except the veto, then measure the speakers.

The bootstrap reference is stated rather than left implicit: it is today's estimator, and its
only job is to make the *scoring* comparable. It never reaches a suppression decision, because
the bootstrap pass's suppressions are discarded.

### A track with no winners falls back, and the fallback direction is the whole point

| winners | reference | reason |
| --- | --- | --- |
| ≥ `bleed.min_attributed_reference_candidates` (default **1**) | percentile of the winners' levels | a winner is direct evidence that this is the wearer speaking |
| zero, but ≥ `bleed.min_reference_candidates` candidates overall | the all-candidates percentile | the overlap-only speaker, below |
| zero, and fewer than that | `None` — the veto cannot be evaluated | a track that only ever *hears* is what the gate exists to suppress |

**The middle row is not a compromise; it is the correction to a regression this ADR's first
draft contained.** A quieter person who speaks *only* during overlap has no uncontested
candidates, so a winners-only rule gives them no reference, disables their veto, and deletes
them — which is precisely the failure ADR-0014 was written against, reintroduced by the fix
for a different one. Today's contaminated reference happens to save that person, and losing
that would trade one silent deletion for another. Found by M8's plan review
(`../reviews/M8-plan-20260803-1729.md`, finding 1); `mutual_bleed_session` cannot show it,
because that fixture gives its quiet speaker three solo utterances.

### Two floors, because the two populations are not equally good evidence

`min_reference_candidates` (3) guards an estimate made from an **unclassified mixture** — one
or two regions that may be speech or may be bleed. `min_attributed_reference_candidates` (1)
guards an estimate made from candidates the gate has already concluded are this wearer
speaking. One winner is stronger evidence than three of the mixture, so reusing the same
number for both would be a floor set for the wrong population.

ADR-0014's amendment warns against "a second knob that selects the same candidates twice".
This is not that: the two select different populations, and both cite **OQ-017**.

## Alternatives considered

- **Raise `REFERENCE_PERCENTILE`.** The obvious fix, and it fails on arithmetic: at six
  speakers the bleed fraction is ~83%, so no fixed percentile below that is safe, and one
  above it is estimated from one or two candidates.
- **Filter by VAD confidence before taking the percentile.** ADR-0014 already rejected a
  confidence filter as a knob that selects the same candidates twice — and it would not help,
  because a detector fires confidently on loud bleed.
- **A winners-only reference with no fallback.** The first draft. Deletes the overlap-only
  speaker, above.
- **No suppression at all for a track with too few winners**, marking every candidate
  ambiguous. The plan review's proposal. Rejected: it makes a track that only ever hears bleed
  unsuppressible, which is the gate's entire purpose, and it breaks
  `delayed_bleed_session`, whose silent listener is deliberately reference-less so that the
  *correlation* half of the rule is tested without the veto in the way.
- **Estimate the reference from the audio directly** — a percentile over the whole track
  rather than over candidates. Measures the room's noise floor on a track whose wearer is
  quiet, and needs a session-length read the candidate path already avoids (INV-07).
- **Iterate to a fixed point** rather than stopping at two passes. Not obviously convergent,
  and a rule whose output depends on how many times it ran is a rule nobody can reason about
  at three in the morning.

## Consequences

- **`ACTIVITY_SEMANTICS_VERSION` bumps**, so every attribution cache misses once (INV-08).
  Correct: the graph's decisions genuinely changed.
- The gate now measures every candidate's level twice as often in the worst case — the pair
  measurements and levels are computed once and reused, so the second pass costs scoring
  arithmetic and no additional audio reads. INV-07 is unaffected.
- **`speech_reference_mbfs` changes meaning slightly**, and M5 reads it as its per-track
  voice-level correction. That is the *point* — the mix was being levelled against bleed — but
  it means M5's `mix_level_correction_clamped` warning needs re-checking against the new
  numbers rather than assuming it still fires for the same reasons.
- The reference is no longer reconstructible from the artifact by re-running a percentile over
  the candidate list, because the population is now a subset. `ActivityTrack` therefore records
  `reference_candidate_count` alongside `candidate_count`, so the estimate is auditable from
  `work/activity.json` without measuring audio — which is how this defect was found, and took
  a day.
- Both floors remain guesses about a real room, cited to **OQ-017**, and a real session is what
  moves them.
