# ADR-0014 — The bleed gate: a score margin, a correlation, and a veto

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M3

## Context

Every lav hears the room, so VAD alone produces the same utterance as a candidate on four or
five tracks. The spec is unusually specific about how to resolve that, and unusually
specific about the failure mode to avoid:

> Suppress a candidate as obvious bleed only when another track is convincingly stronger and
> the signals are strongly related. Default to keeping ambiguous candidates; losing real
> overlapped speech is worse than spending more ASR compute.
>
> Do not use a single global loudness comparison that always awards a time interval to the
> loudest person; that would erase a quieter speaker during real overlap. Source scores
> should combine track-relative speech level, VAD confidence, cross-track dominance, and
> correlation evidence.

M3's charter repeats the warning: *this is the milestone where a "reasonable" simplification
does the most damage.*

The first draft of the rule suppressed on a level dominance in dB plus a correlation
threshold, and computed the four-term score separately as a diagnostic. Independent review
(`../reviews/M3-plan-20260802-1600.md`) pointed out that this satisfies the gate's letter
while leaving the score unable to affect any decision — and produced the case that breaks it:
**two people genuinely speaking at once at unequal levels, each lav carrying the other's
bleed.** Dominance is satisfied, correlation is satisfied, and the quieter of two real
speakers is deleted. The canonical fixture cannot show this, because `tx-d` and `tx-e` have
no mutual bleed.

## Decision

### Similarity is lag-tolerant, normalized, and speech-band

Cross-channel similarity is the peak of the normalized cross-correlation over a bounded lag
window, default ±30 ms (`activity.correlation_max_lag_ms`), computed on band-limited audio.
Both the peak and the lag at which it occurred are recorded for every compared pair.

`timeline.syncqa.measure_lag` already is this function — normalized by both signals'
energies, so a quiet track and a loud one are comparable — and it is reused rather than
reimplemented. Zero-lag correlation would miss bleed entirely: sound crossing a table
arrives milliseconds late, and the fixture's own bleed is 3 ms behind its source.

The band limiting is a checked-in linear-phase FIR (`activity/data/fir_speechband_16k.json`),
data rather than a design run at import time, for the reason ADR-0011 gives for the
decimator: a SciPy upgrade must not silently change what a cached decision was made with.
Its identity hash is part of the attribution cache key.

### Suppression requires three things, and any one of them saves the candidate

A candidate is suppressed **only** when, against some competing candidate on another track:

1. the competitor's **source score** exceeds this candidate's by at least
   `activity.bleed.min_score_margin`; **and**
2. their peak normalized correlation reaches `activity.bleed.min_correlation` within the
   lag window; **and**
3. this candidate's own band-limited level sits **more than `activity.bleed.veto_db` below
   its own track's speech reference**.

The third is the veto, and it is what the reviewer's case needs. A track's speech reference
is a robust band-limited level of that track's own candidates — what this wearer sounds like
when this wearer is talking. A lav hearing its wearer at the wearer's normal level is not
hearing bleed, however loud and however correlated the other track is. That is the whole
content of the spec's word *track-relative*, and without the veto the phrase is decoration.

Everything else is retained. A candidate the numbers condemned — margin **and** correlation
both satisfied — and the veto saved is retained **and marked `ambiguous`**, which is the
spec's "default to keeping ambiguous candidates" made visible rather than implicit.

#### Amended after implementation (M3's verify phase)

Two sentences above originally said something the code does not do. Both are corrected in
place; this section records what changed and why, because an ADR that quietly disagrees with
its implementation is worse than no ADR.

**The reference is the 75th percentile of all of a track's candidates, not the median of its
high-confidence ones.** Two changes, one deliberate and one a simplification worth being
honest about:

* *Percentile.* A track whose wearer spoke twice and heard four other people has more bleed
  candidates than speech ones, and the median of that set **is a bleed level** — which would
  anchor the veto at bleed and disable the protection exactly where it is needed. The upper
  quartile is dominated by real speech under a wider range of mixes. `nearest` interpolation
  keeps the result one of the measured integers rather than an average of two (INV-02).
* *No confidence filter.* The percentile is doing the work a confidence threshold would have
  done, and adding a second knob that selects the same candidates twice is a threshold nobody
  can tune independently. `activity.bleed.min_reference_candidates` remains the guard against
  estimating a reference from one or two regions.

These push the reference in **opposite** directions — including bleed candidates drags it
down, taking the upper quartile pushes it up — and which dominates depends on a real room.
That is exactly **OQ-017**, which the estimator now cites from the code. Both failure modes
are worth stating: a reference set too low fires the veto too often (conservative, and the
direction the spec prefers), and one set too high by a few unusually loud utterances weakens
the veto for that wearer's quieter speech. Neither can be settled against synthetic audio
whose bleed is a delayed copy of its source, so this is tuned on the first real session and
not before.

**`ambiguous` marks the veto case only, not "some but not all conditions".** The original
wording would have flagged every candidate that merely overlapped something and failed one
threshold, which is most of them; a flag that fires on the ordinary case carries no
information. It now means the one thing a human can act on: the numeric evidence said bleed
and the pipeline overrode it.

Relatedly, the per-pair evidence outcome `vetoed_by_track_level` is reported **only for a
comparison that satisfied margin and correlation**. Reporting it for every pair on a vetoed
candidate labelled comparisons the veto had nothing to do with — a competitor that was
quieter or unrelated — which reads as a suppression narrowly averted. The gate's decision
never depended on that label; the audit trail did.

### The decision runs on the score, so the score cannot be decorative

Condition 1 compares source scores, not raw levels. The score is one isolated function
combining, each as a per-mille term recorded beside the total:

* **track-relative level** — this candidate's level against its own track's reference;
* **VAD confidence** — the detector's own probability over the span;
* **cross-track dominance** — its level against the loudest competing track over the overlap;
* **correlation evidence** — how strongly it is related to whatever else was speaking.

Weights are configurable and every term is persisted, so a wrong attribution is debuggable
from the artifact rather than by re-running with print statements.

### Every threshold is a guess about a real room

`min_score_margin`, `min_correlation`, `veto_db`, the VAD thresholds, and the score weights
are all chosen against synthetic audio whose bleed is a delayed attenuated copy of its
source — the easy case. Real bleed crosses a room, reflects, and arrives filtered. Each
defaulted field cites **OQ-017**, and the pipeline records exactly the measurements that
question needs, so answering it is reading one real session's graph.

## Alternatives considered

- **Loudest track wins the interval.** The thing the spec explicitly forbids. It passes
  casual testing on solo speech and erases a quiet speaker during every real overlap.
- **Level dominance plus correlation, without the veto.** The first draft. Deletes the
  quieter of two genuinely simultaneous speakers whenever their lavs also hear each other,
  which is the normal case at a table rather than an exotic one.
- **Suppress on correlation alone.** Two people saying the same word at the same time
  correlate; so do two lavs on the same side of a room picking up the same reflection.
- **Subtract the estimated bleed from the receiving track.** Crosstalk cancellation. An
  explicit non-goal, and it changes samples the mix will use, which a pre-ASR gate must not.
- **Decide bleed after ASR, on text.** That is M4's duplicate collapse, and INV-09 forbids
  its text-dependent conclusions from reaching this graph or the mix.
- **A single global threshold in dBFS.** Fails the moment two wearers have different voices
  or different mic placements, which is every session.

## Consequences

- The gate is deliberately asymmetric: it under-suppresses rather than over-suppresses, so
  M4 will transcribe some bleed. That is the trade the spec asks for, and M4's post-ASR
  collapse is the second line of defence — with text available, which is what makes the
  cheap cases easy there and dangerous here.
- The veto depends on a track having enough of its own high-confidence speech to estimate a
  reference from. A track with too few candidates gets no reference, and the graph records
  that its veto could not be evaluated rather than pretending the reference is zero.
- Scores and their terms are in the frozen artifact (ADR-0012), so changing the scoring
  function is a change to a document two milestones consume, not a private refactor.
- Because the decision consumes scores rather than raw measurements, the attribution cache
  must invalidate on a scoring-weight change while per-track detection must not — which is
  why the two identities are split (ADR-0016).
