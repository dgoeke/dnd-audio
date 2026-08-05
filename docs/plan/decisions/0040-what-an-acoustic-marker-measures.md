# ADR-0040 — Four things an acoustic marker measures, and only one of them is drift

**Status:** accepted
**Date:** 2026-08-05
**Milestone:** M10

## Context

The spec has said since M0 that a start/end clap correlation may be used as synchronization
QA, and that a lag which changes between the two ends is evidence of sample-clock drift:

> Measure each track's relative clap lag near both ends and warn when the lag changes
> materially, because a changing lag is evidence of sample-clock drift rather than a constant
> timecode offset.

Acceptance criterion 15 says the same thing in the same unconditional voice: *"a synthetic
change in start-versus-end clap lag emits a drift warning"*.

**That sentence is false, and it has been false since it was written.** A lag measured between
two microphones from a room sound is the sum of two independent quantities: how far apart the
recordings are placed on the timeline, and how far the sound travelled to each capsule. If the
phone moves half a metre between the start marker and the end marker, or if a wearer leans
back in their chair, the acoustic term changes by about 1.5 ms with two perfectly synchronized
recorders. Six lavs at a table are 0.5–3 m from any one source, which OQ-025 records as a
1.5–9 ms propagation spread — *the same order as the drift being claimed*, since OQ-006
measured ≈1 ppm and bounded it at ±3 ppm, or 14–43 ms across four hours.

M10's charter already says this correctly, because it was **finding 3 of the first M10 plan
review** (`../reviews/M10-plan-20260804-1735.md`): "fixed phone position alone cannot isolate
recorder drift. A moving lav changes propagation delay even when recorder clocks are perfect."
That finding was accepted and propagated into the marker charter, downstream capture plans,
OQ-025, `STATE.md` and
`ROADMAP.md`. It was never propagated into **the spec**, which is the authoritative document.
So the spec, the charter and the gate disagreed, and the working agreement is explicit that
code and spec must never disagree silently — a rule that applies equally to spec and charter.

The second plan review (`../reviews/M10-plan-20260805-0606.md`, finding 7) caught that M10's
working plan proposed only to "name the generated marker alongside the clap", which would have
left the false causal claim standing in both passages.

There is a second, quieter reason this ADR exists. M10 introduces a much more precise
instrument than a clap — a matched filter resolving integer samples where a hand-picked clap
resolved milliseconds — and precision is exactly what makes a wrong causal story dangerous. An
instrument that reports 0.4 ms of "drift" with four significant figures invites belief.

## Decision

**Four quantities, named separately, never collapsed.** M10's analyzer, the spec, the charter
and the operator runbook all use these four words for these four things:

| quantity | what it is | what may be concluded |
| --- | --- | --- |
| **timecode placement** | where a file sits on the session timeline, from `bext.time_reference` | the only thing that places a file, including one with no shared audio at all |
| **acoustic verification** | a marker heard on several tracks agreeing with that placement | the jam worked, or it did not (OQ-023) |
| **differential acoustic arrival** | the per-track lag between the marker's acoustic arrival and its predicted position | geometry **plus** placement, inseparably |
| **recorder drift** | a *change* in differential arrival between two occurrences | only under fixed source **and** lav geometry |

**A start-to-end change is always reported as `differential_arrival_change_samples`.** It is
promoted to `clock_drift_evidence` only when the event log asserts one unchanged acoustic
geometry ID covering the phone *and* every transmitter being compared. Absent that assertion —
which is the normal case for any session with people wearing the microphones — the change is
reported, and reported as inconclusive about clocks.

**The spec is amended in both places**, in the same commit as this ADR. The Milestone-2
synchronization-QA paragraph and acceptance criterion 15 both gain the geometry condition. The
existing `sync_qa` implementation is **not** changed: its `clock_drift_suspected` warning
already says "no correction was applied" and already fires only on a lag change, and M8 gave
it the quantization floor that keeps it honest. What was wrong was the sentence claiming what
that warning *means*.

**Nothing here corrects anything.** INV-12 forbids correcting by an unmeasured amount, and a
measured amount that conflates geometry with clocks is unmeasured for this purpose. The marker
never moves a sample, never overrides valid timecode, and never places a file that did not
record it.

**A marker cannot replace the jam, for a reason unrelated to precision.** Timecode supplies an
origin per *file*, including a file with no overlap at all — a transmitter switched off and
back on mid-session produces a fresh recording that a start-of-session sound cannot place. The
marker is verification; the jam is placement. This holds however good the acoustics turn out
to be.

## Alternatives considered

**Delete the drift claim from the spec entirely** and let M10 own the vocabulary. Rejected:
the spec is the authoritative document, an operator reading only the spec would still be
misled, and "the charter says otherwise" is precisely the silent disagreement the working
agreement prohibits.

**Keep one number and qualify it in prose.** Rejected. M8's whole lesson in `sync_qa` was that
merging two outcomes into one code makes them indistinguishable in practice — "nobody clapped"
and "the jam failed" read identically until they were separated. Geometry change and clock
drift are the same trap one level up, and prose in a report is not something a downstream
consumer branches on.

**Estimate the geometry term and subtract it**, from the spread across six tracks. Rejected as
a fantasy of precision: it needs per-wearer positions this project never knows, it fails
exactly when someone moves, and it would convert an honest inconclusive into a confident wrong
number. Source localization is an explicit non-goal of M10.

**Require fixed geometry for every marker occurrence**, so the distinction never arises.
Rejected: it would make the marker useless in a real session, where people move. The
instrument is worth having for jam verification even when it can say nothing about clocks.

## Consequences

An operator who plays the marker at both ends of an ordinary session gets a real measurement
and an honest label: differential arrival changed by *n* samples, geometry unverified. An
operator who runs a fixed-transmitter experiment gets an additional fixed-endpoint clock-rate
measurement from the same instrument and command. OQ-006 already accepts the no-correction MVP;
new evidence matters only if it shows a material problem.

What this makes harder: the event log becomes load-bearing rather than a convenience, because
the geometry ID is the only thing that can license the stronger claim. An operator who does not
keep one cannot get a drift classification out of this pipeline at all. That is the intended
trade — the alternative is a classification resting on an assumption nobody recorded.

What would make us revisit: a capture method that fixes both endpoints without fixing the
wearers — an electrical injection into each receiver, or a transmitter-mounted emitter — would
remove the geometry term rather than merely labelling it. Both are named non-goals of M10 and
neither is cheap.

This ADR constrains the constants M10 freezes but does not set them. The "material" threshold
for a differential-arrival change is empirical and cites **OQ-025** and **OQ-029** until the
bench resolves it; ADR-0042 freezes it with the measured evidence.
