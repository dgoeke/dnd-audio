# ADR-0043 — Retire dedicated pre-campaign captures in favor of live Session Zero

**Status:** accepted
**Date:** 2026-08-05

## Context

The roadmap reserved two hardware-only steps before ordinary campaign use: a short capture for
metadata breadth and a separate long drift capture. That sequence was reasonable when the DJI
file layout, receiver jam, sample-clock stability, marker reach, and real-model behavior were
mostly assumptions.

The evidence base is now materially different:

- the sample probe established DJI naming, BWF/iXML layout, recorder-domain time references,
  frame quantization, and real PCM variants;
- the jam-verification capture proved that receiver timecode reaches transmitter files and
  bounded relative clock drift at approximately 1 ppm (within a ±3 ppm measurement floor);
- the minimal two-person capture supplied real speech, deliberate overlap, repeated exact-short
  words, handoffs, room bleed, and a 10.6-minute false-positive corpus;
- the intended-phone marker bench exercised all six transmitters and all three jammed receivers,
  found every planned marker occurrence, and measured at most 17 samples (0.35 ms) of
  same-geometry change over about 11.8 minutes; and
- M8 through M10 turned those recordings into guarded production behavior and diagnostics.

The unperformed parts of the two fixtures are either distinctions without a credible pipeline
consequence (a third receiver after three-receiver bench success, six rather than four speech
tracks, a filename counter across a power cycle, receiver wall clock, and receiver-side edit
files) or measurements better made on ordinary play (real-table thresholds, automix taste,
cache footprint, and conversational ASR boundaries).

The owner has therefore chosen the campaign's live Session Zero as the next recording. It is
irreplaceable play, not a laboratory fixture, and its raw sources must be preserved and archived
under the existing invariants.

## Decision

Retire the two dedicated hardware milestones and their controlled-capture runbook. Do not
replace them with another pre-campaign fixture.

Create one final implementation milestone, M11, blocked on the live Session Zero recording. M11
processes the baseline configuration first, evaluates only questions that real play can answer,
and changes a default or architecture only when the session supplies concrete evidence. It also
owns the existing event-first architecture idea as a contingency, not as presumed work.

Questions already settled by the sample probe, minimal capture, jam verification, or marker
bench remain answered evidence records. Questions whose only remaining evidence was broader
hardware count, an artificial capture ritual, or a harmless guarded edge case are dropped with
the reason recorded. They are not transferred into M11.

The normal capture procedure retains the proven operational safeguards: stable physical labels,
receiver timecode jam and display confirmation, internal recording on all six transmitters,
marker v1 (or claps) for acoustic QA, immutable transfer, immediate verified private archive,
and a baseline process run before tuning.

## Consequences

- Live Session Zero is the first full-duration campaign recording and the evidence source for
  remaining tuning. There is no claim that moving-wearer start/end marker differences isolate
  recorder drift; ADR-0040 still governs that distinction.
- M11 owns measured disk/cache use, real-table activity behavior, ASR assembly boundaries,
  automix acceptance, and the decision whether event-first work is warranted.
- No dedicated power-cycle, dual-file, third-receiver, six-versus-four-microphone, or four-hour
  soak obligation remains.
- M7b still waits for an accepted processed session and follows M11, because publication,
  retention, reclamation, and any deletion authority should use the tuned result.
- The product spec, roadmap, state ledger, open-question ledger, source comments, schemas, and
  historical forward-looking notes are amended together so the retired plan cannot survive as
  accidental instructions.
