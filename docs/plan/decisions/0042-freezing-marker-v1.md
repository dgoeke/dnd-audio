# ADR-0042 — Freezing marker v1

**Status:** proposed — **deliberately unfilled until the phone/DJI bench has run.**
**Date:** 2026-08-05 (opened)
**Milestone:** M10

## Context

ADR-0041 decides that `MARKER_SPECS` holds named candidates and that **there is no `v1` key**
until physical evidence selects one. This ADR is the record that adds it, and it is opened
empty on purpose: an ADR that recorded a choice nobody has evidence for would be worse than an
obligation nobody can miss.

**M10 cannot close while this ADR reads `proposed`.** That is its function.

The charter is explicit about why theory is not enough. Exact frequencies, chirp durations,
directions, gaps, sample format and peak level are all properties of what a phone speaker
radiates, what a lav capsule accepts, and what a room does in between — none of which a
synthetic fixture can answer. The provisional three-chirp 500 Hz–8 kHz design in the charter
is a *candidate*, described there as "not frozen by this planning document".

## Decision

**Pending the bench.** When it has run, this ADR records, and nothing else may:

- **Which candidate wins**, from the objective evidence the charter's bench protocol lists:
  correlation peak sharpness and ambiguity on every DJI track; tolerance to phone/browser
  resampling, lav band limiting, reverberation, gain change and moderate clipping; audibility
  at the farthest lav without clipping the nearest; reliable distinction from normal table
  audio; and stable detection across repeated plays on the intended phone.
- **The complete integer PCM sample sequence, by SHA-256**, together with the human-readable
  recipe that regenerates it — both, because a hash alone cannot be reasoned about and a
  recipe alone cannot be verified.
- **The frozen anchor**, as an exact sample relative to the WAV's first sample.
- **The detector thresholds and tolerances**, in the integer permille domain ADR-0041 fixes:
  normalized peak score, runner-up separation, inter-chirp gap tolerance, required chirp
  count, the non-maximum-suppression radius, the bounded cross-track association lag, the
  clipping and weak-signal thresholds, and the "material" differential-arrival change
  threshold above which ADR-0040 permits a warning.
- **The measured tolerance** for repeated same-position lag, which is what makes a later
  change interpretable at all.

Every one of those constants currently cites **OQ-025** or **OQ-029** as provisional. When
this ADR is accepted they cite it instead, and the two open questions record what was measured.

A future marker change takes a **new** semantic version and a new versioned filename. It does
not silently replace v1, and v1's frozen hash stays in this document as history.

## Alternatives considered

**Freeze the charter's provisional candidate now** and treat the bench as confirmation that
could retire it later. Rejected by the charter and by the operator on 2026-08-05. It would put
a golden SHA-256 and a full synthetic regression battery behind a guess, and the two bench
outcomes that would invalidate it — no candidate surviving at the farthest lav, or playback
warping enough to break sequence detection — are exactly the ones theory cannot rule out.

**Never freeze; always pass a spec name.** Rejected: an operator recording Session Zero should
not be choosing a waveform, and a marker whose identity is not frozen cannot be compared
across sessions, which is the entire point of measuring drift between two of them.

## Consequences

Until this is accepted, `marker build OUTPUT_DIRECTORY` exits nonzero naming the bench, and
the only way to produce a marker is the hidden `--marker` option the bench protocol documents.
That is the intended state, and it is what makes the obligation impossible to forget.

Once accepted, the golden test that pins v1's SHA-256 also pins everything underneath it —
the sine table, the integer phase arithmetic, and the RIFF layout in `marker/wav.py`. Any
change to any of those turns that test red, which is the desired behaviour: they are all part
of what the frozen bytes mean.

What would make us revisit: a materially better phone, a receiver firmware change altering the
recorded band, or H2 evidence that the marker is failing in practice. Any of those is a new
version, never an edit to this one.
