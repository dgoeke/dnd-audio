# H2 — Drift soak and first-session validation

**Status:** not started
**Depends on:** H1, M2 (M5 useful), a long recording
**Spec sections:** Milestone 2 (drift paragraph); Owner notes 6

## Goal

Evidence for or against the MVP's no-drift-correction assumption, from either a
~4-hour soak fixture with synchronized transients near both ends or the first real
session's start/end clap measurements.

## Completion gate

- [ ] A ~4-hour recording exists with a distinctive transient near the beginning and
      near the end, across all three kits — or the first real session provides the
      same via its start/end claps.
- [ ] Differential clap lag measured near both ends for every track, and the change
      between them recorded.
- [ ] **OQ-006** marked answered with the measured numbers.
- [ ] The warning threshold is configured from that measurement and documented,
      rather than guessed.
- [ ] A synthetic drift case emits the drift warning **without** applying any
      automatic correction.
- [ ] **OQ-014** and **OQ-016** answered or re-scoped from the same recording: how long a
      real session actually is, and whether one is ever run without a configured origin —
      which is the only case where the shortest-arc day-assignment heuristic can be wrong.
- [ ] `work/` measured after a complete run, so **OQ-013** gets the full-pipeline number M2
      could only bound for its own stage.
- [ ] If drift proves material, an ADR records the finding and the affine-time-warp
      hook's activation is scoped as post-MVP work — not implemented reactively here.
- [ ] For the first real session, raw files and all outputs are retained even if the
      transcript is imperfect; they are the basis for tuning bleed thresholds and
      the automixer.

## Explicitly not in this milestone

- Implementing automatic affine drift correction. Explicitly post-MVP.
- Retuning the automixer — that is a separate pass once real-session diagnostics
  exist.

## What M2 already provides (read before starting)

**The measuring instrument exists.** `session.sync_qa` (disabled by default) correlates
each track against a reference near both ends of the session and reports the lag at each,
never a correction. A constant lag is a timecode disagreement; a lag that *changes* between
the ends is exactly the drift evidence this milestone's gate asks for. Enable it, run
`ingest`, and read the warnings — no new measurement code should be needed.

Two cautions from building it. Below the configured correlation threshold the answer is
"no shared transient found" rather than a number, because two noise floors will always
agree somewhere. And a drift *fixture* must move the transient in the **audio samples**
while holding the metadata identical across tracks; move the metadata instead and the test
passes without the correlator ever being exercised.

## Known risks and open questions

- Depends on **OQ-006**. Also carries **OQ-013**, **OQ-014**, and **OQ-016**, all of which
  need a real session's wall-clock span and disk footprint rather than more code.
- A four-hour soak is cheap to record and expensive to skip: without it, the first
  real session is simultaneously the first drift test and the first everything else.
