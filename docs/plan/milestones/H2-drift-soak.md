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
- [ ] **OQ-018's remaining two parts answered**, both added by M6b and both needing speech
      its 47-second capture could not contain:
      - **item 4**, the text-similarity thresholds — how differently Qwen transcribes *the
        same* utterance heard on two lavs. This is the one that can silently destroy speech,
        which the spec calls the worse failure, so it wants one utterance deliberately heard
        on two transmitters at known relative levels.
      - **the unmeasured half of item 3** — whether splitting a truncated response at the
        quietest interior point resolves it better than at the midpoint. A *natural*
        truncation is required, which needs one continuous utterance long enough to exhaust
        1024 generated tokens; forcing a low ceiling truncates everything and measures
        nothing about the split.
      - **item 5**, presentation gaps — compare logged same-speaker pauses on both sides of
        350 ms, including distinct statements sharing one ASR request and one intended turn
        split near the boundary. Score granular records separately from public turns; request
        batching is not turn evidence (ADR-0034).
- [ ] **OQ-027's M9 remedy is checked over real multi-wearer speech.** With activity and cached
      ASR fixed, compare 0/20/100 ms transcript-only leading grace against logged direct-source
      hard onsets. Confirm recovered direct words, weak-track claims, piece ownership lineage,
      request identities, activity and mix rather than using dropped-pair count alone.
- [ ] Contained-fragment collapse is audited over the long recording: report every
      `contained_fragment` decision and terminating chain, retain unrelated or genuinely
      simultaneous speech, and include exact simultaneous `Yes`/`Okay` ground truth. A false
      deletion requires a new software milestone, not an in-place H2 threshold tweak
      (ADR-0033).

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
