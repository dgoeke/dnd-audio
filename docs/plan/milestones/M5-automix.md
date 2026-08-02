# M5 — Merged MP3 automix

**Status:** not started
**Depends on:** M3 (only — never M4 or M6)
**Spec sections:** Milestone 5; Tests and acceptance criteria 8, 11; automixer
gain-envelope assertions

## Goal

`dnd-audio mix` turns the synchronized 48 kHz originals plus the activity graph
into a listenable mono `session.mp3`: smoothed speech-aware gain envelopes, a
streamed mix, two-pass loudness normalization, and MP3 encode/decode verification.

## Completion gate

- [ ] Conservative per-track voice-level correction estimated from high-confidence
      speech attributed to that track, clamped to a safe range.
- [ ] Activity decisions become continuously smoothed gain envelopes: solo speech
      favors that lav and strongly attenuates the other five; genuine overlap keeps
      each active person's own lav audible via equal-power or otherwise bounded
      gain sharing; silence blends low-level room tone without six noise floors
      adding coherently (Dugan-style normalized gain-share is the baseline).
- [ ] Short attack, longer release/crossfade — no clipped words, clicks, or pumping.
- [ ] **Envelope-level tests** against the deterministic activity graph, with
      explicit configurable tolerances:
  - [ ] after the attack interval a solo speaker's gain dominates every inactive
        channel by at least the configured margin;
  - [ ] during genuine two-person overlap both active channels retain nontrivial gain;
  - [ ] the normalized/equal-power invariant stays bounded at every sample or
        control frame, including silence and transitions;
  - [ ] obvious correlated bleed is not promoted on two channels simultaneously;
  - [ ] envelopes have no discontinuities and respect attack, release, and max-slew limits.
- [ ] Mono output; streamed/windowed mixing, never six full waveforms in RAM (INV-07).
- [ ] Two-pass loudness toward `-16 LUFS` integrated by default.
- [ ] 128 kbps mono MP3 with session ID/title metadata, then **decoded and measured**:
      integrated loudness within 1 LU of target, true peak within the `-1.5 dBTP`
      ceiling plus a documented measurement tolerance, duration within one MP3 frame
      of expected. Pre-encode gain reduction and re-encode from the lossless
      intermediate when the decoded file overshoots; retries bounded; all
      measurements retained in the report; the stage **fails** rather than claiming
      compliance.
- [ ] Lossless mix intermediate kept in `work/` for debugging and cache reuse, not
      as a user-facing deliverable.
- [ ] A simulated transcription failure still produces the MP3 and report, with the
      transcript stage marked `failed` and `process` exiting nonzero (INV-09, INV-13).
- [ ] The mixer imports nothing from the ASR/transcript layer. Verified structurally,
      not just by convention.

## Explicitly not in this milestone

- Stereo, spatial reconstruction, or any phase-coherent processing.
- Naive summing of six channels — explicitly forbidden by the spec.
- Neural source separation or crosstalk subtraction.

## Known risks and open questions

- Decoded loudness alone is *not* evidence of correct channel selection. If the
  only tests are loudness tests, a mix that picks the wrong speaker will pass.
  The envelope assertions are the real gate.
- The true-peak ceiling applies to the decoded MP3, not the pre-encode
  intermediate. Lossy encoding introduces overshoot.
