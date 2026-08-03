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

## What M2 already provides (read before starting)

- **The 48 kHz working path is a segment map, not files.** `work/timeline.json` is the
  authoritative document; `timeline.reader.TrackReader.read(start, n)` returns a bounded
  window of one reconstructed track, silence included. `ingest --materialize-48k` will
  write contiguous RF64 files, but they are **disposable content-addressed cache
  artifacts** and nothing in the mix may depend on their existence (ADR-0011). Mixing means
  stepping six `TrackReader`s over the same window range.
- **Every track answers to the session's aligned `duration_samples`**, returning silence
  past its own end, so the mix does not need to pad or special-case a short track.
- **`timeline.wavwrite` is the streamed float32 writer** for the lossless mix intermediate:
  temp-then-rename, and it chooses RF64 from the *declared* length rather than discovering
  the 4 GiB limit partway through. Use it. `determinism.write_atomic` is for JSON and holds
  its whole payload in memory — reaching for it here is a direct INV-07 violation.
- **`timeline.preflight` sizes a run from the timeline's actual duration and the artifacts
  requested.** M5 must add its own term: the mix intermediate is the third term of
  `doctor`'s original 40 GiB estimate and it does not exist yet (OQ-013).
- **The technique for proving INV-07 over a composed path is in `tests/test_memory.py`.**
  Instrument reads and writes into one ordered event log and assert a write happens before
  the last read — nothing that accumulates a session-length array can satisfy that.
  Bounding one component proves nothing about a caller that collects every window.
- **`TrackReader` holds one file descriptor per audio segment.** Six tracks with a handful
  of chunks each is fine; if M5 ever opens many sessions' worth at once, that is where the
  limit is.

## What M3 already provides (read before starting)

- **`work/activity.json` is the only thing the mix may consult about who was speaking**
  (INV-09, whose *enforcement* this milestone owns). It is frozen at schema version 1
  (ADR-0012) and carries no floats.
- **One track's active intervals, with a confidence, are a direct read.** Filter
  `candidates` to `decision == "retained"` and that `track_id`; each carries
  `probability_permille`, `peak_probability_permille`, `band_level_mbfs`, and
  `relative_level_mb`. `test_activity_artifact.py::TestTheConsumerReads::test_m5_takes_one_tracks_active_intervals_with_a_confidence`
  is that access pattern, written before this milestone existed.
- **`ActivityTrack.speech_reference_mbfs` is the per-track voice level the first gate
  criterion asks you to estimate** — the 75th percentile of that track's own candidate
  levels, band-limited to 300–3400 Hz. It is `None` where the track had too little speech to
  establish one; treat that as "unknown", never as zero. Clamping is still M5's job, and the
  number itself is unsettled until OQ-017 has a real session.
- **`ambiguous` is where obvious correlated bleed nearly won.** It marks a candidate the
  numbers condemned — margin *and* correlation both satisfied — that the track-level veto
  kept (ADR-0014). The "obvious correlated bleed is not promoted on two channels
  simultaneously" criterion is about exactly these; `evidence` names the competitor, the peak
  correlation, and the lag it occurred at.
- **The graph does not contain gain.** It says who was speaking and how sure the pipeline is;
  envelopes are entirely this milestone's (ADR-0012 deliberately kept them out).

## Known risks and open questions

- Decoded loudness alone is *not* evidence of correct channel selection. If the
  only tests are loudness tests, a mix that picks the wrong speaker will pass.
  The envelope assertions are the real gate.
- **The INV-09 field allowlist freezes property names, not prose.**
  `ActivityDecision.detail` and `ActivityNote.message` are unrestricted strings. **The mixer
  must not read either** — they are human-facing audit text, and deriving a single sample
  from them would make the mix depend on content no test constrains. The structural
  "imports nothing from the ASR layer" check does not catch this. Raised by independent
  review in M3's verify phase; see `../reviews/M3-code-20260802-1708.md`.
- **`activity.bleed.compare_pairs` is quadratic in the session's candidate count.** It
  enumerates every pair via `itertools.combinations` and only then rejects non-overlaps.
  Candidates are already sorted by start sample, so a sweep would be `O(n log n)` plus the
  pairs that genuinely overlap. Deferred in M3 with no measured evidence either way — the
  per-pair work rejected is one integer comparison, and a real session's candidate count is
  unknown until H2. If M5 walks the candidate set at scale and it hurts, that is the fix.
- The true-peak ceiling applies to the decoded MP3, not the pre-encode
  intermediate. Lossy encoding introduces overshoot.
