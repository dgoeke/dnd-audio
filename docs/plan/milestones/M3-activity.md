# M3 — Conservative speech activity and bleed rejection

**Status:** not started
**Depends on:** M2
**Spec sections:** Milestone 3; Milestone 5 (activity graph definition); Tests and
acceptance criteria 5, 15

## Goal

Per-track VAD, a conservative pre-ASR bleed gate using lag-tolerant normalized
cross-correlation, and a versioned, model-independent activity/attribution graph
that both the transcript branch and the automixer consume.

## Completion gate

- [ ] VAD runs per 16 kHz track behind an `ActivityDetector` protocol, with a
      deterministic fake / ground-truth-mask implementation used by the default
      suite (INV-10). Synthetic noise is never expected to trigger a specific
      learned Silero release.
- [ ] Silero package and model artifact/revision pinned and loaded locally — no
      unpinned runtime `torch.hub` fetch. Identity appears in cache keys and the
      report (INV-08). CPU or ONNX is the baseline so it does not contend with ASR
      for unified memory.
- [ ] Nearby speech regions merged and boundaries padded; all thresholds
      configurable; VAD probabilities and decisions persisted for debugging.
- [ ] Cross-channel similarity uses normalized speech-band cross-correlation over a
      configurable bounded lag (default ±30 ms), **not** zero-lag correlation. Both
      the peak correlation and its selected lag are recorded.
- [ ] Bleed suppressed only when another track is convincingly stronger *and* the
      signals are strongly related. Ambiguous candidates are kept by default.
- [ ] Source scoring combines track-relative speech level, VAD confidence,
      cross-track dominance, and correlation evidence — never a single global
      loudness comparison. The scoring function is isolated and its diagnostics
      appear in `ingest-report.json`.
- [ ] Tests: solo attribution, genuine two-person overlap survives, quiet bleed is
      suppressed to the right track, and correlated bleed delayed within the lag
      window is still detected with its peak lag reported.
- [ ] **The activity graph schema is checked in, versioned, and frozen** (INV-09).
      It is model-independent: nothing text-derived may enter it.
- [ ] Every retained candidate has a deterministic ID derived from sorted source
      identity and time, not completion order (INV-02).

## Explicitly not in this milestone

- Post-ASR duplicate collapse. That is text-dependent and belongs to M4.
- Generic speaker diarization or clustering. Attribution is "the person mapped to
  that track" for the MVP baseline.
- Gain envelopes. That is M5 consuming this graph.

## What M2 already provides (read before starting)

- **The 16 kHz audio VAD consumes already exists, cached and byte-stable.** Each track's
  `DerivativeRecord` in `work/timeline.json` names its `relative_path` under
  `work/cache/audio/16000/`. Read it through `timeline.pcm.open_pcm`; do not resample
  anything yourself. `DerivativeCache.get()` takes the expected output length, because a
  cache entry that is the wrong length must be a miss rather than a subtly short track.
- **The 48↔16 kHz mapping is a settled contract — use it, do not re-derive it.**
  `timeline.resample.to_source_sample` and `to_derivative_interval`. Output sample `k`
  corresponds to input sample `3k` exactly (the FIR's group delay divides by the decimation
  factor). The reverse direction lands between grid points, so an interval **floors its
  start and ceils its end**. Rounding both ends the same way shrinks a speech region by up
  to two samples, which is how a word loses its first phoneme. M3 is this contract's first
  real consumer.
- **Silence has three causes and they are deliberately indistinguishable** to a
  `TrackReader` caller: before the track started, inside a real gap, and after it stopped.
  A VAD sees zeros in all three. Every track answers to the session's aligned
  `duration_samples`, so do not special-case a track that ended early.
- **The lag-tolerant normalized cross-correlation this milestone's bleed gate needs already
  exists.** `timeline.syncqa.measure_lag` returns the peak correlation and the lag it
  occurred at, over a bounded lag window, normalized by both signals' energy — an
  unnormalized correlation ranks tracks by volume and calls the loudest one the best match.
  Reuse it rather than writing a second one; if M3 needs a variant, extend it there.
- **INV-08 for the activity cache.** Whatever key M3 builds must carry
  `TIMELINE_SEMANTICS_VERSION` and the derivative's own `cache_key`, not just the source
  hashes: a placement fix moves a chunk without changing a source byte, and a stale
  activity graph aligned to a timeline that has moved is not obviously wrong.
- **Commit a cache entry only after INV-01 has been re-verified**, never at publish time.
  M2 shipped the other ordering and it meant a run that correctly *failed* on a changed
  source still left a poisoned entry keyed on the bytes it read. See M2's closeout.
- **`timeline.json`'s schema is frozen at version 1** — additive optional fields only.
  Every interval in it is half-open and there are no floats anywhere in the document.

## Known risks and open questions

- Depends on **OQ-010**.
- This is the milestone where a "reasonable" simplification does the most damage:
  a global loudest-wins rule passes casual testing and erases quiet speakers during
  real overlap. Losing real overlapped speech is worse than extra ASR compute.
- The graph contract is consumed by two downstream milestones. Changing it later
  means redoing both. Spend the time on it here.
