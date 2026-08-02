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

## Known risks and open questions

- Depends on **OQ-010**.
- This is the milestone where a "reasonable" simplification does the most damage:
  a global loudest-wins rule passes casual testing and erases quiet speakers during
  real overlap. Losing real overlapped speech is worse than extra ASR compute.
- The graph contract is consumed by two downstream milestones. Changing it later
  means redoing both. Spend the time on it here.
