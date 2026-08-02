# M2 — Reconstruct six synchronized virtual tracks

**Status:** not started
**Depends on:** M1
**Spec sections:** Milestone 2; Tests and acceptance criteria 2, 3, 4, 13

## Goal

`dnd-audio ingest` turns the manifest into six synchronized virtual tracks: chunks
ordered by embedded timecode, real gaps preserved as silence, a lossless streamed
48 kHz working path, and cached 16 kHz mono derivatives — with an exact recorded
mapping between source samples, working samples, and session time.

## Completion gate

- [ ] Session time zero from the earliest valid source start unless
      `timecode.origin_timecode` supplies an explicit origin on `origin_date`.
      `origin_date` is never inferred from a date-shaped `session_id`.
- [ ] Chunks sorted by parsed start time, not filename order; each chunk's expected
      end validated against the next chunk's start.
- [ ] Real gaps preserved as silence. A transmitter switched off and back on does
      not slide later audio earlier. Verified against synthetic ground truth.
- [ ] Overlaps detected. Only quantization-scale overlaps resolved automatically;
      anything larger warns and requires explicit policy rather than discarding audio.
- [ ] Midnight rollover: `infer_forward` infers a single forward rollover only when
      chunk sequence and session span make it unambiguous, and records the
      decision. Ambiguity demands a dated origin or an override, never an ad hoc
      interpretation.
- [ ] Exact sample-position tests for non-drop, fractional (24000/1001,
      30000/1001), drop-frame, rollover, and explicit-override cases (INV-04).
- [ ] Lossless 48 kHz float working path is streamed/windowed over a segment map,
      never six session-length arrays in RAM; contiguous intermediates use RF64;
      work-space and disk preflighted (INV-07).
- [ ] 16 kHz mono derivatives cached, with resampler delay and end rounding
      accounted for in the 48↔16 kHz mapping.
- [ ] Aligned output duration is set by the latest track end and matches within one
      48 kHz sample.
- [ ] A selected 44.1 kHz source, or chunks within one track disagreeing on sample
      rate, fails before timeline construction with a clear diagnostic.
- [ ] Optional clap cross-correlation runs as QA only: it reports disagreement with
      timecode and never overrides valid timecode. Lag is measured near both ends
      and a materially changed lag warns (drift evidence, not correction).
- [ ] An interface hook exists for a future affine time warp, unused in the MVP.

## Explicitly not in this milestone

- Any automatic drift correction. Warn only.
- VAD, activity, or anything that interprets the audio's content.
- Phase-coherent multichannel processing of any kind.

## What M1 already provides (read before starting)

- **Per-file timing evidence already exists, in typed form** (ADR-0006). The manifest's
  `start_time.evidence` is a discriminated union, and the three variants do not share a
  coordinate system: `bwf_sample_reference` is unsigned samples since midnight **at the
  file's own rate**; `timecode` is an exact integer frame index plus a rational rate;
  `session_offset_samples` is **signed, at 48 kHz, relative to session zero**. Reconciling
  them is this milestone's job. Do not add a fourth "just give me the number" accessor —
  that is the collapse ADR-0006 exists to prevent.
- **This milestone owes acceptance criterion 2 a documented quantization rule.** A frame
  at 30000/1001 fps is `8008/5` samples at 48 kHz, so "the expected integer sample
  position" is a property of a rounding rule, not of the evidence. Define it, write it
  down, and test it at 24000/1001 and 30000/1001 where it actually bites. M1 deliberately
  stopped short of inventing one.
- **A non-48 kHz source is a warning in M1 and must be fatal here**, before timeline
  construction. The manifest already carries `unexpected_sample_rate` per source and the
  container facts that explain it; the diagnostic exists, the refusal does not.
- **`container.sample_count` is exact and needs no decode** — it comes from the RIFF
  `data` size over the block alignment, cross-checked against `duration_ts`, with their
  agreement recorded as `sample_count_agrees` (OQ-011).
- **Every candidate is in the manifest, not only the selected ones**, including files in
  unconfigured directories under `unassigned`. Filter on `role == "selected"` when
  building the timeline; nothing else belongs on it.
- **`_snapshot`/`_verify_unchanged` in `inspection/runner.py` is the INV-01 machinery.**
  If this milestone writes anywhere new, extend the "output inside raw" check — and note
  it compares *resolved* paths, because a lexical comparison was defeated by one symlink.

## Known risks and open questions

- Depends on **OQ-004, OQ-006, OQ-011**, and **settles OQ-013**: the work-space
  preflight this milestone builds is what replaces `doctor`'s estimated 40 GiB warning
  threshold with a number derived from the session's actual length.
- **`dnd_audio.determinism.write_atomic` is for artifacts, not audio.** It holds the
  whole payload in memory, which is right for JSON and a direct INV-07 violation for a
  session-length waveform. The streamed working-audio path is this milestone's to build.
- Exact-time helpers already exist: rates are `Fraction`, and `public_seconds()` is the
  only float-producing conversion, built on an integer-millisecond quantizer with a
  documented tie rule. Do not add a second float path.
- INV-04 and INV-07 are both at maximum risk here. Resist the convenience of a
  float seconds field and of one big NumPy array; both work fine on a two-minute
  fixture and fail on a four-hour session.
- The 48→16 kHz mapping is the most likely source of a subtle, late-discovered
  offset. Test it against known impulse positions, not just durations.
