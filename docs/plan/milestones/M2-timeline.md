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

## Known risks and open questions

- Depends on **OQ-004, OQ-006, OQ-011**.
- INV-04 and INV-07 are both at maximum risk here. Resist the convenience of a
  float seconds field and of one big NumPy array; both work fine on a two-minute
  fixture and fail on a four-hour session.
- The 48→16 kHz mapping is the most likely source of a subtle, late-discovered
  offset. Test it against known impulse positions, not just durations.
