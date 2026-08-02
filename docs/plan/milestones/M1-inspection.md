# M1 — Inspection and immutable ingest manifest

**Status:** not started
**Depends on:** M0
**Spec sections:** Session input contract; Milestone 1; Tests and acceptance
criteria 1, 2, 10

## Goal

`dnd-audio inspect /path/to/session` discovers sources, captures everything
`ffprobe` and a generic RIFF walk can tell us, applies selection and roster rules,
extracts timecode through a testable strategy chain, and writes a deterministic
`work/manifest.json`. The synthetic fixture generator lands here too.

## Completion gate

- [ ] Synthetic fixture generator produces six virtual tracks with multiple chunks
      each, differing start offsets, a real gap, a shared clap, solo activity that
      bleeds quietly into other tracks, and one two-speaker interval. No audio
      binaries are checked into the repository.
- [ ] Per-file capture: relative path, SHA-256, size, displayed duration,
      `duration_ts`, time base, exact PCM sample count where available, codec,
      sample format/bit depth, sample rate, channel count.
- [ ] Complete raw `ffprobe -show_format -show_streams` JSON retained verbatim
      under a content-hash-addressed path beside the manifest, before any
      project-specific parsing.
- [ ] Generic RIFF/RF64 chunk inventory (ID, offset, size) that does not depend on
      `ffprobe` surfacing unknown chunks; bounded textual payloads retained,
      larger ones hashed.
- [ ] Timecode strategy chain: BWF `time_reference` preferred as an integer sample
      count, timecode tag plus configured frame rate as fallback, every assumption
      and fallback recorded. Fails with an actionable diagnostic when nothing
      reliable exists (INV-12).
- [ ] Selection rules: `orig` preferred, `edit` associated but ignored, processed-only
      is fatal unless `allow_processed_audio`, duplicates detected by content hash,
      warnings for unexpected formats and sequence discontinuities.
- [ ] Roster rules: `active_tracks: auto` derives active participants from
      configured directories containing a usable original; explicit lists make
      every listed track required; an unconfigured directory is never attributed
      to a speaker (INV-11); the report shows known/observed/per-track counts and
      lists missing, empty, and extra directories.
- [ ] `recovery.source_time_overrides` honored, keyed by source-relative path, with
      optional SHA-256 verification, exactly one of `start_timecode` or
      `start_offset_samples`, and prominent recording in manifest and report.
- [ ] Inspecting unchanged input twice yields byte-identical `manifest.json`
      (INV-02); no wall-clock or cache telemetry inside it (INV-03).
- [ ] Inspection cache identity includes source hash, exact FFmpeg and FFprobe
      versions, the ffprobe command/options, and the RIFF-parser plus
      manifest-schema versions. A simulated tool-version bump forces re-inspection
      (INV-08).
- [ ] Raw source hashes unchanged before and after a run (INV-01).

## Explicitly not in this milestone

- Chunk ordering, gap/overlap reasoning, or any session timeline. That is M2.
- Decoding audio for anything beyond an exact sample count when `ffprobe` cannot
  supply one.

## Known risks and open questions

- Depends on **OQ-001, OQ-003, OQ-004, OQ-005, OQ-007, OQ-011**. Every guess about
  DJI's real layout must be behind a named strategy in the chain and tagged with
  its `OQ-` ID so H1 can settle it cheaply.
- **Start acquiring the H1 fixture during this milestone.** Do not wait.
- Byte-identical output is easy to lose to dict ordering, path separators, and
  float formatting. Build the canonical writer first and route everything through it.
