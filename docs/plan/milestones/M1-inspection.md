# M1 — Inspection and immutable ingest manifest

**Status:** in progress
**Depends on:** M0
**Spec sections:** Session input contract; Milestone 1; Tests and acceptance
criteria 1 and 10. Criterion 2's *evidence* half lands here — extracting a BWF sample
reference or a timecode tag and preserving it exactly. Mapping that evidence onto
session sample positions, including midnight rollover, is M2's (ADR-0006).

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

## What M0 already provides (read before starting)

Four contracts land in M0 that this milestone inherits. See M0's closeout for the
reasoning behind each.

- **Every stage needs a recorded outcome.** `ReportBuilder.build()` refuses to assemble
  a report with any stage unaccounted for. `inspect` must call `stage_skipped()` with a
  reason for the five stages it does not run.
- **A track's `input` directory must be named for its `track_id`** — enforced in
  `TrackConfig`, which is what makes INV-11 structural. Discovery can rely on it.
- **The manifest schema version is provisional until this milestone closes.** Change
  version 1 freely while M1 is open; after it closes, only additive optional fields, and
  anything else bumps the version (ADR-0005).
- **Build the inspection cache identity on `config_hash()`**, not on raw `session.yaml`
  bytes. The resolved projection materializes defaults and sorts the roster, so a config
  that omits a default hashes identically to one that states it and a reordered roster
  does not invalidate caches. Add FFmpeg/FFprobe versions and the parser versions on top
  of it (INV-08).
- The canonical writer INV-02 needs already exists: `dnd_audio.determinism`. Use
  `write_json_atomic`, and record decisions in the report's `decisions` list rather than
  inventing a second place for them.

## Known risks and open questions

- Depends on **OQ-001, OQ-003, OQ-004, OQ-005, OQ-007, OQ-011**. Every guess about
  DJI's real layout must be behind a named strategy in the chain and tagged with
  its `OQ-` ID so H1 can settle it cheaply.
- **Start acquiring the H1 fixture during this milestone.** Do not wait.
- Byte-identical output is easy to lose to dict ordering, path separators, and
  float formatting. Build the canonical writer first and route everything through it.

## Working plan

_Scratch section, written during the start phase and replaced by the Closeout at the
end. It records what was intended, so the close phase can say what actually happened._

**Amended after independent review** — `docs/plan/reviews/M1-plan-20260802-1046.md`
records every finding, which were accepted, and the two that were rejected and why.
Three were invariant-level: a recovery offset is not a since-midnight sample count, the
cache identity omitted most of the code whose output it caches, and INV-13 was argued
rather than proved on failure paths.

### Evidence gathered before planning

Three assumptions were tested against the real FFmpeg 8.0 in the flake shell, using
hand-built WAV files, because they would each have forced a different design:

1. **`ffprobe` surfaces BWF `time_reference` as a `format.tags` entry.** A file with a
   602-byte `bext` chunk reports `"time_reference": "3283200000"` alongside `date`,
   `creation_time`, `comment`, and `encoded_by`. The preferred strategy has a real input.
2. **`ffprobe` surfaces a `timecode` tag the same way**, so the fallback strategy does too.
3. **`ffprobe` does not surface `iXML` or a private `DJIm` chunk at all** — neither
   appears anywhere in `-show_format -show_streams` output. This is the whole
   justification for the generic RIFF walk, and it means OQ-005 cannot be answered by
   FFprobe no matter what DJI writes.

An RF64 file with a `ds64` chunk and a `0xFFFFFFFF` `data` size also parses correctly,
so the walker's 64-bit path is exercisable without a >4 GiB fixture.

### Files, in build order

Everything after step 1 is tested against step 1, which is why it goes first.

1. **`src/dnd_audio/fixtures/`** — the synthetic fixture generator. Shipped rather than
   test-local so M2–M5 and `scripts/make_fixture.py` can all import one implementation.
   - `wav.py` — RIFF/RF64 writer: `fmt `, `data`, `bext` (v2, with `time_reference`),
     `iXML`, a DJI-shaped private chunk, and an RF64/`ds64` mode.
   - `synth.py` — deterministic signal synthesis: speech-shaped noise, a three-clap
     transient, and an attenuated delayed copy for bleed. Seeded `default_rng`.
   - `session.py` — a declarative spec (`FixtureSession`/`FixtureTrack`/`FixtureChunk`)
     rendered to a session directory plus a `FixtureTruth` record of ground truth.
     `FixtureTruth` also carries the two downstream contracts the spec's fixture recipe
     names and M1 does not itself use: the activity spans `ScriptedActivityDetector`
     consumes and the transcript script `ScriptedTranscriber` consumes. Cheap here,
     expensive to retrofit in M3/M4.
2. **`src/dnd_audio/inspection/riff.py`** — generic RIFF/RF64 chunk inventory. `offset`
   is the chunk **header's** offset. A chunk's SHA-256 covers its **complete** payload,
   streamed; the bounded cap limits how much text is *retained*, never what is hashed,
   because a prefix hash presented as a chunk hash would be a lie. The audio `data`
   payload is inventoried but not hashed — the file hash already covers those bytes and
   INV-07 forbids a second full read. One level of `LIST` recursion. `ds64` table
   entries beyond `data` are resolved when present; a sentinel size with no table entry
   is a recorded warning. A truncated chunk is **recorded and the walk stops** — not
   "recovered", because silent recovery from a bad length turns corruption into
   plausible but false metadata. Never decodes audio.
3. **`src/dnd_audio/inspection/naming.py`** — DJI filename grammar, `orig`/`edit`
   variant, and sequence hints. Every field is a *hint*; none is identity (INV-11).
4. **`src/dnd_audio/inspection/probe.py`** — `ffprobe` invocation with the session
   directory as cwd and a session-relative `-i` path, plus `ffmpeg`/`ffprobe` version
   capture for the cache identity. The sidecar is FFprobe's **stdout bytes**, written
   with `write_atomic` and persisted *before* any project parsing, so it survives a
   parse failure. `write_json_atomic` would reserialize it and it would no longer be
   verbatim.
5. **`src/dnd_audio/timecode.py`** — extended with `frames_since_midnight()` and
   `samples_since_midnight()` returning exact `Fraction`s. See "Charter amendments".
6. **`src/dnd_audio/inspection/starttime.py`** — the timecode strategy chain.
7. **`src/dnd_audio/inspection/discovery.py`** — directory scan, selection rules,
   duplicate detection, roster and `active_tracks` rules.
8. **`src/dnd_audio/artifacts/manifest.py`** — extended to carry the capture. Schema
   version stays 1: it is provisional until this milestone closes (ADR-0005). Two
   contract-level shapes are settled before the leaf fields are: the tagged timing
   union above, and a track-independent **unassigned sources** list. Without the
   latter, a file in an unconfigured directory must either be dropped — losing the
   per-file capture the gate requires — or attached to a track, which is the INV-11
   violation the whole roster design exists to prevent.
9. **`src/dnd_audio/inspection/cache.py`** — per-source cache identity and store. One
   `INSPECTION_SEMANTICS_VERSION` covers probe parsing, naming, the RIFF walk, and
   start-time extraction, so a fix in any of them invalidates the entries it could have
   poisoned. Entries publish atomically, and only after the source's hash still matches
   what was inspected.
10. **`src/dnd_audio/inspection/runner.py`** + `cli.py` — orchestration, report
    contribution, INV-01 verification.
11. **`scripts/make_fixture.py`** — materializes the canonical fixture on disk.
12. Docs: ADRs, `OPEN-QUESTIONS.md` updates, downstream charter edits.

### The timecode strategy chain

Ordered, named, each tagged with the open question it rests on. First match wins; the
chosen strategy, every strategy that declined, and every assumption are recorded.

| Order | Strategy | Evidence it needs | Evidence it yields | Tagged |
| ----- | -------- | ----------------- | ------------------ | ------ |
| 1 | `recovery_override_offset` | `start_offset_samples` | signed integer samples at 48 kHz, **relative to session zero** | — |
| 2 | `recovery_override_timecode` | `start_timecode` | parsed timecode + exact frame index + rate | — |
| 3 | `bwf_time_reference` | `format.tags.time_reference` | integer samples since midnight **at the file's own rate** | OQ-001, OQ-004 |
| 4 | `timecode_tag` | `format.tags.timecode` + configured rate | parsed timecode + exact frame index + rate | OQ-001 |
| — | *none matched* | — | fatal `TimecodeError` naming the file and the override that would fix it (INV-12) | — |

**The three evidence shapes are a tagged union and are never collapsed into one
scalar.** This is the review's first finding and it matters: a recovery offset is signed,
at 48 kHz, and measured from session zero, while a BWF reference is unsigned, at the
file's rate, and measured from midnight. Forcing them together yields a wrong timestamp
whenever session zero is not midnight, and throws away the distinction M2 needs to
reconcile absolute time-of-day evidence with session-relative evidence. Recorded as
ADR-0006.

A `recording_date`-only override supplies no timing. It supplies the calendar day and
lets a later strategy supply the time.

M1 preserves evidence; it does not rasterize. A 29.97 fps frame is `8008/5` samples at
48 kHz, so "the integer sample position" is not a property of the evidence — it is a
property of a quantization rule, and that rule belongs with session zero and rollover in
M2. Where a frame index is exact it is stored as an integer frame count plus its rational
rate, which loses nothing (INV-04).

Filesystem `mtime` is never consulted. The proof is behavioral — perturbing only mtimes
changes neither the timing decision nor a manifest byte, and a source with no timing
evidence but a plausible mtime is still fatal (INV-12). A source grep stays as a cheap
tripwire, but it is not the proof: an alias or a moved helper defeats it.

### Completion-gate criteria mapped to their proof

| # | Criterion | Proof |
| - | --------- | ----- |
| 1 | Fixture generator: six tracks, multiple chunks, differing offsets, real gap, shared clap, quiet bleed, two-speaker interval | `tests/test_fixtures.py::TestCanonicalSession` — one test per property, asserted against `FixtureTruth` and against the written files; `test_no_audio_binaries_are_committed` greps the tree for `.wav` |
| 2 | Per-file capture (path, SHA-256, size, duration, `duration_ts`, time base, exact sample count, codec, sample format, rate, channels) | `tests/test_probe.py::TestCapture` field-by-field against a fixture of known construction; `test_exact_sample_count_matches_the_data_chunk` cross-checks `duration_ts` against the RIFF `data` size (evidence for OQ-011) |
| 3 | Raw `ffprobe` JSON retained verbatim, content-hash-addressed | `tests/test_probe.py::test_sidecar_bytes_are_byte_identical_to_ffprobe_stdout` and `test_sidecar_path_is_its_own_content_hash` |
| 4 | Generic RIFF/RF64 inventory not dependent on FFprobe | `tests/test_riff.py` — `test_private_chunk_is_found_though_ffprobe_never_reports_it` runs both and asserts the asymmetry; plus RF64/`ds64`, `LIST` recursion, odd-size padding, truncated-chunk recovery, text-retained-vs-hashed boundary |
| 5 | Timecode strategy chain, assumptions recorded, fatal when nothing reliable exists | `tests/test_starttime.py` — one test per strategy, one for each declining, `test_no_evidence_is_fatal_and_names_the_fix` (INV-12), `test_mtime_is_never_read` (source grep) |
| 6 | Timecode evidence is exact for non-drop, fractional, drop-frame, and override cases | `tests/test_timecode.py::TestFrameIndex` — exact integer frame index and rational rate at 30F, 25F, 23.98F, 29.97F, 29.97DF, including a drop-frame index that skips labels |
| 7 | Selection rules: `orig` preferred, `edit` associated, processed-only fatal unless allowed, duplicates by hash, warnings | `tests/test_discovery.py::TestSelection` — one case per rule, each asserting the manifest role *and* the stable reason code; `test_a_file_named_for_another_transmitter_warns` covers the spec's more-than-one-transmitter warning; `test_an_unrecognized_filename_is_still_inspected` proves the grammar is not an inclusion filter (OQ-003) |
| 8 | Roster rules incl. INV-11 | `tests/test_discovery.py::TestRoster` — `auto` activates only directories with a usable original; explicit list makes a missing track fatal; `test_an_unconfigured_directory_is_never_attributed_to_a_speaker` asserts its files land in the manifest's *unassigned* list with no `track_id` |
| 9 | `recovery.source_time_overrides` honored, SHA-256 verified, recorded prominently | `tests/test_starttime.py::TestOverrides` + `tests/test_inspect_run.py::test_an_override_appears_in_both_manifest_and_report`; `test_an_override_matching_no_source_is_fatal` |
| 10 | Byte-identical `manifest.json` on rerun (INV-02); no wall-clock (INV-03) | `tests/test_inspect_run.py::test_two_runs_are_byte_identical`, `::test_a_relocated_session_produces_the_same_manifest`, `::test_injected_clock_hostname_and_cache_state_change_no_byte` (the INV-03 proof), `::test_touching_every_source_changes_no_byte` |
| 11 | Cache identity incl. tool versions; a bump forces re-inspection (INV-08) | `tests/test_inspect_cache.py` — a second run probes nothing; each of source hash / config hash / **ffmpeg** version / **ffprobe** version / ffprobe argv / `INSPECTION_SEMANTICS_VERSION` / manifest-schema version changed independently forces a miss, counted with a probe spy; `test_an_interrupted_entry_is_never_a_hit` injects a crash between write and rename |
| 12 | Raw source hashes unchanged before and after (INV-01) | `tests/test_inspect_run.py::test_raw_is_byte_identical_after_a_full_run` snapshots hash + size + mtime of **every** file under `raw/`, not only selected sources; `test_output_paths_inside_raw_are_fatal` |
| 13 | The report shows the roster and the provenance the spec asks for (INV-13) | `tests/test_inspect_report.py` — the serialized report validates against the checked-in schema and carries known/observed/per-track counts, missing/empty/extra directories, the config hash, FFmpeg and FFprobe versions, and the exact FFprobe argv |
| 14 | A failed inspection still writes a report and exits nonzero (INV-13) | `tests/test_inspect_report.py::TestFailurePaths` — five CLI-level cases (no timing evidence, processed-only without permission, required track missing, override hash mismatch, FFprobe failure); each asserts the report exists, `inspect` is `failed` with a structured error, the other five stages are `skipped` with reasons, and the exit code is nonzero |

### Invariants this could plausibly violate, and what stops it

- **INV-01** — the only writes are to `work/` and `output/`; the runner re-hashes every
  source after the run and fails on a mismatch, and refuses a session whose `work/` or
  `output/` resolves inside `raw/`.
- **INV-02** — every artifact goes through `write_json_atomic`. The subtle hazard is
  `ffprobe`'s `format.filename`: probing by absolute path would make the sidecar hash
  depend on where the session lives. Probing with `cwd=session_dir` and a relative `-i`
  path removes it, and a relocation test proves it.
- **INV-03** — proved behaviorally: an injected clock, hostname, and cache hit/miss
  state must not change a manifest byte. The structural "no time-typed field" check
  stays as a tripwire, but it is not the proof — a hostname serializes as a plain
  string and would sail through it.
- **INV-04** — timing evidence is integers and exact rationals end to end; no float
  appears in the start-time path, and no evidence is rounded to reach a common unit.
- **INV-07** — hashing streams (already), the `data` payload is never read, and
  retained chunk text is capped.
- **INV-08** — see criterion 11. The failure this guards against is a fix in
  `starttime.py` still serving the answer the bug produced.
- **INV-11** — `TrackConfig` already ties the directory to the identity; `naming.py`
  returns only hints and there is no code path from a `TX##` label to a `track_id`. A
  file in an unconfigured directory is captured in the unassigned list, never attached
  to a speaker.
- **INV-12** — see the strategy chain, proved behaviorally rather than by grep.
- **INV-13** — proved by running the CLI through five real failures, not by arguing
  that the builder was called. See criterion 14.

### Charter amendments proposed by this plan

1. **The M1/M2 timing boundary is drawn at evidence, and the `Spec sections` line is
   corrected.** The charter claimed spec acceptance criterion 2, which includes midnight
   rollover and final integer sample positions — both of which this milestone defers.
   The boundary that actually holds: M1 extracts and preserves *typed timing evidence*
   exactly; M2 normalizes dates, infers rollover, chooses session zero, and rasterizes
   onto the 48 kHz grid under a documented quantization rule. `timecode.py` gains an
   exact frame-index helper — that is arithmetic on a timecode string, not placement on
   a timeline — and its docstring is amended to say so. Recorded as ADR-0006 and
   propagated to M2's charter.
2. **The inspection cache is keyed on the whole `config_hash()`**, as M0's closeout
   instructed, which means an unrelated `mix.integrated_lufs` edit re-probes every
   source. That is seconds of `ffprobe` on a few dozen files and is the safe direction
   to be wrong in; recorded so it is a known cost rather than a surprise.
3. **A `source_time_overrides` key matching no discovered file is fatal.** ADR-0005
   already rejected an information-free override on the grounds that a silently ignored
   one is the failure the mechanism exists to prevent; an override aimed at a mistyped
   path is the same failure. Recorded as an ADR.
4. **The report gains a typed `RosterSummary` section.** The gate requires the report to
   show known/observed/per-track counts and to list missing, empty, and extra
   directories. Encoding counts as free-text decisions would satisfy the letter and
   leave every consumer parsing prose. The report schema is provisional until M5, so a
   typed section now is cheaper than a migration later.

### Deliberately not done

From the charter's non-goals: no chunk ordering, no gap or overlap reasoning, no session
timeline, no decoding beyond what the container and the RIFF `data` size already state.
Adding to that: no drift analysis, no VAD, no automatic recovery from a missing
timecode, and no DJI-private chunk *parser* — the walker records that a private chunk
exists and hashes it, which is what makes H1 cheap, but inventing a layout for it now is
exactly what the spec forbids.
