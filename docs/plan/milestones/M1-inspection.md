# M1 — Inspection and immutable ingest manifest

**Status:** closed
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

- [x] Synthetic fixture generator produces six virtual tracks with multiple chunks
      each, differing start offsets, a real gap, a shared clap, solo activity that
      bleeds quietly into other tracks, and one two-speaker interval. No audio
      binaries are checked into the repository.
- [x] Per-file capture: relative path, SHA-256, size, displayed duration,
      `duration_ts`, time base, exact PCM sample count where available, codec,
      sample format/bit depth, sample rate, channel count.
- [x] Complete raw `ffprobe -show_format -show_streams` JSON retained verbatim
      under a content-hash-addressed path beside the manifest, before any
      project-specific parsing.
- [x] Generic RIFF/RF64 chunk inventory (ID, offset, size) that does not depend on
      `ffprobe` surfacing unknown chunks; bounded textual payloads retained,
      larger ones hashed.
- [x] Timecode strategy chain: BWF `time_reference` preferred as an integer sample
      count, timecode tag plus configured frame rate as fallback, every assumption
      and fallback recorded. Fails with an actionable diagnostic when nothing
      reliable exists (INV-12).
- [x] Selection rules: `orig` preferred, `edit` associated but ignored, processed-only
      is fatal unless `allow_processed_audio`, duplicates detected by content hash,
      warnings for unexpected formats and sequence discontinuities.
- [x] Roster rules: `active_tracks: auto` derives active participants from
      configured directories containing a usable original; explicit lists make
      every listed track required; an unconfigured directory is never attributed
      to a speaker (INV-11); the report shows known/observed/per-track counts and
      lists missing, empty, and extra directories.
- [x] `recovery.source_time_overrides` honored, keyed by source-relative path, with
      optional SHA-256 verification, exactly one of `start_timecode` or
      `start_offset_samples`, and prominent recording in manifest and report.
- [x] Inspecting unchanged input twice yields byte-identical `manifest.json`
      (INV-02); no wall-clock or cache telemetry inside it (INV-03).
- [x] Inspection cache identity includes source hash, exact FFmpeg and FFprobe
      versions, the ffprobe command/options, and the RIFF-parser plus
      manifest-schema versions. A simulated tool-version bump forces re-inspection
      (INV-08).
- [x] Raw source hashes unchanged before and after a run (INV-01).

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

---

## Closeout

### What works end to end

`uv run dnd-audio inspect /path/to/session` discovers a session's sources, captures
everything FFprobe and a generic RIFF walk can tell us about each one, applies the
selection and roster rules, extracts timing evidence through a named strategy chain, and
writes `work/manifest.json` plus `output/ingest-report.json`.

```
$ python scripts/make_fixture.py /tmp/session-demo
wrote 6 tracks, 12 chunks, 7.0 MB
  session zero   3283200000 samples since midnight
  gap            tx-c: samples 240000-384000

$ uv run dnd-audio inspect /tmp/session-demo
  inspected 12 source(s) across 6/6 active track(s)
  manifest  /tmp/session-demo/work/manifest.json
  report    /tmp/session-demo/output/ingest-report.json
$ echo $?
0
```

A second run is byte-identical and probes nothing (12 cache hits, 0 misses). Beside the
manifest sit twelve content-hash-addressed sidecars holding exactly the bytes FFprobe
wrote.

The synthetic fixture generator lands here too, and everything from M2 onward is meant to
be tested against it: six transmitters, two chunks each, six distinct start offsets, one
real three-second gap, a shared clap all six are recording for, `tx-a`'s solo bleeding
quietly into the four tracks that were live, and a `tx-d`/`tx-e` overlap. It carries the
fake-VAD and fake-ASR contracts the spec's fixture recipe names, which M3 and M4 consume.

### Tests and commands run, with results

`./scripts/gate.sh` — **8 checks, zero skips, 551 tests** (225 of them M1's):

```
== gate summary ==
  pass  system dependencies      pass  pytest (offline, cpu)
  pass  ruff check               pass  lock is current
  pass  ruff format              pass  placeholder scan
  pass  type check               pass  plan consistency

GATE PASSED
```

Per area: fixtures 23, RIFF 21, naming 20, probe 24, start-time 28, timecode 58,
discovery 33, cache 21, end-to-end run 36, report 19.

Every completion-gate criterion was proved with executed output during the verify phase.
The proofs that carry the most weight:

| Criterion | Proof |
| --------- | ----- |
| RIFF inventory independent of FFprobe | `test_riff.py::TestIndependenceFromFfprobe` runs **both tools over the same bytes** and asserts only one sees the private chunk |
| Exact PCM sample count | `test_probe.py::TestExactSampleCount` — all twelve fixture files agree between the `data` chunk and `duration_ts` |
| Byte-identical manifest (INV-02/03) | Second run with a different clock *and* a warm cache; a session copied to a deeper path; every source's mtime set to the epoch |
| `raw/` untouched (INV-01) | Full-tree hash snapshot including a non-audio file; plus a patched probe that corrupts a source mid-run, proving the check can fail |
| Cache identity (INV-08) | Each component varied independently, *and* the runner proved to consult the cache — a perfect key nothing reads would pass the first half alone |
| Report on failure (INV-13) | Seven real failure paths driven through the CLI |

### Decisions made (→ ADRs)

- **[ADR-0006](../decisions/0006-timing-evidence-is-a-tagged-union.md)** — timing evidence
  is a tagged union and M1 does not rasterize it. A BWF reference is unsigned samples
  since midnight at the file's own rate; a timecode is an exact frame index plus a
  rational rate; a recovery offset is **signed, 48 kHz, relative to session zero**. They
  share no coordinate system, and collapsing them into one integer is wrong for every
  session that does not start at midnight. Also corrects the charter's `Spec sections`
  line: M1 owns the *evidence* half of acceptance criterion 2, M2 owns the mapping.
- **[ADR-0007](../decisions/0007-an-override-that-matches-nothing-is-fatal.md)** — a
  `source_time_overrides` key matching no discovered source is fatal, as is a hash
  mismatch. Same reasoning ADR-0005 used to reject an information-free override, applied
  to one with a typo in its path.

### Assumptions made and open questions raised

No new open questions. Three existing ones gained evidence that changes what H1 has to do:

- **OQ-005 is half-answered, and about FFprobe rather than about DJI.** Against FFmpeg
  8.0, a file carrying both an `iXML` chunk and a four-byte-named private chunk produces
  `-show_format -show_streams` output mentioning **neither**. Whatever DJI writes, FFprobe
  will not surface it. That is the entire justification for the generic RIFF walk, and a
  test asserts the asymmetry so it stops being true loudly rather than quietly.
- **OQ-011's synthetic half is answered, and answering it changed the approach.** No decode
  is needed for either half: the RIFF `data` size over the block alignment is exact by
  construction for PCM. So the `data` chunk is the source and `duration_ts` is the
  cross-check, not the reverse. Their agreement is recorded per source, so H1 answers the
  real half by *reading a manifest* rather than running an experiment.
- **OQ-001** records what was built while waiting: both tags are reachable, each is a named
  strategy, and every manifest entry says which one fired and why the others declined.

Still open and still gating on hardware: **OQ-001, OQ-002, OQ-003, OQ-004, OQ-007**. Every
assumption about DJI's layout sits behind a named strategy tagged with its ID, so
`rg 'OQ-004'` finds every place that must change.

### Notes for future implementors

**The verify phase found thirteen defects. Read
`docs/plan/reviews/M1-code-20260802-1144.md` before trusting anything here.** The gate was
green and the milestone looked finished at the point where all thirteen were still
present. Three patterns are worth internalizing:

**An invariant check can be present, look right, and verify nothing.** `_raw_roots()`
dropped `"."` from the protected roots — reasonable, since every relative path is under
`"."` — and for a session configured as `input: "tx-a"` that made the snapshot empty, so
`_verify_unchanged()` compared two empty dicts and INV-01's proof passed unconditionally.
It went unnoticed because the fixture generator always writes `input: raw/<track_id>`, so
that shape had never been built and inspected. **If your milestone adds a verification,
write a test that makes the thing it verifies actually change.**

**Lexical path comparison is not a security boundary.** With `output -> raw/tx-a`, the
"output inside raw" check passed and the run wrote a report into a track's source
directory. Paths are now resolved before comparison. There is a second half people miss:
when the offending location is the *report's own*, writing the failure report there
commits the violation being reported. That one case writes nothing — **INV-01 outranks
INV-13 there**, because a report is regenerable and a source directory written into is
not.

**Two of the defects were tests whose docstrings claimed what the test did not check.**
`test_no_manifest_is_left_behind_by_a_failed_run` started from a clean directory, so it
could not see the stale-manifest case it described. `test_an_override_appears_in_both_
manifest_and_report` never opened the report. Both were written by the same person who
then reviewed them. Independent review caught both; self-review did not.

**Fixing things introduces things.** Making every candidate get probed — correct, the spec
says "for every candidate audio file" — meant one corrupt `.wav` in a directory nobody
configured failed the entire session. Found by re-running the original reproductions after
the fixes, not by the suite, which was green. **Re-run your reproductions, not just your
tests.**

Sharp edges and things that look wrong but are deliberate:

- **The cache key includes the source's path, not only its hash.** Two byte-identical
  files at two paths genuinely have different captures: FFprobe echoes the filename into
  its own output, and which recovery override applies is keyed by path.
- **FFprobe runs with the session directory as cwd and a relative path.** Probing by
  absolute path makes the sidecar's content hash — and therefore the manifest — depend on
  where the session lives, so copying a session would change its manifest.
- **The sidecar is FFprobe's stdout bytes via `write_atomic`, never `write_json_atomic`.**
  Canonical reserialization changes whitespace, key order, and the trailing newline; that
  is not "verbatim". It is persisted *before* parsing, so a capture this code cannot read
  still survives.
- **Duplicate ranking is `(is_unassigned, is_edit, path)` and every term earns its place.**
  Rank by path alone and a byte-identical `_edit` beats its `_orig` (`e` < `o`), or a
  stray `raw/aaa.wav` beats a real `raw/tx-a/TX01_…`. Both were real bugs; both surfaced
  two rules downstream with nothing pointing back.
- **Capture and timing are deliberately asymmetric.** A *selected* source must be
  inspectable and must have timing (INV-12). A source nothing will use is described as
  well as it can be and carries a warning. An asymmetry checked only on the lenient side
  is just leniency, so both directions are tested.
- **Configured track directories are scanned flat; unconfigured ones recursively.** The
  session contract describes the first layout; a directory nobody configured can be
  anything, and a file under `raw/` that never reaches the manifest is a session being
  described inaccurately.
- **The fixture generator's noise floor is seeded by timeline position, not by chunk.**
  That is what lets `test_chunking_does_not_change_a_sample` assert exact equality rather
  than hide a boundary error inside a tolerance.
- **`_EDIT_GAIN` exists because byte-identical `orig`/`edit` pairs turned every orig/edit
  test into an accidental duplicate test.**
- **Errors carry a `code`.** INV-13 wants structured errors; prose gets reworded. Add
  `default_code` to any new `DndAudioError` subclass.
- **`INSPECTION_SEMANTICS_VERSION` covers the whole package.** A fix in `starttime.py`
  must not keep serving the answer the bug produced. Bump it for any behaviour change;
  the cost of being wrong is re-probing a few dozen files.

**Do not add a second float path.** No float appears anywhere in `manifest.json` — checked
by walking the serialized document — and the only division in the timing path is integer
floor division.

### Deviations from this charter, and why

1. **The `Spec sections` line was corrected.** It claimed acceptance criterion 2, which
   includes midnight rollover and final integer sample positions — both deferred to M2.
   M1 owns the evidence half. ADR-0006.
2. **`timecode.frame_index()` landed in M1**, though `timecode.py` previously said
   converting a timecode to a position "is M2's". Counting frames is arithmetic on a
   timecode, not placement on a timeline; the docstring is amended.
3. **The report gained a typed `RosterSummary` and a `commands` list**, and
   `Provenance.config_hash` became optional. The gate requires the report to show the
   roster; the spec requires the exact commands; and a run that never resolved a
   configuration has no hash, so a string of sixty-four zeroes was a valid-looking lie.
4. **`scripts/make_fixture.py` was added** beyond the charter's list, at the user's
   request during the start phase, so the verify phase could run the real command against
   a real directory.
5. **The inspection cache is keyed on the whole `config_hash()`**, as M0's closeout
   instructed. An unrelated `mix.integrated_lufs` edit therefore re-probes every source.
   That is seconds of FFprobe and the safe direction to be wrong in; recorded so it is a
   known cost rather than a surprise.

### Downstream charters updated

- **M2** — records that per-file timing *evidence* already exists in typed form, and that
  M2 owns session zero, rollover, reconciliation, and the documented quantization rule
  that acceptance criterion 2 needs (a 29.97 fps frame is 8008/5 samples at 48 kHz).
  Also: a non-48 kHz source is a warning in M1 and must be fatal before timeline
  construction here.
- **M3/M4** — `FixtureTruth.activity_spans()` and `.transcript_script()` already provide
  the deterministic fake-VAD and fake-ASR contracts the spec's fixture recipe names.
- **`OPEN-QUESTIONS.md`** — OQ-001, OQ-005, and OQ-011 updated with evidence.
- **`INVARIANTS.md`** — INV-01 gains the note that its check compares *resolved* paths and
  that the report is not written when its own location would violate it.
- **`ROADMAP.md`** — unchanged; the dependency graph did not move.

### Next smallest step

Begin M2 — the timeline. Start with session zero and the rollover rules, because
everything else in that milestone hangs off where time zero is, and the evidence they
consume is already in the manifest in typed form.

Real DJI metadata has **still not been validated**. Acquiring the H1 fixture is now the
oldest outstanding item in the project and gates five open questions.
