# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M5 — Automix (not started)
- **Branch:** `main` (M4 merged)
- **Last closed milestone:** M4 — Fake transcript
- **Gate status at HEAD:** passes, zero skips (8 checks, 1768 tests)
- **Blocked on:** nothing for M5. **H1 is still the oldest outstanding item in the
  project** and now gates six open questions (OQ-001, OQ-002, OQ-003, OQ-004, OQ-007,
  OQ-015). It needs a physical recording session, not code. Every DJI layout assumption
  M1 and M2 made sits behind a named strategy or a cited constant tagged with its `OQ-`
  ID, so settling them is cheap once a real file exists — but they stay unsettled until
  one does.

  M3 added a second kind of waiting, and it is not H1's kind. **OQ-017** — what separates
  real speech from lav bleed at a real table — needs H2 or a first real session, because a
  two-minute metadata fixture cannot tune a bleed threshold. Every VAD, bleed, and scoring
  default cites it, and the pipeline already records the numbers that answer it, so this is
  reading one session's graph rather than running an experiment. Nothing is blocked on it:
  the thresholds work on synthetic audio and the gate is conservative by construction.

  M4 added a third, and it is M6b's rather than a room's. **OQ-018** — what Qwen3-ASR and its
  aligner need at a request boundary — covers padding, timestamp stability across two
  overlapping requests, whether a low-energy split beats the midpoint, the retry budget, and
  the text-similarity thresholds. M6b's smoke test settles the first three directly; the last
  needs a real session, or one utterance genuinely heard on two transmitters. Nothing is
  blocked: M4 is correct under whatever the configured values are, and only the *defaults*
  are guesses. `rg 'OQ-018'` finds all twelve sites at once.

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `67b70ed` |
| M1  | Inspection                 | closed      | `fd16931` |
| M2  | Timeline                   | closed      | `f33ad6d` |
| M3  | Activity                   | closed      | `38bc989` |
| M4  | Fake transcript            | closed      | `e8ff55d` |
| M5  | Automix                    | not started | —         |
| M6a | ROCm environment           | not started | —         |
| M6b | Qwen adapter               | not started | —         |
| H1  | Hardware fixture (2 min)   | not started | —         |
| H2  | Drift soak / first session | not started | —         |
| M7  | Archival (sketch)          | sketch      | —         |

Status values: `not started` → `in progress` → `verified` → `closed`.
`blocked` is also valid; say what on. `sketch` means a charter exists to hold the
idea but the work is deliberately unplanned.

## What works end to end

`uv run dnd-audio transcribe /path/to/session --fake-models` — the whole left branch of the
spec's stage DAG: inspection, the timeline, who was speaking, what they said, and both
transcript deliverables.

It does everything `activity` does, then plans ASR requests from the graph's **retained**
candidates only — merging adjacent regions, padding each core, and capping the *padded*
waveform at `max_segment_s` — submits one window at a time (INV-07), resolves a truncated
response by splitting the unpadded core at its quietest interior frame within a global
submission budget, assigns each word to the ownership interval containing its start, collapses
duplicates on overlap **and** similar text **and** the graph's own acoustic evidence, verifies
INV-01 a second time, commits the ASR cache, and writes `work/transcript-records.json`,
`output/transcript.json`, `output/transcript.md` and one report covering five stages.

On the canonical fixture: 4 segments across 4 speakers, 0 collapsed, 2 marked as overlap, all
three deterministic artifacts byte-identical on rerun, 29 ASR cache hits and 0 misses warm.
Alice's line bleeds into four tracks and the scripted ASR is told to transcribe it there;
every copy is gone before a word is submitted, because M3's gate suppressed the candidate.

`uv run dnd-audio render /path/to/session` regenerates both deliverables from the records
alone — proved by deleting the graph, the timeline and the whole cache tree first, with a spy
asserting no model is constructed. Absent records exit nonzero naming
`transcript_records_missing`, and still write a report.

Without `--fake-models`, `transcribe` raises the `DEFERRED: M6b` `NotImplementedError` naming
the missing adapter rather than writing a report that says the session is broken (ADR-0005).

`uv run dnd-audio activity /path/to/session` — inspection, the timeline, then who was
speaking.

It does everything `ingest` does, then runs a VAD per track over the cached 16 kHz
derivative behind an `ActivityDetector` protocol, measures every overlapping candidate pair
with a lag-tolerant normalized speech-band cross-correlation, scores each candidate on four
terms (track-relative level, VAD confidence, cross-track dominance, correlation as
*independence*), applies a bleed gate that suppresses only on margin **and** correlation
**and** a track-relative veto, verifies INV-01, commits four caches at one moment, and
writes `work/activity.json` plus one `output/ingest-report.json` covering three stages.

On the canonical fixture through the real pinned model: 6/6 tracks, 0 candidates — the
correct answer, because the fixture's speech is synthetic noise and INV-10 forbids expecting
a learned release to fire on audio no human made. Attribution is proved against the
deterministic scripted detector over the fixture's declared truth; the real model is
exercised once under `host_smoke` on claims true of any release.

`uv run dnd-audio ingest /path/to/session` — inspection, then the timeline.

It discovers and hashes the session's sources every run (served warm from M1's content
cache), rasterizes each source's timing evidence onto the 48 kHz grid with exact rational
arithmetic, decides where session zero is and which day each chunk belongs to, lays out
each track's chunks with real gaps preserved as explicit silence, builds cached 16 kHz
derivatives through one checked-in FIR, and writes `work/timeline.json` plus
`output/ingest-report.json`. On the canonical fixture: 6/6 tracks, 10.500 s aligned
(504000 samples), byte-identical on rerun, 18 cache hits and 0 misses.

The 48 kHz working path is the **segment map**, not files — `TrackReader` serves bounded
windows over it, and nothing in the pipeline may depend on `--materialize-48k` having run.

`uv run dnd-audio inspect /path/to/session` still does the first half alone, writing the
manifest and the content-hash-addressed sidecars holding exactly the bytes FFprobe wrote.

`uv run dnd-audio models fetch` downloads the Silero VAD model, pinned by release `v6.2.1`,
commit `7e30209a`, and sha256, verified before the file is moved into place. It is the only
command permitted to touch the network (INV-06). `doctor` reports its availability alongside
tool versions, writable paths, and free disk.

`uv run python scripts/make_fixture.py <dir>` materializes the canonical six-transmitter
synthetic session, and nine variants exercise one deviation each: a 44.1 kHz source, a
track whose chunks disagree about their rate, a material overlap, no configured origin,
midnight rollover, 29.97 drop-frame, drift, bleed delayed 25 ms, and two genuine
simultaneous speakers at unequal levels each carrying the other's bleed. No audio binaries
are in the repository.

`transcribe`, `mix`, `render`, and `process` remain registered stubs that exit 3 naming the
milestone they land in.

Underneath, from M0: validated `session.yaml` models, checked-in JSON Schema artifacts
with a drift test, exact rational frame rates, model seams with scripted fakes, a report
writer that cannot lose a stage, and a test suite that is provably offline.

## Next smallest step

Begin M5 — Automix. It depends on M3 only, never on M4: the mix must produce identical samples
whether or not ASR ran, and the graph M4 consumed is unchanged by anything M4 decided. Start
with the gain envelopes, because the envelope-level assertions are the real gate and the
loudness work is meaningless without them — a mix that picks the wrong speaker passes every
loudness test there is. (Claude Code: `/ms-start 5`.)

Read M5's new "What M4 already provides" section first. It is not about data; it is three
runner patterns and one obligation. The obligation:
`tests/test_raw_guard.py::TestCleanupNeverWritesIntoRaw` needs a `mix` parameter the moment
`run_mix` exists, and the reason it is parametrized over every composed command is that M2, M3
and M4 each tested only the runner that milestone added, and all three carried the same INV-01
bug for five milestones.

**Real DJI metadata has still not been validated.** Acquiring the H1 fixture is the oldest
outstanding item in the project. M2 added OQ-015 to what it must settle; M3 added **OQ-017**
and M4 added **OQ-018**, neither of which H1 can answer — a two-minute metadata fixture cannot
tune a bleed threshold or a text-similarity threshold. OQ-017 waits for H2 or a first real
session; OQ-018's first three parts are M6b's smoke test.
