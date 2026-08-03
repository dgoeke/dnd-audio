# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M4 — Fake transcript (verified, not yet closed)
- **Branch:** `milestone/M4-fake-transcript`
- **Last closed milestone:** M3 — Activity
- **Gate status at HEAD:** passes, zero skips (8 checks, 1768 tests)
- **Blocked on:** nothing for M4. **H1 is still the oldest outstanding item in the
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

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `67b70ed` |
| M1  | Inspection                 | closed      | `fd16931` |
| M2  | Timeline                   | closed      | `f33ad6d` |
| M3  | Activity                   | closed      | `38bc989` |
| M4  | Fake transcript            | verified    | —         |
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

Begin M4 — the transcript branch end to end on fake ASR, with no Qwen, no GPU, and no
weights. Start with segment-request construction from retained activity candidates, because
it is the part the rest of the milestone hangs off and the part the frozen graph most
directly constrains. (Claude Code: `/ms-start 4`.)

Read M4's new "What M3 already provides" section first. Two things there will otherwise cost
real time: `ambiguous` does **not** mean "uncertain detection" — it means the numbers said
bleed and the veto overrode them, which makes those candidates the ones duplicate collapse
should look hardest at — and `test_activity_artifact.py::TestTheConsumerReads` is a worked
example of M4's own access pattern, written before M4 existed.

**Real DJI metadata has still not been validated.** Acquiring the H1 fixture is the oldest
outstanding item in the project; M2 added OQ-015 to what it must settle. M3 added **OQ-017**,
which H1 cannot answer — a two-minute metadata fixture cannot tune a bleed threshold, so that
one waits for H2 or a first real session.
