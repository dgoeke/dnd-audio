# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M3 — Activity (not started)
- **Branch:** `main`
- **Last closed milestone:** M2 — Timeline
- **Gate status at HEAD:** passes, zero skips (8 checks, 923 tests)
- **Blocked on:** nothing for M3. **H1 is still the oldest outstanding item in the
  project** and now gates six open questions (OQ-001, OQ-002, OQ-003, OQ-004, OQ-007,
  OQ-015). It needs a physical recording session, not code. Every DJI layout assumption
  M1 and M2 made sits behind a named strategy or a cited constant tagged with its `OQ-`
  ID, so settling them is cheap once a real file exists — but they stay unsettled until
  one does.

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `67b70ed` |
| M1  | Inspection                 | closed      | `fd16931` |
| M2  | Timeline                   | closed      | `f33ad6d` |
| M3  | Activity                   | not started | —         |
| M4  | Fake transcript            | not started | —         |
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

`uv run python scripts/make_fixture.py <dir>` materializes the canonical six-transmitter
synthetic session, and seven variants exercise one deviation each: a 44.1 kHz source, a
track whose chunks disagree about their rate, a material overlap, no configured origin,
midnight rollover, 29.97 drop-frame, and drift. No audio binaries are in the repository.

`doctor` still reports real tool versions, writable paths, and free disk. `transcribe`,
`mix`, `render`, `process`, and `models fetch` remain registered stubs that exit 3 naming
the milestone they land in.

Underneath, from M0: validated `session.yaml` models, checked-in JSON Schema artifacts
with a drift test, exact rational frame rates, model seams with scripted fakes, a report
writer that cannot lose a stage, and a test suite that is provably offline.

## Next smallest step

Begin M3 — activity. Start with the `ActivityDetector` protocol and the deterministic fake
over the canonical fixture's declared truth, before any Silero pinning: the fixture already
carries the fake-VAD contract, the 16 kHz derivatives it consumes are cached and
byte-stable, and getting the graph's shape right is the part M4 and M5 both inherit.
(Claude Code: `/ms-start 3`.)

Read M3's new "What M2 already provides" section first. Two items there will otherwise cost
real time: the 48↔16 kHz interval mapping **floors its start and ceils its end** (rounding
both the same way shrinks a speech region by up to two samples), and the lag-tolerant
normalized cross-correlation M3's bleed gate needs already exists as
`timeline.syncqa.measure_lag`.

**Real DJI metadata has still not been validated.** Acquiring the H1 fixture is the oldest
outstanding item in the project, and M2 added OQ-015 to what it must settle.
