# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M2 — Timeline (verified, closing)
- **Branch:** `main`
- **Last closed milestone:** M1 — Inspection
- **Gate status at HEAD:** passes, zero skips (8 checks, 923 tests)
- **Blocked on:** nothing for M2. **H1 is now the oldest outstanding item in the
  project** and gates five open questions (OQ-001, OQ-002, OQ-003, OQ-004, OQ-007). It
  needs a physical recording session, not code. Every DJI layout assumption M1 made sits
  behind a named strategy tagged with its `OQ-` ID, so settling them is cheap once a real
  file exists — but they stay unsettled until one does.

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `67b70ed` |
| M1  | Inspection                 | closed      | `fd16931` |
| M2  | Timeline                   | verified    | —         |
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

`uv run dnd-audio inspect /path/to/session` — the first stage that touches audio.

It discovers a session's sources, captures everything FFprobe and a generic RIFF walk can
tell us about each candidate, applies the selection and roster rules, extracts timing
evidence through a named strategy chain, and writes `work/manifest.json` plus
`output/ingest-report.json`. A second run is byte-identical and probes nothing. Beside the
manifest sit content-hash-addressed sidecars holding exactly the bytes FFprobe wrote.

`python scripts/make_fixture.py <dir>` materializes the six-transmitter synthetic session
everything from M2 onward is tested against — multiple chunks per track, a real gap, a
shared clap, quiet bleed, a two-speaker overlap, and the fake-VAD/fake-ASR contracts M3
and M4 consume. No audio binaries are in the repository.

`doctor` still reports real tool versions, writable paths, and free disk. `ingest`,
`transcribe`, `mix`, `render`, `process`, and `models fetch` remain registered stubs that
exit 3 naming the milestone they land in.

Underneath, from M0: validated `session.yaml` models, checked-in JSON Schema artifacts
with a drift test, exact rational frame rates, model seams with scripted fakes, a report
writer that cannot lose a stage, and a test suite that is provably offline.

## Next smallest step

Begin M2 — the timeline. Start with session zero and the rollover rules: everything else
in that milestone hangs off where time zero is, and the evidence it consumes is already in
the manifest in typed form. (Claude Code: `/ms-start 2`.)

Read M2's new "What M1 already provides" section first. Two items there are obligations
rather than conveniences: M2 owes acceptance criterion 2 a **documented quantization
rule** (a 29.97 fps frame is 8008/5 samples at 48 kHz, so an integer sample position is a
property of a rounding rule, not of the evidence), and a non-48 kHz source is a warning in
M1 that **must become fatal** before timeline construction.

**Real DJI metadata has still not been validated.** Acquiring the H1 fixture is the oldest
outstanding item in the project.
