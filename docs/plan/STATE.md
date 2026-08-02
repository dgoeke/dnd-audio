# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M1 — Inspection (not started)
- **Branch:** `main`
- **Last closed milestone:** M0 — Foundation
- **Gate status at HEAD:** passes, zero skips (8 checks, 311 tests)
- **Blocked on:** nothing. M1 can start immediately. The H1 hardware fixture is not a
  blocker for starting M1, but acquiring it should begin now — every DJI layout guess
  M1 makes stays unsettled until it exists.

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `5675458` |
| M1  | Inspection                 | not started | —         |
| M2  | Timeline                   | not started | —         |
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

No audio is processed yet — M0 was the foundation and its rails, by design.

`direnv allow` then `cd` gives a shell with Python 3.12, `uv`, FFmpeg, and SoX out of
`/nix/store`; `nix run .#fhs -- -c '<cmd>'` runs inside the FHS sandbox held for M6a.
`uv run dnd-audio doctor` reports real tool versions, writable paths, and free disk.
Every other command is registered and exits 3 naming the milestone it lands in.

Underneath: validated `session.yaml` models, checked-in JSON Schema artifacts with a
drift test, exact rational frame rates, model seams with scripted fakes, a report writer
that cannot lose a stage, and a test suite that is provably offline.

## Next smallest step

Begin M1 — the synthetic fixture generator first, since everything after it is tested
against it. (Claude Code: `/ms-start 1`.)

Read M1's new "What M0 already provides" section before writing code: four contracts
land in M0 that M1 inherits.

**Real DJI metadata has not been validated.** Start acquiring the H1 fixture now; every
layout assumption M1 makes must sit behind a named strategy tagged with its `OQ-` ID.
