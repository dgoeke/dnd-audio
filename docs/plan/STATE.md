# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M0 — Foundation (verified; awaiting close)
- **Branch:** `milestone/M0-foundation`
- **Last closed milestone:** none
- **Gate status at HEAD:** passes, zero skips (8 checks, 311 tests)
- **Blocked on:** nothing

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | verified    | —         |
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

Nothing processes audio yet. The repository contains the spec, this planning scaffold,
and the repo-local Nix development environment: `direnv allow` yields a shell with
Python 3.12, `uv`, FFmpeg, and SoX resolved out of `/nix/store`, and `nix develop .#fhs`
opens the FHS sandbox held for M6a.

## Next smallest step

Continue M0 from inside the activated shell, starting at `pyproject.toml` + `uv.lock`.
The working plan is recorded in `docs/plan/milestones/M0-foundation.md`; everything after
step 1 of its Phase B list is outstanding.
