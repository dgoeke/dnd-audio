# ADR-0002 — Nix flake + direnv for the dev environment; FHS held back for M6a

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** pre-M0

## Context

The spec's Target-host runtime section requires a repository-local Nix/FHS
development environment modeled on the host's existing ComfyUI NixOS module, so that
Python 3.12, `uv`, FFmpeg, SoX, and a build toolchain come from the repository
rather than from whatever happens to be on the host. It did not say how that
environment is expressed — flake or `shell.nix` — and said nothing about how a
developer or agent enters it.

That gap matters more than it looks. If entering the environment is a manual step,
every agent session, every `./scripts/gate.sh` run, and every `uv run` is one
forgotten command away from silently using the host's Python instead of the pinned
one. The gate would then be proving something about the host, not about the project.
On the current target host that is not hypothetical: its system `python3` is 3.13 —
the version `requires-python` excludes — and it has no `sox` installed.

The host already runs `direnv` (2.37.1) with `nix-direnv` enabled through Home
Manager, and other projects on it already use the standard pattern: `.envrc`
containing `use flake`, a `flake.nix` with `devShells.default`, and a committed
`flake.lock`.

The complication is FHS. `buildFHSEnv` is genuinely needed for M6a — AMD's
`gfx1151` Torch wheels pull a `rocm[libraries]` sdist that builds at install time
and expects a `/usr/lib` layout — but an FHS env is a sandbox you `exec` into, not a
set of variables you can source.

## Decision

The environment is a flake with two shells, and direnv loads only the first.

- `flake.nix` pins `nixpkgs` to `github:NixOS/nixpkgs/nixos-25.11`, the same channel
  the host's NixOS configuration tracks, so the repo and the host do not drift.
  `flake.lock` is committed.
- `devShells.default` is an ordinary `mkShell`: Python 3.12, `uv`, FFmpeg, SoX, the
  native libraries CPU wheels link against, and a build toolchain. Everything
  through M5 works here.
- `devShells.fhs` is a `buildFHSEnv` `.env`, entered explicitly with
  `nix develop .#fhs`. M0 proves it opens; M6a is the first milestone that needs it.
- `.envrc` contains `use flake` and is committed. `.direnv/` stays ignored.

The project's Python environment is uv's `.venv` in the repository, usable from
both shells. The host's ComfyUI venv is never read or written.

## Alternatives considered

- **Direnv-load the FHS shell directly** (`use flake .#fhs`). Rejected on evidence,
  not taste. `nix print-dev-env` — what `nix-direnv` sources — ends with
  `eval "${shellHook:-}"`, and `buildFHSEnv`'s `.env` sets a `shellHook` that ends
  in `exec "${cmd[@]}"` into a `bwrap` sandbox. Sourcing it would replace direnv's
  evaluation shell with an interactive bash. Verified against nixpkgs 25.11 by
  inspecting `nix print-dev-env` output for a minimal `buildFHSEnv .env`.
- **FHS only, no plain shell.** Rejected: it makes every M0–M5 session pay for a
  sandbox that only the ROCm work needs, and it forfeits direnv activation
  entirely, which is the property that keeps the gate honest.
- **`shell.nix` + `use nix`.** Rejected: no lock file, so "the environment" is
  whatever channel the host resolved that day. The whole point is reproducibility
  across cleared-context sessions.
- **`nix-ld` instead of FHS.** Rejected for M6a on the same reasoning the host's
  ComfyUI module records: pip's subprocess invocations and source builds are far
  more reliable inside a real FHS. Revisit only if OQ-008 forces it.
- **No repo environment; rely on host `uv` + `nix-shell -p`.** Rejected: the spec
  requires the repo-local environment, and it is the only way `doctor`'s
  system-dependency checks mean the same thing on every run.

## Consequences

- `cd` into the repository is the whole activation story for M0–M5. An agent that
  forgets is corrected by direnv rather than by a confusing test failure.
- Two shells is two things to keep working. M6a must not quietly move base
  dependencies into the FHS shell — if something in the base project stops building
  in `devShells.default`, that is a regression, not a reason to switch shells.
- A first `direnv allow` after a `flake.lock` bump downloads and evaluates, which is
  slow and needs network. The gate itself stays offline; nothing in
  `./scripts/gate.sh` invokes `nix`.
- The nixpkgs pin is a maintenance edge: bumping it can change Python's patch
  version and any native library, so a bump is its own commit with a gate run.
- Whether the FHS shell is sufficient for AMD's Torch install remains **OQ-008**.
  This ADR only commits to where that attempt happens.
- This repository is public, so this ADR names no host, user, or absolute path. The
  concrete values — the host's NixOS config path, the ComfyUI module to model the FHS
  shell on, the current nixpkgs revision — are in the uncommitted `LOCAL.md`.
