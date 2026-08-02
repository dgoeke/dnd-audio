# ADR-0004 — mypy in strict mode is the type checker

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M0

## Context

The spec asks for "a typed `src/` package with `pytest`, `ruff`, and a strict type
checker" without naming one, and the M0 charter records the choice as undecided with an
instruction to make it and write this.

Two constraints narrow the field more than usual:

- **The default test suite runs offline** (INV-05). Whatever the gate invokes must be
  installable from the lock and then work with no network. A checker that fetches a
  runtime at first use is disqualified.
- **Pydantic is the schema layer.** Configuration, manifest, transcript, and report are
  all Pydantic models, so the checker's understanding of them determines how much of the
  project it can actually check.

## Decision

`mypy` with `strict = true` and the `pydantic.mypy` plugin, configured in
`pyproject.toml` and invoked by the gate as
`uv run --no-sync mypy src tests scripts`.

Configuration lives in `pyproject.toml` rather than in the gate script, so an editor,
a manual run, and the gate cannot disagree about what "strict" means. Three extra error
codes are enabled beyond `strict`: `ignore-without-code`, `redundant-expr`, and
`truthy-bool`. The last one earns its place immediately — it catches
`assert model_validate(...)`, an assertion that can never fail because the value is
never falsy.

`scripts/` is checked too. `check_plan.py` and `scan_placeholders.py` gate every
milestone; a bug in one of them fails open.

## Alternatives considered

- **pyright / basedpyright.** Better inference, faster, and its Pydantic understanding
  is good without a plugin. Rejected on packaging: `pyright` from PyPI downloads a Node
  runtime on first use, which the offline gate forbids, and `basedpyright` avoids that
  only by vendoring Node binaries as wheels. Adding a Node toolchain to a project whose
  environment is already carefully pinned across two Nix shells buys inference we do not
  need at this size.
- **`ty` (Astral).** The obvious future fit next to `uv` and `ruff`, and it would make
  the toolchain uniform. Rejected as premature: it is not yet stable, and a strict
  contract enforced across every milestone gate is the wrong place to absorb a
  pre-1.0 checker's changing diagnostics. Revisit once it is stable — the migration is
  mechanical, since the annotations are the artifact and mypy's configuration is small.
- **`--strict` on the command line instead of in `pyproject.toml`.** Rejected: it makes
  an editor's inline diagnostics differ from the gate's, so the first time anyone sees a
  strictness error is in CI.
- **Checking only `src/`.** Rejected. The tests are where the invariants are asserted;
  an untyped test can quietly assert nothing at all.

## Consequences

- Pydantic's plugin is a dependency of the type check, so a Pydantic major upgrade may
  need a plugin bump in the same commit.
- mypy is slower than pyright and will get slower as the project grows. At M0 the whole
  check is about a second; if it becomes a drag, that is the signal to revisit `ty`
  rather than to loosen strictness.
- `strict` makes every new module's untyped edges visible immediately, including in
  M6b's adapter, where Transformers' own annotations are patchy. Expect targeted
  `# type: ignore[code]` there — and `ignore-without-code` guarantees each one names
  what it is silencing.
