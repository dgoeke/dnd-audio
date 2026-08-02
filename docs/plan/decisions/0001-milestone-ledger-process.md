# ADR-0001 — Milestone ledger with cleared context between milestones

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** pre-M0

## Context

The spec is large and will be implemented over many sessions by agents whose
context is cleared between milestones. Anything an agent knows but does not write
down is lost. Conversation history is not a durable record, and the spec describes
the destination rather than the path or the discoveries made along the way.

## Decision

The repository is the memory. Durable knowledge has exactly one home:

- `docs/plan/STATE.md` — where we are. The only file that changes every session.
- `docs/plan/ROADMAP.md` — milestones, dependencies, gates.
- `docs/plan/INVARIANTS.md` — cross-cutting rules, each with an ID and a test.
- `docs/plan/OPEN-QUESTIONS.md` — assumptions awaiting evidence, each with an ID
  cited from the code that depends on it.
- `docs/plan/decisions/` — ADRs for choices the spec left open.
- `docs/plan/milestones/M*.md` — a light charter written up front and a closeout
  written at the end, including notes for future implementors.
- `docs/plan/reviews/` — external review records.

`AGENTS.md` holds the working agreement and points at all of it. `CLAUDE.md` is
one line — `@AGENTS.md` — plus the Claude-only slash-command section, so Claude
and Codex cannot drift apart on project rules. Three slash commands (`/ms-start`,
`/ms-verify`, `/ms-close`) enforce the cycle. `scripts/gate.sh` is the mechanical
gate; `scripts/codex-review.sh` is an independent second opinion that supplies
*role* in its prompt and *context* by pointing at `AGENTS.md`.

Charters are deliberately thin — goal, gate, non-goals, risks — because early
milestones will change what later ones should do. Task-level planning happens at
`/ms-start` time with fresh context and current facts.

## Alternatives considered

- **A single running design document.** Rejected: it becomes a wall of prose that
  nobody can diff, and stale statements are indistinguishable from current ones.
- **Issue tracker / TODO list only.** Rejected: captures what is left to do but
  not why decisions were made, which is the expensive part to lose.
- **Detailed up-front plans for all milestones.** Rejected: the owner explicitly
  wants to avoid over-prescribing, and M1's findings about real DJI metadata will
  invalidate guesses made now.

## Consequences

- Every milestone pays a small closeout tax. That is the point.
- Two places can drift: charters vs. reality, and `STATE.md` vs. the tree.
  `/ms-close` updates both, and `/ms-start` re-reads them before trusting them.
- A milestone that changes a later milestone's premise must edit that charter at
  close time. If it does not, the next agent inherits a false plan.
