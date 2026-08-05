---
description: Close a milestone — record decisions, notes, and state, then commit
argument-hint: "<milestone id, e.g. 0, 2, 6a, 11> [additional context]"
---

**Invocation:** `$ARGUMENTS` — the milestone ID first, then anything else.

Resolve **$0** to its charter under `docs/plan/milestones/`; a bare number means
the M track (`2` is M2). That charter's own ID — `M0`, `M6a`, `M11` — is
what `<ID>` means below. Anything the invoker typed after the ID is something they
want recorded in the closeout. Write it in; it never substitutes for a section you
would otherwise have filled.

Close milestone **<ID>**. This is where the context you are about to lose gets
written down. Treat it as the deliverable, not the paperwork.

Precondition: `/ms-verify <ID>` returned **VERIFIED** and `STATE.md` shows this
milestone as `verified`. If not, stop and verify first.

## 1. Write the closeout

In `docs/plan/milestones/<ID>-*.md`, **replace the `## Working plan` section** with
a filled-in `## Closeout`. Every heading in the template gets a real answer:

- **What works end to end.** What a user can actually run now, and what they get.
- **Tests and commands run, with results.** Real output, not "tests pass".
- **Decisions made.** Each one gets an ADR in `docs/plan/decisions/`; link them.
- **Assumptions made and open questions raised.** New `OQ-NNN` entries added to
  `OPEN-QUESTIONS.md`; existing ones moved to `answered` with their evidence.
- **Notes for future implementors.** The highest-value section. What surprised
  you, what you tried that did not work, where the sharp edges are, what looks
  wrong but is deliberate. Write it for someone with no memory of this work,
  because that is exactly who reads it.
- **Deviations from this charter, and why.**
- **Downstream charters updated.** See phase 2.
- **Next smallest step.**

## 2. Propagate what changed

This is the step that keeps the plan honest, and the one most likely to be skipped:

- Did this milestone change what a later milestone must do? Edit that charter now.
- Did it invalidate an ADR? Mark the old one superseded and write the replacement.
- Did it answer or reframe an `OQ-`? Update the entry and `rg` for the ID to find
  every code site that cited it.
- Did it establish a new cross-cutting rule? Add an `INV-` with an owner and a test.
- Did the roadmap's dependency graph change? Update `ROADMAP.md`.

## 3. Update STATE.md

- Milestone status → `closed`, with the closing commit SHA.
- Current milestone → the next one.
- Refresh **What works end to end** and **Next smallest step**.
- Note anything now blocked, especially waiting on live Session Zero evidence.

## 4. Commit and integrate

Commit on the milestone branch:

```
<ID>: <one-line summary>

<what landed, in a few lines>

Gate: ./scripts/gate.sh passing
Decisions: ADR-NNNN, ADR-NNNN
Open questions: OQ-NNN raised, OQ-NNN answered
```

Then **ask the user** before touching `main`. Propose:

- merge to `main` (`--no-ff` so the milestone stays a visible unit), and
- tag `<ID>-closed` so a later bisect can find the boundary.

Confirm `main` is green after merging.

## 5. Hand off

Print a short handoff for the user:

- What works end to end now.
- Tests and commands run, with results.
- Assumptions made.
- Remaining blockers — explicitly including whether real DJI metadata has been
  validated yet.
- The next smallest implementation step, and the command to start it.

Then tell the user to clear context before running `/ms-start <next>`.
