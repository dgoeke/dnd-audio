---
description: Begin a milestone — orient from the ledger, check preconditions, plan, then implement
argument-hint: "<milestone id, e.g. 0, 2, 6a, H1>"
---

Begin milestone **$1**. Work through these phases in order. Do not skip the
orientation phase even if you think you remember the project — you do not, the
context was cleared.

## 1. Orient

Read, in this order:

- `AGENTS.md` (imported by `CLAUDE.md`, so already in context — reread it anyway)
- `docs/plan/STATE.md`
- `docs/plan/ROADMAP.md`
- `docs/plan/INVARIANTS.md`
- `docs/plan/OPEN-QUESTIONS.md`
- `docs/plan/milestones/` — this milestone's charter, **plus the Closeout section
  of every milestone already marked closed**. Those closeouts contain the notes
  written specifically for you.
- `docs/plan/decisions/` — every ADR.
- The sections of `dnd-audio-ingestion-agent-spec.md` this charter names, in full.

Then read the actual code that this milestone builds on. The ledger says what was
decided; the code says what was done.

## 2. Check preconditions

Report each explicitly, and stop if any fail:

- Working tree is clean (`git status --porcelain` empty).
- Every milestone this one depends on is `closed` in `STATE.md`.
- `./scripts/gate.sh` passes at HEAD. Never start a milestone on a broken tree —
  you will spend the milestone debugging someone else's failure and blame yourself.

## 3. Branch

`git switch -c milestone/M$1-<short-slug>`

## 4. Plan

Enter plan mode and produce a working plan:

- The concrete files you will create or change, and in what order.
- **Every completion-gate criterion mapped to the specific test or command that
  will demonstrate it.** A criterion with no named proof is a criterion you will
  fail to meet.
- Which invariants this milestone could plausibly violate, and what stops that.
- Anything in the charter that looks wrong now that you have read the code. If the
  charter is wrong, say so — amending it is a legitimate outcome of this phase.
- What you will deliberately *not* do, from the charter's non-goals.

Present it and get approval before writing code.

## 5. Record the plan and get an outside opinion

Once approved, write the plan into the charter under a `## Working plan` heading
(placed just above the `---` that precedes Closeout). It gets replaced by the
closeout at the end, so this is a scratch section, not a permanent artifact.

Then run an independent critique — Codex reasons differently and is worth hearing
before the code exists, not only after:

```bash
./scripts/codex-review.sh plan $1
```

Read its output, tell the user which points you accept and which you reject **with
reasons**, and amend the plan for the ones you accept. Codex is a second opinion,
not an authority; disagreeing with it is fine, ignoring it is not.

## 6. Implement

Ground rules while working:

- Write the test that proves a gate criterion alongside the code, not after. A test
  written later tends to assert whatever the code already does.
- No placeholder implementations. No skipped tests without a `reason=` naming a
  milestone or `OQ-`.
- New assumption about the real world → add an `OQ-NNN` entry and cite the ID in
  the code comment that depends on it.
- Choice the spec left open → ADR in `docs/plan/decisions/` at the time you make
  it, while you still remember the alternatives.
- Discovering the charter was wrong → amend it and write an ADR. Do not silently
  drift; the next agent will trust the charter.
- Run `./scripts/gate.sh` often. Commit to the branch at meaningful checkpoints.

When the gate criteria are met, run `/ms-verify $1`.
