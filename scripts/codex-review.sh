#!/usr/bin/env bash
# Independent second opinion from the Codex CLI.
#
# Codex reads the same spec and ledger but reasons differently, which is the
# whole point: it is here to disagree. Run it at least once per milestone.
#
#   scripts/codex-review.sh plan M1          # critique the charter + working plan
#   scripts/codex-review.sh code M1 [base]   # review the milestone branch diff
#
# Output is echoed and saved under docs/plan/reviews/.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODE="${1:-}"
MS="${2:-}"
BASE="${3:-main}"

if [ -z "$MODE" ] || [ -z "$MS" ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

command -v codex >/dev/null 2>&1 || { echo "codex not on PATH"; exit 127; }

MS="${MS#M}"; MS="${MS#m}"
CHARTER=$(ls docs/plan/milestones/[MH]"${MS}"-*.md 2>/dev/null | head -1)
[ -n "$CHARTER" ] || { echo "no charter found for milestone '$MS'"; exit 2; }

mkdir -p docs/plan/reviews
STAMP=$(date +%Y%m%d-%H%M)
OUT="docs/plan/reviews/M${MS}-${MODE}-${STAMP}.md"

COMMON="Read these before forming an opinion, even if some are already in context:

  - AGENTS.md — the shared working agreement and hard rules for this repository.
  - docs/plan/INVARIANTS.md — cross-cutting rules INV-01..INV-13. Treat a
    violation as a top-severity finding.
  - ${CHARTER} — this milestone's charter. Its 'Completion gate' is the contract.
  - dnd-audio-ingestion-agent-spec.md — the authoritative product spec, for the
    sections the charter names.

You are here as an independent second opinion, not as the implementer. Whoever
wrote this has already read the spec and believes they satisfied it; your value is
entirely in the findings they could not generate themselves. Be concrete and
skeptical. Rank by severity. Prefer 'this specific line produces this specific
wrong result' over general advice. Say plainly if you think the charter, or the
spec itself, is wrong.

Any new assumption about DJI hardware, timing, or model behavior must be
registered in docs/plan/OPEN-QUESTIONS.md and cited from the code that depends on
it; an unregistered one is a finding."

case "$MODE" in
  plan)
    PROMPT="Read ${CHARTER}, especially its 'Working plan' section, alongside
dnd-audio-ingestion-agent-spec.md and docs/plan/INVARIANTS.md.

${COMMON}

Critique the plan before it is implemented. Specifically:
  1. What in this plan will not actually satisfy the completion gate?
  2. What is the plan getting wrong that is cheap now and expensive later —
     particularly anything touching determinism, exact time arithmetic, memory
     bounds, cache identity, or an interface two later milestones will consume?
  3. What is the spec asking for that this plan silently omits?
  4. Where is the plan over-building relative to the milestone's stated non-goals?
  5. What would you do differently, and why?

Do not modify any files. Output a review, not a patch."
    echo "codex: critiquing plan for milestone M${MS} -> ${OUT}"
    codex exec -s read-only "$PROMPT" </dev/null 2>&1 | tee "$OUT"
    ;;

  code)
    PROMPT="${COMMON}

Review this milestone's changes against ${CHARTER}'s completion gate. Priorities,
in order:

  1. Correctness bugs that produce wrong audio, wrong timestamps, wrong speaker
     attribution, or silently dropped speech.
  2. Invariant violations (INV-01..INV-13).
  3. Work that only appears complete: tests whose assertions cannot fail, tests
     asserting against output produced by the same code path, fixtures that encode
     the implementation's behavior rather than independent ground truth,
     placeholder implementations, and skipped tests.
  4. Gate criteria that are claimed but not actually demonstrated by a test.
  5. Anything the next implementor will misread six weeks from now.

Ignore formatting and naming preferences; ruff and the type checker already ran."
    SCOPE=(--base "$BASE")
    if [ "${CODEX_UNCOMMITTED:-0}" = "1" ]; then
        SCOPE=(--uncommitted)
    elif [ -n "$(git status --porcelain)" ]; then
        echo "warning: working tree is dirty; uncommitted changes are NOT in this review."
        echo "         commit to the milestone branch first, or re-run with CODEX_UNCOMMITTED=1."
    fi
    echo "codex: reviewing milestone M${MS} (${SCOPE[*]}) -> ${OUT}"
    codex review "${SCOPE[@]}" --title "Milestone M${MS}" "$PROMPT" </dev/null 2>&1 | tee "$OUT"
    ;;

  *)
    echo "unknown mode '$MODE' (expected 'plan' or 'code')"
    exit 2
    ;;
esac

echo
echo "saved: ${OUT}"
