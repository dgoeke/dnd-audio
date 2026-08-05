#!/usr/bin/env bash
# Independent second opinion from the Codex CLI.
#
# Codex reads the same spec and ledger but reasons differently, which is the
# whole point: it is here to disagree. Run it at least once per milestone.
#
#   scripts/codex-review.sh plan M1          # critique the charter + working plan
#   scripts/codex-review.sh code M1 [base]   # review the milestone branch diff
#
# The raw session is echoed and saved as docs/plan/reviews/<ms>-<mode>-<stamp>.raw.md,
# which .gitignore excludes: a reviewer transcript quotes every file it read, and
# LOCAL.md is one of them. This repository is public. Distil the reviewer's actual
# findings into <ms>-<mode>-<stamp>.md — no hostname, username, or absolute home path —
# and commit that.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODE="${1:-}"
MS="${2:-}"
BASE="${3:-main}"

if [ -z "$MODE" ] || [ -z "$MS" ]; then
    # The header comment block below the shebang is the usage message. Read it
    # rather than hardcoding a line range, which silently rots when it is edited.
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 2
fi

command -v codex >/dev/null 2>&1 || { echo "codex not on PATH"; exit 127; }

# Resolve the charter, accepting 0, m1, or M6a. Bare numbers name the M track.
# Getting this wrong reviews the wrong milestone against a prompt that still looks
# plausible, so it is deliberately explicit.
MS="${MS^}"
CHARTER=$(ls docs/plan/milestones/"${MS}"-*.md 2>/dev/null | head -1)
[ -n "$CHARTER" ] || CHARTER=$(ls docs/plan/milestones/M"${MS}"-*.md 2>/dev/null | head -1)
[ -n "$CHARTER" ] || CHARTER=$(ls docs/plan/milestones/[A-Z]"${MS}"-*.md 2>/dev/null | head -1)
[ -n "$CHARTER" ] || { echo "no charter found for milestone '$MS'"; exit 2; }

# Canonical ID from the file that actually matched, so the review filename and
# title name the milestone being reviewed rather than what was typed.
MS=$(basename "$CHARTER" | cut -d- -f1)

mkdir -p docs/plan/reviews
STAMP=$(date +%Y%m%d-%H%M)
OUT="docs/plan/reviews/${MS}-${MODE}-${STAMP}.raw.md"
KEEP="docs/plan/reviews/${MS}-${MODE}-${STAMP}.md"

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
    echo "codex: critiquing plan for milestone ${MS} -> ${OUT}"
    codex exec -s read-only "$PROMPT" </dev/null 2>&1 | tee "$OUT"
    ;;

  code)
    # `codex review` refuses a custom prompt alongside --base ("the argument
    # '--base <BRANCH>' cannot be used with '[PROMPT]'"), and the prompt is the whole
    # point — it is what supplies this project's priorities. So `code` drives
    # `codex exec` the same way `plan` does and names the range in the prompt.
    if [ "${CODEX_UNCOMMITTED:-0}" = "1" ]; then
        SCOPE="the uncommitted work: \`git diff HEAD\` plus anything \`git status --porcelain\` lists as untracked"
    else
        SCOPE="this branch's changes: \`git diff ${BASE}...HEAD\`, with \`git diff --stat ${BASE}...HEAD\` for shape"
        if [ -n "$(git status --porcelain)" ]; then
            echo "warning: working tree is dirty; uncommitted changes are NOT in this review."
            echo "         commit to the milestone branch first, or re-run with CODEX_UNCOMMITTED=1."
        fi
    fi

    PROMPT="${COMMON}

Review ${SCOPE}. Read the changed files in full rather than only the diff hunks —
a test that cannot fail usually looks fine in isolation.

Judge the changes against ${CHARTER}'s completion gate. Priorities, in order:

  1. Correctness bugs that produce wrong audio, wrong timestamps, wrong speaker
     attribution, or silently dropped speech.
  2. Invariant violations (INV-01..INV-13).
  3. Work that only appears complete: tests whose assertions cannot fail, tests
     asserting against output produced by the same code path, fixtures that encode
     the implementation's behavior rather than independent ground truth,
     placeholder implementations, and skipped tests.
  4. Gate criteria that are claimed but not actually demonstrated by a test.
  5. Anything the next implementor will misread six weeks from now.

Ignore formatting and naming preferences; ruff and the type checker already ran.

Do not modify any files. Output a review, not a patch."
    echo "codex: reviewing milestone ${MS} against ${BASE} -> ${OUT}"
    codex exec -s read-only "$PROMPT" </dev/null 2>&1 | tee "$OUT"
    ;;

  *)
    echo "unknown mode '$MODE' (expected 'plan' or 'code')"
    exit 2
    ;;
esac

echo
echo "raw transcript (not committed): ${OUT}"
echo "distil the findings into ${KEEP} and commit that one — see the header comment."
