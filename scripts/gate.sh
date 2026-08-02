#!/usr/bin/env bash
# Mechanical milestone gate.
#
# Must pass with no GPU, no model weights, and no network access.
# Tests that need any of those are marked `host_smoke` and excluded here.
#
# Usage: ./scripts/gate.sh [--verbose]

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Filled in during M0 once the strict type checker is chosen (record an ADR).
# Example: TYPE_CHECK="uv run mypy --strict src tests"
TYPE_CHECK="${TYPE_CHECK:-}"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
PASSED=(); FAILED=(); SKIPPED=()

step() {
    local name="$1"; shift
    printf '\n%s== %s ==%s\n' "$BOLD" "$name" "$OFF"
    if "$@"; then
        PASSED+=("$name")
    else
        FAILED+=("$name")
    fi
}

skip() {
    printf '\n%s== %s ==%s\n%s  skipped: %s%s\n' "$BOLD" "$1" "$OFF" "$YELLOW" "$2" "$OFF"
    SKIPPED+=("$1: $2")
}

# --- system prerequisites -----------------------------------------------------

step "system dependencies" bash -c '
    ok=0
    for tool in uv ffmpeg ffprobe; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf "  %-10s %s\n" "$tool" "$(command -v "$tool")"
        else
            printf "  %-10s MISSING\n" "$tool"
            ok=1
        fi
    done
    exit $ok
'

# --- project checks -----------------------------------------------------------

if [ -f pyproject.toml ]; then
    step "ruff check"        uv run ruff check .
    step "ruff format"       uv run ruff format --check .

    if [ -n "$TYPE_CHECK" ]; then
        step "type check"    bash -c "$TYPE_CHECK"
    else
        skip "type check" "set TYPE_CHECK in scripts/gate.sh (M0)"
    fi

    step "pytest (offline, cpu)" uv run pytest -m 'not host_smoke' -q

    if [ -f uv.lock ]; then
        step "lock is current" uv lock --check
    else
        skip "lock is current" "no uv.lock yet"
    fi
else
    skip "ruff / types / pytest" "no pyproject.toml yet (M0 in progress)"
fi

# --- process checks -----------------------------------------------------------

step "placeholder scan" python3 scripts/scan_placeholders.py

# --- summary ------------------------------------------------------------------

printf '\n%s== gate summary ==%s\n' "$BOLD" "$OFF"
for n in "${PASSED[@]:-}";  do [ -n "$n" ] && printf '  %spass%s  %s\n' "$GREEN" "$OFF" "$n"; done
for n in "${SKIPPED[@]:-}"; do [ -n "$n" ] && printf '  %sskip%s  %s\n' "$YELLOW" "$OFF" "$n"; done
for n in "${FAILED[@]:-}";  do [ -n "$n" ] && printf '  %sFAIL%s  %s\n' "$RED" "$OFF" "$n"; done

if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '\n%sGATE FAILED%s (%d)\n' "$RED" "$OFF" "${#FAILED[@]}"
    exit 1
fi
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    printf '\n%sGate passed with %d skipped check(s).%s A milestone may not close on a skipped check\nunless its charter says so explicitly.\n' \
        "$YELLOW" "${#SKIPPED[@]}" "$OFF"
    exit 0
fi
printf '\n%sGATE PASSED%s\n' "$GREEN" "$OFF"
