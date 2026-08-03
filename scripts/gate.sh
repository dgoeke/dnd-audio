#!/usr/bin/env bash
# Mechanical milestone gate.
#
# Must pass with no GPU, no model weights, and no network access.
# Tests that need any of those are marked `host_smoke` and excluded here.
#
# Usage: ./scripts/gate.sh [--verbose]

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# mypy in strict mode; see ADR-0004. Configuration lives in pyproject.toml so the
# editor, the CLI, and this gate cannot disagree about what "strict" means.
TYPE_CHECK="${TYPE_CHECK:-uv run --no-sync mypy src tests scripts}"

# Empty unless PYTEST_WORKERS is set, so the default is whatever `addopts` in
# pyproject.toml chose and this script never states a worker count of its own.
PYTEST_PARALLEL=()
[ -n "${PYTEST_WORKERS:-}" ] && PYTEST_PARALLEL=(-n "$PYTEST_WORKERS")

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

# Also detects an unactivated shell. The target host's own python3 is 3.13 — the
# version requires-python excludes — and it has no sox at all, so those two are the
# canaries: outside the flake this step fails for a nameable reason instead of the
# suite failing for a confusing one. `nix` is deliberately never invoked here; on a
# cold store it would need the network (ADR-0002).
step "system dependencies" bash -c '
    ok=0
    for tool in uv ffmpeg ffprobe sox python; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf "  %-10s %s\n" "$tool" "$(command -v "$tool")"
        else
            printf "  %-10s MISSING\n" "$tool"
            ok=1
        fi
    done

    version=$(python --version 2>&1)
    location=$(command -v python 2>/dev/null)
    case "$version" in
        "Python 3.12."*) printf "  %-10s %s\n" "version" "$version" ;;
        *)               printf "  %-10s %s (expected 3.12.x)\n" "version" "$version"; ok=1 ;;
    esac
    case "$location" in
        /nix/store/*) ;;
        *)            printf "  %-10s not from the flake: %s\n" "source" "$location"; ok=1 ;;
    esac

    [ "$ok" -eq 0 ] || printf "\n  the project shell is not active — run: direnv allow\n"
    exit $ok
'

# --- project checks -----------------------------------------------------------

# Everything below runs with `uv run --no-sync`, which never touches an index: the
# gate is then provably offline rather than offline-by-habit. The cost is that a
# stale environment fails here instead of being silently repaired, which is the
# trade the invariant asks for (INV-05).
if [ -f pyproject.toml ]; then
    if [ ! -d .venv ]; then
        FAILED+=("environment")
        printf '\n%s== environment ==%s\n' "$BOLD" "$OFF"
        printf '  %sno .venv%s — run: uv sync\n' "$RED" "$OFF"
    else
        step "ruff check"        uv run --no-sync ruff check .
        step "ruff format"       uv run --no-sync ruff format --check .
        step "type check"        bash -c "$TYPE_CHECK"
        # `allow_network` is excluded alongside `host_smoke`: it is the socket block's
        # own escape hatch, reserved for `models fetch` (INV-06). Running an opted-out
        # test here would make "the gate is offline" false the moment one exists.
        #
        # No `-n` here. The suite runs in parallel because `addopts` in pyproject.toml
        # says so, which is what makes an ad-hoc `uv run pytest` parallel too; repeating
        # the worker count here would create a second place for it to drift. Overriding
        # is still possible and the array below is how — PYTEST_WORKERS=0 forces a
        # serial in-process run, PYTEST_WORKERS=16 tunes it for a different box —
        # because a command-line `-n` beats the one in `addopts`.
        #
        # Parallelism is safe here for a reason worth stating: both session-scoped
        # fixtures in `tests/conftest.py` build into their own `tmp_path_factory`
        # directory, so each worker gets its own copy and nothing crosses the process
        # boundary. The autouse socket block runs per worker, because every worker
        # imports the same conftest — parallelism does not widen INV-05.
        step "pytest (offline, cpu)" \
            uv run --no-sync pytest -m 'not host_smoke and not allow_network' -q \
                "${PYTEST_PARALLEL[@]}"

        if [ -f uv.lock ]; then
            step "lock is current" uv lock --check --offline
        else
            skip "lock is current" "no uv.lock yet"
        fi
    fi
else
    skip "ruff / types / pytest" "no pyproject.toml yet (M0 in progress)"
fi

# --- process checks -----------------------------------------------------------

step "placeholder scan" python3 scripts/scan_placeholders.py
step "plan consistency" python3 scripts/check_plan.py

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
