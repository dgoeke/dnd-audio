#!/usr/bin/env bash
# One-time model installation.
#
#   ./scripts/fetch-models.sh          # install everything, then verify
#   ./scripts/fetch-models.sh --plan   # print what would be installed, and touch nothing
#
# This is a *wrapper*, not a second way to download models. `dnd-audio models fetch
# --qwen` does the work and stays the single network authority the spec, INV-06 and M6b's
# completion gate all name; every repository, commit and target directory comes from
# `dnd-audio models plan`, so nothing here restates a pin (ADR-0027).
#
# What the wrapper is for is the environment. `models fetch --qwen` shells out to the `hf`
# CLI, which ships with the `huggingface_hub` that lives in `.venv-rocm` and deliberately
# not in `.venv` (ADR-0025) — so the command has to run from inside the FHS shell, and
# running it from the everyday one fails with a `FileNotFoundError` for `hf` that says
# nothing about which shell you are in. This enters that shell for you.
#
# About 6 GB over two repositories. Re-running is cheap: an already-verified snapshot is
# not downloaded again, so this doubles as an "am I set up?" check.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

ROCM_ENV=".venv-rocm"

if [ "${1:-}" = "--plan" ]; then
    # No network, no `hf`, no ROCm environment needed — the plan is a pure function of
    # the pins compiled into this build.
    exec uv run --no-sync dnd-audio models plan
fi

if [ ! -d "$ROCM_ENV" ]; then
    cat >&2 <<EOF
no $ROCM_ENV — the ROCm environment is not built yet, and \`hf\` lives in it.

Build it first (this is M6a's step, and it needs the FHS shell too):

    nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=$ROCM_ENV uv sync --group asr-qwen'
EOF
    exit 1
fi

command -v nix >/dev/null 2>&1 || { echo "nix is not on PATH" >&2; exit 127; }

# `uv run` rather than the venv's console script, so the ROCm environment's interpreter
# and this repository's source are the pair that runs — the same invocation shape the
# gate uses. `--no-sync` because syncing from inside here would resolve against the wrong
# environment variable and is not this script's job.
exec nix run .#fhs -- -c \
    "UV_PROJECT_ENVIRONMENT=$ROCM_ENV uv run --no-sync dnd-audio models fetch --qwen"
