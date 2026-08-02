#!/usr/bin/env python3
"""Regenerate the checked-in JSON Schema artifacts.

    uv run --no-sync python scripts/gen_schemas.py

Run it after changing any model under ``src/dnd_audio/artifacts/`` or the session
configuration. ``tests/test_schema_drift.py`` fails when the committed files disagree
with the models, so forgetting is a gate failure rather than a silent inconsistency.

This script only writes; the schemas themselves come from
:func:`dnd_audio.schema_export.schema_documents`, which the drift test also calls. One
source, so the two cannot diverge.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dnd_audio.schema_export import SCHEMA_DIRNAME, write_schemas

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    directory = REPO_ROOT / SCHEMA_DIRNAME
    for path in write_schemas(directory):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
