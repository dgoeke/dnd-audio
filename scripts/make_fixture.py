#!/usr/bin/env python3
"""Materialize the canonical synthetic session on disk.

For running the real commands against a real directory — during a milestone's verify
phase, or when a stack trace is easier to read than a test failure. The tests build
their own fixtures in `tmp_path`; this is the same generator pointed somewhere you can
look at afterwards.

    python scripts/make_fixture.py /tmp/session-demo
    uv run dnd-audio inspect /tmp/session-demo

Writes nothing outside the directory it is given, and refuses a directory that already
holds a session rather than merging into it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dnd_audio.fixtures import build_session, canonical_session


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    directory = Path(argv[1]).expanduser()
    if (directory / "session.yaml").exists():
        print(f"{directory} already holds a session.yaml; refusing to write over it")
        return 1

    truth = build_session(canonical_session(), directory)
    total = sum(chunk.size_bytes for chunk in truth.chunks)
    tracks = sorted({chunk.track_id for chunk in truth.chunks})

    print(f"wrote {len(tracks)} tracks, {len(truth.chunks)} chunks, {total / 1e6:.1f} MB")
    print(f"  session.yaml   {directory / 'session.yaml'}")
    print(f"  fake models    {directory / 'fake-models.json'}")
    print(f"  session zero   {truth.session_zero_since_midnight} samples since midnight")
    for track_id, start, end in truth.gaps():
        print(f"  gap            {track_id}: samples {start}-{end}")
    print(f"\n  uv run dnd-audio inspect {directory}")
    # The transcript branch needs something behind the ASR seam and the real adapter lands
    # in M6b, so the fixture's own declared script stands in for it (ADR-0018).
    print(f"  uv run dnd-audio transcribe --fake-models {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
