"""`python -m dnd_audio`, so the CLI is reachable without the installed console script.

The console script `dnd-audio` is the ordinary entry point and is what an operator uses.
This exists for the case where a caller has an interpreter and a `PYTHONPATH` but not a
`bin/` on `PATH` — which is exactly the shape of `tests/test_archive_isolation.py`, where
each command must run as a subprocess under a `sitecustomize` trap.

It is not a convenience. Without it, `python -m dnd_audio.cli` imports the module, defines
`main`, calls nothing, and exits 0 — so a subprocess test invoking it that way asserts
against the output of a program that never ran. That is how INV-06's boundary proof spent
its first draft passing unconditionally, and the guard against it now lives in that test.
"""

from __future__ import annotations

from dnd_audio.cli import main

main()
