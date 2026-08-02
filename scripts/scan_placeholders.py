#!/usr/bin/env python3
"""Fail the gate on unexplained placeholder work.

Two hard rules, both from AGENTS.md:

* A skipped or xfailed test must carry a ``reason=`` naming the milestone
  (``M6b``) or open question (``OQ-004``) that will resolve it.
* ``NotImplementedError`` in ``src/`` must be annotated ``DEFERRED: M<n>`` on the
  same line or the line above.

Everything else (TODO/FIXME/XXX/HACK) is reported as a count only. Those are not
failures, but ``/ms-verify`` is expected to account for them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOTS = ("src", "tests")
JUSTIFIED = re.compile(r"\b(M\d+[a-z]?|H\d+|OQ-\d+)\b")
SKIP_MARK = re.compile(r"\bmark\.(skip|skipif|xfail)\b")
DEFERRED = re.compile(r"DEFERRED:\s*(M\d+[a-z]?|H\d+|OQ-\d+)")
LOOSE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")

# A decorator's reason= may sit a few lines below the mark itself.
WINDOW = 4


def python_files(root: str) -> list[Path]:
    base = Path(root)
    return sorted(base.rglob("*.py")) if base.is_dir() else []


def main() -> int:
    violations: list[str] = []
    loose: list[str] = []

    for root in ROOTS:
        for path in python_files(root):
            lines = path.read_text(encoding="utf-8").splitlines()

            for i, line in enumerate(lines):
                where = f"{path}:{i + 1}"

                if SKIP_MARK.search(line):
                    window = "\n".join(lines[i : i + WINDOW])
                    if "reason=" not in window or not JUSTIFIED.search(window):
                        violations.append(
                            f"{where}: skip/xfail without a reason= naming a "
                            f"milestone or OQ\n      {line.strip()}"
                        )

                if root == "src" and "NotImplementedError" in line:
                    context = line + ("\n" + lines[i - 1] if i else "")
                    if not DEFERRED.search(context):
                        violations.append(
                            f"{where}: NotImplementedError without a "
                            f"'DEFERRED: M<n>' annotation\n      {line.strip()}"
                        )

                if LOOSE.search(line):
                    loose.append(f"{where}: {line.strip()}")

    if loose:
        print(f"  {len(loose)} loose marker(s) — not failures, but account for them:")
        for entry in loose[:20]:
            print(f"    {entry}")
        if len(loose) > 20:
            print(f"    ... and {len(loose) - 20} more")

    if violations:
        print(f"\n  {len(violations)} unexplained placeholder(s):")
        for entry in violations:
            print(f"    {entry}")
        return 1

    print(f"  no unexplained placeholders ({sum(len(python_files(r)) for r in ROOTS)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
