#!/usr/bin/env python3
"""Keep the planning ledger internally consistent.

ADR-0001 names the two places this project can silently drift: charters vs.
reality, and STATE.md vs. the tree. Reality needs a human; the rest is mechanical,
so it runs in the gate:

* every ledger file the working agreement promises actually exists;
* no double-brace scaffold placeholder was left unfilled;
* every milestone charter is well formed — status, dependencies, and a completion
  gate with at least one checkable criterion;
* charters, ``ROADMAP.md``, and ``STATE.md`` agree on which milestones exist;
* every ``INV-``/``OQ-``/``ADR-`` reference anywhere in the repository resolves to
  something defined — a line carrying ``check-plan: ignore`` is exempt, for prose
  that cites an ID illustratively rather than depending on it;
* every invariant names an owning milestone, and every open question has a status.

Exit 0 with a one-line summary, or 1 with the specific inconsistencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PLAN = Path("docs/plan")
MILESTONES = PLAN / "milestones"
DECISIONS = PLAN / "decisions"
STATE = PLAN / "STATE.md"
ROADMAP = PLAN / "ROADMAP.md"
INVARIANTS = PLAN / "INVARIANTS.md"
QUESTIONS = PLAN / "OPEN-QUESTIONS.md"

REQUIRED = (STATE, ROADMAP, INVARIANTS, QUESTIONS,
            MILESTONES / "_template.md", DECISIONS / "0000-template.md")

# Milestone ID prefixes in use — see ROADMAP.md. Add one when a new parallel
# track is introduced. Keep in sync with scripts/scan_placeholders.py.
MILESTONE_PREFIXES = ("M", "H")

# Where references are scanned from: every top-level *.md (the spec included) plus
# these directories, walked for docs and source. Reviews are excluded because an
# external reviewer may cite anything it likes.
SCAN_DIRS = ("docs", "src", "tests", "lib", "app", "scripts", ".claude")
EXCLUDE_DIRS = {"reviews", ".git", "node_modules", "__pycache__", "target", "dist"}
SCAN_SUFFIXES = {".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".rb", ".sh"}

# Files that must contain no unfilled scaffold placeholders.
PLACEHOLDER_ROOTS = ("docs/plan", ".claude/commands", "scripts")

_ID = "|".join(f"{p}\\d+[a-z]?" for p in MILESTONE_PREFIXES)
CHARTER_NAME = re.compile(rf"^({_ID})-.+\.md$")
STATE_ROW = re.compile(rf"^\|\s*({_ID})\s*\|", re.MULTILINE)
GATE_HEADING = re.compile(r"^#{2,3}\s+.*completion gate", re.MULTILINE | re.IGNORECASE)
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")

INV_DEF = re.compile(r"^\*\*(INV-\d+)\b", re.MULTILINE)
OQ_DEF = re.compile(r"^##\s+(OQ-\d+)\b", re.MULTILINE)
INV_REF = re.compile(r"\bINV-\d+\b")
OQ_REF = re.compile(r"\bOQ-\d+\b")
ADR_REF = re.compile(r"\bADR-(\d{4})\b")

# Prose that cites an ID as an example rather than depending on it.
IGNORE_LINE = "check-plan: ignore"


def scan_paths() -> list[Path]:
    paths: list[Path] = sorted(p for p in Path().glob("*.md") if p.is_file())
    for d in SCAN_DIRS:
        base = Path(d)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if EXCLUDE_DIRS & set(path.parts):
                continue
            paths.append(path)
    return paths


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def entry_blocks(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    """Split a definitions file into {id: body} by the given heading pattern."""
    marks = [(m.group(1), m.start()) for m in pattern.finditer(text)]
    blocks: dict[str, str] = {}
    for i, (name, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        blocks[name] = text[start:end]
    return blocks


def main() -> int:
    problems: list[str] = []

    # 1. Required ledger files.
    for path in REQUIRED:
        if not path.is_file():
            problems.append(f"missing ledger file: {path}")
    if problems:
        for p in problems:
            print(f"    {p}")
        print("\n  the planning scaffold is incomplete; nothing else was checked")
        return 1

    # 2. Unfilled scaffold placeholders.
    for root in PLACEHOLDER_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or EXCLUDE_DIRS & set(path.parts):
                continue
            for m in PLACEHOLDER.finditer(read(path)):
                problems.append(f"{path}: unfilled scaffold placeholder {m.group(0)}")

    # 3. Charters are well formed.
    charters: dict[str, Path] = {}
    for path in sorted(MILESTONES.glob("*.md")):
        m = CHARTER_NAME.match(path.name)
        if not m:
            if path.name != "_template.md":
                problems.append(
                    f"{path}: filename does not start with a milestone ID "
                    f"({'/'.join(MILESTONE_PREFIXES)}<n>-<slug>.md)"
                )
            continue
        ms_id = m.group(1)
        charters[ms_id] = path
        text = read(path)
        if "**Status:**" not in text:
            problems.append(f"{path}: no '**Status:**' line")
        if "**Depends on:**" not in text:
            problems.append(f"{path}: no '**Depends on:**' line")
        if not GATE_HEADING.search(text):
            problems.append(f"{path}: no completion-gate section")
        elif not re.search(r"^\s*-\s*\[[ x]\]", text, re.MULTILINE):
            problems.append(f"{path}: completion gate has no checkable criteria")

    if not charters:
        problems.append(f"{MILESTONES}: no milestone charters found")

    # 4. Charters, ROADMAP, and STATE agree.
    state_text = read(STATE)
    roadmap_text = read(ROADMAP)
    state_ids = {m.group(1) for m in STATE_ROW.finditer(state_text)}

    for ms_id in sorted(charters):
        if ms_id not in state_ids:
            problems.append(f"{ms_id}: charter exists but no row in {STATE}")
        if not re.search(rf"\b{ms_id}\b", roadmap_text):
            problems.append(f"{ms_id}: charter exists but not mentioned in {ROADMAP}")
    for ms_id in sorted(state_ids - set(charters)):
        problems.append(f"{ms_id}: row in {STATE} but no charter in {MILESTONES}")

    # 5. Invariants and open questions are defined and complete.
    inv_blocks = entry_blocks(read(INVARIANTS), INV_DEF)
    oq_blocks = entry_blocks(read(QUESTIONS), OQ_DEF)

    if not inv_blocks:
        problems.append(f"{INVARIANTS}: no invariants defined (expected '**INV-01 — …**')")
    if not oq_blocks:
        problems.append(f"{QUESTIONS}: no open questions defined (expected '## OQ-001 — …')")

    for inv, body in sorted(inv_blocks.items()):
        if "Owner:" not in body:
            problems.append(f"{INVARIANTS}: {inv} names no owning milestone ('Owner:')")
    for oq, body in sorted(oq_blocks.items()):
        if "Status:" not in body:
            problems.append(f"{QUESTIONS}: {oq} has no 'Status:' field")
        if "Needs:" not in body:
            problems.append(f"{QUESTIONS}: {oq} has no 'Needs:' field")

    adrs = {
        m.group(1)
        for p in DECISIONS.glob("*.md")
        if (m := re.match(r"^(\d{4})-", p.name))
    }

    # 6. Every reference resolves.
    for path in scan_paths():
        for lineno, line in enumerate(read(path).splitlines(), 1):
            if IGNORE_LINE in line:
                continue
            for ref in sorted(set(INV_REF.findall(line))):
                if ref not in inv_blocks:
                    problems.append(
                        f"{path}:{lineno}: references {ref}, not defined in {INVARIANTS}"
                    )
            for ref in sorted(set(OQ_REF.findall(line))):
                if ref not in oq_blocks:
                    problems.append(
                        f"{path}:{lineno}: references {ref}, not defined in {QUESTIONS}"
                    )
            for num in sorted(set(ADR_REF.findall(line))):
                if num not in adrs:
                    problems.append(
                        f"{path}:{lineno}: references ADR-{num}, no such file in {DECISIONS}"
                    )

    if problems:
        print(f"  {len(problems)} ledger inconsistenc{'y' if len(problems) == 1 else 'ies'}:")
        for p in problems:
            print(f"    {p}")
        return 1

    print(
        f"  ledger consistent: {len(charters)} milestone(s), {len(inv_blocks)} invariant(s), "
        f"{len(oq_blocks)} open question(s), {len(adrs)} ADR(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
