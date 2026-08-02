# dnd-audio — working agreement

Shared by every agent that works on this repository. Claude Code reads it via
`CLAUDE.md`; Codex and other tools read this file directly.

A fully local audio-ingestion and transcription pipeline for long tabletop-RPG
sessions. Six DJI Mic 3 transmitter recordings in; diarized transcript, automix
MP3, and an ingest report out. No audio ever leaves this machine.

`dnd-audio-ingestion-agent-spec.md` is the authoritative product spec. It does not
change casually. If implementation proves part of it wrong, record an ADR under
`docs/plan/decisions/` and amend the spec in the same commit — never let code and
spec disagree silently.

## How work is organized

Work proceeds one milestone at a time, with the agent's context cleared between
milestones. **The repository is the memory.** Before doing anything substantive:

1. `docs/plan/STATE.md` — where the project actually is right now. Read first.
2. `docs/plan/ROADMAP.md` — milestones, dependencies, completion gates.
3. `docs/plan/INVARIANTS.md` — cross-cutting rules (INV-01..INV-13) that no
   milestone may break.
4. `docs/plan/OPEN-QUESTIONS.md` — assumptions still awaiting evidence.
5. `docs/plan/milestones/M<N>-*.md` — charter for the milestone in question, plus
   the closeout sections of every milestone already completed.

Then `LOCAL.md`, if it exists — uncommitted operator notes naming the host, user, and
absolute paths that the public documents refer to only in the abstract. Its absence
means you are on a fresh clone, not that something is wrong.

## The milestone cycle

Every milestone runs through three phases. Documents in `docs/plan/` refer to them
by name. Claude Code has slash commands that drive them; any other agent performs
the same phases directly.

- **Start** — read the ledger and every prior closeout, check preconditions (clean
  tree, dependency milestones closed, gate green at HEAD), branch, plan against the
  charter's completion gate, then implement.
- **Verify** — run the gate, prove every gate criterion with executed output, hunt
  for work that only appears done, and take an independent review.
- **Close** — write the charter's Closeout section, record ADRs and `OQ-` updates,
  propagate changes into downstream charters, update `STATE.md`, and commit.

`./scripts/gate.sh` is the mechanical gate. It must pass with no GPU, no model
weights, and no network.

`./scripts/codex-review.sh` gets an independent second opinion from the Codex CLI —
`plan <N>` before implementing, `code <N>` before closing. Run it at least once per
milestone. It reasons differently on purpose; treat disagreement as signal, and
record which findings were rejected and why.

## Hard rules

- **Never modify anything under a session's `raw/`.** Not renamed, not normalized,
  not rewritten. Hash-verified before and after every run.
- **Never send audio off the machine.** Model downloads during an explicit fetch
  step are the only permitted network traffic, and never in the default test suite.
- **Never commit** session audio, model weights, tokens, generated working audio,
  or output artifacts.
- **No placeholder implementations.** A milestone is not complete because a
  function exists; it is complete when its gate criteria are demonstrated by tests
  that would fail if the behavior regressed.
- **No unexplained skipped tests.** Every `skip`/`xfail` needs a `reason=` naming
  the milestone (`M6b`) or open question (`OQ-004`) that will resolve it.
- **The ledger stays consistent.** `scripts/check_plan.py` runs in the gate: every
  milestone has a charter, a roadmap entry, and a `STATE.md` row, and every
  `INV-`/`OQ-`/`ADR-` reference resolves to something that exists.
- **Deterministic artifacts stay byte-stable.** See `INVARIANTS.md`.

## Recording knowledge as you go

- A choice the spec left open, made deliberately → **ADR** in
  `docs/plan/decisions/`.
- A guess about the real world that evidence could overturn → **`OQ-NNN`** entry in
  `OPEN-QUESTIONS.md`, referenced from the code comment that depends on it.
- Something the next implementor would waste an hour rediscovering → **Notes for
  future implementors** in the current milestone's closeout.
- Anything that changes what a later milestone must do → edit that milestone's
  charter now, while you still remember why.

## Conventions

- Python 3.12, `uv`, typed `src/` layout, `pytest` + `ruff` + strict type checking.
- The dev environment is the repo's Nix flake, activated by `direnv` from `.envrc`
  (`use flake`). If `python --version` is not 3.12 or `uv`/`ffmpeg`/`sox` do not
  resolve into `/nix/store`, run `direnv allow` — do not work around it with host
  tools. `nix develop .#fhs` is the separate FHS shell, needed only for M6a's ROCm
  wheels (ADR-0002).
- This repository is **public**. Committed documents name no hostname, username, or
  absolute home-directory path; they say "the target host", "the invoking user", "the
  host's NixOS configuration". The concrete values live in `LOCAL.md`, which is
  gitignored. Read it for specifics, keep writing the abstractions, and never copy
  its contents into a tracked file.
- Default test suite runs offline on CPU. Tests needing a GPU, model weights, or
  network are marked `host_smoke` and excluded from the gate.
- Times are integer samples or rationals internally; floats only at the public
  millisecond serialization boundary.
