@AGENTS.md

## Claude Code specifics

The working agreement above is shared with every agent that touches this
repository — Codex reads `AGENTS.md` directly. Keep project rules there, not here,
so the two never disagree. This section is Claude-only.

The three phases of the milestone cycle described above map to project slash
commands (defined in `.claude/commands/`):

| Phase  | Command          |
| ------ | ---------------- |
| Start  | `/ms-start <N>`  |
| Verify | `/ms-verify <N>` |
| Close  | `/ms-close <N>`  |

Documents under `docs/plan/` name the phases rather than the commands, because
Codex reads those documents too and has no idea what a slash command is. Where one
is mentioned there, it is marked as Claude-specific.
