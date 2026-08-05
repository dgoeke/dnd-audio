# ADR-0039 — The archive operation report, and the three words for "checked"

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M7a

## Context

`ingest-report.json` is the artifact INV-13 is about, and it is shaped for the
processing DAG: it accounts for six named stages, rolls them up, and refuses to build
with a gap. An archive operation is not one of those stages and never will be — it does
not run under `process`, it has no place in the stage DAG, and forcing it into that
report would mean either inventing a seventh stage that five commands must then skip
with a reason, or writing a report whose `overall_status` describes something other than
a pipeline run.

Three of the five archive commands do not have a session directory at all. `list`,
`verify` and `restore` exist precisely for the case where the local session is gone, so
"write the report under `work/`" is not available to them.

And there is a vocabulary problem that is really a correctness problem. An operator
asking "is my archive good?" can be answered in three genuinely different ways:

- the manifest exists remotely, so an upload once completed;
- that upload read every object back at the time it committed;
- these bytes were downloaded and decompressed **just now** and they are correct.

Only the third is verification. The first two are history. Collapsing them into one word
produces the failure this whole milestone is built against: a green display describing an
archive nobody has actually read.

## Decision

**Every archive operation writes its own local structured report**, separate from
`ingest-report.json`, which is left untouched.

It carries the manifest SHA-256 where one is available, the exact scope (whole session or
one track), per-object outcomes, an overall `complete` / `failed` / `partial`, and
structured secret-free errors. Partial and failed never exit zero (INV-13). It is written
even when the operation failed — that being the whole point of INV-13 — and atomically.

**Three distinct states, never merged:**

| state | means |
| --- | --- |
| `committed` | a manifest exists remotely for this session |
| `previously_verified_at_commit` | the recorded upload read every object back when it committed |
| `verified` | **this operation** downloaded and decompressed these bytes and they are correct |

`status` is cheap and non-authoritative: it may report the first two and **may never
report the third**. Only a current full GET plus decompression produces `verified`.

**Where reports go.** Upload and status write under the session's `work/`, like every
other local artifact. `list`, `verify` and `restore` take `--report PATH` and otherwise
default to `$XDG_STATE_HOME/dnd-audio/archive/` — outside any session, because there may
not be one. No report is ever uploaded: Cold Storage bills anything under 128 KiB as
128 KiB, and the bucket holds exactly one small object per session, the manifest.

**The manifest cannot contain its own hash**, so each local report records it instead —
the same limit and the same remedy ADR-0003 established for `ingest-report.json`.

**No secrets, and it is tested rather than intended.** No endpoint credential, access
key, signed URL, or local absolute machine path reaches a report, a log line, or an
exception message. Credentials are held as `SecretStr` so a stray repr cannot leak one,
and a scan over serialized reports, captured log output and exception text is what proves
it.

## Alternatives considered

**Extend `ingest-report.json` with an archive section.** Rejected: the report's stage
model would have to grow a stage that is not in the DAG, five commands would have to skip
it with reasons, and a remote-only `verify` has no session directory to write it into.

**Upload each report as a small object**, so the archive is self-describing. Rejected on
billing — that is the 128 KiB floor multiplied across every operation — and on principle:
a verification receipt stored in the thing being verified is not independent evidence.

**One `verified` boolean.** Rejected. This is the finding the first plan review made
(P1, "cheap status could misleadingly say `verified`"), and it is the single most
important word in the milestone. A `status` that says `verified` from provider metadata
is a lie that costs nothing to tell and everything to believe.

**Reuse `StageReport`/`OverallStatus` wholesale.** Partially taken: the vocabulary of
`complete`/`failed`/`partial` and the exit-code discipline are reused deliberately, so an
operator reads the same words in both places. The stage machinery is not, because an
archive operation has no stages.

## Consequences

An operator can tell "the archive exists" from "the archive was good once" from "I just
checked it", which is exactly the distinction that decides whether it is safe to lose the
local copy — and M7b's reclamation question, when it comes, is asked entirely in these
three words.

The cost is a second report format, a second schema, and a second place `--report`-shaped
ergonomics have to be thought about.

What would make us revisit: if M7b needs a reclamation decision to consult archive
history programmatically, these reports become an input rather than an audit trail, and
the default location may need to be a queryable directory rather than a convenience.
