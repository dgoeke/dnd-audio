# ADR-0003 — The report hashes every deliverable except itself

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M0

## Context

The spec's product goal lists four deliverables, and `ingest-report.json` is the
fourth. Its error-handling section then requires the report to include "the hashes of
every deliverable that was successfully produced", and INV-13 repeated that wording.

Taken literally those two statements cannot both hold. Writing a hash into the report
changes the report's bytes, which changes its hash. There is no fixed point: any value
written is wrong the instant it is written.

The contradiction surfaced in an independent review of M0's working plan
(`docs/plan/reviews/M0-plan-20260802-0912.md`), before the report writer existed. It is
cheaper to settle now than after five milestones have written to the report.

## Decision

The report carries hashes of every deliverable it did not itself produce —
`manifest.json`, `transcript.json`, `transcript.md`, `session.mp3`, and any future
artifact — and never a hash of `ingest-report.json`.

`docs/plan/INVARIANTS.md` (INV-13) and the spec's error-handling section are amended in
the same commit as this ADR, each with the same one-clause carve-out, so neither
document states an impossibility.

The report's own integrity is a consumer's problem, not the report's: anything that
needs to verify it can hash the file it just read.

## Alternatives considered

- **A sidecar file — `ingest-report.json.sha256`.** Rejected for the MVP. It solves a
  problem nobody has: the report is read locally, immediately after the run that wrote
  it, by a person or a script that already has the bytes in hand. M7's archival work is
  where remote-integrity verification actually matters, and it will need to hash
  *everything* it uploads including the report, which a sidecar written at run time does
  not help with.
- **A two-pass write: write the report, hash it, rewrite it with the hash embedded.**
  Rejected. The second write invalidates the hash again unless the hash field is excluded
  from its own computation, which means defining a canonical "report minus one field"
  serialization that every consumer must reimplement to check anything.
- **Leave both documents as written and quietly omit the self-hash.** Rejected. That is
  exactly the silent code/spec disagreement `AGENTS.md` forbids, and the next implementor
  would read INV-13, notice the missing hash, and "fix" it.

## Consequences

- The report's `provenance.deliverables` list is well defined and finite, and a test can
  assert that `ingest-report.json` never appears in it.
- Verifying a report's own integrity requires the verifier to hash the file. M7 should
  record the report's hash in whatever archival manifest it produces, not in the report.
- This is the first amendment to the spec. It is a one-clause correction of a literal
  impossibility, not a scope or design change, and the spec otherwise stands unaltered.
