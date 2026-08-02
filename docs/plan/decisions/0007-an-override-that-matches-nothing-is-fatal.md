# ADR-0007 — A recovery override that matches nothing is fatal

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M1

## Context

`recovery.source_time_overrides` is keyed by session-relative source path. The spec says
an override "applies only to the named source file" and requires overrides to be recorded
prominently, but it does not say what happens when the named file is not among the ones
discovery found.

Three ways that happens, and only the first is a real recovery:

1. The file exists and the override applies. Nothing to decide.
2. The path is mistyped — a wrong directory, a transposed digit in a timestamp, a
   `.WAV` where the file is `.wav`.
3. The override is stale: it was written for a source that has since been removed or
   renamed, and the session has moved on without it.

Cases 2 and 3 are indistinguishable from inside the pipeline, and both look identical to
"no override was configured": the run proceeds, the file's own metadata is used, and the
operator's correction is silently discarded.

ADR-0005 already settled the neighbouring question. An override supplying neither a
timecode nor an offset nor a date is rejected, "because a silently ignored override is
exactly the failure the recovery mechanism exists to avoid". An override aimed at a path
that does not exist is that same failure with a typo in front of it.

## Decision

An entry in `recovery.source_time_overrides` whose key matches no discovered source is a
fatal `RecoveryError`, naming the unmatched key and listing the paths that were found in
the directory it points at.

The same applies to an override whose configured `sha256` does not match the file's
actual hash: it is fatal rather than ignored, because applying a field-log time to
different bytes is worse than the missing metadata the override was written to repair.

Matching is on the normalized session-relative POSIX path. `RecoveryConfig` already
normalizes and de-duplicates the keys at load time, so `raw/tx-a/f.wav` and
`raw//tx-a/./f.wav` cannot become two entries that each half-apply.

## Alternatives considered

- **Warn and continue.** The obvious option, and the one this rejects. A warning in a
  report nobody reads is how an operator discovers, after a four-hour transcription
  run, that the correction they carefully copied out of a field log did nothing. The
  override mechanism only exists for sessions that already went wrong once.
- **Fatal only under an explicit strict flag.** Rejected: it makes the safe behaviour
  opt-in, and the person who most needs it is the one who has never had to think about
  the flag.
- **Fuzzy matching — basename, or case-insensitive.** Rejected. It converts a clear
  error into a guess about which file the operator meant, which is the same class of
  mistake as inferring timing from a filename (INV-12).

## Consequences

- Removing a source file from a session now requires removing its override too. That is
  a real cost, and it is the direction to be wrong in: the error message names the key
  and shows what was actually found, so the fix is one line.
- A test asserts the fatal path (`test_an_override_matching_no_source_is_fatal`) and
  another asserts the hash mismatch, both through the CLI, so INV-13's "the report is
  still written" holds for these failures too.
- If a future milestone wants to carry overrides across sessions in a shared template,
  this decision is what it will collide with. That is intentional: the spec already says
  a session must be self-describing, and a template's stale entries should not silently
  evaporate.
