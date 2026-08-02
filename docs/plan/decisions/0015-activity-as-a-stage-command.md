# ADR-0015 — `activity` is a stage command, and a composed run writes one report

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M3

## Context

The spec's command list is `process`, `inspect`, `ingest`, `transcribe`, `mix`, `render`,
`doctor`, and `models fetch`. Its stage DAG names `activity` as *"the shared cached
Milestone 3 operation invoked by `transcribe`, `mix`, or `process`"* — a stage, not a
command. Both of the commands that would invoke it land in later milestones.

That leaves M3 with no way to run what it built. Every milestone so far has opened its
closeout with a real command and its real output, and a milestone whose only demonstration
is a test is exactly the "work that only appears done" the verify phase is supposed to hunt.

There is a second question underneath. `activity` needs a timeline, `ingest` needs a
manifest, and INV-13 requires one report accounting for every stage. Three commands each
writing their own report over the same file would leave the last writer's view of the run,
and `ReportBuilder.build()` refuses a report with any stage unaccounted for.

## Decision

### `dnd-audio activity <session-dir>` exists

The spec's own sentence permits it: *"Also expose independently resumable stages for
development and recovery."* `activity` is a stage in the spec's DAG; this exposes it, and
`transcribe`, `mix`, and `process` will call the same `run_activity` rather than
reimplementing it.

### A composed run is one run, with one report

`run_activity` does not shell out to `ingest` and it does not read a `timeline.json` it hopes
is current. It performs the whole chain in one process with one `ReportBuilder`: snapshot
the raw tree once, run inspection, build the timeline, run activity, verify INV-01 once at
the end, then publish every cache and write one report covering `inspect`, `reconstruct`, and
`activity`, with the remaining stages recorded as `skipped` with reasons.

To make that possible, `timeline/runner.py`'s existing `_ingest` body becomes a reusable
stage function taking a builder. `run_ingest` keeps its behaviour exactly and becomes a thin
wrapper. This touches closed-milestone code, as M2's extraction of `raw_guard` did.

### The timeline is rebuilt every run, not validated

M2 already decided the equivalent question for the manifest and gave the reason: a
configuration-hash match is not evidence that an artifact still describes what is on disk,
because a replaced file keeps every hash internally consistent and the INV-01 snapshot only
covers mutations *during* a run. The same argument applies one layer up, and the cost is the
same — near zero, because the inspection cache and the derivative cache both serve warm.

The alternative is staleness detection: compare hashes, decide what is current, and fail
somewhere new when the answer is subtle. Rebuilding deletes that entire class of bug and
every test it would have needed.

## Alternatives considered

- **Fold activity into `ingest`.** Cheapest, and it contradicts the spec's stage boundaries:
  `ingest` is inspection plus the timeline, and `mix` must be able to run activity without
  ever running an ingest command.
- **No command; demonstrate through tests only.** Rejected. Every closeout in this project
  quotes a real command against a real directory, and the two times a milestone's proof was
  only a test, the test turned out to assert less than it claimed.
- **A hidden or `--dev` command.** The same code with a sign on it saying not to look. It is
  a stage in the spec's own DAG.
- **Let each command write its own report and merge them.** M2 already built and then
  deleted report-merge machinery for a better reason than convenience: re-deriving a stage's
  provenance in the same process is strictly stronger than carrying a previous run's forward,
  because it cannot be stale.
- **Read `work/timeline.json` and validate it.** Faster on paper, and it re-introduces the
  stale-artifact question M2 answered. A cached rebuild is not measurably slower.

## Consequences

- `activity` is a resumable entry point from a bare session directory: given only
  `session.yaml` and `raw/`, it inspects, reconstructs, derives, and attributes in one run.
- One report, one exit code, and partial success still never exits zero (INV-13). A failure
  in any composed stage leaves the later ones recorded as `skipped` with a reason naming the
  failure, never absent.
- The INV-01 snapshot covers the composed run end to end, so a source mutated during
  inspection is caught after activity has already read it — and no cache entry from that run
  is committed, because publication happens after verification (INV-08).
- M4's `transcribe` and M5's `mix` inherit the composition rather than repeating it. `process`
  in M5 runs activity **once** and hands the same graph to both branches, which is what
  INV-09 requires.
