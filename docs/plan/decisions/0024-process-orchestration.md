# ADR-0024 — `process`: one snapshot, two branches that fail independently, one report

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M5

## Context

The spec names `process` as the single user-facing command and defines it precisely:

> `process`: dependency-aware orchestration of all applicable stages. Run activity once,
> attempt both downstream branches independently, render the transcript branch when
> transcription succeeds, and always finalize the structured report. A failed transcription
> branch must not cancel or skip the mix branch.

and, in the error-handling section:

> If ASR fails but mixing succeeds, retain the MP3 and report, mark the transcript stage
> failed, and make the top-level `process` command exit nonzero so automation cannot mistake
> partial output for full success.

INV-09 is the same requirement from the other side, and M5's completion gate names `process`
outright. It could not have been built before now: it is the first point at which both
downstream branches exist.

Four M4-era facts constrain the shape. `perform_activity` already exists as the composable
half of `activity`, so a third caller composes rather than reimplements (ADR-0015's argument).
ADR-0021 established that a composed run may commit caches at more than one point, and that
failure cleanup runs *after* the `output_inside_raw` carve-out, never before it.
`ReportBuilder.completed` distinguishes "this stage's artifacts are real and already hashed"
from "this stage has an outcome". And `tests/test_raw_guard.py::TestCleanupNeverWritesIntoRaw`
is parametrized over every composed command precisely so that a new runner is one missing
parameter rather than one missing test file.

## Decision

### One snapshot, activity once, mix first, transcript second

`process` snapshots the raw roots once for the whole run, refuses outputs that would land
inside them, and performs the activity stages exactly once through `perform_activity`. Both
downstream branches then read the graph that pass produced.

The mix branch runs first. Not because ordering implies independence — it does not, and the
next paragraph is what does — but because it makes "the mix cannot have consumed anything the
transcript branch produced" true by construction as well as by test, and because it is the
branch the spec says must survive.

### Independence is a property of the control flow, not of the ordering

Each branch runs in its own handler. A failure in either records a structured error against
that branch's stages and **continues to the other**; neither can short-circuit the run. A
sequential mix-first implementation that lets a mix exception propagate satisfies every
sentence about transcription failure in the spec and still violates the requirement, which is
why this is stated as a decision and why four tests assert it rather than one: activity
executes exactly once; a transcript failure leaves the MP3 present and hashed; a **mix**
failure does not cancel transcribe or render; and either failure accounts for every stage and
exits nonzero.

`render` runs only when transcription succeeded — "render the transcript branch when
transcription succeeds" — and is recorded as failed, with the transcript branch's error, when
it did not.

### Three commit points, and one unconditional final verification

Caches are committed at three points — the activity caches after the activity pass, the mix
cache after the mix, the ASR cache after transcription — each immediately after a
`verify_unchanged` over the raw roots. ADR-0021 already scopes INV-08 to a commit point rather
than to a run, and the mix's own point is load-bearing rather than incidental: **the mix is
the only stage after inspection that reads source audio.** Transcription reads the cached
16 kHz derivatives (ADR-0017) and can therefore never invalidate a source hash; the mix reads
the 48 kHz originals through `TrackReader` and can.

Beyond those three, `process` performs **one unconditional `verify_unchanged` before report
finalization, on the success path and the failure path alike**. With three commit points, a
transcript failure occurring before the ASR commit otherwise means nothing re-verifies the
sources after the mix has read them, and INV-01's guarantee is "hashed before and after a
complete run", not "before and after each cache write". A violation found there is a fatal
error recorded on the report; it does not replace whichever error the branch already reported.

### Failure cleanup, in the order ADR-0021 fixed

When an output path resolves inside a source directory, `process` returns **before** any
cleanup, writing no report — the unlinks would themselves be the INV-01 violation being
reported, and INV-01 outranks INV-13 there. Otherwise the artifacts of every stage that did
**not** complete are removed, and those of every stage that did are kept, because their hashes
are already in the report and deleting them would leave it advertising a file that is gone.

### Where it lives

`src/dnd_audio/orchestrate.py`, not inside either branch's package. Putting it in
`mix/runner.py` would make the mix package import the transcript package and break the
structural half of INV-09 in the most literal way available; putting it in
`transcript/runner.py` inverts the dependency the spec draws. It composes `perform_activity`,
`perform_mix` and the transcript branch's own composable half — which M5 exposes for this
purpose, since M4 left it private.

## Alternatives considered

- **Run the two branches concurrently.** They are genuinely independent and the spec's DAG
  draws them in parallel. Rejected: this is a UMA host where memory pressure kills processes
  (INV-07), the mix streams six 48 kHz tracks while ASR would hold a model, and the report's
  stage ordering would then need to be defended against completion order — which is exactly
  the nondeterminism `STAGE_ORDER` exists to keep out. Sequential costs wall-clock and nothing
  else.
- **`process` as a thin shell calling `run_mix` and `run_transcribe`.** Much less code.
  Rejected: each would snapshot and verify the raw roots independently (hashing every source
  twice), each would write its own report over the other's, and activity would run twice —
  three of the spec's four requirements for this command, broken.
- **Short-circuit on the first branch failure**, reporting the rest as skipped. Rejected
  outright; it is the failure the spec's last sentence exists to forbid.
- **One commit point at the very end.** Simpler to reason about, and it throws away six tracks
  of verified inference plus a completed mix whenever ASR fails — the argument ADR-0021
  already made for two points, with a third now on the same footing.
- **Skip the final verification, on the grounds that each commit point already verified.**
  Rejected: the composition of three local checks is not the global claim INV-01 makes, and
  the gap is reachable — a transcript failure before the ASR commit leaves the post-mix window
  unchecked.

## Consequences

- `process` is the first command in the project that can exit `PARTIAL` from a genuinely
  half-successful run rather than from a refusal, which is what INV-13's exit code 4 was
  reserved for.
- The transcript branch's composable half becomes public API of `transcript.runner`. That is a
  change to closed-milestone code, of the same kind and for the same reason as M2's
  `raw_guard` extraction and M3's cache-ordering fix.
- Four commit/verification points make the INV-08 glob region a per-test question. Each test
  says in its name which region it globs, which is the prescription M4's verify phase wrote
  into the invariant after finding a test whose name promised "anywhere" over a body checking
  one directory.
- Adding a seventh stage later means one more entry in the branch table here and one more
  parameter in `TestCleanupNeverWritesIntoRaw`. Forgetting the second is visible in review,
  which is the whole reason that test is parametrized from one place.
