# ADR-0021 — Failure cleanup runs last, and INV-08's scope is per commit point

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M4 (amending behaviour owned by M2 and M3)

## Context

M4's verify phase found two things about what a *failed* run leaves behind. Both were found
by independent review, and neither was visible from inside the milestone that introduced it.

### Cleanup was running before the INV-01 carve-out

Every composed runner deletes the artifacts a failed run may have written, so that a stale
`timeline.json` cannot sit beside a report calling reconstruction failed. Every composed
runner also has the carve-out INV-01 requires: when an output path resolves inside a source
directory, nothing is written and the run returns without a report, because writing the
failure report would commit the very violation being reported.

The two were in the wrong order. `run_ingest`, `run_activity` and `run_transcribe` all ran
their cleanup *first* and checked for `output_inside_raw` *afterwards*. With a single
`work -> raw/tx-a` symlink — M1's exact defeat, the one the resolved-path comparison exists to
catch — `session_dir / "work" / "timeline.json"` resolves to `raw/tx-a/timeline.json`, and the
cleanup unlinks it. The run that correctly detected the violation committed it on the way out.

Reproduced against all three commands: a file written into `raw/tx-a/` before the run was gone
after it, with `exit_code = FATAL` and `report_written = False` — the report suppressed to
protect a directory the same code path had just deleted from.

The report carve-out was tested. The cleanup was not, in any of the three, because each
milestone's regression test named the runner that milestone had added. This is the lesson
INV-08 already records about caches, arriving a second time about deletions.

### INV-08's "no sidecar anywhere" describes a run M4 does not have

INV-08 says an entry is committed only after INV-01 has been re-verified, and prescribes the
test: *"Assert that a failed run leaves no sidecar anywhere under `work/cache`, by glob rather
than by naming the caches you know about."*

That wording assumes one commit point. M4's composed run has two (see the M4 charter's
decision 7): the activity caches commit after the first verification, the ASR cache after the
second. The reason is cost — an ASR failure reads no source audio and cannot invalidate six
tracks of detection and attribution, and discarding them would make an unrelated failure
expensive. So a failure during ASR legitimately leaves activity sidecars committed, and the
glob assertion is false for a run that is behaving correctly.

The test M4 shipped read as though it enforced the invariant. Its name was
`test_a_failed_run_leaves_no_sidecar_anywhere_under_the_cache`; its body globbed the ASR
directory only and then asserted that the activity sidecars *do* remain. A reader checking
whether INV-08 was still enforced would have taken the name for the answer.

## Decision

### 1. Cleanup runs after the `output_inside_raw` carve-out, in every runner

The carve-out returns first. Nothing is unlinked on the path where an output resolves inside a
source directory, because on that path the unlink *is* the violation. INV-01 outranks the
stale-artifact rule for the same reason it outranks INV-13: a stale artifact is confusing and
regenerable, a source file is neither.

`tests/test_raw_guard.py::TestCleanupNeverWritesIntoRaw` drives all three composed commands
from one parametrized test, deliberately in the file that owns INV-01's machinery rather than
in any one runner's tests. A new runner that forgets the ordering is one missing parameter,
which is visible; a new runner with no test at all is what happened three times already.

### 2. An artifact belonging to a stage the report calls `complete` is not deleted

`remove_activity_artifacts` takes the stages that completed and skips their artifacts. The
composed run records `work/timeline.json` and `work/activity.json` as hashed deliverables at
the first commit point; deleting them on a later ASR failure left the report advertising the
hash of a file that was gone, which is the opposite of what INV-13 asks a report to be.

The rule is symmetric and stated once: **a stage that completed keeps its artifacts, a stage
that did not keeps nothing.** A partial run therefore has a timeline and a graph on disk and a
report that names them, and no transcript, no records, and a report that does not.

### 3. INV-08's test prescription is scoped to a commit point, not to a run

Amended wording, in `INVARIANTS.md`: a failed run leaves no sidecar for any cache whose
verification did not happen, asserted by glob over the caches downstream of the last
successful commit point. A cache committed after a verification that *did* happen was built
from bytes that run confirmed, and keeping it is correct — the hazard the original wording
exists to prevent is an entry keyed on bytes nobody checked, which two commit points do not
create.

The glob technique is preserved, because the failure it was written for is real: naming the
caches you know about is how a composed run's new cache goes unchecked. What changed is the
region globbed, not the method.

## Consequences

- Three runners changed on a milestone that owns one of them. `timeline/runner.py` is M2's and
  `activity/runner.py` is M3's; both are closed, and both had the INV-01 ordering bug. Fixing
  M4's alone would have left a demonstrated violation of the project's hardest rule at HEAD.
- `ReportBuilder.completed` is new, and is deliberately distinct from `recorded`. `recorded`
  answers "does this stage have an outcome" for INV-13's no-gaps rule; `completed` answers
  "did it succeed", which is the question cleanup has to ask.
- A partial run is now more useful: the graph survives, so `mix` can still run against it and
  a re-run of `transcribe` starts from committed caches. That was already the intent of two
  commit points and was silently undone by the cleanup.
- INV-08's amendment is a narrowing of a *test prescription*, not of the invariant. The rule —
  commit only after verification — is unchanged and still holds at both commit points.
