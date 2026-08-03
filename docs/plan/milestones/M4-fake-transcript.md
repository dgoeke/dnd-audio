# M4 — End-to-end transcript with fake ASR

**Status:** closed
**Depends on:** M3
**Spec sections:** Milestone 3 (post-ASR duplicate collapse); Milestone 4; Output
schemas; Tests and acceptance criteria 6, 7, 9, 13, 14

## Goal

The full transcript branch working end to end on synthetic input with no Qwen,
no GPU, and no weights: segment request construction, normalized transcript
records, duplicate collapse, alignment fallback, and `transcript.json` +
`transcript.md` rendering, plus a `render` command that regenerates outputs from
cached records alone.

## Completion gate

- [x] Requests are built from retained activity candidates, not from six full-length
      files. Short adjacent regions merge; every request has padding for word
      recovery and an unpadded core/ownership interval.
- [x] The submitted **padded** waveform never exceeds `max_segment_s` (default 120).
- [x] Words are assigned to core intervals and boundaries stitched deterministically
      so padding cannot duplicate words or utterances.
- [x] Truncation handling: a faked length-stop response triggers a split at a
      natural low-energy boundary in the unpadded core, retries both halves with
      their own padding, and stitches deterministically. Retries are bounded; the
      original response plus a warning is retained when it cannot be resolved.
      No dependence on a private Qwen finish-reason API.
- [x] Post-ASR duplicate collapse requires substantial temporal overlap, strongly
      similar normalized text, **and** supporting acoustic evidence. Short/common
      utterances ("yes", "no") never collapse on text similarity alone. Materially
      different text or ambiguous evidence retains both, marked as overlap.
      Rejected alternatives are recorded.
- [x] Alignment failure on one segment retains the segment-level transcript and
      warns; it never fails the session.
- [x] The unmodified public ASR result is losslessly serialized to a versioned JSON
      artifact before normalization. No pickling.
- [x] `transcript.json` validates against its **checked-in** JSON Schema artifact —
      not merely round-tripped through the Pydantic class that produced it.
- [x] Public times serialize to millisecond precision with stable sorting
      tie-breakers; segment and candidate IDs derive from sorted source identity
      and time (INV-02).
- [x] `overlap` means overlapping another retained, non-duplicate speaker segment by
      at least the configured threshold.
- [x] Markdown renders in the specified format, sorted by start time, overlapping
      turns as separate entries, with user/model text escaped safely.
- [x] `render` regenerates both outputs from cached transcript records without
      loading any model or running the mixer, and fails clearly when records are absent.
- [x] Rerun on unchanged input hits caches and produces byte-stable
      `transcript.json` and `transcript.md` (INV-02).
- [x] No LLM prose cleanup. Only deterministic whitespace/punctuation normalization.

## Explicitly not in this milestone

- Any real model. The fake `Transcriber` is the only implementation exercised.
- Confidence values. Never manufacture one the model does not expose; keep
  signal-quality scores separate from model confidence.

## What M3 already provides (read before starting)

- **`work/activity.json` is frozen at schema version 1** (ADR-0012) — additive optional
  fields only, and every property name is held by a hardcoded allowlist in
  `tests/test_activity_artifact.py`. Adding a field is an ADR-0005 decision, and adding a
  **text-derived** one is an INV-09 violation that fails a test rather than changing a
  contract quietly.
- **Build requests from `decision == "retained"` candidates**, in `candidates` order — the
  document is already sorted by `(start_sample, track_id)` and its ids sort lexically in the
  same order. `test_activity_artifact.py::TestTheConsumerReads` is the worked example of both
  M4's and M5's access patterns, written before either milestone existed.
- **`ambiguous` does not mean "uncertain detection".** It means the numeric evidence said
  bleed — margin *and* correlation both satisfied — and the track-level veto overrode it
  (ADR-0014). Those are the candidates most likely to be a second copy of another track's
  utterance, so they are exactly the ones M4's duplicate collapse should look hardest at.
  It is *not* a reason to drop a candidate before ASR.
- **Every interval appears on both grids** — `start_sample`/`end_sample` at 48 kHz and
  `derivative_*` at 16 kHz — and there are no floats anywhere in the document. Integer
  per-mille and millibels throughout; `public_seconds` is still the only float-producing
  conversion in the project (INV-04).
- **INV-09 runs the other way here.** Nothing M4 decides may flow back into the graph, and
  the mix must produce identical samples whether or not ASR ran at all.

## Known risks and open questions

- **The field allowlist freezes names, not prose.** `ActivityDecision.detail` and
  `ActivityNote.message` are unrestricted strings on that allowlist, so a later stage could
  place text-derived content in either without adding a property, changing the activity
  package's imports, or failing any INV-09 test. Nothing may write ASR-derived text into
  them. Raised by independent review in M3's verify phase and deferred there; see
  `../reviews/M3-code-20260802-1708.md`.

- Depends on **OQ-009** for the eventual real segment limits, but M4 must be
  correct under the configured limit regardless. `config.py` caps `max_segment_s` at 120
  and cites the OQ at the cap; move the cap there if the answer changes.
- **The transcript schema version is provisional until this milestone closes.** Change
  version 1 freely while M4 is open; after it closes, only additive optional fields
  (ADR-0005). `tests/data/transcript-spec-example.json` is the spec's own example held as
  independent ground truth — if a change makes it stop validating, the change is wrong.
- The fake transcriber M0 provides is **scripted**, not content-derived: hand it the
  truncation, alignment-failure, and overlapping-utterance responses this milestone needs
  to exercise. `ScriptedTranscriber.requests` records what it was asked, which is how the
  "no padded waveform exceeds `max_segment_s`" assertion is written.
- Duplicate collapse is where the pipeline is most likely to silently delete real
  speech. Bias every ambiguous case toward keeping both and marking overlap.
- INV-09: nothing decided here may flow back into the activity graph.

## Closeout

### What works end to end

`uv run dnd-audio transcribe /path/to/session --fake-models` — the whole left branch of the
spec's stage DAG in one command: inspect, reconstruct, activity, ASR, transcript render.

It snapshots the raw roots once for the composed run, performs the activity stages through
`perform_activity`, verifies INV-01 and commits four caches, writes the graph — then plans
requests from that graph's **retained** candidates only, submits one padded window at a time,
resolves truncation within a bounded budget, assigns each word to the ownership interval
containing its start, normalizes, collapses duplicates, verifies INV-01 a second time, commits
the ASR cache, and writes `work/transcript-records.json`, `output/transcript.json`,
`output/transcript.md` and one `output/ingest-report.json` covering five stages.

On the canonical fixture:

```
4 segment(s) across 4 speaker(s), 0 collapsed as duplicates, 2 marked as overlap
warn  fake_models_in_use: ... Every text in this transcript was written by whoever
      generated the fixture (ADR-0018).
```

```markdown
# Session 01

**[00:00:05.200] Alice:** We should go back to Zephyrine.

**[00:00:06.800] Dan [overlap]:** Absolutely not.

**[00:00:06.800] Erin [overlap]:** Wait, say that again?

**[00:00:08.500] Carol:** Sorry, my transmitter was off.
```

Four utterances, four tracks, the two genuine simultaneous speakers both marked. Alice's line
bleeds into four other tracks and the scripted ASR is deliberately told to transcribe it
there — every copy is gone before a word is submitted, because M3's gate suppressed the
candidate. That is what "transcribe retained segments rather than six full-length files" buys,
and it is asserted rather than admired (`test_bleed_never_reaches_the_transcript`).

A second run reports **29 cache hits, 0 misses**, and all three deterministic artifacts are
byte-identical (`4eca3424…`, `f3e8f524…`, `c52a47c4…`).

`uv run dnd-audio render /path/to/session` regenerates both deliverables from the records
alone: `rendered 4 segment(s) from cached records`. It is proved rather than asserted — the
test deletes the graph, the timeline, and the entire cache tree first, and a spy proves no
model is constructed.

Without `--fake-models`, `transcribe` raises the `DEFERRED: M6b` `NotImplementedError` naming
the missing adapter. That is deliberately **not** turned into a failed report: "this pipeline
has not built that yet" and "your session is broken" are different answers to different
questions (ADR-0005).

`activity`, `ingest`, `inspect`, `models fetch`, `doctor` and `make_fixture.py` are unchanged
in behaviour, except that all three composed runners had an INV-01 bug fixed (below).
`mix` and `process` remain registered stubs exiting 3.

### Tests and commands run, with results

```
./scripts/gate.sh
  pass  system dependencies      pass  lock is current
  pass  ruff check               pass  placeholder scan
  pass  ruff format              pass  plan consistency
  pass  type check               pass  pytest (offline, cpu) — 1768 passed, 3 deselected
GATE PASSED
```

The 3 deselected are the same three marked tests M3 closed with. No `skip` or `xfail` anywhere.

M4's own files, run during verify:

```
test_transcript_requests   23      test_transcript_cache      31
test_transcript_asr        21      test_transcript_records    25
test_transcript_segments   27      test_transcript_render     35
test_transcript_collapse   26      test_transcript_run        37
test_transcript_normalize  30      test_raw_guard             17
```

**Mutation testing was the verify phase's main instrument**, because a passing test is not
evidence it can fail. 27 deliberate regressions applied to the implementation, suite run,
source restored. 24 were caught. The three survivors were all real coverage holes, and two of
them turned out to be genuine defects seen from the other side (below). After the fixes, every
one of the ten mutations covering new behaviour is caught:

| Mutation | Caught by |
| --- | --- |
| Weakest correlation → strongest | `TestASegmentCoveringSeveralCandidates` |
| Word ownership by start → by overlap | `TestAWordBelongsToTheIntervalContainingItsStart` |
| Cross-piece dedup removed | `TestAdjacentPiecesDoNotDuplicateAWord` |
| Dedup ignores adjacency | `test_a_word_whose_end_reaches_across_a_gap_still_keeps_both` |
| Markdown timestamp via float truncation | `TestTheMarkdownTimestampIsExactToo` |
| Similarity in one direction only / `max` not `min` | `TestSimilarityIsSymmetric` |
| Cleanup before the INV-01 carve-out, in each of three runners | `TestCleanupNeverWritesIntoRaw` |
| A completed stage's artifacts deleted | `TestAPartialRunReportsOnlyWhatSurvived` |

Earlier in the sweep, and still caught: dropping any one of collapse's three conditions,
picking the survivor by text length, planning suppressed candidates, a cap that ignores
padding, never merging adjacent candidates, keeping the boundary repeat, a per-level rather
than global retry budget, a non-atomic truncation fallback, a midpoint split, an unbounded
retry, no `min_split_core_ms` floor, submitting the core instead of the padded window, no
cache size check, request identity out of the key, publishing at commit time, the INV-09 graph
re-hash removed, no Markdown escaping, and the canonical draft sort removed.

**Independent review**: `../reviews/M4-code-20260802-1942.md`. Codex's verdict was *"I would
not close M4"* — eight findings, every one reproducible as described. A second fresh-context
reviewer found one of the same defects independently, from the ADR text rather than the code.
Seven findings fixed, one deferred with reasons, one accepted while its reasoning was
rejected, one out of scope.

Live, on the canonical fixture: byte-stable across two runs, 29 hits / 0 misses warm, five
stages complete and `mix` skipped with a reason, `render` regenerating both outputs from the
records with the graph and caches deleted.

### Decisions made (→ ADRs)

Four recorded before any code was written, one after the review:

- **[ADR-0017](../decisions/0017-the-asr-grid-and-request-ownership.md)** — ASR consumes the
  cached 16 kHz derivative, and requests merge without merging ownership. One retained
  candidate produces one segment; `source_candidate_id` stays singular except in the one case
  that cannot be divided.
- **[ADR-0018](../decisions/0018-session-declared-fake-models.md)** — a session may declare
  its own fake model outputs in `fake-models.json`. Explicit flag, fatal if absent, digest of
  the script in the cache key and the report, and a `fake_models_in_use` warning on every run.
- **[ADR-0019](../decisions/0019-the-transcript-records-artifact.md)** — the records artifact,
  segment identity as position in canonical order, and what the ASR cache key contains.
- **[ADR-0020](../decisions/0020-word-ownership-and-bounded-retry.md)** — who owns a word at a
  boundary, and "bounded" meaning a global submission budget rather than a recursion depth.
  **Its claim that a truncation stitch is the only adjacent boundary was wrong**; the verify
  phase corrected it in code and in the ADR's own prose.
- **[ADR-0021](../decisions/0021-cleanup-ordering-and-per-commit-cache-scope.md)** — failure
  cleanup runs after the INV-01 carve-out in every runner, a completed stage keeps its
  artifacts, and INV-08's test prescription is scoped to a commit point rather than a run.
  This one amends behaviour owned by M2 and M3.

**INV-01 and INV-08 were both amended** in `INVARIANTS.md`, each with the reason and the
milestone that found it.

### Assumptions made and open questions raised

**OQ-018 raised** — what Qwen3-ASR and its aligner need at a request boundary. Four guesses,
each of which M4 had to make a number out of before any model existed to check it against:
padding sufficient for word recovery; timestamp stability across two overlapping requests
(without it the stitch rule stops recognizing a duplicate and emits it twice); truncation
being worth retrying as two halves split at the quietest point, within the configured budget;
and the text-similarity thresholds, which are calibrated against *Qwen's* error distribution.
Every request-shaping and text default in `TranscriptConfig` cites it, plus
`SPLIT_FRAME_SAMPLES`, so `rg 'OQ-018'` finds all twelve sites at once. **M6b's smoke test can
settle the first three directly.**

Nothing was answered. OQ-009 is still cited at the `max_segment_s` cap; OQ-017 still owns the
acoustic half of the duplicate thresholds. **No open question was closed by this milestone,
because none of them can be closed without a model or a room.**

### Notes for future implementors

**Mutation testing here needs `PYTHONDONTWRITEBYTECODE=1`.** This cost real time and produced
two confidently wrong conclusions before it was caught. A same-length source edit (`min` →
`max`) applied and restored within one second leaves `.pyc` bytecode that CPython's
`(mtime, size)` invalidation cannot distinguish from the original, so the interpreter keeps
running the mutant — including in every *subsequent* test run until something else touches the
file. The symptom is a function whose `inspect.getsource` is demonstrably correct returning a
demonstrably wrong answer. Set the variable, or `find -name __pycache__ -exec rm -rf {} +`
between mutations.

**The INV-01 cleanup bug was in all three composed runners at once**, and had been since M2.
Every runner deletes stale artifacts on failure and every runner has the carve-out that
refuses to write when an output path resolves inside `raw/`. They were in the wrong order, so
one `work -> raw/tx-a` symlink turned the correct detection of a violation into a deletion
under `raw/`. Each of M2, M3 and M4 tested the *report* carve-out sitting immediately beside
the bug, and none tested the cleanup — because each wrote a regression test naming the runner
that milestone had added. **This is verbatim the lesson INV-08 already records about caches.**
The pattern to copy is `TestCleanupNeverWritesIntoRaw`: parametrize every composed command
from one place, in the file that owns the invariant's machinery rather than in any one
runner's tests. A runner M5 adds is then one missing parameter, which is visible in review.

**A milestone's own gate criteria are not a list of things to test — they are a list of things
to test *the negation of*.** "Padding cannot duplicate words" was implemented correctly, had a
passing test, and was still broken on a second code path nobody had thought about. What found
it was mutating the rule and noticing the suite did not care, plus a reviewer reading ADR-0020's
word "only" and checking whether it was true. Both are cheap. Neither is a test you write while
implementing.

**`ambiguous` still does not mean "uncertain".** M3's closeout said this and it stayed true:
those are the candidates whose numbers said bleed and whose track-level veto overrode them.
They are always planned and always transcribed; they are the ones collapse should look hardest
at. Nothing in M4 skips a candidate before ASR.

**Collapse is the function to be frightened of.** Two of the four correctness defects found in
verify were in it or fed it, and both deleted speech. The invariant to hold onto: every
ambiguous case keeps both and marks overlap, and every new condition should be able to say
which direction it errs in. `_weakest_correlation` returning `None` is not a gap — it is the
answer "there is no evidence about *these two segments*", which now includes the case where
the graph measured some of a merged segment's candidates and not others.

**`difflib.SequenceMatcher.ratio` is not symmetric**, and the asymmetry can be ~290‰ wide.
Anywhere its output crosses a threshold that deletes data, take the minimum of both
directions. The pair that demonstrates it is kept verbatim in `TestSimilarityIsSymmetric`
rather than reduced to something tidier, because something tidier would not have caught it.

**Two commit points, and why it is not an INV-08 violation.** The activity caches commit after
the first verification and the ASR cache after the second, so an ASR failure — which reads no
source audio — does not discard six tracks of inference. The invariant is "commit only after
verifying", and both points satisfy it. What had to change was the invariant's *test
prescription*, which was written for a single-commit run (ADR-0021). If you find yourself
writing a test whose name promises "anywhere" over a body that checks one directory, that is
the smell.

**A partial run now keeps the artifacts of the stages that completed.** This is load-bearing
for M5: after a failed `transcribe` the graph is still on disk, so `mix` can run against it.
`ReportBuilder.completed` is deliberately distinct from `recorded` — `recorded` answers INV-13's
no-gaps question, `completed` answers cleanup's.

**The records artifact's validators are where the real invariants live.** `document.py` decides
nothing; it hands `TranscriptRecords` the whole picture and the model refuses states that
would make a transcript lie — a duplicate naming nothing, a chain of duplicates, a word
outside its ownership interval, a collapsed segment also marked overlapping. That is why the
word-start clamp in `segments._record` exists and is not cosmetic: the graph's 48 kHz interval
*covers* its derivative one, so converting the first derivative sample back lands up to two
samples before the candidate starts, and the artifact correctly refuses it.

**`tests/data/transcript-spec-example.json` is the spec's own example, byte for byte.** It is
the one piece of ground truth in this milestone that no code here produced. If a change makes
it stop validating, the change is wrong.

### Deferred, with the reproducing case

**A three-way duplicate group can keep a survivor that is not the best source.** With three
mutually-duplicate segments scoring A=800, B=700, C=900 in canonical order, A absorbs B first
and is then forbidden from being absorbed by C — `collapse.py` refuses to let a segment that
has already absorbed another be absorbed itself, because a chain of duplicates has no
surviving text at the end of it. A and C both reach the transcript, and `collapse.py`'s own
docstring says the survivor is the one with the best source score.

Reproduced during verify; not fixed, deliberately. The failure is in the **safe** direction:
it keeps both and marks them overlapping, which is the bias this milestone states outright and
which the gate criterion permits in as many words ("ambiguous evidence retains both, marked as
overlap"). The cost is a duplicated line in a transcript; the fix is a chain-resolution pass
inside the most dangerous function in the milestone, to make an already-safe outcome tidier.

**M6b should revisit it** once real ASR output shows whether three lavs ever agree closely
enough for the shape to occur at all. If they do not, the right change is to delete the
docstring's claim rather than to write the pass.

### Deviations from this charter, and why

- **ADR-0020's "only place" claim was wrong** and is corrected in code, in the ADR, and in the
  proof table. `requests._divide` produces adjacent ownership intervals too.
- **INV-08's glob prescription was amended** rather than followed, because M4's two commit
  points make its literal wording false for a correctly-behaving run (ADR-0021). The rule is
  unchanged.
- **Three runners changed, two of them owned by closed milestones.** Fixing only M4's would
  have left a demonstrated violation of the project's hardest rule at HEAD.
- **The working plan's proof table named six test classes that were never written** under those
  names. The equivalents all exist; the table now points at them. A scratch section is allowed
  to drift, but the charter is the durable record, so it was reconciled rather than deleted.
- **`similarity_permille` takes the lower of two directions**, which the plan did not
  anticipate needing.

### Downstream charters updated

- **M6b** gained a "What M4 already provides" section during implementation — the finished
  seam, the 16 kHz grid, `alignment_status` being stated rather than inferred, the
  `public_document` half M6b still owes, the cache key it extends, budget-bounded truncation,
  the frozen artifacts, and OQ-018 being its to answer. Extended at close with the deferred
  three-way collapse case.
- **M5** gained a "What M4 already provides" section: the runner patterns its own composed
  command must copy (cleanup after the carve-out, `completed` artifacts kept, the parametrized
  INV-01 test it must add a parameter to), and the confirmation that INV-09 holds in the
  direction M5 enforces.
- **INVARIANTS.md**: INV-01 and INV-08 amended, each naming the milestone that found it.

### Next smallest step

Begin **M5 — Automix**. It depends on M3 only, never M4, and the graph M4 consumed is
unchanged by anything M4 decided — asserted by a re-hash inside the composed run and by a
structural import test. Start with the gain envelopes, because the envelope-level assertions
are the real gate and the loudness work is meaningless without them.

Read M5's two "What M2/M3 already provides" sections and its new "What M4 already provides"
one first. The trap M4 would flag: `TestCleanupNeverWritesIntoRaw` in `tests/test_raw_guard.py`
needs a `mix` parameter the moment `run_mix` exists, and the reason it is parametrized is that
three milestones in a row each tested only their own runner and all three had the same bug.
