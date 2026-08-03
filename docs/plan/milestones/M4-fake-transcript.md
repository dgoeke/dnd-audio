# M4 — End-to-end transcript with fake ASR

**Status:** not started
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

- [ ] Requests are built from retained activity candidates, not from six full-length
      files. Short adjacent regions merge; every request has padding for word
      recovery and an unpadded core/ownership interval.
- [ ] The submitted **padded** waveform never exceeds `max_segment_s` (default 120).
- [ ] Words are assigned to core intervals and boundaries stitched deterministically
      so padding cannot duplicate words or utterances.
- [ ] Truncation handling: a faked length-stop response triggers a split at a
      natural low-energy boundary in the unpadded core, retries both halves with
      their own padding, and stitches deterministically. Retries are bounded; the
      original response plus a warning is retained when it cannot be resolved.
      No dependence on a private Qwen finish-reason API.
- [ ] Post-ASR duplicate collapse requires substantial temporal overlap, strongly
      similar normalized text, **and** supporting acoustic evidence. Short/common
      utterances ("yes", "no") never collapse on text similarity alone. Materially
      different text or ambiguous evidence retains both, marked as overlap.
      Rejected alternatives are recorded.
- [ ] Alignment failure on one segment retains the segment-level transcript and
      warns; it never fails the session.
- [ ] The unmodified public ASR result is losslessly serialized to a versioned JSON
      artifact before normalization. No pickling.
- [ ] `transcript.json` validates against its **checked-in** JSON Schema artifact —
      not merely round-tripped through the Pydantic class that produced it.
- [ ] Public times serialize to millisecond precision with stable sorting
      tie-breakers; segment and candidate IDs derive from sorted source identity
      and time (INV-02).
- [ ] `overlap` means overlapping another retained, non-duplicate speaker segment by
      at least the configured threshold.
- [ ] Markdown renders in the specified format, sorted by start time, overlapping
      turns as separate entries, with user/model text escaped safely.
- [ ] `render` regenerates both outputs from cached transcript records without
      loading any model or running the mixer, and fails clearly when records are absent.
- [ ] Rerun on unchanged input hits caches and produces byte-stable
      `transcript.json` and `transcript.md` (INV-02).
- [ ] No LLM prose cleanup. Only deterministic whitespace/punctuation normalization.

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

## Working plan

_Scratch section, written during the start phase and replaced by the Closeout when the
milestone ends. Preserved in the commit history from there._

### Preconditions

Working tree clean at `c88cb39`; M3 `closed` at `38bc989`; `./scripts/gate.sh` green at
HEAD — 8 checks, 1503 passed, 3 deselected. Branch `milestone/M4-fake-transcript`.

### What the code already gives M4 (read, not inferred from the ledger)

- `ActivityGraph.retained(track_id=None)` — the consumer read, already exercised by
  `tests/test_activity_artifact.py::TestTheConsumerReads`.
- **The acoustic evidence duplicate collapse needs is already in the graph.**
  `activity/bleed.py::compare_pairs` measures *every* overlapping cross-track pair, and
  `CandidateEvidence` carries `correlation_permille`, `lag_derivative_samples`,
  `score_margin_permille`, and `level_delta_mb` on both sides. M4 reads those rather than
  correlating audio a second time; reading the graph cannot violate INV-09.
- `interfaces.TranscriptionRequest` already models padded audio plus a core interval and
  validates containment; `fakes.ScriptedTranscriber.requests` records what was asked.
- `timeline/resample.py::to_source_sample` / `to_derivative_interval` — the 48↔16 kHz
  contract. Use them; do not re-derive the floor/ceil asymmetry.
- `activity/cache.py` — the cache pattern to copy exactly: data first, sidecar *staged*,
  committed only after INV-01 is re-verified, size-checked on read.
- `activity/runner.py::run_activity` — composed, not reimplemented.

### Amended after the plan review

`../reviews/M4-plan-20260802-1824.md` holds the independent critique this plan was revised
against, with the implementor's response to each finding. Its verdict was *"not ready to
implement as written"*: six findings accepted, one accepted while its reasoning was rejected,
one rejected, and all three over-building notes accepted. Decisions 6, 7 and 8 below and
several rows of the proof table exist because of it.

### Decisions taken before any code (each becomes an ADR)

1. **ASR consumes the 16 kHz derivative, not the 48 kHz working path.** Qwen3-ASR ingests
   16 kHz mono, the derivative is already built through one checked-in FIR and is cached and
   byte-stable, and resampling at ASR time would put a *second* resampler in the project —
   the failure mode INV-04 names for time and ADR-0011 names for audio.
2. **Session-declared fake models.** `build_session` writes the fixture's already-declared
   truth — fake-VAD spans and fake-ASR utterances with word times — to
   `<session>/fake-models.json`, which is what the spec's fixture recipe asks a fixture to
   carry. `transcribe --fake-models` loads it behind the existing INV-10 seams; without the
   flag the transcriber resolver raises `NotImplementedError` annotated `DEFERRED: M6b`, the
   same shape as `_silero_bundle`. The flag is explicit, the file must exist, both artifacts
   and the report record a scripted identity with a digest of the script, and a
   `fake_models_in_use` warning is emitted. _Scoped to `transcribe` only after the plan
   review: `activity --fake-models` would change a closed milestone's user-facing surface
   without serving this gate, and `run_transcribe` injects the scripted detector through the
   seam that already exists._
3. **`segment_id` is `seg_%06d` over the canonically sorted order** (start sample, then
   track) — derived from sorted source identity and time as the spec requires, and it keeps
   the spec's own `seg_000123` example valid. A `cand_`-style id would break the checked-in
   ground truth in `tests/data/transcript-spec-example.json`.
4. **`work/transcript-records.json` is the render input**, versioned, byte-stable, with a
   checked-in JSON Schema. Collapse and overlap marking happen *before* it is written, so
   `render` is a pure function of it and provably needs no model, no graph, and no mixer.
5. **`TranscriptionResult` gains `alignment_status` and `public_document`.** Word presence
   alone cannot distinguish "the aligner ran and failed" (`segment_only` plus a warning) from
   "no aligner ran" (`not_attempted`); ADR-0005 named all three states and only the adapter
   knows which. `public_document` is the adapter's lossless serialization of its **backend's**
   public result — `None` for a transcriber whose result already *is* its public form, which
   is every fake M4 has — and the raw artifact is an envelope recording which of the two it
   holds. M4 freezes and tests that JSON-preservation contract; **M6b** proves its adapter
   fills it from every public `ASRTranscription` field, which M4 cannot demonstrate with Qwen
   out of scope.
6. **Requests merge; ownership does not.** A request's padded window and core may span several
   adjacent candidates, but every candidate keeps its own ownership subinterval and words are
   assigned to the subinterval containing their start. One retained candidate produces one
   segment, so `source_candidate_id` stays singular as the spec's baseline has it, "keep the
   best `score_permille`" is unambiguous, and collapse reads the exact pairwise evidence M3
   measured. The one case that cannot be split — a wordless `segment_only` result spanning
   several candidates — emits a single record carrying every contributing candidate id, so
   the records artifact holds `source_candidate_ids` as a list that is length one in every
   ordinary case, and collapse then requires **every** existing cross-pair to meet
   `min_correlation` rather than the best one.
7. **`activity/runner.py` splits into a composable core.** `perform_activity` builds, detects,
   attributes, and returns **staged** caches; `run_activity` keeps the snapshot, the
   verification, the commit, the report, and the CLI's failure handling. A composed run then
   hashes `raw/` once for one snapshot, checks output paths once over the union of both
   stages' outputs, and writes one report covering five stages. **Two commit points are kept**
   — activity after the first verification, ASR after the second — rather than the single
   transaction the plan review recommended: one transaction would discard verified, expensive
   inference caches because something unrelated failed later. The cost is a third hash pass
   over `raw/`; the benefit is that an ASR failure never costs six tracks of re-detection.
8. **The ASR cache key includes the request's identity** (track and core interval) alongside
   the audio hash, the transcriber identity, the context hash, the language, and
   `max_new_tokens`. INV-08 requires a key to *include* the spec's list, not to be limited to
   it, and `config.py`'s own bias applies: a too-broad key costs recomputation, a too-narrow
   one is silent. It also stops a scripted fake — which selects on `request_id` and is
   therefore not a function of its audio — from turning a false cache hit into a test that
   lies. The records/render version deliberately stays **out** of this key, so re-rendering
   never costs a re-transcription.

### Files, in implementation order

**A. Contracts** — `config.py` (new `transcript:` section: `pad_ms`, `merge_gap_ms`,
`overlap_min_ms`, `max_truncation_retries`, `min_split_core_ms`, and `duplicate:` with
`min_overlap_ratio`, `min_text_similarity`, `min_text_words`, `min_text_chars`,
`min_correlation`, `min_score_margin`; request-shaping defaults citing **OQ-018**, acoustic
ones OQ-017; classified in `_FIELD_SCOPES` as reaching none of the four cached stages, because
the ASR cache builds its own identity); `interfaces.py`; `artifacts/records.py` (the records
artifact and `TranscriberIdentity`); `schema_export.py` plus regenerated `schemas/`;
**OQ-018 registered in `OPEN-QUESTIONS.md` before any default lands**.

**B. Requests** — `transcript/requests.py`: retained candidates in graph order → merge
adjacent cores within `merge_gap_ms` per track, **preserving each candidate's ownership
subinterval through the merge** → split any core padding would push past `max_segment_s` →
`RequestPlan` carrying ids, intervals, ownership lineage, and **no audio**, so nothing
materializes six tracks at once. Audio is attached per request at submit time.

**C. ASR** — `transcript/cache.py` (identity: segment-audio sha256, request identity,
transcriber identity, context hash, language, `max_new_tokens`, ASR semantics and record
versions; the raw envelope at `work/cache/asr/<key>.raw.json`; sidecar staged and committed
with everything else); `transcript/asr.py` (submit; truncation → split the *unpadded core* at
the lowest-energy interior frame → retry both halves with their own padding → deterministic
stitch, under a **global budget of `max_truncation_retries` extra submissions per original
request** with a minimum child-core length, every child capped like any other, and an **atomic
fallback** to the original response plus `asr_truncation_unresolved` if any descendant is
still truncated; words assigned to the ownership subinterval containing their start, with a
duplicate-word rule at truncation stitch boundaries and the wordless `segment_only` fallback).

**D. Text** — `transcript/normalize.py` (deterministic whitespace and punctuation only, plus
the comparison key) and `transcript/collapse.py` (overlap ratio **and** text similarity
**and** graph-sourced acoustic evidence, with a hard floor on text length so "yes"/"yes" can
never collapse; keep the best `score_permille`; record every rejected alternative; then mark
`overlap` against retained non-duplicate segments of *other* speakers).

**E. Render** — `transcript/render.py`: records → `Transcript` → `output/transcript.json` and
`output/transcript.md` in the spec's format, sorted by start then id, text escaped.

**F. Composition** — `activity/runner.py` (extract `perform_activity`, leaving `run_activity`
as the snapshot/verify/commit/report wrapper — closed-milestone code, flagged), `fakes.py`,
`fixtures/session.py`, `transcript/fakemodels.py`, `transcript/runner.py` (`run_transcribe`,
`run_render`; one INV-01 snapshot around the whole composed run, outputs declared as data over
the union of both stages', verify → commit activity → write graph → ASR → verify → commit ASR
→ write transcript, one report covering five stages), `cli.py`, `scripts/make_fixture.py`.

### Every gate criterion, and the test that proves it

| Criterion | Proof |
| --- | --- |
| Requests from retained candidates; short adjacent regions merge; padding plus an unpadded core | `test_transcript_requests.py::TestFromTheGraph` — a graph with retained, suppressed and ambiguous candidates; suppressed never appear, ambiguous always do, cores tile the merged region and stay inside their padded windows |
| The padded waveform never exceeds `max_segment_s` | `test_transcript_requests.py::TestTheCap` — a core longer than the cap splits; **a core well inside the cap whose padding would push it over has its padding shrunk**; padding shrinks at session edges; **every child request a retry creates is capped too** — plus end to end over `ScriptedTranscriber.requests` in `test_transcript_run.py` |
| Words assigned to ownership intervals, boundaries stitched | `test_transcript_asr.py::TestWordsBelongToCores` — a word inside the padding two requests share appears exactly once; **the same word returned at two *different* timestamps either side of a truncation stitch appears once**; a wordless result's text is kept whole with `segment_only`; a merged request splits its words back onto each candidate's ownership subinterval |
| Truncation: split at a low-energy boundary, retry both halves, stitch, bound, warn | `test_transcript_asr.py::TestTruncation` — resolved split; the boundary chosen at the quiet point rather than the midpoint; **the submission budget counted globally rather than per depth**; a child core below `min_split_core_ms` not split again; the **atomic** fallback keeping the original plus `asr_truncation_unresolved` when any descendant is still truncated; no dependence on a private finish-reason API |
| Collapse needs overlap **and** similar text **and** acoustic evidence | `test_transcript_collapse.py` — collapses with all three; keeps both on materially different text; keeps both when the graph's correlation is weak; `"Yes."`/`"Yes."` on two tracks never collapses; rejected alternatives recorded with the numbers that rejected them |
| Alignment failure retains the segment and warns | `test_transcript_asr.py::TestAlignment` plus the end-to-end run: `alignment_status: segment_only`, a warning, exit 0 |
| The unmodified public result is losslessly serialized, versioned, unpickled | `test_transcript_cache.py::TestRawArtifact` — JSON, every public field, round-trips; plus an assertion that nothing under `transcript/` imports `pickle` |
| `transcript.json` validates against the **checked-in** schema | `test_transcript_render.py` against `schemas/transcript.schema.json`, and `tests/data/transcript-spec-example.json` still validating |
| Millisecond precision, stable tie-breakers, ids from sorted identity and time | `test_transcript_render.py::TestDeterministicIds` — a sample position that is **not** millisecond aligned, asserted against what `public_seconds` produces rather than by counting decimals; two segments starting on one sample ordered by id; ids unchanged when the input order is shuffled |
| Language defaults to English and a configured language reaches the transcriber | `test_transcript_asr.py::TestLanguageAndContext` |
| An existing `glossary.txt` is passed exactly; its absence does not block a run | `test_transcript_asr.py::TestLanguageAndContext` — both directions |
| The report carries transcriber identity, the context hash, and `max_new_tokens` | `test_transcript_run.py::TestReportProvenance` |
| The records declare which graph and configuration they describe | `test_transcript_records.py` — `config_hash`, `timeline_sha256` and the graph's `attribution_cache_key` are present and are the ones the run used |
| The ASR cache is complete and is actually consulted | `test_transcript_cache.py::TestIdentity` — audio, request identity, transcriber identity, context, language and `max_new_tokens` each varied independently; a second run proved to hit; a truncated entry and an orphaned sidecar both refused |
| `overlap` means overlapping a retained, non-duplicate *other speaker* by at least the threshold | `test_transcript_collapse.py::TestOverlapFlag`, including the case where the only overlap is with a collapsed duplicate, which must not set it |
| Markdown format, order, escaping | `test_transcript_render.py::TestMarkdown` — the spec's exact line shape, overlapping turns as separate entries, `*` `_` `[` backtick and newlines escaped |
| `render` regenerates both outputs from records with no model and no mixer, and fails clearly when they are absent | `test_transcript_run.py::TestRender` — run after deleting the graph and the caches; a spy proves no transcriber is constructed; missing records exits nonzero with `transcript_records_missing` and still writes a report (INV-13) |
| A rerun hits caches and is byte-stable | `test_transcript_run.py::TestRerun` — `transcript.json`, `transcript.md` and the records byte-identical across two runs; the second reporting ASR cache hits and zero misses |
| No LLM prose cleanup | `test_transcript_normalize.py` — whitespace and punctuation only; a mangled-but-real sentence survives verbatim |

### Invariants this milestone could break, and what stops it

- **INV-09**, the one that matters here. `transcript/` may import from `activity`; nothing may
  write back. `run_transcribe` re-reads the graph after ASR and asserts its bytes are
  unchanged, a structural test asserts no module under `activity/` imports `transcript`, and
  the hazard M3's review deferred — `ActivityDecision.detail` and `ActivityNote.message` are
  unrestricted strings on the field allowlist — gets its own test that no ASR-derived text
  reaches either.
- **INV-02/INV-03** — records, `transcript.json` and `transcript.md` byte-stable; ids from
  sorted identity; no wall clock outside the report's telemetry.
- **INV-07** — requests are planned without audio and submitted one at a time; a
  `test_memory.py`-style ordered event log asserts a transcription happens before the last
  read, which nothing buffering every request's audio can satisfy.
- **INV-08** — the ASR cache carries the submitted audio's hash, the transcriber identity,
  the context hash, the language and `max_new_tokens`; each is varied independently and the
  cache is proved to be consulted.
- **INV-01/INV-13** — one snapshot around the composed run, outputs declared as data, verify
  → commit caches → write artifacts, a report on every path including the carve-out where the
  report's own location resolves inside `raw/`. A failed run leaves **no** sidecar anywhere
  under `work/cache`, asserted by glob rather than by naming the caches this milestone knows
  about.
