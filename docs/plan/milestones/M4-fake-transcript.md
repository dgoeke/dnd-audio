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
