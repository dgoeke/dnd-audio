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

## Known risks and open questions

- Depends on **OQ-009** for the eventual real segment limits, but M4 must be
  correct under the configured limit regardless.
- Duplicate collapse is where the pipeline is most likely to silently delete real
  speech. Bias every ambiguous case toward keeping both and marking overlap.
- INV-09: nothing decided here may flow back into the activity graph.
