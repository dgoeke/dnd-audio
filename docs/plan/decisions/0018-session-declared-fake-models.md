# ADR-0018 — A session may declare its own fake model outputs

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M4

## Context

M4's whole point is the transcript branch working end to end with no Qwen, no GPU, and no
weights. The seams for that already exist: `Transcriber` and `ActivityDetector` are protocols
with scripted fakes (INV-10). What does not exist is any way to *run* the branch — `dnd-audio
transcribe` needs something behind the transcriber seam, and the only real implementation
lands in M6b.

ADR-0015 already argued the general case, when it made `activity` a command rather than
leaving M3 demonstrable only by its tests: **a milestone whose only demonstration is a test is
exactly the work that only appears done.** M4 inherits a sharper version of the problem,
because M3's closeout established that the canonical fixture through the *real* Silero yields
zero candidates — correctly, since INV-10 forbids expecting a learned release to fire on
audio no human made. So a live `transcribe` run against a fixture, with real models behind
both seams, would produce an empty transcript no matter how well M4 works.

The spec's own fixture recipe already asks for the missing pieces. Among the properties a
synthetic fixture must have, it lists *"deterministic fake-VAD/ground-truth activity
decisions"* and *"deterministic fake-ASR results"*. M1 built both — `FixtureTruth
.activity_spans()` and `.transcript_script()` — and until now they have lived only inside a
Python object that a test can import.

## Decision

### The fixture writes what it already knows to `<session>/fake-models.json`

`fixtures.build_session` writes a versioned JSON document beside `session.yaml` holding the
truth it declared *before* rendering a sample: per track, the ground-truth speech spans, and
per utterance, the text and word times a fake ASR should return for it. Nothing is derived
from the audio; this is the same declaration the truth object has always carried, in a form
something other than a test can read.

### `transcribe --fake-models` loads it behind the existing seams

The flag is explicit and the file must exist — its absence is a fatal, named error rather than
a silent fallback. With it, the run drives `ScriptedActivityDetector` and a session-script
transcriber. Without it, the transcriber resolver raises the builtin `NotImplementedError`
annotated `DEFERRED: M6b` at the raise site, in the same shape as every other unbuilt stage,
so `scripts/scan_placeholders.py` can see it.

### It can never be mistaken for a real run

Three guards, none of them a convention:

- Both artifacts and the report carry the transcriber and detector identity, and a scripted
  one is a `variant_digest` over the whole script — the mechanism `ScriptedActivityDetector`
  already uses, so a cache cannot serve one script's answers under another's key (INV-08).
- A `fake_models_in_use` warning reaches stderr, the records artifact, and the report.
- The flag names what it does. There is no configuration file setting, no environment
  variable, and no automatic detection of the file's presence.

### It stays scoped to `transcribe`

Independent review objected to also adding `activity --fake-models`, and it was right: that
changes a closed milestone's user-facing surface without serving M4's gate. `run_transcribe`
injects the scripted detector through the parameter `run_activity` already takes.

## Alternatives considered

- **No live demonstration; tests only.** The honest minimum, and rejected for ADR-0015's
  reason. It would also mean the first time this code runs as a command is in M6b, against a
  real model, with two milestones' worth of unexercised composition underneath it.
- **A fake ASR without a fake VAD.** Runs the plumbing and demonstrates nothing: with real
  Silero the graph is empty, so there are no requests, no segments, and no rendered line.
- **A content-derived fake that invents plausible text from the audio.** Rejected for the
  reason `fakes.py` already gives: a fake that invents its own answers cannot be asked for a
  specific one, and its "determinism" only proves that a hash function is a function.
- **Auto-detecting `fake-models.json` when it is present.** Rejected. A file dropped into a
  real session directory would then silently replace a model, which is precisely the failure
  the explicit flag exists to make impossible.

## Consequences

- The spec's fixture recipe is satisfied where it can be checked, rather than in a docstring.
- M6b replaces the resolver's `NotImplementedError` with the real adapter and the flag keeps
  working unchanged, which makes it the regression harness for the adapter's first run.
- The fixture generator now writes a file the pipeline reads. `fake-models.json` is an
  *input*, lives beside `session.yaml`, and never under `raw/`; INV-01 is unaffected.
- If a future reviewer finds this too close to production code, the containment boundary is
  one module and one flag — but note that removing it removes the only way to run M4 before
  M6b exists.
