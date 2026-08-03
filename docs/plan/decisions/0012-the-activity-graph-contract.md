# ADR-0012 — The activity graph: what it says, in what units, and what may never enter it

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M3

## Context

The spec asks for "a canonical, model-independent pre-ASR activity/attribution graph from
VAD and the conservative bleed gate", consumed by the automixer, with transcript assembly
starting from the same graph but never feeding its text-dependent decisions back. INV-09 is
that sentence as an invariant, and M3's gate freezes the contract because M4 and M5 both
consume it — changing it later means redoing both.

The first draft of this decision named the Pydantic classes and stopped. Independent review
(`../reviews/M3-plan-20260802-1600.md`) pointed out that naming classes is not a contract:
it left the lag's unit, the reference to a suppressing candidate, the arrangement of
pairwise evidence, and the ordering of nested collections all unstated, and it proposed to
enforce INV-09 with an import test that a single `normalized_text: str` field would walk
straight past.

## Decision

### `work/activity.json`, version 1, frozen at M3's close

Additive optional fields only after that; anything else bumps the version (ADR-0005). The
document is generated from the Pydantic models in `artifacts/activity.py` and validated
against the **checked-in** `schemas/activity.schema.json`, like every other artifact.

### There are no floats in it

Not one, for the reason `timeline.json` has none: this document is deterministic and
byte-stable (INV-02), and a float that is the quotient of two NumPy reductions is not
reliably identical across a library upgrade. Every measured quantity is an integer in a
named unit:

| Quantity | Unit | Field suffix |
| --- | --- | --- |
| Time on the session grid | 48 kHz samples, half-open intervals | `_sample` |
| Time on the detector grid | 16 kHz samples, half-open intervals | `derivative_*_sample` |
| Probabilities, scores, correlation | per-mille, 0…1000 | `_permille` |
| Levels and level differences | millibels (dB × 100), signed | `_mb`, `_mbfs` |
| Correlation lag | **16 kHz samples**, signed | `lag_derivative_samples` |

Quantization is the project's one existing rule — `determinism._quantize`, half away from
zero — applied once at the boundary, never accumulated.

The lag stays on the detector's grid deliberately. Scaling it by three to reach the
canonical grid would produce a number that looks like a 48 kHz measurement and is not one;
the document states `derivative_sample_rate` so a consumer can convert knowingly.

### Every interval is half-open, and both grids are recorded

A candidate carries `[start_sample, end_sample)` at 48 kHz **and**
`[derivative_start_sample, derivative_end_sample)` at 16 kHz. The 16 kHz pair is what the
detector actually decided; the 48 kHz pair is what M5 mixes and M4 requests, converted
through `timeline.resample.to_source_sample` and `to_derivative_interval` and never
re-derived by hand. That conversion floors the start and ceils the end, so the 48 kHz
interval always *covers* the detected one — rounding both ends alike shrinks a speech region
by up to two samples, which is how a word loses its first phoneme.

### Suppressed candidates stay in the document, and name their suppressor

A graph that lists only survivors cannot be audited, and the spec requires rejected
alternatives to be recorded. Every candidate carries a `decision` of `retained` or
`suppressed`; a suppressed one carries `suppressed_by_candidate_id` — the **candidate**, not
merely the track, because "tx-a beat it" does not say which of tx-a's utterances did.

Pairwise evidence is one `CandidateEvidence` record **per compared pair**, not one summary
per candidate, each carrying the compared interval, the peak correlation, its lag, the score
margin, the level difference, and a closed-vocabulary `outcome` saying what that pair
decided. Multiple suppressors therefore stay visible as multiple records rather than
collapsing into one.

### Ordering is stated, not incidental

Tracks by `track_id`. Candidates by `(start_sample, track_id)`. Evidence by
`other_candidate_id`. Warnings and decisions by their own sort keys, as in `timeline.json`.
Candidate IDs are `cand_<track_id>_<start_sample zero-padded to 12>` — derived from sorted
source identity and time, never from completion order (INV-02) — and their uniqueness is
asserted by a validator rather than assumed.

### Attribution is the retained candidates, and nothing else

For the MVP baseline the spec permits attributing every retained candidate to the person
mapped to that track, so there is no separate attribution structure to disagree with the
candidate list. `ActivityTrack` carries the speaker mapping once; a candidate names only its
track.

That makes both consumers' reads explicit:

* **M4** takes retained candidates in order, merges short adjacent ones, and pads them into
  transcription requests. Suppressed candidates are exactly what it must not transcribe.
* **M5** takes each track's retained candidates as that track's active intervals, with
  `score_permille` as the confidence its gain envelope is smoothed from, and
  `speech_reference_mbfs` as the per-track voice-level correction it was asked to estimate.

Both patterns are exercised against a really generated graph, so "M4 and M5 can read this"
is a test rather than a claim.

### INV-09 is enforced by a field allowlist, not by an import graph

`dnd_audio.activity` importing nothing from the transcript layer does not stop a later
milestone adding a local text-derived field. So the frozen document's **every property
name** is listed explicitly in `tests/test_activity_artifact.py`, and the test fails on any
name that is not in the list. Adding a field is then a deliberate edit to a frozen contract
in the same commit, which is the point. The structural import test stays too; the two fail
for different reasons.

## Alternatives considered

- **Serialize probabilities and scores as floats.** Readable, and byte-stability then rests
  on NumPy's reduction order never changing. INV-02 is not a property worth resting on that.
- **Store only retained candidates**, with rejections in the report. Rejected: the report is
  exempt from byte-stability as a whole, and an attribution decision that cannot be replayed
  from the deterministic artifact is not auditable where it matters.
- **A separate attribution section** mapping intervals to speakers. Rejected as a second
  source of truth that can disagree with the candidate list while both look right; it earns
  its place when diarization stops being "the person wearing the lav", which is post-MVP.
- **One evidence summary per candidate** (best competitor only). Smaller, and it hides the
  case this milestone is most likely to get wrong: a candidate that two tracks both nearly
  suppressed.
- **Scaling the lag to 48 kHz** so the document has one time unit. Rejected: it manufactures
  precision. The measurement's resolution is 62.5 µs and the field name now says so.

## Consequences

- The graph's own cache identity (`attribution_cache_key`) is *in* the document, so a
  consumer can tell whether the graph it holds matches the configuration it is reading it
  under without recomputing anything.
- Freezing at version 1 means M4 and M5 may add optional fields but may not change a unit, a
  grid, or an ordering. If either needs to, that is a version bump and a deliberate migration
  of both.
- Recording per-pair evidence makes the document grow with the square of the overlap density
  rather than linearly with the candidates. Bounded in practice — only candidates that
  actually overlap in time are compared, and six tracks cap the fan-out at five — but it is
  the reason evidence is capped to compared *pairs* rather than to all candidate pairs.
