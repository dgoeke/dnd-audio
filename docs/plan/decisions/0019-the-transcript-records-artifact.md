# ADR-0019 — `work/transcript-records.json`, and what identifies a segment

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M4

## Context

The spec requires `render` to "regenerate `transcript.json` and `transcript.md` from existing
normalized transcript records" without invoking ASR or the mixer, and to "fail clearly if the
required transcript records do not exist". It never says what a record *is*, where it lives,
or what it has to carry for that regeneration to be honest.

It also asks for segment ids "derived deterministically from sorted source identity and time,
not from task completion order", and shows an example whose id is `seg_000123` — a form that
carries neither a track nor a sample position. That example is checked in as
`tests/data/transcript-spec-example.json` and held as independent ground truth, so whatever id
scheme M4 chooses has to keep it valid.

And the ASR cache has its own identity question. The spec lists what a key must include:
"the exact segment-audio hash, model/aligner identifiers, context hash, language, and
inference parameters".

## Decision

### The records artifact is the render input, and render reads nothing else

`work/transcript-records.json` holds every normalized segment — text, words in 48 kHz session
samples, alignment status, overlap flag, the collapse decision and the alternatives it
rejected, request lineage, and provenance. Duplicate collapse and overlap marking happen
*before* it is written, so `render` is a pure function of this one file: no model, no
`activity.json`, no timeline, no mixer. That is what makes the spec's requirement checkable
rather than merely intended — a `render` that consulted the graph could not be proved not to
need ASR.

It is versioned, byte-stable (INV-02), and has a checked-in JSON Schema, following
`schema_export.py`'s stated rule that a new deterministic artifact which skipped its schema
would be the one consumers could not validate.

### It declares what it describes

The records carry `config_hash`, the `timeline_sha256` they were placed against, and the
graph's `attribution_cache_key`. A records file sitting beside a graph it does not describe is
then detectable rather than merely unlikely — the same reasoning M3 used when
`ActivityGraph` started carrying the hash of the timeline it was built from.

### A segment id is its position in the canonical order

`seg_%06d`, numbered over segments sorted by `(start_sample, track_id)`. The sort key is
source identity and time, so the id is a function of the inputs and never of completion order,
which is what INV-02 and the spec both ask for. It keeps the spec's own `seg_000123` example
valid, which a `cand_`-style id would not.

Collapsed duplicates keep their ids, so `transcript.json` — which carries only retained
segments — has gaps in its numbering. That is informative: a gap says something was collapsed
there, and the records name what.

### The ASR cache key includes the request's identity as well as its audio

Key components: the submitted audio's sha256, the request identity (track and ownership
interval), the transcriber identity, the context hash, the language, `max_new_tokens`, and the
ASR semantics and record versions.

The request identity is the addition, and it is deliberate. INV-08 requires a key to
*include* the spec's list, not to be limited to it, and `config.py` already states the bias:
a too-broad key costs recomputation, which is slow, and a too-narrow one serves a stale
answer, which is silent. Two requests with byte-identical audio do not occur in a real session
— a candidate came from a VAD and is not digital silence — so the added component costs
nothing measurable. What it buys is that a scripted fake, which selects its response by
`request_id` and is therefore *not* a function of its audio, cannot turn a cache hit into a
test that quietly passes with the wrong text.

**The records and render versions stay out of that key.** Changing how a transcript is
rendered must never cost a re-transcription.

## Alternatives considered

- **Render from `activity.json` plus a cache of ASR results.** Rejected: `render` would then
  depend on the graph and on the cache layout, and "regenerates without ASR" would be a claim
  about which code paths happen not to run rather than a property of its input.
- **`seg_<track>_<start>` ids, matching `candidate_id`.** More self-describing, and it breaks
  the spec's checked-in example. The example is ground truth this project deliberately holds
  independent of its own models; invalidating it to gain a nicer id is the wrong trade.
- **Renumbering only retained segments so `transcript.json` has no gaps.** Rejected: the same
  segment would then have two ids depending on which document you read.
- **Keying the ASR cache on audio alone, as the spec literally lists.** Rejected above. The
  spec's list is a minimum.
- **Storing normalized records inside the ASR cache.** Rejected: it conflates "what the
  backend returned for this audio" with "what this session decided about it", and it would
  make a change to collapse or rendering invalidate expensive inference.

## Consequences

- `render` is trivially testable: delete the graph, the timeline, and every cache, and it
  still produces byte-identical output.
- A records file is small and self-describing enough to be worth reading during a support
  question — which is the same argument `activity.json` made for keeping its evidence per
  pair.
- The records schema is provisional until M4 closes; after that, additive optional fields
  only (ADR-0005).
