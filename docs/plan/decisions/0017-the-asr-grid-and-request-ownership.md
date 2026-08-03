# ADR-0017 — ASR runs on the 16 kHz derivative, and requests merge without merging ownership

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M4

## Context

The spec tells M4 to "transcribe retained VAD segments from their owner's lav rather than
transcribing the six full-length files blindly" and to "merge very short adjacent regions"
while capping a request well below the adapter's limit. It does not say which of this
project's two audio grids the model is handed, and it does not say what happens to the
identity of the candidates a merge swallows.

Both gaps matter more than they look.

**The grid.** M2 built two: the 48 kHz segment map that is the lossless working path
(ADR-0011), and a cached 16 kHz derivative built through one checked-in FIR. Qwen3-ASR
ingests 16 kHz mono. Handing it 48 kHz audio means resampling somewhere, and "somewhere" is
either a second resampler in this project or the model package's own frontend — in both cases
a conversion nobody has pinned, sitting underneath a cache key.

**Ownership.** M3's graph identifies a candidate by track and start sample, and measures
`CandidateEvidence` between *pairs of candidates*. The spec's transcript baseline gives a
segment one `source_candidate_id`. Merging two adjacent candidates into one request and then
emitting one segment for the result breaks both: the segment has two source candidates and
one field to name them in, and post-ASR duplicate collapse — which needs the acoustic
evidence M3 measured — has nothing to look that evidence up by. Independent review of the
working plan produced the concrete failure: candidates A1 and A2 merge, candidate B on
another track overlaps only A2, and collapsing the merged segment against B either deletes A1,
which nobody claimed was a duplicate, or leaves A2's duplicate in place.

## Decision

### ASR consumes the 16 kHz derivative

A transcription request's audio is read from the track's cached 16 kHz derivative — the same
bytes the VAD decided on — and `AudioWindow.sample_rate` says so. Word times come back on
that grid and are converted to canonical 48 kHz session samples through
`timeline.resample.to_source_sample`, the conversion M2 already owns.

The derivative is cached, byte-stable, and content-addressed, so the sha256 of a submitted
segment is stable across runs and across a SciPy upgrade. The 48 kHz path stays what it was
built for: the mix.

### Requests merge; ownership does not

A `RequestPlan` carries an ordered list of **ownership intervals**, one per retained candidate
it covers, each naming its `candidate_id`. Merging joins the *audio* handed to the model — so
it sees a whole sentence with its context rather than eight fragments — and changes nothing
about who owns which sample. Words are assigned to the ownership interval containing their
start, so:

- one retained candidate produces one segment;
- `source_candidate_id` stays singular, exactly as the spec's baseline has it, and no spec
  amendment is needed;
- "keep the version with the best source score" is unambiguous, because a segment has exactly
  one score;
- duplicate collapse reads the exact pairwise evidence M3 measured, with no aggregation.

**One case cannot be split, and it is recorded rather than hidden.** A response with no word
times — the `segment_only` path, where alignment failed — cannot be divided across the
candidates its request covered. It emits a single record naming every contributing candidate.
`TranscriptRecords` therefore holds `source_candidate_ids` as a list that is length one in
every ordinary case, and collapse aggregates across it **conservatively**: every cross-pair
that exists must meet `min_correlation`, not merely the best one, because the bias is toward
keeping speech.

## Alternatives considered

- **Read 48 kHz and let the adapter resample.** Rejected. It puts a second, unpinned
  resampler under a cache key, which is the failure mode INV-04 names for time and ADR-0011
  names for audio. It also makes the segment hash depend on the model package's frontend
  version rather than on this project's checked-in filter.
- **Read 48 kHz and resample here.** Same objection, plus it re-does per request the work M2
  already did once per track and cached.
- **Do not merge adjacent candidates.** The simplest way to keep ownership intact, and it is
  ruled out by the completion gate, which requires short adjacent regions to merge. It would
  also hand the model a stream of fragments with no context, which is the thing padding exists
  to avoid.
- **Emit one segment per request and let `source_candidate_id` name the first.** Rejected.
  It makes the field a lie whenever a merge happened, and it is what produced the reviewer's
  A1/A2/B failure.
- **Extend `transcript.json` to carry a list of source candidates.** Rejected as unnecessary
  once ownership survives the merge: the spec's baseline is correct as written, and amending
  a public schema to work around an internal choice is the wrong direction.

## Consequences

- A word's session time is exact to a 16 kHz sample, or 3 samples at 48 kHz — 62 µs. Public
  transcript times are quantized to milliseconds anyway (INV-04), so nothing is lost at the
  boundary that serializes them.
- If a future adapter genuinely needs full-bandwidth audio, this changes: the request builder
  is the only place that reads audio, and `TrackReader` is already the 48 kHz alternative.
  What must not happen is both grids being used at once for the same request.
- Whether the padding this design pays for is *enough* for word recovery is a property of the
  model, registered as **OQ-018** and cited from the default.
