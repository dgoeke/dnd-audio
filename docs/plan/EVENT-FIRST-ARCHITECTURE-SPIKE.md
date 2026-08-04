# Event-first transcript architecture spike

Status: **hypothesis awaiting acoustic evidence**. This is not an ADR, does not amend the
product spec, and does not authorize a production change. It records the architectural
direction to test after M9 so the reasoning is not lost between sessions.

The smallest input is the
[minimal two-person acoustic direction check](../minimal-acoustic-direction-check.md). H1
remains a separate hardware fixture with different obligations.

## Why reconsider the stage boundary

The current pipeline is a faithful implementation of the original spec: VAD produces
per-track candidates, the activity graph compares those candidates, ASR transcribes retained
audio, and transcript assembly resolves copies using timing, text, graph evidence, and source
quality. M8 and M9 made that design safe and auditable.

The four-file sample also exposes the abstraction's limit. One person saying `Okay` can be
recorded directly on one transmitter and as strong bleed on another. The acoustic evidence can
show that the observations are related without proving, at the required margin, which
transcript record is the disposable copy. Text cannot answer that question: two people can
genuinely say the same short word at the same time. The safe current result therefore keeps
both copies. The later `now I am talking...` fragment is a separate presentation question,
not proof that the earlier acoustic event should be deleted or merged.

This suggests that a future pipeline may need to model **speaker events observed by multiple
microphones before it models words**, rather than treating per-track speech candidates as the
primary objects and repairing their transcripts afterward.

## What should remain unchanged

An architectural experiment starts from the project's existing guarantees, not from a blank
sheet:

- raw recorder files remain immutable and hash-verified;
- timeline placement remains exact and recorder metadata remains distinct from acoustic
  observations;
- all audio and models remain local and the default gate remains CPU/offline;
- activity, mix, transcript records, and rendered views retain explicit versioned semantics;
- evidence, uncertainty, model identity, and transformations remain auditable;
- transcript work must not move activity or alter the mix as a side effect;
- expensive inference caches are invalidated only by inputs or semantics that affect them;
- ambiguous evidence must remain ambiguous rather than being optimized away on a tiny corpus.

The current implementation remains the production baseline until a milestone and ADR say
otherwise.

## Candidate stage architecture

The proposed unit is a latent acoustic event: one or more speakers active over a time range,
with observations on one or more channels. A spike should test this sequence:

1. **Construct the coarse timeline exactly as today.** Keep recorder-domain placement and the
   existing sync QA.
2. **Refine cross-channel acoustic alignment where evidence supports it.** Estimate small
   residual offset and drift from shared sound. Do not rewrite the source timeline or pretend
   weak evidence is precise.
3. **Use generic VAD as a proposal generator.** Per-track regions propose places to inspect;
   they are not yet speaker turns or transcript ownership intervals.
4. **Bootstrap session-local wearer profiles.** Use long, high-confidence solo regions from
   each worn transmitter to characterize its wearer's voice. Record when no reliable profile
   exists.
5. **Infer wearer activity jointly across channels.** Estimate which known wearer or wearers
   are active in each frame using direct-channel evidence, cross-channel lag/correlation,
   relative level, and speaker features. Multiple wearers may be active simultaneously.
6. **Build an acoustic event graph.** Associate channel observations with latent speaker
   events, preserving competing hypotheses and confidence rather than immediately deleting a
   candidate.
7. **Choose or extract source audio per event and speaker.** The wearer's direct transmitter
   is the normal source. Targeted separation is a possible later tool for true overlap, not a
   prerequisite for the first experiment.
8. **Run ASR once per selected event/speaker stream.** Avoid asking ASR to create several
   textual copies that later stages must deduplicate.
9. **Reconcile acoustics and words.** Alignment and language evidence can repair boundaries or
   flag inconsistencies, but text equality alone must never prove that one of two acoustic
   sources did not speak.
10. **Render canonical and editorial views separately.** The canonical transcript reflects
    retained event records and uncertainty. A later local LLM/editorial pass may join clauses,
    repair punctuation, or suggest an obvious placement, but its structured edits and source
    record lineage must remain visible.

One possible event record, intentionally not yet a schema, would include:

```text
event_id
start/end on the shared timeline
speaker hypotheses (zero, one, or several)
per-channel observations and selected source
evidence and confidence
status: same_source_duplicate | distinct_sources | unresolved
downstream ASR and editorial lineage
```

This preserves the useful distinction missing from a transcript-only duplicate decision:
“these waveforms are observations of one source,” “there are two sources,” and “the evidence
does not decide” are different states.

## Where waveforms end and words begin

Waveform evidence should answer:

- how many acoustic sources are active;
- which session-local wearer each source most likely belongs to;
- which microphones observed the same source;
- which channel is the best source for each event;
- whether the result is uncertain or genuinely overlapping.

Word evidence should answer:

- what each selected source said;
- how words align inside the event;
- whether adjacent records read as one sentence in a presentation view;
- whether an editorial cleanup is plausible and traceable.

Words remain valuable, but they should not carry the burden of proving acoustic identity. The
project should spend more waveform/speaker effort before ASR, then use language evidence for
lexical reconciliation and presentation.

## Alternatives deliberately left open

- **Continue tuning only the present pipeline.** This is still the right outcome if the
  acoustic experiment cannot reliably separate one-source bleed from simultaneous speakers.
  M9's conservative unresolved records are a valid result, not a failure.
- **Transcribe every full track and ask an LLM to deduplicate it.** Useful as an optional
  editorial view, but unsafe as the canonical derivation because text cannot distinguish two
  voices saying the same words and the edit would obscure acoustic provenance.
- **Mix first, then transcribe one stream.** Simple, but loses speaker-specific source choice
  and makes overlapping speakers harder to recover.
- **Adopt a monolithic audio-language model.** It may be useful for comparison later, but does
  not by itself meet the local, deterministic, inspectable, and independently cached stage
  requirements.
- **Start with phase-coherent beamforming or full guided source separation.** The unsynchronized
  wearable recorders and small evidence set do not yet justify that complexity. First measure
  whether residual alignment, session-local speaker profiles, and direct-channel dominance are
  enough.

These are experiment branches, not conclusions. The first spike should prefer inspectable
features and frozen intermediate artifacts over a large learned replacement.

## Minimum experiment and measurements

Run the minimal capture in an isolated session under `/tmp`; never alter its originals. Hash
the raw files before and after. Preserve the current pipeline output as the baseline, then
analyze the same aligned audio without changing production defaults.

The experiment should report event-level results for every scripted trial:

- each solo `Okay` becomes one latent source event despite being heard on several tracks;
- each simultaneous two-person `Okay` becomes two source events, even though the text is
  identical;
- both different-word overlap utterances survive;
- solo and quick-handoff events are assigned to the correct wearer/direct transmitter;
- the three-second `Okay ... now` example stays acoustically honest even if an editorial view
  joins it;
- unresolved cases are labeled rather than silently collapsed.

Measure at least false source deletion, false source duplication, speaker attribution, event
fragmentation, overlap retention, ASR calls/audio duration, and runtime. For this project,
deleting genuine speech is worse than retaining an explicit ambiguity. No threshold or model
choice becomes a production default from this one capture.

## Decision after the data returns

There are three acceptable outcomes:

1. **The event-first hypothesis is supported.** Create a new software milestone, separate from
   H1, to define the event schema and one narrow pre-ASR inference path. Take the independent
   plan review, write the ADRs, and amend the product spec before implementation.
2. **The evidence is weak or brittle.** Keep the conservative M9 architecture and add only a
   clearly separate editorial/LLM cleanup layer if desired. Preserve both canonical `Okay`
   records.
3. **The evidence is mixed.** Introduce event grouping first as advisory, non-destructive
   metadata. Let it improve source ranking and diagnostics without authorizing deletion until
   broader table evidence exists.

In every case, H1 and H2 continue to answer their hardware and long-session questions. This
spike decides a software direction; it does not substitute for either fixture.

## Research leads for the spike

These are starting points, not adopted dependencies:

- target-speaker VAD and its multi-speaker extensions for estimating known-speaker activity;
- diarization-guided ASR for separating acoustic speaker inference from lexical decoding;
- distant conversational speech recognition systems that combine diarization, source
  separation, and ASR;
- Qwen ASR's timestamp and alignment behavior, evaluated only through the locally pinned
  adapter used by this project.

Any adopted model or algorithm still needs a pinned, offline, locally executable path and a
deterministic CPU fixture that proves the surrounding semantics without downloading weights.
