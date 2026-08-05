# M11 — Session Zero validation and tuning

**Status:** blocked — awaiting the live Session Zero recording
**Depends on:** M10, M7a, and an immutable live Session Zero recording
**Spec sections:** Session input contract; synchronization; Milestone 3; automix; implementation order

## Goal

Process the campaign's live Session Zero with the production baseline, preserve and archive its
sources, measure the remaining real-play behavior, and make only evidence-backed tuning changes.
The existing pipeline is the expected result; architectural expansion is a contingency, not a
deliverable presumed necessary.

## Inputs and baseline

- Follow [`docs/session-zero-capture-guide.md`](../../session-zero-capture-guide.md).
- Hash every source before analysis and verify the same hashes after every run (INV-01).
- Inspect, then immediately run `archive upload` and `archive verify` before treating the
  off-site copy as established.
- Run `process` once with production defaults before changing any threshold. Preserve that
  baseline's configuration, report, transcript records, transcript views, mix, and diagnostics.
- Record observations from ordinary play. Do not add scripted overlaps, artificial power
  cycles, dual-file exports, or receiver-count experiments to the live session.

## Completion gate

- [ ] All six immutable transmitter sources are inventoried, hash-verified before and after
      processing, and covered by a successfully verified private archive manifest.
- [ ] `inspect`, `ingest`, `activity`, `mix`, `transcribe`, and `process` complete on the live
      recording, or every failure is reproduced, diagnosed, and resolved without weakening an
      invariant. The default CPU/offline gate remains green.
- [ ] The baseline `session.mp3`, `transcript.json`, `transcript.md`, and
      `ingest-report.json` are reviewed together; no metric such as dropped-word count or
      rendered-line count is used as a loss function by itself.
- [ ] **OQ-013** records measured `work/` and cache use for the full run, and any change to
      preflight or reclamation estimates follows from those measurements.
- [ ] **OQ-017** is answered from labeled wearer/table evidence: false source deletion,
      retained bleed, overlap preservation, track references, score margins, and correlation
      are evaluated before any activity default changes.
- [ ] **OQ-018** and **OQ-027** are answered or narrowed from natural conversation: opening
      words, duplicate/contained-fragment decisions, exact-short utterances, presentation
      joins, request lineage, and any naturally occurring truncation are audited. No synthetic
      truncation requirement is invented if the session contains none.
- [ ] **OQ-019** is answered by listening to the automix alongside its graph and level
      corrections. Any changed constant is justified by a named audible defect and a regression
      that preserves bounded gain and overlap.
- [ ] Marker/timecode QA is reviewed under ADR-0040. Moving-wearer differential arrival is
      never relabeled recorder drift, and no automatic timeline correction is introduced from
      geometrically confounded evidence.
- [ ] Every changed default has an evidence note, focused regression, cache/version analysis,
      and—where it changes a decision or artifact meaning—an ADR and spec amendment. Defaults
      unsupported by a clear improvement remain unchanged explicitly.
- [ ] The event-first spike is resolved for this project stage: either the baseline is accepted,
      or a concrete failure trace justifies a separately scoped implementation amendment inside
      M11. Ambiguous duplicate speech alone is not permission to delete content.
- [ ] M7b's charter is updated with measured output/cache sizes and the accepted-session
      boundary; `./scripts/gate.sh` passes with zero unexplained skips.

## Explicitly not in this milestone

- Recreating a short metadata capture or dedicated clock-stability experiment.
- Testing whether six microphones behave differently from four merely because of the count.
- Automatic affine drift correction without fixed-endpoint evidence of a material problem.
- Tuning to make this one transcript look cosmetically perfect, or hiding uncertainty in an
  untraceable editorial pass.
- Publishing or deleting local data; M7b owns those authorities.

## Technical contingencies worth preserving

- If ordinary play exposes material clock divergence that cannot be explained by geometry,
  charter the existing affine-time-warp hook with fixed-endpoint evidence before activation.
- If the track-first activity/ASR boundary demonstrably causes false duplication or source
  deletion that conservative thresholds cannot resolve, use
  [`EVENT-FIRST-ARCHITECTURE-SPIKE.md`](../EVENT-FIRST-ARCHITECTURE-SPIKE.md) as the starting
  hypothesis and prove one narrow event representation before changing production semantics.
- If real cache footprint makes retention unsafe, feed measured sizes into M7b; do not grant
  deletion authority here.
