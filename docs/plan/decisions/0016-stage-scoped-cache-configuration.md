# ADR-0016 — Cache identity carries a stage-scoped projection of the configuration

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M3

## Context

INV-08 requires every cache key to include "the resolved configuration". Until now that has
meant the whole thing: M1's inspection cache and M2's derivative cache both hash
`config_hash(config)`, which covers every section of `session.yaml`. M1's closeout recorded
the cost knowingly — *"an unrelated `mix.integrated_lufs` edit therefore re-probes every
source. That is seconds of FFprobe and the safe direction to be wrong in."*

M3 changes the arithmetic. It adds `activity.vad`, `activity.bleed`, and `activity.scoring`,
and OQ-017 guarantees those numbers get tuned repeatedly against real sessions. Under
whole-config hashing, raising `min_correlation` by a hundredth rebuilds every 16 kHz
derivative in the session — gigabytes of PCM that provably cannot depend on it. Independent
review (`../reviews/M3-plan-20260802-1600.md`) flagged the plan's claim that this
"correctly invalidates M2's derivative caches" as simply untrue.

The seductive fix is to hash only the fields a stage reads. That is also how this invariant
dies: the failure mode of a too-narrow key is a stale artifact served as current, which is
silent, and the failure mode of a too-broad key is recomputation, which is merely slow.

## Decision

### One named projection per stage, defined as data

`config.stage_config(config, stage)` returns the sections of the resolved configuration a
stage's output can depend on, and `stage_config_hash` hashes it. The projections are
declared in one table, so which sections a stage depends on is visible in one place rather
than inferred from call sites:

| Stage | Sections included |
| --- | --- |
| `inspection` | schema version, session identity, `active_tracks`, `tracks`, `timecode`, `recovery` |
| `derivative` | the above (placement determines the segment map, which determines the audio) |
| `detection` | session identity, `tracks`, `activity.vad` |
| `attribution` | session identity, `tracks`, `activity` in full |

Everything not listed is excluded because it provably cannot reach that stage's bytes:
`mix` and `asr` reach none of them, `sync_qa` produces warnings and never a sample, and
`activity.bleed` cannot alter 16 kHz PCM.

### Generous, not minimal

A section is included unless its exclusion is *provable*, not merely plausible. `tracks`
appears everywhere because it carries the roster and therefore which audio exists at all;
`timecode` reaches the derivative because placement decides what the segment map contains.
When a future field's blast radius is unclear, it goes in the broader projection. Being slow
is recoverable; being stale is not.

### Both directions are tested

For each projection, a test varies every included section and asserts the hash moves, and
varies every excluded section and asserts it does not. A test that only checks the first
half is how a projection quietly narrows over time: the key still changes for the reasons
someone thought to test, and no longer changes for one nobody did.

The projections are also asserted to be exhaustive over `SessionConfig`'s fields — a new
configuration section must be classified deliberately, and a field belonging to no projection
fails the test rather than defaulting to "excluded".

## Alternatives considered

- **Keep hashing the whole configuration.** Correct, simple, and it makes the tuning loop
  this milestone creates cost gigabytes per iteration on a real session.
- **Hash only the individual fields each stage reads.** Precise in principle and brittle in
  practice: the coupling is not "which field is read" but "which field can change the bytes",
  and those differ wherever one value is derived from another.
- **Version the caches manually and bump on tuning.** Puts a correctness property in a
  human's hands, on the exact path where forgetting produces a silently stale result.
- **Give the derivative cache its own configuration model** rather than a projection.
  Duplicates the session contract into a second shape that can drift from it.

## Consequences

- M2's derivative identity changes, so every existing cached derivative is a miss once, and
  M2's closed code is edited — the same kind of change as its own extraction of `raw_guard`.
  `derivative_identity_document` keeps its "separate from the hash so a test can assert which
  components are present" property, with the projection as one component.
- Tuning `activity.bleed` re-runs the bleed gate and the scoring, and reuses both the 16 kHz
  derivatives and every per-track VAD result. That is the point.
- Tuning `activity.vad` re-runs detection and therefore attribution, and still reuses the
  derivatives.
- `INSPECTION_SEMANTICS_VERSION` and `TIMELINE_SEMANTICS_VERSION` keep their existing
  package-wide scope. Narrowing *those* would reintroduce exactly the risk this ADR is
  careful about, and they change on code edits rather than on operator tuning.
