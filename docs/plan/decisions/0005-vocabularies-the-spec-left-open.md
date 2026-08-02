# ADR-0005 — The status, exit-code, and enum vocabularies the spec left open

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M0

## Context

The spec fixes the important vocabularies — stage status is `complete`, `failed`, or
`skipped`; devices are `auto`, `cpu`, or `cuda`; the rate labels are DJI's. It leaves
several others implied by a single example value or by a requirement stated in prose.

M0 has to choose them, because they are baked into the checked-in JSON Schemas and into
`extra="forbid"` models that every later milestone validates against. Choosing them
silently would leave the next implementor unable to tell a deliberate decision from an
accident, so they are collected here.

## Decision

**`timecode.rollover_policy`: `infer_forward` | `reject`.** The spec names only
`infer_forward` and describes it precisely: infer a single forward midnight rollover
only when chunk sequence and session span make it unambiguous, and record the decision.
`reject` is the other half — never infer, fail with an actionable diagnostic instead.
It exists because INV-12 forbids inventing timing, and an operator who knows their
session did not cross midnight should be able to say so and have an ambiguity become an
error rather than a guess.

**`overall_status`: `complete` | `partial` | `failed`.** The spec requires the field and
describes the partial case at length — a failed transcript alongside a good mix must
retain the MP3 and exit nonzero — without naming the values. `partial` means at least
one stage failed and at least one produced a deliverable; `failed` means nothing
survived. A skipped stage is not a failure: running only `mix` skips `transcribe` on
purpose.

**Exit codes: `0` ok, `1` fatal, `3` not implemented, `4` partial.** `2` is deliberately
unused — Click spends it on usage errors, and shadowing it would make a mistyped
argument indistinguishable from a pipeline failure. `4` is what makes INV-13's "partial
success never exits zero" checkable by a caller that never opens the report. `3` keeps a
stub that has not been built yet from looking like a broken session.

**`alignment_status`: `aligned` | `segment_only` | `not_attempted`.** The spec requires
that a segment whose forced alignment fails keeps its segment-level transcript and emits
a warning rather than failing the session; `segment_only` is that state, named. There is
deliberately no `failed`, because from a consumer's point of view the distinction that
matters is whether word times are present.

**`asr.dtype`: `auto` | `float32` | `bfloat16`.** The spec discusses exactly these:
BF16 on validated ROCm, float32 after CPU fallback, and BF16 on CPU only when a separate
smoke test succeeds. `float16` is not offered — nothing in the spec asks for it, and an
unvalidated precision option is a way to get a silently worse transcript.

**A recovery override must carry information.** The spec permits an optional
`recording_date` and *either* a `start_timecode` or a `start_offset_samples`. It does
not say what an override with neither means. It is rejected: an override that supplies
only a reason and a hash changes nothing, and accepting it would let a typo look like a
successful recovery. `reason` is required for the same reason — the manifest and report
have to be able to say why a time was not read from the file.

**MP3 bitrate is restricted to the MPEG-1 Layer III set.** The spec defaults to 128 and
does not constrain the field. Any other value is silently rounded by the encoder, which
would make the report's recorded bitrate a fiction.

## Alternatives considered

- **Free-form strings instead of enums.** Rejected. These values are checked into JSON
  Schemas that consumers validate against; a typo would become a schema-valid document
  with a meaning nobody implemented.
- **Only `infer_forward`, with no way to turn inference off.** Rejected as being in
  tension with INV-12: the invariant exists so that missing timing fails loudly, and an
  operator who knows the session did not cross midnight has real evidence the pipeline
  does not.
- **Reusing exit code 2 for "not implemented".** Rejected; see above.
- **Accepting an information-free recovery override as a no-op.** Rejected: a silently
  ignored override is exactly the failure the recovery mechanism exists to avoid.

## Consequences

- Each of these is now part of a checked-in schema. Adding a value is additive and
  cheap; changing the meaning of one bumps the artifact's `schema_version`.
- The M0 manifest and report schemas are **provisional** until the milestone that owns
  the artifact closes (M1 and M5 respectively). Before then, version 1 may change
  freely; after, only optional additive fields, and anything else bumps the version.
- If real hardware or a real session shows that `reject` is never useful, dropping it is
  a schema change, not a code change — which is the right size of decision to revisit.
