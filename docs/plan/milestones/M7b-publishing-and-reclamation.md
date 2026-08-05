# M7b — Accepted-output handoff and local reclamation

**Status:** sketch — deliberately unplanned until live Session Zero has been validated
**Depends on:** M7a, M11, and the accepted live Session Zero outputs
**Spec sections:** archival extension added by M7a; amend its reclamation-related sections
during this milestone's start phase. **The spec's "do not read from or write to the campaign
wiki" scope line is not amended** — ADR-0044 keeps publishing outside this repository.

> M7a owns verified private backup of irreplaceable raw bytes. This charter contains only the
> later work that needs a processed real session and an operator decision about publishing or
> deleting data. Do not pull raw upload or archive integrity back into M7b.

> **ADR-0044 relocated publishing.** The delivery surface is the owner's private Outline wiki,
> and the publisher lives in the wiki host's repository, holding its own credentials. This
> project gains no Outline client and no publish command. What was the largest deliverable here
> is now a handoff: emit an output manifest identifying an accepted run, and let the consumer
> pin the transcript schema version. Retention, cache, and reclamation are untouched by that
> decision and remain the substance of this milestone.

## Goal

Define what makes a processed session *accepted*, emit a manifest that identifies it durably
enough for an external publisher to consume, define retention, prune reproducible working
caches, and provide a separate manual reclamation workflow that can remove local raw files only
after M7a's remote archive is independently re-verified.

## Remaining deliverables

- **Output manifest.** Identify every artifact of an accepted run by session, processing and
  schema version, byte length, SHA-256, and media type. A consumer can distinguish a newer
  render of the same session from a different raw recording — which neither repository can do
  from its own records alone, and which is the whole reason this artifact exists (ADR-0044).
- **Acceptance boundary.** Decide and record what promotes a processed run to *accepted*: the
  operator's judgement after M11, a recorded configuration, or both. The manifest names it.
- **Remote retention.** Decide lifecycle, versioning, and deletion policy for the private raw
  archives. M7b may add explicit remote deletion only with a separate confirmation and audit
  design. Published derivatives are the wiki's retention problem, not this project's.
- **Cache reclamation.** Measure and handle `work/cache/mix/` and other reproducible caches.
  Prefer a dry-run inventory and drop-everything-but-current-identity or age/size policy;
  cache loss must cost computation, never source data or audit lineage.
- **Local raw reclamation.** A manual, dry-run-by-default command may propose deleting local
  raw files only after a fresh M7a full remote readback restores every original hash. It must
  require exact targets and explicit confirmation, operate outside `process`, and produce a
  durable deletion receipt.

## Safety properties

1. Publishing and reclamation are distinct commands, credentials, buckets, and reports —
   enforced structurally by ADR-0044, since publishing is not code this repository contains.
2. A successful upload, HEAD, ETag, old receipt, or provider checksum alone never authorizes
   deletion. Reclamation requires a fresh M7a `archive verify` over complete remote bytes.
3. The delete candidate set is derived from the verified immutable archive manifest and an
   unchanged local raw snapshot. Extra, changed, unarchived, or unrecognized files are refused.
4. Reclamation is dry-run by default, requires the operator to name the session again, and
   never runs from `process` or an unattended lifecycle task.
5. The output manifest names artifacts and hashes. It never carries credentials, bucket names,
   object keys, or absolute local paths, because its consumer is a different repository.
6. Cache pruning cannot traverse symlinks, escape `work/cache/`, or remove authoritative
   outputs, manifests, receipts, or granular audit records.
7. A published session is not a backup. Cache or raw reclamation may never treat the wiki's
   copy of the MP3 or transcript as evidence that anything is safe to delete.

## Decisions settled by ADR-0044

Recorded here so this milestone's start phase does not reopen them: the delivery surface is the
owner's private Outline wiki; the publisher lives in the wiki host's repository; the published
set is `session.mp3` plus the transcript as native document text, with `transcript.md`,
`ingest-report.json`, and activity metadata staying local; privacy is document-scoped rather
than URL-scoped, so no public objects and no expiry scheme; document and attachment identity are
stable across re-publication; and `transcript.json` schema version 1 is the contract, pinned by
the consumer and failing closed on a bump.

## Decisions deferred to M7b start

- Whether the supporting JSON (`transcript.json`, `ingest-report.json`) should also be attached
  to the published document. Declined for now by ADR-0044; revisit once a real session has been
  published and it is clear whether a reader ever wants them.
- Whether local raw reclamation should exist at all after real storage costs and restore times
  are measured. Keeping both copies is a valid decision.
- Local and remote retention periods, legal/privacy expectations for table speech, and who may
  authorize deletion.
- Cache policy and budgets after measuring a complete real run.
- Whether zstd should ever be revisited. Compressor optimization is not a gate: M7a's byte-exact
  zstd archives remain valid regardless.

Each deliberate choice becomes an ADR. Any real-world guess that code relies on becomes an
`OQ-` entry before implementation.

## Explicitly not in this milestone

- Designing or reimplementing raw compression, raw upload, archive key layout, full readback,
  or restore. Those are closed M7a contracts.
- **Any Outline client, wiki credential, publish command, or upload path for processed output.**
  ADR-0044 places all of it in the wiki host's repository, and the spec's scope line forbidding
  this project from touching the campaign wiki stays in force.
- Deleting local files merely because storage is tight.
- Treating generated outputs or caches as substitutes for the raw archive.
- Changing activity, transcript, mix, or ASR semantics.

## Provisional working plan

1. Take M11's accepted Session Zero artifacts and measured cache/output sizes; record the
   expected audience and retrieval/reprocessing costs.
2. Amend the product spec and INV-01 for the narrowly authorized owner reclamation path; write
   retention, cache, and deletion ADRs.
3. Add the output manifest as an artifact of an accepted run, with a checked-in schema and the
   usual drift test, carrying no credentials, keys, or absolute paths.
4. Confirm against the deployed publisher that the manifest is sufficient to identify a
   published render, and that a schema-version bump stops it rather than mangling a page.
5. Implement cache inventory/pruning and raw reclamation as separate dry-run-first commands,
   with exact path and symlink defenses.
6. Prove fresh remote restoration gates every raw delete candidate, run independent reviews,
   and close with a practiced restore-and-reprocess drill.

## Provisional completion gate

- [ ] The owner has made and recorded explicit retention, cache, and deletion decisions from a
      real processed session.
- [ ] An accepted run emits a manifest that hash-identifies every artifact and its processing
      and schema versions, validates against a checked-in schema, and is byte-stable on rerun.
- [ ] A published session can be traced back to the exact run that produced it, and a re-render
      of the same session is distinguishable from a different recording.
- [ ] Cache pruning, local raw reclamation, and any remote deletion are separate commands with
      separate authority and audit records.
- [ ] A reclamation attempt cannot proceed without a fresh full M7a readback and exact restored
      original hashes; stale receipts, ETags, changed raw files, extras, and partial archives
      all fail closed.
- [ ] Reclamation is dry-run by default, requires explicit confirmation of resolved targets,
      and never executes through `process`.
- [ ] Cache pruning is confined to reproducible cache entries and cannot touch raw, outputs,
      manifests, receipts, or paths outside the resolved cache root.
- [ ] The default suite remains offline/CPU/model-free with zero skips, and independent plan
      and code reviews are distilled before close.
- [ ] A documented drill restores an archived session, reprocesses it, and identifies the
      resulting outputs without relying on the deleted local copy.

## Known risks and open questions

- Publication turns private table conversation into a durable access-control problem. The
  default should remain private until the owner chooses otherwise.
- INV-01's exception is intentionally deferred: M7a does not need it, and M7b must prove that
  deletion can be both narrow and worthwhile before weakening the rule.
- Cache use and publish ergonomics cannot be designed honestly from four short files; this
  milestone waits for a complete run on realistic duration and content.
