# M7b — Processed publishing and local reclamation

**Status:** sketch — deliberately unplanned until live Session Zero has been validated
**Depends on:** M7a, M11, and the accepted live Session Zero outputs
**Spec sections:** archival extension added by M7a; amend its publication and reclamation
sections during this milestone's start phase

> M7a owns verified private backup of irreplaceable raw bytes. This charter contains only the
> later work that needs a processed real session and an operator decision about publishing or
> deleting data. Do not pull raw upload or archive integrity back into M7b.

## Goal

Publish the deliberately selected processed artifacts through a wiki-consumable delivery
surface, define retention and privacy, prune reproducible working caches, and provide a
separate manual reclamation workflow that can remove local raw files only after M7a's remote
archive is independently re-verified.

## Remaining deliverables

- **Processed-output publishing.** Upload the accepted `session.mp3`, `transcript.json`,
  `transcript.md`, `ingest-report.json`, and deliberately selected activity/attribution
  metadata to Standard Spaces or another surface capable of the required signed/public/CDN
  delivery. Raw audio never enters the publishing bucket.
- **Publication policy.** Decide public-read versus signed URLs, expiry and rotation, whether
  transcripts or audio may be indexed, and how corrections replace a published processing
  version without erasing provenance.
- **Output manifest.** Identify every published artifact by session, processing/schema
  version, byte length, SHA-256, media type, and privacy class. A consumer can distinguish a
  newer render from a different raw recording.
- **Remote retention.** Decide lifecycle, versioning, and deletion policy for both private raw
  archives and published derivatives. M7b may add explicit remote deletion only with a
  separate confirmation and audit design.
- **Cache reclamation.** Measure and handle `work/cache/mix/` and other reproducible caches.
  Prefer a dry-run inventory and drop-everything-but-current-identity or age/size policy;
  cache loss must cost computation, never source data or audit lineage.
- **Local raw reclamation.** A manual, dry-run-by-default command may propose deleting local
  raw files only after a fresh M7a full remote readback restores every original hash. It must
  require exact targets and explicit confirmation, operate outside `process`, and produce a
  durable deletion receipt.

## Safety properties

1. Publishing and reclamation are distinct commands, credentials, buckets, and reports.
2. A successful upload, HEAD, ETag, old receipt, or provider checksum alone never authorizes
   deletion. Reclamation requires a fresh M7a `archive verify` over complete remote bytes.
3. The delete candidate set is derived from the verified immutable archive manifest and an
   unchanged local raw snapshot. Extra, changed, unarchived, or unrecognized files are refused.
4. Reclamation is dry-run by default, requires the operator to name the session again, and
   never runs from `process` or an unattended lifecycle task.
5. Processed publication never makes raw archives public and never leaks credentials or local
   paths.
6. Cache pruning cannot traverse symlinks, escape `work/cache/`, or remove authoritative
   outputs, manifests, receipts, or granular audit records.

## Decisions deferred to M7b start

- Which artifacts are private, link-shared, or public, and whether the wiki consumes stable
  URLs or versioned URLs.
- Standard Spaces/CDN versus another publication surface; Cold Storage is intentionally not a
  web-delivery mechanism.
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
- Automatic publication from `process` unless a later, explicit policy ADR authorizes it.
- Deleting local files merely because storage is tight.
- Treating generated outputs or caches as substitutes for the raw archive.
- Changing activity, transcript, mix, or ASR semantics.

## Provisional working plan

1. Take M11's accepted Session Zero artifacts and measured cache/output sizes; record the
   expected audience and retrieval/reprocessing costs.
2. Amend the product spec and INV-01 for the narrowly authorized owner reclamation path; write
   privacy/publication, retention, cache, and deletion ADRs.
3. Add an output-publication manifest and a separate provider adapter/configuration boundary
   from M7a's raw archive.
4. Implement publish/status/unpublish behavior with deterministic fake storage tests and no
   network in the default gate.
5. Implement cache inventory/pruning and raw reclamation as separate dry-run-first commands,
   with exact path and symlink defenses.
6. Prove fresh remote restoration gates every raw delete candidate, run independent reviews,
   and close with a practiced restore-and-reprocess drill.

## Provisional completion gate

- [ ] The owner has made and recorded explicit privacy, URL, retention, versioning, cache, and
      deletion decisions from a real processed session.
- [ ] Published outputs are versioned, hash-identified, restorable to a known processing run,
      and never mixed with private raw archives.
- [ ] Publication, cache pruning, local raw reclamation, and any remote deletion are separate
      commands with separate authority and audit records.
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
