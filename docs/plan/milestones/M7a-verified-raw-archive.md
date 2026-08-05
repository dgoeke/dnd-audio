# M7a — Verified private raw archive

**Status:** not started — independent plan review completed; implementation is a separate cycle
**Depends on:** M1 and M8 (closed), an inspected session, and an owner-created private
DigitalOcean Spaces Cold Storage bucket
**Spec sections:** new archival extension — amend the product spec before implementation

## Goal

Before the first live session, provide an explicit, resumable archive path that captures every
regular file protected by a session's immutable source snapshot, compresses it byte-for-byte,
uploads it to owner-controlled private cold storage, reads every archive object back, and proves
that it restores to the original SHA-256. After loss of the local session directory, an operator
must be able to discover, verify, and restore a whole session or one known track without knowing
object keys or possessing the old `session.yaml`.

This milestone is the narrow exception to the project's local-only network rule. It protects
irreplaceable recordings from disk loss. It does **not** publish outputs and does **not** delete,
rename, normalize, or otherwise mutate anything in a source root.

## Evidence already available

DigitalOcean currently offers a distinct Spaces Cold Storage bucket type. Cold buckets are
private S3-compatible storage and omit CDN/public-delivery features, which is the right shape
for raw session audio. Bucket type is chosen when the bucket is created and cannot later be
changed. The implementation must use the documented regional endpoint and signed requests,
not assume that every Standard Spaces feature exists in Cold Storage. Current provider facts
must be rechecked at implementation time against the official
[pricing](https://docs.digitalocean.com/products/spaces/details/pricing/),
[creation](https://docs.digitalocean.com/products/spaces/how-to/create/),
[limits](https://docs.digitalocean.com/products/spaces/details/limits/), and
[S3 compatibility](https://docs.digitalocean.com/products/spaces/reference/s3-compatibility/)
documentation.

Cold Storage bills an object smaller than 128 KiB as 128 KiB. M7a therefore creates exactly
one small remote control object per session—the committed archive manifest—and no per-file JSON
sidecars, FFprobe dumps, receipts, restoration notes, or verification reports. The one manifest
is worth its billing floor because it is the discoverable commit marker; multiplying that floor
across dozens of sidecars is not. Useful bounded inspection identity and complete restoration
instructions are consolidated into the manifest. Operation reports remain local.

A byte-stream zstd trial over four real DJI recordings used `zstd 1.5.7`, `-T0 -10`, reduced
121,617,184 bytes to 84,663,377 bytes (30.4%), and restored all four original SHA-256 values
exactly. This chooses zstd over a first-session WavPack bake-off; it does **not** freeze the v1
encoding recipe because `-T0` may make output depend on the host. The implementation ADR must
pin a byte-stable zstd version, flags, thread/resource settings, and streaming behavior, with a
test over known compressed bytes. OQ-005 found no unknown DJI-private chunk requiring
codec-aware preservation.

## Authoritative archive source set

M7a defines a new `ArchiveSourceSet`; it does not pretend the inspection manifest and the
INV-01 snapshot inventory the same files.

- The source set is every regular, non-symlink file recursively beneath every configured source
  root, using the same session-root `work/` and `output/` carve-out as INV-01 when a source root
  is `.`. It includes audio selected by inspection, ignored edits, duplicates, unassigned audio,
  unexpected file types, nested notes, and any other irreplaceable source-root file.
- Every entry preserves its exact session-relative path, size, and SHA-256. `track_id` is
  optional and is assigned only when the path belongs unambiguously to a configured track input.
  Unassigned entries remain unassigned; M7a never invents identity in violation of INV-11.
- Whole-session upload, verification, and restore include every entry. `--track` includes only
  entries whose optional track ID matches; whole-session restore remains the way to recover
  unassigned files.
- Enumeration uses `lstat`, refuses symlinks at every path component, and proves each resolved
  file remains inside a resolved configured source root. It must not reuse the current
  `raw_guard.snapshot()` traversal blindly because `Path.is_file()` follows a leaf symlink.
- The archive set is hashed once before work and verified unchanged after every success or
  failure path. No archive output, report, lock, or staged file may resolve inside a source root.

## Operator contract

The public CLI distinguishes local comparison from disaster recovery:

```text
dnd-audio archive upload  SESSION_DIRECTORY
dnd-audio archive status  SESSION_DIRECTORY
dnd-audio archive list
dnd-audio archive verify  --session-id SESSION_ID [--track tx-a]
dnd-audio archive restore --session-id SESSION_ID [--track tx-a] --to EMPTY_DIRECTORY
```

- `upload` requires a valid inspection manifest, builds the independent source set, performs a
  local zstd round trip, uploads every immutable object, performs a complete remote GET and
  decompression verification, and publishes the manifest last.
- `status` is cheap and non-authoritative. Against a local directory it reports `absent`,
  `pending`, `committed`, or `divergent` by comparing the source set with the remote manifest.
  It may say `previously_verified_at_commit`; it never calls current remote bytes `verified`.
- `list` discovers committed session IDs and manifest identity without a local session. It
  follows pagination completely; partial listing is never treated as complete.
- `verify` is authoritative and expensive. It requires only a session ID, downloads the
  committed manifest and selected archive objects, hashes compressed bytes, streams them
  through zstd, and hashes/counts restored bytes. Only this current operation reports
  `verified`.
- `restore` likewise requires no lost local metadata. It recreates the manifest's exact
  session-relative paths beneath an empty destination, verifies each temporary output before
  atomic publication, and refuses traversal, symlink escapes, collisions, existing targets,
  destinations inside any current protected source root, and silent overwrite.

There is intentionally no archive-object delete, prune, publication, or raw-reclamation
command in M7a.

## Archive identity, sparse layout, and commit protocol

The remote layout is content-addressed where that prevents ambiguity, versioned before an
encoding change, and still recognizable to a human:

```text
sessions/archive-v1/<session-id>/objects/<encoded-session-relative-path>.<original-sha256>.zst
sessions/archive-v1/<session-id>/archive-manifest.v1.json
```

Path encoding is canonical and reversible; caller strings never become unchecked object keys.
An object key never depends on a fabricated track ID. Archive v1 freezes one byte-stable zstd
recipe. A changed recipe requires archive v2 or a separately versioned object identity, never a
different payload silently written at an existing key.

`archive-manifest.v1.json` is a deterministic, schema-validated commit record containing:

- archive/schema version, session ID, and the complete pinned encoder/decoder recipe;
- one entry per `ArchiveSourceSet` item, sorted by exact session-relative path;
- optional real track ID, original path, byte length and SHA-256;
- compressed byte length and SHA-256 plus immutable object key;
- bounded copied inspection identity when that path has one;
- explicit standalone decoding and path-reconstruction instructions sufficient without this
  repository or the original session directory.

It contains no timestamp, hostname, credential, signed URL, ETag-as-checksum, local absolute
path, or verification telemetry. It is the **only** small JSON object uploaded for a session.
The manifest cannot contain its own final hash; each local operation report records the
manifest hash, following ADR-0003.

One upload for a session ID may run at a time. A local interprocess lock enforces that on the
single supported archive host, and the operator contract forbids concurrent writers elsewhere.
Object keys are immutable. Before committing, `upload` fully GETs an existing fixed manifest:
canonical byte equality is idempotent success and any difference is a fatal divergence. A
HEAD-then-PUT sequence is never presented as a distributed compare-and-swap. The manifest is
PUT last only after all objects pass local and remote verification.

## Integrity, retry, and resource rules

For each source, in deterministic path order:

1. Confirm the original size and SHA-256 from the hardened source set.
2. Preflight worst-case zstd staging space plus operation-report/temporary overhead; never
   budget from the observed 30.4% saving.
3. Compress into at most one staged file outside all source roots using the frozen recipe.
4. Compute compressed size/SHA-256 and stream-decompress locally. Abort immediately if decoded
   bytes would exceed the declared original size; require exact final size and original hash.
5. Upload with a single PUT only within the provider limit. Multipart is mandatory above the
   documented 5 GiB single-PUT limit; non-final parts meet the documented 5 MiB minimum. Persist
   the upload ID before the first part, use bounded injectable exponential backoff for `503 Slow
   Down`, and abort or safely resume interrupted multipart state.
6. Stream a complete remote GET once through compressed hashing and bounded decompression;
   discard restored bytes during verification, enforce the original-size ceiling, and require
   both hashes and sizes.
7. Remove staging and persist the local operation result before moving to the next entry.
8. Recheck the immutable source set before returning, including every failure path. Upload the
   canonical manifest only after every entry completes all checks.

An S3 multipart ETag is never a content checksum. Provider metadata may accelerate `status`,
but cannot replace a full readback. Existing object content is accepted only after full
verification; a conflict is fatal, not an overwrite. Tests inject ENOSPC and interruption at
every phase and prove cleanup. A composed event-log test covers source reader, compressor,
storage adapter, decompressor, and restore writer so no layer can buffer a full source.

Every archive operation produces a mandatory local structured report separate from
`ingest-report.json`. It carries manifest SHA-256 when available, exact scope, per-object
outcomes, current/previous verification distinction, overall `complete`/`failed`/`partial`,
structured secret-free errors, and nonzero partial/failure exit behavior. Upload/status reports
live under session `work/`; remote-only verify and restore use an explicit or documented safe
default report path. Reports and attempts are never uploaded as tiny Cold Storage objects.

## Configuration, permissions, and network boundary

- `ArchiveRuntimeConfig` is separate from `SessionConfig`, `session.yaml`, processing schemas,
  and every processing cache/provenance hash. It is loaded only by archive commands from
  environment variables or a gitignored operator profile. A regression freezes all existing
  stage/cache identities with archive configuration present and absent.
- The bucket is private, has no CDN, and receives no public-read ACL. Endpoint, region, bucket,
  access-key ID, and secret never enter manifests, reports, logs, exceptions, tests, or tracked
  files.
- DigitalOcean bundles multipart-abort capability with broad Read/Write/Delete object
  permission. M7a's upload credential may therefore possess provider-level delete capability,
  but application code exposes no `DeleteObject` operation and never calls it; only
  `AbortMultipartUpload` is allowed for incomplete uploads. `list`, `status`, `verify`, and
  `restore` support a separate read-only credential.
- Only an explicit `archive` subcommand may construct the client or open a socket. `inspect`,
  `ingest`, `activity`, `mix`, `transcribe`, and `process` remain network-denied.
- The storage client sits behind a small interface. Default tests use a deterministic fake and
  the existing socket block. An opt-in `host_smoke` uses only generated non-session bytes and
  exercises remote-only restore plus a forced multipart path without requiring a 5 GiB
  committed fixture.
- M7a amends INV-06 narrowly for owner-controlled archive. It does not weaken cloud-ASR rules.
  INV-01 needs no exception because M7a never deletes source data.
- Versioning, lifecycle policies, object metadata checksums, and provider-native checksum
  headers are defense in depth only; correctness depends on downloaded bytes.

## Explicitly not in this milestone

- Publishing `session.mp3`, transcripts, reports, or wiki artifacts. That is M7b and requires
  Standard Spaces or another delivery surface.
- Local deletion, reclamation, cache sweep, remote-object deletion, retention policy, or
  provider lifecycle automation.
- Changing `process`, making upload automatic, or giving a normal pipeline run network access.
- A global opaque hash-only store, per-source JSON sidecars, remote verification receipts, or
  dozens of sub-128-KiB metadata objects.
- WavPack, adaptive compression, cross-session deduplication, client-side encryption, or
  dependence on provider versioning.
- Uploading real DJI/session audio merely to test the adapter. The provider smoke uses generated
  bytes; the four-file zstd evidence remains a local round trip.
- Treating archive success as permission to discard local raw. M7b must separately justify and
  design any reclamation authority.

## Working plan

1. Amend the product spec and INV-06. Record ADRs for the archive exception, hardened source
   set, zstd v1 recipe and sparse key layout, full-readback commit protocol, provider permission
   boundary, and mandatory archive-operation report.
2. Add `ArchiveRuntimeConfig` outside `SessionConfig`; prove archive settings change no existing
   config, manifest, timeline, activity, mix, transcript, or ASR identity.
3. Implement and test `ArchiveSourceSet` first, including optional track identity, unassigned and
   non-audio entries, root-level layouts, every upload-side symlink/path escape, exact source
   snapshot comparison, and output/report path rejection.
4. Implement local compression/restore with pinned byte-stable zstd output, exact size/hash
   ceilings, worst-case disk preflight, one staged object, atomic restore, ENOSPC cleanup, and a
   composed bounded-resource proof.
5. Implement deterministic manifest and operation-report schemas, remote-only discovery,
   single-writer locking, existing-manifest equality/divergence, and the fake storage client.
6. Implement DigitalOcean upload/read adapter with mandatory multipart thresholds, persisted
   upload IDs, abort-only cleanup, bounded retry, complete pagination, read-only commands, and
   secret sanitization.
7. Implement `upload`, local `status`, remote `list`/`verify`, and whole-session/one-track
   remote-only `restore`; publish the sole small manifest object last.
8. Add CPU/offline negative tests at every corruption/interruption boundary and an opt-in
   generated-data provider smoke plus complete remote-only restore drill. Do not upload real
   session audio as test evidence.
9. Run the independent code review, full zero-skip gate, and `.venv-rocm` default suite; close
   with sanitized evidence and M7b/H1/H2 propagation.

## Completion gate

- [ ] The authoritative product spec, INV-06, schemas, ADRs, OQ references, and downstream
      charters describe the same explicit archive exception, source set, and remote-only
      restoration contract.
- [ ] `upload` archives every hardened `ArchiveSourceSet` entry—including unassigned and
      non-audio regular files—without inventing track identity or mutating a source root, and
      commits the manifest only after local and remote exact restoration.
- [ ] `list`, local `status`, remote-only full/track `verify`, and remote-only full/track
      `restore` require no object-key knowledge; full restore reconstructs every exact relative
      path, size, and original SHA-256 after loss of the session directory.
- [ ] The bucket receives one small manifest object per session and no per-file metadata/report
      objects; every other object is immutable compressed source content.
- [ ] Archive v1 compressed bytes are deterministic under a pinned recipe. Existing objects and
      manifests, repeated/concurrent/interrupted uploads, corruption, multipart boundaries,
      pagination, misleading ETags, and `503` retries fail closed or are provably idempotent.
- [ ] Compression, upload, verification, and restore preflight worst-case disk and are
      bounded-memory streams with output-size ceilings; ENOSPC and interrupted-phase tests leave
      no unsafe staged or published state.
- [ ] Upload and restore independently refuse symlinks at every component, traversal,
      collisions, existing targets, and any resolved path outside the approved root.
- [ ] Mandatory archive reports distinguish `committed`, `previously_verified_at_commit`, and
      current `verified`, carry structured partial failure, and contain no secrets or local
      machine identity.
- [ ] Archive runtime configuration changes no existing processing artifact or cache identity;
      no archive-object delete call or command exists, and no non-archive command gains network
      authority.
- [ ] The fake-backed default suite passes offline/CPU/model-free with zero skips; generated-data
      host smoke proves concrete provider upload, multipart, complete readback, and remote-only
      restore without exposing session audio.
- [ ] Independent Codex plan and code reviews are distilled and every finding is fixed or
      explicitly dispositioned before close.

## Known risks and open questions

- Cold Storage has minimum object billing, minimum storage duration, and retrieval charges;
  authoritative verification intentionally performs full retrieval. Operator messaging must
  show the cost without weakening integrity.
- The provider permission model gives upload credentials more capability than M7a uses. Keeping
  that key only in the upload environment and using read-only credentials elsewhere reduces,
  but does not eliminate, credential-compromise risk.
- The exact provider multipart/checksum/versioning subset may evolve. M7a relies only on signed
  operations verified during implementation and on complete downloaded bytes.
- M7a protects against local loss only after a manifest is committed. Capture failure, transfer
  failure before inspection, compromised credentials, and deletion of both copies remain
  outside what software can prove.
