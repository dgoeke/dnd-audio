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

Scratch section, replaced by the Closeout at the end. Approved by the owner 2026-08-04, after
the provider facts below were re-verified against DigitalOcean's own documentation.

### Provider facts, re-verified 2026-08-04

The charter's numbers hold. Single PUT up to **5 GB**; multipart parts at least **5 MiB**
except the final one; at most **10 000 parts**; 5 TB total. Cold Storage bills an object under
**128 KiB** as 128 KiB, charges a 128 KiB minimum per retrieval, imposes a **30-day** minimum
storage duration, and is rate-limited to **450 write / 250 read / 25 list** requests per
second. Retrieval is **$0.01 per GiB**, waived up to average daily Cold Storage usage — the
cost message says both halves, because quoting the fee alone overstates the bill. Bucket type
is chosen at creation, cannot be changed, and a Cold bucket can only be created in the control
panel. Endpoints are `<region>.digitaloceanspaces.com`. Cross-region
`CopyObject`/`UploadPartCopy` are unsupported; neither is used here.

**One provider fact contradicts itself, and it decides the listing implementation.** The
compatibility page says `ListObjectsV2` is supported. The limits page's **Known Issues**
section says verbatim that "The Spaces API does not currently support `list-objects-v2`
pagination". Archive v1 therefore uses **legacy `ListObjects` marker pagination outright**
rather than V2 with a fallback: a path that runs only when a provider bug is present is a
path nobody exercises. **OQ-028** records the contradiction and the host smoke settles it.

### Dependencies chosen

**`zstandard`, single-threaded.** libzstd travels inside the wheel and is therefore pinned by
`uv.lock` like everything else, which is what makes archive v1's "byte-stable recipe" a
mechanism rather than a promise about a host tool. The trial's `-T0` is deliberately not the
v1 recipe: multithreaded zstd output depends on the host's thread count. The 30.4% saving
stays as *evidence* and is never a disk budget.

**`boto3`, imported lazily inside the provider adapter only.** SigV4, multipart, pagination and
bounded retry are exactly the code that cannot be validated without a real endpoint, and
hand-rolling them was the riskiest part of this milestone. Containment is the same shape
`onnxruntime` and `torch` already get: one module imports it, a mypy override covers it, and a
subprocess import-closure test proves no processing command can reach it.

### Order of work

1. **Ledger, working agreement and spec, before any code.** Four unconditional prohibitions
   forbid this milestone as written and all four are amended together: `AGENTS.md`'s summary
   line ("No audio ever leaves this machine") **and** its hard rule ("Never send audio off
   the machine"); the spec's firm-scope line ("Audio must not be sent to a cloud API"); and
   the spec's target-host line ("must never be sent to a URL or API"). INV-06 moves from
   "planned exception" to the real one, and the spec gains an archival extension section and
   the new command shape. The exception is worded narrowly in every location: **only an
   explicit `archive` command may send immutable source bytes to the configured
   owner-controlled private archive; every processing and model path stays local.**
   `AGENTS.md` is named in the first completion criterion.

   Five ADRs, not seven: the archive network exception **together with** the provider
   permission boundary, since one is the operational face of the other; the hardened source
   set; archive v1 (the `zstandard` recipe, the sparse key layout, and byte-level path and
   session-id encoding — the dependency choice *is* the recipe); the full-readback commit
   protocol; and the mandatory archive operation report.
2. **`ArchiveRuntimeConfig`, outside `SessionConfig`.** Environment variables or a gitignored
   operator profile; secrets as `SecretStr` so no repr, log, or exception can carry one. The
   regression that matters lands here, and it freezes **cache keys rather than output
   bytes**: identical artifacts after a cache miss prove nothing about identity. With every
   `DND_AUDIO_ARCHIVE_*` variable set and unset, the complete identity document *and*
   sidecar path of every cache a run touches — inspection, derivative, detection,
   attribution, mix and **ASR**, whose identity is not a `stage_config_hash` at all — must be
   unchanged, measured once warm and once after clearing the cache tree.
3. **`ArchiveSourceSet`.** `lstat` walk, symlink refused at every path component, resolved-root
   containment, the session-root-only `work`/`output` carve-out, and `track_id` assigned only
   where a path belongs unambiguously to one configured track input.

   **Encoding is frozen over `os.fsencode()` bytes, not text.** Linux permits non-UTF-8
   filenames while `canonical_json` emits UTF-8, so a surrogate-escaped name would fail
   serialization outright. The manifest's authoritative `path` is the pure-ASCII
   percent-encoded byte form, always serializable; a human-readable `path_text` appears only
   when the name is valid UTF-8; restore reconstructs from the byte form. The session id is
   encoded the same way — **`SessionConfig` is not narrowed**, because doing so would move
   every processing identity and make an already-inspected session such as `Session 01`
   unarchivable.
4. **Codec.** The frozen v1 recipe through `zstandard` with `threads=0` and every frame
   parameter stated explicitly, `compress_bound` for worst-case preflight, and decompression
   with a hard output-size ceiling that aborts before the final hash rather than after it.
5. **Manifest, operation report, single-writer lock.** Both new schemas join
   `schema_export.py`, so the existing drift test covers them without a new mechanism.
6. **Storage seam and deterministic fake.** The `ArchiveStorage` protocol has no delete member.
   That alone proves nothing — a grep for `DeleteObject` passes with `client.delete_object(…)`
   sitting in the adapter — so the real proof is a **recording client under an explicit
   operation allowlist**: `delete_object` and `delete_objects` rejected by name, and
   `abort_multipart_upload` the only destructive operation permitted. The fake injects
   ENOSPC, `503 Slow Down`, interruption at each phase, corrupt readback bytes, a misleading
   multipart ETag, and small pagination pages.
7. **DigitalOcean adapter.** Mandatory multipart above the threshold, part size at least 5 MiB
   *and* at least `ceil(size / 10000)`, the upload ID persisted **locally** before the first
   part, bounded injectable backoff on `503`, abort-only cleanup, complete marker pagination.
   The client is constructed with **SDK retries disabled**: botocore retries on its own, and a
   project-level loop layered on top of it bounds nothing.
8. **`upload`, `status`, `list`, `verify`, `restore`**, and the `archive` CLI sub-app.
   **Restore is transactional.** Publishing each file atomically and then refusing existing
   targets strands its own retry — an ENOSPC at file 20 leaves 1–19 behind and the next
   attempt must refuse them. So the whole tree is staged beside the destination, verified
   complete, and renamed into place; a failed restore leaves the destination untouched. Space
   is preflighted as the sum of original sizes plus overhead, not one file at a time.
9. **The adversarial pass**, then the independent code review, the full zero-skip gate, the
   default suite from `.venv-rocm`, the generated-bytes host smoke against the real bucket,
   and close with sanitized evidence plus M7b/H1/H2 propagation.

### Completion-gate criteria, each mapped to its proof

Every proof here must be able to *fail*. The plan review rejected eight of the first table's
ten rows for asserting something weaker than the criterion above it, so each row now names a
test whose failure mode is the criterion's failure mode.

| Criterion | Proof |
| --------- | ----- |
| One explicit exception across `AGENTS.md`, spec, INV-06, schemas, ADRs, OQs, charters | `scripts/check_plan.py` in the gate and `tests/test_schema_drift.py`. **No prose-scanning test**: reading committed wording back is ceremony, and the behavioural boundary tests below are what actually hold |
| Every hardened entry archived without invented identity or source mutation | `tests/test_archive_sourceset.py` over nested notes, unassigned files, non-audio, duplicates, ignored edits, a root-level layout, and `raw/tx-a/work/notes.txt`; exact comparison against `raw_guard.snapshot`; source set re-verified on every exit path including each failure |
| Remote-only recovery needs no object-key knowledge | `tests/test_archive_run.py`: upload, **delete the whole session directory**, then exercise `list`, full and per-track `verify`, and full and per-track `restore` from the session id alone — **not `status`**, which takes a local directory by definition and cannot be part of a drill that deletes one — comparing every relative path, size and SHA-256 — plus an unknown track id, a session whose unassigned files only whole-session restore recovers, and a listing that spans several marker pages |
| One small manifest object per session | compare the **exact remote key set** against `{manifest} ∪ manifest.object_keys`. Counting keys outside `objects/` would miss a report accidentally written beneath it |
| Deterministic bytes; conflicts fail closed or are idempotent | a known-compressed-bytes vector in `tests/test_archive_codec.py`; a byte-identical existing manifest is idempotent success and any difference fatal; an existing object accepted only after full verification; an interrupted upload resumed; **two real processes contending for the lock**, one of which is refused |
| Bounded streams with ceilings, worst-case disk preflight | **phase-typed** event logs asserting each boundary independently — source reads interleave with compression writes, staged reads with upload calls, remote reads with decompressor consumption, restore decoding with destination writes — plus "no size-less `read()` anywhere". One combined ordering assertion is satisfied by an early compressor write while a later verifier buffers a whole object. Preflight is tested **directly at and just below its computed bound**; injected ENOSPC proves cleanup, not arithmetic |
| Symlink, traversal, collision and existing-target refusal | independent upload-side and restore-side path tests, neither reusing the other's helper; encoding tests over `%`, `/`, newline, non-ASCII normalization contrasts, invalid UTF-8 bytes, and the key's UTF-8 **byte** length |
| Reports distinguish the three verification states | `tests/test_archive_report.py`, plus a secret-scan over serialized reports, log output and exception text |
| No processing identity moves; no delete authority appears | the step-2 freeze over **real cache keys**, warm and cold; the recording-client **operation allowlist** rejecting `delete_object`/`delete_objects`; and every non-archive command — `inspect`, `ingest`, `activity`, `transcribe`, `render`, `mix`, `process`, `doctor` — run as a subprocess with a socket-and-client trap on its `PYTHONPATH`, since a subprocess escapes the autouse socket fixture (INV-05) |
| Offline zero-skip default suite; provider smoke separately | `./scripts/gate.sh`; `tests/test_archive_smoke.py` marked `host_smoke`, generated bytes only, with a forced multipart path and a `MaxKeys=1` pagination drill that answers OQ-028 |

### Amendments this plan makes to the charter above

1. **The unconditional prohibitions in `AGENTS.md` are amended too**, not only the spec's and
   INV-06's. The charter's first criterion did not name the working agreement, and
   implementing upload while its hard rule stands is a hard-rule violation.
2. **"requires a valid inspection manifest" becomes concrete.** The manifest must exist, parse,
   and carry a `config_hash` equal to the current resolved configuration; a stale one is fatal
   and names `dnd-audio inspect`.
3. **The remote-only report path** is `--report PATH`, defaulting to
   `$XDG_STATE_HOME/dnd-audio/archive/`.
4. **The v1 recipe is single-threaded**, not the trial's `-T0`.
5. **Object keys are length-bounded** in UTF-8 bytes. Percent-encoding can triple a path's
   length against a 1024-byte limit, so an entry whose key would exceed it is refused, never
   truncated.
6. **The concurrency promise is narrowed to what the design delivers.** DigitalOcean's
   `PutObject` offers no conditional create, so there is no compare-and-swap to be had. The
   charter promises **single-host mutual exclusion through the local lock**, and names the
   "no concurrent writers elsewhere" clause as the operator precondition it is rather than
   something the software proves. Selecting a different provider to obtain conditional
   creation is rejected: the bucket exists, and choosing a delivery surface is M7b's.

**Withdrawn after review.** An earlier amendment proposed refusing a `session_id` outside a
strict pattern. That would make an already valid, already inspected session unarchivable, and
narrowing `SessionConfig` instead would move every processing cache identity — the exact
coupling this milestone exists to avoid. The session id is encoded, not restricted.

### Deliberately not done, from the non-goals above

Publishing anything; any delete, prune, retention or provider lifecycle automation; automatic
upload from `process`; a global hash-only store; per-file sidecars; WavPack; client-side
encryption; local raw reclamation; and uploading real session audio as test evidence.

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
