# M7a — Verified private raw archive

**Status:** closed
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
Object keys are immutable. Before committing, `upload` fully GETs an existing fixed manifest
and compares full entry identity plus the byte-deciding half of the recipe (ADR-0038):
identity equality is idempotent success and any difference is a fatal divergence. A
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

## Completion gate

- [x] The authoritative product spec, INV-06, schemas, ADRs, OQ references, and downstream
      charters describe the same explicit archive exception, source set, and remote-only
      restoration contract.
- [x] `upload` archives every hardened `ArchiveSourceSet` entry—including unassigned and
      non-audio regular files—without inventing track identity or mutating a source root, and
      commits the manifest only after local and remote exact restoration.
- [x] `list`, local `status`, remote-only full/track `verify`, and remote-only full/track
      `restore` require no object-key knowledge; full restore reconstructs every exact relative
      path, size, and original SHA-256 after loss of the session directory.
- [x] The bucket receives one small manifest object per session and no per-file metadata/report
      objects; every other object is immutable compressed source content.
- [x] Archive v1 compressed bytes are deterministic under a pinned recipe. Existing objects and
      manifests, repeated/concurrent/interrupted uploads, corruption, multipart boundaries,
      pagination, misleading ETags, and `503` retries fail closed or are provably idempotent.
- [x] Compression, upload, verification, and restore preflight worst-case disk and are
      bounded-memory streams with output-size ceilings; ENOSPC and interrupted-phase tests leave
      no unsafe staged or published state.
- [x] Upload and restore independently refuse symlinks at every component, traversal,
      collisions, existing targets, and any resolved path outside the approved root.
- [x] Mandatory archive reports distinguish `committed`, `previously_verified_at_commit`, and
      current `verified`, carry structured partial failure, and contain no secrets or local
      machine identity.
- [x] Archive runtime configuration changes no existing processing artifact or cache identity;
      no archive-object delete call or command exists, and no non-archive command gains network
      authority.
- [x] The fake-backed default suite passes offline/CPU/model-free with zero skips; generated-data
      host smoke proves concrete provider upload, multipart, complete readback, and remote-only
      restore without exposing session audio.
- [x] Independent Codex plan and code reviews are distilled and every finding is fixed or
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

---

## Closeout

### What works end to end

`dnd-audio archive upload SESSION` compresses every file in a hardened, independent source
set with a frozen single-threaded zstd recipe, uploads each as one immutable
content-addressed object, **downloads every one of them again and decompresses it back to
the original SHA-256**, re-verifies that no source moved, and only then publishes a single
small manifest as the commit marker. `status` compares a local session cheaply and
structurally cannot claim `verified`. `list`, `verify` and `restore` need no local session
directory at all.

Confirmed against the owner's real Cold Storage bucket on 2026-08-04, and **re-confirmed on
the final code** after the third review's fixes: a synthetic session uploaded and verified,
its directory was deleted outright, and every file came back at its exact relative path,
size and SHA-256 from the session id alone — plus `list` discovering that id remotely, a
forced multipart round trip, and confirmation that a multipart ETag is **not** the content
digest.

Restore is transactional: the whole tree is staged beside the destination and moved in at the
end, so a failure leaves the destination untouched and the retry is just a retry.

### Tests and commands run, with results

- `./scripts/gate.sh` — **8 checks, 2 640 passed, zero skips**, ~39 s.
- Default suite from `.venv-rocm` — passed, after that environment was re-synced for the two
  new dependencies. The first attempt failed with five errors, which is the drift this run
  exists to catch.
- `./scripts/codex-review.sh plan M7a` — **ten findings, nine accepted in full, one in part.**
- `./scripts/codex-review.sh code M7a main` — **eleven findings, four P0, all eleven fixed.**
- Host smoke against the real bucket — upload, multipart, complete readback, `verify`,
  `list`, remote-only whole-session restore, and track-scoped restore.
- **Three mutation checks**, deleting production code to confirm a test goes red: the
  pre-commit INV-01 verification, the compression-streaming proof, and (by inspection) the
  archive-v1 recipe freeze across levels 8–12.

**Then the verify phase, which had been skipped, was run — and it reopened the milestone.**
See `docs/plan/reviews/M7a-code-20260804-2109.md`.

- `./scripts/gate.sh` at the reopened HEAD — **8 checks, 2 656 passed, zero skips**, ~46 s.
  (The number recorded above, 2 640, was never produced by this tree: the close commit's
  HEAD gives 2 629. That mismatch is what first suggested the final gate run had not been
  the one at HEAD.)
- A second `./scripts/codex-review.sh code M7a main`, a fresh-context reviewer agent, and
  the verifier's own pass — **two P0, six P1/P2, two deferred, one rejected.**
- Host smoke against the real Cold Storage bucket, re-run twice during verification —
  **9 passed in 181 s**, then 9 passed again after the `status` reordering, including
  forced multipart and remote-only restore.

- **Host smoke re-run at final HEAD** (2026-08-04, at the operator's instruction before
  closing), which is what the paragraph that used to sit here asked for: the third review's
  fixes touch code the smoke is the only executed proof of — `list` downloading and parsing
  each manifest, `_read_remote_manifest` checking session ownership, the readback draining
  to EOF, restore refusing a leftover staging tree — and its previous proof predated them.
  **9 passed in 7.6 s.** Upload → `list` → `verify` → delete the session directory →
  remote-only whole-session restore, plus track-scoped restore, plus forced multipart with
  its non-content ETag. Both runs' session objects were confirmed present in the bucket
  afterwards, so the drill really transferred rather than short-circuiting.

  **The first attempt failed, and the defect was in the test.** See the note below on
  ordering; the pass above is from the fixed test against a genuinely empty prefix.

- `./scripts/gate.sh` after that fix — **8 checks, 2 656 passed, zero skips**, 40.3 s.
- **Two more mutation checks**, both on the new CLI tests: deleting the report-path INV-01
  guard turns two red, and reverting the protected-roots wiring turns the third red. The
  retry fix was checked the same way — with the handle opened outside the thunk, the stored
  object is `b''`.

### Decisions made (→ ADRs)

- **ADR-0035** — the archive network exception, and the provider permission boundary with it.
- **ADR-0036** — the hardened source set, and byte-level path encoding.
- **ADR-0037** — archive v1: the frozen recipe, the sparse key layout, the two dependencies.
- **ADR-0038** — nothing is committed until it has been read back.
- **ADR-0039** — the archive operation report, and the three words for "checked".

INV-06 is reworded from "session audio never leaves the machine" to "session audio never
reaches anything that processes it", in `AGENTS.md` twice and the spec twice. INV-01 needed
no exception: M7a deletes nothing.

### Assumptions made and open questions raised

**OQ-028 raised and answered.** It asked which listing API paginates, because DigitalOcean's
compatibility page and its limits page contradict each other. The answer is that listing
works — and the finding that mattered was ours, not the provider's. See the notes below.

### Deferred, and why

- **A source replaced at the same path between inspection and archiving is accepted**, and
  the bounded `inspection` block copied into its manifest entry then describes the old file.
  `_require_current_manifest` checks only `config_hash`; `_require_nothing_vanished` checks
  only presence. The archive itself stays correct — the original digest is measured from the
  bytes actually read, so restoration is exact — and the copied identity is decoration.
  Closing it means re-hashing every source against the inspection manifest, a second full
  pass over a session for a decorative field. Worth doing when something else already needs
  that pass.
- **`tests/test_archive_memory.py` proves the staged-read boundary against `WatchedStorage`,
  not `SpacesStorage`.** The real adapter hands boto3 a file handle and the multipart path
  reads fixed-size chunks — bounded by inspection, and exercised for real by the host smoke —
  but that specific boundary has no executed proof over production code. The other four
  boundaries do.
- **`multipart_part_bytes` may be configured up to 5 GB**, and a part is held in memory as
  `bytes` so a retry can resend it. The default is 64 MiB and the ceiling is the provider's
  single-PUT limit rather than anything this host can afford; on a UMA machine `systemd-oomd`
  is watching. Worth lowering the configurable maximum to something a session can actually
  survive. Raised by M7a's third code review alongside the point above.
- **`upload` re-verifies the source set before the manifest PUT, not after it.** The third
  review asked for one more check after the commit. Rejected on cost — `verify_unchanged`
  re-walks and re-hashes the whole session, so closing a window the length of a 2 KB PUT
  would double the hashing of every upload — but recorded because the charter's step 8 says
  "before returning" and the implementation reads "before committing". If a cheap
  incremental source check ever exists, this is where it goes.

### Notes for future implementors

**The endpoint must be the regional one, and getting it wrong is silent.** DigitalOcean's
control panel shows a bucket as `<bucket>.<region>.digitaloceanspaces.com`, which is the
natural thing to paste into `endpoint_url` — and the bucket is *also* a request parameter, so
boto3 addresses path-style and **the bucket name becomes part of every object key**. Upload,
readback, `verify` and `restore` all succeed, because put and head are wrong identically.
Only a listing disagrees. `ArchiveRuntimeConfig` now refuses that endpoint shape at load. If
you are debugging something similar, the lesson generalizes: a system that is
self-consistently wrong passes every test that asks it about itself.

**The region is correctly present in both the URL and the config.** It is the SigV4 signing
scope, not an addressing component. Only the bucket was being applied twice.

**Every layer needs a test that enters through its own front door.** Nine test files, 261
archive tests and complete coverage of the runner did not compensate for the fact that
*nothing ran a command*: `tests/test_cli.py` reached `archive --help` and an unconfigured
`archive list`, which exits above everything interesting. So the CLI's INV-01 guard, its
protected-root wiring and its exit codes were carried by nothing — and a P0 that overwrote a
source recording sat in that gap. Test the seam you actually ship.

**The habit applies to the fixes too, not only to the code being fixed.** The commit that
resolved the first code review shipped a test whose docstring says "This drives the actual
command" above a body that asserts on a helper in isolation — guarding the very P0 the review
had just found, in the same block where the next P0 was hiding. One review's lesson does not
carry itself into the commit that acts on it.

**A retry must be able to run twice.** `put_object` opened the body once and retried the
call; a real request sends the body before it can be told `503`, so the retry PUT zero bytes
at an immutable content-addressed key — and since `_publish` will not overwrite one and there
is no delete command, a single transient error made a session permanently unarchivable. Any
thunk a retry loop re-runs must acquire its own resources inside itself.

**A test that shares remote state with another test is ordered by the scheduler, not by
the file.** `-n 8` is in `addopts`, so the smoke's two pagination tests ran on different
xdist workers in whichever order came up — and one of them uploaded the five objects the
other listed. It passed twice anyway, because the keys are fixed and the *previous* run's
objects were still in the bucket when the listing test went first. Emptying the bucket is
what finally made it fail, on the very run that existed to re-establish OQ-028's evidence
against the final code. Both tests now seed their own objects from a fixture, and the fix
was proved by deleting the five objects and re-running rather than by re-running on top of
them. A test whose green depends on the residue of its own history is not evidence, and
remote state makes that failure survive across processes where `tmp_path` would not.

**Each smoke run leaves a whole session archive in the bucket permanently.** The session id
is derived from `tmp_path`, so every run commits a new one, and M7a deliberately ships no
delete command — cleanup is a console action or a throwaway script outside this project,
under the guard that every key starts with the prefix you meant. Four smoke sessions and
the `smoke/` objects were in the bucket at close, ~1.1 MB of real content against a 128 KiB
per-object billing floor and a 30-day minimum retention.

**Do not trust a proof you have not tried to break.** Eight proofs were rewritten after the
plan review and eleven more defects survived into the implementation, most of them tests that
could not fail. Two were caught only by deleting the production code and watching the test
stay green — including, twice in a row, the INV-01 pre-commit check. The first rewrite
mutated a file the upload loop had not yet reached, so the per-entry re-hash caught it and
the test passed with the real check deleted. **If a test claims to prove a specific line
matters, delete that line once.**

**`python -m dnd_audio.cli` runs nothing.** It has no `__main__` guard, so it imports, defines
`main`, and exits 0. The subprocess network-boundary proof invoked it that way and passed
unconditionally across all eight commands. `src/dnd_audio/__main__.py` exists now, and
`test_the_command_actually_ran` is the guard that keeps it honest.

**`max_attempts` in botocore counts retries, not attempts.** `{"max_attempts": 1}` permits one
SDK retry beneath every project-level attempt. Use `total_max_attempts`.

**Name state files by digest, not by encoded key.** A valid 296-byte object key encodes to a
313-byte filename, past the 255-byte component limit — so multipart failed before its first
part on exactly the long paths the key limit permits.

**Cold Storage costs are real.** 128 KiB minimum per object, 30-day minimum retention,
retrieval charged per GiB (waived up to average daily usage). `verify` is a full download by
design; keep test uploads small and few.

### Deviations from this charter, and why

- The session id is **encoded, not restricted**. An earlier plan narrowed
  `SessionConfig.session_id`; the plan review pointed out that this makes an already-inspected
  `Session 01` unarchivable and would move every processing cache identity.
- The concurrency promise is scoped to a single host. The provider offers no conditional
  create, so there is no compare-and-swap to be had, and the charter now names the operator
  precondition rather than implying a distributed guarantee.
- The manifest comparison is full entry identity plus the recipe, not literal canonical byte
  equality. Compressed fields are implied by the frozen recipe, and comparing them would mean
  re-compressing a whole session to learn whether anything changed.
- The proof table's "delete the session directory, then exercise local `status`" was
  impossible and is corrected: `status` takes a local directory by definition.

### Downstream charters updated

- **M7b** inherits the endpoint lesson, the cost model, and an archive that deliberately has
  no delete, prune or publish command to build on.
- **`OPEN-QUESTIONS.md`** carries OQ-028's answer.
- The product spec, `AGENTS.md`, INV-06 and five ADRs were amended before implementation, not
  after.

### Next smallest step

**M10, then H1.** M7a removes the risk that a lost disk costs a session, which was the reason
to do it before Session Zero. Archive the first real session as soon as it is inspected — the
command is `dnd-audio archive upload`, and `archive verify` afterwards is what makes the
backup a fact rather than a belief.
