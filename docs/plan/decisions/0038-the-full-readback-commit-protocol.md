# ADR-0038 — Nothing is committed until it has been read back

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M7a

## Context

A backup that has never been restored is a belief, not a backup. The specific way this
fails is well known and worth naming: the upload succeeds, the provider returns 200, the
operator sees a green report, and the object is truncated, or the compression was wrong,
or the local file changed between hashing and reading. None of that surfaces until the
day the local copy is gone.

S3 offers several things that look like verification and are not. An **ETag** is the
MD5 of the object for a single PUT and, for a multipart upload, a hash *of the part
hashes* with a `-N` suffix — a value that depends on how the upload was chunked and is
not the content digest of anything. Provider-side checksums and versioning are real, but
they are the provider vouching for itself.

There is also a concurrency question the provider cannot answer. The manifest is the
commit marker, so two uploads of one session must not race. The obvious protocol —
HEAD the manifest, and PUT it if absent — is not a compare-and-swap: both writers can
HEAD, both see nothing, both PUT, and the second silently wins. Genuine
compare-and-swap needs conditional create (`If-None-Match: *`), and DigitalOcean's
documented `PutObject` does not expose it.

## Decision

**For each source, in deterministic path order:**

1. Confirm the original size and SHA-256 from the hardened source set.
2. Preflight worst-case staging space from the compression bound (ADR-0037), plus report
   and temporary overhead. Never from an observed ratio.
3. Compress into **at most one** staged file, outside every source root.
4. Compute the compressed size and digest, then **stream-decompress locally** and require
   the exact original size and digest. Abort the moment decoded bytes would exceed the
   declared original size — at the ceiling, not at the final hash, so a corrupt frame
   cannot expand unbounded first.
5. Upload. Single PUT only within the provider limit; **multipart is mandatory** above it,
   with parts at least 5 MiB and at least `ceil(size / 10000)`. Persist the upload id
   **locally** before the first part. Bounded exponential backoff on `503 Slow Down`, with
   the SDK's own retries disabled so exactly one bound governs.
6. **Stream a complete remote GET once**, through compressed hashing and bounded
   decompression, discarding restored bytes. Require both digests and both sizes.
7. Remove staging, persist the local operation result, move on.
8. Re-check the immutable source set before returning, on every path including failures.

**The manifest is PUT last, and only after every object has passed both the local round
trip and the remote readback.** An interrupted upload therefore leaves objects with no
manifest — visible to `status` as `pending`, not as a committed archive.

**An ETag is never a content checksum.** Provider metadata may make `status` cheap; it
may never replace a full readback. Existing object content is accepted only after full
verification, and a conflict is fatal rather than an overwrite.

**On an existing manifest:** `upload` fully GETs it and compares **full entry identity plus
the byte-deciding half of the recipe** — session id, archive version, and every entry's
path, text path, track id, size, original digest and object key. Identical is **idempotent
success**. Any difference is a **fatal divergence**, never a merge and never an overwrite.

Two exclusions, both deliberate, and this paragraph is the amendment that makes the code and
this document agree — the original said "compares canonical bytes", which the implementation
never did.

- **The compressed size and digest are out.** Archive v1's recipe is frozen and
  single-threaded, so identical sources under an identical recipe produce identical
  compressed bytes; comparing them would mean re-compressing a whole session to discover
  whether anything changed. Nothing is lost that the readback does not already cover.
- **The recorded library versions are out.** `ArchiveCodec.describe()` records the
  `zstandard` and libzstd versions because a *difference* in produced bytes deserves an
  explanation attached, but a zstd frame is readable by any later libzstd and the versions
  decide nothing. Comparing them made an ordinary dependency bump report every archived
  session as `divergent` and fail the next `upload` fatally — a false alarm about a backup,
  which is expensive in the only currency this milestone has. Found by M7a's second code
  review; the first review's `track_id` finding is what put the recipe in the comparison at
  all, and it was drawn one field too wide.

**On concurrency, the promise is scoped to what the design delivers.** A local
interprocess lock gives mutual exclusion between processes on the single supported
archive host, and that is tested with two real processes contending. The charter's "no
concurrent writers elsewhere" is named as the **operator precondition it is**, not as
something the software proves — because with no conditional create available, it cannot
be. Saying otherwise would be the more comfortable and less true thing to write down.

## Alternatives considered

**Trust the ETag** and skip the readback. Rejected — it is not a content hash for
multipart objects, and the readback is the only step that proves the thing this milestone
exists to prove.

**Upload the manifest first**, so a partial upload is discoverable as "committed but
incomplete". Rejected: it makes the commit marker lie during the window that matters
most, and `status` distinguishing `pending` from `committed` gives the same discovery
without the lie.

**HEAD-then-PUT presented as a compare-and-swap.** Rejected as false, and it is the
reason the single-writer promise is scoped rather than dropped: a fake guarantee is worse
than a stated precondition, because only one of them makes the operator careful.

**Verify lazily — commit now, read back on a schedule.** Rejected. The failure being
defended against is a *silent* one, and deferring the only check that catches it to a
later run nobody remembers to make is how backups rot.

**Per-object receipt objects** recording each verification. Rejected: Cold Storage bills
anything under 128 KiB as 128 KiB, so dozens of tiny receipts per session are pure cost,
and the local operation report holds the same information where it is free.

## Consequences

Every archived byte has been compressed, decompressed, uploaded, downloaded and
decompressed again before the archive claims to hold it. That is roughly twice the
network of a naive upload and it is the entire point.

Full verification is genuinely expensive on Cold Storage, which charges per retrieval
($0.01/GiB, waived up to average daily usage). The operator messaging states the cost
without weakening the check, because the alternative is an archive nobody has tested.

What this makes hard: a very large session's `verify` is a long, costly operation, and
there is a real temptation to add a sampling mode. If that is ever added it must be named
something other than `verify`, and must never produce the word "verified".

What would make us revisit: a provider offering conditional create, which would turn the
scoped single-writer promise into a real one and would be worth the migration.
