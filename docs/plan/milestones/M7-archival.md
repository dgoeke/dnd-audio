# M7 — Archival and local disk reclamation

**Status:** sketch — deliberately unplanned
**Depends on:** M4, M5 (a complete run to archive). Realistically after the first
real session has been processed and validated.
**Spec sections:** none — this is beyond the current spec. Amending the spec is
part of planning this milestone.

> This charter is a placeholder so the work is not forgotten. Its completion gate
> is provisional and its decisions are open. Do not treat the sketch below as a
> design; take it through the normal start phase and plan it properly when the
> time comes.

## Goal

Once a session is processed and validated, package the raw transmitter recordings
and the pipeline outputs for durable off-site storage, verify integrity end to end,
publish the wiki-consumable artifacts, and then — as a separate, manual, explicitly
confirmed step — let the owner reclaim the large local WAVs.

## Sketch of the deliverables

- **Compress `raw/`.** WavPack or zstd (see decisions below), one archive per
  source file or per track, with the original bytes recoverable exactly.
- **Sidecar metadata JSON.** Session ID, per-file original SHA-256, compressed
  SHA-256, the exact compressor + version + flags used, decompression instructions,
  the manifest excerpt describing each source, and the pipeline/schema versions
  that produced the outputs. The archive must be self-describing years later
  without this repository.
- **Upload raw archives to DigitalOcean cold storage.**
- **Upload processed outputs to DigitalOcean Spaces** for wiki and downstream use:
  `session.mp3`, `transcript.json`, `transcript.md`, `ingest-report.json`, and the
  activity/attribution metadata.
- **Verify remote integrity** by reading back and hashing, then recording the
  result.
- **Reclaim local disk** — manual, separate, and only after verification passes.
- **Sweep `work/cache/mix/`, or decide deliberately not to.** M5 added the largest cache
  entry in the project — one mono float32 file at the session's full duration, 2.8 GiB for
  four hours — and nothing prunes it. It is content-addressed on the graph's
  `attribution_cache_key`, so a session mixed under two detectors (real Silero once,
  `--fake-models` once) keeps **two** of them side by side; that is the design working, and
  it is also 5.6 GiB. The sidecar-plus-audio layout makes an LRU or a
  drop-everything-but-the-current-identity sweep straightforward, and `MixCache.get` already
  treats a missing entry as a miss rather than an error, so deleting one only costs a re-mix.
  Reclaiming local disk without addressing this leaves the biggest single consumer untouched.

## Safety properties to preserve

These are the parts worth getting right regardless of how the rest is designed:

1. **Never delete on a successful upload alone.** Deletion requires a verified
   read-back hash matching the local original, recorded in the report.
2. **An S3/Spaces multipart ETag is not a SHA-256.** It is a hash-of-hashes with a
   part-count suffix. Do not use it as an integrity check — either store the
   SHA-256 as object metadata and download-and-hash to confirm, or use a checksum
   algorithm the API verifies natively.
3. **Compression must be verified by round-trip**, decompressing to bytes identical
   to the original, before the original is a deletion candidate. Trusting the
   compressor's own exit code is not enough for a 32-bit-float WAV.
4. **Deletion is never part of `process`** and never automatic. A separate command,
   dry-run by default, that refuses to act on any file lacking a complete
   verification record.
5. **Credentials come from the environment or a secret store.** Never committed,
   never in the report, never in a manifest.
6. **The report records what was archived, where, and with which hashes**, so the
   local record survives even after the local audio does not.

## Tension with existing invariants — resolve explicitly, do not ignore

- **INV-01 (`raw/` is immutable).** Archival ends with the owner deleting raw
  files. That is compatible only because deletion happens outside a pipeline run,
  after verification, by explicit human action. No pipeline stage may ever delete
  from `raw/`. Planning this milestone must amend INV-01's wording rather than
  quietly work around it.
- **INV-06 (session audio never leaves the machine).** Archival deliberately
  uploads audio to owner-controlled object storage. That is not the cloud-ASR
  prohibition the invariant exists to prevent, but it is a real exception and must
  be written into INV-06 as one — scoped to archival, opt-in, and never on a
  processing path.

Both invariants have already been annotated with a pointer here.

## Decisions deferred to planning time (each becomes an ADR)

- **WavPack vs zstd.** Decided by measurement, not argument — see the benchmark
  protocol below.
- **What DigitalOcean actually offers for cold storage**, and whether a distinct
  archival tier exists or Spaces with a lifecycle policy is the real answer. Verify
  against current DO documentation; do not design against a remembered feature set.
- **Public-read vs signed URLs** for the wiki-facing outputs, and what that implies
  about session content being world-readable.
- **Archive granularity** — per file, per track, or per session — and how that
  interacts with partial re-uploads and retention.
- **Retention and lifecycle policy**, including whether anything is ever deleted
  remotely.

## Compressor benchmark protocol (owner's method)

Benchmark on the **first complete real session** — all six tracks, real DJI files.
Not synthetic fixtures: the whole question is how each compressor handles real
32-bit-float content and whatever private chunks DJI actually writes (see OQ-005).

```bash
zstd -T0 -10 source.wav -o source.wav.zst
wavpack -h -v -m source.wav -o source.wv
```

Restore with `zstd -d` and `wvunpack` respectively. Measure, across all six tracks:

- total compressed size,
- encoding time,
- restoration time,
- **SHA-256 of each restored file against the original.** Both must match exactly.

### Decision thresholds

Comparing WavPack's total size against zstd's:

| WavPack vs zstd        | Decision                                              |
| ---------------------- | ----------------------------------------------------- |
| less than 15% smaller  | use zstd                                              |
| 15–25% smaller         | either is reasonable; favor zstd for simplicity       |
| more than 25% smaller  | WavPack is worth the extra dependency for long-term storage cost |
| **any** hash mismatch  | use zstd immediately                                  |

Record the measured numbers in the resulting ADR, not just the verdict — the next
person to revisit this should not have to re-run the benchmark to know how close
the call was.

### One caution about the hash requirement

The byte-exact SHA-256 check is doing real work here, and it is the reason
WavPack could lose outright. WavPack's `-v` verify and `-m` MD5 cover the decoded
**audio stream**; byte-identical file reconstruction depends on its RIFF wrapper
preservation carrying every chunk DJI wrote, including any private or iXML chunks
that OQ-005 has not yet identified. zstd is a plain byte-stream compressor and
cannot have this class of problem.

So: run the hash check on real DJI files with their real chunk layout, and treat a
mismatch as disqualifying rather than as something to work around with flags. An
archive that reproduces the audio but not the file is not an archive of the file.

## Provisional completion gate

- [ ] Round-trip decompression of a real session's raw files is byte-identical to
      the originals, verified by hash.
- [ ] Sidecar metadata is sufficient to identify, decompress, and interpret an
      archive with no access to this repository.
- [ ] Remote objects verify by hash after upload, using something stronger than a
      multipart ETag comparison.
- [ ] Reclamation is a separate command, dry-run by default, that refuses to delete
      anything without a complete verification record, and is never invoked by
      `process`.
- [ ] Credentials are absent from every committed file and from the report.
- [ ] The default test suite still passes with no network — the storage client sits
      behind an interface with a fake, like every other external dependency
      (INV-10).
- [ ] INV-01 and INV-06 are amended to describe the archival exception precisely.
