# ADR-0037 — Archive v1: the frozen recipe, the key layout, and what pins them

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M7a

## Context

M7a's charter carries a measurement: a byte-stream zstd trial over four real DJI
recordings using `zstd 1.5.7` at `-T0 -10` reduced 121,617,184 bytes to 84,663,377
(30.4%) and restored all four original SHA-256 values exactly. That settled zstd against
a first-session WavPack bake-off. It deliberately did **not** freeze the recipe, and the
charter says why: `-T0` may make the output depend on the host.

It does. `-T0` asks libzstd to use as many worker threads as there are cores, and
multithreaded zstd partitions its input differently depending on how many workers it
got. The same file compressed on a 32-core box and a 4-core box produces different
bytes — both valid, both decompressing to the original, and different. An archive whose
key is content-addressed and whose manifest records a compressed digest cannot have
that.

There is a second, subtler version of the same problem: even single-threaded, libzstd's
output at a given level can change between library versions. So "pin the flags" is only
half a recipe; the encoder itself has to be pinned by something.

The project had no compression dependency and no HTTP client. `models fetch` uses stdlib
`urllib`. So both were open questions rather than existing choices.

## Decision

**Archive v1 freezes one recipe, and pins the encoder through `uv.lock` rather than
through a host tool.**

Compression goes through the `zstandard` package, which carries libzstd inside its own
wheel. That is the load-bearing half of the choice: the lock file already pins every
other byte-affecting dependency in this project, and routing the encoder through it
makes "byte-stable" a mechanism rather than a promise about whatever `zstd` happens to
be on `PATH`. `zstandard 0.25.0` bundles **libzstd 1.5.7** — the same version the trial
measured, which is a pleasant confirmation rather than a requirement.

The v1 recipe, stated in full and recorded in every manifest:

- format `zstd`, level **10**
- **`threads=0`** — single-threaded, deliberately not the trial's `-T0`
- `write_checksum=True`, `write_content_size=True`, `write_dict_id=False`
- no dictionary, no long-distance matching
- the `zstandard` and libzstd versions that produced it

A test compresses a fixed vector and compares against a **checked-in digest of the
compressed bytes**, so a dependency bump that changes output fails the gate rather than
silently writing different payloads at content-addressed keys.

**The 30.4% figure is evidence, never a budget.** Disk preflight uses zstd's compression
*bound* — worst case, slightly larger than the input — because a preflight that assumes
compression succeeds is a preflight that fails on incompressible data, which is exactly
what a session of already-compressed files would be.

**The key layout is sparse, versioned, and human-recognizable:**

```text
sessions/archive-v1/<encoded-session-id>/objects/<encoded-path>.<original-sha256>.zst
sessions/archive-v1/<encoded-session-id>/archive-manifest.v1.json
```

The version lives in the prefix. **A changed recipe requires archive v2**, never
different payload bytes at an existing key. Encoding is ADR-0036's: `os.fsencode()`
bytes, percent-encoded, uppercase hex, reversible — applied to the session id exactly as
to a path.

**`SessionConfig.session_id` is not narrowed to make this work.** An earlier draft
proposed refusing session ids outside a strict pattern; the plan review pointed out that
this makes an already valid, already inspected session such as `Session 01` unarchivable,
and that constraining the field instead would move every processing configuration and
cache identity — the exact coupling M7a exists to avoid. So the session id is encoded,
not restricted, and nothing in `SessionConfig` changes.

**`boto3` is the S3 client**, imported lazily inside the provider adapter alone. SigV4
signing, multipart, marker pagination and bounded retry are precisely the code that
cannot be validated without a live endpoint, and hand-rolling them against a bucket that
did not exist yet was the riskiest part of this milestone. Containment is the shape
`onnxruntime` and `torch` already have: one module imports it, a mypy override covers it,
and the boundary is proved behaviourally (ADR-0035).

## Alternatives considered

**Shell out to the `zstd` CLI**, matching the trial exactly. Rejected: it pins the
encoder to a flake input rather than the lock, forces `-T1` anyway to be deterministic,
and puts a subprocess in the middle of a streaming path that has to stay bounded.

**A higher level, or `--long`.** Rejected for v1. Level 10 is what was measured on real
recordings, and the marginal ratio above it costs real time on tens of gigabytes. A
better recipe is a legitimate reason to define archive v2; it is not a reason to leave v1
unpinned.

**WavPack**, which understands audio and would compress it better. Rejected by the
charter, and rightly: WavPack decodes to *samples*, and the guarantee here is the
original *file's* SHA-256 — every chunk, the `PAD`, the iXML, the byte offsets. A
codec-aware format has to reproduce a container it does not own. A byte stream has
nothing to interpret (OQ-005).

**Content-addressed keys on hash alone**, dropping the path from the key. Rejected: it
makes the bucket unreadable to a human, and readability during a disaster is worth a
longer key.

**Hand-rolled SigV4 over `urllib`**, keeping the dependency count at zero. Genuinely
attractive for a project this lean, and rejected on risk: about 200 lines of signing and
multipart logic with no endpoint to test against until the bucket existed.

## Consequences

Compressed bytes are reproducible on any machine that installs this lock, so a
content-addressed key means what it says and an idempotent re-upload is genuinely
idempotent.

Two new runtime dependencies, one of which (`boto3`) is large. Neither is imported by any
processing path, and the subprocess boundary test is what keeps that true.

What would make us revisit: a measurably better ratio on real session audio at
acceptable time (define archive v2 — do not change v1), a `zstandard` release whose
output changes at level 10 (the checked-in digest catches it, and the response is a
version bump plus v2, never a silent rewrite), or the encoder gaining a documented
guarantee of cross-version output stability, which would make the pin less load-bearing
than it currently is.
