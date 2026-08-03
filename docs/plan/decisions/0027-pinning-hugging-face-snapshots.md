# ADR-0027 — Pinning Hugging Face snapshots, and who owns the download

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M6b

## Context

M3 pinned Silero as a single 2.3 MB file: a `ModelDescriptor` naming the repository, the
release, the immutable commit, the size, and the sha256, with `find_model` refusing anything
that is not exactly those bytes. `models.py` says outright that this shape is provisional
until M6b, because the ASR stack is a different problem: two logical models, eleven and ten
files, roughly 4.7 GB and 1.2 GB, and a downloader that must stream rather than return
`bytes`.

Three things constrain the answer.

**The spec names one network command.** "`models fetch` is the only command expected to
require network access for model installation; it resolves and records immutable Hugging
Face snapshot revisions for later offline use." INV-06 says the same thing, and so does
M6b's own completion gate. Any design that adds a second network authority has to amend
three documents that agree with each other.

**The owner asked for the `hf` CLI and for one-time setup.** Not a lazy download on first
use, not a hand-rolled HTTP client: a step run once against a machine, which the pipeline
then depends on and refuses to run without.

**Configuration may override the revision.** The spec requires that explicit model and
aligner revisions be settable, and that `process` use the lock rather than re-resolving a
moving branch online. Those two sentences pull against a manifest checked into source: a
manifest can only describe the commit it was written for.

## Decision

**`dnd-audio models fetch --qwen` drives the `hf` CLI.** It shells out to
`hf download <repo> --revision <commit> --local-dir <staging>`. That satisfies the owner's
request — the `hf` CLI does the work, it is one-time setup, the pipeline hard-fails without
it — while leaving exactly one network authority, so the gate, INV-06 and the spec all stay
as written. `scripts/fetch-models.sh` is a wrapper that runs the command inside the FHS
shell, which is necessary rather than decorative: `hf` ships with the `huggingface_hub` that
lives in `.venv-rocm` and is deliberately absent from `.venv`.

**A model installation is keyed by `(repository, resolved commit)`.** Not by a name, and not
by a fixed pair of snapshots. A `SnapshotDescriptor` checked into `models.py` carries the
default commit and every file's path, size and sha256 — `SILERO_VAD`'s treatment, scaled up.

**A configured revision must be a 40-character lowercase hex commit SHA.** Validated at
configuration load, so a branch name is refused before any run begins. This is what makes
"`process` uses the lock rather than re-resolving a moving branch" true by construction:
there is no mutable name left in the system to re-resolve.

**The lock is the authority for an overridden revision; the checked-in manifest is the
authority for the default one.** For the default commit the manifest is stronger, because it
was reviewed and committed. For an override there is nothing else that knows what the tree
should contain, so verification is against the per-file digests `models fetch` recorded when
it installed that commit. This deliberately makes the lock *authoritative* for snapshots,
which is the opposite of `read_lock`'s existing documented stance for Silero — "the lock is
a convenience and an audit trail; `find_model` is the authority, and it consults the bytes".
Both stances are correct for their subject and the difference is stated here rather than
left to be discovered.

**The installed tree is an exact allowlist in both directions.** Every pinned file present
at its pinned size and digest, *and no unpinned file anywhere in the tree*. `hf download`
writes into a staging directory and only the pinned files move into place. An extra
`config.json`, tokenizer file, or custom-code module is something Transformers would load
without comment, so its presence is a verification failure rather than an unnoticed extra —
and `hf`'s own `.cache/huggingface` metadata never reaches the loadable directory at all.

**A snapshot that does not verify is treated exactly as absent**, and the diagnostic names
the command that fixes it. That is `find_model`'s rule, for `find_model`'s reason: a
half-downloaded 4.7 GB shard at the right path is worse than no file, because absence is
diagnosable and corruption surfaces as a slightly wrong transcript.

## Alternatives considered

**Hand-roll the download over `urllib`, as `models.py` already does for Silero.** Genuinely
viable — the endpoints were tested from the target host during planning, `paths-info` yields
per-file sizes and LFS sha256, and `/resolve/<commit>/<path>` streams. It was rejected
because the owner asked for the `hf` CLI, and because roughly 350 lines of retry, redirect
and streaming code would be reimplementing a maintained library to gain nothing the
verification step does not already provide.

**Add `huggingface_hub` to the base dependencies and call `snapshot_download`.** Rejected on
cost: six new packages (`requests`, `tqdm`, `fsspec`, `filelock`, `hf_xet`, `packaging`) in
the environment the offline gate runs in, for a download helper — and `snapshot_download`
verifies considerably less than `find_model` already does.

**A separate network-capable setup script, with the gate criterion and INV-06 amended to
permit it.** This was the first draft of M6b's working plan. The plan review rejected it
correctly: the *spec* also names `models fetch`, so this would have meant changing three
documents that agree, to avoid writing one subcommand.

**Resolve `main` to a commit at fetch time and record only that**, rather than checking a
manifest into source. Weaker in the case that matters: it pins what was downloaded, not what
was reviewed, so a substituted artifact at an unchanged commit would be adopted silently on
a machine that had never fetched before.

**Let `find_snapshot` ignore unpinned files.** Rejected once the review pointed out that
`hf download --local-dir` writes metadata into the target: "the pinned files are correct" is
not the same claim as "this directory contains only what was pinned", and Transformers reads
the directory, not the manifest.

## Consequences

Verifying a snapshot means hashing about 6 GB, which takes a few seconds of disk read. It
happens once per run, at resolution time, before any cache is written — the same place
`silero_bundle` re-hashes the bytes it is about to execute.

Updating either model is a deliberate edit: a new commit and a new file manifest, in a
commit that can be reviewed. That is the intent. It also means the descriptors go stale
silently if upstream publishes a better checkpoint, which is a cost worth paying and a thing
to notice at the next milestone that touches ASR.

`models fetch --qwen` cannot run from `.venv`, because `hf` is not there. The wrapper script
exists so that is a documented step rather than a confusing `FileNotFoundError`, and the
error message when `hf` is missing names the script.

The 40-hex constraint on configured revisions means an operator cannot ask for `main`. That
is the point, and the diagnostic says so with the command that resolves a branch to a
commit.

Nothing here is an assumption about the world that evidence could overturn, so no `OQ-`
entry is attached. The assumption that *is* worth registering — whether the model returns
identical output across cold runs at all — belongs to the adapter and is **OQ-022**.
