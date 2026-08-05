# ADR-0036 — The archive source set is its own enumeration

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M7a

## Context

Two inventories of a session's files already exist, and neither is the one an archive
needs.

`manifest.json` records *candidate audio*: what discovery found, what it selected, what
it ignored as an `edit`, what it called a duplicate, and what it found where no track
was configured. It does not record a `notes.txt` sitting beside the recordings, because
nothing in the pipeline reads one.

`raw_guard.snapshot()` records *every regular file* under the source roots, which is
much closer — it exists to prove INV-01, and "we did not touch what we read" would be a
weaker claim than the invariant makes. But it was built to compare a tree against
itself, and it has a property that is correct there and dangerous here: it tests
`path.is_file()`, which **follows a leaf symlink**. For its own purpose that is fine —
a symlink whose target changed shows up as a changed hash. For an uploader it means a
link named `raw/tx-a/notes.txt` pointing at `~/.ssh/id_ed25519` would be read,
compressed, and sent to a bucket under an innocent key.

The first M7a draft proposed a mandatory `track_id` key for every archived file. The
first plan review rejected it: nested notes have no track, unassigned audio must stay
unassigned (INV-11), and a mandatory track key could represent neither.

## Decision

M7a defines **`ArchiveSourceSet`, a third enumeration**, and does not pretend either
existing one will do.

- **Every regular, non-symlink file** recursively beneath every configured source root,
  using the same session-root-only `work/`/`output/` carve-out INV-01 uses when a source
  root is `.`. That carve-out is at the session root **only** — `raw/tx-a/work/notes.txt`
  is archived, for the same reason M2 found it must be hashed.
- Each entry keeps its exact session-relative path, size and SHA-256. **`track_id` is
  optional**, assigned only where a path belongs unambiguously to one configured track
  input. An unassigned file stays unassigned; M7a never invents identity (INV-11).
- Whole-session operations cover every entry. `--track` selects only genuinely attributed
  entries, so whole-session restore remains the only way to recover unassigned files —
  and the operator contract says so rather than leaving it to be discovered.
- **Enumeration uses `lstat` and refuses a symlink at every path component**, not only at
  the leaf, then proves each resolved path stays inside a resolved configured root. The
  existing traversal is deliberately not reused.
- The set is hashed once before work and re-verified on **every** exit path, including
  each failure path.
- No archive output, staged file, lock, or report may resolve inside a source root, checked
  through `reject_outputs_inside_raw` before the first write.

**Paths are encoded over filesystem bytes, not decoded text.** `os.fsencode()` first,
then percent-encode every byte outside `[A-Za-z0-9._-]` with uppercase hex. The
manifest's authoritative `path` field is that pure-ASCII form. A human-readable
`path_text` appears beside it *only* when the name is valid UTF-8, and restore never
reads it.

That last point is not hypothetical fussiness. Linux permits a filename that is not
valid UTF-8; Python surfaces one through surrogate escapes; and `canonical_json` emits
UTF-8 text and would raise on a surrogate. So a manifest keyed on decoded text cannot
represent a file this milestone promises to archive — the failure would appear as a
crash during upload of a session containing one oddly-named file, at the moment the
archive was most needed.

Keys are bounded in **UTF-8 bytes** against the 1024-byte object-key limit. Encoding can
triple a path's length, so an entry whose key would exceed the bound is refused with a
diagnostic, never silently truncated into a collision.

## Alternatives considered

**Reuse `raw_guard.snapshot()` and add a symlink check afterwards.** Rejected: the
check has to happen *during* traversal, because `rglob` does not descend into a symlinked
directory and its contents are therefore never examined at all — the same shape M6b found
in `verify_tree`, where an unpinned symlinked directory passed a check whose whole claim
was that the tree matched the manifest.

**Archive only what `manifest.json` lists.** Rejected: it silently drops every
irreplaceable non-audio file, and "irreplaceable" is the whole selection criterion.

**Store the decoded path and normalize odd names.** Rejected — normalizing a name means
restoring a file under a different name than it had, which is not the byte-exact
restoration this milestone promises.

**A global content-addressed store keyed on hash alone**, deduplicating across sessions.
Rejected as an explicit non-goal: it makes a human unable to recognize what is in the
bucket, and cross-session deduplication is a saving nobody has asked for against a
recovery story nobody could follow.

## Consequences

Archiving is decoupled from inspection's opinions about what audio matters, which is
correct for a backup and would be wrong for a pipeline stage.

The cost is a third traversal to keep correct, and three enumerations that must not
silently diverge. The test that guards it compares `ArchiveSourceSet` against
`raw_guard.snapshot()` **exactly** on a tree with no symlinks in it, so a change to
either that drops files is loud. On a tree *with* a symlink they legitimately differ,
and that difference is fatal on the archive side rather than reconciled.

`--track` recovering less than the whole session is a real ergonomic sharp edge. It is
the honest one: attributing an unassigned file to a track to make `--track` feel complete
is precisely the INV-11 violation this project has refused since M1.
