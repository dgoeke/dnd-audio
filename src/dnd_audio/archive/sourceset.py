"""What gets archived: a third inventory, because neither existing one will do.

`manifest.json` inventories *candidate audio* — selected, ignored `edit`, duplicate,
unassigned — and would silently drop the `notes.txt` sitting beside the recordings,
because nothing in the pipeline reads one. `raw_guard.snapshot()` inventories every
regular file, which is much closer, but it tests `path.is_file()`, and that **follows a
leaf symlink**. Comparing a tree against itself that is correct: a link whose target moved
shows up as a changed hash. For something that reads and uploads, it means a link named
`raw/tx-a/notes.txt` pointing at a private key would be compressed and sent to a bucket
under an innocent-looking key.

So enumeration here refuses a symlink **at every path component**, not only at the leaf,
and then proves each resolved file is still inside a resolved configured root. The
component-wise part is not belt-and-braces: `rglob` does not descend into a symlinked
directory, so its contents are never examined at all — the same shape M6b found in
`verify_tree`, where an unpinned symlinked directory passed a check whose entire claim was
that a tree matched its manifest.

Track identity is **optional** (INV-11, ADR-0036). A file belongs to a track only when its
path is unambiguously inside one configured track input. Nested notes, unassigned audio and
stray files stay unassigned rather than being attributed to whoever was nearest, which is
why `--track` recovers less than a whole session and why the operator contract says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dnd_audio.archive import (
    ARCHIVE_OBJECTS_DIRNAME,
    ARCHIVE_PREFIX,
    ArchiveError,
)
from dnd_audio.archive.paths import encode_component, require_key_within_limit
from dnd_audio.config import SessionConfig
from dnd_audio.determinism import sha256_file
from dnd_audio.inspection import OUTPUT_DIRNAME, WORK_DIRNAME
from dnd_audio.raw_guard import raw_roots

__all__ = [
    "ArchiveEntry",
    "ArchiveSourceSet",
    "build_source_set",
    "object_key",
]


@dataclass(frozen=True, slots=True, order=True)
class ArchiveEntry:
    """One irreplaceable file, and everything the archive needs to know about it.

    Ordered by `relative_path` first, because every operation walks the set in
    deterministic path order and a set that sorted by size would make two runs of the same
    upload interleave differently.
    """

    #: Session-relative, POSIX-separated. The identity of this entry, and what restore
    #: recreates. May contain surrogate escapes if the filename is not valid UTF-8.
    relative_path: str
    size_bytes: int
    sha256: str
    #: Present only when the path is unambiguously inside one configured track's input
    #: directory. Never inferred from a filename (INV-11).
    track_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSourceSet:
    """Every regular file protected by a session's INV-01 snapshot, hardened.

    Immutable by construction, and re-derived rather than mutated: :meth:`verify_unchanged`
    walks the tree again and compares, which is the only honest way to check something the
    operating system owns.
    """

    session_dir: Path
    roots: tuple[str, ...]
    entries: tuple[ArchiveEntry, ...]

    @property
    def total_bytes(self) -> int:
        """What a whole-session restore will write. Used to preflight, exactly."""
        return sum(entry.size_bytes for entry in self.entries)

    def for_track(self, track_id: str) -> tuple[ArchiveEntry, ...]:
        """Only the entries genuinely attributed to ``track_id``.

        Never falls back to "everything near that directory". An empty result for a
        configured track means that track's directory held no regular files, which is worth
        the caller saying out loud rather than papering over.
        """
        return tuple(entry for entry in self.entries if entry.track_id == track_id)

    def verify_unchanged(self, config: SessionConfig) -> None:
        """Re-walk and compare. INV-01, on every exit path including failures.

        Raises:
            ArchiveError: naming what moved. A file that *appeared* counts as much as one
                that changed: an archive is a claim about a set of files, and a set that
                grew mid-upload was not the set that was hashed.
        """
        current = build_source_set(self.session_dir, config)
        if current.entries == self.entries:
            return

        before = {entry.relative_path: entry for entry in self.entries}
        after = {entry.relative_path: entry for entry in current.entries}
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
        details = []
        if changed:
            details.append(f"modified: {', '.join(changed)}")
        if removed:
            details.append(f"removed: {', '.join(removed)}")
        if added:
            details.append(f"appeared: {', '.join(added)}")
        message = (
            "the session's source files changed while the archive was running, which no "
            "part of this pipeline is permitted to do (INV-01). Nothing was committed. "
            + "; ".join(details)
        )
        raise ArchiveError(message, code="archive_sources_modified")


def object_key(session_id: str, entry: ArchiveEntry) -> str:
    """The immutable object key for one entry.

    ``<prefix>/<encoded-session>/objects/<encoded-path>.<sha256>.zst`` — content-addressed
    on the **original** digest, so re-uploading identical bytes lands on the same key and a
    changed source lands on a different one rather than overwriting. Still readable enough
    that a human scrolling a bucket can tell which session and which file they are looking
    at (ADR-0037).

    Raises:
        ArchiveError: if the encoded key exceeds the provider's limit.
    """
    key = (
        f"{ARCHIVE_PREFIX}/{encode_component(session_id)}/{ARCHIVE_OBJECTS_DIRNAME}/"
        f"{encode_component(entry.relative_path)}.{entry.sha256}.zst"
    )
    return require_key_within_limit(key, subject=entry.relative_path)


def build_source_set(session_dir: Path, config: SessionConfig) -> ArchiveSourceSet:
    """Enumerate and hash every regular file the archive is responsible for.

    Raises:
        ArchiveError: on a symlink at any component, or a file that resolves outside its
            configured root. Both are refusals rather than skips — quietly omitting a file
            from a backup is the failure this milestone exists to prevent, and quietly
            *including* one from outside the session is how a private key gets uploaded.
    """
    roots = raw_roots(config)
    track_inputs = {
        PurePosixPath(track.input).as_posix(): track.track_id for track in config.tracks
    }
    generated = {WORK_DIRNAME, OUTPUT_DIRNAME}
    found: list[ArchiveEntry] = []

    for root in roots:
        directory = session_dir if root == "." else session_dir / root
        if not directory.exists():
            continue
        _refuse_symlink(directory, session_dir)
        if not directory.is_dir():
            continue
        _walk(
            directory,
            session_dir=session_dir,
            root_resolved=directory.resolve(),
            generated=generated,
            track_inputs=track_inputs,
            found=found,
        )

    # De-duplicated by path: two configured roots can nest (`.` and `raw`), and the same
    # file discovered twice would be uploaded twice and counted twice in the preflight.
    unique = {entry.relative_path: entry for entry in found}
    return ArchiveSourceSet(
        session_dir=session_dir,
        roots=roots,
        entries=tuple(sorted(unique.values())),
    )


def _walk(
    directory: Path,
    *,
    session_dir: Path,
    root_resolved: Path,
    generated: set[str],
    track_inputs: dict[str, str],
    found: list[ArchiveEntry],
) -> None:
    """Depth-first with `scandir`, refusing every symlink as it is encountered.

    Refusing during the walk rather than filtering afterwards is what makes the guarantee
    hold for *directories*: a filter over `rglob` never sees inside a symlinked directory
    at all, so its contents are neither archived nor reported — silently absent from a
    backup, which is worse than an error.
    """
    for entry in sorted(os.scandir(directory), key=lambda item: item.name):
        path = Path(entry.path)
        relative = path.relative_to(session_dir).as_posix()

        if entry.is_symlink():
            message = (
                f"{relative} is a symlink. The archive refuses one at every path "
                f"component rather than following it: a link inside a source directory "
                f"can name any file on this machine, and uploading it under a "
                f"session-relative key would put it in the bucket looking like session "
                f"data. Replace it with the file itself, or move it out of the session."
            )
            raise ArchiveError(message, code="archive_symlink_refused")

        # The session's own generated directories, excluded at the session root **only**.
        # Matching the names at any depth is the defect M2 found in INV-01's snapshot:
        # it silently drops `raw/tx-a/work/notes.txt`, which is exactly the irreplaceable
        # nested file this milestone exists to protect.
        if PurePosixPath(relative).parts[0] in generated:
            continue

        if entry.is_dir(follow_symlinks=False):
            _walk(
                path,
                session_dir=session_dir,
                root_resolved=root_resolved,
                generated=generated,
                track_inputs=track_inputs,
                found=found,
            )
            continue

        if not entry.is_file(follow_symlinks=False):
            # A fifo, socket, or device node. Not archivable, and not something to fail a
            # whole session over either — but it must be *said*, because "the archive holds
            # everything" would otherwise be false without anyone knowing.
            message = (
                f"{relative} is not a regular file. The archive stores file contents, and "
                f"a fifo, socket, or device node has none to store. Remove it from the "
                f"session directory."
            )
            raise ArchiveError(message, code="archive_irregular_file")

        _require_inside(path, root_resolved, relative)
        found.append(
            ArchiveEntry(
                relative_path=relative,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                track_id=_attribute(relative, track_inputs),
            )
        )


def _refuse_symlink(path: Path, session_dir: Path) -> None:
    """A configured root that is itself a symlink is refused before it is walked."""
    if path.is_symlink():
        try:
            shown = path.relative_to(session_dir).as_posix()
        except ValueError:  # pragma: no cover - a root is always under the session
            shown = str(path)
        message = (
            f"the configured source root {shown} is a symlink. The archive refuses to "
            f"follow one: what it points at decides what gets uploaded, and that is not "
            f"visible in the session."
        )
        raise ArchiveError(message, code="archive_symlink_refused")


def _require_inside(path: Path, root_resolved: Path, relative: str) -> None:
    """Prove the resolved file is still under the resolved root.

    Belt to the symlink refusal's braces, and not redundant: a bind mount, a hard link
    across trees, or a filesystem this code has not met can put a non-symlink somewhere
    surprising. INV-01 already learned that lexical path comparison is not a boundary
    (M1's verify phase); this is the same lesson applied in the other direction.
    """
    resolved = path.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        message = (
            f"{relative} resolves to {resolved}, outside the configured source root "
            f"{root_resolved}. Nothing outside a session's own sources is archived."
        )
        raise ArchiveError(message, code="archive_path_escapes_root")


def _attribute(relative: str, track_inputs: dict[str, str]) -> str | None:
    """The track this path belongs to, or ``None``. Never a guess (INV-11).

    Attribution requires the path to sit inside exactly one configured track input. A
    session whose configuration nests one track's directory inside another's would make a
    file ambiguous, and an ambiguous file is unassigned rather than assigned to the
    deeper one — `--track` under-recovering is recoverable, and a track restore quietly
    containing another speaker's audio is not.
    """
    parts = PurePosixPath(relative).parts
    matches = {
        track_id
        for input_path, track_id in track_inputs.items()
        if parts[: len(PurePosixPath(input_path).parts)] == PurePosixPath(input_path).parts
    }
    return matches.pop() if len(matches) == 1 else None
