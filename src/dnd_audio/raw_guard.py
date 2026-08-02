"""INV-01, in one place: nothing under a session's sources is ever written.

M1 built this inside `inspection/runner.py`, where it was the only stage that wrote
anything. M2 writes a timeline, derivatives, and optionally materialized audio, and M2's
charter says to extend the "output inside raw" check rather than duplicate it — so it
lives here, and each stage passes the set of paths it intends to produce.

The invariant is verified, not asserted. Two mechanisms, and the order they run in is
load-bearing:

1. :func:`reject_outputs_inside_raw` runs **before the first write**, which is the only
   order in which it is a check rather than a postmortem.
2. :func:`snapshot` before and :func:`verify_unchanged` after, over **every** file under
   the source roots — not only the ones the stage read. "We did not touch what we read" is
   a weaker claim than the invariant makes, and it would miss exactly the accidental
   rename this exists to catch.

Two things M1's verify phase found the hard way, preserved here with the reasons:

**Paths are compared after resolution.** A lexical comparison is defeated by one symlink:
with ``output -> raw/tx-a``, ``output/ingest-report.json`` does not *look* like it is
inside ``raw/``, and a run cheerfully writes into a track's source directory. The snapshot
cannot catch it either, because outputs are written after the snapshot is verified.

**``"."`` stays in the roots.** Dropping it looks reasonable — every relative path is under
``"."`` — and it empties the snapshot for a session configured as ``input: "tx-a"``, so
:func:`verify_unchanged` compares two empty dicts and passes no matter what happened. The
false-positive problem belongs to the output check alone and is handled there.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from dnd_audio.config import SessionConfig
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import DiscoveryError
from dnd_audio.inspection import OUTPUT_DIRNAME, WORK_DIRNAME

__all__ = [
    "RawSnapshot",
    "raw_roots",
    "reject_outputs_inside_raw",
    "snapshot",
    "verify_unchanged",
]

#: ``{session-relative path: (sha256, size)}``.
RawSnapshot = dict[str, tuple[str, int]]


def raw_roots(config: SessionConfig) -> tuple[str, ...]:
    """The directories a session's sources live under, as session-relative paths.

    Derived from the configured inputs rather than hardcoded to ``raw/``: the spec's
    layout is canonical, not mandatory, and INV-01 protects wherever the sources actually
    are.
    """
    roots = {str(PurePosixPath(track.input).parent) or "." for track in config.tracks}
    return tuple(sorted("." if root == "" else root for root in roots))


def snapshot(session_dir: Path, roots: tuple[str, ...]) -> RawSnapshot:
    """Hash and size every file under the source roots.

    The session's own ``work/`` and ``output/`` are excluded: when a track's input sits
    directly in the session root they are inside a scanned root, and they are the two
    directories a run is *supposed* to write.

    **Excluded at the session root only.** Matching those names at any depth — which is
    what an earlier version did — silently drops every file beneath a source directory that
    happens to contain one, so ``raw/tx-a/work/notes.txt`` was never hashed and mutating it
    passed verification unconditionally. Same shape as the defect M1's closeout describes:
    a check that is present, looks right, and verifies nothing.
    """
    generated = {WORK_DIRNAME, OUTPUT_DIRNAME}
    found: RawSnapshot = {}
    for root in roots:
        directory = session_dir if root == "." else session_dir / root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(session_dir).as_posix()
            if PurePosixPath(relative).parts[0] in generated:
                continue
            found[relative] = (sha256_file(path), path.stat().st_size)
    return found


def verify_unchanged(session_dir: Path, roots: tuple[str, ...], before: RawSnapshot) -> None:
    """INV-01, verified rather than asserted.

    Raises:
        DiscoveryError: with code ``raw_sources_modified``, naming what moved. A file that
            appeared counts as much as one that vanished: a stage writing a normalized
            copy beside a source is exactly what this forbids.
    """
    after = snapshot(session_dir, roots)
    if after == before:
        return

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
        "the session's raw sources changed during this run, which no stage of this "
        "pipeline is permitted to do (INV-01). " + "; ".join(details)
    )
    raise DiscoveryError(message, code="raw_sources_modified")


def reject_outputs_inside_raw(
    session_dir: Path,
    config: SessionConfig,
    roots: tuple[str, ...],
    outputs: Mapping[str, Path],
) -> None:
    """The spec's "output paths would overwrite raw inputs" fatal error.

    Args:
        outputs: ``{human-readable label: path}`` this stage intends to write. Each stage
            passes its own set; a stage that adds an output and forgets to declare it here
            is the failure mode this signature exists to make visible.

    Protected: each configured track's input directory, always; and each scan root, except
    when the root *is* the session directory, where ``work/`` and ``output/`` are
    legitimately siblings of the track directories.

    Raises:
        DiscoveryError: with code ``output_inside_raw``.
    """
    protected: dict[str, Path] = {
        track.input: (session_dir / track.input).resolve() for track in config.tracks
    }
    for root in roots:
        if root != ".":
            protected[root] = (session_dir / root).resolve()

    for label, target in sorted(outputs.items()):
        # strict=False: most of these have not been created yet. What matters is where
        # they *would* land, which the symlinks already on the way there decide.
        resolved = target.resolve()
        for name, directory in sorted(protected.items()):
            if resolved == directory or directory in resolved.parents:
                message = (
                    f"{label} would be written to {resolved}, inside the source directory "
                    f"{name} ({directory}). Nothing under a session's raw sources may be "
                    f"written to (INV-01). If a symlink put it there, that counts."
                )
                raise DiscoveryError(message, code="output_inside_raw")
