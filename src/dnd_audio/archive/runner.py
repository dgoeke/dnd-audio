"""The five archive operations.

The ordering inside :func:`run_upload` is the milestone, so it is stated rather than left
to be inferred (ADR-0038):

1. **Require a current inspection manifest.** Not because the archive needs its contents —
   the source set is independent (ADR-0036) — but because a session nobody has inspected is
   a session nobody has looked at, and the archive should not be the first thing to read it.
2. **Build and hash the hardened source set**, before any staging exists.
3. **Refuse output paths inside a source root**, resolved, before the first write (INV-01).
4. **Take the single-writer lock.** Non-blocking: a second upload is refused, not queued.
5. **Preflight worst-case disk** from the compression bound, never from a ratio.
6. **Read any existing manifest in full.** Byte-equal is idempotent success and returns;
   different is fatal divergence. `HEAD` then `PUT` is not a compare-and-swap and is not
   presented as one.
7. **Per entry, in path order:** compress to one staged file, decompress it locally back to
   the original digest, upload, then **download the whole object again** and decompress it
   to the original digest a second time. Then delete the staged file and move on.
8. **Re-verify the source set** — on success and on every failure path.
9. **Only then, PUT the manifest.** An interrupted upload leaves objects with no manifest,
   which `status` calls `pending` rather than an archive.

`verify` and `restore` deliberately take a session id rather than a directory: they exist
for the case where the local session is gone, and a recovery path that needed the thing
that was lost would be theatre.

**Restore is transactional.** The plan review found that publishing each file atomically
and then refusing existing targets strands its own retry — an ENOSPC at file 20 leaves 1–19
behind, and the next attempt must refuse them. So the whole tree is staged beside the
destination, verified complete, and moved into place at the end.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from dnd_audio.archive import (
    ARCHIVE_MANIFEST_FILENAME,
    ARCHIVE_PREFIX,
    ArchiveError,
)
from dnd_audio.archive.codec import (
    ARCHIVE_CODEC_V1,
    compress_bound,
    compress_file,
    decompress_and_measure,
)
from dnd_audio.archive.lock import single_writer
from dnd_audio.archive.manifest import ArchiveManifest, ArchiveManifestEntry
from dnd_audio.archive.paths import decode_component, encode_component
from dnd_audio.archive.report import (
    ArchiveObjectOutcome,
    ArchiveOperation,
    ArchiveReport,
    ArchiveReportError,
    ArchiveScope,
    ObjectResult,
    OperationStatus,
    VerificationState,
)
from dnd_audio.archive.sourceset import ArchiveEntry, ArchiveSourceSet, build_source_set, object_key
from dnd_audio.archive.storage import ArchiveStorage
from dnd_audio.config import SessionConfig, config_hash, load_session_config
from dnd_audio.determinism import canonical_json, sha256_bytes, sha256_file
from dnd_audio.inspection import WORK_DIRNAME
from dnd_audio.inspection.runner import MANIFEST_RELATIVE_PATH
from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw

__all__ = [
    "STAGING_DIRNAME",
    "manifest_key",
    "run_list",
    "run_restore",
    "run_status",
    "run_upload",
    "run_verify",
]

#: Where a compressed object is staged, one at a time. Under `work/`, so it is inside the
#: session's own generated area and outside every source root — and so it is on the same
#: filesystem the preflight measured.
STAGING_DIRNAME: Final = f"{WORK_DIRNAME}/archive"

#: Slack over the worst-case compressed size, for the report and the manifest that share
#: the disk. Generous because the cost of being wrong is a failed upload mid-session.
_PREFLIGHT_SLACK_BYTES: Final = 64 << 20

#: The name a restore stages under, beside its destination. Distinctive so that a leftover
#: one after a hard kill is recognizable rather than mysterious.
_RESTORE_STAGING_PREFIX: Final = ".dnd-audio-restore-"


def manifest_key(session_id: str) -> str:
    """Where a session's commit marker lives."""
    return f"{ARCHIVE_PREFIX}/{encode_component(session_id)}/{ARCHIVE_MANIFEST_FILENAME}"


def _session_prefix(session_id: str) -> str:
    return f"{ARCHIVE_PREFIX}/{encode_component(session_id)}/"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True, slots=True)
class _Staged:
    """One compressed object waiting to be uploaded."""

    entry: ArchiveEntry
    path: Path
    compressed_size: int
    compressed_sha256: str


# --- upload ---------------------------------------------------------------------------


def run_upload(
    session_dir: Path,
    *,
    storage: ArchiveStorage,
    lock_dir: Path | None = None,
    free_bytes: int | None = None,
) -> ArchiveReport:
    """Archive every file in the hardened source set, then commit.

    Args:
        free_bytes: Override the measured free space. For preflight tests, which need to
            drive the arithmetic rather than fill a disk.

    Returns:
        The operation report. Never raises for an ordinary failure — INV-13 wants a written
        report and a nonzero exit, not an exception.
    """
    started = _now()
    config = load_session_config(session_dir / "session.yaml")
    session_id = config.session_id

    try:
        _require_current_manifest(session_dir, config)
        sources = build_source_set(session_dir, config)
        staging_root = session_dir / STAGING_DIRNAME
        reject_outputs_inside_raw(
            session_dir,
            config,
            raw_roots(config),
            {
                "archive staging": staging_root,
                "archive report": _report_path(session_dir, "upload"),
            },
        )
    except Exception as exc:
        return _failed(ArchiveOperation.UPLOAD, session_id, exc, started, entries_in_scope=0)

    with single_writer(session_id, directory=lock_dir):
        return _upload_locked(
            session_dir=session_dir,
            config=config,
            sources=sources,
            storage=storage,
            staging_root=staging_root,
            started=started,
            free_bytes=free_bytes,
        )


def _upload_locked(
    *,
    session_dir: Path,
    config: SessionConfig,
    sources: ArchiveSourceSet,
    storage: ArchiveStorage,
    staging_root: Path,
    started: dt.datetime,
    free_bytes: int | None,
) -> ArchiveReport:
    session_id = config.session_id
    results: list[ObjectResult] = []
    manifest_entries: list[ArchiveManifestEntry] = []

    try:
        _preflight_disk(sources, session_dir, free_bytes=free_bytes)

        existing = _read_remote_manifest(storage, session_id)
        if existing is not None:
            return _resolve_existing_manifest(
                existing=existing, sources=sources, session_id=session_id, started=started
            )

        shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        for entry in sources.entries:
            staged = _compress_and_check(session_dir, entry, staging_root)
            try:
                outcome = _publish(storage, session_id, entry, staged)
                manifest_entries.append(_manifest_entry(session_id, entry, staged))
                results.append(
                    ObjectResult(
                        path=encode_component(entry.relative_path),
                        outcome=outcome,
                        size_bytes=entry.size_bytes,
                        compressed_size_bytes=staged.compressed_size,
                    )
                )
            finally:
                staged.path.unlink(missing_ok=True)

        # INV-01, before anything is committed. A source that moved during the upload means
        # the objects just written describe a set that no longer exists, and publishing a
        # manifest for it would make the archive confidently wrong.
        sources.verify_unchanged(config)

        planned = _plan_manifest(session_id, sources, manifest_entries)
        payload = canonical_json(planned.model_dump(mode="json")).encode("utf-8")
        _put_bytes(storage, manifest_key(session_id), payload, staging_root)

    except Exception as exc:
        _safe_verify(sources, config, results)
        shutil.rmtree(staging_root, ignore_errors=True)
        return ArchiveReport(
            operation=ArchiveOperation.UPLOAD,
            status=OperationStatus.PARTIAL if results else OperationStatus.FAILED,
            scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
            verification=VerificationState.PENDING if results else VerificationState.ABSENT,
            objects=results,
            errors=[_as_error(exc)],
            notes=[
                "No manifest was published, so this session is not committed. Re-running "
                "`archive upload` resumes: objects already present and verified are not "
                "uploaded again."
            ],
            started_at=started,
            finished_at=_now(),
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return ArchiveReport(
        operation=ArchiveOperation.UPLOAD,
        status=OperationStatus.COMPLETE,
        scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
        # `upload` may say this because every object was downloaded again and decompressed
        # before the manifest went up — a current full readback, not provider metadata.
        verification=VerificationState.VERIFIED,
        manifest_sha256=sha256_bytes(payload),
        objects=results,
        started_at=started,
        finished_at=_now(),
    )


def _compress_and_check(session_dir: Path, entry: ArchiveEntry, staging_root: Path) -> _Staged:
    """Compress one file and prove locally that it restores, before any of it is uploaded.

    Two verifications happen per object and both are needed. This one catches a bad
    compression or a source that changed under us, and costs no network; the remote readback
    catches a bad transfer or a provider that stored something else. Neither subsumes the
    other.
    """
    source = session_dir / entry.relative_path
    staged = staging_root / f"{entry.sha256}.zst"

    measured = sha256_file(source)
    if measured != entry.sha256:
        message = (
            f"{entry.relative_path} changed between being inventoried and being read: "
            f"expected {entry.sha256}, found {measured}. Nothing was uploaded (INV-01)."
        )
        raise ArchiveError(message, code="archive_sources_modified")

    fact = compress_file(source, staged)
    with staged.open("rb") as handle:
        restored = decompress_and_measure(handle, max_output_bytes=entry.size_bytes)
    if restored.size_bytes != entry.size_bytes or restored.sha256 != entry.sha256:
        staged.unlink(missing_ok=True)
        message = (
            f"the compressed form of {entry.relative_path} does not decompress back to it: "
            f"got {restored.size_bytes} bytes / {restored.sha256}, expected "
            f"{entry.size_bytes} / {entry.sha256}. Nothing was uploaded."
        )
        raise ArchiveError(message, code="archive_local_roundtrip_failed")

    return _Staged(
        entry=entry,
        path=staged,
        compressed_size=fact.size_bytes,
        compressed_sha256=fact.sha256,
    )


def _publish(
    storage: ArchiveStorage, session_id: str, entry: ArchiveEntry, staged: _Staged
) -> ArchiveObjectOutcome:
    """Upload one object and read it back completely, or accept an identical existing one."""
    key = object_key(session_id, entry)

    head = storage.head_object(key)
    if head is not None:
        # An existing object at a content-addressed key should hold identical bytes. That is
        # checked by downloading it, not by comparing its size or its ETag — a multipart
        # ETag is a hash of part hashes and identifies nothing about the content (ADR-0038).
        _read_back(storage, key, entry, expected_compressed=staged.compressed_sha256)
        return ArchiveObjectOutcome.ALREADY_PRESENT

    storage.put_object(key, staged.path)
    _read_back(storage, key, entry, expected_compressed=staged.compressed_sha256)
    return ArchiveObjectOutcome.UPLOADED


def _read_back(
    storage: ArchiveStorage,
    key: str,
    entry: ArchiveEntry,
    *,
    expected_compressed: str | None,
) -> None:
    """Download the whole object and decompress it, discarding the bytes.

    The step that makes this a backup rather than a belief. It costs a full extra transfer
    per object and that is the price of the guarantee.
    """
    import hashlib

    digest = hashlib.sha256()

    class _Tee:
        """Hashes the compressed stream while the decompressor consumes it.

        Both digests come from one download. Reading the object twice — once to hash the
        compressed form, once to decompress — would double an already expensive step and
        could compare two *different* downloads without noticing.
        """

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def read(self, size: int = -1, /) -> bytes:
            chunk: bytes = self._inner.read(size)  # type: ignore[attr-defined]
            digest.update(chunk)
            return chunk

    with storage.open_object(key) as body:
        restored = decompress_and_measure(_Tee(body), max_output_bytes=entry.size_bytes)

    if restored.size_bytes != entry.size_bytes or restored.sha256 != entry.sha256:
        message = (
            f"the object stored for {entry.relative_path} does not restore to it: got "
            f"{restored.size_bytes} bytes / {restored.sha256}, expected {entry.size_bytes} "
            f"/ {entry.sha256}."
        )
        raise ArchiveError(message, code="archive_readback_mismatch")

    if expected_compressed is not None and digest.hexdigest() != expected_compressed:
        message = (
            f"the object stored for {entry.relative_path} is not the object that was "
            f"uploaded: its compressed bytes hash to {digest.hexdigest()}, not "
            f"{expected_compressed}. It decompresses correctly, which means the difference "
            f"is in the stored frame rather than in the audio — a conflicting write at a "
            f"content-addressed key. Refusing rather than overwriting."
        )
        raise ArchiveError(message, code="archive_object_conflict")


def _resolve_existing_manifest(
    *,
    existing: ArchiveManifest,
    sources: ArchiveSourceSet,
    session_id: str,
    started: dt.datetime,
) -> ArchiveReport:
    """Byte-equality is idempotent success; any difference is fatal (ADR-0038)."""
    planned = _plan_manifest(session_id, sources, _entries_from_sources(session_id, sources))
    same = {(e.path, e.sha256, e.size_bytes) for e in existing.entries} == {
        (e.path, e.sha256, e.size_bytes) for e in planned.entries
    }
    payload = canonical_json(existing.model_dump(mode="json")).encode("utf-8")

    if same:
        return ArchiveReport(
            operation=ArchiveOperation.UPLOAD,
            status=OperationStatus.COMPLETE,
            scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
            # Committed, and not re-verified: this run uploaded nothing and downloaded no
            # object, so claiming `verified` would be exactly the overstatement ADR-0039
            # exists to prevent. `archive verify` is what establishes that.
            verification=VerificationState.COMMITTED,
            manifest_sha256=sha256_bytes(payload),
            notes=[
                "This session is already archived with identical content; nothing was "
                "uploaded. Run `archive verify` to confirm the stored bytes still restore "
                "— that is a separate claim from this one, and it costs a full download."
            ],
            started_at=started,
            finished_at=_now(),
        )

    message = (
        f"session {session_id!r} is already archived, and its committed manifest describes "
        f"different content than this directory holds. Nothing was changed. An archive "
        f"version is immutable by design: if the local session is the correct one, it "
        f"belongs under a new session id; if the archive is, restore from it rather than "
        f"overwriting it."
    )
    return ArchiveReport(
        operation=ArchiveOperation.UPLOAD,
        status=OperationStatus.FAILED,
        scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
        verification=VerificationState.DIVERGENT,
        manifest_sha256=sha256_bytes(payload),
        errors=[ArchiveReportError(code="archive_manifest_divergent", message=message)],
        started_at=started,
        finished_at=_now(),
    )


# --- status, list, verify, restore -----------------------------------------------------


def run_status(session_dir: Path, *, storage: ArchiveStorage) -> ArchiveReport:
    """Compare a local session against the remote archive. Cheap, and never authoritative.

    This is the operation most able to mislead, so what it may say is constrained by the
    report model itself: it can report `absent`, `pending`, `committed`,
    `previously_verified_at_commit` or `divergent`, and it structurally **cannot** report
    `verified` (ADR-0039).
    """
    started = _now()
    config = load_session_config(session_dir / "session.yaml")
    session_id = config.session_id

    try:
        sources = build_source_set(session_dir, config)
    except ArchiveError as exc:
        return _failed(ArchiveOperation.STATUS, session_id, exc, started, entries_in_scope=0)

    remote = _read_remote_manifest(storage, session_id)
    if remote is None:
        objects_present = any(True for _ in _iter_prefix(storage, _session_prefix(session_id)))
        state = VerificationState.PENDING if objects_present else VerificationState.ABSENT
        note = (
            "Objects exist but no manifest does, so an upload was interrupted before it "
            "committed. Re-run `archive upload`."
            if objects_present
            else "This session has never been archived."
        )
        return ArchiveReport(
            operation=ArchiveOperation.STATUS,
            status=OperationStatus.COMPLETE,
            scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
            verification=state,
            notes=[note],
            started_at=started,
            finished_at=_now(),
        )

    planned = {(e.path, e.sha256, e.size_bytes) for e in _entries_from_sources(session_id, sources)}
    committed = {(e.path, e.sha256, e.size_bytes) for e in remote.entries}
    payload = canonical_json(remote.model_dump(mode="json")).encode("utf-8")
    matches = planned == committed

    return ArchiveReport(
        operation=ArchiveOperation.STATUS,
        status=OperationStatus.COMPLETE,
        scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
        verification=(VerificationState.COMMITTED if matches else VerificationState.DIVERGENT),
        manifest_sha256=sha256_bytes(payload),
        notes=[
            "A manifest exists and matches this directory's contents. That is not the same "
            "as the stored bytes still being readable — only `archive verify` establishes "
            "that, by downloading them."
            if matches
            else "The committed manifest describes different content than this directory "
            "holds. Neither was changed."
        ],
        started_at=started,
        finished_at=_now(),
    )


def run_list(*, storage: ArchiveStorage) -> tuple[list[str], ArchiveReport]:
    """Every committed session id, discovered without a local session directory.

    Follows pagination to exhaustion. A partial listing reported as complete would make a
    session look absent during exactly the emergency this command exists for.
    """
    started = _now()
    session_ids: list[str] = []
    for key in _iter_prefix(storage, f"{ARCHIVE_PREFIX}/"):
        if not key.endswith(f"/{ARCHIVE_MANIFEST_FILENAME}"):
            continue
        encoded = key[len(f"{ARCHIVE_PREFIX}/") :].split("/", 1)[0]
        try:
            session_ids.append(decode_component(encoded))
        except ArchiveError:
            # A key this project did not write. Reported rather than silently dropped: an
            # unrecognizable key under our own prefix is worth a human knowing about.
            session_ids.append(f"<unrecognized key: {encoded}>")

    report = ArchiveReport(
        operation=ArchiveOperation.LIST,
        status=OperationStatus.COMPLETE,
        scope=ArchiveScope(session_id="(all)", entries_in_scope=len(session_ids)),
        verification=VerificationState.COMMITTED if session_ids else VerificationState.ABSENT,
        notes=[
            f"{len(session_ids)} committed session(s). Each has a manifest; none has been "
            f"verified by this operation."
        ],
        started_at=started,
        finished_at=_now(),
    )
    return sorted(session_ids), report


def run_verify(
    session_id: str, *, storage: ArchiveStorage, track_id: str | None = None
) -> ArchiveReport:
    """Download every selected object and prove it restores. The authoritative check.

    Expensive on purpose, and the only operation permitted to produce the word `verified`.
    Needs no local session directory.
    """
    started = _now()
    try:
        remote = _require_remote_manifest(storage, session_id)
        selected = _select(remote, track_id)
    except ArchiveError as exc:
        return _failed(ArchiveOperation.VERIFY, session_id, exc, started, entries_in_scope=0)

    results: list[ObjectResult] = []
    errors: list[ArchiveReportError] = []
    for entry in selected:
        try:
            _read_back(
                storage,
                entry.object_key,
                _entry_for_readback(entry),
                expected_compressed=entry.compressed_sha256,
            )
        except ArchiveError as exc:
            results.append(
                ObjectResult(
                    path=entry.path,
                    outcome=ArchiveObjectOutcome.FAILED,
                    size_bytes=entry.size_bytes,
                    error=ArchiveReportError(code=exc.code, message=str(exc), path=entry.path),
                )
            )
            errors.append(ArchiveReportError(code=exc.code, message=str(exc), path=entry.path))
            continue
        results.append(
            ObjectResult(
                path=entry.path,
                outcome=ArchiveObjectOutcome.VERIFIED,
                size_bytes=entry.size_bytes,
                compressed_size_bytes=entry.compressed_size_bytes,
            )
        )

    verified = all(r.outcome is ArchiveObjectOutcome.VERIFIED for r in results)
    gibibytes = sum(e.compressed_size_bytes for e in selected) / (1 << 30)
    return ArchiveReport(
        operation=ArchiveOperation.VERIFY,
        status=OperationStatus.COMPLETE if verified else OperationStatus.PARTIAL,
        scope=ArchiveScope(
            session_id=session_id, track_id=track_id, entries_in_scope=len(selected)
        ),
        verification=(VerificationState.VERIFIED if verified else VerificationState.DIVERGENT),
        objects=results,
        errors=errors,
        notes=[_retrieval_cost_note(gibibytes), *_scope_note(track_id)],
        started_at=started,
        finished_at=_now(),
    )


def run_restore(
    session_id: str,
    destination: Path,
    *,
    storage: ArchiveStorage,
    track_id: str | None = None,
    protected_session_dirs: Sequence[Path] = (),
    free_bytes: int | None = None,
) -> ArchiveReport:
    """Rebuild a session's files under ``destination``, from the archive alone.

    Transactional: the whole tree is staged beside ``destination``, verified complete, and
    then moved in. A failed restore leaves ``destination`` exactly as it found it, so a
    retry is a retry rather than a manual cleanup (plan review, P1).
    """
    started = _now()
    try:
        remote = _require_remote_manifest(storage, session_id)
        selected = _select(remote, track_id)
        _check_destination(destination, protected_session_dirs)
        _preflight_restore(selected, destination, free_bytes=free_bytes)
    except ArchiveError as exc:
        return _failed(ArchiveOperation.RESTORE, session_id, exc, started, entries_in_scope=0)

    staging = destination.parent / f"{_RESTORE_STAGING_PREFIX}{encode_component(session_id)}"
    shutil.rmtree(staging, ignore_errors=True)
    results: list[ObjectResult] = []

    try:
        staging.mkdir(parents=True)
        for entry in selected:
            target = _resolve_restore_target(staging, entry.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as sink, storage.open_object(entry.object_key) as body:
                restored = decompress_and_measure(
                    body, max_output_bytes=entry.size_bytes, sink=sink
                )
            if restored.size_bytes != entry.size_bytes or restored.sha256 != entry.sha256:
                message = (
                    f"{entry.path} restored to {restored.size_bytes} bytes / "
                    f"{restored.sha256}, not {entry.size_bytes} / {entry.sha256}. Nothing "
                    f"was published to the destination."
                )
                raise ArchiveError(message, code="archive_restore_mismatch")
            results.append(
                ObjectResult(
                    path=entry.path,
                    outcome=ArchiveObjectOutcome.RESTORED,
                    size_bytes=entry.size_bytes,
                    compressed_size_bytes=entry.compressed_size_bytes,
                )
            )

        # The destination is empty (checked above), so removing it and renaming the staged
        # tree into its place is one atomic operation on the same filesystem. Either the
        # whole restore is there or none of it is.
        destination.rmdir()
        staging.rename(destination)

    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        return ArchiveReport(
            operation=ArchiveOperation.RESTORE,
            status=OperationStatus.FAILED,
            scope=ArchiveScope(
                session_id=session_id, track_id=track_id, entries_in_scope=len(selected)
            ),
            verification=VerificationState.COMMITTED,
            objects=results,
            errors=[_as_error(exc)],
            notes=[
                "The destination was left untouched, so re-running this command is a "
                "clean retry rather than a manual cleanup."
            ],
            started_at=started,
            finished_at=_now(),
        )

    return ArchiveReport(
        operation=ArchiveOperation.RESTORE,
        status=OperationStatus.COMPLETE,
        scope=ArchiveScope(
            session_id=session_id, track_id=track_id, entries_in_scope=len(selected)
        ),
        # Every restored byte was decompressed and hashed here, but `restore` may not use
        # the word: the report model reserves it for `verify` and `upload`, so an operator
        # reading a restore report is never told the *archive* was verified when what was
        # checked is the copy now on this disk.
        verification=VerificationState.COMMITTED,
        objects=results,
        notes=[
            "Every restored file was decompressed and matched its recorded digest.",
            *_scope_note(track_id),
        ],
        started_at=started,
        finished_at=_now(),
    )


# --- shared helpers --------------------------------------------------------------------


def _require_current_manifest(session_dir: Path, config: SessionConfig) -> None:
    """The session must have been inspected, and inspected as it stands now."""
    path = session_dir / MANIFEST_RELATIVE_PATH
    try:
        document = json.loads(path.read_bytes())
    except OSError as exc:
        message = (
            f"{MANIFEST_RELATIVE_PATH} is missing, so this session has never been "
            f"inspected. Run `dnd-audio inspect` first: the archive should not be the "
            f"first thing that reads a session."
        )
        raise ArchiveError(message, code="archive_manifest_absent") from exc
    except json.JSONDecodeError as exc:
        message = f"{MANIFEST_RELATIVE_PATH} is not readable JSON: {exc}"
        raise ArchiveError(message, code="archive_manifest_absent") from exc

    if document.get("config_hash") != config_hash(config):
        message = (
            f"{MANIFEST_RELATIVE_PATH} was written under a different configuration than "
            f"`session.yaml` now describes, so it does not describe what is on disk. Run "
            f"`dnd-audio inspect` again before archiving."
        )
        raise ArchiveError(message, code="archive_manifest_stale")


def _preflight_disk(
    sources: ArchiveSourceSet, session_dir: Path, *, free_bytes: int | None
) -> None:
    """Refuse an upload that cannot stage its largest object.

    One object is staged at a time, so the requirement is the worst-case *single* bound and
    not the sum — but it is the worst case, from `compress_bound`, never from the measured
    30.4%. Already-compressed sources come out slightly larger than they went in.
    """
    largest = max((entry.size_bytes for entry in sources.entries), default=0)
    needed = compress_bound(largest) + _PREFLIGHT_SLACK_BYTES
    available = shutil.disk_usage(session_dir).free if free_bytes is None else free_bytes
    if available < needed:
        message = (
            f"archiving this session needs {needed / (1 << 20):.1f} MiB free to stage its "
            f"largest file ({largest / (1 << 20):.1f} MiB), and {available / (1 << 20):.1f} "
            f"MiB is available. The requirement is the worst-case compressed size, not the "
            f"expected one: already-compressed data grows slightly."
        )
        raise ArchiveError(message, code="archive_insufficient_space")


def _preflight_restore(
    entries: Sequence[ArchiveManifestEntry], destination: Path, *, free_bytes: int | None
) -> None:
    """Refuse a restore that cannot hold the whole tree.

    The **sum**, not one file at a time: restore stages everything before publishing, so
    running out at file 20 means 19 files of wasted transfer and a failed operation.
    """
    needed = sum(entry.size_bytes for entry in entries) + _PREFLIGHT_SLACK_BYTES
    available = shutil.disk_usage(destination.parent).free if free_bytes is None else free_bytes
    if available < needed:
        message = (
            f"restoring this scope needs {needed / (1 << 20):.1f} MiB and "
            f"{available / (1 << 20):.1f} MiB is free where the destination sits. The whole "
            f"tree is staged before anything is published, so the requirement is the total "
            f"rather than the largest file."
        )
        raise ArchiveError(message, code="archive_insufficient_space")


def _check_destination(destination: Path, protected: Sequence[Path]) -> None:
    """An empty directory, outside every protected source root, reached without a symlink."""
    resolved = destination.resolve()
    for session_dir in protected:
        try:
            config = load_session_config(session_dir / "session.yaml")
        except Exception:
            continue
        for root in raw_roots(config):
            root_path = (session_dir if root == "." else session_dir / root).resolve()
            if resolved == root_path or root_path in resolved.parents:
                message = (
                    f"the restore destination resolves inside {root_path}, which is a "
                    f"session's source directory. Nothing may be written under one "
                    f"(INV-01) — restore somewhere else and compare by hand."
                )
                raise ArchiveError(message, code="archive_destination_protected")

    if destination.is_symlink():
        message = (
            "the restore destination is a symlink. Where it points decides what gets "
            "written, and that is not visible from the command line."
        )
        raise ArchiveError(message, code="archive_symlink_refused")
    if not destination.is_dir():
        message = f"the restore destination {destination} does not exist or is not a directory"
        raise ArchiveError(message, code="archive_destination_unusable")
    if any(destination.iterdir()):
        message = (
            f"the restore destination {destination} is not empty. A restore recreates a "
            f"whole tree and will not merge into existing files — point it at an empty "
            f"directory so what arrives is unambiguous."
        )
        raise ArchiveError(message, code="archive_destination_not_empty")


def _resolve_restore_target(root: Path, encoded_path: str) -> Path:
    """Decode one manifest path into a file under ``root``, refusing anything that escapes.

    The manifest is data, and on a recovery it is data that has been sitting in a bucket.
    So its paths are re-validated here rather than trusted, independently of the upload-side
    checks — the two run on different machines at different times, and a restore that
    depended on the uploader having been careful would be trusting the wrong process.
    """
    relative = decode_component(encoded_path)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        message = (
            f"the manifest names {relative!r}, which is not a relative path inside a "
            f"session. Refusing to write outside the restore destination."
        )
        raise ArchiveError(message, code="archive_path_escapes_root")

    target = root / candidate
    resolved_root = root.resolve()
    if resolved_root not in target.resolve().parents:
        message = f"{relative!r} would be written outside the restore destination"
        raise ArchiveError(message, code="archive_path_escapes_root")
    if target.exists() or target.is_symlink():
        message = (
            f"the manifest names {relative!r} more than once, or the staging tree already "
            f"holds it. Refusing to overwrite during a restore."
        )
        raise ArchiveError(message, code="archive_restore_collision")
    return target


def _iter_prefix(storage: ArchiveStorage, prefix: str) -> Iterator[str]:
    return storage.list_keys(prefix)


def _read_remote_manifest(storage: ArchiveStorage, session_id: str) -> ArchiveManifest | None:
    """Fetch and parse the committed manifest, or ``None`` if none exists."""
    key = manifest_key(session_id)
    if storage.head_object(key) is None:
        return None
    with storage.open_object(key) as body:
        raw = b""
        while chunk := body.read(1 << 20):
            raw += chunk
    try:
        return ArchiveManifest.model_validate_json(raw)
    except ValidationError as exc:
        message = (
            f"the committed manifest for {session_id!r} does not parse as an archive "
            f"manifest: {exc}. Refusing to act on it."
        )
        raise ArchiveError(message, code="archive_manifest_unreadable") from exc


def _require_remote_manifest(storage: ArchiveStorage, session_id: str) -> ArchiveManifest:
    found = _read_remote_manifest(storage, session_id)
    if found is None:
        message = (
            f"no committed archive for session {session_id!r}. `archive list` shows what "
            f"is there; a session whose upload was interrupted has objects but no manifest "
            f"and is not restorable."
        )
        raise ArchiveError(message, code="archive_not_committed")
    return found


def _select(manifest: ArchiveManifest, track_id: str | None) -> Sequence[ArchiveManifestEntry]:
    if track_id is None:
        return manifest.entries
    selected = manifest.for_track(track_id)
    if not selected:
        known = sorted({e.track_id for e in manifest.entries if e.track_id})
        message = (
            f"no archived file is attributed to track {track_id!r}. This archive has "
            f"{', '.join(known) or 'no attributed tracks'}. Unassigned files are recovered "
            f"only by a whole-session operation, which is deliberate: attributing them to "
            f"a track would be inventing identity (INV-11)."
        )
        raise ArchiveError(message, code="archive_unknown_track")
    return selected


def _entries_from_sources(session_id: str, sources: ArchiveSourceSet) -> list[ArchiveManifestEntry]:
    """Manifest entries with the compression fields left at zero, for comparison only.

    Used where only path, size and original digest matter — a divergence check must not
    depend on re-compressing gigabytes to find out whether anything changed.
    """
    return [
        ArchiveManifestEntry(
            path=encode_component(entry.relative_path),
            path_text=_text_or_none(entry.relative_path),
            track_id=entry.track_id,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
            compressed_size_bytes=0,
            compressed_sha256="0" * 64,
            object_key=object_key(session_id, entry),
        )
        for entry in sources.entries
    ]


def _manifest_entry(session_id: str, entry: ArchiveEntry, staged: _Staged) -> ArchiveManifestEntry:
    return ArchiveManifestEntry(
        path=encode_component(entry.relative_path),
        path_text=_text_or_none(entry.relative_path),
        track_id=entry.track_id,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        compressed_size_bytes=staged.compressed_size,
        compressed_sha256=staged.compressed_sha256,
        object_key=object_key(session_id, entry),
    )


def _entry_for_readback(entry: ArchiveManifestEntry) -> ArchiveEntry:
    """Adapt a manifest entry to what `_read_back` needs, so verify and upload share it."""
    return ArchiveEntry(
        relative_path=entry.path_text or entry.path,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        track_id=entry.track_id,
    )


def _plan_manifest(
    session_id: str, sources: ArchiveSourceSet, entries: list[ArchiveManifestEntry]
) -> ArchiveManifest:
    return ArchiveManifest(
        session_id=session_id,
        codec=ARCHIVE_CODEC_V1.describe(),
        entries=entries or _entries_from_sources(session_id, sources),
    )


def _text_or_none(relative_path: str) -> str | None:
    """A human-readable path, or nothing when no faithful one exists.

    A filename that is not valid UTF-8 has no text form that round-trips, and
    `canonical_json` would refuse to serialize its surrogates. Absent is the honest answer,
    and restore never reads this field anyway (ADR-0036).
    """
    try:
        relative_path.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return relative_path


def _put_bytes(storage: ArchiveStorage, key: str, payload: bytes, staging_root: Path) -> None:
    """Upload a small in-memory document by staging it, so one code path uploads files."""
    staged = staging_root / "manifest.tmp"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(payload)
    try:
        storage.put_object(key, staged)
    finally:
        staged.unlink(missing_ok=True)


def _retrieval_cost_note(gibibytes: float) -> str:
    return (
        f"This verification downloaded {gibibytes:.2f} GiB. Cold Storage charges "
        f"$0.01/GiB retrieved, waived up to your average daily Cold Storage usage, with a "
        f"128 KiB minimum per object. Verification is deliberately a full download: "
        f"anything cheaper would not be a verification."
    )


def _scope_note(track_id: str | None) -> list[str]:
    if track_id is None:
        return []
    return [
        f"Scope was track {track_id!r} only. Files not attributed to a track — nested "
        f"notes, unassigned audio — are covered only by a whole-session operation."
    ]


def _safe_verify(
    sources: ArchiveSourceSet, config: SessionConfig, results: list[ObjectResult]
) -> None:
    """Re-check the sources on a failure path without masking the original failure.

    INV-01 is verified on *every* exit path, but a source-set violation discovered while
    handling another error must not replace it: the first failure is what the operator
    needs to see, and a second one raised from an `except` block would hide it.
    """
    try:
        sources.verify_unchanged(config)
    except ArchiveError as exc:
        results.append(
            ObjectResult(
                path="(source set)",
                outcome=ArchiveObjectOutcome.FAILED,
                error=ArchiveReportError(code=exc.code, message=str(exc)),
            )
        )


def _as_error(exc: BaseException) -> ArchiveReportError:
    code = getattr(exc, "code", None)
    return ArchiveReportError(code=str(code) if code else "archive_failed", message=str(exc))


def _failed(
    operation: ArchiveOperation,
    session_id: str,
    exc: BaseException,
    started: dt.datetime,
    *,
    entries_in_scope: int,
) -> ArchiveReport:
    return ArchiveReport(
        operation=operation,
        status=OperationStatus.FAILED,
        scope=ArchiveScope(session_id=session_id, entries_in_scope=entries_in_scope),
        verification=VerificationState.ABSENT,
        errors=[_as_error(exc)],
        started_at=started,
        finished_at=_now(),
    )


def _report_path(session_dir: Path, operation: str) -> Path:
    """Where a session-local operation's report goes."""
    return session_dir / WORK_DIRNAME / f"archive-{operation}-report.json"
