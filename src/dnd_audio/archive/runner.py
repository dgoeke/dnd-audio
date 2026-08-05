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
import re
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from dnd_audio.archive import (
    ARCHIVE_MANIFEST_FILENAME,
    ARCHIVE_PREFIX,
    ArchiveError,
)
from dnd_audio.archive.codec import (
    ARCHIVE_CODEC_V1,
    CHUNK_BYTES,
    compress_bound,
    compress_file,
    decompress_and_measure,
)
from dnd_audio.archive.lock import single_writer
from dnd_audio.archive.manifest import (
    RESTORE_INSTRUCTIONS,
    ArchiveInspectionIdentity,
    ArchiveManifest,
    ArchiveManifestEntry,
)
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

#: A floor on a usable manifest, derived rather than guessed: every manifest carries
#: :data:`RESTORE_INSTRUCTIONS` verbatim, so a real one is always longer than them. Only a
#: size check, because `list` deliberately does not download and parse — that would cost a
#: retrieval per session, and `verify` is the operation that reads bytes. It catches the
#: truncated and zero-byte cases, which is what "committed" must not be claimed for.
_SMALLEST_POSSIBLE_MANIFEST_BYTES: Final = len(RESTORE_INSTRUCTIONS)

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
    # Inside the reporting boundary, like the lock. An unreadable `session.yaml` is an
    # ordinary operator error — the wrong directory, a typo — and it raised straight out of
    # here, so the command that INV-13 requires to leave a report left none. Found by M7a's
    # third code review.
    try:
        config = load_session_config(session_dir / "session.yaml")
    except Exception as exc:
        return _failed(ArchiveOperation.UPLOAD, "(unknown)", exc, started, entries_in_scope=0)
    session_id = config.session_id

    sources: ArchiveSourceSet | None = None
    try:
        document = _require_current_manifest(session_dir, config)
        sources = build_source_set(session_dir, config)
        _require_nothing_vanished(document, sources)
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
        # INV-01 is re-verified here too. The source set exists by the time the output-path
        # check runs, so a failure *after* it built one must still confirm nothing moved —
        # the invariant is about every exit path, not about the successful one.
        results: list[ObjectResult] = []
        if sources is not None:
            _safe_verify(sources, config, results)
        return _failed(
            ArchiveOperation.UPLOAD,
            session_id,
            exc,
            started,
            entries_in_scope=0 if sources is None else len(sources.entries),
            objects=results,
        )

    # The lock is acquired **inside** the reporting boundary. Held outside it, contention
    # raised out of this function and the caller wrote no report at all — so the one
    # failure an operator is most likely to hit produced nothing to read (INV-13).
    try:
        with single_writer(session_id, directory=lock_dir):
            return _upload_locked(
                session_dir=session_dir,
                config=config,
                sources=sources,
                document=document,
                storage=storage,
                staging_root=staging_root,
                started=started,
                free_bytes=free_bytes,
            )
    except Exception as exc:
        held: list[ObjectResult] = []
        _safe_verify(sources, config, held)
        return _failed(
            ArchiveOperation.UPLOAD,
            session_id,
            exc,
            started,
            entries_in_scope=len(sources.entries),
            objects=held,
        )


def _upload_locked(
    *,
    session_dir: Path,
    config: SessionConfig,
    sources: ArchiveSourceSet,
    document: dict[str, Any],
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

        committed = _read_remote_manifest(storage, session_id)
        if committed is not None:
            existing, payload = committed
            # INV-01 before returning, on this path too. It reads no source bytes, but the
            # invariant is a claim about every exit — and an early return that skipped it
            # was exactly the gap M7a's code review found.
            sources.verify_unchanged(config)
            return _resolve_existing_manifest(
                existing=existing,
                payload=payload,
                sources=sources,
                session_id=session_id,
                started=started,
            )

        shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        for position, entry in enumerate(sources.entries):
            try:
                staged = _compress_and_check(session_dir, entry, staging_root)
                try:
                    outcome = _publish(storage, session_id, entry, staged)
                    manifest_entries.append(_manifest_entry(session_id, entry, staged, document))
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
            except Exception as exc:
                # The report used to carry the successful prefix and nothing else, so the
                # object that actually failed appeared only in `errors` and the ones never
                # attempted appeared nowhere — leaving an operator to work out the scope of
                # a partial upload by subtracting two lists. The charter asks for per-object
                # outcomes; these are the outcomes. Found by M7a's third code review.
                results.append(
                    ObjectResult(
                        path=encode_component(entry.relative_path),
                        outcome=ArchiveObjectOutcome.FAILED,
                        size_bytes=entry.size_bytes,
                        error=_as_error(exc, path=encode_component(entry.relative_path)),
                    )
                )
                results.extend(
                    ObjectResult(
                        path=encode_component(remaining.relative_path),
                        outcome=ArchiveObjectOutcome.SKIPPED,
                        size_bytes=remaining.size_bytes,
                    )
                    for remaining in sources.entries[position + 1 :]
                )
                raise

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
        # `partial` means objects reached the bucket, not merely that the report has rows.
        # Once a failure also records the entry that failed and the ones never attempted,
        # `results` is never empty — so keying off it would call a run that failed on its
        # first file a partial success.
        published = any(
            item.outcome in (ArchiveObjectOutcome.UPLOADED, ArchiveObjectOutcome.ALREADY_PRESENT)
            for item in results
        )
        return ArchiveReport(
            operation=ArchiveOperation.UPLOAD,
            status=OperationStatus.PARTIAL if published else OperationStatus.FAILED,
            scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
            verification=VerificationState.PENDING if published else VerificationState.ABSENT,
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
        tee = _Tee(body)
        restored = decompress_and_measure(tee, max_output_bytes=entry.size_bytes)
        # **Drained to EOF, deliberately.** The decoder reads one frame and stops
        # (`read_across_frames=False`), so without this the digest above covers only the
        # bytes the decoder happened to pull — and an object made of a valid frame followed
        # by anything at all could satisfy both hashes while part of it was never read.
        # "The whole object was downloaded and checked" is the claim this function exists
        # to make; it has to be true of the whole object. Found by M7a's third review.
        trailing = 0
        while chunk := tee.read(CHUNK_BYTES):
            trailing += len(chunk)

    if trailing:
        message = (
            f"the object stored for {entry.relative_path} has {trailing} byte(s) after its "
            f"zstd frame. It decompresses correctly, so the archived audio is recoverable — "
            f"but the stored object is not the object this archive wrote, and a "
            f"content-addressed key that holds something else is a conflict, not a variant."
        )
        raise ArchiveError(message, code="archive_object_conflict")

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


#: The recipe fields that decide the compressed bytes. Everything else `describe()` records
#: — the `zstandard` and libzstd versions — explains *why* bytes might differ, and comparing
#: it would turn a dependency bump into a divergent archive (ADR-0037).
_RECIPE_IDENTITY_FIELDS: Final = (
    "format",
    "level",
    "threads",
    "write_checksum",
    "write_content_size",
    "write_dict_id",
)


def _recipe_identity(codec: dict[str, str | int | bool]) -> dict[str, str | int | bool]:
    """The byte-deciding half of a recorded recipe.

    A field this project has never written is kept rather than dropped: an unknown key in a
    committed manifest means the document was written by something that knew more than this
    build does, and quietly ignoring it would compare two recipes as equal on the strength
    of not understanding one of them.
    """
    return {
        name: value
        for name, value in codec.items()
        if name in _RECIPE_IDENTITY_FIELDS or not name.endswith("_version")
    }


def _identity(manifest: ArchiveManifest) -> tuple[object, ...]:
    """Everything about a manifest that is not derived from the compression itself.

    The compressed size and digest are deliberately excluded and nothing is lost by it:
    archive v1's recipe is frozen and single-threaded (ADR-0037), so identical source bytes
    under an identical recorded recipe produce identical compressed bytes. Comparing them
    would require re-compressing the whole session just to discover whether anything
    changed.

    **`track_id` is in here**, and its absence was a real defect: comparing only path, size
    and digest meant that reassigning track ids between directories and re-inspecting
    produced "already archived, identical content" while the committed manifest kept the
    old attribution — and `--track` recovery then returned another speaker's audio. That is
    an INV-11 violation reached through an equality check. Found by M7a's code review.

    **The library versions are not**, which `ArchiveCodec.describe` always said: they are
    description, not identity, because a zstd frame is readable by any later libzstd. They
    were being compared anyway, so an ordinary `uv lock --upgrade` of `zstandard` made
    every already-archived session report `divergent` — the word this project reserves for
    "the bucket and the disk disagree about content" — and made `upload` fail fatally
    telling the operator to pick a new session id or restore. A false alarm about a backup
    is not a cheap thing to raise. Found by M7a's second code review.
    """
    return (
        manifest.session_id,
        manifest.archive_version,
        tuple(sorted(_recipe_identity(manifest.codec).items())),
        tuple(
            (
                entry.path,
                entry.path_text,
                entry.track_id,
                entry.size_bytes,
                entry.sha256,
                entry.object_key,
            )
            for entry in manifest.entries
        ),
    )


def _resolve_existing_manifest(
    *,
    existing: ArchiveManifest,
    payload: bytes,
    sources: ArchiveSourceSet,
    session_id: str,
    started: dt.datetime,
) -> ArchiveReport:
    """Full-identity equality is idempotent success; any difference is fatal (ADR-0038)."""
    planned = _plan_manifest(session_id, sources, _entries_from_sources(session_id, sources))
    same = _identity(existing) == _identity(planned)

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
    try:
        config = load_session_config(session_dir / "session.yaml")
    except Exception as exc:
        return _failed(ArchiveOperation.STATUS, "(unknown)", exc, started, entries_in_scope=0)
    session_id = config.session_id

    try:
        committed = _read_remote_manifest(storage, session_id)
        objects_present = (
            False
            if committed is not None
            else any(True for _ in _iter_prefix(storage, _session_prefix(session_id)))
        )
        # Inventoried **after** the network round trip, which is the ordering that costs
        # nothing and closes the window. Built first, the comparison was against the
        # directory as it stood before a download of unknown duration, so a session being
        # written to during the call could still be reported `committed`.
        #
        # The obvious fix — build first and `verify_unchanged` after, as `upload` does —
        # was rejected: `verify_unchanged` re-walks and re-hashes, and doubling the work
        # would make the one operation this charter calls *cheap* hash a four-hour session
        # twice. `status` writes nothing, so it needs no INV-01 exit check; what it needs is
        # for its answer to be about the present, and reading the directory last is how.
        # Found by M7a's second code review.
        sources = build_source_set(session_dir, config)
    except Exception as exc:
        return _failed(ArchiveOperation.STATUS, session_id, exc, started, entries_in_scope=0)

    if committed is None:
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

    remote, payload = committed
    planned = _plan_manifest(session_id, sources, _entries_from_sources(session_id, sources))
    matches = _identity(remote) == _identity(planned)

    if not matches:
        return ArchiveReport(
            operation=ArchiveOperation.STATUS,
            status=OperationStatus.COMPLETE,
            scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
            verification=VerificationState.DIVERGENT,
            manifest_sha256=sha256_bytes(payload),
            notes=[
                "The committed manifest describes different content than this directory "
                "holds. Neither was changed."
            ],
            started_at=started,
            finished_at=_now(),
        )

    # The one place `previously_verified_at_commit` comes from. It is a claim about
    # *history*, so its only honest source is the report the committing upload wrote — and
    # that report exists locally, beside the session, which is exactly where this operation
    # is standing. Absent one, `committed` is all that can be said (ADR-0039).
    verified_at_commit = _upload_report_says_verified(session_dir, session_id, payload)
    return ArchiveReport(
        operation=ArchiveOperation.STATUS,
        status=OperationStatus.COMPLETE,
        scope=ArchiveScope(session_id=session_id, entries_in_scope=len(sources.entries)),
        verification=(
            VerificationState.PREVIOUSLY_VERIFIED_AT_COMMIT
            if verified_at_commit
            else VerificationState.COMMITTED
        ),
        manifest_sha256=sha256_bytes(payload),
        notes=[
            "A manifest exists and matches this directory's contents."
            + (
                " The upload that committed it read every object back at the time — which "
                "is history, not a statement about the bytes in the bucket today."
                if verified_at_commit
                else ""
            )
            + " Only `archive verify` establishes that the stored bytes still restore, by "
            "downloading them."
        ],
        started_at=started,
        finished_at=_now(),
    )


def _upload_report_says_verified(session_dir: Path, session_id: str, manifest_bytes: bytes) -> bool:
    """Whether the local upload report vouches for **this** committed manifest.

    Deliberately tolerant: a missing, unreadable, or unrecognized report simply means the
    weaker answer. Treating a parse failure as "verified" would be the one direction that
    matters, and treating it as an error would make `status` fail over an artifact it does
    not need.

    **The report must name this session and this manifest.** It used to be believed on the
    strength of saying `upload`, `complete` and `verified` and nothing else, so a report
    left behind by an earlier archive of a different session id — or by an upload of an
    earlier manifest for this one — vouched for bytes it had never seen. A claim about
    history is only worth making about the history it is actually a record of. Found by
    M7a's second code review.
    """
    path = _report_path(session_dir, "upload")
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    scope = document.get("scope")
    return (
        document.get("operation") == ArchiveOperation.UPLOAD.value
        and document.get("verification") == VerificationState.VERIFIED.value
        and document.get("status") == OperationStatus.COMPLETE.value
        and isinstance(scope, dict)
        and scope.get("session_id") == session_id
        and document.get("manifest_sha256") == sha256_bytes(manifest_bytes)
    )


def run_list(*, storage: ArchiveStorage) -> tuple[list[str], ArchiveReport]:
    """Every committed session id, discovered without a local session directory.

    Follows pagination to exhaustion. A partial listing reported as complete would make a
    session look absent during exactly the emergency this command exists for (OQ-028).

    **A key is not a manifest, and neither is a plausible size.** Each candidate is
    downloaded, parsed, and checked to belong to the key it sits at, because "committed" is
    what an operator reads before deciding a recovery is possible — and a size floor accepts
    any sufficiently large garbage. The charter asks this command for "manifest identity",
    which cannot be produced without reading the document, so the digest of each is reported.

    The retrieval this costs is one manifest per session, billed at Cold Storage's 128 KiB
    floor. An earlier version skipped it to avoid exactly that cost and called the objects
    committed on the strength of their length; the trade was wrong, because this command
    exists to be believed during a recovery. `verify` remains the expensive operation — it
    downloads every *object* — and this one still reads nothing but the commit markers.
    """
    started = _now()
    session_ids: list[str] = []
    unreadable: list[str] = []
    digests: dict[str, str] = {}

    try:
        for key in _iter_prefix(storage, f"{ARCHIVE_PREFIX}/"):
            if not key.endswith(f"/{ARCHIVE_MANIFEST_FILENAME}"):
                continue
            encoded = key[len(f"{ARCHIVE_PREFIX}/") :].split("/", 1)[0]
            try:
                session_id = decode_component(encoded)
            except ArchiveError:
                # A key this project did not write. Reported rather than dropped: an
                # unrecognizable key under our own prefix is worth a human knowing about.
                unreadable.append(encoded)
                continue

            # The key must be **the** manifest key for that session, not merely a key that
            # ends in the manifest filename. `.../<session>/objects/x/archive-manifest.v1.
            # json` was being read as a committed session whose id came from the first
            # component, which is a session id this project never wrote a manifest for.
            # Found by M7a's second code review.
            if key != manifest_key(session_id):
                unreadable.append(session_id)
                continue

            head = storage.head_object(key)
            if head is None or head.size_bytes < _SMALLEST_POSSIBLE_MANIFEST_BYTES:
                unreadable.append(session_id)
                continue

            # Read and parsed, not merely measured. `_read_remote_manifest` also checks the
            # document names this session and points its entries at this session's prefix,
            # so a listing cannot announce a session whose manifest belongs to another.
            try:
                found = _read_remote_manifest(storage, session_id)
            except ArchiveError:
                unreadable.append(session_id)
                continue
            if found is None:  # pragma: no cover - the head above already found it
                unreadable.append(session_id)
                continue
            digests[session_id] = sha256_bytes(found[1])
            session_ids.append(session_id)
    except Exception as exc:
        return [], _failed(ArchiveOperation.LIST, "(all)", exc, started, entries_in_scope=0)

    notes = [
        f"{len(session_ids)} committed session(s), each with a manifest that was downloaded "
        f"and parsed. No *object* was read, so nothing here says a session still restores — "
        f"`archive verify --session-id ...` is what establishes that.",
        *(
            [
                "manifest digests: "
                + ", ".join(f"{name} {digests[name][:12]}" for name in sorted(session_ids))
            ]
            if session_ids
            else []
        ),
    ]
    if unreadable:
        notes.append(
            f"{len(unreadable)} key(s) under this project's prefix are not usable "
            f"manifests (empty, truncated, unparseable, or describing a different "
            f"session): {', '.join(sorted(unreadable)[:5])}. They are not listed as "
            f"sessions, because an operator reads this before deciding a recovery is "
            f"possible."
        )

    report = ArchiveReport(
        operation=ArchiveOperation.LIST,
        status=OperationStatus.COMPLETE,
        scope=ArchiveScope(session_id="(all)", entries_in_scope=len(session_ids)),
        verification=(VerificationState.COMMITTED if session_ids else VerificationState.ABSENT),
        notes=notes,
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
        remote, manifest_bytes = _require_remote_manifest(storage, session_id)
        selected = _select(remote, track_id)
    except Exception as exc:
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
        except Exception as exc:
            results.append(
                ObjectResult(
                    path=entry.path,
                    outcome=ArchiveObjectOutcome.FAILED,
                    size_bytes=entry.size_bytes,
                    error=_as_error(exc, path=entry.path),
                )
            )
            errors.append(_as_error(exc, path=entry.path))
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
        manifest_sha256=sha256_bytes(manifest_bytes),
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
        # Before the network, deliberately. A destination inside a source root is illegal
        # whatever the bucket holds, and checking it second meant the INV-01 refusal was
        # reached only for sessions that happened to be committed — so on a fresh bucket
        # the operation failed with `archive_not_committed` and the guard never ran. It
        # also spends a manifest GET to learn something local.
        _check_destination(destination, protected_session_dirs)
        remote, manifest_bytes = _require_remote_manifest(storage, session_id)
        selected = _select(remote, track_id)
        _preflight_restore(selected, destination, free_bytes=free_bytes)
    except Exception as exc:
        return _failed(ArchiveOperation.RESTORE, session_id, exc, started, entries_in_scope=0)

    # Named by digest, for the reason the multipart state files were: a valid session id
    # can be long enough that percent-encoding it passes the 255-byte filename-component
    # limit, so an upload that succeeded — object keys have a 1 024-byte budget — could not
    # be restored. Locks and multipart records were converted after M7a's first code
    # review; this one was missed. Found by the second.
    staging = destination.parent / (
        _RESTORE_STAGING_PREFIX + sha256_bytes(encode_component(session_id).encode("ascii"))[:32]
    )
    results: list[ObjectResult] = []

    # **Refused if it already exists, rather than removed.** The name is derived from the
    # session and the destination's parent, so two concurrent restores of one session
    # underneath one directory land on the same tree — and blindly `rmtree`-ing it meant
    # each run could delete the other's half-written files, or a track-scoped restore could
    # publish files a whole-session restore had put there. Refusing makes the collision an
    # error rather than a corruption, and makes a leftover from a hard kill something an
    # operator is told about instead of something silently reused. Found by M7a's third
    # code review.
    if staging.exists() or staging.is_symlink():
        message = (
            f"a restore of this session into this directory is already in progress, or one "
            f"was killed partway and left {staging.name!r} behind. Refusing to reuse it: "
            f"two restores sharing a staging tree can publish each other's files. If "
            f"nothing else is running, remove that directory and retry."
        )
        return _failed(
            ArchiveOperation.RESTORE,
            session_id,
            ArchiveError(message, code="archive_restore_in_progress"),
            started,
            entries_in_scope=len(selected),
        )

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
            manifest_sha256=sha256_bytes(manifest_bytes),
            # Relabelled, because the restore is transactional: the staging tree these
            # entries were written into has just been deleted and the destination is
            # untouched, so `restored` would name files that do not exist anywhere. They
            # were downloaded and verified and then deliberately discarded, which is
            # `skipped`. Found by M7a's second code review.
            objects=[_as_unpublished(result) for result in results],
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
        manifest_sha256=sha256_bytes(manifest_bytes),
        objects=results,
        notes=[
            "Every restored file was decompressed and matched its recorded digest.",
            *_scope_note(track_id),
        ],
        started_at=started,
        finished_at=_now(),
    )


# --- shared helpers --------------------------------------------------------------------


def _require_current_manifest(session_dir: Path, config: SessionConfig) -> dict[str, Any]:
    """The session must have been inspected, and inspected as it stands now.

    Returns the parsed manifest so :func:`_require_nothing_vanished` can cross-check the
    source set against it without reading the file twice.
    """
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

    # Checked before anything is read *out* of the document, which is the only order that
    # works: `.get` on a JSON array raises `AttributeError`, so the guard below used to sit
    # downstream of the call it was guarding and could never fire.
    if not isinstance(document, dict):
        message = f"{MANIFEST_RELATIVE_PATH} is not a manifest object"
        raise ArchiveError(message, code="archive_manifest_absent")
    if document.get("config_hash") != config_hash(config):
        message = (
            f"{MANIFEST_RELATIVE_PATH} was written under a different configuration than "
            f"`session.yaml` now describes, so it does not describe what is on disk. Run "
            f"`dnd-audio inspect` again before archiving."
        )
        raise ArchiveError(message, code="archive_manifest_stale")
    return document


def _require_nothing_vanished(document: dict[str, Any], sources: ArchiveSourceSet) -> None:
    """Every file inspection recorded must still be in the source set.

    The refusal in `build_source_set` catches a whole *root* going missing. This catches
    the narrower and likelier case: one file deleted or renamed inside a root that is still
    there, between inspection and archiving. Without it the upload commits a manifest
    describing a session that has quietly lost a recording — and reports success, which is
    the outcome that makes an operator confident it is safe to lose the local copy.

    Uses inspection's inventory as the reference rather than a second traversal, because
    the manifest is the only record of what was there *before*.
    """
    recorded: set[str] = set()
    for track in document.get("tracks", []):
        for source in track.get("sources", []):
            recorded.add(str(source["relative_path"]))
    for source in document.get("unassigned", []):
        recorded.add(str(source["relative_path"]))

    present = {entry.relative_path for entry in sources.entries}
    missing = sorted(recorded - present)
    if missing:
        message = (
            f"{len(missing)} file(s) that inspection recorded are no longer present: "
            f"{', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}. Nothing was "
            f"archived. Archiving what remains would commit a manifest describing a "
            f"session that has lost a recording, and report success while doing it."
        )
        raise ArchiveError(message, code="archive_source_vanished")


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
    """An empty directory, outside every protected source root, reached without a symlink.

    Every message here names paths by their final component rather than in full. These
    strings reach the operation report, which the completion gate requires to carry no
    local machine identity — and an absolute destination is a home directory and a username
    (ADR-0039). Nothing is lost: there is exactly one destination per invocation and the
    operator typed it. Found by M7a's second code review.
    """
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
                    f"the restore destination resolves inside the source directory "
                    f"{root!r} of session {session_dir.name!r}. Nothing may be written "
                    f"under one (INV-01) — restore somewhere else and compare by hand."
                )
                raise ArchiveError(message, code="archive_destination_protected")

    # Every component, not only the leaf. A symlinked *parent* — `~/backups -> /mnt/other`
    # — decides where a whole restored session lands just as completely as a symlinked
    # destination does, and the leaf-only check accepted it. The same "at every component"
    # rule the upload side applies (ADR-0036). Found by M7a's code review.
    for component in (destination, *destination.parents):
        if component.is_symlink():
            shown = (
                "the restore destination"
                if component == destination
                else f"the path component {component.name!r} above the restore destination"
            )
            message = (
                f"{shown} is a symlink. Where it points decides what gets written, and "
                f"that is not visible from the command line — the archive refuses a link "
                f"at every path component rather than following one. Give a real path."
            )
            raise ArchiveError(message, code="archive_symlink_refused")
    if not destination.is_dir():
        message = (
            f"the restore destination {destination.name!r} does not exist or is not a directory"
        )
        raise ArchiveError(message, code="archive_destination_unusable")
    if any(destination.iterdir()):
        message = (
            f"the restore destination {destination.name!r} is not empty. A restore "
            f"recreates a whole tree and will not merge into existing files — point it at "
            f"an empty directory so what arrives is unambiguous."
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


def _read_remote_manifest(
    storage: ArchiveStorage, session_id: str
) -> tuple[ArchiveManifest, bytes] | None:
    """Fetch and parse the committed manifest, with **the bytes that arrived**.

    The bytes are returned rather than reserialized from the parsed model, because the
    report records the manifest's digest and a reserialization identifies the model, not
    the stored object. Those differ the moment the schema gains an optional field — and
    the hash would then name something that is not in the bucket (ADR-0003, ADR-0039).
    """
    key = manifest_key(session_id)
    if storage.head_object(key) is None:
        return None
    with storage.open_object(key) as body:
        raw = b""
        while chunk := body.read(1 << 20):
            raw += chunk
    try:
        manifest = ArchiveManifest.model_validate_json(raw)
    except ValidationError as exc:
        message = (
            f"the committed manifest for {session_id!r} does not parse as an archive "
            f"manifest: {exc}. Refusing to act on it."
        )
        raise ArchiveError(message, code="archive_manifest_unreadable") from exc
    _require_manifest_belongs_to(manifest, session_id)
    return manifest, raw


def _require_manifest_belongs_to(manifest: ArchiveManifest, session_id: str) -> None:
    """The document at a session's key must actually be that session's manifest.

    Schema validity was the only thing checked, so a valid manifest for session B placed at
    session A's key made `verify A` and `restore A` operate on B's objects while reporting
    A — and every digest matches, because they are B's digests. Recovery reading correct
    bytes for the wrong session is worse than a failure: it looks like success.

    Both halves are needed. The session id catches a whole document in the wrong place; the
    key prefix catches a document that names the right session while pointing its entries
    somewhere else. Keys are still not recomputed field by field — the entry's own digest is
    what proves the bytes, and `_read_back` checks it — but they must live under this
    session's own object prefix. Found by M7a's third code review.
    """
    if manifest.session_id != session_id:
        message = (
            f"the manifest stored at {session_id!r}'s key says it describes "
            f"{manifest.session_id!r}. Refusing to restore or verify one session from "
            f"another's record — every digest in it would match, and the recovery would "
            f"look like a success."
        )
        raise ArchiveError(message, code="archive_manifest_session_mismatch")

    prefix = f"{_session_prefix(session_id)}objects/"
    stray = sorted(
        entry.path for entry in manifest.entries if not entry.object_key.startswith(prefix)
    )
    if stray:
        message = (
            f"{len(stray)} entry/entries in {session_id!r}'s manifest name objects outside "
            f"that session's own prefix, starting with {stray[0]!r}. A manifest may not "
            f"redirect a recovery to another session's audio."
        )
        raise ArchiveError(message, code="archive_manifest_foreign_object")


def _require_remote_manifest(
    storage: ArchiveStorage, session_id: str
) -> tuple[ArchiveManifest, bytes]:
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


def _inspection_identity(
    document: dict[str, Any], relative_path: str
) -> ArchiveInspectionIdentity | None:
    """The bounded container facts inspection recorded for one path, if any.

    Read out of `manifest.json` rather than re-probed. The archive runs no FFprobe, and
    deriving these from the bytes would be a second implementation of something M1 owns —
    which is how two answers to "what format is this" end up in one project.
    """
    for track in document.get("tracks", []):
        for source in track.get("sources", []):
            if source.get("relative_path") == relative_path:
                return _identity_from(source)
    for source in document.get("unassigned", []):
        if source.get("relative_path") == relative_path:
            return _identity_from(source)
    return None


def _identity_from(source: dict[str, Any]) -> ArchiveInspectionIdentity | None:
    """Copy the bounded subset, or nothing when the file was never probed.

    A duplicate or an ignored `edit` is recorded and left alone by inspection, so it has no
    container record — and absent is the honest answer rather than a set of nulls.
    """
    container = source.get("container")
    if not isinstance(container, dict):
        return None
    return ArchiveInspectionIdentity(
        codec_name=container.get("codec_name"),
        sample_format=container.get("sample_format"),
        sample_rate=container.get("sample_rate"),
        channels=container.get("channels"),
        sample_count=container.get("sample_count"),
    )


def _manifest_entry(
    session_id: str,
    entry: ArchiveEntry,
    staged: _Staged,
    document: dict[str, Any] | None = None,
) -> ArchiveManifestEntry:
    return ArchiveManifestEntry(
        path=encode_component(entry.relative_path),
        path_text=_text_or_none(entry.relative_path),
        track_id=entry.track_id,
        size_bytes=entry.size_bytes,
        sha256=entry.sha256,
        compressed_size_bytes=staged.compressed_size,
        compressed_sha256=staged.compressed_sha256,
        object_key=object_key(session_id, entry),
        inspection=(
            None if document is None else _inspection_identity(document, entry.relative_path)
        ),
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


def _as_unpublished(result: ObjectResult) -> ObjectResult:
    """One restore result, demoted because the transaction rolled back over it."""
    if result.outcome is not ArchiveObjectOutcome.RESTORED:
        return result
    return result.model_copy(update={"outcome": ArchiveObjectOutcome.SKIPPED})


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


def _as_error(exc: BaseException, *, path: str | None = None) -> ArchiveReportError:
    """Any exception as a structured error a caller can branch on (INV-13).

    Reads `code` when the exception carries one and falls back to a generic code otherwise,
    so an unexpected exception type still produces a *structured* failure rather than
    escaping the operation and leaving no report at all.
    """
    code = getattr(exc, "code", None)
    return ArchiveReportError(
        code=str(code) if code else "archive_failed",
        message=_without_absolute_paths(str(exc) or type(exc).__name__),
        path=path,
    )


#: Any absolute POSIX path in an error message. Matched greedily enough to catch a filename
#: with spaces up to the end of the segment, which is what an `OSError` carries.
_ABSOLUTE_PATH = re.compile(r"(?<![\w/])/(?:[^\s'\"<>|]+)")


def _without_absolute_paths(message: str) -> str:
    """Reduce every absolute path in a message to its final component.

    Report errors reach a durable file the completion gate requires to carry no local
    machine identity, and an absolute path is a home directory and a username. The
    destination messages were written carefully; the ones that leak are the ones nobody
    writes — a real `ENOSPC` is `[Errno 28] No space left on device: '/home/…/tx-a/x.zst'`,
    and `_as_error` passed it through verbatim. The ENOSPC tests inject an exception with
    no filename, so they could not see it. Found by M7a's third code review.

    The final component is kept because it is the actionable half: which *file* ran out of
    room is worth knowing, and where the operator's home directory is is not.
    """
    return _ABSOLUTE_PATH.sub(lambda found: Path(found.group()).name or "/", message)


def _failed(
    operation: ArchiveOperation,
    session_id: str,
    exc: BaseException,
    started: dt.datetime,
    *,
    entries_in_scope: int,
    objects: list[ObjectResult] | None = None,
) -> ArchiveReport:
    return ArchiveReport(
        operation=operation,
        status=OperationStatus.FAILED,
        scope=ArchiveScope(session_id=session_id, entries_in_scope=entries_in_scope),
        verification=VerificationState.ABSENT,
        objects=objects or [],
        errors=[_as_error(exc)],
        started_at=started,
        finished_at=_now(),
    )


def _report_path(session_dir: Path, operation: str) -> Path:
    """Where a session-local operation's report goes."""
    return session_dir / WORK_DIRNAME / f"archive-{operation}-report.json"
