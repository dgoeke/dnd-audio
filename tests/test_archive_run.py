"""The five operations, end to end against the deterministic fake.

The test that matters most is `TestDisasterRecovery`: upload a session, **delete the whole
session directory**, and rebuild it from the session id alone. Everything else in this
milestone is machinery in service of that one drill working on a day when the local copy is
gone and nobody remembers an object key.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

import pytest

from dnd_audio.archive import ARCHIVE_MANIFEST_FILENAME
from dnd_audio.archive.fakes import FakeArchiveStorage, StorageFault
from dnd_audio.archive.manifest import ArchiveManifest, ArchiveManifestEntry
from dnd_audio.archive.paths import encode_component
from dnd_audio.archive.report import (
    ArchiveObjectOutcome,
    OperationStatus,
    VerificationState,
)
from dnd_audio.archive.runner import (
    manifest_key,
    run_list,
    run_restore,
    run_status,
    run_upload,
    run_verify,
)
from dnd_audio.archive.sourceset import build_source_set
from dnd_audio.config import load_session_config
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.runner import run_inspect


@pytest.fixture
def inspected(canonical_fixture: FixtureTruth) -> FixtureTruth:
    """A session that has been inspected, which `upload` requires."""
    result = run_inspect(canonical_fixture.session_dir)
    assert result.exit_code is ExitCode.OK
    return canonical_fixture


@pytest.fixture
def storage() -> FakeArchiveStorage:
    return FakeArchiveStorage()


@pytest.fixture
def lock_dir(tmp_path: Path) -> Path:
    return tmp_path / "locks"


def session_id_of(fixture: FixtureTruth) -> str:
    return load_session_config(fixture.session_dir / "session.yaml").session_id


class TestUpload:
    def test_it_archives_every_entry_and_commits_last(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.COMPLETE
        assert report.exit_code() is ExitCode.OK

        sources = build_source_set(
            inspected.session_dir, load_session_config(inspected.session_dir / "session.yaml")
        )
        assert report.scope.entries_in_scope == len(sources.entries)
        assert len(report.objects) == len(sources.entries)
        assert all(item.outcome is ArchiveObjectOutcome.UPLOADED for item in report.objects)

        # The manifest is the last thing written. Anything else would make an interrupted
        # upload look committed during the window that matters most (ADR-0038).
        puts = [key for operation, key in storage.calls if operation == "put"]
        assert puts[-1] == manifest_key(session_id_of(inspected))

    def test_every_object_is_read_back_before_the_manifest_goes_up(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """A backup that has never been restored is a belief, not a backup."""
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        manifest_index = next(
            i
            for i, (operation, key) in enumerate(storage.calls)
            if operation == "put" and key.endswith(ARCHIVE_MANIFEST_FILENAME)
        )
        gets_before = [
            key for operation, key in storage.calls[:manifest_index] if operation == "get"
        ]
        object_puts = [
            key
            for operation, key in storage.calls[:manifest_index]
            if operation == "put" and not key.endswith(ARCHIVE_MANIFEST_FILENAME)
        ]
        assert set(object_puts) <= set(gets_before)
        assert len(gets_before) >= len(object_puts)

    def test_the_bucket_holds_exactly_one_small_object_per_session(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """Cold Storage bills anything under 128 KiB as 128 KiB.

        Compared as an **exact key set** rather than by counting keys outside `objects/`,
        which would miss a report accidentally written beneath it.
        """
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        session_id = session_id_of(inspected)
        stored = _manifest_from(storage, session_id)
        expected = {entry.object_key for entry in stored.entries} | {manifest_key(session_id)}
        assert set(storage.objects) == expected

    def test_no_report_or_sidecar_is_uploaded(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert not any("report" in key for key in storage.objects)
        assert sum(key.endswith(".json") for key in storage.objects) == 1

    def test_it_leaves_no_staging_behind(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert not (inspected.session_dir / "work" / "archive").exists()

    def test_it_does_not_modify_the_sources(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """INV-01: the archive reads, and that is all it does."""
        config = load_session_config(inspected.session_dir / "session.yaml")
        before = build_source_set(inspected.session_dir, config)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        before.verify_unchanged(config)

    def test_an_uninspected_session_is_refused(
        self, canonical_fixture: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        report = run_upload(canonical_fixture.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_manifest_absent"
        assert "dnd-audio inspect" in report.errors[0].message
        assert not storage.objects

    def test_a_stale_manifest_is_refused(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """A manifest written under a different configuration does not describe this disk."""
        import yaml

        path = inspected.session_dir / "session.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["title"] = "a different title"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_manifest_stale"
        assert not storage.objects

    def test_a_symlink_stops_the_upload_before_anything_is_sent(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        secret = inspected.session_dir.parent / "private"
        secret.write_text("not session data\n", encoding="utf-8")
        (inspected.session_dir / "raw" / "innocent.wav").symlink_to(secret)

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_symlink_refused"
        assert not storage.objects

    def test_insufficient_disk_is_refused_before_compressing(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """Preflight is driven directly, at the bound, rather than by filling a disk.

        Injecting ENOSPC would prove cleanup works; it would say nothing about whether the
        arithmetic is right (plan review, P0).
        """
        report = run_upload(
            inspected.session_dir, storage=storage, lock_dir=lock_dir, free_bytes=1024
        )
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_insufficient_space"
        assert not storage.objects

    def test_preflight_passes_just_above_the_bound(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """The other side of the boundary, so the threshold is a threshold."""
        from dnd_audio.archive.codec import compress_bound
        from dnd_audio.archive.runner import _PREFLIGHT_SLACK_BYTES

        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        largest = max(entry.size_bytes for entry in sources.entries)
        needed = compress_bound(largest) + _PREFLIGHT_SLACK_BYTES

        assert (
            run_upload(
                inspected.session_dir, storage=storage, lock_dir=lock_dir, free_bytes=needed - 1
            ).status
            is OperationStatus.FAILED
        )
        storage.objects.clear()
        assert (
            run_upload(
                inspected.session_dir, storage=storage, lock_dir=lock_dir, free_bytes=needed
            ).status
            is OperationStatus.COMPLETE
        )


class TestIdempotenceAndDivergence:
    def test_re_uploading_identical_content_is_success_without_re_uploading(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        before = len(storage.calls)

        second = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert second.status is OperationStatus.COMPLETE
        assert second.verification is VerificationState.COMMITTED
        assert not any(operation == "put" for operation, _ in storage.calls[before:]), (
            "an idempotent re-upload must not write anything"
        )

    def test_an_idempotent_rerun_does_not_claim_to_have_verified_anything(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """It downloaded no object, so `verified` would be exactly ADR-0039's overstatement."""
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        second = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert second.verification is not VerificationState.VERIFIED
        assert "archive verify" in " ".join(second.notes)

    def test_changed_local_content_against_a_commit_is_fatal_divergence(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """Never a merge and never an overwrite: an archive version is immutable."""
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        stored_before = dict(storage.objects)

        (inspected.session_dir / "raw" / "new-note.txt").write_text("later\n", encoding="utf-8")
        run_inspect(inspected.session_dir)

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.verification is VerificationState.DIVERGENT
        assert report.errors[0].code == "archive_manifest_divergent"
        assert storage.objects == stored_before

    def test_a_zstd_library_upgrade_is_not_a_divergent_archive(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """Library versions are description, not identity — as `describe()` always said.

        They were being compared anyway, so an ordinary `uv lock --upgrade` of `zstandard`
        made every already-archived session report `divergent` and made `upload` fail
        fatally, telling the operator to choose a new session id or restore. A false alarm
        about a backup is expensive. Found by M7a's second code review.

        The committed manifest is rewritten with a *different recorded library version* and
        nothing else, which is exactly the shape a dependency bump leaves behind.
        """
        import json

        from dnd_audio.determinism import canonical_json

        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        session_id = session_id_of(inspected)
        document = json.loads(storage.objects[manifest_key(session_id)])
        assert document["codec"]["zstandard_version"], "the recipe must record its library"
        document["codec"]["zstandard_version"] = "99.0.0"
        document["codec"]["libzstd_version"] = "9.9.9"
        storage.objects[manifest_key(session_id)] = canonical_json(document).encode("utf-8")

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.COMPLETE
        assert report.verification is VerificationState.COMMITTED

        status = run_status(inspected.session_dir, storage=storage)
        assert status.verification is not VerificationState.DIVERGENT

    def test_a_changed_compression_level_is_still_divergent(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """The other half, without which the test above is a way of comparing nothing.

        Level, thread count and every frame flag decide the bytes, so they stay in the
        identity — only the library versions left it.
        """
        import json

        from dnd_audio.determinism import canonical_json

        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        session_id = session_id_of(inspected)
        document = json.loads(storage.objects[manifest_key(session_id)])
        document["codec"]["level"] = 11
        storage.objects[manifest_key(session_id)] = canonical_json(document).encode("utf-8")

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.verification is VerificationState.DIVERGENT


class TestInterruptedUpload:
    def test_a_failure_leaves_no_manifest_and_says_so(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        from dnd_audio.archive.sourceset import object_key

        doomed = object_key(config.session_id, sources.entries[2])
        storage.arm(StorageFault(key=doomed, operation="put", kind="error"))

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.PARTIAL
        assert report.exit_code() is ExitCode.PARTIAL
        assert report.verification is VerificationState.PENDING
        assert manifest_key(config.session_id) not in storage.objects
        assert not (inspected.session_dir / "work" / "archive").exists()

    def test_a_resumed_upload_accepts_objects_already_present(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """And accepts them only after downloading them, never on a size or an ETag."""
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        from dnd_audio.archive.sourceset import object_key

        doomed = object_key(config.session_id, sources.entries[2])
        storage.arm(StorageFault(key=doomed, operation="put", kind="error"))
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        storage.faults.clear()
        resumed = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert resumed.status is OperationStatus.COMPLETE
        assert any(item.outcome is ArchiveObjectOutcome.ALREADY_PRESENT for item in resumed.objects)

    def test_a_corrupt_readback_fails_the_object_rather_than_committing_it(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        from dnd_audio.archive.sourceset import object_key

        storage.arm(
            StorageFault(
                key=object_key(config.session_id, sources.entries[0]),
                operation="get",
                kind="corrupt",
            )
        )
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert manifest_key(config.session_id) not in storage.objects

    def test_a_source_that_changes_mid_upload_stops_the_commit(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """INV-01 is verified **before the manifest**, and this test can prove it.

        The first version of this completed an upload, mutated a file afterwards, and
        asserted a failure — which it got, from existing-manifest divergence. It would have
        passed with the pre-commit verification deleted entirely. Found by M7a's code
        review, and it is the exact shape M1's closeout warns about.

        The second version still did not. It mutated a file the loop had **not yet
        reached**, so `_compress_and_check`'s own re-hash caught it and the test passed with
        the pre-commit verification deleted — the same defect, one layer down. Verified by
        deleting the call and watching it stay green.

        This one mutates the **first** entry after it has already been compressed, uploaded
        and read back. Nothing revisits it, so the only thing standing between that
        mutation and a committed manifest is `sources.verify_unchanged(config)` before the
        manifest PUT. Delete that call and this test fails.
        """
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        victim = inspected.session_dir / sources.entries[0].relative_path

        class MutatingStorage:
            """Changes a source file the moment the second object is uploaded."""

            def __init__(self, inner: FakeArchiveStorage) -> None:
                self.inner = inner
                self.puts = 0

            def put_object(self, key: str, source: Path) -> None:
                self.puts += 1
                self.inner.put_object(key, source)
                # After the *first* object is fully published, so the entry it belongs to
                # is behind the loop and only the pre-commit check can still see it.
                if self.puts == 1:
                    victim.write_bytes(victim.read_bytes() + b"\x00")

            def head_object(self, key: str):  # type: ignore[no-untyped-def]
                return self.inner.head_object(key)

            def open_object(self, key: str):  # type: ignore[no-untyped-def]
                return self.inner.open_object(key)

            def list_keys(self, prefix: str):  # type: ignore[no-untyped-def]
                return self.inner.list_keys(prefix)

        mutating = MutatingStorage(storage)
        report = run_upload(inspected.session_dir, storage=mutating, lock_dir=lock_dir)

        assert report.status is not OperationStatus.COMPLETE
        assert manifest_key(config.session_id) not in storage.objects, (
            "a manifest was committed describing a source that changed during the upload"
        )
        assert any(
            "archive_sources_modified" in (error.code or "") for error in report.errors
        ) or any(
            "archive_sources_modified" in (item.error.code if item.error else "")
            for item in report.objects
        )


class TestSourcesThatVanish:
    """The worst failure a backup can have: committing successfully while incomplete."""

    def test_a_missing_configured_root_is_refused(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """An unmounted disk must not become a smaller archive reported as complete.

        `build_source_set` used to `continue` past a root that was not there, so the upload
        archived the tracks that remained and committed. Found by M7a's code review.
        """
        import shutil as _shutil

        _shutil.rmtree(inspected.session_dir / "raw")
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_source_root_missing"
        assert not storage.objects

    def test_a_file_deleted_after_inspection_is_refused(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """The narrower and likelier case: one recording gone, its directory still there."""
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        (inspected.session_dir / sources.entries[0].relative_path).unlink()

        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_source_vanished"
        assert sources.entries[0].relative_path in report.errors[0].message
        assert not storage.objects

    def test_a_file_added_after_inspection_is_not_treated_as_a_loss(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """Only *disappearance* is fatal here.

        A file appearing after inspection is an ordinary thing an operator does — dropping
        a notes file beside the recordings — and the archive should take it, not refuse the
        session. The manifest cross-check is deliberately one-directional.
        """
        (inspected.session_dir / "raw" / "added-later.txt").write_text("ok\n", encoding="utf-8")
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.COMPLETE
        assert any("added-later" in item.path for item in report.objects)


class TestLockFailuresStillReport:
    def test_contention_produces_a_report_rather_than_an_exception(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """INV-13: the failure an operator is most likely to hit must leave something to read.

        The lock used to be acquired outside the reporting boundary, so contention raised
        straight out of `run_upload` and the CLI wrote no report at all.
        """
        from dnd_audio.archive.lock import single_writer

        config = load_session_config(inspected.session_dir / "session.yaml")
        with single_writer(config.session_id, directory=lock_dir):
            report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_upload_in_progress"
        assert report.exit_code() is not ExitCode.OK


class TestStatus:
    def test_an_unarchived_session_is_absent(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage
    ) -> None:
        report = run_status(inspected.session_dir, storage=storage)
        assert report.verification is VerificationState.ABSENT

    def test_a_committed_session_is_committed_and_never_verified(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """The word `status` may never say, enforced by the report model itself."""
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        report = run_status(inspected.session_dir, storage=storage)
        assert report.verification is VerificationState.COMMITTED
        # And the note says so in words, because the operator deciding whether it is safe
        # to lose the local copy is reading prose, not an enum. That `status` structurally
        # *cannot* say `verified` is asserted in tests/test_archive_manifest.py.
        assert "archive verify" in " ".join(report.notes)

    def test_it_reports_previously_verified_only_when_a_commit_report_says_so(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """The state that was defined and never produced, until code review found it.

        `previously_verified_at_commit` is a claim about history, so its only honest source
        is the report the committing upload wrote — which lives beside the session, where
        `status` is already standing. Absent one, `committed` is all that can be said.
        """
        from dnd_audio.archive.report import write_report
        from dnd_audio.archive.runner import _report_path

        upload = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert upload.verification is VerificationState.VERIFIED

        # Before the upload's report is on disk, history is not available to be read.
        assert (
            run_status(inspected.session_dir, storage=storage).verification
            is VerificationState.COMMITTED
        )

        write_report(upload, _report_path(inspected.session_dir, "upload"))
        after = run_status(inspected.session_dir, storage=storage)
        assert after.verification is VerificationState.PREVIOUSLY_VERIFIED_AT_COMMIT
        assert "history" in " ".join(after.notes)

    def test_an_unreadable_upload_report_yields_the_weaker_answer(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """A corrupt report must never be read as evidence of verification."""
        from dnd_audio.archive.runner import _report_path

        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        path = _report_path(inspected.session_dir, "upload")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        assert (
            run_status(inspected.session_dir, storage=storage).verification
            is VerificationState.COMMITTED
        )

    def test_objects_without_a_manifest_are_pending(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        from dnd_audio.archive.sourceset import object_key

        storage.arm(
            StorageFault(
                key=object_key(config.session_id, sources.entries[2]),
                operation="put",
                kind="error",
            )
        )
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        report = run_status(inspected.session_dir, storage=storage)
        assert report.verification is VerificationState.PENDING
        assert "interrupted" in " ".join(report.notes)

    def test_changed_local_content_is_divergent(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        (inspected.session_dir / "raw" / "extra.txt").write_text("new\n", encoding="utf-8")
        report = run_status(inspected.session_dir, storage=storage)
        assert report.verification is VerificationState.DIVERGENT

    def test_a_source_that_changes_during_the_comparison_is_caught(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """`status` was the one operation that never re-checked INV-01.

        It inventoried the sources, then made a network round trip that takes as long as it
        takes, then reported on a directory that may have moved underneath the comparison —
        so a session actively being written to could still be reported `committed`. The
        source is mutated from inside the storage layer, which is where the wait is.
        Found by M7a's second code review.
        """
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        victim = inspected.session_dir / "raw" / "midflight.txt"
        victim.write_text("before\n", encoding="utf-8")
        run_inspect(inspected.session_dir)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        class MutatingOnRead:
            """Changes a source while `status` is waiting on the manifest download."""

            def __init__(self, inner: FakeArchiveStorage) -> None:
                self.inner = inner

            def head_object(self, key: str):  # type: ignore[no-untyped-def]
                return self.inner.head_object(key)

            def open_object(self, key: str):  # type: ignore[no-untyped-def]
                victim.write_text("changed mid-flight\n", encoding="utf-8")
                return self.inner.open_object(key)

            def put_object(self, key: str, source: Path) -> None:
                self.inner.put_object(key, source)

            def list_keys(self, prefix: str):  # type: ignore[no-untyped-def]
                return self.inner.list_keys(prefix)

        report = run_status(inspected.session_dir, storage=MutatingOnRead(storage))
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_sources_modified"

    def test_another_sessions_upload_report_does_not_vouch_for_this_one(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """A history claim must be a record of *this* history.

        The check read `operation`, `status` and `verification` and nothing else, so a
        report left behind by an archive of a different session — or of an earlier manifest
        for this one — vouched for bytes it had never seen. Found by M7a's second review.
        """
        import json

        from dnd_audio.archive.report import write_report
        from dnd_audio.archive.runner import _report_path

        upload = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        path = _report_path(inspected.session_dir, "upload")
        write_report(upload, path)
        assert (
            run_status(inspected.session_dir, storage=storage).verification
            is VerificationState.PREVIOUSLY_VERIFIED_AT_COMMIT
        ), "the positive control failed, so the assertions below prove nothing"

        document = json.loads(path.read_bytes())
        document["scope"]["session_id"] = "some-other-session"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert (
            run_status(inspected.session_dir, storage=storage).verification
            is VerificationState.COMMITTED
        )

        document["scope"]["session_id"] = session_id_of(inspected)
        document["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(document), encoding="utf-8")
        assert (
            run_status(inspected.session_dir, storage=storage).verification
            is VerificationState.COMMITTED
        ), "a report about a different manifest vouched for this one"


class TestList:
    def test_it_discovers_sessions_without_a_local_directory(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        found, report = run_list(storage=storage)
        assert found == [session_id_of(inspected)]
        assert report.status is OperationStatus.COMPLETE

    def test_it_follows_pagination_to_exhaustion(self, storage: FakeArchiveStorage) -> None:
        """The fake pages every two keys, so a caller that stops early is visibly wrong.

        Completeness is the whole contract: a partial listing reported as complete makes a
        session look absent during exactly the emergency this command exists for (OQ-028).
        """
        for index in range(7):
            storage.objects[manifest_key(f"session-{index}")] = _minimal_manifest_bytes(
                f"session-{index}"
            )
        found, _ = run_list(storage=storage)
        assert found == [f"session-{index}" for index in range(7)]

    def test_an_empty_archive_lists_nothing_rather_than_failing(
        self, storage: FakeArchiveStorage
    ) -> None:
        found, report = run_list(storage=storage)
        assert found == []
        assert report.status is OperationStatus.COMPLETE
        assert report.verification is VerificationState.ABSENT

    def test_a_manifest_filename_somewhere_else_is_not_a_session(
        self, storage: FakeArchiveStorage
    ) -> None:
        """The key must be *the* manifest key, not merely end in the manifest filename.

        A key one level deeper was read as a committed session whose id came from the
        first path component — announcing a recovery is possible for a session that has no
        manifest at all. Found by M7a's second code review.
        """
        storage.objects[f"sessions/archive-v1/real/objects/x/{ARCHIVE_MANIFEST_FILENAME}"] = (
            _minimal_manifest_bytes("real")
        )
        found, report = run_list(storage=storage)
        assert found == []
        assert "not usable manifests" in " ".join(report.notes)

    def test_a_truncated_manifest_is_not_a_session(self, storage: FakeArchiveStorage) -> None:
        """A one-byte object passed the "not empty" check and was called committed."""
        storage.objects[manifest_key("half-written")] = b"{"
        found, report = run_list(storage=storage)
        assert found == []
        assert "half-written" in " ".join(report.notes)

    def test_a_real_manifest_is_still_listed(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """The positive control for the two refusals above."""
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        found, _ = run_list(storage=storage)
        assert found == [session_id_of(inspected)]


class TestVerify:
    def test_it_downloads_everything_and_reports_verified(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        before = len(storage.calls)

        report = run_verify(session_id_of(inspected), storage=storage)
        assert report.status is OperationStatus.COMPLETE
        assert report.verification is VerificationState.VERIFIED
        gets = [key for operation, key in storage.calls[before:] if operation == "get"]
        assert len(gets) >= report.scope.entries_in_scope

    def test_it_states_the_retrieval_cost_without_weakening_the_check(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        notes = " ".join(run_verify(session_id_of(inspected), storage=storage).notes)
        assert "$0.01/GiB" in notes
        assert "waived" in notes

    def test_a_corrupt_object_is_caught_and_named(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        stored = _manifest_from(storage, session_id_of(inspected))
        target = stored.entries[0]
        storage.arm(StorageFault(key=target.object_key, operation="get", kind="corrupt"))

        report = run_verify(session_id_of(inspected), storage=storage)
        assert report.status is OperationStatus.PARTIAL
        assert report.exit_code() is ExitCode.PARTIAL
        assert any(item.outcome is ArchiveObjectOutcome.FAILED for item in report.objects)
        assert report.errors[0].path == target.path

    def test_a_track_scope_verifies_only_that_track(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        report = run_verify(session_id_of(inspected), storage=storage, track_id="tx-a")
        assert report.scope.track_id == "tx-a"
        assert report.scope.entries_in_scope > 0
        assert "whole-session" in " ".join(report.notes)

    def test_an_unknown_track_is_refused_with_what_is_there(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        report = run_verify(session_id_of(inspected), storage=storage, track_id="tx-zz")
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_unknown_track"
        assert "tx-a" in report.errors[0].message

    def test_an_uncommitted_session_is_refused(self, storage: FakeArchiveStorage) -> None:
        report = run_verify("never-archived", storage=storage)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_not_committed"


class TestDisasterRecovery:
    """Upload, delete everything local, and rebuild from the session id alone."""

    def test_a_whole_session_restores_after_the_directory_is_gone(
        self,
        inspected: FixtureTruth,
        storage: FakeArchiveStorage,
        lock_dir: Path,
        tmp_path: Path,
    ) -> None:
        config = load_session_config(inspected.session_dir / "session.yaml")
        expected = {
            entry.relative_path: (entry.size_bytes, entry.sha256)
            for entry in build_source_set(inspected.session_dir, config).entries
        }
        session_id = config.session_id

        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        # The disaster. Nothing local survives — no session.yaml, no manifest, no notes.
        shutil.rmtree(inspected.session_dir)

        destination = tmp_path / "recovered"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage)
        assert report.status is OperationStatus.COMPLETE

        rebuilt = {
            path.relative_to(destination).as_posix(): (
                path.stat().st_size,
                sha256_file(path),
            )
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        assert rebuilt == expected

    def test_the_session_id_is_discoverable_without_knowing_it(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """`list` is what makes the drill above possible when nobody remembers the id."""
        expected = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        shutil.rmtree(inspected.session_dir)
        found, _ = run_list(storage=storage)
        assert expected in found

    def test_a_single_track_restores_on_its_own(
        self,
        inspected: FixtureTruth,
        storage: FakeArchiveStorage,
        lock_dir: Path,
        tmp_path: Path,
    ) -> None:
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        shutil.rmtree(inspected.session_dir)

        destination = tmp_path / "one-track"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage, track_id="tx-a")
        assert report.status is OperationStatus.COMPLETE
        restored = [
            path.relative_to(destination).as_posix()
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        ]
        assert restored
        assert all(path.startswith("raw/tx-a/") for path in restored)

    def test_unassigned_files_come_back_only_from_a_whole_session_restore(
        self,
        inspected: FixtureTruth,
        storage: FakeArchiveStorage,
        lock_dir: Path,
        tmp_path: Path,
    ) -> None:
        """The documented cost of never inventing identity (INV-11), asserted both ways."""
        (inspected.session_dir / "raw" / "field-notes.txt").write_text("kept\n", encoding="utf-8")
        run_inspect(inspected.session_dir)
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        by_track = tmp_path / "by-track"
        by_track.mkdir()
        run_restore(session_id, by_track, storage=storage, track_id="tx-a")
        assert not (by_track / "raw" / "field-notes.txt").exists()

        whole = tmp_path / "whole"
        whole.mkdir()
        run_restore(session_id, whole, storage=storage)
        assert (whole / "raw" / "field-notes.txt").read_text(encoding="utf-8") == "kept\n"


class TestRestoreRefusals:
    def test_a_non_empty_destination_is_refused(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        destination = tmp_path / "occupied"
        destination.mkdir()
        (destination / "already-here.txt").write_text("mine\n", encoding="utf-8")

        report = run_restore(session_id, destination, storage=storage)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_destination_not_empty"
        assert (destination / "already-here.txt").exists()

    def test_a_destination_inside_a_protected_source_root_is_refused(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        """INV-01 applies to a restore as much as to a run."""
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        destination = inspected.session_dir / "raw" / "restore-here"
        destination.mkdir()

        report = run_restore(
            session_id,
            destination,
            storage=storage,
            protected_session_dirs=[inspected.session_dir],
        )
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_destination_protected"

    def test_a_symlinked_destination_is_refused(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)

        report = run_restore(session_id, link, storage=storage)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_symlink_refused"

    def test_a_refusal_names_no_absolute_path(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        """Report errors carry no local machine identity, and a home directory is some.

        Every destination refusal embedded the full path, which the completion gate forbids
        in a report (ADR-0039). Nothing is lost by naming the final component: there is one
        destination per invocation and the operator typed it. Found by M7a's second review.
        """
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        occupied = tmp_path / "not-empty"
        occupied.mkdir()
        (occupied / "something.txt").write_text("here\n", encoding="utf-8")
        report = run_restore(session_id, occupied, storage=storage)

        assert report.errors[0].code == "archive_destination_not_empty"
        serialized = report.model_dump_json()
        assert str(tmp_path) not in serialized, "the report carries an absolute local path"
        assert "not-empty" in report.errors[0].message, "it must still say which one"

    def test_a_session_id_too_long_to_be_a_filename_still_restores(
        self, tmp_path: Path, storage: FakeArchiveStorage
    ) -> None:
        """Restore staged under a directory named from the encoded session id.

        Object keys have a 1 024-byte budget, so a 300-character session id uploads
        perfectly — and then percent-encoding it produced a staging basename past the
        255-byte filename-component limit, so the archive could not be restored. Locks and
        multipart records were converted to digest names after the first code review; this
        one was missed. Found by the second.
        """
        from dnd_audio.archive.codec import ARCHIVE_CODEC_V1, compress_file
        from dnd_audio.archive.manifest import ArchiveManifest, ArchiveManifestEntry
        from dnd_audio.determinism import canonical_json, sha256_bytes

        session_id = "a-very-long-session-name-" * 12
        assert len(session_id) > 255

        payload = b"one restored recording\n"
        source = tmp_path / "one.wav"
        source.write_bytes(payload)
        staged = tmp_path / "one.zst"
        fact = compress_file(source, staged)

        key = f"sessions/archive-v1/x/objects/one.wav.{sha256_bytes(payload)}.zst"
        storage.objects[key] = staged.read_bytes()
        manifest = ArchiveManifest(
            session_id=session_id,
            codec=ARCHIVE_CODEC_V1.describe(),
            entries=[
                ArchiveManifestEntry(
                    path=encode_component("raw/tx-a/one.wav"),
                    path_text="raw/tx-a/one.wav",
                    track_id="tx-a",
                    size_bytes=len(payload),
                    sha256=sha256_bytes(payload),
                    compressed_size_bytes=fact.size_bytes,
                    compressed_sha256=fact.sha256,
                    object_key=key,
                )
            ],
        )
        storage.objects[manifest_key(session_id)] = canonical_json(
            manifest.model_dump(mode="json")
        ).encode("utf-8")

        destination = tmp_path / "recovered"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage)

        assert report.status is OperationStatus.COMPLETE, report.errors
        assert (destination / "raw" / "tx-a" / "one.wav").read_bytes() == payload

    def test_insufficient_space_is_refused_before_downloading(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        before = len(storage.calls)

        destination = tmp_path / "small"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage, free_bytes=1024)
        assert report.status is OperationStatus.FAILED
        assert report.errors[0].code == "archive_insufficient_space"
        assert not [key for operation, key in storage.calls[before:] if operation == "get"][1:], (
            "nothing beyond the manifest should have been downloaded"
        )


class TestRestoreIsTransactional:
    def test_a_failure_partway_leaves_the_destination_untouched(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        """The plan review's P1: per-file publication strands its own retry.

        With whole-tree staging, a failure at object three leaves the destination exactly
        as it was — so the operator's next move is "run it again", not "work out which of
        the nineteen files that arrived are safe to delete".
        """
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        stored = _manifest_from(storage, session_id)
        storage.arm(StorageFault(key=stored.entries[2].object_key, operation="get", kind="corrupt"))

        destination = tmp_path / "recovered"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage)

        assert report.status is OperationStatus.FAILED
        assert destination.is_dir()
        assert list(destination.iterdir()) == []
        assert not list(tmp_path.glob(".dnd-audio-restore-*"))

    def test_a_rolled_back_restore_does_not_call_anything_restored(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        """The report must not name files that exist nowhere.

        Entries written into the staging tree before the failure kept the outcome
        `restored`, and then the staging tree was deleted and the destination left empty —
        so the report listed recovered files an operator could not find. They were
        downloaded, verified, and deliberately discarded, which is `skipped`. Found by
        M7a's second code review.
        """
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        stored = _manifest_from(storage, session_id)
        storage.arm(StorageFault(key=stored.entries[2].object_key, operation="get", kind="corrupt"))

        destination = tmp_path / "recovered"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage)

        assert report.status is OperationStatus.FAILED
        assert report.objects, "entries were written before the failure, so this is not vacuous"
        assert not any(item.outcome is ArchiveObjectOutcome.RESTORED for item in report.objects)
        assert all(item.outcome is ArchiveObjectOutcome.SKIPPED for item in report.objects), [
            item.outcome for item in report.objects
        ]

    def test_a_failed_restore_can_simply_be_retried(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        stored = _manifest_from(storage, session_id)
        storage.arm(StorageFault(key=stored.entries[2].object_key, operation="get", kind="corrupt"))

        destination = tmp_path / "recovered"
        destination.mkdir()
        assert run_restore(session_id, destination, storage=storage).status is (
            OperationStatus.FAILED
        )

        storage.faults.clear()
        assert run_restore(session_id, destination, storage=storage).status is (
            OperationStatus.COMPLETE
        )

    def test_an_interrupted_transfer_leaves_nothing_published(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path, tmp_path: Path
    ) -> None:
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        stored = _manifest_from(storage, session_id)
        storage.arm(
            StorageFault(key=stored.entries[1].object_key, operation="get", kind="interrupt")
        )

        destination = tmp_path / "recovered"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage)
        assert report.status is OperationStatus.FAILED
        assert list(destination.iterdir()) == []


class TestDiskExhaustion:
    """ENOSPC at each phase, which the charter requires and the first pass omitted.

    Preflight is tested separately and directly, at and just below its computed bound —
    injecting ENOSPC proves *cleanup*, not that the arithmetic is right, and conflating the
    two was one of the plan review's findings. These tests are the cleanup half.
    """

    def _no_space(self) -> OSError:
        return OSError(errno.ENOSPC, "No space left on device")

    def test_exhaustion_while_compressing_leaves_no_staging_and_no_commit(
        self,
        inspected: FixtureTruth,
        storage: FakeArchiveStorage,
        lock_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = load_session_config(inspected.session_dir / "session.yaml")
        import dnd_audio.archive.runner as runner_module

        def failing(*args: object, **kwargs: object) -> None:
            raise self._no_space()

        monkeypatch.setattr(runner_module, "compress_file", failing)
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        assert report.status is OperationStatus.FAILED
        assert manifest_key(config.session_id) not in storage.objects
        assert not (inspected.session_dir / "work" / "archive").exists()

    def test_exhaustion_while_uploading_leaves_no_manifest(
        self, inspected: FixtureTruth, storage: FakeArchiveStorage, lock_dir: Path
    ) -> None:
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        from dnd_audio.archive.sourceset import object_key

        class Exhausted:
            def __init__(self, inner: FakeArchiveStorage, doomed: str) -> None:
                self.inner = inner
                self.doomed = doomed

            def put_object(self, key: str, source: Path) -> None:
                if key == self.doomed:
                    raise OSError(errno.ENOSPC, "No space left on device")
                self.inner.put_object(key, source)

            def head_object(self, key: str):  # type: ignore[no-untyped-def]
                return self.inner.head_object(key)

            def open_object(self, key: str):  # type: ignore[no-untyped-def]
                return self.inner.open_object(key)

            def list_keys(self, prefix: str):  # type: ignore[no-untyped-def]
                return self.inner.list_keys(prefix)

        exhausted = Exhausted(storage, object_key(config.session_id, sources.entries[1]))
        report = run_upload(inspected.session_dir, storage=exhausted, lock_dir=lock_dir)

        assert report.status is not OperationStatus.COMPLETE
        assert manifest_key(config.session_id) not in storage.objects
        assert not (inspected.session_dir / "work" / "archive").exists()

    def test_exhaustion_while_restoring_leaves_the_destination_untouched(
        self,
        inspected: FixtureTruth,
        storage: FakeArchiveStorage,
        lock_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The transactional restore's whole point, under the failure it exists for."""
        session_id = session_id_of(inspected)
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)

        import dnd_audio.archive.runner as runner_module

        real = runner_module.decompress_and_measure  # type: ignore[attr-defined]
        state = {"calls": 0}

        def failing(*args: object, **kwargs: object) -> object:
            state["calls"] += 1
            if state["calls"] == 3:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(runner_module, "decompress_and_measure", failing)

        destination = tmp_path / "recovered"
        destination.mkdir()
        report = run_restore(session_id, destination, storage=storage)

        assert report.status is OperationStatus.FAILED
        assert list(destination.iterdir()) == []
        assert not list(tmp_path.glob(".dnd-audio-restore-*"))

    def test_a_report_still_describes_the_failure(
        self,
        inspected: FixtureTruth,
        storage: FakeArchiveStorage,
        lock_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """INV-13: running out of disk is exactly when an operator needs a report."""
        import dnd_audio.archive.runner as runner_module

        def failing(*args: object, **kwargs: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(runner_module, "compress_file", failing)
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.errors
        assert report.exit_code() is not ExitCode.OK


def _manifest_from(storage: FakeArchiveStorage, session_id: str) -> ArchiveManifest:
    return ArchiveManifest.model_validate_json(storage.objects[manifest_key(session_id)])


def _minimal_manifest_bytes(session_id: str) -> bytes:
    """A committed manifest with one plausible entry, for listing tests."""
    from dnd_audio.archive.codec import ARCHIVE_CODEC_V1
    from dnd_audio.determinism import canonical_json

    document = ArchiveManifest(
        session_id=session_id,
        codec=ARCHIVE_CODEC_V1.describe(),
        entries=[
            ArchiveManifestEntry(
                path=encode_component("raw/tx-a/one.wav"),
                path_text="raw/tx-a/one.wav",
                track_id="tx-a",
                size_bytes=1,
                sha256="0" * 64,
                compressed_size_bytes=1,
                compressed_sha256="1" * 64,
                object_key="k",
            )
        ],
    )
    return canonical_json(document.model_dump(mode="json")).encode("utf-8")
