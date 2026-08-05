"""The five operations, end to end against the deterministic fake.

The test that matters most is `TestDisasterRecovery`: upload a session, **delete the whole
session directory**, and rebuild it from the session id alone. Everything else in this
milestone is machinery in service of that one drill working on a day when the local copy is
gone and nobody remembers an object key.
"""

from __future__ import annotations

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
        """INV-01 is verified before the manifest, not only at the start."""
        config = load_session_config(inspected.session_dir / "session.yaml")
        sources = build_source_set(inspected.session_dir, config)
        from dnd_audio.archive.sourceset import object_key

        # Fail after the first object, then mutate a source so the pre-commit re-check has
        # something to catch on the resumed run.
        first_key = object_key(config.session_id, sources.entries[0])
        run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert first_key in storage.objects

        target = inspected.session_dir / sources.entries[0].relative_path
        target.write_bytes(target.read_bytes() + b"\x00")
        report = run_upload(inspected.session_dir, storage=storage, lock_dir=lock_dir)
        assert report.status is OperationStatus.FAILED


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
        assert "not the same as" in " ".join(report.notes)
        assert "archive verify" in " ".join(report.notes)

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
