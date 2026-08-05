"""The commit record, the operation report, and the single-writer lock.

The report tests carry most of the weight here, and specifically the ones about the word
`verified`. That distinction is the milestone's central correctness property: an operator
deciding whether it is safe to lose a local copy is deciding entirely on which of three
words they were shown, and a cheap `status` that could say the strongest one is the lie
this artifact was designed to make impossible (ADR-0039).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.codec import ARCHIVE_CODEC_V1
from dnd_audio.archive.lock import lock_path, single_writer
from dnd_audio.archive.manifest import (
    RESTORE_INSTRUCTIONS,
    ArchiveManifest,
    ArchiveManifestEntry,
)
from dnd_audio.archive.paths import encode_component
from dnd_audio.archive.report import (
    ArchiveObjectOutcome,
    ArchiveOperation,
    ArchiveReport,
    ArchiveReportError,
    ArchiveScope,
    ObjectResult,
    OperationStatus,
    VerificationState,
    write_report,
)
from dnd_audio.determinism import canonical_json
from dnd_audio.errors import ExitCode

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
INSTANT = dt.datetime(2026, 8, 15, 19, 0, 0, tzinfo=dt.UTC)
LATER = dt.datetime(2026, 8, 15, 19, 5, 0, tzinfo=dt.UTC)


def entry(path: str = "raw%2Ftx-a%2Fone.wav", **overrides: object) -> ArchiveManifestEntry:
    values: dict[str, object] = {
        "path": path,
        "path_text": None,
        "track_id": "tx-a",
        "size_bytes": 1000,
        "sha256": DIGEST_A,
        "compressed_size_bytes": 700,
        "compressed_sha256": DIGEST_B,
        "object_key": f"sessions/archive-v1/s/objects/{path}.{DIGEST_A}.zst",
    }
    values.update(overrides)
    return ArchiveManifestEntry.model_validate(values)


def manifest(*entries: ArchiveManifestEntry) -> ArchiveManifest:
    return ArchiveManifest(
        session_id="session-2026-08-15",
        codec=ARCHIVE_CODEC_V1.describe(),
        entries=list(entries) or [entry()],
    )


class TestManifest:
    def test_it_is_byte_stable_and_sorted_by_path(self) -> None:
        """Directory order must never reach the bytes (INV-02)."""
        forwards = manifest(entry("b.wav", object_key="k-b"), entry("a.wav", object_key="k-a"))
        backwards = manifest(entry("a.wav", object_key="k-a"), entry("b.wav", object_key="k-b"))
        assert [item.path for item in forwards.entries] == ["a.wav", "b.wav"]
        assert canonical_json(forwards.model_dump(mode="json")) == canonical_json(
            backwards.model_dump(mode="json")
        )

    def test_it_carries_standalone_restore_instructions(self) -> None:
        """A recovery that needs this repository to still exist is not a recovery."""
        document = manifest().model_dump(mode="json")
        assert document["restore_instructions"] == RESTORE_INSTRUCTIONS
        assert "zstd -d" in RESTORE_INSTRUCTIONS
        assert "sha256sum" in RESTORE_INSTRUCTIONS
        assert "uppercase hex" in RESTORE_INSTRUCTIONS
        assert "not decode it as text" in RESTORE_INSTRUCTIONS

    def test_it_records_the_complete_recipe(self) -> None:
        codec = manifest().codec
        assert codec["level"] == 10
        assert codec["threads"] == 0
        assert codec["format"] == "zstd"
        assert "libzstd_version" in codec

    def test_it_holds_no_timestamp_hostname_or_credential(self) -> None:
        """Nothing here may vary between two identical uploads, and nothing may identify
        the machine or the bucket."""
        text = canonical_json(manifest().model_dump(mode="json")).lower()
        for forbidden in ("timestamp", "hostname", "endpoint", "access_key", "secret", "etag"):
            assert forbidden not in text

    def test_two_entries_for_one_path_are_refused(self) -> None:
        """Restoring whichever came last, silently, is the failure this prevents."""
        with pytest.raises(ValueError, match="more than once"):
            manifest(entry("same.wav", object_key="k1"), entry("same.wav", object_key="k2"))

    def test_two_entries_sharing_an_object_key_are_refused(self) -> None:
        with pytest.raises(ValueError, match="share an object key"):
            manifest(entry("a.wav", object_key="same"), entry("b.wav", object_key="same"))

    def test_an_empty_manifest_is_refused(self) -> None:
        """A session with no files is not an archive; it is a mistake worth reporting."""
        with pytest.raises(ValueError, match="at least 1"):
            ArchiveManifest(session_id="s", codec=ARCHIVE_CODEC_V1.describe(), entries=[])

    def test_track_scope_selects_only_attributed_entries(self) -> None:
        found = manifest(
            entry("a.wav", object_key="k1", track_id="tx-a"),
            entry("b.wav", object_key="k2", track_id="tx-b"),
            entry("notes.txt", object_key="k3", track_id=None),
        )
        assert [item.path for item in found.for_track("tx-a")] == ["a.wav"]
        assert found.total_original_bytes == 3000
        assert found.total_compressed_bytes == 2100

    def test_it_validates_against_the_committed_schema(self, repo_root: Path) -> None:
        schema = json.loads(
            (repo_root / "schemas" / "archive-manifest.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(manifest().model_dump(mode="json"), schema)

    def test_a_path_that_is_not_valid_utf8_survives_serialization(self) -> None:
        """The whole reason `path` is the encoded byte form (ADR-0036).

        `canonical_json` raises on a surrogate, so a manifest holding decoded text could
        not represent this file at all — and the crash would land partway through an
        upload rather than at validation time.
        """
        original = os.fsdecode(b"raw/tx-a/broken-\xff.wav")
        encoded = encode_component(original)
        document = manifest(entry(encoded, object_key="k", path_text=None))
        text = canonical_json(document.model_dump(mode="json"))
        assert "%FF" in text
        json.loads(text)


class TestReportVocabulary:
    """The three words, and the rule that keeps them apart."""

    def build(self, **overrides: object) -> ArchiveReport:
        values: dict[str, object] = {
            "operation": ArchiveOperation.VERIFY,
            "status": OperationStatus.COMPLETE,
            "scope": ArchiveScope(session_id="s", entries_in_scope=1),
            "verification": VerificationState.VERIFIED,
            "started_at": INSTANT,
            "finished_at": LATER,
        }
        values.update(overrides)
        return ArchiveReport.model_validate(values)

    def test_verify_may_report_verified(self) -> None:
        assert self.build().verification is VerificationState.VERIFIED

    def test_upload_may_report_verified_because_it_reads_everything_back(self) -> None:
        """The readback inside `upload` is a current full download (ADR-0038)."""
        assert self.build(operation=ArchiveOperation.UPLOAD).verification is (
            VerificationState.VERIFIED
        )

    @pytest.mark.parametrize(
        "operation",
        [ArchiveOperation.STATUS, ArchiveOperation.LIST, ArchiveOperation.RESTORE],
    )
    def test_a_cheap_operation_may_never_report_verified(self, operation: ArchiveOperation) -> None:
        """The finding the first plan review raised, enforced in one place for five commands.

        A `status` that says `verified` from provider metadata costs nothing to emit and
        everything to believe.
        """
        with pytest.raises(ValueError, match="may not report `verified`"):
            self.build(operation=operation)

    def test_status_may_report_what_it_actually_knows(self) -> None:
        for state in (
            VerificationState.ABSENT,
            VerificationState.PENDING,
            VerificationState.COMMITTED,
            VerificationState.PREVIOUSLY_VERIFIED_AT_COMMIT,
            VerificationState.DIVERGENT,
        ):
            report = self.build(operation=ArchiveOperation.STATUS, verification=state)
            assert report.verification is state


class TestReportDiscipline:
    def base(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "operation": ArchiveOperation.UPLOAD,
            "status": OperationStatus.COMPLETE,
            "scope": ArchiveScope(session_id="s", entries_in_scope=2),
            "verification": VerificationState.COMMITTED,
            "started_at": INSTANT,
            "finished_at": LATER,
        }
        values.update(overrides)
        return values

    def test_partial_never_exits_zero(self) -> None:
        """INV-13, restated for this artifact rather than assumed from the other one."""
        report = ArchiveReport.model_validate(
            self.base(
                status=OperationStatus.PARTIAL,
                errors=[ArchiveReportError(code="x", message="one object failed")],
            )
        )
        assert report.exit_code() is ExitCode.PARTIAL
        assert report.exit_code() != 0

    def test_failed_exits_nonzero_and_must_carry_an_error(self) -> None:
        with pytest.raises(ValueError, match="structured error"):
            ArchiveReport.model_validate(self.base(status=OperationStatus.FAILED))
        report = ArchiveReport.model_validate(
            self.base(
                status=OperationStatus.FAILED,
                errors=[ArchiveReportError(code="archive_failed", message="no")],
            )
        )
        assert report.exit_code() is ExitCode.FATAL

    def test_complete_exits_zero(self) -> None:
        assert ArchiveReport.model_validate(self.base()).exit_code() is ExitCode.OK

    def test_a_failed_object_must_say_why(self) -> None:
        with pytest.raises(ValueError, match="without a structured error"):
            ObjectResult(path="a.wav", outcome=ArchiveObjectOutcome.FAILED)

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            ArchiveReport.model_validate(
                self.base(started_at=dt.datetime(2026, 8, 15, 19, 0, 0))  # noqa: DTZ001
            )

    def test_objects_serialize_in_path_order(self) -> None:
        report = ArchiveReport.model_validate(
            self.base(
                objects=[
                    ObjectResult(path="b.wav", outcome=ArchiveObjectOutcome.UPLOADED),
                    ObjectResult(path="a.wav", outcome=ArchiveObjectOutcome.UPLOADED),
                ]
            )
        )
        assert [item.path for item in report.objects] == ["a.wav", "b.wav"]

    def test_the_scope_says_how_many_were_in_it(self) -> None:
        """So "3 verified" is read against "of 3" rather than looking complete alone."""
        report = ArchiveReport.model_validate(
            self.base(scope=ArchiveScope(session_id="s", track_id="tx-a", entries_in_scope=3))
        )
        assert report.scope.entries_in_scope == 3
        assert report.scope.track_id == "tx-a"

    def test_it_is_written_atomically_and_validates_against_its_schema(
        self, tmp_path: Path, repo_root: Path
    ) -> None:
        report = ArchiveReport.model_validate(self.base(manifest_sha256=DIGEST_A))
        path = write_report(report, tmp_path / "archive-report.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(
            (repo_root / "schemas" / "archive-report.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(document, schema)
        assert document["manifest_sha256"] == DIGEST_A

    def test_the_report_is_where_the_manifests_own_hash_lives(self) -> None:
        """ADR-0003's fixed point, one level up: the manifest cannot contain its own hash."""
        report = ArchiveReport.model_validate(self.base(manifest_sha256=DIGEST_A))
        assert report.manifest_sha256 == DIGEST_A
        assert "manifest_sha256" not in manifest().model_dump(mode="json")


class TestSingleWriterLock:
    def test_it_lives_outside_every_session(self, tmp_path: Path) -> None:
        """A lock inside a source root would itself violate INV-01."""
        path = lock_path("session-1", directory=tmp_path)
        assert path.parent == tmp_path
        assert path.name.endswith(".upload.lock")

    def test_it_is_stable_for_a_session_and_distinct_between_them(self, tmp_path: Path) -> None:
        assert lock_path("a", directory=tmp_path) == lock_path("a", directory=tmp_path)
        assert lock_path("a", directory=tmp_path) != lock_path("b", directory=tmp_path)

    def test_a_session_id_with_a_slash_cannot_name_a_lock_elsewhere(self, tmp_path: Path) -> None:
        path = lock_path("a/b", directory=tmp_path)
        assert path.parent == tmp_path
        assert "/" not in path.name

    def test_a_very_long_session_id_still_produces_a_usable_filename(self, tmp_path: Path) -> None:
        """Named by digest, because an encoded long id blows past the 255-byte limit.

        The upload-id record had the same defect against object keys, where it was worse:
        multipart failed before its first part on exactly the long paths the key limit
        permits. Found by M7a's code review.
        """
        path = lock_path("s" * 900, directory=tmp_path)
        assert len(path.name.encode("utf-8")) < 255
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert path.is_file()

    def test_it_is_held_and_released(self, tmp_path: Path) -> None:
        with single_writer("s", directory=tmp_path):
            pass
        with single_writer("s", directory=tmp_path):
            pass

    def test_two_real_processes_cannot_both_hold_it(self, tmp_path: Path) -> None:
        """Two processes, not one process asserting about itself.

        `flock` is per open file description, so a same-process test can pass while the
        cross-process guarantee — the only one that matters — is absent entirely.
        """
        script = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from dnd_audio.archive.lock import single_writer\n"
            "with single_writer('s', directory=Path(sys.argv[1])):\n"
            "    Path(sys.argv[2]).write_text('held')\n"
            "    time.sleep(float(sys.argv[3]))\n"
        )
        marker = tmp_path / "held.marker"
        holder = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(marker), "10"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = 30.0
            waited = 0.0
            while not marker.exists() and waited < deadline:
                if holder.poll() is not None:
                    pytest.fail(f"the holder exited early: {holder.communicate()}")
                waited += 0.05
                _sleep(0.05)
            assert marker.exists(), "the first process never acquired the lock"

            with pytest.raises(ArchiveError) as caught, single_writer("s", directory=tmp_path):
                pass
            assert caught.value.code == "archive_upload_in_progress"
        finally:
            holder.kill()
            holder.wait(timeout=30)

    def test_a_different_session_is_not_blocked(self, tmp_path: Path) -> None:
        """The lock is per session, so archiving two sessions at once is fine."""
        with single_writer("one", directory=tmp_path), single_writer("two", directory=tmp_path):
            pass


def _sleep(seconds: float) -> None:
    """Named so the polling loop above reads as waiting on a condition, not as a delay."""
    import time

    time.sleep(seconds)
