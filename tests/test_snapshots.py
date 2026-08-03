"""The snapshot half of the model store: pinning a repository at one commit (ADR-0027).

Every test here builds its own tiny snapshot rather than touching the real ones. The two
pinned descriptors are six gigabytes between them, and a suite that needed them would need
a GPU host to run at all (INV-05). What is asserted about the *real* descriptors is only
what can be checked without their bytes — that the manifests are well-formed, canonical,
and pinned to commits rather than to names.

The negative cases carry the weight. `verify_snapshot` returning `None` on a good tree
proves very little; what matters is that it refuses a truncated shard, a substituted file,
an unpinned extra, and a revision nothing installed — because each of those is a way a
model runs on bytes nobody checked and produces a slightly wrong transcript instead of an
error.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from dnd_audio.models import (
    LOCK_VERSION,
    MODEL_HASH_MISMATCH,
    MODEL_LOCK_FILENAME,
    MODEL_REVISION_NOT_INSTALLED,
    MODEL_SIZE_MISMATCH,
    MODEL_UNAVAILABLE,
    MODEL_UNPINNED_FILE,
    QWEN3_ALIGNER,
    QWEN3_ASR,
    QWEN_SNAPSHOTS,
    REVISION_PATTERN,
    SILERO_VAD,
    SNAPSHOT_FETCH_COMMAND,
    ModelError,
    SnapshotDescriptor,
    SnapshotDownloader,
    SnapshotFile,
    find_snapshot,
    install_snapshot,
    lock_path,
    lock_record,
    measure_snapshot,
    read_lock,
    read_snapshots,
    record_snapshot_in_lock,
    require_snapshot,
    snapshot_dir,
    snapshot_lock_record,
    snapshot_manifest,
    snapshot_present,
    verify_snapshot,
    write_lock,
)

#: A second commit, used wherever a test needs a revision that is not the descriptor's.
#: Forty hex characters, because that is the only shape this project accepts.
OTHER_REVISION = "b" * 40

_CONTENTS = {
    "config.json": b'{"model_type": "toy"}',
    "model.safetensors": b"\x00\x01\x02\x03" * 64,
    "nested/tokenizer.json": b'{"vocab": {}}',
}


def _describe(contents: dict[str, bytes], *, revision: str = "a" * 40) -> SnapshotDescriptor:
    """A descriptor whose manifest is the truth about ``contents``."""
    return SnapshotDescriptor(
        key="toy-model",
        repository="Toy/Toy-Model",
        revision=revision,
        files=tuple(
            SnapshotFile(path, len(body), hashlib.sha256(body).hexdigest())
            for path, body in sorted(contents.items())
        ),
    )


def _materialize(root: Path, contents: dict[str, bytes]) -> None:
    for name, body in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """An empty models directory."""
    directory = tmp_path / "models"
    directory.mkdir()
    return directory


@pytest.fixture
def toy(store: Path) -> SnapshotDescriptor:
    """A descriptor with its snapshot already correctly installed."""
    descriptor = _describe(_CONTENTS)
    _materialize(snapshot_dir(descriptor, directory=store), _CONTENTS)
    return descriptor


class TestThePinnedDescriptors:
    """What can be asserted about the real snapshots without downloading them."""

    @pytest.mark.parametrize("descriptor", QWEN_SNAPSHOTS, ids=lambda d: d.key)
    def test_the_revision_is_a_commit_and_not_a_name(self, descriptor: SnapshotDescriptor) -> None:
        """A branch is a moving pointer; the spec forbids resolving one during `process`."""
        assert re.match(REVISION_PATTERN, descriptor.revision), descriptor.revision

    @pytest.mark.parametrize("descriptor", QWEN_SNAPSHOTS, ids=lambda d: d.key)
    def test_the_manifest_is_canonical_and_free_of_duplicates(
        self, descriptor: SnapshotDescriptor
    ) -> None:
        paths = [entry.path for entry in descriptor.files]
        assert paths == sorted(paths)
        assert len(paths) == len(set(paths))

    @pytest.mark.parametrize("descriptor", QWEN_SNAPSHOTS, ids=lambda d: d.key)
    def test_every_digest_is_lowercase_hex_and_every_size_is_positive(
        self, descriptor: SnapshotDescriptor
    ) -> None:
        for entry in descriptor.files:
            assert re.match(r"^[0-9a-f]{64}$", entry.sha256), entry
            assert entry.size_bytes > 0, entry

    @pytest.mark.parametrize("descriptor", QWEN_SNAPSHOTS, ids=lambda d: d.key)
    def test_no_manifest_path_escapes_the_snapshot(self, descriptor: SnapshotDescriptor) -> None:
        """The manifest decides which files get opened, so it is data that is validated."""
        for entry in descriptor.files:
            assert not Path(entry.path).is_absolute(), entry
            assert ".." not in Path(entry.path).parts, entry

    @pytest.mark.parametrize("descriptor", QWEN_SNAPSHOTS, ids=lambda d: d.key)
    def test_the_weights_are_pinned_not_only_the_configuration(
        self, descriptor: SnapshotDescriptor
    ) -> None:
        """The failure this guards is a manifest of small JSON files that verifies while
        the multi-gigabyte tensors beside it are whatever happens to be on disk."""
        weights = [entry for entry in descriptor.files if entry.path.endswith(".safetensors")]
        assert weights, descriptor.key
        assert sum(entry.size_bytes for entry in weights) > 1_000_000_000

    def test_the_two_models_are_distinct_repositories_at_distinct_commits(self) -> None:
        assert QWEN3_ASR.key != QWEN3_ALIGNER.key
        assert QWEN3_ASR.repository != QWEN3_ALIGNER.repository
        assert QWEN3_ASR.revision != QWEN3_ALIGNER.revision

    def test_readme_and_gitattributes_are_deliberately_unpinned(self) -> None:
        """They are not downloaded and must not be, because an unpinned file in the tree
        is a verification failure — see `TestVerification`."""
        for descriptor in QWEN_SNAPSHOTS:
            names = {entry.path for entry in descriptor.files}
            assert "README.md" not in names
            assert ".gitattributes" not in names


class TestTheDirectoryLayout:
    def test_the_directory_is_keyed_by_commit(self, store: Path) -> None:
        """Two revisions of one model must not share a tree: verifying against one and
        loading the other's leftovers would report the revision that was asked for."""
        descriptor = _describe(_CONTENTS)
        default = snapshot_dir(descriptor, directory=store)
        other = snapshot_dir(descriptor, revision=OTHER_REVISION, directory=store)

        assert default != other
        assert default.name == descriptor.revision
        assert other.name == OTHER_REVISION
        assert default.parent == other.parent == store / descriptor.key

    def test_snapshots_live_outside_any_session(self, store: Path) -> None:
        assert store in snapshot_dir(_describe(_CONTENTS), directory=store).parents


class TestVerification:
    """The reasons a tree is refused. Each one is a way a model runs on unchecked bytes."""

    def test_a_correct_tree_verifies(self, toy: SnapshotDescriptor, store: Path) -> None:
        assert verify_snapshot(toy, directory=store) is None
        assert find_snapshot(toy, directory=store) == snapshot_dir(toy, directory=store)

    def test_an_absent_directory_is_refused(self, store: Path) -> None:
        descriptor = _describe(_CONTENTS)
        assert verify_snapshot(descriptor, directory=store) is not None
        assert find_snapshot(descriptor, directory=store) is None

    def test_a_missing_file_is_refused(self, toy: SnapshotDescriptor, store: Path) -> None:
        (snapshot_dir(toy, directory=store) / "config.json").unlink()
        assert "config.json" in (verify_snapshot(toy, directory=store) or "")
        assert find_snapshot(toy, directory=store) is None

    def test_a_truncated_file_is_refused(self, toy: SnapshotDescriptor, store: Path) -> None:
        """The interrupted-download case: a plausible-looking file at the right path."""
        path = snapshot_dir(toy, directory=store) / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        reason = verify_snapshot(toy, directory=store) or ""
        assert "bytes," in reason
        assert find_snapshot(toy, directory=store) is None

    def test_a_substituted_file_of_the_right_size_is_refused(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """Where the size check alone would pass and only the digest can see it."""
        path = snapshot_dir(toy, directory=store) / "model.safetensors"
        path.write_bytes(b"\xff" * len(path.read_bytes()))
        assert "does not hash" in (verify_snapshot(toy, directory=store) or "")
        assert find_snapshot(toy, directory=store) is None

    def test_an_unpinned_file_is_refused(self, toy: SnapshotDescriptor, store: Path) -> None:
        """Transformers loads a directory, not a manifest. Every pinned file being correct
        does not make the tree correct."""
        (snapshot_dir(toy, directory=store) / "chat_template.json").write_text("{}")
        assert "is not pinned" in (verify_snapshot(toy, directory=store) or "")
        assert find_snapshot(toy, directory=store) is None

    def test_an_unpinned_file_in_a_subdirectory_is_refused(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """`hf download --local-dir` writes `.cache/huggingface` metadata into its target,
        so the walk has to be recursive rather than one level deep."""
        stray = snapshot_dir(toy, directory=store) / ".cache" / "huggingface" / "note.json"
        stray.parent.mkdir(parents=True)
        stray.write_text("{}")
        assert "is not pinned" in (verify_snapshot(toy, directory=store) or "")

    def test_an_unpinned_symlinked_directory_is_refused(
        self, toy: SnapshotDescriptor, store: Path, tmp_path: Path
    ) -> None:
        """The hole the two-directional check had, found by M6b's verify phase.

        A symlink *to a directory* answers `is_dir()` truthfully, so testing `is_dir()`
        alone skipped it — and `rglob` does not descend into it, so its contents were never
        walked either. The result was that a plain unpinned file was refused while a
        symlinked directory holding a whole second model went through, which is the exact
        inverse of what the rule claims. Not hypothetical: `hf download --local-dir` is a
        tool that has created symlinks into a shared cache, and Transformers loads a
        *directory*, so anything reachable inside one is a file a model may read.
        """
        elsewhere = tmp_path / "outside"
        elsewhere.mkdir()
        (elsewhere / "other.safetensors").write_bytes(b"\xff" * 32)
        (snapshot_dir(toy, directory=store) / "extra").symlink_to(
            elsewhere, target_is_directory=True
        )

        assert "is not pinned" in (verify_snapshot(toy, directory=store) or "")
        assert find_snapshot(toy, directory=store) is None

    def test_an_unpinned_symlinked_file_is_refused_too(
        self, toy: SnapshotDescriptor, store: Path, tmp_path: Path
    ) -> None:
        """The same rule, stated for the simpler case so the fix cannot regress to
        "symlinks are always fine"."""
        target = tmp_path / "stray.bin"
        target.write_bytes(b"\x00")
        (snapshot_dir(toy, directory=store) / "stray.bin").symlink_to(target)

        assert "is not pinned" in (verify_snapshot(toy, directory=store) or "")

    def test_a_pinned_file_may_still_be_a_symlink_to_the_right_bytes(
        self, toy: SnapshotDescriptor, store: Path, tmp_path: Path
    ) -> None:
        """Content is the rule, not inode identity. `hf` has stored a snapshot as symlinks
        into a shared blob cache, and a manifest file whose bytes hash correctly is the
        pinned artifact however it got there — refusing it would break a supported layout
        for no gain."""
        root = snapshot_dir(toy, directory=store)
        body = (root / "config.json").read_bytes()
        blob = tmp_path / "blob"
        blob.write_bytes(body)
        (root / "config.json").unlink()
        (root / "config.json").symlink_to(blob)

        assert verify_snapshot(toy, directory=store) is None

    def test_an_empty_directory_beside_the_files_is_not_a_failure(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """Only files are unpinned content. A directory holds nothing a model can read."""
        (snapshot_dir(toy, directory=store) / "empty").mkdir()
        assert verify_snapshot(toy, directory=store) is None

    def test_a_manifest_path_that_escapes_the_snapshot_is_refused(self, store: Path) -> None:
        """A lock is a file a person can edit, and the manifest decides what gets opened."""
        descriptor = SnapshotDescriptor(
            key="toy-model",
            repository="Toy/Toy-Model",
            revision="a" * 40,
            files=(SnapshotFile("../escape.txt", 1, "00" * 32),),
        )
        snapshot_dir(descriptor, directory=store).mkdir(parents=True)
        assert "relative path" in (verify_snapshot(descriptor, directory=store) or "")

    def test_an_unknown_revision_is_refused(self, toy: SnapshotDescriptor, store: Path) -> None:
        """Nothing knows what this commit should contain — not the build, not the lock."""
        reason = verify_snapshot(toy, revision=OTHER_REVISION, directory=store) or ""
        assert "no manifest" in reason


class TestRequireSnapshot:
    """The fatal presentation of the same verification, with actionable codes."""

    def test_it_returns_the_directory_when_the_tree_is_good(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        assert require_snapshot(toy, directory=store) == snapshot_dir(toy, directory=store)

    @pytest.mark.parametrize(
        ("damage", "code"),
        [
            ("absent", MODEL_UNAVAILABLE),
            ("truncated", MODEL_SIZE_MISMATCH),
            ("substituted", MODEL_HASH_MISMATCH),
            ("unpinned", MODEL_UNPINNED_FILE),
        ],
    )
    def test_each_failure_carries_its_own_code(
        self, toy: SnapshotDescriptor, store: Path, damage: str, code: str
    ) -> None:
        """Codes are the stable part of an error; prose gets reworded. A caller deciding
        whether to suggest a re-download needs to tell these four apart."""
        root = snapshot_dir(toy, directory=store)
        weights = root / "model.safetensors"
        if damage == "absent":
            weights.unlink()
        elif damage == "truncated":
            weights.write_bytes(weights.read_bytes()[:-4])
        elif damage == "substituted":
            weights.write_bytes(b"\xff" * len(weights.read_bytes()))
        else:
            (root / "extra.json").write_text("{}")

        with pytest.raises(ModelError) as caught:
            require_snapshot(toy, directory=store)
        assert caught.value.code == code

    def test_an_uninstalled_revision_is_told_apart_from_an_absent_model(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """The store is healthy at another commit, so "you never fetched it" is wrong and
        the operator needs to know which revision is missing."""
        with pytest.raises(ModelError) as caught:
            require_snapshot(toy, revision=OTHER_REVISION, directory=store)
        assert caught.value.code == MODEL_REVISION_NOT_INSTALLED
        assert OTHER_REVISION in str(caught.value)

    def test_every_failure_names_the_command_that_fixes_it(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        (snapshot_dir(toy, directory=store) / "config.json").unlink()
        with pytest.raises(ModelError) as caught:
            require_snapshot(toy, directory=store)
        assert SNAPSHOT_FETCH_COMMAND in str(caught.value)


class TestSnapshotPresent:
    """`doctor`'s cheap check, and the reason it is a separate function."""

    def test_it_is_true_for_a_correct_tree(self, toy: SnapshotDescriptor, store: Path) -> None:
        assert snapshot_present(toy, directory=store) is True

    def test_it_is_false_when_a_file_is_missing(self, toy: SnapshotDescriptor, store: Path) -> None:
        (snapshot_dir(toy, directory=store) / "config.json").unlink()
        assert snapshot_present(toy, directory=store) is False

    def test_it_does_not_hash_and_says_so_by_disagreeing_with_find_snapshot(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """The one behaviour that makes this fast, asserted rather than described.

        A substituted file of the right size is *present* and is not *usable*. That is why
        `doctor` may call this and why nothing that loads a model may: hashing six
        gigabytes is a minute `doctor` should not spend and a minute an ASR run must.
        """
        path = snapshot_dir(toy, directory=store) / "model.safetensors"
        path.write_bytes(b"\xff" * len(path.read_bytes()))

        assert snapshot_present(toy, directory=store) is True
        assert find_snapshot(toy, directory=store) is None


class TestTheLock:
    def test_a_recorded_snapshot_reads_back(self, toy: SnapshotDescriptor, store: Path) -> None:
        record_snapshot_in_lock(toy, directory=store)
        records = read_snapshots(directory=store)
        assert records == {toy.key: snapshot_lock_record(toy)}

    def test_recording_a_snapshot_leaves_the_models_section_alone(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """Silero, the ASR model and the aligner share one lock. M3's merge rule, applied
        across the two sections rather than only within one."""
        write_lock({SILERO_VAD.key: lock_record(SILERO_VAD)}, directory=store)
        record_snapshot_in_lock(toy, directory=store)

        assert set(read_lock(directory=store)) == {SILERO_VAD.key}
        assert set(read_snapshots(directory=store)) == {toy.key}

    def test_recording_a_model_leaves_the_snapshots_section_alone(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """The other direction, which is the one that breaks if `_record_in_lock` forgets
        to pass the snapshots through."""
        record_snapshot_in_lock(toy, directory=store)
        write_lock(
            {SILERO_VAD.key: lock_record(SILERO_VAD)},
            snapshots=read_snapshots(directory=store),
            directory=store,
        )
        assert set(read_snapshots(directory=store)) == {toy.key}

    def test_two_snapshots_coexist(self, toy: SnapshotDescriptor, store: Path) -> None:
        other = SnapshotDescriptor(
            key="other-model", repository="Toy/Other", revision="c" * 40, files=toy.files
        )
        record_snapshot_in_lock(toy, directory=store)
        record_snapshot_in_lock(other, directory=store)
        assert set(read_snapshots(directory=store)) == {toy.key, other.key}

    def test_a_version_one_lock_reads_as_no_lock(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """The bump's cost, stated: one re-verification of files already on disk."""
        lock_path(directory=store).write_text(
            json.dumps({"lock_version": 1, "snapshots": {toy.key: snapshot_lock_record(toy)}}),
            encoding="utf-8",
        )
        assert read_snapshots(directory=store) == {}

    def test_an_entry_missing_a_required_field_is_dropped(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        broken = snapshot_lock_record(toy)
        del broken["revision"]
        lock_path(directory=store).write_text(
            json.dumps({"lock_version": LOCK_VERSION, "snapshots": {toy.key: broken}}),
            encoding="utf-8",
        )
        assert read_snapshots(directory=store) == {}

    def test_the_record_is_canonical(self, toy: SnapshotDescriptor, store: Path) -> None:
        """INV-02: the lock is rewritten on every fetch and must not churn."""
        shuffled = SnapshotDescriptor(
            key=toy.key,
            repository=toy.repository,
            revision=toy.revision,
            files=tuple(reversed(toy.files)),
        )
        assert snapshot_lock_record(shuffled) == snapshot_lock_record(toy)


class TestAConfiguredRevisionVerifiesAgainstTheLock:
    """ADR-0027's asymmetry, which is the subtlest thing in this module.

    A checked-in manifest cannot describe a commit it was not written for. So for an
    overridden revision the lock recorded at install time is the manifest, and that makes
    the lock authoritative here in a way it deliberately is not for `find_model`.
    """

    def test_the_default_revision_uses_the_checked_in_manifest(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        assert snapshot_manifest(toy, directory=store) == toy.files

    def test_an_override_with_no_lock_entry_has_no_manifest(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        assert snapshot_manifest(toy, revision=OTHER_REVISION, directory=store) is None

    def test_an_override_verifies_against_what_the_lock_recorded(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        contents = dict(_CONTENTS, **{"config.json": b'{"model_type": "other"}'})
        installed = _describe(contents, revision=OTHER_REVISION)
        _materialize(snapshot_dir(toy, revision=OTHER_REVISION, directory=store), contents)
        record_snapshot_in_lock(
            toy, revision=OTHER_REVISION, files=installed.files, directory=store
        )

        assert verify_snapshot(toy, revision=OTHER_REVISION, directory=store) is None
        assert find_snapshot(toy, revision=OTHER_REVISION, directory=store) is not None
        # And the default revision is untouched by any of it.
        assert verify_snapshot(toy, directory=store) is None

    def test_a_lock_entry_for_a_different_revision_does_not_authorize_this_one(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """The failure that would make the override path meaningless: any lock entry at
        all being taken as permission for any revision."""
        record_snapshot_in_lock(toy, directory=store)  # records the *default* revision
        assert snapshot_manifest(toy, revision=OTHER_REVISION, directory=store) is None

    def test_an_override_whose_bytes_disagree_with_the_lock_is_refused(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        installed = _describe(_CONTENTS, revision=OTHER_REVISION)
        root = snapshot_dir(toy, revision=OTHER_REVISION, directory=store)
        _materialize(root, _CONTENTS)
        record_snapshot_in_lock(
            toy, revision=OTHER_REVISION, files=installed.files, directory=store
        )
        (root / "config.json").write_bytes(b'{"model_type": "tampered!!"}')

        assert verify_snapshot(toy, revision=OTHER_REVISION, directory=store) is not None


class TestInstallation:
    """`install_snapshot`, driven through its real body with a fake `hf` (INV-05).

    The seam is the download, not the installer, for the same reason `silero.py` puts its
    seam at the ONNX session rather than at the detector: everything worth asserting here
    — that staging keeps a failed download out of the tree, that only manifest files move,
    that verification happens before the lock is written — is behaviour of the code below
    the seam, and a fake installer would replace it.
    """

    @staticmethod
    def _downloader(
        contents: dict[str, bytes], *, extras: dict[str, bytes] | None = None
    ) -> tuple[SnapshotDownloader, list[tuple[str, str]]]:
        """A fake `hf download` that writes ``contents`` plus repository furniture."""
        calls: list[tuple[str, str]] = []

        def download(repository: str, revision: str, target: Path) -> None:
            calls.append((repository, revision))
            _materialize(target, {**contents, **(extras or {})})

        return download, calls

    def test_it_downloads_verifies_moves_and_records(self, store: Path) -> None:
        descriptor = _describe(_CONTENTS)
        download, calls = self._downloader(_CONTENTS)

        path, downloaded = install_snapshot(descriptor, directory=store, download=download)

        assert downloaded is True
        assert calls == [(descriptor.repository, descriptor.revision)]
        assert path == snapshot_dir(descriptor, directory=store)
        assert verify_snapshot(descriptor, directory=store) is None
        assert read_snapshots(directory=store) == {descriptor.key: snapshot_lock_record(descriptor)}

    def test_an_already_verified_snapshot_is_not_downloaded_again(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """What makes this safe to re-run, and therefore usable as "am I set up?"."""
        download, calls = self._downloader(_CONTENTS)
        path, downloaded = install_snapshot(toy, directory=store, download=download)

        assert downloaded is False
        assert calls == []
        assert path == snapshot_dir(toy, directory=store)

    def test_it_repairs_a_deleted_lock_without_downloading(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """`fetch` has done this for Silero since M3; the snapshot half did not until M6b's
        verify phase.

        The early return above skipped straight past the lock, so deleting `models.lock.json`
        and re-running `models fetch --qwen` reported a lock path it had not written and left
        the snapshot section missing — for a six-gigabyte download whose whole point is that
        it is cheap to re-run as an "am I set up?" check. Raised by M6b's code review.
        """
        # The `toy` fixture materializes correct bytes and no lock, which *is* the state
        # a deleted lock leaves behind.
        assert not (store / MODEL_LOCK_FILENAME).exists()

        download, calls = self._downloader(_CONTENTS)
        _, downloaded = install_snapshot(toy, directory=store, download=download)

        assert downloaded is False, "verified bytes must not be downloaded again"
        assert calls == [], "repairing a lock must not reach the network"
        assert read_snapshots(directory=store) == {toy.key: snapshot_lock_record(toy)}

    def test_repository_furniture_never_reaches_the_snapshot(self, store: Path) -> None:
        """`hf` fetches the whole repository; a verified tree holds only model files.

        Without this the unpinned-file rule and the download would contradict each other
        on the very first fetch — README.md would arrive and then fail verification.
        """
        descriptor = _describe(_CONTENTS)
        download, _ = self._downloader(
            _CONTENTS,
            extras={
                "README.md": b"# Toy",
                ".gitattributes": b"*.safetensors filter=lfs\n",
                ".cache/huggingface/download/config.json.metadata": b"{}",
            },
        )

        path, _ = install_snapshot(descriptor, directory=store, download=download)

        assert verify_snapshot(descriptor, directory=store) is None
        assert not (path / "README.md").exists()
        assert not (path / ".gitattributes").exists()
        assert not (path / ".cache").exists()

    def test_a_file_upstream_grew_does_not_smuggle_itself_in(self, store: Path) -> None:
        """The pinned revision moves *the manifest's* files, not what happens to arrive."""
        descriptor = _describe(_CONTENTS)
        download, _ = self._downloader(_CONTENTS, extras={"surprise.py": b"import os\n"})

        path, _ = install_snapshot(descriptor, directory=store, download=download)

        assert not (path / "surprise.py").exists()
        assert verify_snapshot(descriptor, directory=store) is None

    def test_a_failed_download_leaves_no_tree_and_no_lock_entry(self, store: Path) -> None:
        descriptor = _describe(_CONTENTS)

        def download(repository: str, revision: str, target: Path) -> None:
            _materialize(target, {"config.json": _CONTENTS["config.json"]})
            raise ModelError("hf exited 1", code=MODEL_UNAVAILABLE)

        with pytest.raises(ModelError):
            install_snapshot(descriptor, directory=store, download=download)

        assert find_snapshot(descriptor, directory=store) is None
        assert read_snapshots(directory=store) == {}

    def test_a_download_missing_a_pinned_file_is_fatal_and_records_nothing(
        self, store: Path
    ) -> None:
        descriptor = _describe(_CONTENTS)
        partial = {k: v for k, v in _CONTENTS.items() if k != "model.safetensors"}
        download, _ = self._downloader(partial)

        with pytest.raises(ModelError) as caught:
            install_snapshot(descriptor, directory=store, download=download)

        assert "model.safetensors" in str(caught.value)
        assert read_snapshots(directory=store) == {}

    def test_wrong_bytes_are_fatal_and_the_lock_records_nothing(self, store: Path) -> None:
        """The lock must never vouch for bytes nobody checked — INV-08's rule, one level
        up from a cache entry. Verification happens before the record is written."""
        descriptor = _describe(_CONTENTS)
        tampered = dict(
            _CONTENTS, **{"model.safetensors": b"\xff" * len(_CONTENTS["model.safetensors"])}
        )
        download, _ = self._downloader(tampered)

        with pytest.raises(ModelError) as caught:
            install_snapshot(descriptor, directory=store, download=download)

        assert caught.value.code == MODEL_HASH_MISMATCH
        assert read_snapshots(directory=store) == {}
        assert find_snapshot(descriptor, directory=store) is None

    def test_a_stale_tree_is_replaced_rather_than_merged_into(
        self, toy: SnapshotDescriptor, store: Path
    ) -> None:
        """A leftover file from an older install would fail the unpinned-file rule
        forever, so installing has to clear the target rather than write over it."""
        (snapshot_dir(toy, directory=store) / "leftover.bin").write_bytes(b"old")
        download, _ = self._downloader(_CONTENTS)

        path, downloaded = install_snapshot(toy, directory=store, download=download)

        assert downloaded is True
        assert not (path / "leftover.bin").exists()
        assert verify_snapshot(toy, directory=store) is None

    def test_an_overridden_revision_records_what_arrived(self, store: Path) -> None:
        """Nothing in this build knows what that commit contains, so the lock becomes its
        manifest — and it is written from measured bytes, after they verified."""
        descriptor = _describe(_CONTENTS)
        contents = dict(_CONTENTS, **{"config.json": b'{"model_type": "newer"}'})
        download, calls = self._downloader(contents, extras={"README.md": b"# Toy"})

        path, downloaded = install_snapshot(
            descriptor, revision=OTHER_REVISION, directory=store, download=download
        )

        assert downloaded is True
        assert calls == [(descriptor.repository, OTHER_REVISION)]
        assert path.name == OTHER_REVISION
        assert verify_snapshot(descriptor, revision=OTHER_REVISION, directory=store) is None
        recorded = read_snapshots(directory=store)[descriptor.key]
        assert recorded["revision"] == OTHER_REVISION
        assert {row["path"] for row in recorded["files"]} == set(contents)

    def test_installing_a_second_snapshot_leaves_the_first(self, store: Path) -> None:
        first = _describe(_CONTENTS)
        second = SnapshotDescriptor(
            key="other-model",
            repository="Toy/Other",
            revision="c" * 40,
            files=first.files,
        )
        download, _ = self._downloader(_CONTENTS)

        install_snapshot(first, directory=store, download=download)
        install_snapshot(second, directory=store, download=download)

        assert verify_snapshot(first, directory=store) is None
        assert verify_snapshot(second, directory=store) is None
        assert set(read_snapshots(directory=store)) == {first.key, second.key}


class TestMeasureSnapshot:
    def test_it_skips_repository_furniture_and_hf_bookkeeping(self, tmp_path: Path) -> None:
        _materialize(
            tmp_path,
            {
                **_CONTENTS,
                "README.md": b"# Toy",
                ".gitattributes": b"lfs\n",
                ".cache/huggingface/x.metadata": b"{}",
            },
        )
        assert {entry.path for entry in measure_snapshot(tmp_path)} == set(_CONTENTS)

    def test_it_measures_what_is_actually_there(self, tmp_path: Path) -> None:
        _materialize(tmp_path, _CONTENTS)
        measured = {entry.path: entry for entry in measure_snapshot(tmp_path)}
        body = _CONTENTS["config.json"]
        assert measured["config.json"].size_bytes == len(body)
        assert measured["config.json"].sha256 == hashlib.sha256(body).hexdigest()

    def test_it_is_canonical(self, tmp_path: Path) -> None:
        _materialize(tmp_path, _CONTENTS)
        paths = [entry.path for entry in measure_snapshot(tmp_path)]
        assert paths == sorted(paths)
