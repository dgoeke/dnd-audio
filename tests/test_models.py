"""The model store refuses anything it cannot prove is the pinned artifact.

Every test here is offline except one, and that is the point rather than a convenience:
`fetch` takes an injected downloader, so the code path that would open a socket is
exercised without one (INV-05). The single exception is marked ``allow_network`` and
excluded from the gate — it is the only way to find out that the pinned URL still serves
the pinned bytes, and that question cannot be answered from a fixture.

The tests are written against a *stand-in* descriptor of thirty-odd bytes rather than
the real 2.3 MB model, because the interesting behaviour is "what does it do when the
bytes are wrong", and no fixture can hold the right ones. The real pin is checked
separately, for shape here and for content over the network.

Nothing here reads or writes the invoking user's cache directory: every test either
overrides ``DND_AUDIO_MODELS_DIR`` or passes an explicit directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from dnd_audio import models
from dnd_audio.determinism import sha256_bytes
from dnd_audio.models import (
    LOCK_VERSION,
    MODEL_HASH_MISMATCH,
    MODEL_LOCK_FILENAME,
    MODEL_SIZE_MISMATCH,
    MODEL_UNAVAILABLE,
    SILERO_VAD,
    ModelDescriptor,
    ModelError,
    default_download,
    fetch,
    find_model,
    lock_path,
    lock_record,
    model_path,
    models_dir,
    read_lock,
    require_model,
    write_lock,
)

#: Stands in for an ONNX graph. Its length and digest are what the fake descriptor pins,
#: so "the bytes are right" and "the bytes are wrong" are both expressible.
PAYLOAD: Final = b"pretend this is an ONNX graph\n"

#: A substitute of exactly the same length. A size check alone cannot tell this from the
#: real thing, which is why there is also a digest.
SUBSTITUTE: Final = b"x" * len(PAYLOAD)


class RecordingDownloader:
    """A downloader that answers with fixed bytes and remembers being asked.

    The memory is the assertion: "already present" is only meaningful if it can be shown
    that nothing was fetched, and a downloader that merely returns the right bytes would
    let a re-download pass unnoticed.
    """

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        return self.payload


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A models directory that does not exist yet. `fetch` must create it."""
    return tmp_path / "models"


@pytest.fixture
def descriptor() -> ModelDescriptor:
    """The real pin's shape at 1/70000th of its size."""
    commit = "0" * 40
    return ModelDescriptor(
        key="fake-vad",
        filename="fake_vad.onnx",
        repository="example/fake-vad",
        release="v0.0.1",
        commit=commit,
        path_in_repository="data/fake_vad.onnx",
        url=f"https://raw.githubusercontent.com/example/fake-vad/{commit}/data/fake_vad.onnx",
        size_bytes=len(PAYLOAD),
        sha256=sha256_bytes(PAYLOAD),
    )


class TestModelsDir:
    """Resolution order, proved without going anywhere near the real cache."""

    def test_the_explicit_override_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path / "elsewhere"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert models_dir() == tmp_path / "elsewhere"

    def test_falls_back_to_xdg_cache_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DND_AUDIO_MODELS_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert models_dir() == tmp_path / "cache" / "dnd-audio" / "models"

    def test_falls_back_to_the_home_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DND_AUDIO_MODELS_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert models_dir() == tmp_path / "home" / ".cache" / "dnd-audio" / "models"

    def test_an_empty_override_counts_as_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`DND_AUDIO_MODELS_DIR=` must not resolve to the working directory."""
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", "")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert models_dir() == tmp_path / "cache" / "dnd-audio" / "models"

    def test_model_path_is_inside_the_models_directory(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        assert model_path(descriptor, directory=store) == store / descriptor.filename


class TestFetch:
    def test_writes_and_verifies(self, descriptor: ModelDescriptor, store: Path) -> None:
        downloader = RecordingDownloader(PAYLOAD)

        path = fetch(descriptor, download=downloader, directory=store)

        assert downloader.calls == [descriptor.url]
        assert path == store / descriptor.filename
        assert path.read_bytes() == PAYLOAD
        assert find_model(descriptor, directory=store) == path

    def test_records_the_lock(self, descriptor: ModelDescriptor, store: Path) -> None:
        fetch(descriptor, download=RecordingDownloader(PAYLOAD), directory=store)

        records = read_lock(directory=store)
        assert set(records) == {descriptor.key}
        entry = records[descriptor.key]
        assert entry["commit"] == descriptor.commit
        assert entry["release"] == descriptor.release
        assert entry["sha256"] == descriptor.sha256
        assert entry["size_bytes"] == descriptor.size_bytes
        assert entry["url"] == descriptor.url

    def test_a_hash_mismatch_is_fatal_and_leaves_nothing_behind(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        """The test that matters most.

        A right-sized file with the wrong contents is exactly what a substituted or
        corrupted artifact looks like, and if it survives the failed fetch then the next
        run finds a model at the expected path and loads it.
        """
        downloader = RecordingDownloader(SUBSTITUTE)

        with pytest.raises(ModelError) as raised:
            fetch(descriptor, download=downloader, directory=store)

        assert raised.value.code == MODEL_HASH_MISMATCH
        assert not model_path(descriptor, directory=store).exists()
        assert not lock_path(directory=store).exists()
        assert find_model(descriptor, directory=store) is None

    def test_a_size_mismatch_is_caught_on_its_own(self, store: Path) -> None:
        """Even when the digest would have matched the bytes that arrived.

        This descriptor pins the payload's own digest but the wrong length, so a
        check that hashed first and never compared sizes would accept it.
        """
        commit = "1" * 40
        wrong_size = ModelDescriptor(
            key="fake-vad",
            filename="fake_vad.onnx",
            repository="example/fake-vad",
            release="v0.0.1",
            commit=commit,
            path_in_repository="data/fake_vad.onnx",
            url=f"https://raw.githubusercontent.com/example/fake-vad/{commit}/data/fake_vad.onnx",
            size_bytes=len(PAYLOAD) + 1,
            sha256=sha256_bytes(PAYLOAD),
        )

        with pytest.raises(ModelError) as raised:
            fetch(wrong_size, download=RecordingDownloader(PAYLOAD), directory=store)

        assert raised.value.code == MODEL_SIZE_MISMATCH
        assert not model_path(wrong_size, directory=store).exists()

    def test_a_present_model_is_not_downloaded_again(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        downloader = RecordingDownloader(PAYLOAD)
        fetch(descriptor, download=downloader, directory=store)

        again = fetch(descriptor, download=downloader, directory=store)

        assert downloader.calls == [descriptor.url]
        assert again == model_path(descriptor, directory=store)

    def test_a_corrupted_model_is_replaced(self, descriptor: ModelDescriptor, store: Path) -> None:
        """Present-but-wrong is absence, so a fetch over it is a real fetch."""
        store.mkdir(parents=True)
        (store / descriptor.filename).write_bytes(SUBSTITUTE)
        downloader = RecordingDownloader(PAYLOAD)

        path = fetch(descriptor, download=downloader, directory=store)

        assert downloader.calls == [descriptor.url]
        assert path.read_bytes() == PAYLOAD

    def test_it_repairs_a_deleted_lock_without_downloading(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        downloader = RecordingDownloader(PAYLOAD)
        fetch(descriptor, download=downloader, directory=store)
        lock_path(directory=store).unlink()

        fetch(descriptor, download=downloader, directory=store)

        assert downloader.calls == [descriptor.url]
        assert set(read_lock(directory=store)) == {descriptor.key}

    def test_it_honours_the_environment_when_no_directory_is_given(
        self, descriptor: ModelDescriptor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DND_AUDIO_MODELS_DIR", str(tmp_path / "env-models"))

        path = fetch(descriptor, download=RecordingDownloader(PAYLOAD))

        assert path == tmp_path / "env-models" / descriptor.filename

    def test_the_default_downloader_refuses_a_non_https_url(self, tmp_path: Path) -> None:
        """Belt and braces around the one function that opens a socket.

        Nothing constructs such a descriptor today; the check exists so that a future
        pin edited to `http://` or `file://` fails loudly rather than fetching.
        """
        with pytest.raises(ModelError):
            default_download(f"file://{tmp_path / 'model.onnx'}")


class TestFindModel:
    def test_absent_is_none(self, descriptor: ModelDescriptor, store: Path) -> None:
        assert find_model(descriptor, directory=store) is None

    def test_a_corrupted_model_is_treated_as_absent(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        """Same size, different bytes — the case only the digest can catch."""
        store.mkdir(parents=True)
        (store / descriptor.filename).write_bytes(SUBSTITUTE)

        assert find_model(descriptor, directory=store) is None

    def test_a_truncated_model_is_treated_as_absent(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        """What an interrupted download leaves at a perfectly plausible path."""
        store.mkdir(parents=True)
        (store / descriptor.filename).write_bytes(PAYLOAD[:-1])

        assert find_model(descriptor, directory=store) is None

    def test_a_directory_in_the_way_is_treated_as_absent(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        (store / descriptor.filename).mkdir(parents=True)

        assert find_model(descriptor, directory=store) is None


class TestRequireModel:
    def test_an_absent_model_is_fatal_and_names_the_fix(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        with pytest.raises(ModelError) as raised:
            require_model(descriptor, directory=store)

        assert raised.value.code == MODEL_UNAVAILABLE
        assert "models fetch" in str(raised.value)

    def test_a_verified_model_is_returned(self, descriptor: ModelDescriptor, store: Path) -> None:
        fetch(descriptor, download=RecordingDownloader(PAYLOAD), directory=store)

        assert require_model(descriptor, directory=store) == model_path(descriptor, directory=store)


class TestLock:
    def test_it_round_trips(self, descriptor: ModelDescriptor, store: Path) -> None:
        write_lock({descriptor.key: lock_record(descriptor)}, directory=store)

        assert read_lock(directory=store) == {descriptor.key: lock_record(descriptor)}

    def test_an_unchanged_rewrite_is_byte_identical(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        """INV-02. Canonical JSON, so a re-fetch produces no diff."""
        path = write_lock({descriptor.key: lock_record(descriptor)}, directory=store)
        first = path.read_bytes()

        write_lock(read_lock(directory=store), directory=store)

        assert path.read_bytes() == first

    def test_it_lives_beside_the_models(self, store: Path) -> None:
        assert lock_path(directory=store) == store / MODEL_LOCK_FILENAME

    def test_an_absent_lock_reads_as_empty(self, store: Path) -> None:
        assert read_lock(directory=store) == {}

    def test_malformed_json_reads_as_empty(self, store: Path) -> None:
        store.mkdir(parents=True)
        lock_path(directory=store).write_text("{not json", encoding="utf-8")

        assert read_lock(directory=store) == {}

    def test_an_unrecognized_version_reads_as_empty(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        """Half-understanding a future lock is worse than not reading it."""
        store.mkdir(parents=True)
        lock_path(directory=store).write_text(
            json.dumps(
                {
                    "lock_version": LOCK_VERSION + 1,
                    "models": {descriptor.key: lock_record(descriptor)},
                }
            ),
            encoding="utf-8",
        )

        assert read_lock(directory=store) == {}

    def test_an_entry_missing_a_field_is_dropped(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        record = lock_record(descriptor)
        del record["sha256"]
        store.mkdir(parents=True)
        lock_path(directory=store).write_text(
            json.dumps({"lock_version": LOCK_VERSION, "models": {descriptor.key: record}}),
            encoding="utf-8",
        )

        assert read_lock(directory=store) == {}

    def test_fetching_one_model_keeps_the_others(
        self, descriptor: ModelDescriptor, store: Path
    ) -> None:
        """M6b fetches more models into this same lock. It must merge, not replace."""
        store.mkdir(parents=True)
        other = lock_record(descriptor) | {"key": "some-asr-model"}
        write_lock({"some-asr-model": other}, directory=store)

        fetch(descriptor, download=RecordingDownloader(PAYLOAD), directory=store)

        assert set(read_lock(directory=store)) == {"some-asr-model", descriptor.key}


class TestSileroPin:
    """The real descriptor's shape. Its contents are checked over the network below."""

    def test_the_url_is_built_from_the_commit_not_the_tag(self) -> None:
        """A tag can be moved onto different bytes; a commit cannot (ADR-0013)."""
        assert SILERO_VAD.commit in SILERO_VAD.url
        assert SILERO_VAD.release not in SILERO_VAD.url
        assert SILERO_VAD.url.startswith("https://")
        assert SILERO_VAD.url.endswith(SILERO_VAD.path_in_repository)

    def test_the_pin_is_the_one_adr_0013_recorded(self) -> None:
        assert SILERO_VAD.key == "silero-vad"
        assert SILERO_VAD.repository == "snakers4/silero-vad"
        assert SILERO_VAD.release == "v6.2.1"
        assert SILERO_VAD.commit == "7e30209a3e901f9842f81b225f3e93d8199902b1"
        assert SILERO_VAD.size_bytes == 2327524
        assert SILERO_VAD.sha256 == (
            "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
        )

    def test_the_digest_is_a_lowercase_sha256(self) -> None:
        assert len(SILERO_VAD.sha256) == 64
        assert SILERO_VAD.sha256 == SILERO_VAD.sha256.lower()
        assert set(SILERO_VAD.sha256) <= set("0123456789abcdef")


@pytest.mark.allow_network
def test_the_pinned_url_still_serves_the_pinned_bytes() -> None:
    """The only test in this project permitted to reach the network (INV-06).

    Excluded from the gate, and deliberately *not* ``host_smoke``: needing the internet
    is not the same as needing a GPU. Run it explicitly when the pin is changed, or when
    a fetch fails and the question is whether upstream moved:

        uv run --no-sync pytest -m allow_network -q

    It downloads into memory and writes nothing — the models directory is the fetch
    command's business, not this test's.
    """
    payload = models.default_download(SILERO_VAD.url)

    assert len(payload) == SILERO_VAD.size_bytes
    assert sha256_bytes(payload) == SILERO_VAD.sha256
