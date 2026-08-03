"""The ASR cache, and the raw artifact written before any normalization.

Two properties, and the second is the one that is easy to fake. The identity has to *include*
everything that could change what the model said — varied one component at a time, because a
key that changes for the right reason in one test can still be missing a component, and the
missing one is always the one that matters later. And the cache has to be *consulted*: a
perfect key nothing reads would pass the first half on its own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dnd_audio.artifacts.records import TranscriberIdentity
from dnd_audio.interfaces import TranscribedWord, TranscriptionResult
from dnd_audio.transcript import ASR_DIRNAME
from dnd_audio.transcript.cache import (
    ASR_CACHE_RECORD_VERSION,
    RAW_DOCUMENT_VERSION,
    AsrCache,
    asr_identity,
    asr_identity_document,
    audio_sha256,
    raw_document,
    raw_relative_path,
)

HASH = "d" * 64


def an_identity(**overrides: Any) -> TranscriberIdentity:
    fields: dict[str, Any] = {
        "name": "scripted",
        "max_new_tokens": 1024,
        "language": "English",
        "variant_digest": HASH,
    }
    return TranscriberIdentity(**{**fields, **overrides})


def a_key(**overrides: Any) -> str:
    fields: dict[str, Any] = {
        "audio_hash": HASH,
        "request_id": "req_tx-a_000000048000",
        "track_id": "tx-a",
        "core_start_sample": 16_000,
        "core_end_sample": 32_000,
        "transcriber": an_identity(),
    }
    return asr_identity(**{**fields, **overrides})


def a_result(**overrides: Any) -> TranscriptionResult:
    fields: dict[str, Any] = {
        "request_id": "req_tx-a_000000048000",
        "text": "We should go back to Zephyrine.",
        "words": (TranscribedWord(start_sample=16_000, end_sample=16_400, text="We"),),
        "alignment_status": "aligned",
    }
    return TranscriptionResult(**{**fields, **overrides})


class TestIdentity:
    def test_the_document_names_every_component(self) -> None:
        """Asserted by name, not by "some change produced some different hash"."""
        document = asr_identity_document(
            audio_hash=HASH,
            request_id="req_tx-a_000000048000",
            track_id="tx-a",
            core_start_sample=16_000,
            core_end_sample=32_000,
            transcriber=an_identity(),
        )
        assert set(document) == {
            "audio_sha256",
            "cache_record_version",
            "core_end_sample",
            "core_start_sample",
            "request_id",
            "track_id",
            "transcriber",
            "transcript_semantics_version",
        }
        # The spec's own list reaches the key through the transcriber identity rather than
        # being restated in a second place that could disagree with the first.
        assert set(document["transcriber"]) >= {
            "model",
            "model_revision",
            "aligner",
            "aligner_revision",
            "language",
            "max_new_tokens",
            "context_sha256",
        }

    def test_the_audio_moves_the_key(self) -> None:
        assert a_key() != a_key(audio_hash="e" * 64)

    def test_max_new_tokens_moves_the_key(self) -> None:
        """Acceptance criterion 14, and the spec names it as an inference parameter."""
        assert a_key() != a_key(transcriber=an_identity(max_new_tokens=512))

    def test_the_context_moves_the_key(self) -> None:
        assert a_key() != a_key(transcriber=an_identity(context_sha256="b" * 64))

    def test_the_language_moves_the_key(self) -> None:
        assert a_key() != a_key(transcriber=an_identity(language="German"))

    def test_the_model_and_its_revision_move_the_key(self) -> None:
        assert a_key() != a_key(transcriber=an_identity(model="Qwen/Qwen3-ASR-1.7B"))
        assert a_key(transcriber=an_identity(model="q", model_revision="abc")) != a_key(
            transcriber=an_identity(model="q", model_revision="def")
        )

    def test_the_aligner_and_its_revision_move_the_key(self) -> None:
        assert a_key() != a_key(transcriber=an_identity(aligner="Qwen/Qwen3-ForcedAligner-0.6B"))
        assert a_key(transcriber=an_identity(aligner="a", aligner_revision="abc")) != a_key(
            transcriber=an_identity(aligner="a", aligner_revision="def")
        )

    def test_a_scripted_variant_moves_the_key(self) -> None:
        """Two scripted transcribers with different scripts are different transcribers."""
        assert a_key() != a_key(transcriber=an_identity(variant_digest="c" * 64))

    def test_the_request_identity_moves_the_key(self) -> None:
        """The component the spec's list does not name, and the reason it is here.

        A scripted fake selects its response by `request_id`, so it is not a function of its
        audio. Without this, two requests with byte-identical audio and different scripted
        answers would share an entry and the first would be served for the second — a cache
        hit that makes a test pass with the wrong text (ADR-0019).
        """
        assert a_key() != a_key(request_id="req_tx-a_000000096000")
        assert a_key() != a_key(track_id="tx-b")
        assert a_key() != a_key(core_start_sample=16_001)
        assert a_key() != a_key(core_end_sample=32_001)

    def test_identical_inputs_give_an_identical_key(self) -> None:
        assert a_key() == a_key()

    def test_audio_is_hashed_as_little_endian_float32(self) -> None:
        """Explicitly, so a cache written on one machine is not silently missed on another."""
        samples = np.array([0.5, -0.25], dtype=np.float32)
        assert audio_sha256(samples) == audio_sha256(np.asarray(samples, dtype="<f4"))
        assert audio_sha256(samples) != audio_sha256(np.array([0.5, 0.25], dtype=np.float32))


class TestTheRawArtifact:
    def test_a_fake_result_is_recorded_as_its_own_public_form(self) -> None:
        document = raw_document("k", a_result())
        assert document["raw_schema_version"] == RAW_DOCUMENT_VERSION
        assert document["source"] == "result"
        assert document["document"] == {
            "alignment_status": "aligned",
            "language": "English",
            "text": "We should go back to Zephyrine.",
            "truncated": False,
            "words": [{"start_sample": 16_000, "end_sample": 16_400, "text": "We"}],
        }

    def test_a_backend_document_is_carried_through_unmodified(self) -> None:
        """M6b fills this from Qwen's public result; M4 freezes the preservation contract."""
        public = {"language": "en", "text": "hi", "timestamps": [[0.0, 0.5, "hi"]]}
        document = raw_document("k", a_result(public_document=public))
        assert document["source"] == "backend"
        assert document["document"] == public

    def test_the_artifact_is_json_and_round_trips(self, tmp_path: Path) -> None:
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result())
        written = json.loads((tmp_path / raw_relative_path(key)).read_text(encoding="utf-8"))
        assert written == raw_document(key, a_result())

    def test_nothing_in_the_package_pickles(self, repo_root: Path) -> None:
        """The spec says so in as many words: "Do not pickle the Python object".

        Matched on *use* rather than on the word, because the module that must not pickle is
        also the one that has to explain why.
        """
        used = re.compile(r"^\s*(?:import pickle|from pickle\b)|\bpickle\.\w+\(", re.MULTILINE)
        for path in (repo_root / "src" / "dnd_audio" / "transcript").rglob("*.py"):
            assert used.search(path.read_text(encoding="utf-8")) is None, path


class TestTheCacheIsConsulted:
    def test_a_published_entry_is_a_hit_after_commit(self, tmp_path: Path) -> None:
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result())
        assert cache.get(key) is None  # staged, not committed: nothing is findable yet
        cache.commit()

        found = AsrCache(session_dir=tmp_path).get(key)
        assert found is not None
        assert found.as_result() == a_result()

    def test_a_staged_entry_discarded_never_becomes_a_hit(self, tmp_path: Path) -> None:
        """A run that failed INV-01 verification must leave nothing findable behind."""
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result())
        cache.discard()
        cache.commit()
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_hits_and_misses_are_counted(self, tmp_path: Path) -> None:
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        assert cache.get(key) is None
        cache.publish(key, a_result())
        cache.commit()
        assert cache.get(key) is not None
        assert (cache.hits, cache.misses) == (1, 1)

    def test_reading_can_be_turned_off_without_losing_the_write(self, tmp_path: Path) -> None:
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result())
        cache.commit()
        cold = AsrCache(session_dir=tmp_path, read_enabled=False)
        assert cold.get(key) is None
        assert (tmp_path / raw_relative_path(key)).exists()


class TestAnIncompleteEntryIsNeverAHit:
    def _committed(self, tmp_path: Path) -> str:
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result())
        cache.commit()
        return key

    def test_a_missing_raw_document_is_a_miss(self, tmp_path: Path) -> None:
        key = self._committed(tmp_path)
        (tmp_path / raw_relative_path(key)).unlink()
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_a_truncated_raw_document_is_a_miss(self, tmp_path: Path) -> None:
        """The size check is what makes "incomplete is never a hit" true, not merely intended."""
        key = self._committed(tmp_path)
        path = tmp_path / raw_relative_path(key)
        path.write_text(path.read_text(encoding="utf-8")[:-20], encoding="utf-8")
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_a_sidecar_naming_another_file_is_a_miss(self, tmp_path: Path) -> None:
        key = self._committed(tmp_path)
        sidecar = tmp_path / f"{ASR_DIRNAME}/{key}.json"
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        document["raw_relative_path"] = f"{ASR_DIRNAME}/somewhere-else.raw.json"
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_a_sidecar_under_another_key_is_a_miss(self, tmp_path: Path) -> None:
        key = self._committed(tmp_path)
        sidecar = tmp_path / f"{ASR_DIRNAME}/{key}.json"
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        document["key"] = "0" * 64
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_an_old_record_version_is_a_miss(self, tmp_path: Path) -> None:
        key = self._committed(tmp_path)
        sidecar = tmp_path / f"{ASR_DIRNAME}/{key}.json"
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        document["cache_record_version"] = ASR_CACHE_RECORD_VERSION + 1
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_an_entry_claiming_alignment_with_no_words_is_a_miss(self, tmp_path: Path) -> None:
        """The consistency the seam enforces; a hand-edited entry must not bypass it."""
        key = self._committed(tmp_path)
        sidecar = tmp_path / f"{ASR_DIRNAME}/{key}.json"
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        document["words"] = []
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_unparseable_json_is_a_miss_rather_than_an_error(self, tmp_path: Path) -> None:
        key = self._committed(tmp_path)
        (tmp_path / f"{ASR_DIRNAME}/{key}.json").write_text("{not json", encoding="utf-8")
        assert AsrCache(session_dir=tmp_path).get(key) is None

    def test_an_absent_entry_is_a_miss(self, tmp_path: Path) -> None:
        assert AsrCache(session_dir=tmp_path).get(a_key()) is None


class TestWhatTheCacheDoesNotCarry:
    def test_a_hit_does_not_restore_the_backend_document(self, tmp_path: Path) -> None:
        """It lives in the raw artifact, for a human. Restoring it would make a hit and a
        fresh call differ in a field nothing downstream reads."""
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result(public_document={"text": "hi"}))
        cache.commit()
        found = AsrCache(session_dir=tmp_path).get(key)
        assert found is not None
        assert found.as_result().public_document is None

    def test_publishing_twice_is_stable(self, tmp_path: Path) -> None:
        cache = AsrCache(session_dir=tmp_path)
        key = a_key()
        cache.publish(key, a_result())
        first = (tmp_path / raw_relative_path(key)).read_bytes()
        cache.publish(key, a_result())
        assert (tmp_path / raw_relative_path(key)).read_bytes() == first


@pytest.mark.parametrize("truncated", [True, False])
def test_truncation_survives_a_round_trip(tmp_path: Path, truncated: bool) -> None:
    """M4 branches on it, so a cached entry that lost it would skip the retry entirely."""
    cache = AsrCache(session_dir=tmp_path)
    key = a_key()
    cache.publish(key, a_result(truncated=truncated))
    cache.commit()
    found = AsrCache(session_dir=tmp_path).get(key)
    assert found is not None
    assert found.truncated is truncated
