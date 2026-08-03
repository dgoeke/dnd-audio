"""The per-request ASR cache, and the raw artifact that is written before normalization.

The spec asks for both, and they are different things kept in different files on purpose.

**The cache** is keyed by everything that could change what the model would say: the exact
bytes submitted, the request's own identity, the transcriber's identity, the context hash, the
language, and `max_new_tokens` (ADR-0019). The request identity is the component the spec's
list does not name, and it is deliberate — INV-08 requires a key to *include* that list rather
than to be limited to it, and `config.py` states the bias this project applies: a too-broad key
costs recomputation, which is slow, while a too-narrow one serves a stale answer, which is
silent. It also stops a scripted fake, which chooses its response by `request_id` and is
therefore not a function of its audio, from turning a cache hit into a test that passes with
the wrong text.

It also means the word times stored here are **session-absolute on the derivative grid**,
because the position they were measured against is pinned by the key. An entry cannot be
served for the same audio somewhere else, so it never needs rebinding.

**The raw artifact** is the spec's "save the unmodified public result before applying pipeline
normalization... losslessly serialize all public fields... do not pickle the Python object".
It is an envelope: a version, the request it belongs to, and either the backend's own public
document or — for a transcriber whose result already *is* its public form, which is every fake
M4 has — this project's serialization of that result, with `source` saying which. M6b's adapter
fills `public_document` from Qwen's `ASRTranscription`; what M4 can freeze and test is the
preservation contract, not that a model this milestone does not have fills it correctly.

Publication order is M2's, for M2's reason: the raw document is written temp-then-rename, the
sidecar that makes the entry findable is **staged in memory**, and staged sidecars are
committed only once the caller has re-verified that every source is byte-identical to what it
read (INV-01). An entry is a hit only when the sidecar parses, agrees with itself about its own
key and paths, and the raw document it names exists at exactly the recorded size — the size
check being what makes "an incomplete entry is never a hit" true rather than merely intended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from dnd_audio.artifacts.records import TranscriberIdentity
from dnd_audio.artifacts.transcript import AlignmentStatus
from dnd_audio.determinism import canonical_json, sha256_bytes, write_atomic, write_json_atomic
from dnd_audio.interfaces import TranscribedWord, TranscriptionResult
from dnd_audio.transcript import ASR_DIRNAME, TRANSCRIPT_SEMANTICS_VERSION

__all__ = [
    "ASR_CACHE_RECORD_VERSION",
    "AUDIO_DTYPE",
    "RAW_DOCUMENT_VERSION",
    "AsrCache",
    "CachedTranscription",
    "asr_identity",
    "asr_identity_document",
    "audio_sha256",
    "raw_document",
    "raw_relative_path",
]

#: The shape of a sidecar. Separate from the semantics version: one is "what we computed",
#: the other is "how we wrote it down".
ASR_CACHE_RECORD_VERSION: Final = 1

#: The version of the raw envelope. Bumped if the envelope changes; the document *inside* it
#: is the backend's and is not this project's to version.
RAW_DOCUMENT_VERSION: Final = 1

#: How submitted audio is hashed: little-endian float32, explicitly rather than natively, so a
#: cache written on one machine is not silently missed on another. The same reason the PCM
#: reader spells out ``<f4``.
AUDIO_DTYPE: Final = "<f4"


def audio_sha256(samples: npt.NDArray[np.float32]) -> str:
    """The hash of exactly the bytes submitted to the model."""
    return sha256_bytes(np.ascontiguousarray(samples, dtype=AUDIO_DTYPE).tobytes())


def asr_identity_document(
    *,
    audio_hash: str,
    request_id: str,
    track_id: str,
    core_start_sample: int,
    core_end_sample: int,
    transcriber: TranscriberIdentity,
) -> dict[str, Any]:
    """Everything one transcription depends on, before it is hashed.

    Separate from :func:`asr_identity` so a test can assert *which components are present*
    rather than only that some change produced some different hash. A key that changes for the
    right reason in one test can still be missing a component, and the missing one is always
    the one that matters later — M2 learned this about derivative identity and it is the same
    lesson here.

    The transcriber identity carries the model, both revisions, the language, the context hash
    and `max_new_tokens`, so those reach the key without being restated in a second place that
    could disagree with the first.
    """
    return {
        "audio_sha256": audio_hash,
        "cache_record_version": ASR_CACHE_RECORD_VERSION,
        "core_end_sample": core_end_sample,
        "core_start_sample": core_start_sample,
        "request_id": request_id,
        "track_id": track_id,
        "transcriber": transcriber.model_dump(mode="json"),
        "transcript_semantics_version": TRANSCRIPT_SEMANTICS_VERSION,
    }


def asr_identity(
    *,
    audio_hash: str,
    request_id: str,
    track_id: str,
    core_start_sample: int,
    core_end_sample: int,
    transcriber: TranscriberIdentity,
) -> str:
    """The full cache identity of one transcription (INV-08)."""
    return sha256_bytes(
        canonical_json(
            asr_identity_document(
                audio_hash=audio_hash,
                request_id=request_id,
                track_id=track_id,
                core_start_sample=core_start_sample,
                core_end_sample=core_end_sample,
                transcriber=transcriber,
            )
        ).encode("utf-8")
    )


def raw_relative_path(key: str) -> str:
    """Where one request's unmodified public result lives, session-relative."""
    return f"{ASR_DIRNAME}/{key}.raw.json"


def raw_document(key: str, result: TranscriptionResult) -> dict[str, Any]:
    """The versioned envelope holding the unmodified public result.

    ``source`` is the honest part. ``backend`` means the adapter handed over its own
    serialization of what the model package returned; ``result`` means the transcriber's
    result object *is* the public form and this is it, field for field. Recording which
    stops a reader from believing a fake's output came from a model.
    """
    document: dict[str, Any]
    if result.public_document is not None:
        source, document = "backend", result.public_document
    else:
        source, document = "result", _serialize(result)
    return {
        "document": document,
        "key": key,
        "raw_schema_version": RAW_DOCUMENT_VERSION,
        "request_id": result.request_id,
        "source": source,
    }


def _serialize(result: TranscriptionResult) -> dict[str, Any]:
    """Every public field of a result, losslessly. Never a pickle."""
    return {
        "alignment_status": result.alignment_status,
        "language": result.language,
        "text": result.text,
        "truncated": result.truncated,
        "words": [
            {
                "start_sample": word.start_sample,
                "end_sample": word.end_sample,
                "text": word.text,
            }
            for word in result.words
        ],
    }


@dataclass(frozen=True, slots=True)
class CachedTranscription:
    """A transcription entry that is present and complete."""

    key: str
    request_id: str
    text: str
    words: tuple[TranscribedWord, ...]
    language: str
    truncated: bool
    alignment_status: AlignmentStatus
    raw_relative_path: str

    def as_result(self) -> TranscriptionResult:
        """The entry as the seam's own type, so a hit and a miss are the same downstream.

        ``public_document`` is deliberately not restored: it lives in the raw artifact, which
        exists to be read by a human or a later tool rather than to be fed back into the
        pipeline. Carrying it here would make a cache hit and a fresh call differ in a field
        nothing downstream reads, which is how a byte-stability bug starts.
        """
        return TranscriptionResult(
            request_id=self.request_id,
            text=self.text,
            words=self.words,
            language=self.language,
            truncated=self.truncated,
            alignment_status=self.alignment_status,
        )


@dataclass
class AsrCache:
    """One transcription per key, addressed by everything that could change it."""

    session_dir: Path
    read_enabled: bool = True
    hits: int = 0
    misses: int = 0
    _staged: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def get(self, key: str) -> CachedTranscription | None:
        """The complete entry for ``key``, or ``None``. Counts the hit or the miss."""
        if not self.read_enabled:
            self.misses += 1
            return None

        document = _parse(self._sidecar_path(key))
        entry = None if document is None else self._read(key, document)
        if entry is None or document is None:
            self.misses += 1
            return None

        try:
            size = (self.session_dir / entry.raw_relative_path).stat().st_size
        except OSError:
            self.misses += 1
            return None
        if size != document.get("raw_size_bytes"):
            self.misses += 1
            return None

        self.hits += 1
        return entry

    def publish(self, key: str, result: TranscriptionResult) -> CachedTranscription:
        """Write the raw document and stage the sidecar that will make ``key`` findable.

        The raw document lands first and the sidecar is staged, never the other way round:
        the sidecar is what makes an entry a hit, so writing it first would advertise a file
        that does not exist yet.
        """
        relative = raw_relative_path(key)
        # Serialized once and written from the same string the size is measured on: building
        # it twice would let the recorded size describe a document other than the one on disk.
        payload = canonical_json(raw_document(key, result))
        write_atomic(self.session_dir / relative, payload)

        entry = CachedTranscription(
            key=key,
            request_id=result.request_id,
            text=result.text,
            words=result.words,
            language=result.language,
            truncated=result.truncated,
            alignment_status=result.alignment_status,
            raw_relative_path=relative,
        )
        self._staged[key] = {
            "alignment_status": entry.alignment_status,
            "cache_record_version": ASR_CACHE_RECORD_VERSION,
            "key": key,
            "language": entry.language,
            "raw_relative_path": relative,
            "raw_size_bytes": len(payload.encode("utf-8")),
            "request_id": entry.request_id,
            "text": entry.text,
            "truncated": entry.truncated,
            "words": [
                {
                    "start_sample": word.start_sample,
                    "end_sample": word.end_sample,
                    "text": word.text,
                }
                for word in entry.words
            ],
        }
        return entry

    def commit(self) -> int:
        """Write every staged sidecar atomically. Returns how many were written."""
        written = 0
        for key, payload in sorted(self._staged.items()):
            write_json_atomic(self._sidecar_path(key), payload)
            written += 1
        self._staged.clear()
        return written

    def discard(self) -> None:
        """Drop everything staged. The raw documents remain, and without a sidecar are inert."""
        self._staged.clear()

    def _sidecar_path(self, key: str) -> Path:
        return self.session_dir / f"{ASR_DIRNAME}/{key}.json"

    def _read(self, key: str, document: dict[str, Any]) -> CachedTranscription | None:
        """Parse a sidecar, refusing anything that disagrees with itself."""
        if document.get("key") != key:
            return None
        try:
            version = int(document["cache_record_version"])
            relative = str(document["raw_relative_path"])
            alignment: AlignmentStatus = document["alignment_status"]
            words = tuple(
                TranscribedWord(
                    start_sample=int(item["start_sample"]),
                    end_sample=int(item["end_sample"]),
                    text=str(item["text"]),
                )
                for item in document["words"]
            )
            entry = CachedTranscription(
                key=key,
                request_id=str(document["request_id"]),
                text=str(document["text"]),
                words=words,
                language=str(document["language"]),
                truncated=bool(document["truncated"]),
                alignment_status=alignment,
                raw_relative_path=relative,
            )
        except (KeyError, TypeError, ValueError):
            return None

        # A sidecar that disagrees with itself is not a usable entry: the reader looks at the
        # *canonical* path, so a record naming another file would grant a hit on the strength
        # of a file nothing goes on to read.
        if version != ASR_CACHE_RECORD_VERSION or relative != raw_relative_path(key):
            return None
        if alignment not in ("aligned", "segment_only", "not_attempted"):
            return None
        if (alignment == "aligned") != bool(words):
            # The same consistency the seam enforces. A hand-edited entry claiming alignment
            # with no words would produce a segment whose status is a lie.
            return None
        return entry


def _parse(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or ``None`` for anything that is not one.

    Every form of unreadability is a miss rather than an error: a corrupted cache should cost
    time, not a session.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None
