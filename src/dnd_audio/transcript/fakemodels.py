"""Loading a session's declared fake model outputs (ADR-0018).

`transcribe --fake-models` is how this milestone runs end to end before M6b exists. It reads
`fake-models.json` — written beside `session.yaml` by the fixture generator, holding the
fake-VAD spans and fake-ASR utterances the spec's fixture recipe asks a fixture to carry — and
puts them behind the seams INV-10 already defines.

Three properties make this safe to have in production code:

**It is never automatic.** The flag is explicit and the file must be present; a missing file is
a named, fatal error rather than a silent fall back to a real model, and a file that happens to
be lying in a real session directory does nothing unless somebody asked for it.

**It is visible in everything the run produces.** The detector and the transcriber both carry a
`variant_digest` over the whole script, so a cache cannot serve one script's answers under
another's key (INV-08), and the caller emits a warning into the report and the records.

**It cannot pretend to be a model.** The identities carry no model name and no revision, so
`transcript.json` records `session-script` where a real run records `Qwen/Qwen3-ASR-1.7B`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnd_audio.activity.runner import DetectorBundle
from dnd_audio.artifacts.activity import DetectorIdentity
from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.errors import ConfigError
from dnd_audio.fakes import ScriptedActivityDetector, ScriptedUtterance, SessionScriptTranscriber
from dnd_audio.interfaces import SpeechSpan
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.transcript import FAKE_MODELS_FILENAME

__all__ = ["FakeModels", "load_fake_models"]

#: The name both identities carry. Deliberately not a model name.
_NAME = "session-script"


@dataclass(frozen=True, slots=True)
class FakeModels:
    """A session's declared detector and transcriber, and the digest of the script."""

    detector: DetectorBundle
    transcriber: SessionScriptTranscriber
    digest: str
    name: str = _NAME


def load_fake_models(session_dir: Path) -> FakeModels:
    """Read `fake-models.json` and put its contents behind the two model seams.

    Raises:
        ConfigError: if the file is missing, unreadable, not an object, or written to a
            version this code does not know. Every one of those is fatal rather than a
            fallback: the flag was an explicit request for *this* script, and quietly
            substituting a real model — or nothing — would be the surprise the flag exists
            to avoid.
    """
    path = session_dir / FAKE_MODELS_FILENAME
    document = _read(path)
    digest = sha256_bytes(canonical_json(document).encode("utf-8"))
    sample_rate = int(document.get("sample_rate", 0))
    if sample_rate <= 0:
        message = f"{path} does not say what sample rate its positions are counted at"
        raise ConfigError(message)

    detector = ScriptedActivityDetector(_spans(document, sample_rate, path))
    return FakeModels(
        detector=DetectorBundle(
            # The scripted detector's own identity already digests its spans; this replaces
            # the name so a graph built this way says so in the artifact rather than looking
            # like any other scripted run.
            identity=DetectorIdentity(
                name=_NAME, variant_digest=detector.identity().variant_digest
            ),
            make=lambda _track_id: detector,
        ),
        transcriber=SessionScriptTranscriber(_utterances(document, path), sample_rate=sample_rate),
        digest=digest,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        message = (
            f"--fake-models needs {path}, which is not readable: {exc}. It is written by "
            f"`scripts/make_fixture.py`; a real session does not have one, and the real ASR "
            f"adapter lands in M6b."
        )
        raise ConfigError(message) from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"{path} is not valid JSON: {exc}"
        raise ConfigError(message) from exc

    if not isinstance(document, dict):
        message = f"{path} must contain a JSON object, got {type(document).__name__}"
        raise ConfigError(message)

    from dnd_audio.fixtures.fakemodels import FAKE_MODELS_VERSION

    if document.get("fake_models_version") != FAKE_MODELS_VERSION:
        message = (
            f"{path} is version {document.get('fake_models_version')!r}; this build reads "
            f"version {FAKE_MODELS_VERSION}. Regenerate the fixture rather than editing it."
        )
        raise ConfigError(message)
    return document


def _spans(
    document: dict[str, Any], sample_rate: int, path: Path
) -> dict[str, tuple[SpeechSpan, ...]]:
    """The declared speech regions, on the grid the detector actually sees.

    Converted here rather than by the caller because the file states 48 kHz session samples —
    what the fixture declared — and a detector is handed the 16 kHz derivative. The start
    floors and the end ceils, the covering rule M2 owns: rounding both alike would shrink each
    region and hand M3 a region that starts after the word does.
    """
    scale = sample_rate // DERIVATIVE_SAMPLE_RATE
    if scale < 1 or sample_rate % DERIVATIVE_SAMPLE_RATE:
        message = (
            f"{path} is at {sample_rate} Hz, which does not divide by {DERIVATIVE_SAMPLE_RATE}"
        )
        raise ConfigError(message)

    found: dict[str, tuple[SpeechSpan, ...]] = {}
    for track_id, spans in sorted(dict(document.get("activity", {})).items()):
        converted = [
            SpeechSpan(start_sample=int(start) // scale, end_sample=-(-int(end) // scale))
            for start, end in spans
            if -(-int(end) // scale) > int(start) // scale
        ]
        found[str(track_id)] = tuple(
            sorted(converted, key=lambda span: (span.start_sample, span.end_sample))
        )
    return found


def _utterances(document: dict[str, Any], path: Path) -> list[ScriptedUtterance]:
    """The declared transcript, still on the session grid; the transcriber converts."""
    found: list[ScriptedUtterance] = []
    for item in document.get("asr", []):
        try:
            found.append(
                ScriptedUtterance(
                    track_id=str(item["track_id"]),
                    start_sample=int(item["start_sample"]),
                    end_sample=int(item["end_sample"]),
                    text=str(item["text"]),
                    words=tuple(
                        (int(word["start_sample"]), int(word["end_sample"]), str(word["text"]))
                        for word in item.get("words", [])
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            message = f"{path} has an unusable utterance: {exc}"
            raise ConfigError(message) from exc
    return sorted(found, key=lambda item: (item.start_sample, item.track_id))
