"""The declared fake model outputs a synthetic session carries (ADR-0018).

The spec's fixture recipe asks a synthetic fixture for "deterministic fake-VAD/ground-truth
activity decisions" and "deterministic fake-ASR results". :class:`~dnd_audio.fixtures.session
.FixtureTruth` has always held both; this writes them beside `session.yaml` so something other
than a test can read them, which is what lets `transcribe --fake-models` run the whole branch
end to end before any model exists.

Nothing here is derived from the audio. Every number is the fixture's own declaration, made
before a sample was written — which is the property that makes a test against it mean anything.

**Word times are invented, deliberately and visibly.** A fixture that declared text without
times would leave the `aligned` path unexercised outside unit tests, and one that measured
times from synthetic noise would be pretending to align speech no human spoke (INV-10). So the
text is divided evenly across its interval, stated here in the generator rather than guessed at
run time, and the artifact records that its transcriber was a script.
"""

from __future__ import annotations

from typing import Any, Final

from dnd_audio.fixtures.session import FixtureTruth, SpeechInterval

__all__ = ["FAKE_MODELS_VERSION", "fake_models_document"]

#: Bumped if the shape below changes. The loader refuses anything else rather than guessing.
FAKE_MODELS_VERSION: Final = 1


def fake_models_document(truth: FixtureTruth) -> dict[str, Any]:
    """What the fixture would have a detector and a transcriber say, on the 48 kHz grid.

    The activity half is the **leaky** truth — speech *and* the bleed each track received —
    because that is what a real detector would find, and a fixture that declared only who
    spoke would make the bleed gate untestable. The ASR half matches it: a track that heard
    someone else's voice is scripted to transcribe that voice, which is exactly the duplicate
    the post-ASR collapse exists to catch.
    """
    return {
        "activity": {
            track: [[span.start_sample, span.end_sample] for span in spans]
            for track, spans in truth.leaky_activity_spans().items()
        },
        "asr": _utterances(truth),
        "fake_models_version": FAKE_MODELS_VERSION,
        "sample_rate": truth.sample_rate,
    }


def _utterances(truth: FixtureTruth) -> list[dict[str, Any]]:
    """Every utterance a fake ASR should return, on the track that can hear it.

    The delay comes from the interval itself, the same value the renderer used, so the
    scripted text lands on the samples the bleed actually occupies rather than near them.
    """
    found: list[dict[str, Any]] = []
    for interval in truth.speech:
        if not interval.text:
            continue
        found.append(_utterance(interval.track_id, interval.start_sample, interval))
        for target in interval.bleeds_into:
            found.append(
                _utterance(target, interval.start_sample + interval.delay_samples, interval)
            )
    return sorted(found, key=lambda item: (item["start_sample"], item["track_id"]))


def _utterance(track_id: str, start_sample: int, interval: SpeechInterval) -> dict[str, Any]:
    words = interval.text.split()
    total = interval.n_samples
    return {
        "end_sample": start_sample + total,
        "start_sample": start_sample,
        "text": interval.text,
        "track_id": track_id,
        "utterance_id": f"{interval.utterance_id or 'utt'}@{track_id}",
        "words": [
            {
                "end_sample": start_sample + (index + 1) * total // len(words),
                "start_sample": start_sample + index * total // len(words),
                "text": word,
            }
            for index, word in enumerate(words)
        ],
    }
