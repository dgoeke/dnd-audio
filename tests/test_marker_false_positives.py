"""What the marker detector does with real speech that contains no marker.

Every accepted sequence here is a **false positive**, because none of these recordings
contains a marker — they predate it. That makes this the one half of the bench's
false-positive question that needs no bench: real DJI hardware, real rooms, real voices,
already on disk.

**It answers only that half.** Whether a marker played from a phone speaker is *found* on a
lav across a table is a question about acoustics, and nothing here touches it. See
`docs/M10-marker-bench-protocol.md`.

Marked `host_smoke` and excluded from `./scripts/gate.sh`, because the recordings are
gitignored session audio. The default suite proves the same property against synthetic
speech and music in `test_marker_detect.py`; this is the same claim against a real capture
chain, which INV-10's reasoning says is the one that counts.

Recordings are **discovered rather than named**, the convention `test_qwen_smoke.py`
established for the same reason: pinning a filename turns a replaced corpus into a silent
skip, which is worse than either passing or failing.

Run it the way the rest of the host smokes run:

    uv run pytest -m host_smoke tests/test_marker_false_positives.py -n 0 -s
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from dnd_audio.marker.detect import DetectorThresholds, _normalized_scores, detect_occurrences
from dnd_audio.marker.spec import MARKER_SPECS, MarkerSpec
from dnd_audio.marker.synth import marker_templates
from dnd_audio.timeline.pcm import PcmReader, open_pcm

pytestmark = pytest.mark.host_smoke

#: Both corpora of real DJI audio this host carries. `samples/` is one person announcing
#: microphones; `minimal-test-samples/` is two people talking over each other on purpose,
#: with hand claps at both ends — the closest thing on disk to a hostile negative control,
#: since a clap is broadband and a marker chirp is what a clap would be mistaken for.
CORPORA: Final = ("samples", "minimal-test-samples")

#: How close a single chirp may come to its acceptance threshold on real speech before this
#: stops being reassuring. Measured 2026-08-05: the worst case across 13.7 minutes was 186
#: permille against a 550 threshold, so this leaves the measurement room to drift by more
#: than twice its observed value and still fail before the threshold does.
HEADROOM_CEILING_PERMILLE: Final = 400


def _recordings() -> list[Path]:
    return sorted(path for corpus in CORPORA for path in Path(corpus).glob("*.wav"))


RECORDINGS: Final = _recordings()

_no_audio = pytest.mark.skipif(
    not RECORDINGS,
    reason=(
        "OQ-025 — needs real recordings in samples/ or minimal-test-samples/, which are "
        "gitignored session audio. This asks what the detector does with a real capture "
        "chain, and INV-10's reasoning is that synthetic speech cannot answer it; "
        "test_marker_detect.py carries the synthetic half in the default gate"
    ),
)


def _signal(path: Path) -> np.ndarray:
    """One recording as float64, through the pipeline's own decoder rather than a second one."""
    source = open_pcm(path)
    with PcmReader(source) as reader:
        return reader.read(0, source.n_samples).astype(np.float64)


def _best_chirp_permille(signal: np.ndarray, spec: MarkerSpec) -> int:
    """The strongest single-chirp match anywhere in ``signal``.

    The accepted-sequence count is the property that matters, but on its own it says only
    that a line was not crossed and never by how much. This is the margin.
    """
    from dnd_audio.marker.detect import to_permille

    best = 0
    for template in marker_templates(spec):
        scores = _normalized_scores(signal, template.astype(np.float64))
        if scores.size:
            best = max(best, to_permille(float(scores.max())))
    return best


@_no_audio
class TestRealSpeechIsNotAMarker:
    """The negative control, on hardware rather than on a synthesizer."""

    @pytest.mark.parametrize("marker_name", sorted(MARKER_SPECS))
    def test_no_recording_yields_an_accepted_sequence(self, marker_name: str) -> None:
        spec = MARKER_SPECS[marker_name]
        thresholds = DetectorThresholds()
        for path in RECORDINGS:
            source = open_pcm(path)
            with PcmReader(source) as reader:
                found = detect_occurrences(
                    reader, spec, interval=(0, source.n_samples), thresholds=thresholds
                )
            assert found == [], (
                f"{marker_name} was 'detected' {len(found)} time(s) in {path.name}, which "
                f"contains no marker. Anchors: {[item.anchor_sample for item in found]}"
            )

    @pytest.mark.parametrize("marker_name", sorted(MARKER_SPECS))
    def test_real_speech_stays_far_below_the_chirp_threshold(self, marker_name: str) -> None:
        """Not merely under the line — under it with the margin the line was chosen for."""
        spec = MARKER_SPECS[marker_name]
        threshold = DetectorThresholds().min_chirp_score_permille
        worst = 0
        where = ""
        for path in RECORDINGS:
            best = _best_chirp_permille(_signal(path), spec)
            if best > worst:
                worst, where = best, path.name

        print(f"\n{marker_name}: worst real-speech chirp match {worst}/1000 ({where})")
        assert worst < HEADROOM_CEILING_PERMILLE, (
            f"{marker_name} reached {worst} permille on real speech in {where}, within "
            f"{threshold - worst} of the {threshold} acceptance threshold. The margin this "
            f"detector's false-positive claim rests on has narrowed; re-measure before "
            f"trusting ADR-0042's thresholds"
        )
