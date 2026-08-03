#!/usr/bin/env python3
"""Regenerate the checked-in 16 kHz speech-band band-pass filter.

The coefficients live in the repository rather than being designed at import time, for
the reason ``scripts/design_fir.py`` gives for the decimator and ADR-0014 repeats for
this one: a SciPy upgrade must not silently change what a cached bleed decision was made
with (INV-08). The filter's identity hash is part of the attribution cache key, so the
array moving is a commit rather than a surprise.

What the filter is *for* determines every number below. Bleed rejection compares how loud
one voice is on two different lavs. A DC offset, the HVAC rumble under the table, and the
hiss above the speech band all contribute to a broadband level and none of them are the
voice, so they are removed before any level is measured — 300 Hz to 3400 Hz is the
telephony speech band, chosen because it is what intelligible speech occupies and because
it is a band two microphones can be expected to agree about.

Two tests guard the result and they guard different things:

* ``tests/test_speech_band.py::TestDesignIsReproducible`` re-runs this design and
  compares, so a hand-edited coefficient is caught.
* ``tests/test_speech_band.py::TestFrequencyResponse`` measures the checked-in array
  against the declared passband, stopband, ripple, and attenuation. That one is the real
  acceptance test: without it, "the speech band filter" is an arbitrary array that
  happens to have the right length.

Usage: ``uv run python scripts/design_speech_band.py [--check]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy.signal import firwin

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dnd_audio.activity.band import SPEECH_BAND_PATH  # noqa: E402
from dnd_audio.determinism import canonical_json  # noqa: E402

#: The design, stated once. Every number here is a decision.
#:
#: ``LENGTH`` is odd, which is what makes the filter exactly symmetric and its group delay
#: the integer ``(LENGTH - 1) / 2`` rather than a half sample. Unlike the decimator there is
#: no second sample grid here, so the delay has nothing to divide by — 175 samples is
#: 10.9 ms of delay that :func:`~dnd_audio.activity.band.band_limited` removes with a slice.
#:
#: ``LOWER_CUTOFF_HZ`` and ``UPPER_CUTOFF_HZ`` are firwin's -6 dB points, placed half a
#: transition width outside each declared passband edge, so the declared passband is inside
#: the flat region rather than on the shoulder. ``KAISER_BETA`` then buys the stopband
#: attenuation. The pair was chosen by sweeping length and beta against the response
#: contract below and taking the design with the most margin on both stopbands — the
#: relationship is not monotonic in beta, because for a band-pass the two transition bands
#: interact, so the sweep found a better answer than the textbook estimate did.
SAMPLE_RATE: Final = 16000
LENGTH: Final = 351
KAISER_BETA: Final = 6.75
TRANSITION_WIDTH_HZ: Final = 200.0
PASSBAND_LOW_HZ: Final = 300.0
PASSBAND_HIGH_HZ: Final = 3400.0
LOWER_CUTOFF_HZ: Final = PASSBAND_LOW_HZ - TRANSITION_WIDTH_HZ / 2
UPPER_CUTOFF_HZ: Final = PASSBAND_HIGH_HZ + TRANSITION_WIDTH_HZ / 2

DESIGN: Final[dict[str, Any]] = {
    "method": "kaiser_window_sinc",
    "designer": "scipy.signal.firwin",
    "sample_rate": SAMPLE_RATE,
    "length": LENGTH,
    "kaiser_beta": KAISER_BETA,
    "lower_cutoff_hz": LOWER_CUTOFF_HZ,
    "upper_cutoff_hz": UPPER_CUTOFF_HZ,
    "cutoff_note": "firwin's -6 dB points, half a transition width outside each passband edge",
    "transition_width_hz": TRANSITION_WIDTH_HZ,
    # A band-pass has essentially no gain at DC, so the decimator's unity-DC-gain
    # normalization is meaningless here. firwin scales the taps to unit gain at the center
    # of the passband instead, which is what makes a band-limited level comparable to the
    # level of the signal it came from rather than offset by an arbitrary constant.
    "normalization": "unity_gain_at_passband_center",
}

#: What the filter must *do*, independent of how it was designed. Asserted against the
#: checked-in coefficients by the frequency-response test. The passband is the telephony
#: speech band; the stopband edges are one transition width outside it. Nothing here is
#: physics the way the decimator's Nyquist edge was — it is a judgement about what part of
#: a signal is a voice — which is why the numbers are declared in the artifact and measured
#: by a test rather than left implicit in a tap count.
RESPONSE: Final[dict[str, float]] = {
    "passband_low_hz": PASSBAND_LOW_HZ,
    "passband_high_hz": PASSBAND_HIGH_HZ,
    "max_passband_ripple_db": 1.0,
    "lower_stopband_edge_hz": PASSBAND_LOW_HZ - TRANSITION_WIDTH_HZ,
    "upper_stopband_edge_hz": PASSBAND_HIGH_HZ + TRANSITION_WIDTH_HZ,
    "min_stopband_attenuation_db": 60.0,
}


def design() -> list[float]:
    """The coefficients, as firwin produces and scales them.

    ``pass_zero=False`` is what makes this a band-pass rather than a band-stop, and it is
    the one argument whose omission would produce a plausible-looking array that passes
    exactly the frequencies this filter exists to remove.
    """
    taps = np.asarray(
        firwin(
            LENGTH,
            [LOWER_CUTOFF_HZ, UPPER_CUTOFF_HZ],
            window=("kaiser", KAISER_BETA),
            pass_zero=False,
            fs=SAMPLE_RATE,
        ),
        dtype=np.float64,
    )
    return [float(value) for value in taps]


def document() -> str:
    """The checked-in file's exact bytes."""
    if LENGTH % 2 == 0:
        message = (
            f"a length-{LENGTH} filter has a half-sample group delay, which cannot be "
            f"compensated by a slice. Choose an odd length."
        )
        raise ValueError(message)

    return canonical_json(
        {
            "name": "fir_speechband_16k",
            "design": DESIGN,
            "response": RESPONSE,
            "group_delay_samples": (LENGTH - 1) // 2,
            "coefficients": design(),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the checked-in file differs, without rewriting it.",
    )
    args = parser.parse_args()

    text = document()
    if args.check:
        current = SPEECH_BAND_PATH.read_text(encoding="utf-8") if SPEECH_BAND_PATH.exists() else ""
        if current == text:
            print(f"  {SPEECH_BAND_PATH.relative_to(REPO_ROOT)} is current")
            return 0
        print(f"  {SPEECH_BAND_PATH.relative_to(REPO_ROOT)} differs from this design")
        return 1

    SPEECH_BAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEECH_BAND_PATH.write_text(text, encoding="utf-8")
    print(f"  wrote {SPEECH_BAND_PATH.relative_to(REPO_ROOT)} ({LENGTH} taps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
