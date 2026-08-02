#!/usr/bin/env python3
"""Regenerate the checked-in 48 kHz → 16 kHz decimation filter.

The coefficients live in the repository rather than being designed at import time, so
that a SciPy upgrade cannot silently change what every 16 kHz derivative in every cached
session was built with (INV-08). This script is how they got there, and running it is how
they change.

Two tests guard the result and they guard different things:

* ``tests/test_fir.py::TestDesignIsReproducible`` re-runs this design and compares, so a
  hand-edited coefficient is caught.
* ``tests/test_fir.py::TestFrequencyResponse`` measures the checked-in array against the
  declared passband, stopband, ripple, and attenuation. That one is the real acceptance
  test: without it, "fixed decimator" degrades into an arbitrary set of numbers that
  happens to produce the expected sample count.

Usage: ``uv run python scripts/design_fir.py [--check]``
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

from dnd_audio.determinism import canonical_json  # noqa: E402
from dnd_audio.timeline.fir import FIR_PATH  # noqa: E402

#: The design, stated once. Every number here is a decision:
#:
#: ``length`` is odd so the filter is symmetric with an integer group delay, and
#: ``(length - 1) / 2 = 129`` is divisible by the decimation factor, so that delay is a
#: whole number of samples in **both** grids — 129 at 48 kHz, 43 at 16 kHz. Without that
#: the 48↔16 mapping would be off by a fraction of a sample and no amount of testing
#: would make it exact.
#:
#: ``cutoff_hz`` is firwin's -6 dB point, placed between the declared passband edge and
#: the declared stopband edge. ``kaiser_beta`` then buys the stopband attenuation. The
#: pair was chosen by sweeping both against the response contract below and taking the
#: design with the most margin on each side.
INPUT_RATE: Final = 48000
OUTPUT_RATE: Final = 16000
DECIMATION: Final = 3
LENGTH: Final = 259
KAISER_BETA: Final = 9.0
CUTOFF_HZ: Final = 7450.0

DESIGN: Final[dict[str, Any]] = {
    "method": "kaiser_window_sinc",
    "designer": "scipy.signal.firwin",
    "input_rate": INPUT_RATE,
    "output_rate": OUTPUT_RATE,
    "decimation": DECIMATION,
    "length": LENGTH,
    "kaiser_beta": KAISER_BETA,
    "cutoff_hz": CUTOFF_HZ,
    "cutoff_note": "firwin's -6 dB point, between the declared passband and stopband edges",
    "normalization": "unity_dc_gain",
}

#: What the filter must *do*, independent of how it was designed. Asserted against the
#: checked-in coefficients by the frequency-response test. The stopband edge is 16 kHz's
#: Nyquist: everything above it aliases into the derivative, so it is the one number here
#: that is physics rather than preference.
RESPONSE: Final[dict[str, float]] = {
    "passband_edge_hz": 7000.0,
    "max_passband_ripple_db": 0.1,
    "stopband_edge_hz": 8000.0,
    "min_stopband_attenuation_db": 80.0,
}


def design() -> list[float]:
    """The coefficients, normalized to unity DC gain.

    Normalizing matters: an unnormalized windowed sinc has a DC gain near but not exactly
    one, which would make every derivative quietly a fraction of a decibel off from its
    source.
    """
    taps = np.asarray(
        firwin(LENGTH, CUTOFF_HZ / (INPUT_RATE / 2), window=("kaiser", KAISER_BETA)),
        dtype=np.float64,
    )
    return [float(value) for value in taps / taps.sum()]


def document() -> str:
    """The checked-in file's exact bytes."""
    delay_input = (LENGTH - 1) // 2
    if delay_input % DECIMATION:
        message = (
            f"a length-{LENGTH} filter has group delay {delay_input} at {INPUT_RATE} Hz, "
            f"which is not a whole number of output samples at 1/{DECIMATION} rate. "
            f"Choose a length where (length - 1) / 2 divides by {DECIMATION}."
        )
        raise ValueError(message)

    return canonical_json(
        {
            "name": "fir_48k_16k",
            "design": DESIGN,
            "response": RESPONSE,
            "group_delay_samples_input": delay_input,
            "group_delay_samples_output": delay_input // DECIMATION,
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
        current = FIR_PATH.read_text(encoding="utf-8") if FIR_PATH.exists() else ""
        if current == text:
            print(f"  {FIR_PATH.relative_to(REPO_ROOT)} is current")
            return 0
        print(f"  {FIR_PATH.relative_to(REPO_ROOT)} differs from this design")
        return 1

    FIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIR_PATH.write_text(text, encoding="utf-8")
    print(f"  wrote {FIR_PATH.relative_to(REPO_ROOT)} ({LENGTH} taps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
