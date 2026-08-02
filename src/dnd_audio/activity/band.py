"""The checked-in speech-band filter, and the level measurement built on it.

ADR-0014's gate compares *voices*: how loud this candidate is on this lav against how loud
the competing candidate is on that one. A broadband level does not answer that question. A
DC offset from a cheap preamp, the HVAC rumble under the table, and the hiss above the
speech band all move a broadband RMS by decibels while saying nothing about who was
talking, and two lavs with different placements accumulate different amounts of each. So
every level this project compares is measured through one fixed 300–3400 Hz band-pass.

The coefficients are data in the repository, not something designed at import time — the
reason `timeline/fir.py` gives for the decimator, and ADR-0014 repeats for this filter. A
SciPy upgrade must not silently change what a cached bleed decision was made with (INV-08),
so :attr:`SpeechBandFilter.identity` is part of the attribution cache key and the array
moving is a commit with a frequency-response test standing in front of it.

`scripts/design_speech_band.py` is how the file is produced. ``tests/test_speech_band.py``
guards it twice, and the two guards catch different failures: `TestDesignIsReproducible`
catches a hand-edited coefficient, and `TestFrequencyResponse` measures the committed array
against the passband, stopband, ripple, and attenuation it declares. The second is the real
acceptance test — without it "the speech band filter" is an arbitrary array of the right
length, and every level comparison downstream is measured through something nobody checked.

One property of the design is load-bearing rather than aesthetic: **odd length, exactly
symmetric**, which makes the phase linear and the group delay the integer ``(N - 1) / 2``.
:func:`band_limited` removes that delay with a slice, so a band-limited sample sits at the
same index as the sample it came from and a level measured over a candidate's span is a
level of that candidate rather than of the 11 ms next door. Unlike the decimator there is no
second sample grid here, so the delay has nothing to divide by; the only constraint is that
it be a whole sample.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from dnd_audio.determinism import sha256_bytes
from dnd_audio.errors import DndAudioError

__all__ = [
    "SILENCE_FLOOR_MB",
    "SPEECH_BAND_PATH",
    "SpeechBandError",
    "SpeechBandFilter",
    "band_limited",
    "level_millibels",
    "load_speech_band_filter",
    "rms_millibels",
]

#: The committed coefficient file. Not loaded at import: `scripts/design_speech_band.py`
#: imports this module to find out where to *write* it, and a module that reads its own
#: output on import cannot be used to create that output.
SPEECH_BAND_PATH: Final = Path(__file__).parent / "data" / "fir_speechband_16k.json"

#: The level reported for digital silence, in millibels relative to full scale (-120 dB).
#:
#: Silence has no logarithm. Returning ``-inf`` would make the artifact's integer level
#: field impossible to fill (there is no such int), would poison every subtraction it
#: reaches — a margin against it is ``inf`` or ``nan`` depending on which side is silent —
#: and cannot be serialized as JSON at all (INV-02 rejects non-finite floats). A floor is a
#: lie of a known size instead: -120 dB is roughly 20 dB below the quietest thing 16-bit
#: audio can represent, so no real recording reaches it and nothing above it is clamped.
SILENCE_FLOOR_MB: Final = -12000


class SpeechBandError(DndAudioError):
    """The checked-in speech-band filter is missing, unreadable, or not what it claims."""

    default_code = "speech_band_filter_unusable"


@dataclass(frozen=True, slots=True)
class SpeechBandFilter:
    """One fixed speech-band band-pass, and the response contract it is held to."""

    name: str
    coefficients: npt.NDArray[np.float64]
    sample_rate: int
    #: Samples of delay the filter introduces. An integer by design, which is what lets
    #: :func:`band_limited` compensate it with a slice rather than an interpolation.
    group_delay: int
    #: The declared response. Asserted against the coefficients by `tests/test_speech_band.py`,
    #: not by this module — a self-check on every import would be a slow way to discover at
    #: runtime what a test discovers at commit time.
    passband_low_hz: float
    passband_high_hz: float
    max_passband_ripple_db: float
    lower_stopband_edge_hz: float
    upper_stopband_edge_hz: float
    min_stopband_attenuation_db: float
    #: SHA-256 of the file's exact bytes. Part of the attribution cache identity: a changed
    #: filter must rebuild every decision it ever produced (INV-08, ADR-0014).
    identity: str

    @property
    def length(self) -> int:
        return int(self.coefficients.shape[0])


@functools.cache
def load_speech_band_filter() -> SpeechBandFilter:
    """Read, validate, and cache the checked-in speech-band filter.

    Raises:
        SpeechBandError: if the file is absent, malformed, or internally inconsistent — an
            even length, a broken symmetry, or a declared delay that disagrees with the
            length. Each of those means the filtered signal is offset from the signal it
            was measured against, and a level measured over the wrong span is wrong in a
            way that looks entirely plausible in the artifact.
    """
    try:
        raw = SPEECH_BAND_PATH.read_bytes()
    except OSError as exc:
        message = (
            f"the speech-band filter {SPEECH_BAND_PATH} could not be read: {exc}. "
            f"Regenerate it with `uv run python scripts/design_speech_band.py`."
        )
        raise SpeechBandError(message) from exc

    document = _parsed(raw)
    design = _mapping(document, "design")
    response = _mapping(document, "response")
    coefficients = np.asarray(document.get("coefficients"), dtype=np.float64)

    group_delay = _integer(document, "group_delay_samples")
    _validate(coefficients, group_delay)

    return SpeechBandFilter(
        name=str(document.get("name", SPEECH_BAND_PATH.stem)),
        coefficients=coefficients,
        sample_rate=_integer(design, "sample_rate"),
        group_delay=group_delay,
        passband_low_hz=_number(response, "passband_low_hz"),
        passband_high_hz=_number(response, "passband_high_hz"),
        max_passband_ripple_db=_number(response, "max_passband_ripple_db"),
        lower_stopband_edge_hz=_number(response, "lower_stopband_edge_hz"),
        upper_stopband_edge_hz=_number(response, "upper_stopband_edge_hz"),
        min_stopband_attenuation_db=_number(response, "min_stopband_attenuation_db"),
        identity=sha256_bytes(raw),
    )


def band_limited(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Filter to the speech band, delay-compensated, same length as the input.

    Output sample *i* corresponds to input sample *i*. That alignment is the whole reason
    the filter is symmetric: the convolution delays everything by ``group_delay`` samples
    equally, so removing it is the slice ``convolved[delay : delay + n]`` rather than a
    frequency-dependent correction that could not be exact.

    **What happens at the two ends, precisely.** ``numpy.convolve`` in ``full`` mode treats
    everything outside the input as zero, and the slice keeps ``n`` of the ``n + 2 * delay``
    samples it produces. So the first ``group_delay`` and the last ``group_delay`` output
    samples are computed against implicit zeros where the real signal's past and future
    would have been. They are not wrong for a signal that genuinely begins and ends in
    silence; they are an approximation, of at most 11 ms at each end, whenever this is
    called on a *window cut out of* a longer track. Callers measuring a level over a speech
    candidate should pass more audio than the candidate spans where they have it, and every
    caller should know that a window shorter than ``2 * group_delay`` is ramp at both ends
    with nothing in between. Filtering in one pass over a whole track and slicing afterwards
    is exact; this function cannot do that for the caller, because INV-07 forbids holding a
    session-length array to make it possible.

    The convolution is a direct one rather than an FFT one on purpose. An FFT convolution's
    rounding depends on the transform length the library happens to pick, so the same audio
    could produce a level one millibel different under a different SciPy build — and that
    millibel is a cache-key input and a comparison the bleed gate makes decisions with.
    """
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        message = f"band_limited takes one channel, got shape {array.shape}"
        raise ValueError(message)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)

    speech_band = load_speech_band_filter()
    delay = speech_band.group_delay
    convolved = np.convolve(array, speech_band.coefficients)
    aligned: npt.NDArray[np.float64] = convolved[delay : delay + array.size]
    return aligned.astype(np.float32)


def level_millibels(samples: npt.NDArray[np.float32]) -> int:
    """Band-limit ``samples``, then measure. The convenience form of :func:`rms_millibels`.

    A caller that already holds a band-limited signal — because it also correlated it —
    must use :func:`rms_millibels` instead. Filtering twice is not a rounding difference:
    it applies the passband ripple twice and rolls the band edges off at double the rate,
    so a voice with energy near 300 Hz reads quieter than it is.
    """
    return rms_millibels(band_limited(samples))


def rms_millibels(samples: npt.NDArray[np.float32]) -> int:
    """RMS of an **already band-limited** signal, in millibels relative to full scale.

    Millibels — decibels scaled by a hundred, as an integer — because ``activity.json``
    contains no floats at all (INV-02): a level that is the quotient of two NumPy reductions
    is not reliably identical across a library upgrade, and an artifact that must be
    byte-stable on rerun cannot contain one. Negative for every real signal, since a
    full-scale sine is -3.01 dB RMS and nothing quieter is louder.

    The rounding is half away from zero, matching :mod:`dnd_audio.determinism`. That module
    deliberately owns the *only* quantizer this project has, and this is not a second one in
    the sense INV-04 means: INV-04 is about time, where two quantizers with different tie
    rules make a millisecond and a sample position disagree about the same instant. A
    decibel is not a time, `to_samples` takes seconds and a sample rate, and calling it with
    a decibel and the number 100 would make the code say something false about what it is
    doing. What matters is that the tie rule is the same one, stated rather than inherited
    from :func:`round` (which is banker's rounding, so ``round(0.5) != round(1.5)`` in the
    way that matters), and :func:`_millibels` is where it is stated.

    Digital silence returns :data:`SILENCE_FLOOR_MB` exactly, and any level below the floor
    is clamped to it.
    """
    filtered = np.asarray(samples, dtype=np.float64)
    if filtered.size == 0:
        return SILENCE_FLOOR_MB
    mean_square = float(np.mean(filtered * filtered))
    if mean_square <= 0.0:
        return SILENCE_FLOOR_MB
    # 10·log10 of the mean square, not 20·log10 of its root: one logarithm instead of a
    # square root and a logarithm, and identical arithmetic either way.
    return max(_millibels(10.0 * math.log10(mean_square)), SILENCE_FLOOR_MB)


def _millibels(decibels: float) -> int:
    """Quantize decibels to integer millibels, halves away from zero.

    ``Fraction(decibels)`` is the float's exact binary value, so the scaling by 100 and the
    comparison against a half are exact and the result does not depend on which way a
    borderline product rounds in binary. This is the rule
    :func:`dnd_audio.determinism.to_samples` uses, restated for a quantity that is not a
    time; see :func:`level_millibels` for why it is restated rather than reused.
    """
    scaled = Fraction(decibels) * 100
    magnitude, remainder = divmod(abs(scaled.numerator), scaled.denominator)
    if 2 * remainder >= scaled.denominator:
        magnitude += 1
    return -magnitude if scaled.numerator < 0 else magnitude


def _validate(coefficients: npt.NDArray[np.float64], group_delay: int) -> None:
    length = int(coefficients.shape[0])
    if coefficients.ndim != 1 or length == 0:
        message = (
            f"the speech-band filter must be a non-empty 1-D array, got shape {coefficients.shape}"
        )
        raise SpeechBandError(message)
    if length % 2 == 0:
        message = (
            f"the speech-band filter has {length} taps. An even-length filter has a "
            f"half-sample group delay, which cannot be compensated by a slice."
        )
        raise SpeechBandError(message)
    if not np.array_equal(coefficients, coefficients[::-1]):
        message = (
            "the speech-band filter is not symmetric, so its phase is not linear. A "
            "non-linear phase delays one part of the band more than another, which would "
            "make a band-limited level a measurement of a smeared signal rather than of "
            "the candidate it was cut from."
        )
        raise SpeechBandError(message)
    if group_delay != (length - 1) // 2:
        message = (
            f"the file declares a group delay of {group_delay} samples, but a symmetric "
            f"length-{length} filter delays by {(length - 1) // 2}"
        )
        raise SpeechBandError(message)


def _parsed(raw: bytes) -> dict[str, Any]:
    import json

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"the speech-band filter {SPEECH_BAND_PATH} is not valid JSON: {exc}"
        raise SpeechBandError(message) from exc
    if not isinstance(document, dict):
        message = f"the speech-band filter {SPEECH_BAND_PATH} is not a JSON object"
        raise SpeechBandError(message)
    return document


def _mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        message = f"the speech-band filter has no {key!r} object"
        raise SpeechBandError(message)
    return value


def _integer(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"the speech-band filter's {key!r} is {value!r}, expected an integer"
        raise SpeechBandError(message)
    return value


def _number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"the speech-band filter's {key!r} is {value!r}, expected a number"
        raise SpeechBandError(message)
    return float(value)
