"""The checked-in 48 kHz → 16 kHz decimation filter.

The coefficients are data in the repository, not something designed at import time. A
SciPy upgrade must not silently change what every cached 16 kHz derivative in every
session was built with (INV-08); making the array a committed file turns that into a
commit, with a frequency-response test standing between the commit and the pipeline.

`scripts/design_fir.py` is how the file is produced. Two tests guard it and they guard
different failures: `TestDesignIsReproducible` catches a hand-edited coefficient, and
`TestFrequencyResponse` measures the array against the passband, stopband, ripple, and
attenuation it declares. The second is the real acceptance test — without it, "one fixed
decimator" decays into an arbitrary set of numbers that happens to produce the expected
sample count.

Two properties of the design are load-bearing rather than aesthetic:

* **Odd length, exactly symmetric.** That is what makes the phase linear and the group
  delay an integer instead of a fraction of a sample.
* **The group delay divides by the decimation factor.** 129 samples at 48 kHz is exactly
  43 at 16 kHz, so compensating for it is a slice rather than an interpolation, and
  ``sample16 = sample48 // 3`` is true to the sample rather than nearly true.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from dnd_audio.determinism import sha256_bytes
from dnd_audio.errors import DndAudioError

__all__ = ["FIR_PATH", "DecimationFilter", "FilterError", "load_decimation_filter"]

#: The committed coefficient file. Not loaded at import: `scripts/design_fir.py` imports
#: this module to find out where to *write* it, and a module that reads its own output on
#: import cannot be used to create that output.
FIR_PATH: Final = Path(__file__).parent / "data" / "fir_48k_16k.json"


class FilterError(DndAudioError):
    """The checked-in filter is missing, unreadable, or not the shape it claims."""

    default_code = "decimation_filter_unusable"


@dataclass(frozen=True, slots=True)
class DecimationFilter:
    """One fixed anti-aliasing decimator, and the response contract it is held to."""

    name: str
    coefficients: npt.NDArray[np.float64]
    input_rate: int
    output_rate: int
    decimation: int
    #: Samples of delay the filter introduces, in each grid. Both are integers by design.
    group_delay_input: int
    group_delay_output: int
    #: The declared response. Asserted against the coefficients by `tests/test_fir.py`,
    #: not by this module — a self-check that runs on every import would be a slow way to
    #: discover at runtime what a test discovers at commit time.
    passband_edge_hz: float
    max_passband_ripple_db: float
    stopband_edge_hz: float
    min_stopband_attenuation_db: float
    #: SHA-256 of the file's exact bytes. Part of every derivative's cache identity: a
    #: changed filter must rebuild every derivative it ever produced (INV-08).
    identity: str

    @property
    def length(self) -> int:
        return int(self.coefficients.shape[0])


@functools.cache
def load_decimation_filter() -> DecimationFilter:
    """Read, validate, and cache the checked-in filter.

    Raises:
        FilterError: if the file is absent, malformed, or internally inconsistent — an
            even length, a broken symmetry, or a group delay that is not a whole number of
            output samples. Each of those would make the 48↔16 mapping wrong by a
            fraction of a sample, which is the class of error that is invisible until
            someone compares a transcript against the audio.
    """
    try:
        raw = FIR_PATH.read_bytes()
    except OSError as exc:
        message = (
            f"the decimation filter {FIR_PATH} could not be read: {exc}. "
            f"Regenerate it with `uv run python scripts/design_fir.py`."
        )
        raise FilterError(message) from exc

    document = _parsed(raw)
    design = _mapping(document, "design")
    response = _mapping(document, "response")
    coefficients = np.asarray(document.get("coefficients"), dtype=np.float64)

    decimation = _integer(design, "decimation")
    delay_input = _integer(document, "group_delay_samples_input")
    delay_output = _integer(document, "group_delay_samples_output")
    _validate(coefficients, decimation, delay_input, delay_output)

    return DecimationFilter(
        name=str(document.get("name", FIR_PATH.stem)),
        coefficients=coefficients,
        input_rate=_integer(design, "input_rate"),
        output_rate=_integer(design, "output_rate"),
        decimation=decimation,
        group_delay_input=delay_input,
        group_delay_output=delay_output,
        passband_edge_hz=_number(response, "passband_edge_hz"),
        max_passband_ripple_db=_number(response, "max_passband_ripple_db"),
        stopband_edge_hz=_number(response, "stopband_edge_hz"),
        min_stopband_attenuation_db=_number(response, "min_stopband_attenuation_db"),
        identity=sha256_bytes(raw),
    )


def _validate(
    coefficients: npt.NDArray[np.float64], decimation: int, delay_input: int, delay_output: int
) -> None:
    length = int(coefficients.shape[0])
    if coefficients.ndim != 1 or length == 0:
        message = f"the decimation filter must be a non-empty 1-D array, got shape {coefficients.shape}"
        raise FilterError(message)
    if length % 2 == 0:
        message = (
            f"the decimation filter has {length} taps. An even-length filter has a "
            f"half-sample group delay, which cannot be compensated by a slice."
        )
        raise FilterError(message)
    if not np.array_equal(coefficients, coefficients[::-1]):
        message = (
            "the decimation filter is not symmetric, so its phase is not linear. A "
            "non-linear phase smears transients by a frequency-dependent amount, which "
            "is exactly what a word-timing pipeline must not do."
        )
        raise FilterError(message)
    if delay_input != (length - 1) // 2:
        message = (
            f"the file declares a group delay of {delay_input} samples, but a symmetric "
            f"length-{length} filter delays by {(length - 1) // 2}"
        )
        raise FilterError(message)
    if decimation <= 0 or delay_input % decimation or delay_output != delay_input // decimation:
        message = (
            f"a group delay of {delay_input} input samples is not {delay_output} whole "
            f"output samples at 1/{decimation} rate. The delay must divide by the "
            f"decimation factor, or the two grids cannot be aligned exactly."
        )
        raise FilterError(message)


def _parsed(raw: bytes) -> dict[str, Any]:
    import json

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"the decimation filter {FIR_PATH} is not valid JSON: {exc}"
        raise FilterError(message) from exc
    if not isinstance(document, dict):
        message = f"the decimation filter {FIR_PATH} is not a JSON object"
        raise FilterError(message)
    return document


def _mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        message = f"the decimation filter has no {key!r} object"
        raise FilterError(message)
    return value


def _integer(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"the decimation filter's {key!r} is {value!r}, expected an integer"
        raise FilterError(message)
    return value


def _number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"the decimation filter's {key!r} is {value!r}, expected a number"
        raise FilterError(message)
    return float(value)
