"""A checked-in integer sine table, so no ``libm`` result reaches a frozen SHA-256.

ADR-0041 freezes the marker's exact integer PCM by content hash. That promise is only
meaningful if the bytes are a function of this repository and not of the platform's maths
library: ``math.sin`` is not required to be correctly rounded, implementations differ in the
last unit in the last place, and one flipped bit after quantization is a different file.

So the quarter wave is **data**, exactly as M2 made the decimation FIR data
(``activity/data/fir_speechband_16k.json``, ``scripts/design_fir.py``), and for the same
reason stated there: making the array a committed file turns a library upgrade into a commit,
with a test standing between the commit and the pipeline. ``scripts/design_sine_table.py`` is
how it is produced.

**The tests are the contract, not the array.** ``tests/test_marker_synth.py`` asserts the
endpoints, the quarter wave's monotonicity, the symmetry identities :func:`sine_scaled`
relies on, and a stated maximum error against ``math.sin``. It deliberately does not compare
the table to a stored copy of itself, which would only prove the file had not been edited.

Everything here is exact integer arithmetic. :func:`sine_at` returns a *numerator over a
caller-supplied denominator* rather than a rounded value, so a caller composing several
factors — a chirp's sine, an envelope, a peak amplitude — rounds **once**, at the end, where
fixed point becomes an integer sample.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Final

from dnd_audio.errors import DndAudioError

__all__ = [
    "QUARTER_STEPS",
    "TABLE_PATH",
    "TABLE_SCALE",
    "TURN_STEPS",
    "SineTable",
    "SineTableError",
    "load_sine_table",
    "round_half_away",
]

#: The committed table. Not read at import: ``scripts/design_sine_table.py`` imports this
#: module to learn where to *write*, and a module that reads its own output on import cannot
#: be used to produce it (the note ``timeline/fir.py`` records about ``FIR_PATH``).
TABLE_PATH: Final = Path(__file__).parent / "data" / "sine_table.json"

#: Entries per quarter turn. The table holds ``QUARTER_STEPS + 1`` values covering
#: ``[0, pi/2]`` inclusive, so both endpoints are exact rather than interpolated.
#:
#: Linear interpolation of a sine over a step ``h`` has worst-case error ``h**2 / 8``. At
#: 1024 steps per quarter, ``h = 2*pi/4096`` and the error is 2.9e-7 — about 130 dB below
#: full scale, and two orders of magnitude below a 16-bit sample's own quantization step.
#: Chosen so the table is never the limiting factor and never has to be revisited.
QUARTER_STEPS: Final = 1024

#: Steps in a full turn. The phase domain every caller indexes into.
TURN_STEPS: Final = 4 * QUARTER_STEPS

#: Fixed-point scale of a table entry: ``sin(x) * TABLE_SCALE``, rounded once at design
#: time. A power of two so that scaling by it is exact.
TABLE_SCALE: Final = 1 << 30


class SineTableError(DndAudioError):
    """The checked-in sine table is missing, unreadable, or not the shape it claims."""

    default_code = "sine_table_unusable"


def round_half_away(numerator: int, denominator: int) -> int:
    """``numerator / denominator`` as an integer, halves away from zero.

    The project's stated tie rule (``determinism._quantize``), restated here for *amplitude*
    rather than time. INV-04's "exactly one quantizer" is a rule about the sample grid —
    ``determinism.to_samples`` — and this is the amplitude counterpart, the same role
    ``fixtures/wav.quantize`` plays for fixture audio. Naming it once and calling it from one
    place in :mod:`dnd_audio.marker.synth` is what keeps that from becoming two rules.

    Python's ``round`` is banker's rounding and ``//`` floors toward negative infinity, so
    neither is usable for a value that must round the same way on both sides of zero.

    Raises:
        ValueError: if ``denominator`` is not positive. A negative denominator would mirror
            the tie rule, which is exactly the kind of silent asymmetry this exists to avoid.
    """
    if denominator <= 0:
        message = f"denominator must be positive, got {denominator}"
        raise ValueError(message)
    magnitude, remainder = divmod(abs(numerator), denominator)
    if 2 * remainder >= denominator:
        magnitude += 1
    return -magnitude if numerator < 0 else magnitude


class SineTable:
    """One quarter wave of integer sine values, and exact evaluation over a full turn.

    Immutable and cached: ``load_sine_table()`` returns the same instance every call, because
    every chirp sample in every candidate reads it.
    """

    __slots__ = ("_quarter", "_sha256")

    def __init__(self, quarter: tuple[int, ...], sha256: str) -> None:
        self._quarter = quarter
        self._sha256 = sha256

    @property
    def sha256(self) -> str:
        """Digest of the committed file, so it can enter a marker's recorded identity."""
        return self._sha256

    @property
    def quarter(self) -> tuple[int, ...]:
        """``QUARTER_STEPS + 1`` values, ``sin(i * pi/2 / QUARTER_STEPS) * TABLE_SCALE``."""
        return self._quarter

    def value(self, step: int) -> int:
        """``sin(2*pi*step / TURN_STEPS) * TABLE_SCALE`` for any integer ``step``.

        Quarter-wave symmetry, written out rather than folded into an index expression,
        because the folded form is where the sign of the third quadrant goes wrong and no
        test that only checks a chirp's spectrum would notice.
        """
        step %= TURN_STEPS
        quadrant, offset = divmod(step, QUARTER_STEPS)
        if quadrant == 0:
            return self._quarter[offset]
        if quadrant == 1:
            return self._quarter[QUARTER_STEPS - offset]
        if quadrant == 2:
            return -self._quarter[offset]
        return -self._quarter[QUARTER_STEPS - offset]

    def sine_at(self, phase_numerator: int, phase_denominator: int) -> tuple[int, int]:
        """Sine of ``phase_numerator / phase_denominator`` **turns**, as an exact fraction.

        Returns ``(numerator, denominator)`` with the value scaled by :data:`TABLE_SCALE`:
        the sine is ``numerator / denominator / TABLE_SCALE``. Nothing is rounded here.

        Returning a fraction rather than a rounded integer is the whole point. A caller
        multiplies this by an envelope and a peak amplitude and rounds the product **once**
        (ADR-0041); rounding at each factor instead would be three tie rules where the ADR
        promises one, and the error would be systematic rather than random because a chirp's
        phase advances monotonically.

        Raises:
            ValueError: if ``phase_denominator`` is not positive.
        """
        if phase_denominator <= 0:
            message = f"phase_denominator must be positive, got {phase_denominator}"
            raise ValueError(message)

        # One integer division carries the phase onto the table's grid; the remainder is
        # the interpolation weight, kept exact rather than converted to a float.
        step, remainder = divmod(phase_numerator * TURN_STEPS, phase_denominator)
        low = self.value(step)
        high = self.value(step + 1)
        return low * phase_denominator + (high - low) * remainder, phase_denominator


@functools.lru_cache(maxsize=1)
def load_sine_table() -> SineTable:
    """Read and validate the committed table.

    Raises:
        SineTableError: if the file is absent, unparseable, the wrong length, wrongly
            scaled, or does not start and end where a quarter wave must.
    """
    from dnd_audio.determinism import sha256_bytes

    try:
        raw = TABLE_PATH.read_bytes()
    except OSError as exc:
        message = (
            f"the sine table at {TABLE_PATH} cannot be read: {exc}. Regenerate it with "
            f"`uv run python scripts/design_sine_table.py`."
        )
        raise SineTableError(message) from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"the sine table at {TABLE_PATH} is not valid JSON: {exc}"
        raise SineTableError(message) from exc

    if not isinstance(document, dict) or "quarter" not in document:
        message = f"the sine table at {TABLE_PATH} has no `quarter` array"
        raise SineTableError(message)

    values = document["quarter"]
    if not isinstance(values, list) or not all(isinstance(item, int) for item in values):
        message = f"the sine table at {TABLE_PATH} must hold a list of integers"
        raise SineTableError(message)

    quarter = tuple(int(item) for item in values)
    _validate(quarter, document)
    return SineTable(quarter, sha256_bytes(raw))


def _validate(quarter: tuple[int, ...], document: dict[str, object]) -> None:
    """Refuse a table that is the wrong shape, before anything is built from it.

    Checked rather than trusted because this file is the root of a frozen content hash: a
    truncated or rescaled table would still produce *a* waveform, and the first sign of
    trouble would be a golden hash that moved for no stated reason.
    """
    declared_steps = document.get("quarter_steps")
    declared_scale = document.get("scale")
    if declared_steps != QUARTER_STEPS or declared_scale != TABLE_SCALE:
        message = (
            f"the sine table at {TABLE_PATH} declares quarter_steps={declared_steps!r} and "
            f"scale={declared_scale!r}, but this build uses {QUARTER_STEPS} and "
            f"{TABLE_SCALE}. Regenerate it with `uv run python scripts/design_sine_table.py`."
        )
        raise SineTableError(message)

    if len(quarter) != QUARTER_STEPS + 1:
        message = (
            f"the sine table at {TABLE_PATH} holds {len(quarter)} values; a quarter wave "
            f"inclusive of both endpoints is {QUARTER_STEPS + 1}"
        )
        raise SineTableError(message)

    if quarter[0] != 0 or quarter[-1] != TABLE_SCALE:
        message = (
            f"the sine table at {TABLE_PATH} runs from {quarter[0]} to {quarter[-1]}; a "
            f"quarter wave runs from 0 to {TABLE_SCALE} exactly. Those two endpoints are "
            f"what make sin(0) and sin(pi/2) exact rather than interpolated."
        )
        raise SineTableError(message)
