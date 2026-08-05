#!/usr/bin/env python3
"""Regenerate the checked-in integer sine table the marker is synthesized from.

The table lives in the repository rather than being computed at import time so that a
platform's ``libm`` cannot silently change the bytes a frozen marker SHA-256 describes
(ADR-0041). This script is how it got there, and running it is how it changes — which means
a change is a commit with a diff, rather than a different answer on a different machine.

Exactly M2's arrangement for the decimation filter (``scripts/design_fir.py``), including
the division of labour between its two tests:

* ``tests/test_marker_synth.py::TestTheTableIsReproducible`` re-runs this design and
  compares, so a hand-edited entry is caught.
* ``tests/test_marker_synth.py::TestTheTableIsASine`` measures the checked-in array against
  the endpoints, the monotonicity, the symmetry identities the evaluator relies on, and a
  declared maximum error against ``math.sin``. **That is the real acceptance test.** Without
  it, "integer sine table" degrades into an arbitrary array that happens to produce a
  waveform.

The design is computed in :class:`~decimal.Decimal` at high precision rather than in
``float``, so the rounding to an integer is decided far above the last bit a double could
argue about. ``math.sin`` is used only to *check* the result, never to produce it.

Usage: ``uv run python scripts/design_sine_table.py [--check]``
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dnd_audio.determinism import canonical_json  # noqa: E402
from dnd_audio.marker.sine import (  # noqa: E402
    QUARTER_STEPS,
    TABLE_PATH,
    TABLE_SCALE,
    round_half_away,
)

#: Working precision for the series below. Fifty digits against a table scaled by 2**30
#: means the integer rounding is never in doubt: the value being rounded is correct to
#: roughly forty digits more than the one place where it matters.
PRECISION: Final = 50

#: What the table must *be*, independent of how it was computed. Asserted against the
#: checked-in array by the acceptance test.
CONTRACT: Final[dict[str, Any]] = {
    "quarter_steps": QUARTER_STEPS,
    "scale": TABLE_SCALE,
    "domain": "sin(i * pi/2 / quarter_steps) for i in [0, quarter_steps], inclusive",
    "rounding": "half away from zero, once, at design time",
    "max_absolute_error_vs_libm_units": 1,
    "max_linear_interpolation_error_relative": 3.0e-7,
}


def _pi() -> Decimal:
    """Pi to the working precision, by the Chudnovsky-free Machin-like formula.

    Computed rather than pasted so the digits are checkable: a literal would be one more
    thing nobody verifies, and this is the root of every value in the table.
    """
    return 16 * _arctan_inverse(5) - 4 * _arctan_inverse(239)


def _converged(total: Decimal, addend: Decimal) -> Decimal | None:
    """``total + addend``, or ``None`` once the addend no longer changes it.

    The termination rule for both series below, and it is the one that is easy to get
    wrong: a term never becomes *exactly* zero. Decimal's exponent range runs to about
    -1000000, so ``while term != 0`` keeps dividing long after the sum has stopped moving —
    tens of thousands of iterations producing digits that are discarded at the working
    precision. Measured: 49 ms per value against 0.05 ms for this. Converging on the sum
    rather than on the term is both correct and three orders of magnitude cheaper.
    """
    updated = total + addend
    return None if updated == total else updated


def _arctan_inverse(x: int) -> Decimal:
    """``arctan(1/x)`` by its alternating series, to the working precision."""
    total = term = Decimal(1) / x
    x_squared = Decimal(x) * x
    n = 1
    while True:
        term = -term / x_squared
        n += 2
        updated = _converged(total, term / n)
        if updated is None:
            return total
        total = updated


def _sin(x: Decimal) -> Decimal:
    """``sin(x)`` by its Taylor series, for ``x`` in ``[0, pi/2]``.

    The series converges quickly over that range, and the range is never exceeded because
    only a quarter wave is ever generated — the evaluator recovers the rest by symmetry.
    """
    total = term = x
    n = 1
    while True:
        n += 2
        term = -term * x * x / ((n - 1) * n)
        updated = _converged(total, term)
        if updated is None:
            return total
        total = updated


def design() -> list[int]:
    """The quarter wave, as integers scaled by :data:`TABLE_SCALE`."""
    getcontext().prec = PRECISION
    half_pi = _pi() / 2
    values: list[int] = []
    for index in range(QUARTER_STEPS + 1):
        exact = _sin(half_pi * index / QUARTER_STEPS) * TABLE_SCALE
        numerator, denominator = exact.as_integer_ratio()
        values.append(round_half_away(numerator, denominator))
    # The endpoints are what make sin(0) and sin(pi/2) exact rather than interpolated, and
    # they are the two the series is least able to guarantee: the first is exactly zero and
    # the last sits one ulp below a power of two. Pinned here so a precision change cannot
    # quietly move them.
    if values[0] != 0 or values[-1] != TABLE_SCALE:
        message = (
            f"the design produced endpoints {values[0]} and {values[-1]}, expected 0 and "
            f"{TABLE_SCALE}. Raise PRECISION."
        )
        raise ValueError(message)
    return values


def document() -> str:
    """The checked-in file's exact bytes."""
    return canonical_json({"name": "marker_sine_quarter", **CONTRACT, "quarter": design()})


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
        current = TABLE_PATH.read_text(encoding="utf-8") if TABLE_PATH.exists() else ""
        if current == text:
            print(f"  {TABLE_PATH.relative_to(REPO_ROOT)} is current")
            return 0
        print(f"  {TABLE_PATH.relative_to(REPO_ROOT)} differs from this design")
        return 1

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(text, encoding="utf-8")
    print(f"  wrote {TABLE_PATH.relative_to(REPO_ROOT)} ({QUARTER_STEPS + 1} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
