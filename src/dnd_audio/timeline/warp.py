"""The seam a future affine drift correction plugs into, unused in the MVP.

The spec is explicit on both halves: "Add a hook/interface for a future affine time warp,
but do not make it an MVP dependency", and "Sample-clock drift correction is a future
enhancement." So this exists, it is wired into the real placement path, and the only
implementation shipped is the identity.

**Wired in, not merely declared.** :func:`~dnd_audio.timeline.origin.determine_origin`
takes a warp and applies it to every placement, so a test can supply a non-identity warp
and watch the timeline move. A protocol nothing calls is decoration, and M0's closeout
records what happens to rails that cannot fire.

**It operates on exact rational time, before quantization.** That is the only place it can
go: a drift correction is a scaling, and applying one to an already-rounded sample index
would accumulate the rounding error it exists to remove (INV-04, ADR-0008).

Why the MVP does not correct drift: jammed timecode is timeline synchronization, not a
shared word clock, and how far the three kits' sample clocks actually diverge over four
hours is bounded by **OQ-006** and accepted for the no-correction MVP. Correcting without new
fixed-endpoint evidence of a material problem would still invent timing (INV-12). M2 warns
instead; see :mod:`~dnd_audio.timeline.syncqa`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Protocol, runtime_checkable

__all__ = ["IdentityWarp", "TimeWarp"]


@runtime_checkable
class TimeWarp(Protocol):
    """Maps a track's elapsed session time to corrected elapsed session time.

    Takes and returns time *relative to session zero*, in exact seconds. Relative because
    a drift correction is proportional to elapsed time, so the origin has to be the thing
    it is measured from; exact because the result is quantized once, afterwards.
    """

    def warp(self, track_id: str, elapsed_seconds: Fraction) -> Fraction: ...


class IdentityWarp:
    """No correction. The MVP's only implementation.

    Deliberately a real class rather than ``None``: a default of ``None`` would put an
    ``if warp is not None`` in the placement path, and the branch that is never taken in
    production is the branch that stops working.
    """

    def warp(self, track_id: str, elapsed_seconds: Fraction) -> Fraction:  # noqa: ARG002
        return elapsed_seconds
