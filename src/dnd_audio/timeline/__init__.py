"""The session timeline — M2's stage.

M1 captured *evidence* and deliberately did not place it (ADR-0006). This package does the
placing, and the modules are ordered the way the data flows:

* :mod:`~dnd_audio.timeline.rasterize` turns each kind of evidence into exact rational
  time and, once, into an integer sample position (ADR-0008).
* :mod:`~dnd_audio.timeline.origin` decides where session zero is and unwraps the 24-hour
  cycle in each evidence domain's own units (ADR-0009).
* :mod:`~dnd_audio.timeline.layout` orders a track's chunks, preserves real gaps, and
  applies the overlap policy (ADR-0010).
* :mod:`~dnd_audio.timeline.pcm` and :mod:`~dnd_audio.timeline.reader` read the audio the
  map points at, in bounded windows (ADR-0011, INV-07).
* :mod:`~dnd_audio.timeline.resample` derives the 16 kHz working audio with one fixed
  filter, run across the whole virtual track.
* :mod:`~dnd_audio.timeline.runner` orchestrates and writes ``work/timeline.json``.

Two rules hold everywhere in here. **Time is exact until the last step** — `Fraction`
internally, one quantizer, no accumulated running totals (INV-04). **Nothing is
session-length** — every audio path is a bounded window over a segment map, never an
array (INV-07).
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "DERIVATIVE_SAMPLE_RATE",
    "TIMELINE_DIRNAME",
    "TIMELINE_RELATIVE_PATH",
    "TIMELINE_SEMANTICS_VERSION",
]

#: The rate the mix and the timeline are expressed at. A selected source that is not this
#: rate is fatal here, where M1 only warned: resampling a lossless timeline silently is
#: not on offer.
CANONICAL_SAMPLE_RATE: Final = 48000

#: What VAD and ASR consume. Exactly one third of the working rate, which is what makes
#: the mapping between the two grids an integer relationship rather than an approximation.
DERIVATIVE_SAMPLE_RATE: Final = 16000

#: Session-relative. The authoritative segment map.
TIMELINE_RELATIVE_PATH: Final = "work/timeline.json"

#: Where derived and materialized audio lives. Everything under it is regenerable cache;
#: deleting it costs a rebuild and nothing else (ADR-0011).
TIMELINE_DIRNAME: Final = "work/cache/audio"

#: Bumped when **any** module in this package changes what it would produce from unchanged
#: inputs — a placement rule, the overlap policy, the PCM reader's conventions, or the
#: resampler's boundary handling.
#:
#: One version for the package, for the reason M1 gives for
#: :data:`~dnd_audio.inspection.INSPECTION_SEMANTICS_VERSION`: a cache identity that
#: varied one module's version but not another's keeps serving the answer a fixed bug
#: produced. Bump it for a behaviour change; the cost of being wrong is rebuilding
#: derivatives that are regenerable by construction.
TIMELINE_SEMANTICS_VERSION: Final = 2
