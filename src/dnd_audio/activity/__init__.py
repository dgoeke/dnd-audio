"""Speech activity and bleed rejection — M3's stage.

M2 reconstructed *where every sample sits*. This package decides *who was speaking*, and
does it without ever looking at text: the graph it produces is the model-independent
contract the automixer consumes, so nothing an ASR model says may flow back into it
(INV-09). The modules are ordered the way the data flows:

* :mod:`~dnd_audio.activity.detect` runs an :class:`~dnd_audio.interfaces.ActivityDetector`
  over a track's 16 kHz derivative in bounded windows and turns per-frame probabilities into
  padded, merged speech candidates.
* :mod:`~dnd_audio.activity.silero` is the real detector: a commit-pinned ONNX artifact under
  a pinned runtime, with no Torch anywhere in the process (ADR-0013).
* :mod:`~dnd_audio.activity.band` band-limits to the speech band through a checked-in filter,
  so a level comparison is about voices rather than about room rumble.
* :mod:`~dnd_audio.activity.scoring` combines four pieces of evidence into one source score.
* :mod:`~dnd_audio.activity.bleed` compares overlapping candidates across tracks and
  suppresses only what is convincingly someone else's voice (ADR-0014).
* :mod:`~dnd_audio.activity.runner` orchestrates and writes ``work/activity.json``.

Two rules hold everywhere in here. **Nothing text-derived enters the graph** — that is the
invariant this milestone exists to freeze. And **losing real overlapped speech is worse than
extra ASR compute**, which is why every ambiguous case is retained and marked rather than
resolved.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "ACTIVITY_CACHE_DIRNAME",
    "ACTIVITY_RELATIVE_PATH",
    "ACTIVITY_SEMANTICS_VERSION",
    "ATTRIBUTION_DIRNAME",
    "DETECTION_DIRNAME",
    "DETECTOR_CONTEXT_SAMPLES",
    "DETECTOR_FRAME_SAMPLES",
    "PERMILLE",
    "to_permille",
    "to_permille_array",
]

#: Full scale for every probability, score, and correlation in this package. Integers,
#: because they reach a byte-stable artifact and a float that is the quotient of two
#: reductions is not reliably identical across a library upgrade (INV-02).
PERMILLE: Final = 1000

#: Session-relative. The activity graph, and the only artifact this stage publishes.
ACTIVITY_RELATIVE_PATH: Final = "work/activity.json"

#: Everything under here is regenerable cache; deleting it costs a rebuild and nothing else.
ACTIVITY_CACHE_DIRNAME: Final = "work/cache/activity"

#: Per-track detection: the candidates and the per-frame probabilities behind them.
DETECTION_DIRNAME: Final = f"{ACTIVITY_CACHE_DIRNAME}/detect"

#: Per-session attribution: the assembled graph, keyed on every detection that fed it.
ATTRIBUTION_DIRNAME: Final = f"{ACTIVITY_CACHE_DIRNAME}/graph"

#: Silero's frame, in derivative samples: 512 at 16 kHz is 32 ms. Fixed by the model rather
#: than chosen by us — the ONNX graph rejects any other length (ADR-0013).
DETECTOR_FRAME_SAMPLES: Final = 512

#: Samples of the previous frame prepended to each one. Also the model's, not ours.
DETECTOR_CONTEXT_SAMPLES: Final = 64

#: Bumped when **any** module in this package changes what it would produce from unchanged
#: inputs — a threshold's meaning, the region assembler's boundary handling, the scoring
#: function, or the suppression rule.
#:
#: One version for the package, for the reason M1 and M2 both give: a cache identity that
#: varied one module's version but not another's keeps serving the answer a fixed bug
#: produced.
ACTIVITY_SEMANTICS_VERSION: Final = 2


def to_permille(value: float) -> int:
    """Quantize a ratio in [0, 1] to per-mille, halves away from zero, clamped.

    **One rule for the whole package.** Four modules produce per-mille values — the
    rasterizer, the Silero adapter, the scorer, and the bleed gate — and NumPy's ``rint`` and
    Python's ``round`` are both half-to-*even*, so mixing them with an away-from-zero rule
    would make two quantities a thousandth apart order differently depending on which module
    computed them. INV-04 states that argument for *time* and names a second quantizer as how
    the rule dies; the same reasoning applies here, to a different unit, which is why this is
    one function rather than four call sites that happen to agree today.

    Deliberately not :mod:`dnd_audio.determinism`'s: that one takes an exact
    :class:`~fractions.Fraction` and a rate, because time must never accumulate in floats. A
    ratio of two measured energies is a float already, and pretending otherwise by routing it
    through a rational would be a lie about where the precision went.
    """
    return max(0, min(PERMILLE, int(abs(value) + 0.5) if value >= 0 else -int(-value + 0.5)))


def to_permille_array(values: npt.NDArray[np.float64]) -> npt.NDArray[np.uint16]:
    """:func:`to_permille` over an array of non-negative ratios.

    ``floor(x + 0.5)`` rather than ``rint``: identical for every value except an exact half,
    and an exact half is the whole reason this function exists.
    """
    return np.clip(np.floor(values + 0.5), 0, PERMILLE).astype(np.uint16)
