"""The automix — M5's stage.

M3 decided *who was speaking* and froze that graph (ADR-0012). This package turns it into a
listenable mono `session.mp3` and nothing else reads it: the mix is the branch that must
survive a transcription failure, and the branch nothing text-derived may reach (INV-09).

The modules follow the data:

* :mod:`~dnd_audio.mix.levels` estimates each track's voice-level correction from that track's
  own `speech_reference_mbfs`, and clamps it.
* :mod:`~dnd_audio.mix.envelope` turns retained candidates into a gain per track per control
  frame — slew-limited ramps, two weight floors, a Dugan-style normalized share — produced in
  bounded chunks with carried state (ADR-0022).
* :mod:`~dnd_audio.mix.render` steps every track's reader and that envelope over the same
  window range and streams one mono float32 intermediate.
* :mod:`~dnd_audio.mix.cache` is that intermediate's identity, so a second run does not re-mix.
* :mod:`~dnd_audio.mix.loudness` measures a file with FFmpeg's `ebur128`, counting the decoded
  samples in the same pass (ADR-0023).
* :mod:`~dnd_audio.mix.encode` encodes, decodes, measures, and walks the gain down under a
  bounded retry budget — failing the stage rather than claiming a compliance it did not see.
* :mod:`~dnd_audio.mix.runner` orchestrates, and owns `mix`.

Two rules hold everywhere in here.

**The invariant is about the coefficient that reaches a sample.** The normalized share sums to
one, but what multiplies a sample is that share times the track's level correction, and only
the second statement bounds anything audible. Both are checked as frames are produced.

**Nothing is session-length.** Not the audio, and not the gains: 1 kHz of control frames over
six tracks and four hours is 690 MB, so the envelope is an iterator with carried slew state,
exactly as the 3:1 decimator carries filter state across windows (INV-07).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "MIX_CACHE_DIRNAME",
    "MIX_SEMANTICS_VERSION",
    "MP3_RELATIVE_PATH",
    "MixNote",
]

#: Session-relative. The one deliverable this stage publishes.
MP3_RELATIVE_PATH: Final = "output/session.mp3"

#: The lossless intermediate, content-addressed. Everything under here is regenerable cache;
#: deleting it costs a re-mix and nothing else. The spec asks for it to be kept "for
#: debugging/cache reuse, not as a required user-facing deliverable", which is what living
#: under `work/cache/` means in this project.
MIX_CACHE_DIRNAME: Final = "work/cache/mix"

#: Bumped when **any** module in this package changes what it would produce from unchanged
#: inputs — the weight rule, the slew limit, the sharing law, the interpolation, or the
#: correction estimator.
#:
#: One version for the package, for the reason M1, M2, M3 and M4 all give: a cache identity
#: that varied one module's version but not another's keeps serving the answer a fixed bug
#: produced.
MIX_SEMANTICS_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class MixNote:
    """Something an operator should look at that did not stop the mix.

    A plain dataclass rather than a pydantic model, because unlike `TimelineNote`,
    `ActivityNote` and `TranscriptNote` this one belongs to no artifact: M5 publishes an MP3
    and a report, and the report is where these end up (ADR-0022). The three attribute names
    are the ones the runners' `_Note` protocol reads.
    """

    code: str
    message: str
    path: str | None = None
