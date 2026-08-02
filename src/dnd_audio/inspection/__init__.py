"""Discovery, capture, and the deterministic ingest manifest — M1's stage.

The order the modules are built in is the order the data flows:

* :mod:`~dnd_audio.inspection.riff` walks a file's chunk structure without decoding it.
* :mod:`~dnd_audio.inspection.naming` reads what a DJI filename *hints* at. Hints only:
  the configured directory is the identity (INV-11).
* :mod:`~dnd_audio.inspection.probe` runs `ffprobe` and keeps its output verbatim.
* :mod:`~dnd_audio.inspection.starttime` turns captured metadata into typed timing
  evidence through a named strategy chain, and never invents timing (INV-12).
* :mod:`~dnd_audio.inspection.discovery` applies the selection and roster rules.
* :mod:`~dnd_audio.inspection.cache` decides when captured work may be reused (INV-08).
* :mod:`~dnd_audio.inspection.runner` orchestrates, writes the manifest, and contributes
  to the report.

Nothing here modifies anything under a session's ``raw/`` (INV-01), and nothing here
decodes audio: the container and the RIFF ``data`` size already state the sample count.
"""

from __future__ import annotations

from typing import Final

__all__ = ["INSPECTION_SEMANTICS_VERSION", "OUTPUT_DIRNAME", "WORK_DIRNAME"]

#: The two session-relative directories this pipeline generates. Named here rather than
#: in the runner because discovery needs them too: when a track's input sits directly in
#: the session root, these are siblings of the track directories and must not be mistaken
#: for unconfigured source directories.
WORK_DIRNAME: Final = "work"
OUTPUT_DIRNAME: Final = "output"

#: Bumped when **any** module in this package changes what it would produce from
#: unchanged bytes.
#:
#: One version rather than one per module, because the alternative failed review: a
#: cache identity that varied the RIFF-parser version but not the strategy chain would
#: keep serving the answer a fixed bug produced. INV-08 asks for "the
#: implementation/schema version", and the implementation is the whole package — probe
#: parsing, filename hints, the chunk walk, and start-time extraction all feed one
#: cached record, so they share one number.
#:
#: Bump this for a behaviour change, not for a comment or a refactor that provably
#: cannot alter output. When in doubt, bump: the cost is re-probing a few dozen files.
INSPECTION_SEMANTICS_VERSION: Final = 1
