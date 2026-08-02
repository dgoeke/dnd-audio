"""What a DJI filename *hints* at. Hints only.

Two rules govern everything in this module, and both are invariants rather than style
preferences:

**A filename is never an identity (INV-11).** ``TX01`` is a receiver-assigned
pairing-order label; DJI documents it as changing after re-pairing, so two independent
kits can both produce one — the canonical fixture deliberately has three ``TX01``s. The
configured directory is the track. Nothing here returns a ``track_id``, and nothing here
can be turned into one.

**A filename is never a time (INV-12).** The date and time components are recorded for
diagnostics and for cross-checking, and are never a timing source. A file whose only
apparent timing is in its name has no timing, and inspection says so.

**And the grammar is a guess** until the H1 fixture lands (OQ-003). Parsing is therefore
fail-soft in one specific direction: an unrecognized name yields hints that say so, and
the file remains a candidate. If this grammar became an inclusion filter, real hardware
whose names differ from the guess would be silently omitted — the worst possible failure
for a milestone whose job is to notice what is there.
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "AUDIO_SUFFIXES",
    "FilenameHints",
    "Variant",
    "parse_filename",
    "sequence_discontinuities",
]

Variant = Literal["orig", "edit", "unknown"]

#: What discovery considers a candidate at all. Deliberately by extension rather than by
#: name shape: see the module docstring.
AUDIO_SUFFIXES: Final = frozenset({".wav"})

#: The grammar OQ-003 assumes: ``TX##_MIC###_YYYYMMDD_HHMMSS_<variant>.wav``.
_DJI_NAME: Final = re.compile(
    r"^(?P<tx>TX\d{2})"
    r"_MIC(?P<sequence>\d{3,})"
    r"_(?P<date>\d{8})"
    r"_(?P<time>\d{6})"
    r"_(?P<variant>orig|edit)$",
    re.IGNORECASE,
)

#: Recognizing the variant on its own, for a name the full grammar does not match. A
#: recorder that changes its naming is far more likely to keep this suffix than to keep
#: the whole shape, and getting it right is what keeps a processed file from being
#: consumed as if it were the original.
_VARIANT_SUFFIX: Final = re.compile(r"_(?P<variant>orig|edit)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FilenameHints:
    """Everything a filename suggests, and nothing it decides."""

    filename: str
    #: Whether the whole assumed grammar matched. False is not an error.
    recognized: bool
    variant: Variant
    #: DJI's pairing-order label. A validation hint, never an identity (INV-11).
    tx_label: str | None = None
    #: The monotonic counter. A secondary ordering hint only — chunk order comes from
    #: embedded timecode, not from this (OQ-003).
    sequence: int | None = None
    #: Parsed from the name, and never usable as a timing source (INV-12).
    named_date: dt.date | None = None
    named_time: dt.time | None = None

    @property
    def stem_is_dji_shaped(self) -> bool:
        return self.recognized


def parse_filename(filename: str) -> FilenameHints:
    """Read a source filename for hints.

    Never raises. A name this does not understand is still a file that exists, and
    inspection's job is to describe it rather than to decline to look at it.
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename

    matched = _DJI_NAME.match(stem)
    if matched is None:
        loose = _VARIANT_SUFFIX.search(stem)
        variant: Variant = _as_variant(loose.group("variant")) if loose else "unknown"
        return FilenameHints(filename=filename, recognized=False, variant=variant)

    return FilenameHints(
        filename=filename,
        recognized=True,
        variant=_as_variant(matched.group("variant")),
        tx_label=matched.group("tx").upper(),
        sequence=int(matched.group("sequence")),
        named_date=_parse_date(matched.group("date")),
        named_time=_parse_time(matched.group("time")),
    )


def sequence_discontinuities(hints: list[FilenameHints]) -> tuple[tuple[int, int], ...]:
    """Gaps in one transmitter's counter, as ``(before, after)`` pairs.

    The spec asks for a warning on sequence discontinuities, which is a diagnostic about
    *files that may be missing*, not an ordering mechanism. A power cycle can legitimately
    produce one, so it warns rather than failing.

    Only recognized names participate: a counter this module could not read is not
    evidence of anything.
    """
    numbers = sorted(hint.sequence for hint in hints if hint.sequence is not None)
    return tuple(
        (earlier, later) for earlier, later in itertools.pairwise(numbers) if later - earlier > 1
    )


def _as_variant(text: str) -> Variant:
    return "edit" if text.lower() == "edit" else "orig"


def _parse_date(text: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(text, "%Y%m%d").replace(tzinfo=dt.UTC).date()
    except ValueError:
        return None


def _parse_time(text: str) -> dt.time | None:
    try:
        return dt.datetime.strptime(text, "%H%M%S").replace(tzinfo=dt.UTC).time()
    except ValueError:
        return None
