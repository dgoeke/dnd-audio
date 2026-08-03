"""The only text processing in this project, and it is deliberately almost nothing.

The spec is explicit: *"Do not add an LLM prose-cleanup pass in the MVP. Preserve what ASR
produced, with only deterministic whitespace/punctuation normalization necessary for
rendering."* So this module does three things and refuses to do a fourth.

**Normalization is for rendering.** Unicode is put in NFC so two spellings of the same
accented character cannot make one transcript differ from another byte for byte (INV-02); runs
of whitespace — including the newlines a model may emit mid-utterance, which would break the
one-line Markdown format — collapse to single spaces; and a space stranded before a closing
punctuation mark is removed. Nothing decides that a sentence should have ended differently,
nothing capitalizes, nothing spells anything out, and nothing rewrites a word the model chose.

**The comparison key is not the text.** Duplicate collapse asks whether two tracks heard the
same utterance, and that question should not turn on a comma or on which lav's transcript
happened to capitalize a name. So the key casefolds, drops punctuation, and collapses
whitespace — and it is used *only* for comparison. What reaches the transcript is always the
normalized text, never the key.

**Similarity is an integer.** `similarity_permille` quantizes through the project's one ratio
rounding rule, so a threshold comparison cannot depend on binary floating point and the number
recorded beside a collapse decision is the number the decision was made on.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Final

from dnd_audio.activity import PERMILLE, to_permille

__all__ = [
    "comparison_key",
    "normalize_text",
    "similarity_permille",
    "word_count",
]

#: Any run of whitespace, including the newlines and tabs a model may emit.
_WHITESPACE: Final = re.compile(r"\s+")

#: A space before a mark that closes a clause. Removing it is presentation, not editing.
_SPACE_BEFORE_PUNCTUATION: Final = re.compile(r"\s+([,.;:!?%)\]}])")

#: Everything that is not a letter, a digit, or a space, for the comparison key alone.
_NOT_WORD: Final = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """The text as it will be recorded and rendered.

    Deterministic and idempotent: normalizing an already-normalized string returns it
    unchanged, which is what lets a rerun produce byte-identical records (INV-02).
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", normalized)


def comparison_key(text: str) -> str:
    """The form two texts are compared in, and that nothing is ever rendered from.

    Casefolded rather than lowercased: casefolding is the operation defined for comparison,
    and it handles the cases lowercasing does not.
    """
    key = unicodedata.normalize("NFKC", text).casefold()
    key = _NOT_WORD.sub(" ", key)
    return _WHITESPACE.sub(" ", key).strip()


def word_count(text: str) -> int:
    """Words in the comparison key. Zero for text that is only punctuation."""
    key = comparison_key(text)
    return len(key.split()) if key else 0


def similarity_permille(first: str, second: str) -> int:
    """How alike two utterances are, in thousandths, comparing words rather than characters.

    Characters would rate "no" and "now" at 800 per-mille, which is most of the way to a
    collapse threshold for two utterances that mean opposite things. Words are the unit a
    transcript is wrong in.

    ``autojunk`` is off. It is a heuristic that starts ignoring elements appearing in more
    than one percent of a long sequence, which would make the score depend on how long the
    two utterances happen to be — a determinism hazard rather than a correctness one, and
    this is a comparison whose result is written into an artifact.
    """
    left = comparison_key(first).split()
    right = comparison_key(second).split()
    if not left and not right:
        return PERMILLE
    if not left or not right:
        return 0
    ratio = difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    return to_permille(ratio * PERMILLE)
