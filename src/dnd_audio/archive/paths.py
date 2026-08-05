"""Turning a filename into an object key, reversibly, over bytes rather than text.

The obvious implementation percent-encodes a `str`, and it is wrong on a filesystem this
project actually runs on. Linux filenames are **bytes**, not text: any byte sequence
without `/` or NUL is a legal name, and plenty of real ones are not valid UTF-8 — a file
copied from a FAT card, a name typed in a different locale, a recorder that wrote Latin-1.
Python surfaces those through surrogate escapes (`surrogateescape`), and
:func:`~dnd_audio.determinism.canonical_json` emits UTF-8 and **raises** on a surrogate.

So a manifest keyed on decoded text cannot represent a file M7a promises to archive, and
the failure would appear as a crash partway through uploading a session containing one
oddly-named file — at the moment the archive was most wanted. Encoding over
:func:`os.fsencode` bytes removes the whole class: the encoded form is pure ASCII by
construction, so it always serializes, and :func:`os.fsdecode` puts the original bytes
back on restore (ADR-0036).

The same encoder handles the session id, which is why this module is not called
``sourceset``. `SessionConfig.session_id` accepts any non-empty string, and the first
draft of this milestone proposed narrowing it. The plan review was right to refuse:
narrowing it would move every processing cache identity, and would make an already valid,
already inspected session called ``Session 01`` unarchivable. It is encoded, not
restricted.
"""

from __future__ import annotations

import os
from typing import Final
from urllib.parse import unquote_to_bytes

from dnd_audio.archive import MAX_OBJECT_KEY_BYTES, ArchiveError

__all__ = [
    "UNRESERVED",
    "decode_component",
    "encode_component",
    "key_length_bytes",
    "require_key_within_limit",
]

#: Bytes that survive encoding unchanged. Deliberately narrower than RFC 3986's unreserved
#: set minus nothing: ``~`` is excluded because some tooling normalizes it, and every other
#: punctuation byte — including ``/`` — is encoded, which is what collapses a whole
#: session-relative path into a single opaque key component. Nothing in a key can then be
#: read as a path separator by a proxy, a CDN, or a filesystem on the way back.
UNRESERVED: Final = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")

#: Uppercase, so encoding is canonical. ``%2f`` and ``%2F`` decode identically but are
#: different object keys, and an archive whose key depends on which one a Python version
#: happened to emit is not content-addressed in any useful sense.
_HEX: Final = "0123456789ABCDEF"


def encode_component(text: str) -> str:
    """Encode one path or session id into a single canonical ASCII key component.

    Reversible by :func:`decode_component` for every possible filename, including ones
    that are not valid UTF-8.

    Args:
        text: A session-relative POSIX path, or a session id. Never a whole key.

    Returns:
        Pure ASCII. Every byte outside :data:`UNRESERVED` becomes ``%XX`` with uppercase
        hex, ``/`` included.
    """
    encoded: list[str] = []
    for byte in os.fsencode(text):
        if byte in UNRESERVED:
            encoded.append(chr(byte))
        else:
            encoded.append(f"%{_HEX[byte >> 4]}{_HEX[byte & 0x0F]}")
    return "".join(encoded)


def decode_component(encoded: str) -> str:
    """Recover the original path or session id from an encoded component.

    The exact inverse of :func:`encode_component`, through the byte layer rather than
    through text — :func:`os.fsdecode` reproduces the surrogate escapes a non-UTF-8 name
    arrived with, so the name written on restore is the name that was archived.

    Raises:
        ArchiveError: if the input is not canonically encoded. A key this project did not
            write is not a key this project will act on: lowercase hex, a stray literal
            byte that should have been escaped, or a truncated escape all mean the value
            came from somewhere else, and decoding it leniently is how a caller string
            becomes an unchecked path.
    """
    if encoded != encode_component(_decode_permissively(encoded)):
        message = (
            f"{encoded!r} is not a canonically encoded archive component. It decodes, but "
            f"not to something this project would have encoded that way — lowercase hex "
            f"or an unescaped byte both do this. Refusing rather than guessing."
        )
        raise ArchiveError(message, code="archive_key_not_canonical")
    return _decode_permissively(encoded)


def _decode_permissively(encoded: str) -> str:
    """Percent-decode to bytes and back to a filesystem string, without the canonical check.

    Split out so :func:`decode_component` can compare a round trip against it. Doing the
    check inline would need the decode twice anyway, and naming it makes clear that this
    one is *not* the safe entry point.
    """
    try:
        raw = unquote_to_bytes(encoded.encode("ascii"))
    except UnicodeEncodeError as exc:
        message = f"an archive key component must be ASCII; {encoded!r} is not"
        raise ArchiveError(message, code="archive_key_not_canonical") from exc
    return os.fsdecode(raw)


def key_length_bytes(key: str) -> int:
    """How long a key is in **UTF-8 bytes**, which is what the limit is stated in."""
    return len(key.encode("utf-8"))


def require_key_within_limit(key: str, *, subject: str) -> str:
    """Return ``key``, or refuse it for being too long.

    Encoding can triple a path's length — every byte of a non-ASCII name becomes three
    characters — against a 1024-byte object-key limit, so this is a bound a real session
    with deeply nested, non-ASCII filenames can actually reach.

    Truncating instead would be catastrophic in the quietest possible way: two long paths
    sharing a prefix would truncate to the same key, and the second upload would overwrite
    the first while both manifests claimed success.

    Raises:
        ArchiveError: naming the entry, because "key too long" without one is unactionable.
    """
    length = key_length_bytes(key)
    if length > MAX_OBJECT_KEY_BYTES:
        message = (
            f"the object key for {subject!r} is {length} bytes, over the "
            f"{MAX_OBJECT_KEY_BYTES}-byte limit. Percent-encoding expands a path by up to "
            f"three times, so a deeply nested or non-ASCII name reaches this. Shorten the "
            f"path inside the session — nothing is truncated, because two long paths "
            f"sharing a prefix would truncate to one key and silently overwrite."
        )
        raise ArchiveError(message, code="archive_key_too_long")
    return key
