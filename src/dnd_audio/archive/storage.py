"""The seam between what the archive means and how one provider spells it.

Narrow on purpose. Four operations, and the shape of them is decided by what the commit
protocol needs rather than by what S3 offers: put an object, look at whether one exists,
read one back completely, and enumerate what is there. Everything provider-specific —
signing, multipart thresholds, part sizes, marker pagination, `503` backoff — lives behind
this in the adapter, because it is exactly the code that cannot be validated without a
live endpoint.

**There is no delete operation here, and that is load-bearing but not sufficient.** The
application must never delete a committed object (ADR-0035). Omitting the method from this
protocol expresses the intent, and a test that greps for `DeleteObject` would appear to
enforce it — but boto3 spells the call `client.delete_object(...)`, so such a test passes
with the forbidden call sitting in the adapter, and the adapter can reach its own client
regardless of what this protocol says. The real enforcement is a recording client under an
operation allowlist, in `tests/test_archive_spaces.py`. This protocol is the statement of
intent; that test is the proof.

**Reads are streams, not bytes.** `open_object` yields something with a bounded
`read(size)` and nothing else, so a four-hour recording's object cannot be materialized by
a caller that meant well. `BinaryReader` is the same one-method protocol
`dnd_audio.determinism` uses for hashing, which is what lets a remote body and a local file
share the decompression path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dnd_audio.determinism import BinaryReader

__all__ = ["ArchiveStorage", "ObjectHead"]


@dataclass(frozen=True, slots=True)
class ObjectHead:
    """What a cheap existence check can honestly report.

    Deliberately carries no digest field. An S3 ETag is the MD5 for a single PUT and, for a
    multipart object, a hash *of the part hashes* with a `-N` suffix — a value that depends
    on how the upload was chunked and identifies nothing about the content. Providing it
    here as `sha256` or even as `checksum` would invite exactly the shortcut the commit
    protocol forbids (ADR-0038), so the field is named for what it is and the size is the
    only number a caller may reason about.
    """

    key: str
    size_bytes: int
    #: Opaque provider metadata. **Never a content checksum.** Useful only for logging
    #: a mismatch a human might recognize.
    etag: str | None = None


class ArchiveStorage(Protocol):
    """Object storage, as much of it as this milestone is allowed to touch."""

    def put_object(self, key: str, source: Path) -> None:
        """Upload ``source``'s bytes to ``key``, streaming.

        Multipart above the provider's limit is the implementation's business, not the
        caller's: the threshold, the minimum part size and the maximum part count are
        provider facts, and a caller that knew them would be a second place for them to
        drift.

        An existing key is **overwritten**. That is safe only because every key this
        project writes is content-addressed on the original digest, so identical bytes land
        on the same key and different bytes land on a different one — and because the
        caller has already verified any existing object before deciding to write. Nothing
        here should be read as permission to overwrite an arbitrary key.
        """
        ...

    def head_object(self, key: str) -> ObjectHead | None:
        """Whether an object exists, and how large it is. ``None`` if it does not."""
        ...

    def open_object(self, key: str) -> AbstractContextManager[BinaryReader]:
        """Open ``key`` for a complete streamed read.

        Raises:
            ArchiveError: if the object does not exist. An absent object during
                verification is a failure, not an empty stream.
        """
        ...

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Every key under ``prefix``, following pagination to exhaustion.

        Completeness is the contract, and it is the reason this returns an iterator rather
        than accepting a page token: a caller that could stop early is a caller that will,
        and a partial listing reported as complete makes `list` say a session is absent
        when it is not.

        DigitalOcean's documentation contradicts itself about which listing API paginates
        — the compatibility page says `ListObjectsV2` is supported, the limits page's known
        issues say its pagination is not — so the adapter uses legacy marker pagination
        outright (**OQ-028**).
        """
        ...
