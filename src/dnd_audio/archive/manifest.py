"""`archive-manifest.v1.json` — the commit marker, and the only small object per session.

Cold Storage bills anything under 128 KiB as 128 KiB, so a session gets exactly one such
object and it has to carry everything a recovery needs. That budget is why there are no
per-file sidecars, no verification receipts, and no uploaded reports: multiplying a billing
floor across dozens of tiny objects buys nothing that this one document cannot hold.

**It is written last**, only after every object has passed both a local round trip and a
complete remote readback (ADR-0038). An interrupted upload therefore leaves objects with no
manifest, which `status` reports as `pending` rather than as an archive.

**It must be usable without this repository.** The person restoring may be doing so years
later, from a laptop that has never seen this project, possibly without Python. So the
document carries explicit decoding and path-reconstruction instructions in prose, and the
compressed objects are plain zstd frames that `zstd -d` reads. A manifest that could only be
understood by the program that wrote it would be a single point of failure sitting inside a
disaster-recovery mechanism.

**Paths are the encoded byte form.** ``path`` is pure ASCII and always serializable;
``path_text`` appears only when the name is valid UTF-8 and is decoration for humans.
Restore reconstructs from ``path``, never from ``path_text`` — a filename that is not valid
UTF-8 has no faithful text form, and :func:`~dnd_audio.determinism.canonical_json` would
refuse to serialize its surrogates at all (ADR-0036).

Nothing here is a timestamp, a hostname, a credential, a signed URL, an ETag, or a local
absolute path. The manifest cannot contain its own hash — ADR-0003's fixed-point problem,
one level up from `ingest-report.json` — so each local operation report records it instead.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dnd_audio.archive import ARCHIVE_VERSION

__all__ = [
    "ARCHIVE_MANIFEST_SCHEMA_VERSION",
    "RESTORE_INSTRUCTIONS",
    "ArchiveManifest",
    "ArchiveManifestEntry",
]

#: Bumped when the *shape* of this document changes. Distinct from
#: :data:`~dnd_audio.archive.ARCHIVE_VERSION`, which changes when the compressed bytes or
#: the key layout do: a manifest can gain an optional field without any stored object
#: becoming unreadable, and conflating the two would force a pointless re-upload.
ARCHIVE_MANIFEST_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

#: Carried verbatim in every manifest. Prose rather than a schema field because its reader
#: is a person under pressure, and because a recovery that depends on this project still
#: existing is not a recovery. Deliberately mentions the percent-encoding rule and the
#: uppercase-hex convention, which are the two things nobody would guess.
RESTORE_INSTRUCTIONS: Final = (
    "Each entry's object holds one original file compressed as a single standard zstd "
    "frame. To restore by hand, download the object at `object_key` and run "
    "`zstd -d < OBJECT > FILE`; any zstd 1.4 or later can read it, and the archive's own "
    "compression settings do not affect decoding. Verify the result with "
    "`sha256sum FILE` against `sha256`, and check its length against `size_bytes`. "
    "Recreate the file at the session-relative path in `path`, which is percent-encoded "
    "over raw filesystem bytes: every byte outside A-Z a-z 0-9 . _ - is written as %XX "
    "with uppercase hex, including the / separators. Decode it by replacing each %XX with "
    "that byte and writing the result as a filename; do not decode it as text first, "
    "because a filename need not be valid UTF-8. `path_text` is a human-readable copy and "
    "is absent when no faithful one exists — never restore from it."
)


class ArchiveManifestEntry(BaseModel):
    """One archived file: where it came from, what it is, and where its bytes are."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Percent-encoded over filesystem bytes. Authoritative, and always ASCII.
    path: str = Field(min_length=1)
    #: The same path as text, when the filename is valid UTF-8. Decoration only.
    path_text: str | None = None
    #: Only where the file belongs unambiguously to one configured track (INV-11). Absent
    #: for nested notes, unassigned audio, and anything a session did not configure — and
    #: absent is a real answer, not a gap to fill in.
    track_id: str | None = None
    size_bytes: int = Field(ge=0)
    sha256: Sha256Hex
    compressed_size_bytes: int = Field(ge=0)
    compressed_sha256: Sha256Hex
    #: Immutable, and derivable from the fields above — recorded anyway so a recovery does
    #: not have to reimplement this project's key construction to find a byte.
    object_key: str = Field(min_length=1)


class ArchiveManifest(BaseModel):
    """The deterministic commit record for one session's archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = ARCHIVE_MANIFEST_SCHEMA_VERSION
    #: The key-layout and encoding-recipe version, which is what the prefix encodes.
    archive_version: Literal[1] = ARCHIVE_VERSION
    #: As configured, not encoded. The encoded form is in the object keys; a human reading
    #: this document should see the name they gave the session.
    session_id: str = Field(min_length=1)
    #: The complete pinned encoder recipe, from `ArchiveCodec.describe()`.
    codec: dict[str, str | int | bool]
    entries: list[ArchiveManifestEntry] = Field(min_length=1)
    restore_instructions: str = RESTORE_INSTRUCTIONS

    @model_validator(mode="after")
    def _sort_and_check(self) -> Self:
        """Sort by path, and refuse a document that describes an impossible archive.

        Sorting during validation rather than at the call site for `Manifest`'s reason: no
        future caller can forget, and directory iteration order can never reach the bytes
        (INV-02).
        """
        object.__setattr__(self, "entries", sorted(self.entries, key=lambda item: item.path))

        paths = [entry.path for entry in self.entries]
        if len(set(paths)) != len(paths):
            duplicated = sorted({path for path in paths if paths.count(path) > 1})
            message = (
                f"the manifest lists {', '.join(duplicated)} more than once. Two entries "
                f"for one path would restore whichever came last, silently."
            )
            raise ValueError(message)

        keys = [entry.object_key for entry in self.entries]
        if len(set(keys)) != len(keys):
            # Reachable only through an encoding collision, which is the failure the key
            # length bound exists to prevent. Checked anyway: the cost of being wrong here
            # is one file overwriting another in the bucket, and nothing else would notice.
            message = (
                "two manifest entries share an object key, so one file's bytes would "
                "overwrite another's. This should be unreachable — the key encoding is "
                "injective and over-long keys are refused — so treat it as a defect in "
                "the encoder rather than as bad input."
            )
            raise ValueError(message)
        return self

    def entry_for(self, path: str) -> ArchiveManifestEntry | None:
        """The entry for an encoded path, or ``None``."""
        return next((entry for entry in self.entries if entry.path == path), None)

    def for_track(self, track_id: str) -> list[ArchiveManifestEntry]:
        """Entries genuinely attributed to ``track_id``. Never a near-miss."""
        return [entry for entry in self.entries if entry.track_id == track_id]

    @property
    def total_original_bytes(self) -> int:
        """What a whole-session restore writes. What restore preflights against."""
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def total_compressed_bytes(self) -> int:
        """What a whole-session verify downloads. What its cost message is built from."""
        return sum(entry.compressed_size_bytes for entry in self.entries)
