"""Archive v1's frozen compression recipe, and the ceilings around it.

Three properties, and each exists because its absence is a specific failure:

**The recipe is frozen and single-threaded** (ADR-0037). The four-file trial used
``zstd -T0 -10``, and ``-T0`` asks libzstd for one worker per core — which partitions the
input differently on a 32-core box than on a 4-core one and produces different bytes. Both
decompress correctly, and both are wrong for an archive whose object key is
content-addressed and whose manifest records a compressed digest. ``threads=0`` here means
*no worker threads*, which is the deterministic setting.

**Decompression carries an output ceiling.** A corrupt or hostile frame can declare a
small content size and expand without bound; discovering that at the final hash means it
has already been written or held. The counter is checked as each chunk emerges, so the
abort happens at the ceiling rather than after it.

**Nothing here holds a file.** Every function streams in bounded chunks, because the
inputs are multi-gigabyte recordings and INV-07 is not satisfied by a component that
merely *could* stream — `tests/test_archive_memory.py` asserts the interleaving over the
composed path.

`zstandard` is imported at module scope rather than lazily: this module is only reached
from the archive package, and the lazy import that matters for INV-06 is `boto3`'s, in the
provider adapter. Compression opens no socket.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, cast

import zstandard

from dnd_audio.archive import ArchiveError
from dnd_audio.determinism import BinaryReader

__all__ = [
    "ARCHIVE_CODEC_V1",
    "CHUNK_BYTES",
    "ArchiveCodec",
    "CompressedFact",
    "compress_bound",
    "compress_file",
    "decompress_and_measure",
]

#: Bytes moved per iteration everywhere in this module. Large enough that syscall overhead
#: is irrelevant on a four-hour recording, small enough that a dozen of them in flight is
#: nothing on a machine `systemd-oomd` is watching (INV-07).
CHUNK_BYTES: Final = 1 << 20

#: A ceiling on the decompressor's own window, independent of the ceiling on its output.
#: Frames this project writes at level 10 use a window far below it; a downloaded frame
#: claiming a gigabyte window is refused rather than honoured. Defense against a hostile
#: object, which is a thing a bucket can hold even when nobody meant it to.
_MAX_WINDOW_BYTES: Final = 1 << 27

#: Streaming adds a frame header and epilogue on top of `ZSTD_COMPRESSBOUND`. 64 bytes is
#: far more than either needs and costs nothing in a preflight.
_FRAME_MARGIN_BYTES: Final = 64


@dataclass(frozen=True, slots=True)
class ArchiveCodec:
    """Everything that decides the compressed bytes, stated rather than defaulted.

    Every field is recorded in the manifest, so a restore performed years later by
    something that is not this program knows exactly what it is reading — and so a
    dependency bump that changes output is visible as a changed recipe rather than as
    mysteriously different payloads.
    """

    level: int
    #: Zero means *no worker threads*, which is the whole point. Not a tuning knob: a
    #: nonzero value makes output depend on the machine, and archive v1 cannot have that.
    threads: int
    write_checksum: bool
    write_content_size: bool
    write_dict_id: bool

    def describe(self) -> dict[str, str | int | bool]:
        """The recipe as it appears in a manifest, including library versions.

        Versions are part of the description rather than of the identity: zstd frames are
        decodable by any later libzstd, so a restore does not need this exact build. It is
        here so that a *difference* in produced bytes has an explanation attached.
        """
        return {
            "format": "zstd",
            "level": self.level,
            "threads": self.threads,
            "write_checksum": self.write_checksum,
            "write_content_size": self.write_content_size,
            "write_dict_id": self.write_dict_id,
            "zstandard_version": zstandard.__version__,
            "libzstd_version": ".".join(str(part) for part in zstandard.ZSTD_VERSION),
        }

    def _compressor(self) -> zstandard.ZstdCompressor:
        return zstandard.ZstdCompressor(
            level=self.level,
            threads=self.threads,
            write_checksum=self.write_checksum,
            write_content_size=self.write_content_size,
            write_dict_id=self.write_dict_id,
        )


#: The recipe. Level 10 is what the four-file trial measured on real DJI recordings
#: (30.4%); the rest is stated explicitly so no libzstd default can move underneath it.
ARCHIVE_CODEC_V1: Final = ArchiveCodec(
    level=10,
    threads=0,
    write_checksum=True,
    write_content_size=True,
    write_dict_id=False,
)


@dataclass(frozen=True, slots=True)
class CompressedFact:
    """What a compression actually produced. Measured, never predicted."""

    size_bytes: int
    sha256: str


def compress_bound(size_bytes: int) -> int:
    """The largest a compressed frame of ``size_bytes`` can be.

    ``ZSTD_COMPRESSBOUND``'s documented arithmetic plus a streaming margin. Used to
    preflight disk, and it is why the charter forbids budgeting from the observed 30.4%
    saving: a session of already-compressed files compresses to slightly *more* than it
    started as, and a preflight that assumed otherwise would pass and then run out of disk
    halfway through.
    """
    if size_bytes < 0:
        message = f"cannot bound a negative size: {size_bytes}"
        raise ValueError(message)
    margin = ((128 << 10) - size_bytes) >> 11 if size_bytes < (128 << 10) else 0
    return size_bytes + (size_bytes >> 8) + margin + _FRAME_MARGIN_BYTES


def compress_file(
    source: Path,
    target: Path,
    *,
    codec: ArchiveCodec = ARCHIVE_CODEC_V1,
    chunk_bytes: int = CHUNK_BYTES,
) -> CompressedFact:
    """Compress ``source`` to ``target``, streaming, and measure what came out.

    The digest is computed on the way past rather than by re-reading the file, so a
    multi-gigabyte object is hashed once instead of twice.

    Returns:
        The compressed size and digest — the two values the manifest records and the
        remote readback later has to reproduce.
    """
    digest = hashlib.sha256()
    written = 0
    compressor = codec._compressor()

    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as raw, target.open("wb") as out:

        class _Counting:
            """Hashes and counts on the way to disk, so nothing is read back to measure."""

            @staticmethod
            def write(data: bytes) -> int:
                nonlocal written
                digest.update(data)
                written += len(data)
                out.write(data)
                return len(data)

        # `stream_writer` is annotated `IO[bytes]` but only ever calls `.write()`, which is
        # what makes the counting wrapper above possible at all. Narrowed here rather than
        # by giving `_Counting` a dozen unused `IO` methods that would be lies.
        with compressor.stream_writer(cast("IO[bytes]", _Counting()), closefd=False) as sink:
            while chunk := raw.read(chunk_bytes):
                sink.write(chunk)

    return CompressedFact(size_bytes=written, sha256=digest.hexdigest())


def decompress_and_measure(
    compressed: BinaryReader,
    *,
    max_output_bytes: int,
    sink: object = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> CompressedFact:
    """Decompress a stream, measuring the result and refusing to exceed its declared size.

    Args:
        compressed: Anything with a bounded ``read(size)`` — a local file, or a remote
            response body. The seam that lets verification and restore share this code.
        max_output_bytes: The original size the manifest declares. Decoding stops the
            moment output would pass it.
        sink: Optional object with ``write(bytes)``. ``None`` discards, which is what
            *verification* wants: it needs the digest, not the bytes.

    Returns:
        The restored size and digest, for comparison against the original.

    Raises:
        ArchiveError: if the frame is malformed, or if it would decode past
            ``max_output_bytes``. The ceiling is checked per chunk, before the chunk is
            handed on, so a decompression bomb is stopped at the limit rather than after
            the whole thing has been written somewhere.
    """
    digest = hashlib.sha256()
    produced = 0
    decompressor = zstandard.ZstdDecompressor(max_window_size=_MAX_WINDOW_BYTES)

    try:
        # Same narrowing as the writer above: `stream_reader` only calls `.read(size)`,
        # which is exactly what `BinaryReader` promises. Keeping the parameter as
        # `BinaryReader` is what lets a remote response body and a local file share this
        # path without either pretending to be a full `IO[bytes]`.
        source = cast("IO[bytes]", compressed)
        with decompressor.stream_reader(source, read_across_frames=False) as reader:
            while chunk := reader.read(chunk_bytes):
                produced += len(chunk)
                if produced > max_output_bytes:
                    message = (
                        f"the archived object decodes to more than the "
                        f"{max_output_bytes} bytes its manifest declares. Stopped at the "
                        f"ceiling rather than writing it: a frame that expands past its "
                        f"declared size is corrupt or hostile, and either way is not the "
                        f"file that was archived."
                    )
                    raise ArchiveError(message, code="archive_decoded_size_exceeded")
                digest.update(chunk)
                if sink is not None:
                    sink.write(chunk)  # type: ignore[attr-defined]
    except zstandard.ZstdError as exc:
        message = (
            f"the archived object is not a readable zstd frame: {exc}. The bytes that "
            f"arrived are not the bytes that were stored."
        )
        raise ArchiveError(message, code="archive_frame_unreadable") from exc

    return CompressedFact(size_bytes=produced, sha256=digest.hexdigest())
