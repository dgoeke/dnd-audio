"""`dnd-audio marker build`: the WAV, the page, and the manifest that says they are complete.

**Publication order is the interesting part, and it is not what the charter first said.**
"Manifest last" is a completeness marker for a *first* build. It is not one for a **rebuild**:
a crash between replacing the WAV and replacing the page would leave the previous manifest
sitting beside a mixed set, describing bytes that are no longer there — and it would look
complete, because the marker of completeness is present. So the manifest is **removed first**,
then the two artifacts are written and validated, then it is published. Every interrupted
state is therefore manifest-less, which is detectable (ADR-0041, second plan review P2-10).

**INV-01 is checked before anything is created.** This command takes an arbitrary destination
and has no session argument, so ``dnd-audio marker build SESSION/raw/tx-a`` would write three
files under a source root. That is verbatim the P0 M7a's second code review found, where a
guard conditioned on *having a session directory* was defeated by the one command that never
has one. The guard here is driven by the resolved destination instead, and it runs before the
directory is created, before a candidate file is written, and before the old manifest is
unlinked — because on that path the unlink is itself the violation (ADR-0021's ordering rule,
one command further out).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnd_audio.determinism import sha256_bytes, write_atomic, write_json_atomic
from dnd_audio.marker import MARKER_MANIFEST_FILENAME, artifact_stem
from dnd_audio.marker.manifest import MarkerArtifact, MarkerManifest, describe
from dnd_audio.marker.page import marker_page_html, payload_from_html
from dnd_audio.marker.spec import MarkerSpec
from dnd_audio.marker.wav import marker_wav_bytes

__all__ = ["MarkerBuild", "build_marker"]


@dataclass(frozen=True, slots=True)
class MarkerBuild:
    """What a build produced, for the CLI to report and a test to inspect."""

    manifest: MarkerManifest
    wav_path: Path
    page_path: Path
    manifest_path: Path


def build_marker(spec: MarkerSpec, destination: Path) -> MarkerBuild:
    """Write ``spec``'s WAV, standalone page and manifest into ``destination``.

    The caller is responsible for having refused a destination inside a session's sources —
    :func:`dnd_audio.cli._reject_path_inside_any_session` does that before this is reached, so
    that the check happens before ``mkdir`` rather than after it.

    Raises:
        ValueError: if the page's embedded payload does not decode back to the WAV's exact
            bytes. That cannot happen while both come from one call to
            :func:`~dnd_audio.marker.wav.marker_wav_bytes`, and it is checked anyway: this is
            the milestone's central equivalence claim, and a claim nothing verifies is a claim
            that quietly stops being true.
    """
    stem = artifact_stem(spec.name)
    wav_path = destination / f"{stem}.wav"
    page_path = destination / f"{stem}.html"
    manifest_path = destination / MARKER_MANIFEST_FILENAME

    wav_bytes = marker_wav_bytes(spec)
    page_text = marker_page_html(spec, wav_bytes)

    extracted = payload_from_html(page_text)
    if extracted != wav_bytes:  # pragma: no cover - one source, so unreachable by construction
        message = (
            f"the page's embedded payload is {len(extracted)} bytes and the WAV is "
            f"{len(wav_bytes)}; they must be the same bytes, not two encodings of the same "
            f"samples"
        )
        raise ValueError(message)

    destination.mkdir(parents=True, exist_ok=True)

    # Removed first. A rebuild interrupted between the two artifacts must not leave a
    # manifest describing the previous pair — see the module docstring.
    manifest_path.unlink(missing_ok=True)

    write_atomic(wav_path, wav_bytes)
    write_atomic(page_path, page_text.encode("utf-8"))

    page_bytes = page_path.read_bytes()
    manifest = describe(
        spec,
        wav=MarkerArtifact(
            filename=wav_path.name, size_bytes=len(wav_bytes), sha256=sha256_bytes(wav_bytes)
        ),
        page=MarkerArtifact(
            filename=page_path.name, size_bytes=len(page_bytes), sha256=sha256_bytes(page_bytes)
        ),
    )

    # Read back from disk rather than trusting what was just handed to `write_atomic`: the
    # digests in the manifest are a claim about the files an operator will copy to a phone,
    # and hashing the in-memory value would make that claim about something else.
    if sha256_bytes(wav_path.read_bytes()) != manifest.wav.sha256:  # pragma: no cover
        message = f"{wav_path} does not read back as the bytes just written to it"
        raise ValueError(message)

    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    return MarkerBuild(
        manifest=manifest,
        wav_path=wav_path,
        page_path=page_path,
        manifest_path=manifest_path,
    )
