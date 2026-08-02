"""Run `ffprobe`, keep what it said verbatim, and read the container properties.

Two decisions here are load-bearing and neither is obvious.

**FFprobe runs with the session directory as its working directory and is given a
session-relative path.** Its output contains `format.filename` echoed back, so probing
by absolute path would make the sidecar's bytes — and therefore its content hash, and
therefore the manifest — depend on where the session happens to live on disk. Copying a
session would change its manifest, which is not what INV-02 means by byte-stable.

**Running and parsing are separate functions.** The caller runs, persists the bytes, and
only then parses, so a document this code cannot understand still leaves the evidence on
disk. A capture that is discarded when parsing fails is worst-case useless exactly when
it would have been most useful (OQ-001).

Nothing here decodes audio. The exact sample count comes from the container: the RIFF
`data` size divided by the block alignment is exact by construction, and `duration_ts`
is a cross-check rather than the source. Which of the two to trust on real hardware is
OQ-011, and :func:`exact_sample_count` records their agreement so the fixture answers
the synthetic half of it now.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from dnd_audio.determinism import sha256_bytes
from dnd_audio.errors import DndAudioError

__all__ = [
    "FFPROBE_ARGS",
    "AudioProperties",
    "ProbeError",
    "ProbeResult",
    "SampleCount",
    "ToolVersions",
    "exact_sample_count",
    "format_tags",
    "parse_probe",
    "read_audio_properties",
    "run_ffprobe",
    "tool_versions",
]

#: The exact invocation, and part of every inspection cache identity (INV-08). Changing
#: it changes what was captured, so a cached capture taken under different options must
#: not be reused.
FFPROBE_ARGS: Final = (
    "-v",
    "error",
    "-print_format",
    "json",
    "-show_format",
    "-show_streams",
)

_TIMEOUT_S: Final = 60.0
_BITS_PER_BYTE: Final = 8


class ProbeError(DndAudioError):
    """FFprobe could not be run, failed, or produced something unreadable."""


@dataclass(frozen=True, slots=True)
class ToolVersions:
    """Exact external tool versions. A change to either re-runs inspection (INV-08)."""

    ffmpeg: str
    ffprobe: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """FFprobe's output, unmodified."""

    #: Exactly the bytes FFprobe wrote. Not reserialized — that is what "verbatim" means.
    raw: bytes
    sha256: str

    @property
    def sidecar_name(self) -> str:
        """Content-addressed filename. Identical output is stored once."""
        return f"{self.sha256}.json"


@dataclass(frozen=True, slots=True)
class AudioProperties:
    """The container facts the spec requires the manifest to carry."""

    codec_name: str
    sample_format: str
    bits_per_sample: int
    sample_rate: int
    channels: int
    duration_ts: int | None
    time_base: str | None
    #: FFprobe's own displayed duration, kept as the string it reported. Parsing it into
    #: a float here would introduce the only float in the timing path for no gain
    #: (INV-04); it is a human-facing cross-check, not arithmetic.
    duration_text: str | None

    @property
    def block_align(self) -> int:
        """Bytes per frame across all channels."""
        return self.channels * (self.bits_per_sample // _BITS_PER_BYTE)


@dataclass(frozen=True, slots=True)
class SampleCount:
    """How many PCM frames the file holds, and how we know.

    ``agrees`` is the evidence OQ-011 asks for: whether the container's `duration_ts`
    matches the count derived from the `data` chunk size. ``None`` when only one of the
    two was available.
    """

    samples: int | None
    source: Literal["data_chunk", "duration_ts", "none"]
    agrees: bool | None = None


def tool_versions() -> ToolVersions:
    """Read the exact FFmpeg and FFprobe versions.

    Raises:
        ProbeError: if either tool is missing or will not report a version. Inspecting
            without knowing the tool version would produce a cache entry that cannot be
            invalidated by an upgrade, which is a quiet INV-08 violation.
    """
    return ToolVersions(ffmpeg=_version_of("ffmpeg"), ffprobe=_version_of("ffprobe"))


def run_ffprobe(session_dir: Path, relative_path: str) -> ProbeResult:
    """Probe one source, returning FFprobe's bytes without interpreting them.

    Args:
        session_dir: Used as the working directory, so the recorded filename is
            relative and the output does not depend on where the session lives.
        relative_path: Session-relative POSIX path of the file to probe.

    Raises:
        ProbeError: if FFprobe is missing, times out, or exits nonzero.
    """
    executable = shutil.which("ffprobe")
    if executable is None:
        message = "ffprobe is not on PATH — run `direnv allow` to enter the project shell"
        raise ProbeError(message)

    try:
        completed = subprocess.run(
            [executable, *FFPROBE_ARGS, "-i", relative_path],
            cwd=session_dir,
            capture_output=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        message = f"ffprobe timed out after {_TIMEOUT_S:.0f}s on {relative_path}"
        raise ProbeError(message) from exc
    except OSError as exc:
        message = f"could not run ffprobe on {relative_path}: {exc}"
        raise ProbeError(message) from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        message = f"ffprobe failed on {relative_path} (exit {completed.returncode}): {detail}"
        raise ProbeError(message)

    return ProbeResult(raw=completed.stdout, sha256=sha256_bytes(completed.stdout))


def parse_probe(raw: bytes) -> dict[str, Any]:
    """Parse FFprobe's JSON.

    Separate from :func:`run_ffprobe` so the caller can persist the bytes first.

    Raises:
        ProbeError: if the output is not a JSON object.
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = f"ffprobe output is not valid JSON: {exc}"
        raise ProbeError(message) from exc
    if not isinstance(document, dict):
        message = f"ffprobe output is a {type(document).__name__}, expected an object"
        raise ProbeError(message)
    return document


def read_audio_properties(document: dict[str, Any]) -> AudioProperties:
    """Pull the container facts out of a parsed probe.

    Raises:
        ProbeError: if there is no audio stream, or a required numeric field is missing
            or unreadable. A source whose sample rate cannot be established is not a
            source this pipeline can use, and defaulting one would be inventing it.
    """
    streams = document.get("streams")
    if not isinstance(streams, list):
        message = "ffprobe output has no streams array"
        raise ProbeError(message)

    audio = [s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"]
    if not audio:
        message = "the file has no audio stream"
        raise ProbeError(message)
    if len(audio) > 1:
        message = f"the file has {len(audio)} audio streams; a transmitter recording has one"
        raise ProbeError(message)

    stream = audio[0]
    return AudioProperties(
        codec_name=str(stream.get("codec_name", "")),
        sample_format=str(stream.get("sample_fmt", "")),
        bits_per_sample=_required_int(stream, "bits_per_sample"),
        sample_rate=_required_int(stream, "sample_rate"),
        channels=_required_int(stream, "channels"),
        duration_ts=_optional_int(stream, "duration_ts"),
        time_base=_optional_str(stream, "time_base"),
        duration_text=_optional_str(stream, "duration"),
    )


def format_tags(document: dict[str, Any]) -> dict[str, str]:
    """Format-level metadata tags, keyed in lower case.

    Where a BWF `time_reference` and an `INFO`/`ISMP` timecode both surface, and the
    only place the strategy chain looks for them.
    """
    container = document.get("format")
    if not isinstance(container, dict):
        return {}
    tags = container.get("tags")
    if not isinstance(tags, dict):
        return {}
    return {str(key).lower(): str(value) for key, value in tags.items()}


def exact_sample_count(
    *, data_size: int | None, block_align: int, duration_ts: int | None
) -> SampleCount:
    """The PCM frame count, preferring the container's own arithmetic.

    The `data` chunk size divided by the block alignment is exact by construction for
    PCM, needs no decode, and is available from the RIFF walk we already did. FFprobe's
    `duration_ts` should agree; when both exist, whether they do is recorded rather than
    assumed (OQ-011).
    """
    from_data: int | None = None
    if data_size is not None and block_align > 0 and data_size % block_align == 0:
        from_data = data_size // block_align

    if from_data is not None:
        agrees = None if duration_ts is None else duration_ts == from_data
        return SampleCount(samples=from_data, source="data_chunk", agrees=agrees)
    if duration_ts is not None:
        return SampleCount(samples=duration_ts, source="duration_ts")
    return SampleCount(samples=None, source="none")


def _version_of(tool: str) -> str:
    executable = shutil.which(tool)
    if executable is None:
        message = f"{tool} is not on PATH — run `direnv allow` to enter the project shell"
        raise ProbeError(message)
    try:
        completed = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        message = f"could not read the {tool} version: {exc}"
        raise ProbeError(message) from exc

    first = completed.stdout.strip().splitlines()
    if not first:
        message = f"{tool} reported no version"
        raise ProbeError(message)
    return first[0].strip()


def _required_int(stream: dict[str, Any], key: str) -> int:
    value = _optional_int(stream, key)
    if value is None:
        message = f"the audio stream has no usable {key!r}"
        raise ProbeError(message)
    return value


def _optional_int(stream: dict[str, Any], key: str) -> int | None:
    raw = stream.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _optional_str(stream: dict[str, Any], key: str) -> str | None:
    raw = stream.get(key)
    return None if raw is None else str(raw)
