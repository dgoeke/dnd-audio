"""Reading the session's existing artifacts, and refusing stale ones — without writing.

`marker analyze` is the one command in this project that **validates** rather than rebuilds.
Every other composed runner re-runs inspection and reconstruction on the reasoning ADR-0015
gives: a configuration-hash match is not evidence that an artifact still describes what is on
disk, and rebuilding deletes a whole class of staleness bug. That reasoning is still right; it
just cannot be applied here, because rebuilding would rewrite `timeline.json` and
`ingest-report.json`, and M10's charter forbids touching either.

**So the departure has to be paid for by checking more, not less.** Comparing the manifest
digest alone would accept a timeline built by obsolete placement logic that is still perfectly
consistent with the same manifest. Every identity component the timeline records is compared
against the current one, each with its own diagnostic code and its own test.

**And it genuinely does not write.** The first draft of M10's working plan proposed re-running
inspection "in memory, publishing nothing", which is false: on a cold or missing sidecar,
`inspection/runner.py::_inspect_one` writes `work/ffprobe/…`. "Warm from the content cache" is
an assumption about the machine rather than a contract — the same shape as the six tests M6b
found asserting a property of the machine instead of the code. Nothing here runs inspection at
all; it reads two files and compares them to constants and to the resolved configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy
import scipy
from pydantic import ValidationError

from dnd_audio.artifacts.manifest import Manifest, StartEvidenceRecord
from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.config import SessionConfig, config_hash
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import DndAudioError
from dnd_audio.inspection import INSPECTION_SEMANTICS_VERSION
from dnd_audio.inspection.runner import MANIFEST_RELATIVE_PATH
from dnd_audio.timeline import (
    CANONICAL_SAMPLE_RATE,
    TIMELINE_RELATIVE_PATH,
    TIMELINE_SEMANTICS_VERSION,
)

__all__ = ["SessionArtifacts", "StaleArtifactError", "read_session_artifacts"]


class StaleArtifactError(DndAudioError):
    """A required artifact is missing, unreadable, or no longer describes this session."""

    default_code = "timeline_stale"


@dataclass(frozen=True, slots=True)
class SessionArtifacts:
    """The validated inputs `marker analyze` reads, and the identities it records."""

    timeline: Timeline
    manifest: Manifest
    manifest_sha256: str
    config_hash: str

    def start_evidence(self) -> tuple[StartEvidenceRecord, ...]:
        """Every selected source's timing evidence, for the quantization floor.

        `syncqa.offset_floor_samples` needs this to derive the finest offset the session's
        own evidence could express, and M8's reason for reading it from the *evidence*
        rather than from `timecode.frame_rate` holds here: a receiver set to 60 fps still
        wrote 1600-sample boundaries (OQ-024), so the configured rate would give a 16.7 ms
        threshold against timing that still moves in 33.3 ms steps.
        """
        return tuple(
            source.start_time.evidence
            for track in self.manifest.tracks
            for source in track.sources
            if source.start_time is not None
        )


def read_session_artifacts(session_dir: Path, config: SessionConfig) -> SessionArtifacts:
    """Load `manifest.json` and `timeline.json`, refusing anything stale.

    Raises:
        StaleArtifactError: with a code naming *which* component disagrees —
            ``manifest_missing``, ``manifest_unreadable``, ``timeline_missing``,
            ``timeline_unreadable``,
            ``timeline_stale_config``, ``timeline_stale_manifest``,
            ``timeline_stale_semantics``, ``timeline_stale_numerics``, or
            ``timeline_unsupported_rate``. Distinct codes rather than one, because
            "your timeline is stale" sends an operator to re-run `ingest` without telling
            them why it went stale — and a NumPy upgrade and an edited `session.yaml` want
            different reactions.
    """
    manifest_path = session_dir / MANIFEST_RELATIVE_PATH
    timeline_path = session_dir / TIMELINE_RELATIVE_PATH

    if not manifest_path.is_file():
        message = (
            f"there is no manifest at {MANIFEST_RELATIVE_PATH}. Marker analysis reads the "
            f"session's existing artifacts and never rebuilds them — run `dnd-audio ingest` "
            f"first."
        )
        raise StaleArtifactError(message, code="manifest_missing")

    if not timeline_path.is_file():
        message = (
            f"there is no timeline at {TIMELINE_RELATIVE_PATH}. Marker analysis needs the "
            f"48 kHz segment map and never builds one — run `dnd-audio ingest` first."
        )
        raise StaleArtifactError(message, code="timeline_missing")

    try:
        timeline = Timeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        message = (
            f"{TIMELINE_RELATIVE_PATH} cannot be read as a timeline this build understands: "
            f"{exc}. Re-run `dnd-audio ingest`."
        )
        raise StaleArtifactError(message, code="timeline_unreadable") from exc

    try:
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        message = (
            f"{MANIFEST_RELATIVE_PATH} cannot be read as a manifest this build understands: "
            f"{exc}. Re-run `dnd-audio inspect`."
        )
        raise StaleArtifactError(message, code="manifest_unreadable") from exc

    manifest_digest = sha256_file(manifest_path)
    current_config = config_hash(config)

    # Every component, not just the manifest digest. A timeline can agree with its manifest
    # and still have been produced by placement logic two milestones old.
    _require(
        timeline.config_hash == current_config,
        code="timeline_stale_config",
        detail=(
            f"the timeline was built from configuration {timeline.config_hash[:12]} and "
            f"session.yaml now resolves to {current_config[:12]}"
        ),
    )
    _require(
        timeline.manifest_sha256 == manifest_digest,
        code="timeline_stale_manifest",
        detail=(
            f"the timeline was built from manifest {timeline.manifest_sha256[:12]} and the "
            f"manifest on disk is {manifest_digest[:12]}"
        ),
    )
    _require(
        timeline.provenance.timeline_semantics_version == TIMELINE_SEMANTICS_VERSION,
        code="timeline_stale_semantics",
        detail=(
            f"the timeline was built by timeline semantics version "
            f"{timeline.provenance.timeline_semantics_version} and this build is "
            f"{TIMELINE_SEMANTICS_VERSION}"
        ),
    )
    _require(
        timeline.provenance.inspection_semantics_version == INSPECTION_SEMANTICS_VERSION,
        code="timeline_stale_semantics",
        detail=(
            f"the timeline rests on inspection semantics version "
            f"{timeline.provenance.inspection_semantics_version} and this build is "
            f"{INSPECTION_SEMANTICS_VERSION}"
        ),
    )
    _require(
        timeline.provenance.numpy_version == numpy.__version__
        and timeline.provenance.scipy_version == scipy.__version__,
        code="timeline_stale_numerics",
        detail=(
            f"the timeline was built with numpy {timeline.provenance.numpy_version} / scipy "
            f"{timeline.provenance.scipy_version} and this environment has "
            f"numpy {numpy.__version__} / scipy {scipy.__version__}. Those decide the "
            f"derivatives and the placement arithmetic, so M2 records them (INV-08)"
        ),
    )
    _require(
        timeline.sample_rate == CANONICAL_SAMPLE_RATE,
        code="timeline_unsupported_rate",
        detail=(
            f"the timeline is on a {timeline.sample_rate} Hz grid and the marker is built "
            f"and searched at {CANONICAL_SAMPLE_RATE} Hz; resampling one to meet the other "
            f"would put an inexact step on the detection path"
        ),
    )

    return SessionArtifacts(
        timeline=timeline,
        manifest=manifest,
        manifest_sha256=manifest_digest,
        config_hash=current_config,
    )


def _require(condition: bool, *, code: str, detail: str) -> None:
    """Refuse with a code naming the component, or return."""
    if condition:
        return
    message = (
        f"{detail}. Marker analysis reads the session's existing artifacts rather than "
        f"rebuilding them, so a stale one is refused rather than silently trusted — re-run "
        f"`dnd-audio ingest` and analyze again."
    )
    raise StaleArtifactError(message, code=code)
