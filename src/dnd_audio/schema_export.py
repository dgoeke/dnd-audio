"""Generate the checked-in JSON Schema artifacts from the authoritative models.

The spec requires schemas for `session.yaml`, `manifest.json`, `transcript.json`, and
`ingest-report.json`, generated from the Pydantic models and checked in, and requires
tests to validate real output against *those files* rather than round-tripping through
the model that produced them. `timeline.json` is M2's and follows the same rule: the
spec names the artifacts it knew about, and a new deterministic artifact that skipped
the schema would be the one consumers could not validate.

:func:`schema_documents` is the single source both the generator script and the drift
test use. If they each built the schema their own way, a drift test could pass while the
committed file was wrong — which is precisely the failure it exists to catch.

The input schema (`session.yaml`) is generated in validation mode and the output schemas
in serialization mode. The two differ wherever a field has a default or a computed
representation, and using one mode for both would describe a document nobody writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel

from dnd_audio.archive.manifest import ArchiveManifest
from dnd_audio.archive.report import ArchiveReport
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.manifest import Manifest
from dnd_audio.artifacts.records import TranscriptRecords
from dnd_audio.artifacts.report import IngestReport
from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.artifacts.transcript import Transcript
from dnd_audio.config import SessionConfig
from dnd_audio.determinism import canonical_json, write_atomic

__all__ = ["JSON_SCHEMA_DIALECT", "SCHEMA_DIRNAME", "schema_documents", "write_schemas"]

#: Directory, relative to the repository root, the artifacts are committed to.
SCHEMA_DIRNAME: Final = "schemas"

#: Pydantic v2 emits this dialect. Stating it makes the files usable by any validator
#: rather than only by one that guesses the same default.
JSON_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"


def schema_documents() -> dict[str, str]:
    """Every schema artifact, as ``{filename: canonical JSON text}``."""
    return {
        "session-config.schema.json": _document(SessionConfig, mode="validation"),
        "manifest.schema.json": _document(Manifest, mode="serialization"),
        "timeline.schema.json": _document(Timeline, mode="serialization"),
        "activity.schema.json": _document(ActivityGraph, mode="serialization"),
        "transcript.schema.json": _document(Transcript, mode="serialization"),
        "transcript-records.schema.json": _document(TranscriptRecords, mode="serialization"),
        "ingest-report.schema.json": _document(IngestReport, mode="serialization"),
        # M7a's two artifacts. The manifest is the remote commit record and has to be
        # readable by something that is not this program (ADR-0038); the report is local
        # and is the only place the manifest's own hash can live (ADR-0003, ADR-0039).
        "archive-manifest.schema.json": _document(ArchiveManifest, mode="serialization"),
        "archive-report.schema.json": _document(ArchiveReport, mode="serialization"),
    }


def write_schemas(directory: Path) -> list[Path]:
    """Write every schema artifact into ``directory``. Returns the paths written."""
    written: list[Path] = []
    for filename, text in sorted(schema_documents().items()):
        path = directory / filename
        write_atomic(path, text)
        written.append(path)
    return written


def _document(model: type[BaseModel], *, mode: Literal["validation", "serialization"]) -> str:
    schema = model.model_json_schema(mode=mode)
    schema["$schema"] = JSON_SCHEMA_DIALECT
    return canonical_json(schema)
