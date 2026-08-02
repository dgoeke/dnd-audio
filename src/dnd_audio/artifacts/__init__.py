"""Output artifact schemas.

Every model here is serialized through :func:`dnd_audio.determinism.canonical_json` and
has a checked-in JSON Schema under ``schemas/``. Tests validate real output against
those files rather than round-tripping through the model that produced it — a
round-trip proves only that pydantic agrees with itself.

**Schema versions are provisional until the milestone that owns the artifact closes.**
``manifest.json`` is M1's, ``transcript.json`` is M4's, ``ingest-report.json`` accretes
across every milestone. Before its owner closes, a milestone may change version 1
freely; after, only optional additive fields, and anything else bumps the version.
M0 checks in skeletons so the drift rail exists from the start, not because their
shapes are settled.
"""

from __future__ import annotations

__all__: list[str] = []
