"""Output artifact schemas.

Every model here is serialized through :func:`dnd_audio.determinism.canonical_json` and
has a checked-in JSON Schema under ``schemas/``. Tests validate real output against
those files rather than round-tripping through the model that produced it — a
round-trip proves only that pydantic agrees with itself.

**Schema versions are provisional until the milestone that owns the artifact closes.**
``manifest.json`` is M1's, ``timeline.json`` is M2's, ``activity.json`` is M3's,
``transcript.json`` is M4's, and ``ingest-report.json`` accretes across every milestone.
Before its owner closes, a milestone may change version 1 freely; after, only optional
additive fields, and anything else bumps the version. M0 checks in skeletons so the drift
rail exists from the start, not because their shapes are settled.

``timeline.json`` and ``activity.json`` are both **frozen** — the first at M2's close, the
second at M3's — and both are read by two later milestones. `activity.json` additionally
carries a field allowlist test, because INV-09 is about which fields may exist rather than
about which modules import which.
"""

from __future__ import annotations

__all__: list[str] = []
