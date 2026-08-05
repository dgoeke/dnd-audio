"""The acoustic synchronization marker: one canonical waveform, generated and detected.

M10 builds a distinctive sound that can be produced through the CLI, played from a single
offline HTML file on a phone, and found automatically at integer-sample positions on every
track. It **verifies** the LTC jam and measures differential acoustic arrival. It never
places a file, never overrides valid timecode, and never becomes a hidden timeline
correction — see [ADR-0040](../../../docs/plan/decisions/0040-what-an-acoustic-marker-measures.md)
for the four quantities this milestone is careful never to conflate, only one of which is
recorder drift.

**There is one synthesizer.** The WAV the CLI writes, the bytes embedded in the phone page,
and the templates the detector correlates against all come from
:func:`dnd_audio.marker.synth.marker_samples`. A second implementation — a JavaScript
oscillator, a detector-side formula — is the failure this design exists to prevent, because
two approximately equivalent generators disagree in exactly the conditions nobody tests.

**There is one frozen ``v1``.** The physical phone/DJI bench selected the longer,
narrower-band ``cand-b`` recipe; :data:`~dnd_audio.marker.spec.MARKER_SPECS` retains all three
candidate names as evidence and adds a separate ``v1`` entry with those exact waveform
parameters. ADR-0042 records the measurements, margins, and frozen WAV SHA-256.

Three semantic versions, because three different things can move the result and merging them
would let one change hide behind another's number:

* :data:`MARKER_SEMANTICS_VERSION` — the waveform. Moving it changes the bytes.
* :data:`DETECTOR_SEMANTICS_VERSION` — matched filtering and sequence acceptance.
* :data:`MARKER_ANALYSIS_SEMANTICS_VERSION` — everything above the detector: occurrence
  grouping, cross-track association, role assignment, geometry classification, and the
  source-coordinate mapping. The second plan review found this one missing, which would have
  let a change to grouping move the analysis without moving its claimed identity.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ANALYSIS_RELATIVE_PATH",
    "DEFAULT_WINDOW_SECONDS",
    "DETECTOR_SEMANTICS_VERSION",
    "MARKER_ANALYSIS_SEMANTICS_VERSION",
    "MARKER_MANIFEST_FILENAME",
    "MARKER_REPORT_RELATIVE_PATH",
    "MARKER_SAMPLE_RATE",
    "MARKER_SEMANTICS_VERSION",
    "artifact_stem",
]

#: The waveform's semantics. Bump for any change to synthesis, the sine table, the envelope,
#: the container layout, or a spec's numbers — every one of them moves the frozen SHA-256.
MARKER_SEMANTICS_VERSION: Final = 1

#: Matched filtering, per-chirp templates, sequence acceptance, and the thresholds that
#: decide it. Bump when a detection that used to be accepted stops being one, or vice versa.
DETECTOR_SEMANTICS_VERSION: Final = 3

#: Occurrence grouping, one-to-one cross-track association, role assignment against the
#: event log, geometry classification, and anchor → `(source, sample)` mapping.
MARKER_ANALYSIS_SEMANTICS_VERSION: Final = 2

#: The grid the marker is built on and searched at. The session's own working rate: a
#: marker built at any other rate would need resampling to correlate, and resampling is the
#: one operation that would put a second, inexact implementation on the path.
MARKER_SAMPLE_RATE: Final = 48000

#: How much of each end of the session `marker analyze` searches when no event log says
#: otherwise. It lives here rather than in `runner` so the CLI can name it as an option
#: default without importing NumPy to draw its `--help`.
DEFAULT_WINDOW_SECONDS: Final = 120

#: Published beside the WAV and the page, last, as the completeness marker (ADR-0041).
MARKER_MANIFEST_FILENAME: Final = "marker-manifest.json"

#: Deterministic, byte-stable, INV-02. Under `work/` because it is derived from the session.
ANALYSIS_RELATIVE_PATH: Final = "work/sync-marker-analysis.json"

#: Per-run, INV-13, at its own command boundary rather than as a seventh stage in
#: `ingest-report.json` — the precedent ADR-0039 set for the archive's report.
MARKER_REPORT_RELATIVE_PATH: Final = "output/marker-report.json"


def artifact_stem(marker_name: str) -> str:
    """The filename stem both published artifacts share, for ``marker_name``.

    A candidate is named for itself and `v1` is named `v1`, so a bench take and a production
    marker can never be confused by filename alone — which matters because the operator is
    carrying these onto a phone by hand.
    """
    return f"dnd-audio-sync-marker-{marker_name}"
