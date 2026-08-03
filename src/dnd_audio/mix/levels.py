"""Per-track voice-level correction: making six wearers equally loud, conservatively.

The spec's first automixer requirement:

> Estimate a conservative per-track voice-level correction from high-confidence speech
> attributed to that track; clamp correction to a safe range.

M3 already did the estimating half and said so: `ActivityTrack.speech_reference_mbfs` is "what
this wearer sounds like when this wearer is talking" — the 75th percentile of that track's own
band-limited candidate levels (ADR-0014). M5's job is to turn six of those into six gains, and
to clamp them.

Three things this deliberately does not do.

**A missing reference is not a reference of zero.** `speech_reference_mbfs` is `None` where the
track had too little speech to establish one, and M3 chose `None` over a default precisely so
that a consumer could not read silence as full scale. Such a track is corrected by **zero** and
warned about, never lifted to a target it was never measured against.

**The target is the tracks' own median, not a fixed dBFS.** The point is that six wearers end up
level with *each other*; where that lands absolutely is the two-pass loudness normalization's
question (ADR-0023), and choosing a fixed target here would fight it. The median is taken with
integer millibels and `nearest` interpolation, the same way `speech_references` takes its
percentile, so the result is an integer and does not move with a NumPy upgrade (INV-02).

**The correction does not enter the gain share.** It multiplies the audio; the share decides who
is heard. Folding it into the weights would count track-relative level twice, since the graph's
`score_permille` already carries a track-relative term — and it would quietly undo the
correction, because normalizing by the corrected sum removes exactly the level equalization the
correction was applied for. ADR-0022 records that, and records the price: the correction erodes
the dominance margin by up to twice its clamp, which is why the clamp is part of the
achievability validator rather than a free parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.config import EnvelopeConfig
from dnd_audio.mix import MixNote

__all__ = ["MILLIBELS_PER_DB", "LevelCorrections", "TrackCorrection", "level_corrections"]

#: The activity graph quotes levels in millibels — decibels scaled by a hundred — so that a
#: byte-stable artifact carries no floats (ADR-0012). This package keeps the same unit for
#: every level it records, and converts to a linear factor at exactly one place.
MILLIBELS_PER_DB: int = 100


@dataclass(frozen=True, slots=True)
class TrackCorrection:
    """One track's correction, and enough of the reasoning to audit it."""

    track_id: str
    #: What M3 measured this wearer's speech at, or ``None`` if it could not.
    reference_mbfs: int | None
    #: Signed millibels applied to this track's audio. Zero when there is no reference.
    correction_mb: int
    #: Whether the clamp bound it. Recorded because a clamped correction means the target was
    #: further away than the configuration is willing to move, which is worth a human's
    #: attention on a session where one lav was mounted wrong.
    clamped: bool

    @property
    def gain(self) -> float:
        """The linear factor this correction is, evaluated once."""
        return float(10.0 ** (self.correction_mb / (20.0 * MILLIBELS_PER_DB)))


@dataclass(frozen=True, slots=True)
class LevelCorrections:
    """Every track's correction, the target they were levelled to, and what to warn about."""

    #: The median of the references that existed, or ``None`` when none did.
    target_mbfs: int | None
    corrections: tuple[TrackCorrection, ...]
    warnings: tuple[MixNote, ...]

    def gains(self, track_ids: tuple[str, ...]) -> np.ndarray:
        """The linear factors for ``track_ids``, in that order.

        Raises:
            KeyError: for a track this does not cover. Every track in the mix comes from the
                same graph these corrections were built from, so a miss means two different
                track sets reached the renderer — which would silently apply one track's
                correction to another's audio.
        """
        by_id = {item.track_id: item for item in self.corrections}
        return np.array([by_id[track_id].gain for track_id in track_ids], dtype=np.float64)


def level_corrections(graph: ActivityGraph, *, settings: EnvelopeConfig) -> LevelCorrections:
    """Level the graph's tracks against each other, within the configured clamp.

    Args:
        graph: The frozen activity graph. Only `ActivityTrack.speech_reference_mbfs` is read.
        settings: Supplies `max_level_correction_db`, the spec's "safe range".
    """
    clamp_mb = round(settings.max_level_correction_db * MILLIBELS_PER_DB)
    measured = [
        track.speech_reference_mbfs
        for track in graph.tracks
        if track.speech_reference_mbfs is not None
    ]
    target = _median_mbfs(measured)

    corrections: list[TrackCorrection] = []
    warnings: list[MixNote] = []
    for track in graph.tracks:
        reference = track.speech_reference_mbfs
        if reference is None or target is None:
            corrections.append(
                TrackCorrection(
                    track_id=track.track_id,
                    reference_mbfs=reference,
                    correction_mb=0,
                    clamped=False,
                )
            )
            warnings.append(
                MixNote(
                    code="mix_level_uncorrected",
                    message=(
                        f"{track.track_id} has no speech reference, so its level is left "
                        f"exactly as recorded. A missing reference is not a reference of zero "
                        f"— lifting an unmeasured track to the session's target would amplify "
                        f"whatever it did record (OQ-019)."
                    ),
                    path=track.track_id,
                )
            )
            continue

        wanted = target - reference
        correction = max(-clamp_mb, min(clamp_mb, wanted))
        clamped = correction != wanted
        corrections.append(
            TrackCorrection(
                track_id=track.track_id,
                reference_mbfs=reference,
                correction_mb=correction,
                clamped=clamped,
            )
        )
        if clamped:
            warnings.append(
                MixNote(
                    code="mix_level_correction_clamped",
                    message=(
                        f"{track.track_id} speaks at {reference / MILLIBELS_PER_DB:.2f} dBFS "
                        f"against a session target of {target / MILLIBELS_PER_DB:.2f}, which "
                        f"wants {wanted / MILLIBELS_PER_DB:+.2f} dB of correction. Clamped to "
                        f"{correction / MILLIBELS_PER_DB:+.2f} dB. A lav this far out is "
                        f"usually a mounting or gain problem the mix should not hide."
                    ),
                    path=track.track_id,
                )
            )

    return LevelCorrections(
        target_mbfs=target,
        corrections=tuple(corrections),
        warnings=tuple(warnings),
    )


def _median_mbfs(values: list[int]) -> int | None:
    """The median reference, as an integer, or ``None`` when nothing was measured.

    `nearest` rather than the default linear interpolation, for the reason
    `bleed.speech_references` gives about its own percentile: the result reaches a decision and
    must be an integer that does not move with a library upgrade. With an even count that means
    the upper of the two middle values, deterministically.
    """
    if not values:
        return None
    return int(np.percentile(np.asarray(values, dtype=np.int64), 50, method="nearest"))
