"""One source score from four pieces of evidence, and every term kept.

The spec is specific about this function and about the thing it must not be:

    Do not use a single global loudness comparison that always awards a time interval to
    the loudest person; that would erase a quieter speaker during real overlap. Source
    scores should combine track-relative speech level, VAD confidence, cross-track
    dominance, and correlation evidence. Keep the scoring function isolated and make its
    diagnostics visible in `ingest-report.json`.

So it is isolated here, it takes measurements rather than audio, and it returns its four
terms alongside the total — which is what makes a wrong attribution debuggable from the
artifact instead of by re-running with print statements.

Two of the terms are easy to state backwards, so they are stated here:

**Track-relative level** asks *is this as loud as this wearer usually is*, not *is this
loud*. A softly-spoken player at their own normal level must score like a loud one at
theirs; that is the entire difference between this and the global loudness rule the spec
forbids.

**Correlation contributes as independence, not as similarity.** A candidate strongly
correlated with another track is more likely to be a copy of it than a voice of its own, so
a high correlation *lowers* the score. Adding correlation with the other sign would rank
the best-recorded copy of someone else's voice above the original.

Every threshold and weight here is a guess about a real room (**OQ-017**). The terms are
clamped at both ends so one outlier measurement cannot carry the total.
"""

from __future__ import annotations

from dataclasses import dataclass

from dnd_audio.activity.detect import PERMILLE
from dnd_audio.config import ScoringConfig

__all__ = ["NEUTRAL_PERMILLE", "ScoreTerms", "score_candidate"]

#: What a term reads when the evidence for it does not exist — a track with no speech
#: reference, or a candidate nothing else overlaps. Deliberately the midpoint: an absent
#: measurement must neither reward nor punish, and defaulting it to zero would make every
#: solo utterance on a quiet track look like bleed.
NEUTRAL_PERMILLE = PERMILLE // 2

#: Millibels per decibel. Levels are integers in this project (INV-02); a hundredth of
#: a decibel is finer than any of these thresholds needs.
_MB_PER_DB = 100


@dataclass(frozen=True, slots=True)
class ScoreTerms:
    """The four terms and their weighted total, all per-mille."""

    level_permille: int
    confidence_permille: int
    dominance_permille: int
    correlation_permille: int
    total_permille: int


def score_candidate(
    *,
    relative_level_mb: int | None,
    probability_permille: int,
    level_delta_mb: int | None,
    peak_correlation_permille: int | None,
    settings: ScoringConfig,
) -> ScoreTerms:
    """Combine four measurements into one score in [0, 1000].

    Args:
        relative_level_mb: This candidate's band-limited level minus its own track's speech
            reference. ``None`` when the track has too little speech for a reference, which
            reads neutral rather than zero.
        probability_permille: The detector's own confidence over the candidate.
        level_delta_mb: The *best competitor's* level minus this candidate's, over the
            interval they share. ``None`` when nothing overlaps this candidate.
        peak_correlation_permille: The strongest correlation against any competitor.
            ``None`` when nothing overlaps.
    """
    level = (
        NEUTRAL_PERMILLE
        if relative_level_mb is None
        else _ramp(relative_level_mb, -settings.level_span_db * _MB_PER_DB, 0.0)
    )
    confidence = _clamp(probability_permille)
    dominance = (
        PERMILLE
        if level_delta_mb is None
        else _ramp(
            -level_delta_mb,
            -settings.dominance_span_db * _MB_PER_DB,
            settings.dominance_span_db * _MB_PER_DB,
        )
    )
    independence = (
        PERMILLE
        if peak_correlation_permille is None
        else PERMILLE - _clamp(peak_correlation_permille)
    )

    weights = (
        settings.level_weight,
        settings.confidence_weight,
        settings.dominance_weight,
        settings.correlation_weight,
    )
    terms = (level, confidence, dominance, independence)
    # Normalized by the sum rather than required to sum to one: weights are relative, so
    # doubling all four must change nothing. The configuration rejects a zero sum.
    total = sum(weight * term for weight, term in zip(weights, terms, strict=True)) / sum(weights)

    return ScoreTerms(
        level_permille=level,
        confidence_permille=confidence,
        dominance_permille=dominance,
        correlation_permille=independence,
        total_permille=_clamp(_round_half_up(total)),
    )


def _ramp(value: float, low: float, high: float) -> int:
    """Map ``value`` linearly onto [0, 1000] across ``[low, high]``, clamped at both ends."""
    if high <= low:  # pragma: no cover - the configuration's bounds forbid it
        message = f"a ramp needs low < high, got [{low}, {high}]"
        raise ValueError(message)
    return _clamp(_round_half_up((value - low) / (high - low) * PERMILLE))


def _clamp(value: int) -> int:
    return max(0, min(PERMILLE, value))


def _round_half_up(value: float) -> int:
    """Halves away from zero, matching the project's one rounding rule.

    Python's :func:`round` is banker's rounding, so 0.5 and 1.5 would disagree about which
    way a half goes and two scores a thousandth apart could order differently on different
    inputs. :mod:`dnd_audio.determinism` owns this rule for *time*; this is the same rule
    applied to a dimensionless ratio, which is deliberately not a second time quantizer.
    """
    magnitude = int(abs(value) + 0.5)
    return -magnitude if value < 0 else magnitude
