"""The conservative bleed gate: a score margin, a correlation, and a veto (ADR-0014).

Every lav hears the room, so VAD alone produces the same utterance as a candidate on four
or five tracks. Deciding which one is the speaker is the most dangerous thing this project
does, because the failure is silent: a deleted candidate leaves no trace in the audio, and a
transcript missing one side of an argument reads exactly like an argument that only had one
side.

So suppression needs **all three** of:

1. another track's candidate scoring at least `min_score_margin` better;
2. a normalized speech-band correlation of at least `min_correlation` within the lag window;
3. this candidate's own level sitting more than `veto_db` **below its own track's speech
   reference**.

Any one of them failing keeps the candidate. The third is the one that took a review to
find: without it, two people genuinely talking at once at unequal levels — each lav also
carrying the other's voice — satisfy the first two, and the quieter *real* speaker is
deleted. A lav hearing its wearer at the wearer's normal speaking level is not hearing
someone across the table, however loud and however correlated that other track is. That is
what the spec's word *track-relative* is for.

**A candidate kept only by the veto is marked `ambiguous`**, because that is the one case
where the numeric evidence pointed at bleed and the pipeline overrode it. Marking every
merely-overlapping candidate ambiguous would make the flag noise; this way it means
something a human can act on.

**Levels are measured over shared intervals, never over whole candidates.** Two tracks are
compared over the samples they actually have in common, band-limited first so the comparison
is about voices rather than about which lav is closer to the air conditioning. Every read is
bounded by `correlation_window_ms` (INV-07): a candidate may be minutes long, and a
correlation over minutes is neither affordable nor more informative than one over seconds.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from dnd_audio.activity import PERMILLE, to_permille
from dnd_audio.activity.band import band_limited, rms_millibels
from dnd_audio.activity.scoring import ScoreTerms, score_candidate
from dnd_audio.artifacts.activity import EvidenceOutcome
from dnd_audio.config import ActivityConfig
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.syncqa import measure_lag

__all__ = [
    "REFERENCE_PERCENTILE",
    "Attribution",
    "AttributionResult",
    "AudioReader",
    "CandidateInput",
    "EvidenceRecord",
    "PairMeasurement",
    "attribute",
    "compare_pairs",
    "measure_levels",
    "speech_references",
]

#: Which percentile of a track's candidate levels becomes its speech reference (**OQ-017**).
#:
#: Not the median, which ADR-0014 originally specified: a track whose wearer spoke twice and
#: heard four other people has more bleed candidates than speech ones, and the median of that
#: is a bleed level — which would set the veto at bleed and disable the protection exactly
#: where it is needed. `nearest` interpolation makes the result one of the measured integers
#: rather than an average of two (INV-02).
#:
#: This is a guess about a real room, and it is registered as one. Two effects fight here and
#: only a real session says which wins: including bleed candidates drags the reference *down*
#: (the veto fires more often — conservative, the direction the spec asks for), while taking
#: the upper quartile pushes it *up* (a few unusually loud utterances weaken the veto for that
#: wearer's quieter speech). The graph records every candidate's level and its track's
#: reference, so answering this is reading one real session rather than running an experiment.
REFERENCE_PERCENTILE = 75

#: Returns ``[start, start + n)`` derivative samples of one track, in the session's own
#: derivative coordinates. The seam that keeps this module testable with synthetic audio and
#: keeps every read bounded.
AudioReader = Callable[[str, int, int], npt.NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class CandidateInput:
    """One detected region, before anything has been decided about it."""

    track_id: str
    start_sample: int
    end_sample: int
    derivative_start_sample: int
    derivative_end_sample: int
    probability_permille: int
    peak_probability_permille: int


@dataclass(frozen=True, slots=True)
class PairMeasurement:
    """Two candidates on different tracks, measured over the interval they share.

    Held per *unordered* pair and read from both directions, so the two candidates can never
    disagree about the correlation between them. The lag is antisymmetric by construction
    rather than by a second measurement: negating it is exact, while correlating the other
    way round could land on a different argmax at a tie.
    """

    left: int
    right: int
    overlap_start_sample: int
    overlap_end_sample: int
    compared_derivative_samples: int
    correlation_permille: int
    #: Positive means ``left``'s audio arrives later than ``right``'s.
    lag_derivative_samples: int
    left_level_mb: int
    right_level_mb: int

    def other(self, index: int) -> int:
        return self.right if index == self.left else self.left

    def lag_for(self, index: int) -> int:
        return self.lag_derivative_samples if index == self.left else -self.lag_derivative_samples

    def level_delta_for(self, index: int) -> int:
        """The other candidate's level minus this one's, over the shared interval."""
        if index == self.left:
            return self.right_level_mb - self.left_level_mb
        return self.left_level_mb - self.right_level_mb


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """What one comparison decided, from one candidate's point of view."""

    other_index: int
    overlap_start_sample: int
    overlap_end_sample: int
    compared_derivative_samples: int
    correlation_permille: int
    lag_derivative_samples: int
    score_margin_permille: int
    level_delta_mb: int
    outcome: EvidenceOutcome


@dataclass(frozen=True, slots=True)
class Attribution:
    """Everything decided about one candidate."""

    index: int
    band_level_mbfs: int
    relative_level_mb: int | None
    terms: ScoreTerms
    decision: Literal["retained", "suppressed"]
    ambiguous: bool
    suppressed_by: int | None
    evidence: tuple[EvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """The gate's output for a whole session."""

    attributions: tuple[Attribution, ...]
    #: Per track, the level its wearer speaks at. ``None`` where there was too little
    #: speech to establish one, which disables that track's veto rather than defaulting it.
    speech_references: dict[str, int | None]


def attribute(
    candidates: Sequence[CandidateInput], *, read: AudioReader, config: ActivityConfig
) -> AttributionResult:
    """Measure, score, and gate every candidate in one session.

    Four passes, in this order and not another: levels are needed before references,
    references and pair measurements before scores, and scores before any suppression can be
    decided — because the rule compares scores rather than raw levels, which is what stops
    the four-term score from being a decoration beside a loudness comparison.
    """
    levels = measure_levels(candidates, read=read, config=config)
    references = speech_references(candidates, levels, config=config)
    pairs = compare_pairs(candidates, read=read, config=config)

    by_candidate: dict[int, list[PairMeasurement]] = {index: [] for index in range(len(candidates))}
    for pair in pairs:
        by_candidate[pair.left].append(pair)
        by_candidate[pair.right].append(pair)

    relative = [
        _relative_level(levels[index], references.get(candidate.track_id))
        for index, candidate in enumerate(candidates)
    ]
    scores = [
        score_candidate(
            relative_level_mb=relative[index],
            probability_permille=candidate.probability_permille,
            level_delta_mb=_worst_level_delta(by_candidate[index], index),
            peak_correlation_permille=_peak_correlation(by_candidate[index]),
            settings=config.scoring,
        )
        for index, candidate in enumerate(candidates)
    ]

    return AttributionResult(
        attributions=tuple(
            _gate(
                index,
                levels[index],
                relative[index],
                scores,
                by_candidate[index],
                config=config,
            )
            for index in range(len(candidates))
        ),
        speech_references=references,
    )


def _relative_level(level: int, reference: int | None) -> int | None:
    """This candidate's level against its own track's speech reference.

    ``None`` where the track has no reference, and deliberately not zero: "as loud as this
    wearer usually is" and "we do not know how loud this wearer usually is" are different
    facts, and the veto must not fire on the second (ADR-0014).
    """
    return None if reference is None else level - reference


def measure_levels(
    candidates: Sequence[CandidateInput], *, read: AudioReader, config: ActivityConfig
) -> list[int]:
    """Band-limited level of each candidate, in millibels relative to full scale.

    Measured over a bounded window **centred** on the candidate. A candidate longer than
    `correlation_window_ms` is therefore characterized by its middle, which is deliberate:
    taking the first two seconds instead would measure onsets, and every utterance begins
    quietly.
    """
    cap = _window_samples(config)
    levels: list[int] = []
    for candidate in candidates:
        start, length = _centred(
            candidate.derivative_start_sample, candidate.derivative_end_sample, cap
        )
        levels.append(rms_millibels(band_limited(read(candidate.track_id, start, length))))
    return levels


def speech_references(
    candidates: Sequence[CandidateInput], levels: Sequence[int], *, config: ActivityConfig
) -> dict[str, int | None]:
    """What each wearer sounds like when they are the one talking.

    :data:`REFERENCE_PERCENTILE` of that track's own candidate levels — see its note for why
    that rather than the median, and for the open question it rests on (**OQ-017**).

    ``None`` for a track with fewer than `min_reference_candidates` candidates: a reference
    estimated from one or two regions is as likely to be measuring bleed as speech, and a
    veto built on it would fire in the wrong direction. Recording the absence keeps that
    visible in the graph instead of turning into a reference of zero.
    """
    grouped: dict[str, list[int]] = {}
    for candidate, level in zip(candidates, levels, strict=True):
        grouped.setdefault(candidate.track_id, []).append(level)

    references: dict[str, int | None] = {}
    for track_id, found in sorted(grouped.items()):
        if len(found) < config.bleed.min_reference_candidates:
            references[track_id] = None
            continue
        references[track_id] = int(
            np.percentile(np.asarray(found, dtype=np.int64), REFERENCE_PERCENTILE, method="nearest")
        )
    return references


def compare_pairs(
    candidates: Sequence[CandidateInput], *, read: AudioReader, config: ActivityConfig
) -> list[PairMeasurement]:
    """Correlate every pair of overlapping candidates that sit on different tracks.

    Two candidates on one track are two utterances, not a duplication of one, so they are
    never compared. Candidates that do not overlap in time cannot be the same sound.
    """
    cap = _window_samples(config)
    max_lag = max(1, config.correlation_max_lag_ms * DERIVATIVE_SAMPLE_RATE // 1000)
    measurements: list[PairMeasurement] = []

    for (left, first), (right, second) in itertools.combinations(enumerate(candidates), 2):
        if first.track_id == second.track_id:
            continue
        overlap_start = max(first.start_sample, second.start_sample)
        overlap_end = min(first.end_sample, second.end_sample)
        if overlap_end <= overlap_start:
            continue

        shared_start = max(first.derivative_start_sample, second.derivative_start_sample)
        shared_end = min(first.derivative_end_sample, second.derivative_end_sample)
        start, length = _centred(shared_start, shared_end, cap)
        if length <= 0:  # pragma: no cover - a 48 kHz overlap always covers a derivative sample
            continue

        left_audio = band_limited(read(first.track_id, start, length))
        right_audio = band_limited(read(second.track_id, start, length))
        lag, correlation = measure_lag(right_audio, left_audio, max_lag_samples=max_lag)

        measurements.append(
            PairMeasurement(
                left=left,
                right=right,
                overlap_start_sample=overlap_start,
                overlap_end_sample=overlap_end,
                compared_derivative_samples=length,
                correlation_permille=to_permille(correlation * PERMILLE),
                lag_derivative_samples=lag,
                left_level_mb=rms_millibels(left_audio),
                right_level_mb=rms_millibels(right_audio),
            )
        )
    return measurements


def _gate(
    index: int,
    level: int,
    relative_level: int | None,
    scores: Sequence[ScoreTerms],
    pairs: Sequence[PairMeasurement],
    *,
    config: ActivityConfig,
) -> Attribution:
    """Apply ADR-0014's rule to one candidate against every competitor it overlaps."""
    settings = config.bleed
    minimum_margin = round(settings.min_score_margin * PERMILLE)
    minimum_correlation = round(settings.min_correlation * PERMILLE)
    vetoed = relative_level is not None and relative_level >= -round(settings.veto_db * 100)

    evidence: list[EvidenceRecord] = []
    for pair in sorted(pairs, key=lambda item: item.other(index)):
        other = pair.other(index)
        margin = scores[other].total_permille - scores[index].total_permille
        evidence.append(
            EvidenceRecord(
                other_index=other,
                overlap_start_sample=pair.overlap_start_sample,
                overlap_end_sample=pair.overlap_end_sample,
                compared_derivative_samples=pair.compared_derivative_samples,
                correlation_permille=pair.correlation_permille,
                lag_derivative_samples=pair.lag_for(index),
                score_margin_permille=margin,
                level_delta_mb=pair.level_delta_for(index),
                outcome=_outcome(
                    vetoed=vetoed,
                    correlated=pair.correlation_permille >= minimum_correlation,
                    dominated=margin >= minimum_margin,
                ),
            )
        )

    suppressing = [item for item in evidence if item.outcome == "suppresses"]
    # Largest margin wins, and the lowest index breaks a tie — never the order the
    # comparisons happened to be measured in (INV-02).
    best = max(
        suppressing, key=lambda item: (item.score_margin_permille, -item.other_index), default=None
    )

    return Attribution(
        index=index,
        band_level_mbfs=level,
        relative_level_mb=relative_level,
        terms=scores[index],
        decision="suppressed" if best is not None else "retained",
        # Ambiguous means the numbers said bleed and the veto kept it anyway. Flagging every
        # merely-overlapping candidate would make this mean nothing.
        #
        # The margin and correlation are still checked here rather than inferred from the
        # outcome label. `_outcome` now only returns `vetoed_by_track_level` once both have
        # passed, so the three agree — but the flag is what a downstream milestone reads, and
        # it should not silently change meaning if that precedence is ever revisited.
        ambiguous=best is None
        and vetoed
        and any(
            item.outcome == "vetoed_by_track_level"
            and item.score_margin_permille >= minimum_margin
            and item.correlation_permille >= minimum_correlation
            for item in evidence
        ),
        suppressed_by=None if best is None else best.other_index,
        evidence=tuple(evidence),
    )


def _outcome(*, vetoed: bool, correlated: bool, dominated: bool) -> EvidenceOutcome:
    """Why this comparison did not suppress — or that it did.

    The veto is reported **last**, for a comparison the other two conditions had already
    satisfied, because that is the only case where the veto is what changed the answer.

    Reporting it first — whenever it merely *applied* — labelled every pair on a vetoed
    candidate `vetoed_by_track_level`, including pairs whose competitor was quieter or
    unrelated and where nothing was overridden at all. An operator auditing why a speaker
    survived then read "the veto saved this" against a competitor that never threatened it,
    which is worse than no diagnostic (M3's verify phase). The candidate-level `ambiguous`
    flag was always computed from the margin and correlation directly, so this changes what
    the evidence *says*, never what the gate decides.
    """
    if not correlated:
        return "insufficient_correlation"
    if not dominated:
        return "insufficient_margin"
    if vetoed:
        return "vetoed_by_track_level"
    return "suppresses"


def _worst_level_delta(pairs: Sequence[PairMeasurement], index: int) -> int | None:
    """How much louder the loudest competitor is. ``None`` when there is none."""
    deltas = [pair.level_delta_for(index) for pair in pairs]
    return max(deltas) if deltas else None


def _peak_correlation(pairs: Sequence[PairMeasurement]) -> int | None:
    """The strongest relationship to any competitor. ``None`` when there is none."""
    return max((pair.correlation_permille for pair in pairs), default=None)


def _window_samples(config: ActivityConfig) -> int:
    return max(1, config.bleed.correlation_window_ms * DERIVATIVE_SAMPLE_RATE // 1000)


def _centred(start: int, end: int, cap: int) -> tuple[int, int]:
    """A bounded window of at most ``cap`` samples, centred on ``[start, end)``."""
    length = min(end - start, cap)
    return start + (end - start - length) // 2, length
