"""Finding the marker: per-chirp matched filters, then a sequence check on the gaps.

**Why per-chirp and not one correlation against the whole waveform.** A single template gives
the sharpest peak on a clean signal and is the obvious implementation. It is also the one that
fails hardest if a phone's media pipeline resamples or reschedules playback, because a
whole-template correlation degrades with *total* elapsed error while the marker is over a
second long. Correlating each chirp separately and checking the gaps afterwards degrades with
the error over one chirp instead, and turns a timing perturbation into a measurable gap
residual rather than a lost detection. **ADR-0042** records the physical phone-bench result and
freezes the tolerance.

**The gaps are the code.** With identical chirps — which two of the three candidates have — a
per-chirp peak cannot tell the first sweep from the third; every position scores alike. The
sequence is identified entirely by the *asymmetric* inter-chirp gaps, which is also what makes
a reversed pattern fail: play the marker backwards and the gaps arrive in the other order. A
strong isolated chirp is never a detection.

**Bounded memory, and it is not only the correlator.** Blocks are fixed size with a
template-length carry, so the correlation working set is independent of the searched range
(INV-07). Accumulated *occurrences* are the other half, and the obvious streaming proof misses
them: non-maximum suppression bounds nearby candidates and says nothing about the number of
separated ones. So there is an explicit ceiling, and reaching it **fails** — this project does
not truncate silently, and a truncated occurrence list is indistinguishable from a session that
genuinely had that many (ADR-0041).

**Scores are integer permille and every comparison happens there.** The correlation itself is
floating point, but it is quantized once, at the boundary, and no threshold or tie-break ever
compares two floats. That is the defect M8 removed from `sync_qa`, and it is easier to keep out
than to remove twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from dnd_audio.errors import DndAudioError
from dnd_audio.marker.spec import MarkerSpec
from dnd_audio.marker.synth import marker_templates

__all__ = [
    "BLOCK_SAMPLES",
    "PERMILLE",
    "ChirpHit",
    "DetectorThresholds",
    "MarkerOccurrence",
    "OccurrenceCeilingError",
    "WindowReader",
    "detect_occurrences",
    "to_permille",
]

#: The score domain. One part in a thousand is finer than any acoustic distinction this
#: instrument can make and coarse enough that a last-bit difference in an FFT cannot move a
#: decision.
PERMILLE: Final = 1000

#: Samples per correlation block. Fixed, so peak memory is a property of this constant and
#: the longest template rather than of how much audio was asked for. About 1.4 s at 48 kHz:
#: large enough that per-block FFT overhead is irrelevant against a multi-minute window,
#: small enough that six tracks' working sets are megabytes.
BLOCK_SAMPLES: Final = 1 << 16


class OccurrenceCeilingError(DndAudioError):
    """More accepted occurrences than the configured ceiling permits.

    A failure rather than a truncation, deliberately. Silently keeping the first *n* would
    make a pathological input — a stuck loop of the marker, or a threshold set so low that
    room tone qualifies — indistinguishable from a session that genuinely contained that many,
    and the analysis would look complete (ADR-0041).
    """

    default_code = "marker_occurrence_ceiling"


class WindowReader(Protocol):
    """Anything that yields ``[start, start + n)`` of one track as float32.

    Exactly :meth:`dnd_audio.timeline.reader.TrackReader.read`'s shape, narrowed to the one
    method the detector needs — so `tests/test_memory.py` can hand it an instrumented reader
    and prove the read size is bounded, rather than trusting this module's own account.
    """

    def read(self, start_sample: int, n_samples: int, /) -> npt.NDArray[np.float32]: ...


def to_permille(value: float) -> int:
    """A normalized score as integer permille, halves away from zero.

    The one place a float becomes a comparable number. Every threshold, every runner-up
    separation and every tie-break below operates on the result, so a decision can never
    depend on the last bit of an FFT.
    """
    if not math.isfinite(value):
        return 0
    scaled = value * PERMILLE
    return math.floor(scaled + 0.5) if scaled >= 0 else -math.floor(-scaled + 0.5)


def _is_locally_ambiguous(score: int, runner_up: int, minimum_separation: int) -> bool:
    """Whether an unclaimed local alternative is too close, entirely in permille."""
    return runner_up > 0 and score - runner_up < minimum_separation


@dataclass(frozen=True, slots=True)
class DetectorThresholds:
    """Every number that decides whether a sound is the marker.

    Frozen for marker v1 by **ADR-0042**, against both the physical phone/DJI positive bench
    and the 13.7-minute real-speech negative sweep. The failure directions are not symmetric
    and are worth stating: too strict and the marker
    is undetectable on the one device it will ever be played from, which is visible
    immediately; too loose and three unrelated transients become an occurrence, which is
    invisible until it moves a start/end pair.
    """

    #: A single chirp must reach this to be a candidate peak. Low enough to survive a lav's
    #: band limiting and a room's reverberation, which both cost correlation (ADR-0042).
    #:
    #: **Both sides are measured.**
    #: Across 13.7 minutes of real DJI recordings — two voices overlapping on purpose, plus
    #: hand claps, the broadband transient most likely to be mistaken for a chirp — no
    #: sequence was accepted by any candidate with this forced as low as **100 permille**,
    #: and the strongest single chirp anywhere reached only 186. What rejects speech is
    #: mostly the three-chirp *gap structure*, not this number
    #: (`docs/fixtures/2026-08-05-marker-false-positive-sweep.md`). So the bench may lower
    #: this substantially if the farthest lav needs it. The bench-selected v1's weakest fixed-
    #: position sequence was 404 permille; 300 retains 104 permille of positive margin while
    #: staying three times above the lowest threshold proved clean on real speech (ADR-0042).
    min_chirp_score_permille: int = 300
    #: The assembled sequence's score — the **weakest** of its chirps, so a sequence is only
    #: as good as its worst link rather than as good as its best.
    min_sequence_score_permille: int = 300
    #: A clean occurrence's weakest selected chirp must beat the strongest unclaimed local
    #: same-chirp alternative by this much. Fifty permille is deliberately independent of
    #: the absolute 300-permille acceptance floor: it asks whether the arrival is decisive,
    #: not whether it is loud. The phone/DJI bench left no local alternatives at all, while
    #: 50 keeps a useful guard against a room echo almost as persuasive as the direct path
    #: (ADR-0042).
    min_runner_up_separation_permille: int = 50
    #: How far each measured inter-chirp gap may sit from the canonical one — 30 ms, generous
    #: against scheduling jitter between one chirp and the next.
    #:
    #: **This is not what bounds tolerance to a clock difference, despite the name.** Measured
    #: 2026-08-05 across all three candidates: detection survives to roughly 1000 ppm of time
    #: stretch and fails by 2000, while 2000 ppm moves even the longest gap by only 31 samples
    #: — two percent of this tolerance. What actually fails first is *per-chirp* correlation,
    #: because stretching a chirp detunes it against its own template, and the loss scales
    #: with time-bandwidth product. So the real constant governing clock tolerance is the
    #: chirp's duration and bandwidth, which is a property of the spec rather than of the
    #: detector. V1 measured at most 29 samples, leaving 1411 samples of margin without fitting
    #: a cross-device tolerance to one phone (ADR-0042).
    gap_tolerance_samples: int = 1440
    #: Suppression radius around a per-chirp peak, in samples. Fifty milliseconds collapses
    #: one correlation lobe/reflection family; each chirp has its own peak list, so this cannot
    #: suppress another chirp in the sequence (ADR-0042).
    nms_radius_samples: int = 2400
    #: Complete sequences closer than 150 ms are one acoustic event. This is deliberately
    #: separate from per-chirp NMS: the synthetic room response produces a coherent echo at
    #: 106 ms, while a local competing chirp beyond 50 ms must remain visible to the runner-
    #: up diagnostic. A 1.47-second v1 cannot legitimately be replayed 150 ms apart.
    sequence_nms_radius_samples: int = 7200
    #: How far another track's occurrence may sit from the reference's and still be the same
    #: acoustic event. Generous against 0.5-3 m of propagation spread (1.5-9 ms) plus the
    #: 33.3 ms timecode quantum M8 measured — 100 ms covers both with room. V1 measured a
    #: 1878-sample maximum against this 4800-sample bound (ADR-0042).
    association_lag_samples: int = 4800
    #: Above this fraction of samples at or near full scale inside a detection, the arrival is
    #: reported as clipped: the peak position is still usable, its score is not. No track
    #: crossed this at approximately 90% phone volume (ADR-0042).
    clipping_ratio_permille: int = 10
    #: Below this RMS the window carries no usable signal, which is a different outcome from
    #: a weak match and gets its own diagnostic — the distinction M8 had to add to `sync_qa`.
    #: No bench track crossed it (ADR-0042).
    weak_signal_rms_permille: int = 1
    #: A fixed-geometry start/end lag change at or above one millisecond is material enough
    #: to warn. V1's approximately 11.8-minute fixed-position repeat changed by at most 17
    #: samples; 48 samples leaves 31 samples of measured repeat margin and is still far below
    #: DJI's 1600-sample timecode quantum (ADR-0042).
    material_arrival_change_samples: int = 48
    #: Accepted occurrences retained per track before the run fails. Thirty-two is far above any
    #: plausible bench take (three plays at each end, plus two moved-phone diagnostics) and
    #: far below anything that threatens memory.
    max_occurrences_per_track: int = 32

    def __post_init__(self) -> None:
        if self.min_sequence_score_permille < self.min_chirp_score_permille:
            message = (
                f"min_sequence_score_permille={self.min_sequence_score_permille} is below "
                f"min_chirp_score_permille={self.min_chirp_score_permille}, so the sequence "
                f"threshold could never reject anything the chirp threshold admitted"
            )
            raise ValueError(message)
        for name, value in (
            ("min_chirp_score_permille", self.min_chirp_score_permille),
            ("min_runner_up_separation_permille", self.min_runner_up_separation_permille),
            ("min_sequence_score_permille", self.min_sequence_score_permille),
        ):
            if not 0 < value <= PERMILLE:
                message = f"{name}={value} is outside (0, {PERMILLE}]"
                raise ValueError(message)
        for name, value in (
            ("gap_tolerance_samples", self.gap_tolerance_samples),
            ("nms_radius_samples", self.nms_radius_samples),
            ("sequence_nms_radius_samples", self.sequence_nms_radius_samples),
            ("association_lag_samples", self.association_lag_samples),
            ("max_occurrences_per_track", self.max_occurrences_per_track),
            ("material_arrival_change_samples", self.material_arrival_change_samples),
        ):
            if value <= 0:
                message = f"{name}={value} must be positive"
                raise ValueError(message)

    def identity(self) -> dict[str, int]:
        """Every threshold, by name, for the analysis identity document.

        Separate from any hash of it, so a test can assert *which* components are present —
        M2's `derivative_identity_document` lesson: a key that changes for the right reason
        can still be missing the component that matters later.
        """
        return {
            "association_lag_samples": self.association_lag_samples,
            "clipping_ratio_permille": self.clipping_ratio_permille,
            "gap_tolerance_samples": self.gap_tolerance_samples,
            "max_occurrences_per_track": self.max_occurrences_per_track,
            "material_arrival_change_samples": self.material_arrival_change_samples,
            "min_chirp_score_permille": self.min_chirp_score_permille,
            "min_runner_up_separation_permille": self.min_runner_up_separation_permille,
            "min_sequence_score_permille": self.min_sequence_score_permille,
            "nms_radius_samples": self.nms_radius_samples,
            "sequence_nms_radius_samples": self.sequence_nms_radius_samples,
            "weak_signal_rms_permille": self.weak_signal_rms_permille,
        }


@dataclass(frozen=True, slots=True)
class ChirpHit:
    """One chirp of one occurrence, where it was found and how well it matched."""

    chirp_index: int
    #: Session sample of the chirp's **first** sample, not of its correlation peak. The two
    #: differ by the template length, and conflating them is how a lag acquires a constant
    #: offset that every track shares and nothing reveals.
    start_sample: int
    score_permille: int


@dataclass(frozen=True, slots=True)
class MarkerOccurrence:
    """One complete marker sequence found on one track."""

    #: Session sample of the marker's frozen anchor — the first sample of the first chirp
    #: (ADR-0041). This is the quantity every lag is computed from.
    anchor_sample: int
    #: The weakest chirp's score. A sequence is as good as its worst link.
    score_permille: int
    hits: tuple[ChirpHit, ...]
    #: Measured minus canonical, per gap. Zero on clean synthetic audio; the quantity
    #: ADR-0042 measures, and the reason the detector reports it rather than only using it.
    gap_errors_samples: tuple[int, ...]
    clipped: bool = False
    weak: bool = False
    #: The best unclaimed same-chirp peak local to this occurrence. Peaks belonging to every
    #: accepted occurrence are excluded first, so repeated valid plays cannot masquerade as
    #: ambiguity (ADR-0041).
    runner_up_permille: int = 0
    ambiguous: bool = False
    diagnostics: dict[str, int] = field(default_factory=dict)


def _normalized_scores(
    signal: npt.NDArray[np.float64], template: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Normalized correlation of ``template`` against every position in ``signal``.

    Normalized by both energies, so a quiet track and a loud one are comparable — the same
    reason `syncqa.measure_lag` normalizes, and without it the detector would rank arrivals by
    how close the phone happened to be.

    Returns one score per valid start position; ``len(signal) - len(template) + 1`` of them.
    """
    length = template.size
    if signal.size < length:
        return np.zeros(0, dtype=np.float64)

    energy = float(np.linalg.norm(template))
    if energy == 0.0:  # pragma: no cover - a spec with a silent chirp cannot be constructed
        return np.zeros(signal.size - length + 1, dtype=np.float64)

    products = correlate(signal, template, mode="valid", method="fft")
    windows = np.sqrt(
        np.maximum(correlate(signal * signal, np.ones(length), mode="valid", method="fft"), 0.0)
    )
    denominator = energy * windows
    scores = np.zeros_like(products)
    usable = denominator > 0.0
    scores[usable] = np.abs(products[usable]) / denominator[usable]
    # FFT correlation of a normalized pair can land a hair above 1.0 on rounding alone; the
    # permille quantizer would then report 1001, which is not a score.
    return np.clip(scores, 0.0, 1.0)


def _peaks(
    scores: npt.NDArray[np.float64], *, minimum: int, radius: int, offset: int
) -> list[tuple[int, int]]:
    """Local maxima above ``minimum`` permille, suppressed within ``radius``.

    Greedy: take the best, suppress its neighbourhood, repeat. Returns ``(sample, permille)``
    in descending score order, with ``offset`` added so positions are in the caller's domain.

    Suppression is what bounds this list per block. It does **not** bound the number of
    separated peaks, which is why the caller carries a ceiling.
    """
    if scores.size == 0:
        return []
    permille = np.array([to_permille(float(value)) for value in scores], dtype=np.int64)
    found: list[tuple[int, int]] = []
    remaining = permille.copy()
    while True:
        index = int(np.argmax(remaining))
        best = int(remaining[index])
        if best < minimum:
            return found
        found.append((index + offset, best))
        low = max(0, index - radius)
        high = min(remaining.size, index + radius + 1)
        remaining[low:high] = -1


def _assemble(
    spec: MarkerSpec,
    per_chirp: list[list[tuple[int, int]]],
    thresholds: DetectorThresholds,
) -> list[tuple[int, int, tuple[ChirpHit, ...], tuple[int, ...]]]:
    """Turn per-chirp peaks into complete sequences, by the gap structure.

    Walks from each candidate first chirp and requires every later chirp to appear at its
    canonical offset within tolerance. The canonical offsets come from the spec's own chirp
    intervals, so the gaps checked here are the gaps that were built — there is no second
    table of expected spacings to drift.
    """
    intervals = spec.chirp_intervals()
    canonical = tuple(start - intervals[0][0] for start, _ in intervals)
    gaps = spec.gap_intervals()
    canonical_gaps = tuple(end - start for start, end in gaps)

    by_position: list[dict[int, int]] = [dict(peaks) for peaks in per_chirp]
    assembled: list[tuple[int, int, tuple[ChirpHit, ...], tuple[int, ...]]] = []

    for anchor, first_score in per_chirp[0]:
        hits = [ChirpHit(chirp_index=0, start_sample=anchor, score_permille=first_score)]
        ok = True
        for index in range(1, len(spec.chirps)):
            expected = anchor + canonical[index]
            best: tuple[int, int] | None = None
            for position, score in by_position[index].items():
                if abs(position - expected) > thresholds.gap_tolerance_samples:
                    continue
                # Prefer the strongest, then the one closest to where it should be — never
                # the first encountered, which would make the result depend on dict order.
                key = (score, -abs(position - expected))
                if best is None or key > (best[1], -abs(best[0] - expected)):
                    best = (position, score)
            if best is None:
                ok = False
                break
            hits.append(ChirpHit(chirp_index=index, start_sample=best[0], score_permille=best[1]))
        if not ok:
            continue

        measured_gaps = tuple(
            hits[i + 1].start_sample - (hits[i].start_sample + spec.chirps[i].duration_samples)
            for i in range(len(hits) - 1)
        )
        errors = tuple(
            measured - expected
            for measured, expected in zip(measured_gaps, canonical_gaps, strict=True)
        )
        score = min(hit.score_permille for hit in hits)
        if score < thresholds.min_sequence_score_permille:
            continue
        assembled.append((anchor, score, tuple(hits), errors))

    return assembled


def _suppress_sequences(
    assembled: list[tuple[int, int, tuple[ChirpHit, ...], tuple[int, ...]]],
    *,
    radius: int,
) -> list[tuple[int, int, tuple[ChirpHit, ...], tuple[int, ...]]]:
    """Keep the best sequence within each ``radius``, deterministically.

    Ties choose the **lower sample**, which is the tie-break ADR-0041 fixes — never list
    order, which would depend on how the blocks happened to be cut.
    """
    ordered = sorted(assembled, key=lambda item: (-item[1], item[0]))
    kept: list[tuple[int, int, tuple[ChirpHit, ...], tuple[int, ...]]] = []
    for candidate in ordered:
        if all(abs(candidate[0] - other[0]) > radius for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item[0])


def detect_occurrences(
    reader: WindowReader,
    spec: MarkerSpec,
    *,
    interval: tuple[int, int],
    thresholds: DetectorThresholds | None = None,
    block_samples: int = BLOCK_SAMPLES,
) -> list[MarkerOccurrence]:
    """Every complete marker sequence inside the half-open ``interval`` of one track.

    Args:
        reader: The track to search, read in bounded windows.
        spec: Which marker to look for. Its templates are slices of the canonical waveform.
        interval: Half-open ``[start, end)`` in session samples. The caller is responsible
            for having added the matching halo — an occurrence whose anchor sits near an edge
            needs its whole length inside what is read (ADR-0041).
        block_samples: Correlation block size. A parameter so a memory test can shrink it and
            still exercise the seam logic, not something a caller should tune. Must be at
            least the longest template.

    Returns:
        Accepted occurrences in ascending anchor order.

    Raises:
        OccurrenceCeilingError: if more sequences are accepted than
            ``thresholds.max_occurrences_per_track``. Never truncated.
        ValueError: if ``block_samples`` is smaller than the longest chirp template.
    """
    settings = thresholds if thresholds is not None else DetectorThresholds()
    templates = [template.astype(np.float64) for template in marker_templates(spec)]
    longest = max(template.size for template in templates)
    start, end = interval
    if end - start < spec.total_samples:
        return []
    if block_samples <= 0:
        message = f"block_samples must be positive, got {block_samples}"
        raise ValueError(message)
    if block_samples < longest:
        # Refused rather than quietly raised to a workable value. Overlap-save carries
        # `longest - 1` samples between blocks, so a block shorter than a template makes
        # every segment too short to correlate — and the loop below would return **no
        # detections at all**, which reads exactly like a session where nobody played the
        # marker. A silent wrong answer is the one outcome this project does not ship.
        message = (
            f"block_samples={block_samples} is below the longest chirp template "
            f"({longest} samples), so no block could ever contain one. Use at least "
            f"{longest}; the default is {BLOCK_SAMPLES}."
        )
        raise ValueError(message)

    per_chirp: list[list[tuple[int, int]]] = [[] for _ in templates]
    position = start
    # Overlap-save: each block carries the previous block's tail so a template straddling the
    # seam is still matched. Without the carry, an occurrence landing on a block boundary
    # would be invisible — and would be invisible only at certain search offsets, which is the
    # worst shape of bug to find later.
    carry = longest - 1
    while position < end:
        stop = min(position + block_samples, end)
        read_from = max(start, position - carry)
        segment = np.asarray(reader.read(read_from, stop - read_from), dtype=np.float64)
        if segment.size < longest:
            break
        for index, template in enumerate(templates):
            scores = _normalized_scores(segment, template)
            per_chirp[index].extend(
                _peaks(
                    scores,
                    minimum=settings.min_chirp_score_permille,
                    radius=settings.nms_radius_samples,
                    offset=read_from,
                )
            )
        position = stop

    # Blocks overlap by `carry`, so a peak in the overlap is found twice — once per block,
    # at the same absolute position. Deduplicate on position, keeping the higher score.
    deduplicated: list[list[tuple[int, int]]] = []
    for peaks in per_chirp:
        best: dict[int, int] = {}
        for sample, score in peaks:
            best[sample] = max(best.get(sample, 0), score)
        deduplicated.append(sorted(best.items(), key=lambda item: (-item[1], item[0])))

    assembled = _assemble(spec, deduplicated, settings)
    kept = _suppress_sequences(assembled, radius=settings.sequence_nms_radius_samples)

    if len(kept) > settings.max_occurrences_per_track:
        message = (
            f"{len(kept)} marker occurrences were accepted in "
            f"[{start}, {end}), above the configured ceiling of "
            f"{settings.max_occurrences_per_track}. Nothing was truncated: a shortened list "
            f"would be indistinguishable from a session that genuinely contained that many. "
            f"Either the searched window is far wider than intended, or the thresholds are "
            f"low enough that ordinary audio qualifies."
        )
        raise OccurrenceCeilingError(message)

    return [
        _describe(reader, spec, settings, anchor, score, hits, errors, deduplicated, kept)
        for anchor, score, hits, errors in kept
    ]


def _describe(
    reader: WindowReader,
    spec: MarkerSpec,
    settings: DetectorThresholds,
    anchor: int,
    score: int,
    hits: tuple[ChirpHit, ...],
    errors: tuple[int, ...],
    per_chirp: list[list[tuple[int, int]]],
    accepted: list[tuple[int, int, tuple[ChirpHit, ...], tuple[int, ...]]],
    *,
    full_scale_permille: int = 990,
) -> MarkerOccurrence:
    """Attach the diagnostics that separate "found it" from "found it, and trust it".

    Clipping and weak signal are distinct outcomes, not degrees of one: a clipped arrival has
    a usable *position* and an untrustworthy *score*, while a silent window has neither. M8
    had to add exactly that distinction to `sync_qa` after conflating them cost six correct
    measurements.
    """
    span = spec.total_samples
    window = np.asarray(reader.read(anchor, span), dtype=np.float64)
    if window.size == 0:  # pragma: no cover - the anchor came from inside the read range
        return MarkerOccurrence(anchor, score, hits, errors)

    peak = float(np.abs(window).max())
    clipped_samples = int(np.count_nonzero(np.abs(window) >= full_scale_permille / PERMILLE))
    rms = float(np.sqrt(np.mean(window * window)))
    clipped_ratio = to_permille(clipped_samples / max(window.size, 1))

    runner_up = 0
    for hit, peaks in zip(hits, per_chirp, strict=True):
        for sample, other in peaks:
            if abs(sample - hit.start_sample) > settings.sequence_nms_radius_samples:
                continue
            claimed = any(
                abs(sample - occurrence_hits[hit.chirp_index].start_sample)
                <= settings.nms_radius_samples
                for _, _, occurrence_hits, _ in accepted
            )
            if not claimed:
                runner_up = max(runner_up, other)

    return MarkerOccurrence(
        anchor_sample=anchor,
        score_permille=score,
        hits=hits,
        gap_errors_samples=errors,
        clipped=clipped_ratio >= settings.clipping_ratio_permille,
        weak=to_permille(rms) < settings.weak_signal_rms_permille,
        runner_up_permille=runner_up,
        ambiguous=_is_locally_ambiguous(
            score, runner_up, settings.min_runner_up_separation_permille
        ),
        diagnostics={
            "peak_permille": to_permille(peak),
            "rms_permille": to_permille(rms),
            "clipped_samples": clipped_samples,
        },
    )
