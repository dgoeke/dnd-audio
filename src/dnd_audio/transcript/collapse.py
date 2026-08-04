"""Post-ASR duplicate collapse, and the overlap flag.

Every lav hears the room, so the same utterance can survive M3's bleed gate on two tracks —
that is deliberate, because the gate keeps anything ambiguous. This is where text finally gets
a vote, and it is the most dangerous code in the milestone: a wrong collapse deletes speech
and leaves nothing in the audio to show it happened.

So the spec's three conditions are three *independent* conditions, all required:

1. **Substantial temporal overlap**, measured against the shorter of the two segments and
   compared by integer cross-multiplication — no float ratio decided at a boundary.
2. **Strongly similar normalized text**, with a hard floor on length underneath it. The spec
   names the failure: "yes" and "no" match perfectly and mean two people agreeing with each
   other. Below `min_text_words` or `min_text_chars`, text similarity is not evidence at all
   and nothing here can collapse.
3. **Supporting acoustic evidence, taken from the activity graph** rather than measured again
   (ADR-0017). M3 already correlated every overlapping cross-track candidate pair and recorded
   the peak and its lag; reading those numbers is exact where a second correlator would be a
   second opinion. Where a segment covers several candidates, **every** pair that exists must
   clear the correlation threshold rather than the best one — the conservative direction — or
   the source scores must differ by a compelling margin, which is the spec's own "or
   compelling source-dominance evidence".

Two segments the graph never compared have no acoustic evidence, and are therefore kept. That
is not a gap: candidates M3 did not compare are candidates that did not overlap in time, and
two utterances that did not overlap are not one utterance heard twice.

**The survivor is the one with the best source score**, the model-independent number M3
produced — never the longer text or the more confident-sounding transcript, which would let
what was said decide who said it. Ties go to the lower segment id, which is a function of time
and track and therefore of the input rather than of iteration order (INV-02).

Everything rejected is recorded with the numbers that rejected it. Only what was actually
rejected: every evaluated pair would be quadratic growth in the artifact for no audit value.
"""

from __future__ import annotations

from dataclasses import dataclass

from dnd_audio.activity import PERMILLE
from dnd_audio.artifacts.activity import ActivityGraph, CandidateEvidence
from dnd_audio.artifacts.records import (
    CollapseRule,
    RejectedAlternative,
    SegmentDecision,
    TranscriptDecision,
    segment_id,
)
from dnd_audio.config import DuplicateConfig
from dnd_audio.transcript.normalize import comparison_key, similarity_permille, word_count
from dnd_audio.transcript.segments import SegmentDraft

__all__ = ["CollapseResult", "SegmentVerdict", "collapse"]


@dataclass(frozen=True, slots=True)
class SegmentVerdict:
    """What was decided about one draft segment."""

    decision: SegmentDecision
    duplicate_of_segment_id: str | None
    collapse_rule: CollapseRule | None
    overlap: bool
    rejected_alternatives: tuple[RejectedAlternative, ...]


@dataclass(frozen=True, slots=True)
class CollapseResult:
    """One verdict per draft, in the drafts' own order, plus the audit trail."""

    verdicts: tuple[SegmentVerdict, ...]
    decisions: tuple[TranscriptDecision, ...]


@dataclass(frozen=True, slots=True)
class _Comparison:
    """Two segments, and every number the three conditions are decided on."""

    overlap_samples: int
    overlap_permille: int
    similarity_permille: int
    #: The weakest correlation among the candidate pairs behind them; ``None`` when the graph
    #: compared no pair, which is itself a reason not to collapse.
    correlation_permille: int | None
    #: The better segment's source score minus the worse one's, always non-negative.
    score_margin_permille: int
    #: Index of the segment that would survive.
    winner: int
    loser: int


def collapse(
    drafts: list[SegmentDraft],
    graph: ActivityGraph,
    *,
    settings: DuplicateConfig,
    overlap_min_samples: int,
) -> CollapseResult:
    """Decide which drafts are duplicates and which retained segments overlap.

    ``drafts`` must already be in canonical order — sorted by start sample then track — because
    a segment's id is its position in that order (ADR-0019), and the ids are what a collapse
    decision refers to.
    """
    scores = {candidate.candidate_id: candidate.score_permille for candidate in graph.candidates}
    evidence = _evidence(graph)
    speakers = {track.track_id: track.speaker_id for track in graph.tracks}

    comparisons = [
        _compare(drafts, index, other, scores=scores, evidence=evidence)
        for index, other in _pairs(drafts)
    ]
    # **The best-scoring segment absorbs first** (ADR-0032). In canonical order, A=800 absorbs
    # B=700 and is then forbidden from being absorbed by C=900, so A and C both reach the
    # transcript — contradicting this module's own rule that the survivor is the copy the
    # model-independent evidence prefers. Resolving `(C, A)` before `(A, B)` makes that shape
    # unreachable. It is a sort, not new logic: `_is_duplicate` still gates every pair, so the
    # only thing that changes is *which* of two mutual duplicates survives.
    #
    # The tie-break is on segment ids, which are a function of time and track — of the input
    # rather than of iteration order (INV-02).
    comparisons.sort(
        key=lambda item: (-_score(drafts[item.winner], scores), item.winner, item.loser)
    )

    duplicate_of: dict[int, int] = {}
    absorbed: dict[int, list[_Comparison]] = {}
    contained_losers: set[int] = set()
    # The complete legacy algorithm is the first global pass. A containment edge cannot
    # preempt an existing similarity decision merely because its winner scores higher
    # (ADR-0033; M9 plan review finding 1).
    for comparison in comparisons:
        if comparison.winner in duplicate_of or comparison.loser in duplicate_of:
            continue
        # A segment that has already absorbed another cannot itself be absorbed: a chain of
        # duplicates has no surviving text at the end of it, and the records artifact refuses
        # one. Only the *loser* is constrained — one survivor absorbing several copies of one
        # utterance is the ordinary case with six lavs in a room.
        if comparison.loser in absorbed:
            continue
        if not _is_duplicate(comparison, drafts, settings=settings):
            continue
        duplicate_of[comparison.loser] = comparison.winner
        absorbed.setdefault(comparison.winner, []).append(comparison)

    # Only survivors of the unchanged pass are eligible for the distinct conservative rule.
    for comparison in comparisons:
        if comparison.winner in duplicate_of or comparison.loser in duplicate_of:
            continue
        if not _is_contained_fragment(comparison, drafts, settings=settings):
            continue
        duplicate_of[comparison.loser] = comparison.winner
        absorbed.setdefault(comparison.winner, []).append(comparison)
        contained_losers.add(comparison.loser)

    return CollapseResult(
        verdicts=tuple(
            _verdict(
                index,
                drafts,
                duplicate_of=duplicate_of,
                absorbed=absorbed,
                contained_losers=contained_losers,
                speakers=speakers,
                overlap_min_samples=overlap_min_samples,
            )
            for index in range(len(drafts))
        ),
        decisions=_decisions(drafts, absorbed, contained_losers=contained_losers),
    )


def _pairs(drafts: list[SegmentDraft]) -> list[tuple[int, int]]:
    """Every cross-track pair that overlaps in time, in canonical order.

    Two segments on one track are two utterances, never one heard twice — the same rule the
    bleed gate applies to candidates.
    """
    found: list[tuple[int, int]] = []
    for index, first in enumerate(drafts):
        for other in range(index + 1, len(drafts)):
            second = drafts[other]
            if first.track_id == second.track_id:
                continue
            shared = min(first.end_sample, second.end_sample) - max(
                first.start_sample, second.start_sample
            )
            if shared > 0:
                found.append((index, other))
    return found


def _evidence(graph: ActivityGraph) -> dict[tuple[str, str], CandidateEvidence]:
    """Every pairwise measurement M3 recorded, addressable from either direction."""
    found: dict[tuple[str, str], CandidateEvidence] = {}
    for candidate in graph.candidates:
        for item in candidate.evidence:
            found[candidate.candidate_id, item.other_candidate_id] = item
    return found


def _compare(
    drafts: list[SegmentDraft],
    index: int,
    other: int,
    *,
    scores: dict[str, int],
    evidence: dict[tuple[str, str], CandidateEvidence],
) -> _Comparison:
    first, second = drafts[index], drafts[other]
    overlap = min(first.end_sample, second.end_sample) - max(
        first.start_sample, second.start_sample
    )
    shorter = min(first.end_sample - first.start_sample, second.end_sample - second.start_sample)

    first_score = _score(first, scores)
    second_score = _score(second, scores)
    # The better source score survives. A tie goes to the lower index, which is the lower
    # segment id, which is a function of time and track rather than of iteration order.
    winner, loser = (index, other) if first_score >= second_score else (other, index)

    return _Comparison(
        overlap_samples=overlap,
        overlap_permille=min(PERMILLE, overlap * PERMILLE // shorter) if shorter else 0,
        similarity_permille=similarity_permille(first.text, second.text),
        correlation_permille=_weakest_correlation(first, second, evidence),
        score_margin_permille=abs(first_score - second_score),
        winner=winner,
        loser=loser,
    )


def _score(draft: SegmentDraft, scores: dict[str, int]) -> int:
    """The segment's source score: the best of the candidates it owns.

    A segment covering several candidates is the wordless case, where its text is one string
    the model produced for all of them; the strongest candidate is what that string was most
    likely heard through.
    """
    return max((scores.get(candidate, 0) for candidate in draft.candidate_ids), default=0)


def _weakest_correlation(
    first: SegmentDraft,
    second: SegmentDraft,
    evidence: dict[tuple[str, str], CandidateEvidence],
) -> int | None:
    """The lowest peak correlation among the candidate pairs behind two segments.

    The weakest rather than the strongest, because collapsing deletes speech: a segment whose
    candidates correlate with the other's in some places and not others is exactly the case to
    keep. ``None`` when there is no evidence this pair of segments can be decided on.

    ``None`` covers two cases, and the second one was a hole. The graph having compared *no*
    pair is the obvious one. The other is a segment covering several candidates — the wordless
    case ADR-0017 names — where the graph compared only *some* of them: a merged `A1+A2` whose
    `A2` correlates strongly with `B` while `A1` was never compared to anything on B's track.
    Taking the minimum over the pairs that happen to exist silently treated `A1`'s unmeasured
    speech as covered by `A2`'s evidence, and collapsing the merged record then deleted `A1`'s
    words outright. Every candidate on both sides must have been measured against the other
    side, or there is no evidence about this pair of *segments* (M4's verify phase, found by
    independent review).
    """
    measured = {
        (left, right): evidence[left, right].correlation_permille
        for left in first.candidate_ids
        for right in second.candidate_ids
        if (left, right) in evidence
    }
    if not measured:
        return None
    if {left for left, _ in measured} != set(first.candidate_ids):
        return None
    if {right for _, right in measured} != set(second.candidate_ids):
        return None
    return min(measured.values())


def _is_duplicate(
    comparison: _Comparison, drafts: list[SegmentDraft], *, settings: DuplicateConfig
) -> bool:
    """All three conditions, evaluated in integers."""
    first, second = drafts[comparison.winner], drafts[comparison.loser]
    shorter = min(first.end_sample - first.start_sample, second.end_sample - second.start_sample)
    # Integer cross-multiplication rather than a float ratio compared at a boundary.
    required = round(settings.min_overlap_ratio * PERMILLE)
    if comparison.overlap_samples * PERMILLE < required * shorter:
        return False

    if not _long_enough(first, settings) or not _long_enough(second, settings):
        return False
    if comparison.similarity_permille < round(settings.min_text_similarity * PERMILLE):
        return False

    correlated = comparison.correlation_permille is not None and (
        comparison.correlation_permille >= round(settings.min_correlation * PERMILLE)
    )
    dominant = comparison.score_margin_permille >= round(settings.min_score_margin * PERMILLE)
    # The graph having compared *something* is required either way: two segments it never
    # compared are two candidates that did not overlap, and score dominance alone would then
    # be a comparison of unrelated speech.
    return comparison.correlation_permille is not None and (correlated or dominant)


def _is_contained_fragment(
    comparison: _Comparison, drafts: list[SegmentDraft], *, settings: DuplicateConfig
) -> bool:
    """The separate proper-containment rule, evaluated only after legacy collapse.

    Exact short utterances are structurally excluded: proper containment requires the
    acoustically preferred survivor to have strictly more normalized words (ADR-0033,
    OQ-018).
    """
    winner, loser = drafts[comparison.winner], drafts[comparison.loser]
    shorter = min(
        winner.end_sample - winner.start_sample,
        loser.end_sample - loser.start_sample,
    )
    required = round(settings.min_overlap_ratio * PERMILLE)
    if comparison.overlap_samples * PERMILLE < required * shorter:
        return False
    if comparison.correlation_permille is None:
        return False
    if comparison.score_margin_permille < round(settings.contained_min_score_margin * PERMILLE):
        return False
    outer = comparison_key(winner.text).split()
    inner = comparison_key(loser.text).split()
    if not inner or len(outer) <= len(inner):
        return False
    # Contiguous, not a bag of words: collapsing may lose no unique normalized word.
    return any(
        outer[index : index + len(inner)] == inner for index in range(len(outer) - len(inner) + 1)
    )


def _long_enough(draft: SegmentDraft, settings: DuplicateConfig) -> bool:
    """Whether this text is long enough for similarity to mean anything at all."""
    return (
        word_count(draft.text) >= settings.min_text_words
        and len(comparison_key(draft.text)) >= settings.min_text_chars
    )


def _verdict(
    index: int,
    drafts: list[SegmentDraft],
    *,
    duplicate_of: dict[int, int],
    absorbed: dict[int, list[_Comparison]],
    contained_losers: set[int],
    speakers: dict[str, str],
    overlap_min_samples: int,
) -> SegmentVerdict:
    if index in duplicate_of:
        return SegmentVerdict(
            decision="duplicate",
            duplicate_of_segment_id=segment_id(duplicate_of[index]),
            collapse_rule=("contained_fragment" if index in contained_losers else None),
            overlap=False,
            rejected_alternatives=tuple(
                _alternative(drafts[comparison.loser], comparison, speakers)
                for comparison in sorted(absorbed.get(index, []), key=lambda item: item.loser)
            ),
        )
    return SegmentVerdict(
        decision="retained",
        duplicate_of_segment_id=None,
        collapse_rule=None,
        overlap=_overlaps_another_speaker(
            index,
            drafts,
            duplicate_of=duplicate_of,
            speakers=speakers,
            overlap_min_samples=overlap_min_samples,
        ),
        rejected_alternatives=tuple(
            _alternative(drafts[comparison.loser], comparison, speakers)
            for comparison in sorted(absorbed.get(index, []), key=lambda item: item.loser)
        ),
    )


def _overlaps_another_speaker(
    index: int,
    drafts: list[SegmentDraft],
    *,
    duplicate_of: dict[int, int],
    speakers: dict[str, str],
    overlap_min_samples: int,
) -> bool:
    """The spec's definition, exactly: a *retained, non-duplicate* other speaker's segment.

    A segment whose only overlap is with something that was collapsed is not overlapping
    speech — the collapsed one is not in the transcript to overlap with.
    """
    draft = drafts[index]
    speaker = speakers.get(draft.track_id, draft.track_id)
    for other, candidate in enumerate(drafts):
        if other == index or other in duplicate_of:
            continue
        if speakers.get(candidate.track_id, candidate.track_id) == speaker:
            continue
        shared = min(draft.end_sample, candidate.end_sample) - max(
            draft.start_sample, candidate.start_sample
        )
        if shared > 0 and shared >= overlap_min_samples:
            return True
    return False


def _alternative(
    draft: SegmentDraft, comparison: _Comparison, speakers: dict[str, str]
) -> RejectedAlternative:
    return RejectedAlternative(
        segment_id=segment_id(comparison.loser),
        track_id=draft.track_id,
        speaker_id=speakers.get(draft.track_id, draft.track_id),
        text=draft.text,
        overlap_permille=comparison.overlap_permille,
        text_similarity_permille=comparison.similarity_permille,
        correlation_permille=comparison.correlation_permille,
        score_margin_permille=comparison.score_margin_permille,
    )


def _decisions(
    drafts: list[SegmentDraft],
    absorbed: dict[int, list[_Comparison]],
    *,
    contained_losers: set[int],
) -> tuple[TranscriptDecision, ...]:
    """One auditable record per collapse, with the numbers that produced it."""
    found: list[TranscriptDecision] = []
    for winner, comparisons in sorted(absorbed.items()):
        for comparison in sorted(comparisons, key=lambda item: item.loser):
            loser = drafts[comparison.loser]
            contained = comparison.loser in contained_losers
            text_evidence = (
                "the weaker normalized words are properly contained by the survivor, "
                if contained
                else (f"their normalized text matches at {comparison.similarity_permille}/1000, ")
            )
            found.append(
                TranscriptDecision(
                    code=("contained_fragment_collapsed" if contained else "duplicate_collapsed"),
                    subject=segment_id(comparison.loser),
                    detail=(
                        f"{loser.track_id} at sample {loser.start_sample} was collapsed into "
                        f"{segment_id(winner)}: the two overlap by "
                        f"{comparison.overlap_permille}/1000 of the shorter segment, "
                        f"{text_evidence}"
                        f"and the activity graph's weakest correlation between their "
                        f"candidates is {comparison.correlation_permille}/1000 with a source "
                        f"score margin of {comparison.score_margin_permille}/1000."
                    ),
                )
            )
    return tuple(found)
