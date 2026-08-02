"""The source score, held to the two properties the spec names and one it implies.

`score_candidate` is the only place where four measurements become one number, and
ADR-0014 puts that number on the decision path: a candidate is suppressed as bleed only
when a competitor's *score* beats it by a margin. So an inverted term here does not merely
mislabel a diagnostic, it deletes the wrong speaker.

Three things are therefore asserted harder than the rest:

**The level term is track-relative.** The spec forbids "a single global loudness
comparison that always awards a time interval to the loudest person". Two candidates at
the same level *relative to their own wearers* must score identically on that term, and
nothing in the call may say how loud either actually was.

**Correlation contributes as independence.** A higher peak correlation must *lower* the
score. With the other sign the function ranks the best-recorded copy of someone else's
voice above the original, which is a wrong attribution that looks confident.

**Absent evidence reads neutral, not zero.** A track with no speech reference and a
candidate nothing overlaps are both ordinary — the first utterance of a session is exactly
that — and a zero default would make every one of them look like bleed.

Every expectation below is arithmetic done by hand from the configured spans and weights,
not a value read back out of the implementation.
"""

from __future__ import annotations

import inspect

import pytest

from dnd_audio.activity.scoring import NEUTRAL_PERMILLE, ScoreTerms, score_candidate
from dnd_audio.config import ScoringConfig

#: A candidate sitting in the middle of every range, so no term is clamped and a change in
#: any single input is visible in the total. With the default spans and weights its terms
#: are level 800, confidence 500, dominance 500, independence 500, total 605.
_BASELINE_LEVEL_MB = -600
_BASELINE_PROBABILITY = 500
_BASELINE_DELTA_MB = 0
_BASELINE_CORRELATION = 500

_DEFAULT_SETTINGS = ScoringConfig()

#: Weights that make the total the plain mean of the four terms, so a hand-computed
#: expectation needs no weighting arithmetic.
_EQUAL_WEIGHTS = ScoringConfig(
    level_weight=1.0, confidence_weight=1.0, dominance_weight=1.0, correlation_weight=1.0
)


def _score(
    *,
    relative_level_mb: int | None = _BASELINE_LEVEL_MB,
    probability_permille: int = _BASELINE_PROBABILITY,
    level_delta_mb: int | None = _BASELINE_DELTA_MB,
    peak_correlation_permille: int | None = _BASELINE_CORRELATION,
    settings: ScoringConfig = _DEFAULT_SETTINGS,
) -> ScoreTerms:
    """Score the baseline candidate with whatever this test varies, and nothing else."""
    return score_candidate(
        relative_level_mb=relative_level_mb,
        probability_permille=probability_permille,
        level_delta_mb=level_delta_mb,
        peak_correlation_permille=peak_correlation_permille,
        settings=settings,
    )


def _others(terms: ScoreTerms, *, without: str) -> dict[str, int]:
    """The three terms a single-input test is holding fixed."""
    names = ["level_permille", "confidence_permille", "dominance_permille", "correlation_permille"]
    return {name: getattr(terms, name) for name in names if name != without}


class TestEachTermMovesTheTotal:
    """One input at a time, so a term wired to nothing cannot hide behind the other three.

    A single combined test would pass with any one term dropped from the sum, because the
    other three would still move the total in the same direction.
    """

    def test_a_candidate_nearer_its_own_tracks_reference_scores_higher(self) -> None:
        near = _score(relative_level_mb=-600)  # level (3000-600)/3000 -> 800
        far = _score(relative_level_mb=-1800)  # level (3000-1800)/3000 -> 400

        assert near.level_permille == 800
        assert far.level_permille == 400
        # 0.35*800 + 0.25*500 + 0.25*500 + 0.15*500 = 605, against 0.35*400 + 325 = 465.
        assert near.total_permille == 605
        assert far.total_permille == 465
        assert _others(near, without="level_permille") == _others(far, without="level_permille")

    def test_a_more_confident_detection_scores_higher(self) -> None:
        confident = _score(probability_permille=900)
        unsure = _score(probability_permille=500)

        assert confident.confidence_permille == 900
        # 280 + 0.25*900 + 125 + 75 = 705, against 605.
        assert confident.total_permille == 705
        assert unsure.total_permille == 605
        assert _others(confident, without="confidence_permille") == _others(
            unsure, without="confidence_permille"
        )

    def test_dominating_every_competitor_scores_higher_than_being_dominated(self) -> None:
        """`level_delta_mb` is the competitor minus this candidate, so negative is louder."""
        dominant = _score(level_delta_mb=-2000)  # 20 dB above the best competitor
        dominated = _score(level_delta_mb=2000)  # 20 dB below it

        assert dominant.dominance_permille == 1000
        assert dominated.dominance_permille == 0
        # 280 + 125 + 0.25*1000 + 75 = 730, against 280 + 125 + 0 + 75 = 480.
        assert dominant.total_permille == 730
        assert dominated.total_permille == 480
        assert _others(dominant, without="dominance_permille") == _others(
            dominated, without="dominance_permille"
        )

    def test_a_less_correlated_candidate_scores_higher(self) -> None:
        independent = _score(peak_correlation_permille=100)
        entangled = _score(peak_correlation_permille=900)

        assert independent.correlation_permille == 900
        assert entangled.correlation_permille == 100
        # 280 + 125 + 125 + 0.15*900 = 665, against 280 + 125 + 125 + 15 = 545.
        assert independent.total_permille == 665
        assert entangled.total_permille == 545
        assert _others(independent, without="correlation_permille") == _others(
            entangled, without="correlation_permille"
        )


class TestTheLevelTermIsTrackRelative:
    """The spec's central prohibition, and the one a "reasonable" simplification breaks."""

    def test_two_candidates_at_the_same_relative_level_score_the_same_on_it(self) -> None:
        """A softly-spoken player at their normal level must not lose to a loud one.

        Both candidates are 2 dB under their own wearer's speech reference. One of those
        wearers may be 20 dB quieter than the other in absolute terms — the function
        cannot tell, and that is the property. Under the global loudness rule the spec
        forbids, the quieter wearer would score far lower here and be suppressed as the
        louder one's bleed every time the two genuinely spoke at once.
        """
        soft_speaker = _score(
            relative_level_mb=-200, probability_permille=950, peak_correlation_permille=100
        )
        loud_speaker = _score(
            relative_level_mb=-200, probability_permille=400, peak_correlation_permille=800
        )

        # (3000 - 200) / 3000 * 1000 = 933.33 -> 933.
        assert soft_speaker.level_permille == 933
        assert soft_speaker.level_permille == loud_speaker.level_permille
        # And the rest of the score did move, so the equality above is not two identical calls.
        assert soft_speaker.total_permille != loud_speaker.total_permille

    def test_nothing_in_the_call_says_how_loud_either_candidate_actually_was(self) -> None:
        """There is no absolute level to compare, so a global comparison cannot be written.

        Pinned as a signature assertion because the prohibition is structural: adding a
        `level_dbfs` parameter is the first step of reintroducing the rule the spec bans,
        and it would otherwise be a silent, plausible-looking change.
        """
        parameters = set(inspect.signature(score_candidate).parameters)

        assert parameters == {
            "relative_level_mb",
            "probability_permille",
            "level_delta_mb",
            "peak_correlation_permille",
            "settings",
        }

    def test_the_ramp_is_linear_between_exact_endpoints(self) -> None:
        """At the configured span the term is 0, at the reference 1000, halfway 500."""
        span_mb = 30 * 100  # level_span_db defaults to 30 dB, and levels are millibels.

        assert _score(relative_level_mb=-span_mb).level_permille == 0
        assert _score(relative_level_mb=-span_mb // 2).level_permille == 500
        assert _score(relative_level_mb=0).level_permille == 1000

    def test_the_ramp_follows_the_configured_span_rather_than_a_constant(self) -> None:
        wide = ScoringConfig(level_span_db=40.0)

        assert _score(relative_level_mb=-4000, settings=wide).level_permille == 0
        assert _score(relative_level_mb=-2000, settings=wide).level_permille == 500
        assert _score(relative_level_mb=-1000, settings=wide).level_permille == 750
        assert _score(relative_level_mb=0, settings=wide).level_permille == 1000


class TestCorrelationContributesAsIndependence:
    def test_a_higher_correlation_lowers_the_score(self) -> None:
        """The sign that decides which of two tracks keeps an utterance.

        A candidate strongly correlated with another track is more likely a copy of it than
        a voice of its own. Inverted, this term rewards being a copy — and since ADR-0014's
        gate suppresses on a score margin, the best-recorded copy of someone else's voice
        would win the interval from the person who actually spoke.
        """
        scores = [
            _score(peak_correlation_permille=value).total_permille for value in (0, 500, 1000)
        ]

        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1]

    def test_the_term_is_the_complement_of_the_measured_correlation(self) -> None:
        assert _score(peak_correlation_permille=0).correlation_permille == 1000
        assert _score(peak_correlation_permille=250).correlation_permille == 750
        assert _score(peak_correlation_permille=1000).correlation_permille == 0


class TestAbsentEvidenceReadsNeutral:
    def test_a_track_with_no_speech_reference_gives_a_neutral_level_term(self) -> None:
        terms = _score(relative_level_mb=None)

        assert terms.level_permille == NEUTRAL_PERMILLE == 500

    def test_a_candidate_nothing_overlaps_gets_full_marks_for_dominance(self) -> None:
        """Nothing contests it, so there is no competitor it could be quieter than."""
        assert _score(level_delta_mb=None).dominance_permille == 1000

    def test_a_candidate_nothing_overlaps_gets_full_marks_for_independence(self) -> None:
        assert _score(peak_correlation_permille=None).correlation_permille == 1000

    def test_a_solo_utterance_on_an_unreferenced_track_still_scores_well(self) -> None:
        """The consequence, which is why the neutral default is not a stylistic choice.

        The first confident utterance of a session, on a track that has not yet
        accumulated enough speech for a reference, has all three of these absences at once.
        Defaulting them to zero would score it 225 — below any plausible margin — and
        ADR-0014's gate would read a person talking alone as somebody else's bleed.
        """
        alone = _score(
            relative_level_mb=None,
            probability_permille=900,
            level_delta_mb=None,
            peak_correlation_permille=None,
        )
        # 0.35*500 + 0.25*900 + 0.25*1000 + 0.15*1000 = 175 + 225 + 250 + 150.
        assert alone.total_permille == 800

        as_if_zero = _score(
            relative_level_mb=-3000,  # the level term's floor
            probability_permille=900,
            level_delta_mb=2000,  # dominated by the full span
            peak_correlation_permille=1000,  # perfectly correlated
        )
        assert as_if_zero.total_permille == 225
        assert alone.total_permille > 750


class TestClamping:
    """One outlier measurement must not carry the total past either end."""

    @pytest.mark.parametrize(
        ("relative_level_mb", "expected"),
        [(500, 1000), (60_000, 1000), (-3000, 0), (-50_000, 0)],
    )
    def test_the_level_term_saturates_at_both_ends(
        self, relative_level_mb: int, expected: int
    ) -> None:
        """Above its own reference is not better than at it; 500 dB down is not worse than 30."""
        assert _score(relative_level_mb=relative_level_mb).level_permille == expected

    @pytest.mark.parametrize(
        ("level_delta_mb", "expected"),
        [(-2000, 1000), (-100_000, 1000), (2000, 0), (100_000, 0)],
    )
    def test_the_dominance_term_saturates_outside_its_span(
        self, level_delta_mb: int, expected: int
    ) -> None:
        assert _score(level_delta_mb=level_delta_mb).dominance_permille == expected

    def test_an_out_of_range_probability_is_clamped(self) -> None:
        assert _score(probability_permille=1500).confidence_permille == 1000
        assert _score(probability_permille=-20).confidence_permille == 0

    def test_an_out_of_range_correlation_is_clamped(self) -> None:
        assert _score(peak_correlation_permille=1500).correlation_permille == 0
        assert _score(peak_correlation_permille=-20).correlation_permille == 1000


class TestWeightsAreRelative:
    def test_doubling_every_weight_changes_nothing(self) -> None:
        """Normalization by the sum, asserted rather than assumed.

        An operator raising all four weights means "no change", and a function that
        required them to sum to one would instead double every score.
        """
        doubled = ScoringConfig(
            level_weight=_DEFAULT_SETTINGS.level_weight * 2,
            confidence_weight=_DEFAULT_SETTINGS.confidence_weight * 2,
            dominance_weight=_DEFAULT_SETTINGS.dominance_weight * 2,
            correlation_weight=_DEFAULT_SETTINGS.correlation_weight * 2,
        )

        assert _score(settings=doubled) == _score(settings=_DEFAULT_SETTINGS)

    @pytest.mark.parametrize(
        ("weight_field", "term_field"),
        [
            ("level_weight", "level_permille"),
            ("confidence_weight", "confidence_permille"),
            ("dominance_weight", "dominance_permille"),
            ("correlation_weight", "correlation_permille"),
        ],
    )
    def test_zeroing_three_weights_leaves_the_total_equal_to_the_survivor(
        self, weight_field: str, term_field: str
    ) -> None:
        weights = dict.fromkeys(
            ("level_weight", "confidence_weight", "dominance_weight", "correlation_weight"), 0.0
        )
        weights[weight_field] = 1.0
        settings = ScoringConfig(**weights)

        # Four deliberately distinct terms: 800, 470, 750, 650.
        terms = _score(
            relative_level_mb=-600,
            probability_permille=470,
            level_delta_mb=-1000,
            peak_correlation_permille=350,
            settings=settings,
        )

        assert terms.total_permille == getattr(terms, term_field)
        assert len({terms.level_permille, terms.confidence_permille}) == 2


class TestRounding:
    """Halves go away from zero, which is not what :func:`round` does."""

    def test_a_total_landing_on_an_exact_half_rounds_up(self) -> None:
        """With banker's rounding this would be 998, and two scores could swap order.

        The four terms sum to 3994 under equal weights, so the total is exactly 998.5.
        `round(998.5)` is 998 — Python rounds a half to the even neighbour — while
        `round(999.5)` is 1000, so the direction a half goes would depend on the value.
        """
        terms = _score(
            relative_level_mb=0,  # 1000
            probability_permille=1000,  # 1000
            level_delta_mb=-2000,  # 1000
            peak_correlation_permille=6,  # 994
            settings=_EQUAL_WEIGHTS,
        )

        assert (
            terms.level_permille
            + terms.confidence_permille
            + terms.dominance_permille
            + terms.correlation_permille
        ) == 3994
        assert terms.total_permille == 999

    @pytest.mark.parametrize(("relative_level_mb", "expected"), [(-7996, 1), (-7988, 2)])
    def test_a_ramp_landing_on_an_exact_half_rounds_up(
        self, relative_level_mb: int, expected: int
    ) -> None:
        """An 80 dB span puts the ramp on exact halves: 0.5 -> 1 and 1.5 -> 2.

        Banker's rounding would answer 0 and 2, disagreeing with itself about which way a
        half goes — the inconsistency the module's own rounding rule exists to avoid.
        """
        settings = ScoringConfig(level_span_db=80.0)

        assert _score(relative_level_mb=relative_level_mb, settings=settings).level_permille == (
            expected
        )


class TestEveryTermStaysInRange:
    @pytest.mark.parametrize("relative_level_mb", [None, -100_000, 0, 100_000])
    @pytest.mark.parametrize("probability_permille", [-5000, 500, 5000])
    @pytest.mark.parametrize("level_delta_mb", [None, -100_000, 100_000])
    @pytest.mark.parametrize("peak_correlation_permille", [None, -5000, 5000])
    def test_no_input_can_push_a_term_outside_zero_to_one_thousand(
        self,
        relative_level_mb: int | None,
        probability_permille: int,
        level_delta_mb: int | None,
        peak_correlation_permille: int | None,
    ) -> None:
        """Per-mille is the artifact's unit; a term outside it is a corrupt document."""
        terms = _score(
            relative_level_mb=relative_level_mb,
            probability_permille=probability_permille,
            level_delta_mb=level_delta_mb,
            peak_correlation_permille=peak_correlation_permille,
            settings=ScoringConfig(
                level_weight=1.0,
                confidence_weight=0.0,
                dominance_weight=0.5,
                correlation_weight=0.25,
                level_span_db=0.5,
                dominance_span_db=120.0,
            ),
        )

        for value in (
            terms.level_permille,
            terms.confidence_permille,
            terms.dominance_permille,
            terms.correlation_permille,
            terms.total_permille,
        ):
            assert 0 <= value <= 1000
