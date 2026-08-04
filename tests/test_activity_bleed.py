"""The bleed gate: what it suppresses, and — more importantly — what it refuses to.

This is the milestone's dangerous module. A wrongly suppressed candidate leaves no trace in
the audio, and a transcript missing one side of an argument reads exactly like an argument
that only had one side. So the tests are arranged around the *four* ways a comparison can
go, one class each, rather than around one happy path and one sad one:

* both conditions met and the level below the veto → suppressed;
* louder but uncorrelated → kept;
* correlated but not dominant → kept;
* correlated, dominant, and at the track's own speech level → **kept, by the veto**.

The last one is the case independent review produced against the first plan, and it has its
own contrast test: the same audio with the track's reference removed *is* suppressed. Without
that contrast, "the quiet speaker survived" would prove only that some threshold happened not
to be met.

The audio here is synthetic and built per test, because these are properties of the *rule*
and a fixture would only make them harder to read. The end-to-end proofs over the real
fixtures are in `tests/test_activity_run.py`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.activity.bleed import (
    REFERENCE_PERCENTILE,
    CandidateInput,
    attribute,
    compare_pairs,
    measure_levels,
    speech_references,
)
from dnd_audio.config import ActivityConfig, BleedConfig
from dnd_audio.fixtures import synth
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

#: One second at the detector's rate.
SECOND: Final = DERIVATIVE_SAMPLE_RATE

#: 3 ms — sound crossing a table, and the canonical fixture's own bleed delay. Far enough
#: outside zero lag that a correlator pinned there misses it entirely.
BLEED_LAG: Final = 48


class Room:
    """Per-track derivative audio, served the way the runner serves it.

    Reads are clamped and zero-filled past the end, matching `_DerivativeAudio`: the point of
    the seam is that this module never opens a file, so a test can state the acoustics
    directly instead of synthesizing a session to imply them.
    """

    def __init__(self, tracks: dict[str, npt.NDArray[np.float32]]) -> None:
        self.tracks = tracks
        self.reads: list[tuple[str, int, int]] = []

    def read(self, track_id: str, start: int, n_samples: int) -> npt.NDArray[np.float32]:
        self.reads.append((track_id, start, n_samples))
        samples = self.tracks[track_id]
        window = np.zeros(n_samples, dtype=np.float32)
        available = samples[start : start + n_samples]
        window[: available.shape[0]] = available
        return window


def voice(seed: int, *, gain: float, n_samples: int = 2 * SECOND) -> npt.NDArray[np.float32]:
    """Speech-shaped noise at the derivative rate. Content is irrelevant; level is not."""
    return synth.speech_shaped(n_samples, DERIVATIVE_SAMPLE_RATE, seed=seed, gain=gain)


def place(
    signal: npt.NDArray[np.float32], *, at: int, total: int, delay: int = 0
) -> npt.NDArray[np.float32]:
    """One event on an otherwise near-silent track, with a self-noise floor under it."""
    track = synth.noise_floor(total, seed=at + delay)
    start = at + delay
    end = min(start + signal.shape[0], total)
    track[start:end] += signal[: end - start]
    return track


def candidate(track_id: str, start: int, end: int, *, probability: int = 900) -> CandidateInput:
    """A candidate on the derivative grid, with its 48 kHz interval derived exactly."""
    return CandidateInput(
        track_id=track_id,
        start_sample=start * 3,
        end_sample=end * 3,
        derivative_start_sample=start,
        derivative_end_sample=end,
        probability_permille=probability,
        peak_probability_permille=min(1000, probability + 50),
    )


def two_speaker_room(
    *, speaker_gain: float, listener_gain: float, correlated: bool, total: int = 6 * SECOND
) -> tuple[Room, list[CandidateInput]]:
    """`tx-a` speaks at 1 s; `tx-b` carries either a copy of it or its own unrelated sound."""
    speech = voice(1, gain=speaker_gain)
    heard = (
        synth.bleed_of(speech, delay_samples=BLEED_LAG, attenuation_db=0.0) * listener_gain
        if correlated
        else voice(2, gain=listener_gain * speaker_gain)
    )
    room = Room(
        {
            "tx-a": place(speech, at=SECOND, total=total),
            "tx-b": place(np.asarray(heard, dtype=np.float32), at=SECOND, total=total),
        }
    )
    return room, [
        candidate("tx-a", SECOND, 3 * SECOND),
        candidate("tx-b", SECOND, 3 * SECOND),
    ]


def settings(**overrides: object) -> ActivityConfig:
    return ActivityConfig(bleed=BleedConfig(**overrides))  # type: ignore[arg-type]


def decisions(
    room: Room, candidates: list[CandidateInput], config: ActivityConfig
) -> dict[str, str]:
    found = attribute(candidates, read=room.read, config=config)
    return {candidates[item.index].track_id: item.decision for item in found.attributions}


class TestConservatism:
    """The four quadrants. Suppression needs every condition; anything else keeps."""

    def test_a_loud_correlated_neighbour_suppresses(self) -> None:
        """The case the gate exists for: tx-b is hearing tx-a and nothing else."""
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        assert decisions(room, candidates, settings()) == {
            "tx-a": "retained",
            "tx-b": "suppressed",
        }

    def test_a_loud_but_uncorrelated_neighbour_does_not(self) -> None:
        """Two people talking at once, one louder. Both are speaking; both are kept.

        A loudness comparison alone would delete the quieter one here, which is the failure
        the spec names outright.
        """
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.06, correlated=False)
        assert decisions(room, candidates, settings()) == {
            "tx-a": "retained",
            "tx-b": "retained",
        }

    def test_a_correlated_but_undominant_neighbour_does_not(self) -> None:
        """Two lavs hearing the same thing at nearly the same level.

        Which one owns it is genuinely unclear, so both survive and M4 decides with text —
        which is exactly the division of labour INV-09 requires.
        """
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.95, correlated=True)
        assert decisions(room, candidates, settings()) == {
            "tx-a": "retained",
            "tx-b": "retained",
        }

    def test_neither_condition_keeps_both(self) -> None:
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.9, correlated=False)
        assert decisions(room, candidates, settings()) == {
            "tx-a": "retained",
            "tx-b": "retained",
        }

    def test_candidates_that_do_not_overlap_are_never_compared(self) -> None:
        """Bleed is simultaneous by definition. Two sounds an hour apart are two sounds."""
        speech = voice(1, gain=0.3)
        room = Room(
            {
                "tx-a": place(speech, at=SECOND, total=8 * SECOND),
                "tx-b": place(voice(1, gain=0.01), at=5 * SECOND, total=8 * SECOND),
            }
        )
        candidates = [
            candidate("tx-a", SECOND, 3 * SECOND),
            candidate("tx-b", 5 * SECOND, 7 * SECOND),
        ]
        found = attribute(candidates, read=room.read, config=settings())
        assert [item.evidence for item in found.attributions] == [(), ()]
        assert all(item.decision == "retained" for item in found.attributions)

    def test_two_candidates_on_one_track_are_never_compared(self) -> None:
        """They are two utterances by one person, not one utterance heard twice."""
        speech = voice(1, gain=0.3)
        room = Room({"tx-a": place(speech, at=SECOND, total=8 * SECOND)})
        candidates = [
            candidate("tx-a", SECOND, 3 * SECOND),
            candidate("tx-a", 2 * SECOND, 4 * SECOND),
        ]
        found = attribute(candidates, read=room.read, config=settings())
        assert [item.evidence for item in found.attributions] == [(), ()]


class TestTheVeto:
    """A lav hearing its wearer at the wearer's normal level is not hearing bleed."""

    def room(
        self, *, overlap_gain: float, quiet_gain: float = 0.03
    ) -> tuple[Room, list[CandidateInput]]:
        """Both wearers speak alone three times, then `tx-a` speaks loudly at 12 s.

        `overlap_gain` is how much of its own voice `tx-b` contributes during that overlap:
        its solo level means both are genuinely talking, and zero means `tx-b` is silent and
        its lav is carrying nothing but `tx-a`.

        Three solos **per track**, not just for `tx-b`: a track with too few candidates has no
        reference, its level term reads neutral, and its score drops for a reason that has
        nothing to do with the acoustics. That artefact quietly collapsed the score margin in
        an earlier version of this fixture and made the veto look load-bearing when it was
        not.
        """
        total = 24 * SECOND
        loud, quiet = 0.30, quiet_gain
        tracks = {
            "tx-a": synth.noise_floor(total, seed=11),
            "tx-b": synth.noise_floor(total, seed=12),
        }
        candidates: list[CandidateInput] = []

        for index, at in enumerate((SECOND, 4 * SECOND, 7 * SECOND)):
            tracks["tx-a"][at : at + SECOND] += voice(20 + index, gain=loud, n_samples=SECOND)
            candidates.append(candidate("tx-a", at, at + SECOND))
        for index, at in enumerate((2 * SECOND, 5 * SECOND, 8 * SECOND)):
            tracks["tx-b"][at : at + SECOND] += voice(30 + index, gain=quiet, n_samples=SECOND)
            candidates.append(candidate("tx-b", at, at + SECOND))

        at = 12 * SECOND
        speech = voice(40, gain=loud)
        heard = np.asarray(
            synth.bleed_of(speech, delay_samples=BLEED_LAG, attenuation_db=18.0), dtype=np.float32
        )
        tracks["tx-a"][at : at + speech.shape[0]] += speech
        tracks["tx-b"][at : at + heard.shape[0]] += heard
        if overlap_gain:
            own = voice(41, gain=overlap_gain)
            tracks["tx-b"][at : at + own.shape[0]] += own
        candidates.append(candidate("tx-a", at, at + 2 * SECOND))
        candidates.append(candidate("tx-b", at, at + 2 * SECOND))

        return Room(tracks), candidates

    def overlap(self, room: Room, candidates: list[CandidateInput], config: ActivityConfig):  # type: ignore[no-untyped-def]
        """The attribution of `tx-b`'s candidate during the overlap — the one under test."""
        found = attribute(candidates, read=room.read, config=config)
        assert candidates[-1].track_id == "tx-b"
        return found, found.attributions[len(candidates) - 1]

    def test_a_quiet_speaker_at_their_own_level_survives_a_dominant_correlated_neighbour(
        self,
    ) -> None:
        """The reviewer's case. Both numeric conditions say bleed; the veto overrules them."""
        room, candidates = self.room(overlap_gain=0.03)
        _, decided = self.overlap(room, candidates, settings())

        assert decided.decision == "retained"
        record = decided.evidence[0]
        assert record.outcome == "vetoed_by_track_level"
        # Both numeric conditions really were met. Without this the test would pass whenever
        # some threshold happened not to be reached, and prove nothing about the veto.
        assert record.correlation_permille >= 500
        assert record.score_margin_permille >= 150

    def test_the_same_audio_is_suppressed_when_the_track_has_no_reference(self) -> None:
        """The contrast that makes the previous test mean something.

        One thing — whether `tx-b` has enough of its own speech to know what its wearer sounds
        like — flips the outcome on identical audio. That is the veto being load-bearing
        rather than incidental.

        **Both floors are raised, and that is the only change ADR-0029 made here.** After the
        two-pass estimator there are two populations a reference can come from — the
        candidates that won attribution, and the unclassified mixture it falls back to — so
        putting a track beyond reach of a reference means putting it beyond reach of both. The
        claim under test is unchanged: no reference, no veto, and the identical overlap is
        suppressed.
        """
        room, candidates = self.room(overlap_gain=0.03)
        found, decided = self.overlap(
            room,
            candidates,
            settings(min_reference_candidates=99, min_attributed_reference_candidates=99),
        )

        assert found.speech_references["tx-b"] is None
        assert decided.decision == "suppressed"
        assert decided.evidence[0].outcome == "suppresses"

    def test_pure_bleed_on_a_track_with_a_reference_is_still_suppressed(self) -> None:
        """The veto must not retain everything on a track that happens to have a reference.

        `tx-b`'s wearer speaks loudly when they speak, and is silent here — the lav carries
        only `tx-a`, well below what this wearer sounds like. The veto correctly declines to
        fire, which is the half of the rule that keeps it from being "retain everything".
        """
        room, candidates = self.room(overlap_gain=0.0, quiet_gain=0.30)
        found, decided = self.overlap(room, candidates, settings())

        assert found.speech_references["tx-b"] is not None
        assert decided.relative_level_mb is not None
        assert decided.relative_level_mb < -1200, "the veto's own threshold is 12 dB"
        assert decided.decision == "suppressed"

    def test_bleed_that_arrives_at_the_wearers_own_level_is_kept(self) -> None:
        """A real limit of the veto, asserted rather than discovered later.

        A loud speaker's bleed into a *quiet* wearer's lav can arrive at that wearer's own
        speaking level. The veto then cannot distinguish it from that wearer talking, and
        keeps it. That is the conservative direction the spec asks for — losing real
        overlapped speech is worse than extra ASR compute — but it means the gate does not
        suppress everything a human would, and M4's post-ASR collapse is what catches the
        rest. Recorded here so the behaviour is a decision rather than a surprise (OQ-017).
        """
        room, candidates = self.room(overlap_gain=0.0, quiet_gain=0.03)
        _, decided = self.overlap(room, candidates, settings())

        assert decided.relative_level_mb is not None
        assert decided.relative_level_mb > -1200
        assert decided.decision == "retained"
        assert decided.evidence[0].outcome == "vetoed_by_track_level"


class TestTheEvidenceSaysWhatActuallyHappened:
    """`vetoed_by_track_level` is a claim about a comparison, not about the candidate.

    The evidence is per compared *pair*, and it is what an operator reads to find out why a
    speaker survived or vanished. The veto is a candidate-level fact, so reporting it on
    every pair the moment it applied labelled comparisons the veto had nothing to do with —
    including ones where the competitor was quieter or unrelated and suppression was never
    on the table. That reads as "this was nearly suppressed and the veto saved it", which is
    false, and on the most dangerous decision in the pipeline a false diagnostic is worse
    than none (M3's verify phase).
    """

    def test_an_unrelated_competitor_is_not_reported_as_a_veto(self) -> None:
        """Same audio, same vetoed candidate — only the correlation threshold moves.

        `tx-b`'s wearer is genuinely talking, so the veto still applies and the candidate is
        still retained. But with the correlation bar above what the pair measured, nothing
        was overridden: the two signals simply were not related, and that is what the record
        has to say.
        """
        veto = TestTheVeto()
        room, candidates = veto.room(overlap_gain=0.03)

        _, vetoed = veto.overlap(room, candidates, settings())
        assert vetoed.decision == "retained"
        assert vetoed.evidence[0].outcome == "vetoed_by_track_level"
        measured = vetoed.evidence[0].correlation_permille

        _, decided = veto.overlap(room, candidates, settings(min_correlation=0.999))

        assert measured < 999, "precondition: the pair must fall below the raised bar"
        assert decided.relative_level_mb is not None
        assert decided.relative_level_mb > -1200, "precondition: the veto still applies"
        assert decided.decision == "retained"
        assert decided.evidence[0].outcome == "insufficient_correlation"
        assert decided.ambiguous is False, (
            "nothing was overridden, so this is an ordinary retention rather than a "
            "candidate the numbers condemned and the veto saved"
        )

    def test_an_undominant_competitor_is_not_reported_as_a_veto(self) -> None:
        """The other half: correlated, vetoed, but the competitor never out-scored it."""
        veto = TestTheVeto()
        room, candidates = veto.room(overlap_gain=0.03)
        _, decided = veto.overlap(room, candidates, settings(min_score_margin=0.99))

        assert decided.decision == "retained"
        assert decided.evidence[0].outcome == "insufficient_margin"
        assert decided.ambiguous is False


class TestSpeechReferences:
    def test_a_track_below_the_minimum_has_no_reference(self) -> None:
        """Recorded as absent rather than defaulted: a reference estimated from one region of
        an *unclassified mixture* is as likely to be measuring bleed as speech, and a veto
        built on it fires backwards.

        The two tracks here take the two different paths ADR-0029 separates, on one run:

        * `tx-a` speaks and wins its candidate, so **one winner is enough** — the gate has
          already concluded that region is this wearer, which is direct evidence in a way
          three candidates of a mixture are not.
        * `tx-b` only ever *hears* `tx-a`. It wins nothing, so it falls back to the mixture,
          where one candidate is below `min_reference_candidates` and the answer is `None`.
          That is what keeps a pure listener suppressible, which is the gate's whole purpose.
        """
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        found = attribute(candidates, read=room.read, config=settings())

        assert found.speech_references["tx-b"] is None
        assert found.reference_candidate_counts["tx-b"] == 0
        assert found.speech_references["tx-a"] is not None
        assert found.reference_candidate_counts["tx-a"] == 1

    def test_the_reference_is_the_upper_quartile_of_a_tracks_own_levels(self) -> None:
        """Not the median.

        A track whose wearer spoke twice and heard four other people has more bleed
        candidates than speech ones, and the median of that set is a bleed level — which
        would set the veto at bleed and disable it exactly where it is needed.
        """
        total = 30 * SECOND
        track = synth.noise_floor(total, seed=5)
        # Four quiet regions and two loud ones: the median sits among the quiet, the upper
        # quartile among the loud.
        gains = (0.01, 0.01, 0.01, 0.01, 0.3, 0.3)
        candidates = []
        for index, gain in enumerate(gains):
            at = (2 + 3 * index) * SECOND
            track[at : at + SECOND] += voice(40 + index, gain=gain, n_samples=SECOND)
            candidates.append(candidate("tx-a", at, at + SECOND))

        room = Room({"tx-a": track})
        levels = measure_levels(candidates, read=room.read, config=settings())
        references = speech_references(candidates, levels, config=settings())

        assert REFERENCE_PERCENTILE == 75
        assert references["tx-a"] is not None
        assert references["tx-a"] > sorted(levels)[len(levels) // 2], (
            "the reference must sit above the median, or a track that mostly hears other "
            "people would set its own speech level at bleed"
        )
        assert references["tx-a"] in levels, "nearest, not interpolated: no invented level"


class TestTheReferenceComesFromTheWinners:
    """ADR-0029, and the defect that produced it.

    `speech_references` — the bootstrap estimator — takes the upper quartile of *every*
    candidate on a track. On the 2026-08-03 jam capture that put `tx-d`'s reference at
    -57.80 dBFS, which is the level of the bleed it was hearing, not the level its wearer
    speaks at. One extra bleed candidate moved it 17 dB, because `nearest` interpolation
    lands on the largest of three values and the second-largest of four.

    The arithmetic gets worse with the roster, not better: at six speakers roughly 83% of any
    track's candidates are bleed, so the upper quartile sits in bleed territory for *every*
    participant. That is why raising the percentile is not a fix and why the population had to
    change instead.
    """

    def room(self, *, bleed_candidates: int) -> tuple[Room, list[CandidateInput]]:
        """One own-speech candidate on `tx-a`, and `bleed_candidates` copies of `tx-b`.

        Calibrated to the jam capture's measured acoustics: `tx-a`'s wearer is 17.4 dB above
        what `tx-a` hears of `tx-b`, which is the separation that recording actually had.
        """
        total = (4 + 3 * (bleed_candidates + 1)) * SECOND
        speaker = synth.noise_floor(total, seed=11)
        other = synth.noise_floor(total, seed=12)

        candidates: list[CandidateInput] = []
        own_at = 2 * SECOND
        speaker[own_at : own_at + SECOND] += voice(21, gain=0.30, n_samples=SECOND)
        candidates.append(candidate("tx-a", own_at, own_at + SECOND))

        for index in range(bleed_candidates):
            at = (5 + 3 * index) * SECOND
            spoken = voice(30 + index, gain=0.30, n_samples=SECOND)
            other[at : at + SECOND] += spoken
            # 17.4 dB down on `tx-a`: what a lav across the table hears of someone else.
            speaker[at : at + SECOND] += synth.bleed_of(
                spoken, delay_samples=BLEED_LAG, attenuation_db=17.4
            )[:SECOND]
            candidates.append(candidate("tx-a", at, at + SECOND))
            candidates.append(candidate("tx-b", at, at + SECOND))

        return Room({"tx-a": speaker, "tx-b": other}), candidates

    @pytest.mark.parametrize("bleed_candidates", range(1, 9))
    def test_the_reference_lands_on_own_speech_however_much_bleed_there_is(
        self, bleed_candidates: int
    ) -> None:
        """The completion criterion, spanning the boundary the old estimator fails at.

        The old rule passes at two bleed candidates and fails from three onward, so a test at
        a single N would have been a coin flip on which side it landed. `tx-a` has exactly one
        candidate of its own throughout; everything else on that track is someone else.
        """
        room, candidates = self.room(bleed_candidates=bleed_candidates)
        found = attribute(candidates, read=room.read, config=settings())

        levels = measure_levels(candidates, read=room.read, config=settings())
        own = levels[0]
        reference = found.speech_references["tx-a"]

        assert reference == own, (
            f"with {bleed_candidates} bleed candidate(s) the reference is {reference}, not "
            f"tx-a's own speech at {own}. A reference anchored on bleed sets the veto at "
            f"bleed, which protects bleed from suppression."
        )
        assert found.reference_candidate_counts["tx-a"] == 1

    def test_the_old_estimator_really_does_fail_here(self) -> None:
        """A contrast, so the parametrized test above is known to be capable of failing.

        Asserting that a fixed estimator succeeds proves nothing unless the broken one is
        shown to fail on the identical audio. Three bleed candidates is where `nearest`
        interpolation crosses over.
        """
        room, candidates = self.room(bleed_candidates=3)
        levels = measure_levels(candidates, read=room.read, config=settings())
        bootstrap = speech_references(candidates, levels, config=settings())

        assert bootstrap["tx-a"] != levels[0], (
            "the all-candidates estimator is supposed to be wrong here — if it is not, this "
            "fixture no longer reproduces the defect ADR-0029 exists for"
        )

    def test_a_speaker_who_only_ever_overlaps_is_not_deleted(self) -> None:
        """The regression the plan review produced, and the reason the fallback exists.

        A quieter person who speaks *only* while someone else is speaking wins nothing in the
        first pass — every one of their candidates is contested. A winners-only reference
        would therefore leave them with none, disable their veto, and suppress them: exactly
        the failure ADR-0014 was written against, reintroduced by the fix for a different one.

        `mutual_bleed_session` cannot show this, because it gives its quiet speaker three solo
        utterances. Here `tx-b` has none: every one of its candidates is dominated by `tx-a`
        and correlated with it, so the first pass condemns all three.

        **The fixture is asserted to have that shape rather than assumed to.** A first draft
        of this test used a gain that left `tx-b` one surviving candidate in the first pass,
        so the fallback never ran and the test passed for the wrong reason — it was caught by
        reverting the fallback and watching nothing fail. `reference_candidate_count` counts
        *attributed* candidates, so zero-with-a-reference is the fallback's signature and is
        what makes the mechanism visible from the public result.
        """
        total = 14 * SECOND
        loud = synth.noise_floor(total, seed=13)
        quiet = synth.noise_floor(total, seed=14)

        candidates: list[CandidateInput] = []
        for index in range(3):
            at = (2 + 4 * index) * SECOND
            speech = voice(50 + index, gain=0.30, n_samples=2 * SECOND)
            answer = voice(60 + index, gain=0.02, n_samples=2 * SECOND)
            loud[at : at + 2 * SECOND] += speech
            quiet[at : at + 2 * SECOND] += answer
            # tx-b's lav carries tx-a as well as its own wearer, which is what makes every one
            # of tx-b's candidates dominated *and* correlated — both numeric conditions for
            # suppression, on every candidate it has.
            quiet[at : at + 2 * SECOND] += synth.bleed_of(
                speech, delay_samples=BLEED_LAG, attenuation_db=18.0
            )[: 2 * SECOND]
            candidates.append(candidate("tx-a", at, at + 2 * SECOND))
            candidates.append(candidate("tx-b", at, at + 2 * SECOND))

        room = Room({"tx-a": loud, "tx-b": quiet})
        found = attribute(candidates, read=room.read, config=settings())
        by_track = {candidates[item.index].track_id: item.decision for item in found.attributions}

        assert found.reference_candidate_counts["tx-b"] == 0, (
            "this fixture is supposed to leave tx-b with no attributed candidates at all — "
            "if it has one, the winners population is non-empty and the fallback under test "
            "never runs"
        )
        assert found.speech_references["tx-b"] is not None, (
            "tx-b won nothing, so the reference must come from the fallback — otherwise the "
            "veto cannot fire and a real speaker is deleted"
        )
        assert by_track["tx-b"] == "retained"


class TestTheMeasurements:
    def test_the_reported_lag_is_the_real_acoustic_delay(self) -> None:
        """3 ms of air, reported on the grid it was measured on."""
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        pairs = compare_pairs(candidates, read=room.read, config=settings())
        assert len(pairs) == 1
        # A pair's lag is stated from its `left` member, which is `tx-a` — the speaker. Its
        # audio arrives *earlier* than the copy `tx-b` heard, so the sign is negative here
        # and positive from `tx-b`'s side. The antisymmetry test below pins the other half.
        assert pairs[0].left == 0
        assert pairs[0].lag_derivative_samples == -BLEED_LAG
        assert pairs[0].correlation_permille >= 900

    def test_a_lag_outside_the_window_is_not_found(self) -> None:
        """The window is a real bound, not decoration.

        A delay beyond `correlation_max_lag_ms` reads as an unrelated signal, which is the
        conservative direction: an unfound relationship keeps both candidates.
        """
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        narrow = ActivityConfig(correlation_max_lag_ms=1)
        pairs = compare_pairs(candidates, read=room.read, config=narrow)
        assert abs(pairs[0].lag_derivative_samples) <= 16
        assert pairs[0].correlation_permille < 900

    def test_the_lag_and_the_level_delta_are_antisymmetric(self) -> None:
        """The two candidates in a pair can never disagree about the pair."""
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        found = attribute(candidates, read=room.read, config=settings())
        first, second = found.attributions[0].evidence[0], found.attributions[1].evidence[0]
        assert first.lag_derivative_samples == -second.lag_derivative_samples
        assert first.level_delta_mb == -second.level_delta_mb
        assert first.correlation_permille == second.correlation_permille
        assert first.score_margin_permille == -second.score_margin_permille

    def test_every_read_is_bounded_by_the_correlation_window(self) -> None:
        """INV-07. A candidate may be minutes long; a read may not.

        Without the cap, one long candidate pulls its whole span into memory — and on a
        four-hour session with six tracks that is the shape of the failure this project's
        UMA host dies from.
        """
        total = 120 * SECOND
        room = Room(
            {
                "tx-a": place(voice(1, gain=0.3, n_samples=100 * SECOND), at=0, total=total),
                "tx-b": place(voice(2, gain=0.3, n_samples=100 * SECOND), at=0, total=total),
            }
        )
        candidates = [candidate("tx-a", 0, 100 * SECOND), candidate("tx-b", 0, 100 * SECOND)]
        config = settings(correlation_window_ms=2000)
        attribute(candidates, read=room.read, config=config)

        cap = 2000 * DERIVATIVE_SAMPLE_RATE // 1000
        assert room.reads, "the gate read nothing, so this proves nothing"
        assert max(length for _, _, length in room.reads) <= cap

    def test_a_capped_window_is_centred_on_the_candidate(self) -> None:
        """A candidate longer than the window is characterized by its middle.

        Taking the first two seconds instead would measure onsets, and every utterance begins
        quietly — so every long candidate would read as quieter than it is.
        """
        total = 60 * SECOND
        room = Room({"tx-a": place(voice(1, gain=0.3, n_samples=40 * SECOND), at=0, total=total)})
        candidates = [candidate("tx-a", 0, 40 * SECOND)]
        measure_levels(candidates, read=room.read, config=settings(correlation_window_ms=2000))

        _, start, length = room.reads[0]
        assert length == 2000 * DERIVATIVE_SAMPLE_RATE // 1000
        assert start == (40 * SECOND - length) // 2


class TestAmbiguity:
    def test_a_candidate_kept_only_by_the_veto_is_marked(self) -> None:
        """The one case where the numbers said bleed and the pipeline overrode them."""
        veto = TestTheVeto()
        room, candidates = veto.room(overlap_gain=0.03)
        _, decided = veto.overlap(room, candidates, settings())
        assert decided.ambiguous is True

    def test_an_ordinary_overlap_is_not_marked_ambiguous(self) -> None:
        """Otherwise the flag means "these candidates overlapped", which is not news."""
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.9, correlated=False)
        found = attribute(candidates, read=room.read, config=settings())
        assert [item.ambiguous for item in found.attributions] == [False, False]

    def test_a_suppressed_candidate_is_never_ambiguous(self) -> None:
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        found = attribute(candidates, read=room.read, config=settings())
        suppressed = [item for item in found.attributions if item.decision == "suppressed"]
        assert suppressed
        assert all(not item.ambiguous for item in suppressed)


class TestDeterminism:
    def test_the_same_audio_decides_the_same_way_twice(self) -> None:
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        first = attribute(candidates, read=room.read, config=settings())
        second = attribute(candidates, read=room.read, config=settings())
        assert first == second

    @pytest.mark.parametrize("order", [(0, 1), (1, 0)])
    def test_the_order_candidates_arrive_in_does_not_change_a_decision(
        self, order: tuple[int, int]
    ) -> None:
        """A pair is measured once and read from both directions, so neither position is
        privileged. If it were, a session's attribution would depend on dictionary order."""
        room, candidates = two_speaker_room(speaker_gain=0.3, listener_gain=0.05, correlated=True)
        reordered = [candidates[index] for index in order]
        found = attribute(reordered, read=room.read, config=settings())
        by_track = {reordered[item.index].track_id: item.decision for item in found.attributions}
        assert by_track == {"tx-a": "retained", "tx-b": "suppressed"}
