"""The affine time-warp seam: unused in the MVP, and able to fire.

The spec asks for a hook and explicitly forbids depending on it. Both halves matter, and
the second is the one that rots: a protocol nothing calls is decoration, and M0's closeout
records what happens to rails that cannot fail — two of its own tests short-circuited before
reaching the code they claimed to exercise, and independent review is what caught it.

So there is a test-local affine implementation here, and the assertion is that supplying it
*moves the timeline*. If someone later "simplifies" `determine_origin` by dropping the warp
parameter, this fails immediately rather than in whichever milestone finally needs drift
correction.

The MVP does not correct drift because how far the three kits' clocks actually diverge is
**OQ-006** — unmeasured until H2 — and correcting by an unmeasured amount is inventing
timing (INV-12).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from dnd_audio.artifacts.manifest import Manifest
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.origin import determine_origin
from dnd_audio.timeline.warp import IdentityWarp, TimeWarp
from tests.manifests import bwf, config, manifest, source

RATE = CANONICAL_SAMPLE_RATE
HOUR = 3600 * RATE


@dataclass(frozen=True)
class AffineWarp:
    """What a real drift correction would look like: a rate scaling per track.

    Deliberately lives in the tests rather than in `src`. Shipping an unused
    implementation would be a placeholder, and the seam's job is to make the future change
    small — not to pre-write it against a measurement nobody has taken.
    """

    scale: Fraction
    only: str | None = None

    def warp(self, track_id: str, elapsed_seconds: Fraction) -> Fraction:
        if self.only is not None and track_id != self.only:
            return elapsed_seconds
        return elapsed_seconds * self.scale


def two_tracks() -> Manifest:
    return manifest(
        {
            "tx-a": [source("raw/tx-a/one.wav", bwf(19 * HOUR))],
            "tx-b": [source("raw/tx-b/one.wav", bwf(19 * HOUR + 100 * RATE))],
        }
    )


def placements(warp: TimeWarp | None) -> dict[str, int]:
    found = determine_origin(
        two_tracks(),
        config(origin_date="2026-08-15", origin_timecode="19:00:00:00"),
        warp=warp,
    )
    return {start.relative_path: start.session_start_sample for start in found.starts}


class TestTheSeamIsReal:
    def test_a_non_identity_warp_moves_the_timeline(self) -> None:
        """The assertion that keeps the hook from being decoration.

        A one-part-per-thousand scaling — about what a badly mismatched pair of clocks
        would do over a long session — moves a source 100 seconds in by 100 ms.
        """
        unwarped = placements(None)
        assert unwarped["raw/tx-b/one.wav"] == 100 * RATE

        stretched = placements(AffineWarp(scale=Fraction(1001, 1000)))
        assert stretched["raw/tx-b/one.wav"] == 100 * RATE + RATE // 10
        assert stretched != unwarped

    def test_it_applies_per_track(self) -> None:
        """Drift is a property of a kit's clock, so the seam is keyed by track."""
        only_b = placements(AffineWarp(scale=Fraction(2), only="tx-b"))
        assert only_b["raw/tx-a/one.wav"] == 0
        assert only_b["raw/tx-b/one.wav"] == 200 * RATE

    def test_it_operates_before_the_quantization(self) -> None:
        """A scaling applied after rounding would accumulate the error it removes.

        Half a sample of elapsed time, scaled by three, must land on two samples — which
        it only can if the multiplication happened in exact rational arithmetic. Applied
        after rounding, 0.5 becomes 1 and then 3.
        """
        found = determine_origin(
            manifest(
                {
                    "tx-a": [source("raw/tx-a/one.wav", bwf(0))],
                    "tx-b": [source("raw/tx-b/one.wav", bwf(1))],
                }
            ),
            config(origin_date="2026-08-15", origin_timecode="00:00:00:00"),
            warp=AffineWarp(scale=Fraction(5, 2)),
        )
        placed = {start.relative_path: start.session_start_sample for start in found.starts}
        # One sample, scaled by 5/2, is 2.5 — which rounds away from zero to 3.
        assert placed["raw/tx-b/one.wav"] == 3

    def test_the_default_changes_nothing(self) -> None:
        assert placements(None) == placements(IdentityWarp())


class TestIdentity:
    def test_it_returns_its_input_exactly(self) -> None:
        warp = IdentityWarp()
        for value in (Fraction(0), Fraction(1, 3), Fraction(-7, 11), Fraction(86400)):
            assert warp.warp("tx-a", value) == value

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(IdentityWarp(), TimeWarp)
        assert isinstance(AffineWarp(scale=Fraction(1)), TimeWarp)

    def test_it_is_a_class_rather_than_none(self) -> None:
        """A `None` default would put an untaken branch in the placement path.

        The branch that never runs in production is the branch that stops working, so the
        MVP's "no correction" is an implementation rather than a special case.
        """
        assert IdentityWarp().warp("tx-a", Fraction(3)) == Fraction(3)


class TestTheMvpDoesNotUseIt:
    def test_ingest_places_nothing_differently(self, canonical_fixture: FixtureTruth) -> None:
        """The other half of the spec's sentence: a hook, not a dependency."""
        from dnd_audio.timeline.runner import run_ingest

        result = run_ingest(canonical_fixture.session_dir)
        assert result.timeline is not None
        declared = {chunk.relative_path: chunk.start_sample for chunk in canonical_fixture.chunks}
        placed = {
            segment.source_relative_path: segment.session_start_sample
            for track in result.timeline.tracks
            for segment in track.segments
            if segment.kind == "audio"
        }
        assert placed == declared

    def test_no_drift_correction_is_configurable(self) -> None:
        """There is no session.yaml field that turns one on, and that is deliberate.

        Adding one before H2 has measured anything would let an operator correct by a
        number nobody has evidence for, which is what INV-12 forbids.
        """
        from dnd_audio.config import SessionConfig

        fields = set(SessionConfig.model_fields) | set(
            SessionConfig.model_fields["timecode"].annotation.model_fields  # type: ignore[union-attr]
        )
        assert not [name for name in fields if "warp" in name or "drift_correct" in name]
