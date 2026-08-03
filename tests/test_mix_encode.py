"""The encode/verify/retry loop, and the failure it must produce rather than a claim.

> Bound the retry count, retain all measurements in the report, and **fail the mix stage
> rather than claim compliance** if the decoded MP3 remains outside configured tolerances.

The retry logic runs against a scripted measurer, because the cases that matter — an
overshoot that resolves, an overshoot that never does, a duration that is wrong for reasons a
gain cannot fix — are ones a real encoder produces rarely or never on a six-second tone. The
seam is `measurer`, and it exists for exactly this.

The rest runs through real FFmpeg, because "128 kbps mono MP3 with metadata containing the
session ID/title" is a claim about a file and nothing else can check it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dnd_audio.config import EncodeConfig, MixConfig
from dnd_audio.mix.encode import (
    MP3_FRAME_SAMPLES,
    EncodeError,
    encode_command,
    encode_mp3,
    probe_mp3,
)
from dnd_audio.mix.loudness import Measurement, measure
from dnd_audio.timeline.wavwrite import WavWriter

RATE = 48_000
SECONDS = 6
SAMPLES = SECONDS * RATE


@pytest.fixture
def intermediate(tmp_path: Path) -> Path:
    """A six-second unity-gain mix, written through the project's own streamed writer."""
    rng = np.random.default_rng(20260802)
    samples = (
        0.2 * np.sin(2 * np.pi * 440 * np.arange(SAMPLES) / RATE)
        + 0.01 * rng.standard_normal(SAMPLES)
    ).astype(np.float32)
    path = tmp_path / "mix.wav"
    with WavWriter(path, sample_rate=RATE, n_samples=SAMPLES) as writer:
        for start in range(0, SAMPLES, RATE):
            writer.write(samples[start : start + RATE])
    return path


def a_measurement(**overrides: object) -> Measurement:
    """A comfortably compliant decode. Every case below states only what it changes."""
    base = Measurement(
        integrated_lufs_mb=-1600,
        true_peak_dbtp_mb=-200,
        n_samples=SAMPLES,
        command="ffmpeg ...",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _headroom() -> Measurement:
    """An intermediate quiet enough to normalize and peaky enough not to be ceiling-limited.

    Stated explicitly because the alternative is a default that is *itself* a ceiling-limited
    case, which would silently switch off the loudness assertions in every test below it.
    """
    return a_measurement(integrated_lufs_mb=-1760, true_peak_dbtp_mb=-3000)


class Scripted:
    """A measurer that answers from a list, and records how often it was asked."""

    def __init__(self, *answers: Measurement) -> None:
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, path: Path) -> Measurement:
        self.calls += 1
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]


def _encode(
    intermediate: Path,
    tmp_path: Path,
    measurer: Scripted,
    *,
    settings: MixConfig | None = None,
    source: Measurement | None = None,
) -> object:
    return encode_mp3(
        intermediate,
        tmp_path / "session.mp3",
        settings=settings or MixConfig(),
        session_id="2026-08-15",
        title="Session 01",
        source_measurement=source or _headroom(),
        expected_samples=SAMPLES,
        measurer=measurer,
    )


class TestTheRetryLoop:
    def test_a_compliant_first_encode_is_accepted_without_a_retry(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        measurer = Scripted(a_measurement())
        result = _encode(intermediate, tmp_path, measurer)
        assert measurer.calls == 1
        assert len(result.attempts) == 1  # type: ignore[attr-defined]
        assert result.accepted.compliant  # type: ignore[attr-defined]

    def test_a_true_peak_overshoot_is_retried_at_a_lower_gain_and_resolves(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The spec's "reduce the pre-encode gain ... and re-encode from the lossless
        intermediate". The second attempt must actually be quieter, not merely a second try."""
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=50), a_measurement())
        result = _encode(intermediate, tmp_path, measurer)

        attempts = result.attempts  # type: ignore[attr-defined]
        assert len(attempts) == 2
        assert attempts[0].failures == ("true_peak",)
        assert attempts[1].gain_mb < attempts[0].gain_mb
        assert attempts[1].compliant

    def test_the_reduction_is_the_overshoot_it_measured(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """Walking down by a fixed step would burn the budget on a large overshoot and
        overshoot the correction on a small one."""
        settings = MixConfig()
        ceiling_mb = round(settings.true_peak_dbtp * 100)
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=ceiling_mb + 250), a_measurement())
        result = _encode(intermediate, tmp_path, measurer, settings=settings)

        attempts = result.attempts  # type: ignore[attr-defined]
        assert attempts[0].gain_mb - attempts[1].gain_mb == 250

    def test_an_attempt_exactly_on_the_ceiling_still_moves(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """A reduction of zero would re-encode an identical file until the budget ran out."""
        settings = MixConfig()
        ceiling_mb = round(settings.true_peak_dbtp * 100)
        tolerance_mb = round(settings.encode.true_peak_tolerance_db * 100)
        measurer = Scripted(
            a_measurement(true_peak_dbtp_mb=ceiling_mb + tolerance_mb + 1), a_measurement()
        )
        result = _encode(intermediate, tmp_path, measurer, settings=settings)
        attempts = result.attempts  # type: ignore[attr-defined]
        assert attempts[1].gain_mb < attempts[0].gain_mb

    def test_an_overshoot_that_never_resolves_exhausts_the_budget_and_fails(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The clause the whole loop exists for: fail rather than claim compliance."""
        settings = MixConfig(encode=EncodeConfig(max_retries=2))
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=50))
        with pytest.raises(EncodeError, match="after 3 attempt"):
            _encode(intermediate, tmp_path, measurer, settings=settings)
        assert measurer.calls == 3

    def test_the_failure_names_every_attempt_and_every_number(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """ "Retain all measurements" — a compliance claim nobody can audit is worth less than
        a failure that says what it saw."""
        settings = MixConfig(encode=EncodeConfig(max_retries=1))
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=50))
        with pytest.raises(EncodeError) as raised:
            _encode(intermediate, tmp_path, measurer, settings=settings)

        message = str(raised.value)
        assert "attempt 0" in message
        assert "attempt 1" in message
        assert "0.50 dBTP" in message
        assert f"{SAMPLES} samples decoded" in message
        assert "true_peak" in message

    def test_a_retry_budget_of_zero_means_one_attempt(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        settings = MixConfig(encode=EncodeConfig(max_retries=0))
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=50))
        with pytest.raises(EncodeError, match="after 1 attempt"):
            _encode(intermediate, tmp_path, measurer, settings=settings)
        assert measurer.calls == 1

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"integrated_lufs_mb": -1900}, "integrated_loudness"),
            ({"n_samples": SAMPLES + 5 * MP3_FRAME_SAMPLES}, "duration"),
        ],
    )
    def test_a_failure_a_gain_cannot_fix_is_not_retried(
        self, intermediate: Path, tmp_path: Path, overrides: dict[str, object], expected: str
    ) -> None:
        """Walking the gain would be guessing. A duration mismatch is not a level problem, and
        a loudness miss with no overshoot means the target itself is out of reach."""
        measurer = Scripted(a_measurement(**overrides))
        with pytest.raises(EncodeError, match=expected):
            _encode(intermediate, tmp_path, measurer)
        assert measurer.calls == 1


class TestTheTolerances:
    @pytest.mark.parametrize("offset_mb", [-100, 0, 100])
    def test_loudness_within_the_configured_tolerance_is_accepted(
        self, intermediate: Path, tmp_path: Path, offset_mb: int
    ) -> None:
        settings = MixConfig()
        target_mb = round(settings.integrated_lufs * 100)
        measurer = Scripted(a_measurement(integrated_lufs_mb=target_mb + offset_mb))
        assert _encode(intermediate, tmp_path, measurer, settings=settings)

    def test_loudness_one_millibel_outside_it_is_not(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The boundary, from the failing side. A tolerance nothing is ever outside is not a
        tolerance."""
        settings = MixConfig(encode=EncodeConfig(max_retries=0))
        target_mb = round(settings.integrated_lufs * 100)
        tolerance_mb = round(settings.encode.loudness_tolerance_lu * 100)
        measurer = Scripted(a_measurement(integrated_lufs_mb=target_mb + tolerance_mb + 1))
        with pytest.raises(EncodeError, match="integrated_loudness"):
            _encode(intermediate, tmp_path, measurer, settings=settings)

    def test_a_true_peak_inside_the_documented_measurement_tolerance_is_accepted(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The spec's "does not exceed the true-peak target beyond a documented measurement
        tolerance" (OQ-020)."""
        settings = MixConfig()
        ceiling_mb = round(settings.true_peak_dbtp * 100)
        tolerance_mb = round(settings.encode.true_peak_tolerance_db * 100)
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=ceiling_mb + tolerance_mb))
        assert _encode(intermediate, tmp_path, measurer, settings=settings)

    def test_a_duration_within_the_configured_frames_is_accepted(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        settings = MixConfig()
        allowed = settings.encode.duration_tolerance_frames * MP3_FRAME_SAMPLES
        measurer = Scripted(a_measurement(n_samples=SAMPLES + allowed))
        assert _encode(intermediate, tmp_path, measurer, settings=settings)

    def test_a_duration_one_sample_further_out_is_not(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        settings = MixConfig()
        allowed = settings.encode.duration_tolerance_frames * MP3_FRAME_SAMPLES
        measurer = Scripted(a_measurement(n_samples=SAMPLES - allowed - 1))
        with pytest.raises(EncodeError, match="duration"):
            _encode(intermediate, tmp_path, measurer, settings=settings)


class TestTheMasterGain:
    def test_the_first_gain_moves_the_measured_loudness_onto_the_target(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """Two-pass loudness: the intermediate is measured, and the difference is the gain."""
        measurer = Scripted(a_measurement())
        result = _encode(
            intermediate,
            tmp_path,
            measurer,
            source=a_measurement(integrated_lufs_mb=-2000, true_peak_dbtp_mb=-2000),
        )
        assert result.attempts[0].gain_mb == 400  # type: ignore[attr-defined]

    def test_the_first_gain_already_aims_at_the_true_peak_ceiling(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """ "Reduce the pre-encode gain or true-peak target" — applied before the first encode
        rather than after the first failure, so the ordinary case needs no retry (OQ-020)."""
        measurer = Scripted(a_measurement())
        result = _encode(
            intermediate,
            tmp_path,
            measurer,
            source=a_measurement(integrated_lufs_mb=-3000, true_peak_dbtp_mb=-300),
        )
        # Loudness alone wants +14 dB; the peak allows only 1.5 dB of headroom.
        assert result.attempts[0].gain_mb == 150  # type: ignore[attr-defined]

    def test_a_mix_below_the_silence_floor_is_not_normalized_and_says_so(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """Not hypothetical. A session where the detector found no speech has every track at
        the room-tone share, and normalizing that to -16 LUFS is fifty decibels of amplified
        noise floor (OQ-019). FFmpeg reports -70 LUFS for silence, not -inf, so the guard is
        a threshold rather than a null check.
        """
        measurer = Scripted(a_measurement(integrated_lufs_mb=-7000))
        settings = MixConfig(
            encode=EncodeConfig(loudness_tolerance_lu=1.0, silence_floor_lufs=-50.0)
        )
        result = _encode(
            intermediate,
            tmp_path,
            measurer,
            settings=settings,
            source=a_measurement(integrated_lufs_mb=-7000, true_peak_dbtp_mb=-7000),
        )
        assert result.attempts[0].gain_mb == 0  # type: ignore[attr-defined]
        assert [note.code for note in result.warnings] == ["mix_not_normalized"]  # type: ignore[attr-defined]

    def test_a_silent_mix_is_not_then_failed_for_being_silent(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The trap in the guard above: declining to normalize and then failing the loudness
        check would make every silent session a failed stage rather than a warned one."""
        measurer = Scripted(a_measurement(integrated_lufs_mb=-7000))
        result = _encode(
            intermediate,
            tmp_path,
            measurer,
            source=a_measurement(integrated_lufs_mb=-7000, true_peak_dbtp_mb=-7000),
        )
        assert result.accepted.compliant  # type: ignore[attr-defined]

    def test_a_target_the_true_peak_ceiling_forbids_is_warned_about_not_failed(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The case the canonical fixture actually produces, and it must not fail the stage.

        Peaky material 31 dB down wants +15.6 dB and the ceiling allows +1.6. The ceiling is a
        hard limit on clipping; the loudness figure is a target. Honouring the first and
        warning about the second is the only reading that does not throw away a good mix and
        make `process` exit nonzero on a session that produced exactly the MP3 it should have.
        """
        settings = MixConfig()
        ceiling_mb = round(settings.true_peak_dbtp * 100)
        measurer = Scripted(a_measurement(integrated_lufs_mb=-3000, true_peak_dbtp_mb=ceiling_mb))
        result = _encode(
            intermediate,
            tmp_path,
            measurer,
            settings=settings,
            source=a_measurement(integrated_lufs_mb=-3160, true_peak_dbtp_mb=-310),
        )
        assert result.attempts[0].gain_mb == 160  # type: ignore[attr-defined]
        assert result.accepted.compliant  # type: ignore[attr-defined]
        assert [n.code for n in result.warnings] == ["mix_loudness_target_unreachable"]  # type: ignore[attr-defined]
        assert "14.0 LU quieter" in result.warnings[0].message  # type: ignore[attr-defined]

    def test_the_ceiling_is_still_enforced_on_the_decode(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """Declining to chase the loudness target does not suspend the true-peak check: that
        one is a claim about the file rather than about a target."""
        settings = MixConfig(encode=EncodeConfig(max_retries=0))
        measurer = Scripted(a_measurement(integrated_lufs_mb=-3000, true_peak_dbtp_mb=500))
        with pytest.raises(EncodeError, match="true_peak"):
            _encode(
                intermediate,
                tmp_path,
                measurer,
                settings=settings,
                source=a_measurement(integrated_lufs_mb=-3160, true_peak_dbtp_mb=-310),
            )

    def test_a_gain_beyond_the_clamp_is_clamped_and_says_so(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        settings = MixConfig(encode=EncodeConfig(max_master_gain_db=5.0, silence_floor_lufs=-70.0))
        measurer = Scripted(a_measurement())
        result = _encode(
            intermediate,
            tmp_path,
            measurer,
            settings=settings,
            source=a_measurement(integrated_lufs_mb=-4000, true_peak_dbtp_mb=-4000),
        )
        assert result.attempts[0].gain_mb == 500  # type: ignore[attr-defined]
        assert [note.code for note in result.warnings] == ["mix_master_gain_clamped"]  # type: ignore[attr-defined]


class TestTheRealEncode:
    """Through FFmpeg, because "128 kbps mono MP3 with metadata" is a claim about a file."""

    def test_it_produces_a_mono_mp3_at_the_configured_bitrate(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        source = measure(intermediate)
        result = encode_mp3(
            intermediate,
            tmp_path / "session.mp3",
            settings=MixConfig(),
            session_id="2026-08-15",
            title="Session 01",
            source_measurement=source,
            expected_samples=SAMPLES,
        )
        assert result.facts.codec == "mp3"
        assert result.facts.channels == 1
        assert result.facts.sample_rate == RATE
        assert result.facts.bit_rate_kbps == 128

    def test_the_session_id_and_title_are_in_the_tags(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The spec asks for "metadata containing the session ID/title" by name."""
        source = measure(intermediate)
        result = encode_mp3(
            intermediate,
            tmp_path / "session.mp3",
            settings=MixConfig(),
            session_id="2026-08-15",
            title="Session 01",
            source_measurement=source,
            expected_samples=SAMPLES,
        )
        assert result.facts.tags["title"] == "Session 01"
        assert result.facts.tags["album"] == "2026-08-15"

    def test_the_decoded_file_meets_every_configured_target(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """The gate's acceptance criterion 8, through real encoding and real decoding: within
        1 LU of the target, under the ceiling, and within one MP3 frame of the length."""
        settings = MixConfig()
        source = measure(intermediate)
        result = encode_mp3(
            intermediate,
            tmp_path / "session.mp3",
            settings=settings,
            session_id="2026-08-15",
            title="Session 01",
            source_measurement=source,
            expected_samples=SAMPLES,
        )
        decoded = result.accepted.measurement
        assert decoded.integrated_lufs_mb is not None
        assert abs(decoded.integrated_lufs_mb - round(settings.integrated_lufs * 100)) <= round(
            settings.encode.loudness_tolerance_lu * 100
        )
        assert decoded.true_peak_dbtp_mb is not None
        assert decoded.true_peak_dbtp_mb <= round(settings.true_peak_dbtp * 100) + round(
            settings.encode.true_peak_tolerance_db * 100
        )
        assert abs(decoded.n_samples - SAMPLES) <= (
            settings.encode.duration_tolerance_frames * MP3_FRAME_SAMPLES
        )

    def test_a_different_loudness_target_lands_somewhere_different(
        self, intermediate: Path, tmp_path: Path
    ) -> None:
        """So the target is read rather than baked. Without this the loudness assertion above
        passes for a mixer that always produced -16 LUFS by coincidence of its inputs."""
        source = measure(intermediate)
        found = []
        for target in (-16.0, -23.0):
            result = encode_mp3(
                intermediate,
                tmp_path / f"session{target}.mp3",
                settings=MixConfig(integrated_lufs=target),
                session_id="2026-08-15",
                title="Session 01",
                source_measurement=source,
                expected_samples=SAMPLES,
            )
            measured = result.accepted.measurement.integrated_lufs_mb
            assert measured is not None
            found.append(measured)
        assert found[0] - found[1] == pytest.approx(700, abs=150)

    def test_encoding_a_file_that_is_not_there_fails_and_leaves_nothing_behind(
        self, tmp_path: Path
    ) -> None:
        destination = tmp_path / "session.mp3"
        with pytest.raises(EncodeError, match="ffmpeg exited"):
            encode_mp3(
                tmp_path / "absent.wav",
                destination,
                settings=MixConfig(),
                session_id="2026-08-15",
                title="Session 01",
                source_measurement=a_measurement(),
                expected_samples=SAMPLES,
            )
        assert not destination.exists()

    def test_probing_something_that_is_not_audio_fails_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "not-audio.mp3"
        path.write_bytes(b"this is not an mp3")
        with pytest.raises(EncodeError):
            probe_mp3(path)


class TestTheCommand:
    def test_the_gain_is_an_encode_parameter_rather_than_part_of_the_mix(
        self, tmp_path: Path
    ) -> None:
        """ADR-0023: a true-peak retry then costs one encode rather than one re-mix of six
        four-hour tracks, and changing the loudness target reuses the intermediate."""
        command = " ".join(
            encode_command(
                tmp_path / "mix.wav",
                tmp_path / "session.mp3",
                settings=MixConfig(),
                gain_mb=-250,
                session_id="2026-08-15",
                title="Session 01",
            )
        )
        assert "volume=-2.50dB" in command
        assert "libmp3lame" in command
        assert "-b:a 128k" in command
        assert "-ac 1" in command

    def test_nothing_from_the_intermediate_leaks_into_the_deliverable(self, tmp_path: Path) -> None:
        """`-map_metadata -1`: the MP3's tags are the ones this project put there."""
        command = encode_command(
            tmp_path / "mix.wav",
            tmp_path / "session.mp3",
            settings=MixConfig(),
            gain_mb=0,
            session_id="2026-08-15",
            title="Session 01",
        )
        assert command[command.index("-map_metadata") + 1] == "-1"

    def test_every_attempts_command_is_recorded(self, intermediate: Path, tmp_path: Path) -> None:
        """The spec asks the report for "the exact commands/parameters used for FFmpeg
        outputs", and a retry is a different command."""
        measurer = Scripted(a_measurement(true_peak_dbtp_mb=50), a_measurement())
        result = _encode(intermediate, tmp_path, measurer)
        commands = result.commands  # type: ignore[attr-defined]
        assert len(commands) == 2
        assert commands[0] != commands[1]
