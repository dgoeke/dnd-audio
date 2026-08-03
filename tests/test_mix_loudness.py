"""Measuring a file, and the parse that would otherwise break silently on an upgrade.

Two halves, tested two ways on purpose.

**The parse** runs against captured `ebur128` output, including output this project's own
FFmpeg has never produced — a `-inf` summary, a summary with the true-peak line missing
because `peak=true` was not passed, and no summary at all. Driving it only through a live
FFmpeg would test one version's formatting and call it a contract.

**The measurement** runs against real files, because the property that matters is that the
sample count is the *decoded* one. A container duration cannot disagree with a decode in a
test that reads the container.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dnd_audio.mix.loudness import (
    BYTES_PER_SAMPLE,
    LoudnessError,
    ffmpeg_version,
    measure,
    measure_command,
    parse_summary,
)
from dnd_audio.timeline.wavwrite import WavWriter

RATE = 48_000

#: Captured from FFmpeg 8.0 on a 1 kHz tone. Kept verbatim rather than reduced to the two
#: lines that are parsed, because something tidier would not catch a change in the block's
#: shape — which is the failure this is here for.
SUMMARY = """[Parsed_ebur128_0 @ 0x706888003680] Summary:

  Integrated loudness:
    I:         -21.1 LUFS
    Threshold: -31.1 LUFS

  Loudness range:
    LRA:         0.0 LU
    Threshold: -41.1 LUFS
    LRA low:   -21.1 LUFS
    LRA high:  -21.1 LUFS

  True peak:
    Peak:      -18.1 dBFS
"""


def _tone(path: Path, *, n_samples: int, amplitude: float = 0.2, rate: int = RATE) -> None:
    """A 1 kHz tone with a little noise, written through the project's own streamed writer."""
    rng = np.random.default_rng(20260802)
    samples = (
        amplitude * np.sin(2 * np.pi * 1000 * np.arange(n_samples) / rate)
        + 0.001 * rng.standard_normal(n_samples)
    ).astype(np.float32)
    with WavWriter(path, sample_rate=rate, n_samples=n_samples) as writer:
        for start in range(0, n_samples, rate):
            writer.write(samples[start : start + rate])


class TestTheSummaryParse:
    def test_it_reads_both_numbers_as_millibels(self) -> None:
        """Millibels, so a threshold comparison is integer arithmetic rather than a float
        that has been through a text round trip."""
        assert parse_summary(SUMMARY) == (-2110, -1810)

    def test_minus_infinity_becomes_none_rather_than_a_very_small_number(self) -> None:
        """What FFmpeg prints for digital silence. Turning it into -70 or -999 would make a
        silent mix look merely quiet to every threshold downstream — and the normalizer would
        then apply the clamp instead of declining to normalize."""
        silent = SUMMARY.replace("I:         -21.1 LUFS", "I:          -inf LUFS")
        assert parse_summary(silent) == (None, -1810)

    def test_a_missing_true_peak_line_is_none_rather_than_an_error(self) -> None:
        """`peak=true` produces it and nothing else does. Its absence means the filter ran
        without peak mode, which is a measurement of loudness and not of peak."""
        without = SUMMARY.split("  True peak:")[0]
        assert parse_summary(without) == (-2110, None)

    def test_no_summary_at_all_is_an_error_rather_than_a_measurement(self) -> None:
        with pytest.raises(LoudnessError, match="no integrated-loudness line"):
            parse_summary("ffmpeg version 8.0\nsome unrelated warning\n")

    def test_the_error_quotes_what_ffmpeg_actually_said(self) -> None:
        """An operator debugging a format change needs the output, not a summary of it."""
        with pytest.raises(LoudnessError, match="Unknown filter"):
            parse_summary("No such filter: 'ebur128x'\nUnknown filter\n")

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [("-16.0", -1600), ("0.0", 0), ("-0.5", -50), ("-70.5", -7050), ("3.2", 320)],
    )
    def test_a_decimal_becomes_hundredths_exactly(self, printed: str, expected: int) -> None:
        text = SUMMARY.replace("-21.1 LUFS", f"{printed} LUFS")
        assert parse_summary(text)[0] == expected

    def test_a_value_with_no_decimal_point_still_parses(self) -> None:
        """FFmpeg prints one decimal today. A build that printed an integer would otherwise
        take this path from "a measurement" to "no summary at all"."""
        assert parse_summary(SUMMARY.replace("-21.1 LUFS", "-21 LUFS"))[0] == -2100


class TestMeasuringARealFile:
    def test_the_sample_count_is_the_decoded_one(self, tmp_path: Path) -> None:
        """The gate says the MP3 is *decoded* and measured. This is the number that claim
        rests on, and it comes from counting bytes off the decoder rather than from a header
        that could describe a file the decoder could not finish."""
        path = tmp_path / "tone.wav"
        _tone(path, n_samples=3 * RATE)
        found = measure(path)
        assert found.n_samples == 3 * RATE

    def test_it_measures_a_known_signal_in_the_right_direction(self, tmp_path: Path) -> None:
        """Not against a hardcoded LUFS figure — that would pin one FFmpeg's arithmetic.
        Halving the amplitude must move the measurement by very close to 6 dB.
        """
        loud, quiet = tmp_path / "loud.wav", tmp_path / "quiet.wav"
        _tone(loud, n_samples=3 * RATE, amplitude=0.4)
        _tone(quiet, n_samples=3 * RATE, amplitude=0.2)

        louder = measure(loud).integrated_lufs_mb
        quieter = measure(quiet).integrated_lufs_mb
        assert louder is not None
        assert quieter is not None
        assert louder - quieter == pytest.approx(600, abs=20)

    def test_digital_silence_measures_at_ffmpegs_floor_rather_than_minus_infinity(
        self, tmp_path: Path
    ) -> None:
        """Measured, not assumed — and the answer is not what the parse's `-inf` branch
        suggests.

        FFmpeg 8.0's `ebur128` reports **-70.0 LUFS** for a file of digital silence, which is
        its gating floor, rather than `-inf`. That is why the normalizer's guard is a
        *threshold* (`encode.silence_floor_lufs`) and not an `is None` check: a guard written
        against `-inf` would never fire on real silence, and the mix of a session nobody spoke
        in would be lifted by the full master-gain clamp. The `-inf` branch stays because a
        different build may still produce it, and it costs one comparison.
        """
        path = tmp_path / "silence.wav"
        with WavWriter(path, sample_rate=RATE, n_samples=RATE) as writer:
            writer.write(np.zeros(RATE, dtype=np.float32))
        found = measure(path)
        assert found.integrated_lufs_mb == -7000
        assert not found.silent
        assert found.n_samples == RATE

    def test_a_true_peak_is_reported_and_is_near_the_sample_peak(self, tmp_path: Path) -> None:
        """True peak is measured on an oversampled signal, so it sits at or a little above
        the sample peak — never below it, which is the direction that matters for a ceiling."""
        path = tmp_path / "tone.wav"
        _tone(path, n_samples=RATE, amplitude=0.5)
        found = measure(path)
        assert found.true_peak_dbtp_mb is not None
        sample_peak_mb = round(20 * np.log10(0.5) * 100)
        assert found.true_peak_dbtp_mb >= sample_peak_mb - 20

    def test_a_missing_file_fails_rather_than_returning_a_measurement(self, tmp_path: Path) -> None:
        with pytest.raises(LoudnessError, match="ffmpeg exited"):
            measure(tmp_path / "absent.wav")

    def test_the_command_is_recorded_verbatim(self, tmp_path: Path) -> None:
        """The spec asks the report for "the exact commands/parameters used for FFmpeg
        outputs", so the string the run used is the string the report gets."""
        path = tmp_path / "tone.wav"
        _tone(path, n_samples=RATE)
        assert measure(path).command == " ".join(measure_command(path))

    def test_the_command_asks_for_true_peak_and_a_quiet_frame_log(self, tmp_path: Path) -> None:
        """`peak=true` is what produces the ceiling measurement at all, and `framelog=quiet`
        is what keeps a four-hour measurement from printing 144 000 lines into a pipe its
        caller is not draining."""
        command = " ".join(measure_command(tmp_path / "x.wav"))
        assert "ebur128=peak=true:framelog=quiet" in command
        assert "-f f32le" in command

    def test_a_long_measurement_does_not_deadlock_on_its_own_stderr(self, tmp_path: Path) -> None:
        """The failure `framelog=quiet` and the stderr temp file both exist to prevent.

        A minute of audio would print 600 frame lines without it; four hours would print
        144 000 and fill the pipe while this side is reading stdout. A minute is enough to
        prove the plumbing rather than the arithmetic.
        """
        path = tmp_path / "long.wav"
        _tone(path, n_samples=60 * RATE)
        assert measure(path).n_samples == 60 * RATE


class TestTheMeasurementIsBounded:
    def test_a_decode_is_counted_rather_than_collected(self, tmp_path: Path) -> None:
        """INV-07: the whole point of counting bytes off the pipe is that a four-hour decode
        never becomes an array. Asserted by watching how much is read at a time.
        """
        path = tmp_path / "tone.wav"
        _tone(path, n_samples=10 * RATE)
        sizes: list[int] = []
        original = subprocess.Popen

        def watched(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            process: subprocess.Popen[bytes] = original(*args, **kwargs)
            stream = process.stdout
            assert stream is not None
            read = stream.read

            def watched_read(size: int = -1) -> bytes:
                block: bytes = read(size)
                sizes.append(len(block))
                return block

            stream.read = watched_read  # type: ignore[method-assign]
            return process

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(subprocess, "Popen", watched)
            found = measure(path)

        assert found.n_samples == 10 * RATE
        total = 10 * RATE * BYTES_PER_SAMPLE
        assert max(sizes) < total
        assert sum(sizes) == total


class TestTheVersionReachesProvenance:
    def test_it_reports_a_real_version_string(self) -> None:
        """INV-08 and the spec both want the tool version recorded, because this milestone's
        measurements are FFmpeg's arithmetic rather than ours (ADR-0023)."""
        version = ffmpeg_version()
        assert version.startswith("ffmpeg version")
        assert version != "unknown"
