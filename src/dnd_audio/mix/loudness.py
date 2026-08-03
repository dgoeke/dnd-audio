"""Measuring a file: integrated loudness, true peak, and how many samples it really has.

ADR-0023 records why this is FFmpeg's job rather than a BS.1770 implementation of our own,
and why that is not a reversal of ADR-0011's rejection of FFmpeg for the canonical derivative:
a measurement is read, acted on once, and recorded beside the tool version that produced it,
where a cached artifact's identity must not move when a tool is upgraded for unrelated
reasons.

**One decode serves everything.** `ffmpeg -af ebur128=peak=true -f f32le -` puts the R128
summary on stderr and the decoded samples on stdout. The samples are counted in bounded chunks
and thrown away (INV-07), and that exact integer count is what the duration tolerance is
applied to. Taking the duration from `ffprobe` instead was this milestone's first plan and is
wrong for the reason its review gave: a container or header duration can stay entirely
plausible while decoding yields fewer samples, and the gate says *decoded*.

**`framelog=quiet` matters.** Without it `ebur128` prints a line per 100 ms — 144 000 lines
for a four-hour session — and a subprocess whose stderr pipe fills while its caller is reading
stdout deadlocks. stderr goes to a temporary file anyway, which is belt and braces, but the
option is what keeps the output small enough to read at all.

**A nonzero exit is a failure, not a measurement.** A decode that stopped halfway produces a
perfectly parseable summary of the part it managed, and a compliance check run on half a file
is worse than no check.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dnd_audio.errors import DndAudioError

__all__ = [
    "BYTES_PER_SAMPLE",
    "LoudnessError",
    "Measurement",
    "ffmpeg_version",
    "measure",
    "measure_command",
    "parse_summary",
]

#: The decode format this reads: 32-bit float, one channel.
BYTES_PER_SAMPLE: Final = 4

#: `I:  -21.1 LUFS` inside the Summary block.
_INTEGRATED: Final = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?|-inf)\s*LUFS\s*$", re.MULTILINE)

#: `Peak:  -18.1 dBFS`, the true-peak line, which only appears with `peak=true`.
_PEAK: Final = re.compile(r"^\s*Peak:\s*(-?\d+(?:\.\d+)?|-inf)\s*dBFS\s*$", re.MULTILINE)

#: How much decoded audio to pull from the pipe at a time. Bounded, because the whole point
#: of counting rather than reading is that a four-hour decode never becomes an array.
_READ_CHUNK: Final = 1 << 20


class LoudnessError(DndAudioError):
    """FFmpeg could not measure the file, or said something this cannot read."""

    default_code = "loudness_measurement_failed"


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one decode-and-measure pass found.

    Loudness and peak are kept in **millibels** — decibels scaled by a hundred, the unit the
    activity graph already uses — so a comparison against a configured threshold is integer
    arithmetic rather than a float that has been through a text round trip.
    """

    #: Integrated loudness, in hundredths of a LU relative to full scale.
    integrated_lufs_mb: int | None
    #: True peak, in hundredths of a dB relative to full scale.
    true_peak_dbtp_mb: int | None
    #: Samples the decoder actually produced. Not a container field.
    n_samples: int
    #: The exact command, for the report. The spec asks for FFmpeg parameters by name.
    command: str

    @property
    def silent(self) -> bool:
        """Whether FFmpeg reported no measurable loudness at all.

        **Rarely true, and not the test for silence.** FFmpeg 8.0 reports `-70.0 LUFS` — its
        gating floor — for a file of digital silence, not `-inf`, which is measured in
        `tests/test_mix_loudness.py`. So the normalizer's guard is a *threshold*
        (`encode.silence_floor_lufs`); a guard written against this property would never fire
        on real silence and would lift a session nobody spoke in by the full clamp. This
        exists for a build that does print `-inf`, and costs one comparison.
        """
        return self.integrated_lufs_mb is None


def measure_command(path: Path) -> list[str]:
    """The invocation, built in one place so the report and the run cannot disagree."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        "ebur128=peak=true:framelog=quiet",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-",
    ]


def measure(path: Path) -> Measurement:
    """Decode ``path`` once, measuring loudness and true peak and counting its samples.

    Raises:
        LoudnessError: if FFmpeg is missing, exits nonzero, produces a partial sample, or
            prints a summary without an integrated-loudness line. Each of those would
            otherwise become a compliance claim about something nobody measured.
    """
    command = measure_command(path)
    printable = " ".join(command)
    decoded = 0
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        except OSError as exc:
            message = f"cannot run ffmpeg to measure {path.name}: {exc}"
            raise LoudnessError(message, code="ffmpeg_unavailable") from exc

        assert process.stdout is not None
        with process.stdout as stream:
            while block := stream.read(_READ_CHUNK):
                decoded += len(block)
        status = process.wait()

        errors.seek(0)
        text = errors.read().decode("utf-8", errors="replace")

    if status != 0:
        message = (
            f"ffmpeg exited {status} while measuring {path.name}. A summary of the part it "
            f"managed to decode is not a measurement of the file.\n{text.strip()[-2000:]}"
        )
        raise LoudnessError(message, code="ffmpeg_failed")
    if decoded % BYTES_PER_SAMPLE:
        message = (
            f"ffmpeg produced {decoded} bytes decoding {path.name}, which is not a whole "
            f"number of float32 samples. The decode was truncated."
        )
        raise LoudnessError(message, code="decode_truncated")

    integrated, peak = parse_summary(text, source=path.name)
    return Measurement(
        integrated_lufs_mb=integrated,
        true_peak_dbtp_mb=peak,
        n_samples=decoded // BYTES_PER_SAMPLE,
        command=printable,
    )


def parse_summary(text: str, *, source: str = "the input") -> tuple[int | None, int | None]:
    """Pull integrated loudness and true peak out of `ebur128`'s Summary block, in millibels.

    Separate from :func:`measure` so it can be tested against captured output rather than only
    against a live FFmpeg — the parse is the fragile half, and it is the half that would break
    silently on an upgrade.

    ``-inf`` becomes ``None`` rather than a very negative number, so a caller can tell "no
    measurement" from "very quiet". Note that FFmpeg 8.0 does **not** print it for digital
    silence — it prints its `-70.0 LUFS` gating floor — so this branch is defensive rather
    than the silence path; see :attr:`Measurement.silent`.

    Raises:
        LoudnessError: if there is no integrated-loudness line at all.
    """
    integrated = _INTEGRATED.search(text)
    if integrated is None:
        message = (
            f"ffmpeg printed no integrated-loudness line while measuring {source}. Either the "
            f"ebur128 filter did not run or its summary format has changed; either way this "
            f"is not a measurement.\n{text.strip()[-2000:]}"
        )
        raise LoudnessError(message, code="loudness_unreadable")
    peak = _PEAK.search(text)
    return _millibels(integrated.group(1)), (None if peak is None else _millibels(peak.group(1)))


def ffmpeg_version() -> str:
    """FFmpeg's version string, for the report's provenance. ``"unknown"`` if it cannot run."""
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    first = completed.stdout.splitlines()[0] if completed.stdout else ""
    return first.strip() or "unknown"


def _millibels(value: str) -> int | None:
    """A decibel figure as hundredths, or ``None`` for ``-inf``."""
    if value.lstrip("-").lower() == "inf":
        return None
    # Half away from zero, matching the project's one rounding rule. FFmpeg prints one
    # decimal, so this is exact in practice and stated rather than left to `round`'s
    # half-to-even, which would make two adjacent values order inconsistently.
    scaled = float(value) * 100.0
    return int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)
