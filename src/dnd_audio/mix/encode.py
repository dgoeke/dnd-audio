"""Encode, decode, measure, and walk the gain down — or fail saying so.

The spec is unusually specific here, and every clause is load-bearing:

> Encode a 128 kbps mono MP3 with metadata containing the session ID/title, decode it, and
> measure integrated loudness and true peak. Because lossy encoding can introduce peak
> overshoot, reduce the pre-encode gain or true-peak target and re-encode from the lossless
> intermediate when necessary. Bound the retry count, retain all measurements in the report,
> and fail the mix stage rather than claim compliance if the decoded MP3 remains outside
> configured tolerances.

Three consequences, and the first is the one that shapes the loop.

**The master gain is an encode parameter, not part of the mix** (ADR-0023). The intermediate
is written at unity, so "re-encode from the lossless intermediate" costs one encode rather
than one re-mix of six four-hour tracks — and changing the loudness target does not invalidate
the most expensive artifact in the pipeline.

**The first attempt already aims at the ceiling.** The gain is the smaller of what the
loudness target asks for and what the intermediate's own measured true peak allows, so the
ordinary case needs no retry at all. That is the spec's "reduce the pre-encode gain or
true-peak target", applied before the first encode rather than after the first failure.

**Exhausting the budget fails the stage.** Every attempt's measurements are kept either way,
because a compliance claim nobody can audit is worth less than a failure that names the
numbers.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from dnd_audio.config import MixConfig
from dnd_audio.errors import DndAudioError
from dnd_audio.mix import MixNote
from dnd_audio.mix.levels import MILLIBELS_PER_DB
from dnd_audio.mix.loudness import Measurement, measure

__all__ = [
    "MP3_FRAME_SAMPLES",
    "EncodeAttempt",
    "EncodeError",
    "EncodeResult",
    "Mp3Facts",
    "encode_command",
    "encode_mp3",
    "probe_mp3",
]

#: One MPEG-1 Layer III frame. 24 ms at 48 kHz, and the unit `duration_tolerance_frames`
#: counts in — the spec's "within one MP3 frame (or another documented codec-appropriate
#: tolerance)".
MP3_FRAME_SAMPLES: Final = 1152


class EncodeError(DndAudioError):
    """The MP3 could not be produced, or could not be shown to meet its targets.

    Carries whatever the loop managed to measure. The spec asks for **all measurements
    retained in the report**, and the interesting run is the one that failed — so the caller
    records `attempts` and `commands` before re-raising rather than leaving the report with a
    single error string and nothing to audit.
    """

    default_code = "mp3_encode_failed"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        attempts: tuple[EncodeAttempt, ...] = (),
        commands: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message, code=code)
        self.attempts = attempts
        self.commands = commands


@dataclass(frozen=True, slots=True)
class EncodeAttempt:
    """One encode and the measurement of what came back out of it.

    Kept for every attempt, compliant or not: the spec asks for all measurements to be
    retained, and the interesting one is usually the attempt that failed.
    """

    index: int
    #: The master gain applied at encode time, in millibels.
    gain_mb: int
    measurement: Measurement
    #: Empty when this attempt was accepted. Otherwise the reasons it was not, in a closed
    #: vocabulary a reader can branch on rather than parse.
    failures: tuple[str, ...] = ()

    @property
    def compliant(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class Mp3Facts:
    """What the container says about itself. Separate from what decoding it produced."""

    codec: str
    channels: int
    sample_rate: int
    bit_rate_kbps: int | None
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """The accepted MP3, everything tried on the way there, and what to warn about."""

    path: Path
    facts: Mp3Facts
    attempts: tuple[EncodeAttempt, ...]
    warnings: tuple[MixNote, ...] = ()
    commands: tuple[str, ...] = field(default_factory=tuple)
    #: Whether this run aimed at the configured loudness target at all.
    #:
    #: False when one of ADR-0023's three guards fired, and then the decoded loudness was
    #: **not** compared against the target. The report says so rather than claiming a
    #: compliance nothing checked; `warnings` names which guard it was.
    normalized: bool = True

    @property
    def accepted(self) -> EncodeAttempt:
        return self.attempts[-1]


def encode_command(
    source: Path,
    destination: Path,
    *,
    settings: MixConfig,
    gain_mb: int,
    session_id: str,
    title: str,
) -> list[str]:
    """The invocation, built in one place so the report and the run cannot disagree.

    The gain is a `volume` filter rather than baked into the intermediate, and the metadata is
    the session id and title the spec names. `-map_metadata -1` first, so nothing from the
    intermediate's own headers leaks into a deliverable.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-af",
        f"volume={gain_mb / MILLIBELS_PER_DB:.2f}dB",
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{settings.mp3_bitrate_kbps}k",
        "-ac",
        "1",
        "-metadata",
        f"title={title}",
        "-metadata",
        f"album={session_id}",
        "-metadata",
        f"comment=dnd-audio automix of session {session_id}",
        str(destination),
    ]


def encode_mp3(
    source: Path,
    destination: Path,
    *,
    settings: MixConfig,
    session_id: str,
    title: str,
    source_measurement: Measurement,
    expected_samples: int,
    measurer: Callable[[Path], Measurement] = measure,
) -> EncodeResult:
    """Encode the intermediate, verify the decode, and retry a true-peak overshoot.

    Args:
        source_measurement: The intermediate's own loudness and true peak, which is what the
            first attempt's gain is computed from.
        expected_samples: The session's aligned duration. Compared against the **decoded**
            sample count, as integers.
        measurer: The decode-and-measure function. A seam, so the retry logic can be driven
            through cases a real encoder produces rarely or never.

    Raises:
        EncodeError: if FFmpeg fails, or if the retry budget is exhausted without a compliant
            decode. The spec says to fail rather than claim compliance, and INV-13 turns a
            failed stage into a nonzero exit.
    """
    encode = settings.encode
    target_mb = round(settings.integrated_lufs * MILLIBELS_PER_DB)
    ceiling_mb = round(settings.true_peak_dbtp * MILLIBELS_PER_DB)

    gain_mb, warnings, normalized = _initial_gain(
        source_measurement, target_mb=target_mb, ceiling_mb=ceiling_mb, settings=settings
    )

    attempts: list[EncodeAttempt] = []
    commands: list[str] = []
    for index in range(encode.max_retries + 1):
        command = encode_command(
            source,
            destination,
            settings=settings,
            gain_mb=gain_mb,
            session_id=session_id,
            title=title,
        )
        commands.append(" ".join(command))
        _run(command, destination)

        decoded = measurer(destination)
        failures = _failures(
            decoded,
            target_mb=target_mb,
            ceiling_mb=ceiling_mb,
            expected_samples=expected_samples,
            settings=settings,
            normalized=normalized,
        )
        attempts.append(
            EncodeAttempt(
                index=index, gain_mb=gain_mb, measurement=decoded, failures=tuple(failures)
            )
        )
        if not failures:
            return EncodeResult(
                path=destination,
                facts=probe_mp3(destination),
                attempts=tuple(attempts),
                warnings=tuple(warnings),
                commands=tuple(commands),
                normalized=normalized,
            )

        reduction = _reduction(decoded, ceiling_mb=ceiling_mb, failures=failures)
        if reduction is None:
            break
        gain_mb -= reduction

    raise EncodeError(
        _exhausted(attempts, encode.max_retries),
        code="mp3_not_compliant",
        attempts=tuple(attempts),
        commands=tuple(commands),
    )


def probe_mp3(path: Path) -> Mp3Facts:
    """What the container says about itself: codec, channels, rate, bitrate, and tags.

    Deliberately *not* where the duration comes from — that is the decoded sample count, for
    the reason ADR-0023 gives. These are the facts a header can state truthfully.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,bit_rate:format_tags",
        "-select_streams",
        "a:0",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        message = f"cannot run ffprobe on {path.name}: {exc}"
        raise EncodeError(message, code="ffprobe_unavailable") from exc
    if completed.returncode != 0:
        message = f"ffprobe could not read {path.name}: {completed.stderr.strip()[-2000:]}"
        raise EncodeError(message, code="mp3_unreadable")

    try:
        document: dict[str, Any] = json.loads(completed.stdout)
        stream = document["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        message = f"ffprobe reported no audio stream in {path.name}"
        raise EncodeError(message, code="mp3_unreadable") from exc

    raw_rate = stream.get("bit_rate")
    tags = document.get("format", {}).get("tags", {})
    return Mp3Facts(
        codec=str(stream.get("codec_name", "")),
        channels=int(stream.get("channels", 0)),
        sample_rate=int(stream.get("sample_rate", 0)),
        bit_rate_kbps=None if raw_rate is None else int(raw_rate) // 1000,
        tags={str(key): str(value) for key, value in tags.items()},
    )


def _initial_gain(
    source: Measurement, *, target_mb: int, ceiling_mb: int, settings: MixConfig
) -> tuple[int, list[MixNote], bool]:
    """The gain the first encode uses, whether it aimed at the target, and what to warn about.

    The third return value is load-bearing rather than informational: **a run that did not aim
    at the loudness target must not then be failed for missing it.** Without it a perfectly
    good mix becomes a failed stage, and `process` exits nonzero on a session that produced
    exactly the MP3 it should have. Three ways that happens, none of them hypothetical:

    * **The mix is below the silence floor.** A session where the detector found no speech has
      every track at the room-tone share, and normalizing that to -16 LUFS is fifty decibels of
      amplified noise floor. FFmpeg reports its -70 LUFS gating floor rather than `-inf` for
      digital silence, so this is a threshold rather than a null check.
    * **The gain the target wants exceeds the clamp.** A very quiet session should not be
      lifted by forty decibels on the strength of one measurement.
    * **The true-peak ceiling forbids it.** This is the one the canonical fixture actually
      produces: peaky material 31 dB down wants +15.6 dB and the ceiling allows +1.6, so the
      MP3 lands 14 LU below target. The ceiling is a hard limit on clipping and the loudness
      figure is a target; honouring the first and warning about the second is the only reading
      that does not throw away a good mix. Real speech has a lower crest factor than shaped
      noise, but a session with a loud clap in it has this shape too (OQ-020).
    """
    warnings: list[MixNote] = []
    floor_mb = round(settings.encode.silence_floor_lufs * MILLIBELS_PER_DB)

    if source.integrated_lufs_mb is None or source.integrated_lufs_mb < floor_mb:
        measured = (
            "digital silence"
            if source.integrated_lufs_mb is None
            else f"{source.integrated_lufs_mb / MILLIBELS_PER_DB:.1f} LUFS"
        )
        warnings.append(
            MixNote(
                code="mix_not_normalized",
                message=(
                    f"the mix measures {measured}, below the "
                    f"{settings.encode.silence_floor_lufs:.1f} LUFS floor, so it is encoded "
                    f"without loudness normalization. Normalizing it would amplify a noise "
                    f"floor by tens of decibels. A session where the detector found no speech "
                    f"produces exactly this (OQ-019)."
                ),
            )
        )
        return 0, warnings, False

    wanted = target_mb - source.integrated_lufs_mb
    limit_mb = round(settings.encode.max_master_gain_db * MILLIBELS_PER_DB)
    if abs(wanted) > limit_mb:
        clamped = limit_mb if wanted > 0 else -limit_mb
        warnings.append(
            MixNote(
                code="mix_master_gain_clamped",
                message=(
                    f"reaching {settings.integrated_lufs:.1f} LUFS from "
                    f"{source.integrated_lufs_mb / MILLIBELS_PER_DB:.1f} needs "
                    f"{wanted / MILLIBELS_PER_DB:+.1f} dB; clamped to "
                    f"{clamped / MILLIBELS_PER_DB:+.1f} dB (OQ-019)."
                ),
            )
        )
        wanted = clamped
        return wanted, warnings, False

    # Aim at the ceiling before the first encode rather than after the first failure — the
    # spec's "reduce the pre-encode gain or true-peak target", applied before the first
    # attempt rather than after it (OQ-020).
    peak_mb = source.true_peak_dbtp_mb
    headroom = None if peak_mb is None else ceiling_mb - peak_mb
    if peak_mb is not None and headroom is not None and headroom < wanted:
        warnings.append(
            MixNote(
                code="mix_loudness_target_unreachable",
                message=(
                    f"reaching {settings.integrated_lufs:.1f} LUFS needs "
                    f"{wanted / MILLIBELS_PER_DB:+.1f} dB, but the "
                    f"{settings.true_peak_dbtp:.1f} dBTP ceiling allows only "
                    f"{headroom / MILLIBELS_PER_DB:+.1f} dB above this mix's true peak of "
                    f"{peak_mb / MILLIBELS_PER_DB:.1f} dBTP. The ceiling "
                    f"wins: it is a limit on clipping, and the loudness figure is a target. "
                    f"The MP3 will be about "
                    f"{(wanted - headroom) / MILLIBELS_PER_DB:.1f} LU quieter than asked for "
                    f"(OQ-020)."
                ),
            )
        )
        return headroom, warnings, False
    return wanted, warnings, True


def _failures(
    decoded: Measurement,
    *,
    target_mb: int,
    ceiling_mb: int,
    expected_samples: int,
    settings: MixConfig,
    normalized: bool,
) -> list[str]:
    """Every configured tolerance this decode is outside, as a closed vocabulary.

    ``normalized`` is False when this run deliberately did not aim at the loudness target —
    the mix was below the silence floor, the gain the target wanted exceeded the clamp, or the
    true-peak ceiling forbade it. The loudness check is then skipped, because failing a run
    for missing a target it was told not to aim at throws away a good MP3 and makes `process`
    exit nonzero on a session that produced exactly what it should have. That waiver is a
    documented amendment to the spec's acceptance criterion 8 rather than a silent one — see
    ADR-0023 — and every waived run carries a warning naming which of the three guards fired.
    The true-peak and duration checks still apply either way: those are claims about the file
    rather than about a target.

    **A measurement nobody took is never a pass.** Two ways that used to slip through, both
    found by M5's code review:

    * a decode that reported `-inf` loudness on a run that *was* aiming at the target — the
      MP3 is silent, and skipping the comparison because the number is ``None`` reports the
      silence as compliant;
    * a summary with no true-peak line at all, which means `peak=true` did not take effect.
      That is a different thing from `Peak: -inf dBFS`, which FFmpeg prints for digital
      silence and which is a real measurement infinitely below any ceiling.
    """
    encode = settings.encode
    found: list[str] = []

    tolerance_mb = round(encode.loudness_tolerance_lu * MILLIBELS_PER_DB)
    if normalized and (
        decoded.integrated_lufs_mb is None
        or abs(decoded.integrated_lufs_mb - target_mb) > tolerance_mb
    ):
        found.append("integrated_loudness")

    peak_tolerance_mb = round(encode.true_peak_tolerance_db * MILLIBELS_PER_DB)
    if not decoded.true_peak_reported:
        found.append("true_peak_unmeasured")
    elif (
        decoded.true_peak_dbtp_mb is not None
        and decoded.true_peak_dbtp_mb > ceiling_mb + peak_tolerance_mb
    ):
        found.append("true_peak")

    allowed = encode.duration_tolerance_frames * MP3_FRAME_SAMPLES
    if abs(decoded.n_samples - expected_samples) > allowed:
        found.append("duration")
    return found


def _reduction(decoded: Measurement, *, ceiling_mb: int, failures: list[str]) -> int | None:
    """How far to pull the gain down before the next attempt, or ``None`` to stop.

    Only a true-peak overshoot is worth retrying. A duration mismatch is not a gain problem,
    and a loudness miss without an overshoot means the target itself is unreachable — walking
    the gain in either direction would be guessing, and the spec asks for a failure rather
    than a guess.
    """
    if failures != ["true_peak"] or decoded.true_peak_dbtp_mb is None:
        return None
    # One millibel past the overshoot, so an attempt that lands exactly on the ceiling still
    # moves. A reduction of zero would burn the whole budget re-encoding the same file.
    return max(1, decoded.true_peak_dbtp_mb - ceiling_mb)


def _run(command: list[str], destination: Path) -> None:
    """Encode, or raise. A partial MP3 left behind is worse than none."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        message = f"cannot run ffmpeg to encode {destination.name}: {exc}"
        raise EncodeError(message, code="ffmpeg_unavailable") from exc
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        message = (
            f"ffmpeg exited {completed.returncode} encoding {destination.name}: "
            f"{completed.stderr.strip()[-2000:]}"
        )
        raise EncodeError(message, code="ffmpeg_failed")


def _exhausted(attempts: list[EncodeAttempt], budget: int) -> str:
    """The diagnostic for a mix that never met its targets, with every number in it."""
    lines = [
        f"the encoded MP3 is still outside its configured tolerances after {len(attempts)} "
        f"attempt(s) (budget {budget} retries). The stage fails rather than claiming a "
        f"compliance nothing demonstrated."
    ]
    for attempt in attempts:
        loudness = _describe(attempt.measurement.integrated_lufs_mb, "LUFS")
        peak = _describe(attempt.measurement.true_peak_dbtp_mb, "dBTP")
        lines.append(
            f"  attempt {attempt.index}: gain {attempt.gain_mb / MILLIBELS_PER_DB:+.2f} dB, "
            f"{loudness}, {peak}, {attempt.measurement.n_samples} samples decoded — "
            f"{', '.join(attempt.failures)}"
        )
    return "\n".join(lines)


def _describe(millibels: int | None, unit: str) -> str:
    if millibels is None:
        return f"-inf {unit}"
    return f"{millibels / MILLIBELS_PER_DB:.2f} {unit}"
