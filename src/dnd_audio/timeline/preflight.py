"""Check there is room before expanding a session (INV-07).

The spec asks for work-space usage to be preflighted "before expanding long sessions", and
`doctor` already warns below a fixed 40 GiB derived from arithmetic in `doctor.py` about an
assumed four-hour session. This is the version that knows the real numbers: the timeline
has already been built, so the session's actual duration and the artifacts actually
requested are both known, and the estimate is arithmetic rather than a guess.

Two of the three terms in `doctor`'s original estimate turn out not to exist. There is no
15 GiB of materialized 48 kHz working audio unless `--materialize-48k` asks for it
(ADR-0011), and the mix intermediate belongs to M5. So this **partially answers OQ-013**
and does not close it: what a full pipeline consumes still needs a real session to measure.

Running out of disk halfway through writing six derivatives leaves a directory of
half-files that the cache correctly refuses and that nothing cleans up. Refusing before the
first byte is cheaper for everyone.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dnd_audio.artifacts.timeline import TimelineNote
from dnd_audio.errors import DndAudioError
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE, DERIVATIVE_SAMPLE_RATE

__all__ = ["HEADROOM_FACTOR", "WorkspaceError", "WorkspaceEstimate", "estimate", "preflight"]

_BYTES_PER_SAMPLE: Final = 4

#: How much more than the estimate should be free before the run proceeds quietly. Not a
#: safety margin on the estimate — that part is exact — but on everything else the machine
#: is doing with the same disk while a four-hour session is being processed.
HEADROOM_FACTOR: Final = 2


class WorkspaceError(DndAudioError):
    """There is not enough disk to write what this run would produce."""

    default_code = "insufficient_work_space"


@dataclass(frozen=True, slots=True)
class WorkspaceEstimate:
    """What this run will write, and what is available.

    Exact rather than approximate: every artifact's length is known from the timeline, and
    float32 is four bytes. The only uncertainty is what *else* will use the disk.
    """

    duration_samples: int
    track_count: int
    derivative_bytes: int
    materialized_bytes: int
    free_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.derivative_bytes + self.materialized_bytes

    @property
    def sufficient(self) -> bool:
        return self.free_bytes >= self.total_bytes

    @property
    def comfortable(self) -> bool:
        return self.free_bytes >= self.total_bytes * HEADROOM_FACTOR


def estimate(
    session_dir: Path, *, duration_samples: int, track_count: int, materialize_48k: bool
) -> WorkspaceEstimate:
    """Size this run's output from the timeline it is about to write.

    Every track's derivative is the session's full duration, not the track's own: a track
    that stopped early is still readable to the aligned duration, and the derivative is
    what M3 and M4 read.
    """
    per_track_16k = _decimated_length(duration_samples) * _BYTES_PER_SAMPLE
    per_track_48k = duration_samples * _BYTES_PER_SAMPLE if materialize_48k else 0
    return WorkspaceEstimate(
        duration_samples=duration_samples,
        track_count=track_count,
        derivative_bytes=per_track_16k * track_count,
        materialized_bytes=per_track_48k * track_count,
        free_bytes=shutil.disk_usage(session_dir).free,
    )


def preflight(found: WorkspaceEstimate) -> list[TimelineNote]:
    """Refuse an impossible run; warn about a tight one.

    Raises:
        WorkspaceError: when the estimate exceeds the free space. Naming both numbers,
            because "not enough disk" without them is a message an operator cannot act on.
    """
    if not found.sufficient:
        message = (
            f"this run would write {_gib(found.total_bytes)} of working audio for "
            f"{found.track_count} track(s) over "
            f"{found.duration_samples / CANONICAL_SAMPLE_RATE / 3600:.1f} hours, and only "
            f"{_gib(found.free_bytes)} is free. Free some space, or drop "
            f"--materialize-48k if it was requested."
        )
        raise WorkspaceError(message)

    if found.comfortable:
        return []
    return [
        TimelineNote(
            code="work_space_tight",
            message=(
                f"{_gib(found.free_bytes)} free against an estimated "
                f"{_gib(found.total_bytes)} of working audio. The run will fit, with "
                f"little room for anything else this machine is doing."
            ),
        )
    ]


def _decimated_length(n_samples: int) -> int:
    """`ceil`, matching the resampler's own rule so the estimate is not one short."""
    factor = CANONICAL_SAMPLE_RATE // DERIVATIVE_SAMPLE_RATE
    return -(-n_samples // factor)


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.2f} GiB"
