"""What a marker *is*: a frozen registry of named waveform specifications.

Every number here is a decision, and none of them is frozen yet. ADR-0041 records why the
registry holds **candidates** rather than a `v1`: the exact band, chirp duration, direction,
gaps and level are properties of what a phone speaker radiates, what a lav capsule accepts,
and what a room does in between — none of which a synthetic fixture can answer. A physical
bench selects one, and ADR-0042 adds the `v1` entry with its exact integer PCM frozen by
SHA-256. Until then :func:`resolve` refuses to guess.

The three candidates are chosen to span the questions the bench exists to answer rather than
to be three flavours of one guess. See each one's ``rationale``.

**Two structural properties every candidate shares**, because the detector depends on both:

* **Three chirps, with asymmetric gaps.** The asymmetry is what makes the whole sequence far
  less likely than one speech or music sweep, and it is what lets the detector reject a
  reversed, truncated or partly obscured pattern. Two equal gaps would make a
  time-reversed sequence indistinguishable from the real one.
* **The anchor is the first sample of the first chirp** — not the first sample of the file,
  which is silence, and not a correlation peak, which is a measurement. A frozen anchor
  expressed against the WAV's own start is what makes ``relative_lag_samples`` comparable
  between two occurrences recorded weeks apart (ADR-0041).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dnd_audio.errors import DndAudioError
from dnd_audio.marker import MARKER_SAMPLE_RATE

__all__ = [
    "MARKER_SPECS",
    "ChirpSpec",
    "MarkerSpec",
    "UnknownMarkerError",
    "resolve",
]


class UnknownMarkerError(DndAudioError):
    """A marker name that is not in the registry, or no name where none can be inferred."""

    default_code = "unknown_marker"


def _ms(milliseconds: int) -> int:
    """Whole samples for a whole number of milliseconds at the marker rate.

    Every duration in this module is stated in milliseconds and converted here, so a spec
    reads the way an operator thinks and still lands on an exact integer sample. 48 kHz makes
    every millisecond exactly 48 samples; the assertion is that this stays true rather than
    becoming a rounding nobody notices.
    """
    samples, remainder = divmod(milliseconds * MARKER_SAMPLE_RATE, 1000)
    if remainder:  # pragma: no cover - unreachable at 48 kHz, kept as the guard it is
        message = f"{milliseconds} ms is not a whole number of samples at {MARKER_SAMPLE_RATE} Hz"
        raise ValueError(message)
    return samples


@dataclass(frozen=True, slots=True)
class ChirpSpec:
    """One linear frequency sweep, with a raised-cosine fade at each end.

    ``start_hz`` and ``end_hz`` are integers because the phase arithmetic in
    :mod:`dnd_audio.marker.synth` is exact only for integer frequencies — the closed form
    ``n*f0*(N-1) + (f1-f0)*n*(n-1)/2`` is an integer expression, and a fractional hertz would
    reintroduce the floating-point phase this design exists to avoid.
    """

    start_hz: int
    end_hz: int
    duration_samples: int
    #: Raised-cosine fade length at *each* end. Without it a chirp starts on a discontinuity,
    #: which spreads energy across the whole spectrum and makes the matched filter's job
    #: harder rather than easier — and clicks audibly on a phone speaker.
    fade_samples: int

    def __post_init__(self) -> None:
        if self.start_hz <= 0 or self.end_hz <= 0:
            message = f"chirp frequencies must be positive, got {self.start_hz}..{self.end_hz}"
            raise ValueError(message)
        if self.start_hz == self.end_hz:
            message = (
                f"a chirp sweeps; {self.start_hz} Hz to itself is a tone, whose correlation "
                f"peak is ambiguous by whole cycles (OQ-025)"
            )
            raise ValueError(message)
        nyquist = MARKER_SAMPLE_RATE // 2
        if max(self.start_hz, self.end_hz) >= nyquist:
            message = (
                f"{max(self.start_hz, self.end_hz)} Hz is at or above the {nyquist} Hz "
                f"Nyquist limit of the {MARKER_SAMPLE_RATE} Hz marker grid, so it would "
                f"alias rather than sweep"
            )
            raise ValueError(message)
        if self.duration_samples < 2:
            message = f"a chirp needs at least two samples, got {self.duration_samples}"
            raise ValueError(message)
        if self.fade_samples < 0 or 2 * self.fade_samples > self.duration_samples:
            message = (
                f"fade_samples={self.fade_samples} does not fit twice inside "
                f"duration_samples={self.duration_samples}"
            )
            raise ValueError(message)

    @property
    def rises(self) -> bool:
        """Whether this chirp sweeps upward. Recorded so a manifest states the direction."""
        return self.end_hz > self.start_hz


@dataclass(frozen=True, slots=True)
class MarkerSpec:
    """A complete, self-describing marker waveform.

    Everything needed to produce the exact bytes, and nothing else — no filenames, no output
    paths, no run-specific state. Two builds of the same spec are byte-identical by
    construction rather than by care.
    """

    name: str
    chirps: tuple[ChirpSpec, ...]
    #: One fewer than the number of chirps. Deliberately unequal — see the module docstring.
    gaps_samples: tuple[int, ...]
    lead_silence_samples: int
    trail_silence_samples: int
    #: Peak sample value in output units. An integer rather than a decibel figure, so the
    #: level is exact and needs no rounding: 16384 is 2**14, exactly 6.02 dB below a 16-bit
    #: full scale of 32767. Conservative on purpose — the nearest lav must not clip, and
    #: OQ-025 records that headroom is worth more here than loudness.
    peak_amplitude: int
    #: Why this candidate exists, in the operator's terms. Reaches the manifest, so a WAV on
    #: a phone can be traced back to the question it was built to answer.
    rationale: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("-", "").isalnum():
            message = (
                f"a marker name becomes a filename and a manifest key, so it must be "
                f"alphanumeric with hyphens; got {self.name!r}"
            )
            raise ValueError(message)
        if len(self.chirps) < 2:
            message = (
                f"a marker is a *sequence*; {len(self.chirps)} chirp(s) has no gap structure "
                f"to distinguish it from an ordinary sweep (ADR-0041)"
            )
            raise ValueError(message)
        if len(self.gaps_samples) != len(self.chirps) - 1:
            message = (
                f"{len(self.chirps)} chirps need {len(self.chirps) - 1} gaps, got "
                f"{len(self.gaps_samples)}"
            )
            raise ValueError(message)
        if any(gap <= 0 for gap in self.gaps_samples):
            message = f"every gap must be positive, got {self.gaps_samples}"
            raise ValueError(message)
        if len(set(self.gaps_samples)) != len(self.gaps_samples):
            message = (
                f"the gaps must all differ, got {self.gaps_samples}. Equal gaps make a "
                f"time-reversed sequence indistinguishable from the real one, which is the "
                f"rejection the asymmetry exists to buy."
            )
            raise ValueError(message)
        if self.lead_silence_samples < 0 or self.trail_silence_samples < 0:
            message = "silence lengths cannot be negative"
            raise ValueError(message)
        if not 0 < self.peak_amplitude < 32768:
            message = (
                f"peak_amplitude={self.peak_amplitude} is outside the range a 16-bit sample "
                f"can hold without clipping"
            )
            raise ValueError(message)

    @property
    def anchor_sample(self) -> int:
        """The frozen anchor: the first sample of the first chirp, relative to WAV start.

        Not sample zero, which is silence and carries no energy to correlate against, and not
        a peak position, which is a measurement rather than a definition.
        """
        return self.lead_silence_samples

    @property
    def total_samples(self) -> int:
        """Length of the complete WAV in samples."""
        return (
            self.lead_silence_samples
            + sum(chirp.duration_samples for chirp in self.chirps)
            + sum(self.gaps_samples)
            + self.trail_silence_samples
        )

    def chirp_intervals(self) -> tuple[tuple[int, int], ...]:
        """Half-open ``[start, end)`` of each chirp, relative to the WAV's first sample.

        The detector's per-chirp templates are slices at exactly these positions, and the
        manifest publishes them, so an operator can confirm what was played against what was
        looked for without reading any code.
        """
        intervals: list[tuple[int, int]] = []
        position = self.lead_silence_samples
        for index, chirp in enumerate(self.chirps):
            intervals.append((position, position + chirp.duration_samples))
            position += chirp.duration_samples
            if index < len(self.gaps_samples):
                position += self.gaps_samples[index]
        return tuple(intervals)

    def gap_intervals(self) -> tuple[tuple[int, int], ...]:
        """Half-open ``[start, end)`` of each inter-chirp gap.

        Published for the same reason as the chirps, and consumed by the detector's sequence
        check: the gaps *are* the code, and their asymmetry is what a reversed pattern fails.
        """
        chirps = self.chirp_intervals()
        return tuple((chirps[i][1], chirps[i + 1][0]) for i in range(len(chirps) - 1))


def _candidate(
    name: str,
    *,
    band: tuple[int, int],
    chirp_ms: int,
    directions: str,
    gaps_ms: tuple[int, int],
    rationale: str,
) -> MarkerSpec:
    """Build one candidate from the terms the bench protocol talks in.

    ``directions`` is a string of ``u``/``d`` per chirp, so a candidate's shape is legible in
    the registry below rather than buried in a list of frequency pairs.
    """
    low, high = band
    chirps = tuple(
        ChirpSpec(
            start_hz=low if step == "u" else high,
            end_hz=high if step == "u" else low,
            duration_samples=_ms(chirp_ms),
            # 10 ms at each end. Long enough to remove the discontinuity, short enough that
            # the faded region is a small fraction of even the shortest candidate chirp.
            fade_samples=_ms(10),
        )
        for step in directions
    )
    return MarkerSpec(
        name=name,
        chirps=chirps,
        gaps_samples=(_ms(gaps_ms[0]), _ms(gaps_ms[1])),
        lead_silence_samples=_ms(100),
        trail_silence_samples=_ms(100),
        peak_amplitude=1 << 14,
        rationale=rationale,
    )


#: The candidates the bench chooses between. **No `v1` key** — see the module docstring and
#: ADR-0042. Ordered as they are played at the bench.
MARKER_SPECS: Final[dict[str, MarkerSpec]] = {
    spec.name: spec
    for spec in (
        _candidate(
            "cand-a",
            band=(500, 8000),
            chirp_ms=180,
            directions="uuu",
            gaps_ms=(150, 250),
            rationale=(
                "The charter's provisional design: three rising 500 Hz - 8 kHz sweeps. The "
                "reference point every other candidate is measured against."
            ),
        ),
        _candidate(
            "cand-b",
            band=(800, 6000),
            chirp_ms=250,
            directions="uuu",
            gaps_ms=(200, 320),
            rationale=(
                "Narrower band, longer sweeps. A phone speaker radiates little below about "
                "700 Hz and a lav capsule rolls off at the top, so this trades bandwidth for "
                "time-bandwidth product: more matched-filter processing gain in the band the "
                "hardware actually passes. The candidate to beat at the farthest seat."
            ),
        ),
        _candidate(
            "cand-c",
            band=(400, 10000),
            chirp_ms=120,
            directions="udu",
            gaps_ms=(90, 160),
            rationale=(
                "Wider band, shorter sweeps, and alternating direction. Asks two questions at "
                "once: whether a short chirp survives room reverberation, and whether "
                "direction asymmetry buys rejection that gap asymmetry alone does not. The "
                "least audible interruption of the three if it works."
            ),
        ),
    )
}


def resolve(name: str | None) -> MarkerSpec:
    """The spec for ``name``, or the frozen `v1` when no name is given.

    Raises:
        UnknownMarkerError: when ``name`` is absent and no `v1` exists — which is the state
            until the bench selects one, and the reason this refuses rather than defaulting
            to a candidate. An operator who accidentally recorded Session Zero against an
            unvalidated waveform would have no way to know (ADR-0041, ADR-0042).
    """
    if name is None:
        if "v1" not in MARKER_SPECS:
            message = (
                "no marker is frozen as v1 yet, so there is nothing for `marker build` to "
                "build by default. A waveform becomes v1 only after the phone/DJI bench "
                "selects it from measured evidence (ADR-0042); until then, build a candidate "
                f"for the bench with --marker: {', '.join(sorted(MARKER_SPECS))}. The "
                "protocol is docs/M10-marker-bench-protocol.md."
            )
            raise UnknownMarkerError(message, code="marker_not_selected")
        return MARKER_SPECS["v1"]

    try:
        return MARKER_SPECS[name]
    except KeyError:
        message = f"no marker named {name!r}; this build carries {', '.join(sorted(MARKER_SPECS))}"
        raise UnknownMarkerError(message) from None
