"""The one synthesizer: a :class:`~dnd_audio.marker.spec.MarkerSpec` to exact integer PCM.

Everything that needs the marker's samples comes here — the WAV the CLI writes, the bytes
embedded in the phone page, and the per-chirp templates the detector correlates against.
ADR-0041 makes that a rule rather than a habit: two approximately equivalent generators
disagree in exactly the conditions nobody tests, and the whole value of embedding the CLI's
bytes in the page evaporates if the detector matches against a third implementation.

**No floating point anywhere on this path.** A frozen SHA-256 over floating-point output
would be a promise about one platform's ``libm``, so the phase of a linear chirp is computed
in closed form, in integers, and the sine comes from a checked-in integer table
(:mod:`dnd_audio.marker.sine`).

The closed form is the part worth reading twice. Instantaneous frequency sweeps linearly,
``f(k) = f0 + (f1 - f0)*k/(N - 1)``, so the phase in turns at sample ``n`` is the sum of
``f(k)/R`` over ``k < n`` — and that sum has a closed form with every term integral:

    phase(n) = [ n*f0*(N - 1) + (f1 - f0)*n*(n - 1)/2 ] / ( R*(N - 1) )

``n*(n - 1)/2`` is always an integer, so for integer frequencies the numerator is exact.
Being a **closed form rather than an accumulator** is what matters: a phase accumulator
compounds its rounding, and a chirp is precisely the signal where that error grows
monotonically and lands as a frequency offset at the end of the sweep.

**One rounding, at the end.** :meth:`~dnd_audio.marker.sine.SineTable.sine_at` returns an
exact fraction rather than a value, the envelope is an exact fraction, and the peak amplitude
is an integer — so sine, envelope and level are composed as one rational and quantized once,
half away from zero. Rounding each factor instead would be three tie rules where ADR-0041
promises one, and the error would be systematic rather than random because a chirp's phase
advances in one direction.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt

from dnd_audio.marker import MARKER_SAMPLE_RATE
from dnd_audio.marker.sine import TABLE_SCALE, load_sine_table, round_half_away
from dnd_audio.marker.spec import ChirpSpec, MarkerSpec

__all__ = ["chirp_phase_numerator", "marker_samples", "marker_templates"]


def chirp_phase_numerator(chirp: ChirpSpec, sample: int) -> int:
    """Phase at ``sample``, in turns, over the denominator ``rate * (duration - 1)``.

    Exposed rather than inlined so a test can check the arithmetic directly — that the
    instantaneous frequency really sweeps from ``start_hz`` to ``end_hz``, by differencing
    successive phases, which is a property of the formula rather than of the spectrum it
    happens to produce.
    """
    span = chirp.duration_samples - 1
    return sample * chirp.start_hz * span + (chirp.end_hz - chirp.start_hz) * (
        sample * (sample - 1) // 2
    )


def _envelope(chirp: ChirpSpec, sample: int) -> tuple[int, int]:
    """The raised-cosine amplitude at ``sample``, as an exact fraction in ``[0, 1]``.

    Unity across the middle and ``(1 - cos(pi*p/fade))/2`` across each end, mirrored so the
    first and last samples of a chirp are exactly zero. The cosine is evaluated through the
    same table as the chirp — ``cos(t) = sin(t + pi/2)`` — so there is no second trigonometric
    implementation here either.
    """
    fade = chirp.fade_samples
    if fade == 0:
        return 1, 1

    last = chirp.duration_samples - 1
    if sample < fade:
        position = sample
    elif sample > last - fade:
        position = last - sample
    else:
        return 1, 1

    # cos(pi*position/fade) = sin(pi*position/fade + pi/2); in turns that is
    # position/(2*fade) + 1/4, i.e. (2*position + fade) / (4*fade).
    table = load_sine_table()
    cosine_numerator, denominator = table.sine_at(2 * position + fade, 4 * fade)
    scaled = denominator * TABLE_SCALE
    return scaled - cosine_numerator, 2 * scaled


@functools.lru_cache(maxsize=8)
def marker_samples(spec: MarkerSpec) -> npt.NDArray[np.int32]:
    """The complete marker waveform for ``spec``, as integer samples.

    Cached because every build, every page, and every detector run asks for the same handful
    of specs, and the arithmetic below is arbitrary-precision by design. The array is marked
    read-only: a caller that mutated it would corrupt the shared copy, and mutating the
    canonical waveform is never a legitimate thing to do.

    Deliberately a plain Python loop over arbitrary-precision integers rather than vectorized
    NumPy. The intermediate products reach roughly 1e35, which overflows int64 several times
    over, and the alternative — rescaling until everything fits — would trade an obviously
    exact computation for a fast one that is exact only if the scaling analysis is right. This
    runs once per build in well under a second.
    """
    table = load_sine_table()
    samples = np.zeros(spec.total_samples, dtype=np.int32)

    for chirp, (start, _) in zip(spec.chirps, spec.chirp_intervals(), strict=True):
        phase_denominator = MARKER_SAMPLE_RATE * (chirp.duration_samples - 1)
        for index in range(chirp.duration_samples):
            sine_numerator, sine_denominator = table.sine_at(
                chirp_phase_numerator(chirp, index), phase_denominator
            )
            envelope_numerator, envelope_denominator = _envelope(chirp, index)
            samples[start + index] = round_half_away(
                spec.peak_amplitude * sine_numerator * envelope_numerator,
                sine_denominator * TABLE_SCALE * envelope_denominator,
            )

    samples.setflags(write=False)
    return samples


def marker_templates(spec: MarkerSpec) -> tuple[npt.NDArray[np.int32], ...]:
    """Each chirp on its own, as a slice of the canonical waveform.

    The detector's matched filters are **these exact arrays**. Slicing rather than
    re-synthesizing is the point: a detector-side formula would be the second implementation
    ADR-0041 forbids, and slicing makes it structurally impossible for the templates to drift
    from the bytes that were played.
    """
    samples = marker_samples(spec)
    return tuple(samples[start:end] for start, end in spec.chirp_intervals())
