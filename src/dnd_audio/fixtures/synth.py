"""Deterministic signals for the synthetic fixtures.

Three sounds, which are the three the spec's fixture recipe asks for: speech-shaped
activity, a shared transient, and a quiet delayed copy of one track's speech leaking
into the others.

**Nothing here is speech.** INV-10 is explicit that synthetic noise must never be
expected to trigger a particular learned Silero release. These signals exist so that
correlation, gain, and timing behave the way a real recording's would; the ground truth
about *what is speaking when* comes from the fixture's truth record, never from a
detector's opinion about this audio.

Determinism comes from an explicitly constructed :class:`~numpy.random.PCG64`, whose
stream NumPy's versioning policy guarantees stable across releases. It is spelled out
rather than left to :func:`~numpy.random.default_rng` because "the default" is a thing
NumPy is allowed to change, and a fixture whose bytes depend on it would be reproducible
only until the next upgrade.

Seeding per *event* rather than per file is what lets a chunked renderer produce the
same bytes as a whole-track one: an event is rendered once at full length and then
sliced, so where the chunk boundaries fall cannot change the samples.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "CLAP_INTERVAL_S",
    "CLAP_PATTERN",
    "bleed_of",
    "clap",
    "noise_floor",
    "speech_shaped",
]

#: The spec's fixture recipe asks for "a distinctive three-clap pattern".
CLAP_PATTERN: Final = 3

#: Seconds between claps in that pattern.
CLAP_INTERVAL_S: Final = 0.25

#: Roughly the first formant region. The envelope is a smooth band rather than a filter
#: design: this needs to look like speech to a correlator, not to a phonetician.
_FORMANT_HZ: Final = 500.0
_ROLLOFF_HZ: Final = 1800.0

#: Syllables per second. Real conversational speech is 3-5.
_SYLLABLE_HZ: Final = 4.0


def _rng(seed: int) -> np.random.Generator:
    """A generator pinned to a named bit generator, not to whatever the default is."""
    return np.random.Generator(np.random.PCG64(seed))


def speech_shaped(
    n_samples: int, sample_rate: int, *, seed: int, gain: float = 0.2
) -> npt.NDArray[np.float32]:
    """Noise with a speech-like spectrum and a syllabic amplitude envelope.

    Shaped in the frequency domain rather than through an IIR filter: a one-pole filter
    is a Python loop over a quarter-million samples per event, and the vectorized
    version is both faster and exactly reproducible without depending on accumulation
    order.
    """
    if n_samples <= 0:
        message = f"n_samples must be positive, got {n_samples}"
        raise ValueError(message)

    white = _rng(seed).standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    # A resonance at the first formant, rolling off above it. +1 in the denominator
    # keeps DC finite.
    envelope = np.exp(-(((freqs - _FORMANT_HZ) / _ROLLOFF_HZ) ** 2))
    shaped = np.fft.irfft(spectrum * envelope, n=n_samples)

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    syllables = 0.5 * (1.0 - np.cos(2.0 * np.pi * _SYLLABLE_HZ * time)) ** 2
    voiced = shaped * syllables * _fade(n_samples, sample_rate)

    peak = float(np.max(np.abs(voiced)))
    scale = gain / peak if peak > 0 else 0.0
    return (voiced * scale).astype(np.float32)


def clap(
    n_samples: int, sample_rate: int, *, seed: int, gain: float = 0.7
) -> npt.NDArray[np.float32]:
    """A three-burst transient: the fixture's shared synchronization landmark.

    Bursts are short and broadband, which is what makes the cross-correlation M2 uses
    for sync QA have an unambiguous peak.
    """
    out = np.zeros(n_samples, dtype=np.float64)
    burst_len = max(1, int(0.004 * sample_rate))
    decay = np.exp(-np.arange(burst_len, dtype=np.float64) / (burst_len / 4.0))

    for index in range(CLAP_PATTERN):
        start = int(index * CLAP_INTERVAL_S * sample_rate)
        end = min(start + burst_len, n_samples)
        if start >= n_samples:
            break
        burst = _rng(seed + index).standard_normal(end - start) * decay[: end - start]
        out[start:end] += burst

    peak = float(np.max(np.abs(out)))
    scale = gain / peak if peak > 0 else 0.0
    return (out * scale).astype(np.float32)


def bleed_of(
    signal: npt.NDArray[np.float32], *, delay_samples: int, attenuation_db: float
) -> npt.NDArray[np.float32]:
    """A quiet, delayed copy — one transmitter picking up another wearer's voice.

    The delay is what makes the bleed gate a real test: a correlator measuring only zero
    lag would find a delayed copy of the same speech uncorrelated, which is why the
    configuration has ``activity.correlation_max_lag_ms`` at all.
    """
    if delay_samples < 0:
        message = f"delay_samples must not be negative, got {delay_samples}"
        raise ValueError(message)
    gain = 10.0 ** (-abs(attenuation_db) / 20.0)
    delayed = np.concatenate([np.zeros(delay_samples, dtype=np.float32), signal])
    attenuated: npt.NDArray[np.float32] = (delayed * gain).astype(np.float32)
    return attenuated


def noise_floor(n_samples: int, *, seed: int, level: float = 1e-4) -> npt.NDArray[np.float32]:
    """The self-noise every real recording has, and no digital silence does.

    Without it a fixture's "silence" is exactly zero, which lets a bug that keys off
    ``== 0`` pass here and fail on the first real session.
    """
    return (_rng(seed).standard_normal(n_samples) * level).astype(np.float32)


def _fade(n_samples: int, sample_rate: int) -> npt.NDArray[np.float64]:
    """A short raised-cosine fade at both ends, so an event has no edge click."""
    fade_len = min(int(0.01 * sample_rate), n_samples // 2)
    window = np.ones(n_samples, dtype=np.float64)
    if fade_len <= 0:
        return window
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len, dtype=np.float64) / fade_len))
    window[:fade_len] = ramp
    window[-fade_len:] = ramp[::-1]
    return window
