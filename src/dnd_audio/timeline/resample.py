"""Stream 48 kHz down to 16 kHz through one fixed filter, without ever resetting it.

The mapping this produces is the most likely source of a subtle, late-discovered offset in
the whole pipeline: a word timestamp that is consistently 40 ms early looks like a bad
aligner, not like a resampler. So the arithmetic is arranged to be exact rather than
nearly exact, and every constant that makes it so is checked.

**The filter runs across the whole virtual track and is never reset at a chunk or gap
boundary.** A reset would put a transient at every boundary and make the derivative depend
on how DJI happened to split the recording — the same class of bug as placing chunks
relative to each other. State and decimation phase are carried between windows, and
`tests/test_resample.py` asserts a streamed run is byte-identical to a one-shot one at
every window partitioning.

**Output sample `k` corresponds to input sample `3k`, exactly.** Two facts make that true
rather than approximately true: 48000/16000 is exactly 3, and the filter's group delay of
129 input samples divides by 3, so compensating for it is a slice of 43 output samples
rather than an interpolation. Both are asserted in `tests/test_fir.py`; neither is assumed
here.

That correspondence is exact in one direction only, which is worth being precise about
because it is where an off-by-one gets in. An output index maps *back* to an input index
exactly (`3k`). An input index maps *forward* onto a grid it need not land on, so
converting an interval uses :func:`to_derivative_interval` — floor the start, ceil the end
— and never a single rounding rule for both. Rounding both ends the same way shrinks an
interval by up to two samples at 48 kHz, which is how a word loses its first phoneme. A
*filtered impulse* peaks at the nearest grid point rather than the floor, which is a fact
about the filter and not about the mapping; `tests/test_resample.py` asserts it as such.

**The output length is `ceil(n / 3)`.** Rounding down would silently drop the tail of every
track whose length is not a multiple of three — up to two samples, every time, in the same
direction. The input is zero-padded to reach it, which is the only invented sample in the
pipeline and is confined to the last two of a derivative.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import numpy.typing as npt
from scipy.signal import upfirdn

from dnd_audio.timeline.fir import DecimationFilter, load_decimation_filter

__all__ = [
    "Decimator",
    "decimate_stream",
    "output_length",
    "to_derivative_interval",
    "to_source_sample",
]


def output_length(n_input: int, decimation: int) -> int:
    """How many output samples ``n_input`` produces. `ceil`, and stated once."""
    return -(-n_input // decimation)


def to_source_sample(output_sample: int, decimation: int) -> int:
    """The input sample an output sample corresponds to. Exact, by construction."""
    return output_sample * decimation


def to_derivative_interval(start_sample: int, end_sample: int, decimation: int) -> tuple[int, int]:
    """Map a half-open input interval onto the derivative grid without losing either end.

    Floor the start and ceil the end, so the result always *covers* the input interval.
    Rounding both ends the same way would shrink it by up to two input samples — and the
    intervals this converts are speech regions, where the missing samples are the start of
    a word and the end of one.
    """
    if end_sample < start_sample:
        message = f"interval [{start_sample}, {end_sample}) runs backwards"
        raise ValueError(message)
    return start_sample // decimation, -(-end_sample // decimation)


class Decimator:
    """One track's 3:1 decimation, fed in arbitrary blocks.

    The state carried between blocks is the last ``len(h) - 1`` input samples, which is
    exactly what the convolution needs to produce the next output as if the whole track
    had been passed at once. It is 258 samples here, and 258 divides by 3 — that is what
    lets each block's outputs be sliced at a fixed offset instead of the decimation phase
    drifting with the block sizes.

    Args:
        n_input: The track's full length. Known before the first sample arrives, because
            the timeline says so, and needed so the tail can be flushed to exactly the
            right length rather than to however much the last block happened to contain.
    """

    def __init__(self, n_input: int, *, decimation_filter: DecimationFilter | None = None) -> None:
        self._filter = decimation_filter or load_decimation_filter()
        self._down = self._filter.decimation
        self._taps = self._filter.coefficients
        self._history = self._taps.shape[0] - 1
        if self._history % self._down:  # pragma: no cover - the loader forbids it
            message = (
                f"a {self._taps.shape[0]}-tap filter carries {self._history} samples of "
                f"history, which is not a whole number of output samples at 1/{self._down}"
            )
            raise ValueError(message)

        self._n_input = n_input
        self._expected = output_length(n_input, self._down)
        #: Outputs to discard: the filter's group delay, in output samples. Discarding
        #: them is what makes output k correspond to input 3k rather than to 3k - 129.
        self._skip = self._filter.group_delay_output
        self._state = np.zeros(self._history, dtype=np.float64)
        self._pending = np.zeros(0, dtype=np.float64)
        self._consumed = 0
        self._emitted = 0
        self._discarded = 0

    @property
    def expected_output(self) -> int:
        return self._expected

    def process(self, block: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Feed one block; return whatever output is ready. May be empty."""
        self._consumed += int(block.shape[0])
        return self._advance(np.asarray(block, dtype=np.float64))

    def flush(self) -> npt.NDArray[np.float32]:
        """Finish the track, zero-padding the tail to the declared output length.

        The filter still holds 129 samples of the track inside its delay line when the
        last input sample arrives. Padding pushes them out; stopping without it would cut
        43 output samples — 2.7 ms — off the end of every track.
        """
        parts: list[npt.NDArray[np.float32]] = []
        if self._pending.size:
            padding = self._down - int(self._pending.size) % self._down
            parts.append(self._advance(np.zeros(padding, dtype=np.float64)))

        # Enough zeros to drive the delay line out and reach the last output we owe.
        while self._emitted < self._expected:
            parts.append(self._advance(np.zeros(self._history, dtype=np.float64)))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _advance(self, samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        """Convolve and decimate a run of input, carrying state across the boundary."""
        buffered = np.concatenate((self._pending, samples)) if self._pending.size else samples
        usable = (buffered.shape[0] // self._down) * self._down
        self._pending = buffered[usable:]
        if usable == 0:
            return np.zeros(0, dtype=np.float32)

        block = buffered[:usable]
        # `upfirdn` over history + block computes the same convolution the whole track
        # would have produced; the first `history / down` outputs re-derive samples the
        # previous block already emitted and are dropped.
        decimated = upfirdn(self._taps, np.concatenate((self._state, block)), up=1, down=self._down)
        fresh = np.asarray(decimated[self._history // self._down :][: usable // self._down])
        self._state = np.concatenate((self._state, block))[-self._history :]

        if self._discarded < self._skip:
            drop = min(self._skip - self._discarded, int(fresh.shape[0]))
            self._discarded += drop
            fresh = fresh[drop:]

        remaining = self._expected - self._emitted
        if fresh.shape[0] > remaining:
            fresh = fresh[:remaining]
        self._emitted += int(fresh.shape[0])
        return np.asarray(fresh, dtype=np.float32)


def decimate_stream(
    blocks: Iterable[npt.NDArray[np.float32]],
    n_input: int,
    *,
    decimation_filter: DecimationFilter | None = None,
) -> Iterator[npt.NDArray[np.float32]]:
    """Decimate a stream of input blocks into a stream of output blocks.

    A generator over a generator: neither the input nor the output is ever whole in
    memory, which is what makes a four-hour derivative cost a window rather than a
    gigabyte (INV-07). Empty results are not yielded, so a consumer counting writes counts
    real ones.
    """
    decimator = Decimator(n_input, decimation_filter=decimation_filter)
    for block in blocks:
        produced = decimator.process(block)
        if produced.size:
            yield produced
    tail = decimator.flush()
    if tail.size:
        yield tail
