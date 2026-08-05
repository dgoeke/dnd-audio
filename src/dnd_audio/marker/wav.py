"""The marker's RIFF container, stated where its bytes are frozen.

ADR-0042 freezes marker v1 by the SHA-256 of a complete file, so the container's layout is
part of what that hash means — not an implementation detail underneath it. That is why the
twenty lines below exist rather than a call into :mod:`dnd_audio.fixtures.wav`, which is
fixture support, says so in its own docstring, and writes with ``path.write_bytes`` where this
needs bytes it can hand to :func:`~dnd_audio.determinism.write_atomic`.

**The container is verified by two parsers this project already owns and this module shares
no code with**: :func:`dnd_audio.inspection.riff.read_inventory` walks the chunks, and
:func:`dnd_audio.timeline.pcm.open_pcm` reads the audio back. Agreement between three
independent implementations is much stronger evidence than sharing a table with the writer
would be — the same argument `fixtures/wav.py` makes for keeping its layout separate from
`inspection/riff.py`.

**Nothing but ``fmt `` and ``data``.** No ``bext``, no ``iXML``, no ``LIST``. The marker is a
generated asset with no recording history to describe, and every optional chunk is another
thing whose bytes would have to be frozen and explained. A player that cannot read a
canonical two-chunk mono WAV cannot read anything.

**16-bit rather than 24 or float.** The marker is a playback asset: 96 dB of dynamic range is
far more than a phone speaker and a room will preserve, the file is the payload embedded in an
HTML page the operator carries on a phone, and ``pcm_s16le`` converts to float32 exactly
(ADR-0030), so the detector's templates are the played bytes without a lossy step.
"""

from __future__ import annotations

import struct
from typing import Final

import numpy as np
import numpy.typing as npt

from dnd_audio.marker import MARKER_SAMPLE_RATE
from dnd_audio.marker.spec import MarkerSpec
from dnd_audio.marker.synth import marker_samples

__all__ = ["BITS_PER_SAMPLE", "CHANNELS", "SAMPLE_FORMAT", "marker_wav_bytes"]

#: One transmitter records one channel and one phone speaker plays one; a stereo marker would
#: be two copies of the same thing and twice the payload.
CHANNELS: Final = 1

BITS_PER_SAMPLE: Final = 16

#: FFprobe's name for what this writes, so a manifest and a probe of the file agree.
SAMPLE_FORMAT: Final = "pcm_s16le"

_WAVE_FORMAT_PCM: Final = 1
_BYTES_PER_SAMPLE: Final = BITS_PER_SAMPLE // 8


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """One RIFF chunk: four-byte id, little-endian 32-bit size, payload, padded to even.

    The pad byte is **not** counted in the size field. Getting that wrong shifts every
    later chunk's offset by one, which is why `inspection/riff.py` has an odd-size test and
    why this file is read back through it.
    """
    body = chunk_id + struct.pack("<I", len(payload)) + payload
    return body + b"\x00" if len(payload) % 2 else body


def marker_wav_bytes(spec: MarkerSpec) -> bytes:
    """The complete WAV file for ``spec``, as bytes.

    Returns bytes rather than writing, for two reasons that both matter: the builder writes
    them atomically through :func:`~dnd_audio.determinism.write_atomic`, and the page embeds
    **these exact bytes** rather than a second encoding of the same samples. One array, one
    container, one payload — which is what makes "the WAV extracted from the page is
    byte-identical to the CLI's WAV" true by construction rather than by testing.
    """
    samples: npt.NDArray[np.int32] = marker_samples(spec)
    if int(np.abs(samples).max(initial=0)) >= 1 << (BITS_PER_SAMPLE - 1):
        # Unreachable while `MarkerSpec` validates `peak_amplitude`, and checked anyway:
        # silently wrapping a sample would put a full-scale discontinuity in the middle of a
        # chirp, and the resulting file would still play.
        message = (
            f"a sample reaches {int(np.abs(samples).max())}, which does not fit in "
            f"{BITS_PER_SAMPLE}-bit PCM"
        )
        raise ValueError(message)

    frames = samples.astype("<i2").tobytes()
    byte_rate = MARKER_SAMPLE_RATE * CHANNELS * _BYTES_PER_SAMPLE
    block_align = CHANNELS * _BYTES_PER_SAMPLE
    fmt = struct.pack(
        "<HHIIHH",
        _WAVE_FORMAT_PCM,
        CHANNELS,
        MARKER_SAMPLE_RATE,
        byte_rate,
        block_align,
        BITS_PER_SAMPLE,
    )

    body = _chunk(b"fmt ", fmt) + _chunk(b"data", frames)
    return b"RIFF" + struct.pack("<I", len(b"WAVE") + len(body)) + b"WAVE" + body
