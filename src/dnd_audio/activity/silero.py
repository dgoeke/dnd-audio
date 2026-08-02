"""Silero VAD driven directly, through ONNX Runtime, with no Torch in the process.

The `silero-vad` package hard-depends on Torch and torchaudio, which INV-05 keeps out of the
environment the default suite runs in and which M6a intends to source from AMD's index rather
than from PyPI. Driving the model needs none of it: the ONNX artifact is 2.3 MB, its call
protocol is three inputs and two outputs, and this module is the whole of what the package
would have provided. ADR-0013 records that decision, the measurements behind it, and the
alternatives it rejects.

**The model is recurrent, and the protocol it sits behind is not.**
:meth:`~dnd_audio.interfaces.ActivityDetector.detect` takes a window and returns spans, which
reads as a pure function of that window. It is not: 128 numbers of recurrent state and 64
samples of context cross every frame boundary, window boundaries included. Two ways of using
a stateless-looking detector are therefore wrong, and both produce *plausible* output —

* one instance shared across tracks leaks one speaker's state into another's audio;
* one instance rebuilt per window makes the answer depend on how the reader happened to
  partition the track.

Neither shows up as a crash or as an obviously bad number, so neither is left to convention:
an instance belongs to one track, windows must arrive in order and contiguously, and a
violation **raises**. ADR-0013 chose that over extending the M0 protocol with a stream
lifecycle, because two later milestones already type against that protocol and an assertion
expresses the constraint just as precisely.

**Everything the model needs to answer is in the identity, not just its weights.** A future
release with the same name and a 256-sample frame would otherwise answer differently under an
unchanged cache key. So :class:`DetectorInterface` — frame size, context size, state shape,
input names, sample rate — is recorded beside the model's sha256, and it is not decoration:
this adapter is *driven* by those fields, so a changed interface is a changed call and a
changed cache key at once (INV-08).

**The runtime import is lazy on purpose.** Importing this module must stay free for the
default suite, which never loads a model (INV-05); ``onnxruntime`` is imported inside
:func:`load_silero_session` and nowhere else. Every offline test drives this production code
path through the :class:`OnnxSession` seam instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from dnd_audio.activity import DETECTOR_CONTEXT_SAMPLES, DETECTOR_FRAME_SAMPLES
from dnd_audio.activity.detect import PERMILLE
from dnd_audio.artifacts.activity import DetectorIdentity, DetectorInterface
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import DndAudioError
from dnd_audio.interfaces import AudioWindow, SpeechSpan
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE

if TYPE_CHECKING:  # The runner is this module's caller; see `silero_bundle`.
    from dnd_audio.activity.runner import DetectorBundle

__all__ = [
    "CPU_EXECUTION_PROVIDER",
    "DETECTOR_MISUSED",
    "MODEL_HASH_MISMATCH",
    "MODEL_UNAVAILABLE",
    "ONNX_RUNTIME_NAME",
    "SILERO_DETECTOR_NAME",
    "SILERO_INTERFACE",
    "OnnxSession",
    "SileroActivityDetector",
    "SileroError",
    "SileroIdentity",
    "detector_identity",
    "load_silero_session",
    "onnxruntime_version",
    "silero_bundle",
    "silero_factory",
]

#: Stable machine-readable codes (see :mod:`dnd_audio.errors`). The first two deliberately
#: carry the same strings as :mod:`dnd_audio.models`: a caller branching on "the pinned model
#: is not usable" must not have to know whether the store or the adapter noticed it.
MODEL_UNAVAILABLE: Final = "model_unavailable"
MODEL_HASH_MISMATCH: Final = "model_hash_mismatch"

#: A detector was driven in a way its recurrent state cannot survive. A programming error
#: rather than an operator one, which is why it is loud rather than repaired.
DETECTOR_MISUSED: Final = "detector_misused"

SILERO_DETECTOR_NAME: Final = "silero-vad"
ONNX_RUNTIME_NAME: Final = "onnxruntime"

#: CPU, and the reasoning is in ADR-0013: it keeps Torch and ROCm out of the default
#: environment, the GPU's scarce resource during a session is ASR compute, and 2.3 MB over
#: 16 kHz audio does not make the question interesting.
CPU_EXECUTION_PROVIDER: Final = "CPUExecutionProvider"

#: ``(2, batch, 128)``. The batch dimension is part of the shape rather than implied, because
#: this is what gets recorded in the cache key and a reader should not have to infer it.
SILERO_STATE_SHAPE: Final = (2, 1, 128)

#: In call order: the 64 + 512 sample chunk, the recurrent state, the int64 sample rate.
SILERO_INPUT_NAMES: Final = ("input", "state", "sr")

#: How the pinned artifact is called. Frozen, shared, and part of every detection cache key.
SILERO_INTERFACE: Final = DetectorInterface(
    frame_samples=DETECTOR_FRAME_SAMPLES,
    context_samples=DETECTOR_CONTEXT_SAMPLES,
    state_shape=list(SILERO_STATE_SHAPE),
    input_names=list(SILERO_INPUT_NAMES),
    sample_rate=DERIVATIVE_SAMPLE_RATE,
)

#: A probability and the next state. Anything shorter is not the pinned graph.
_EXPECTED_OUTPUTS: Final = 2

#: How far outside [0, 1] a returned probability may sit before the artifact is not the model
#: this adapter thinks it is. A sigmoid output lands inside; a wrong graph does not.
_PROBABILITY_TOLERANCE: Final = 1e-3


class SileroError(DndAudioError):
    """The pinned VAD model cannot be loaded, or a detector was driven in a way it cannot be.

    One type, three conditions a caller genuinely wants to tell apart, so each raise names its
    own code: the artifact is absent, the artifact is not the pinned bytes, or the detector's
    contract was broken. Prose gets reworded; codes do not.
    """

    default_code: ClassVar[str] = "vad_model_unusable"


@runtime_checkable
class OnnxSession(Protocol):
    """The one method this adapter calls on an ONNX Runtime session.

    **This seam is the most important thing in this file.** INV-05 says the default test suite
    loads no model, and INV-10 says a detector's correctness is proved against a deterministic
    fake — but the properties that matter here are properties of *statefulness*: that the
    recurrent state crosses window boundaries, that the context carries, that a track's state
    is its own. None of those can be proved by a fake detector, because a fake detector
    replaces the very code that carries the state. They can only be proved by driving this
    module's real frame loop against a fake *session*.

    So the seam sits one level lower than the obvious place. A test supplies a stateful fake
    session, the production adapter runs unchanged, and partition invariance and cross-track
    isolation become offline assertions about the code that actually ships.

    Parameters are positional-only so that conformance does not depend on ``onnxruntime``'s
    argument names, which are not part of any contract we pinned.
    """

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, npt.NDArray[Any]],
        /,
    ) -> Sequence[npt.NDArray[Any]]: ...


@dataclass(frozen=True, slots=True)
class SileroIdentity:
    """Everything except the call protocol that decides what this detector would answer.

    Separate from :class:`~dnd_audio.artifacts.activity.DetectorIdentity` because that one is
    an artifact record and this one is a constructor argument: the interface half is the
    adapter's to state, not the caller's, and a caller able to declare an interface it is not
    being run under could write a cache key that is not true.
    """

    release: str | None = None
    commit: str | None = None
    model_sha256: str | None = None
    runtime_version: str | None = None
    name: str = SILERO_DETECTOR_NAME
    runtime: str = ONNX_RUNTIME_NAME
    execution_provider: str = CPU_EXECUTION_PROVIDER


def onnxruntime_version() -> str:
    """The installed runtime's version, imported lazily for the same reason as the session.

    Part of the detection cache key: "reproducible inference" is the artifact *and* the thing
    executing it (ADR-0013), and a runtime upgrade that changed a kernel's rounding would
    otherwise serve the old answers under the new numbers.
    """
    import onnxruntime  # Lazy on purpose; see the module docstring.

    version: str = onnxruntime.__version__
    return version


def load_silero_session(path: Path, *, expected_sha256: str | None = None) -> OnnxSession:
    """Open the pinned ONNX artifact on the CPU execution provider.

    The hash is verified here as well as in :mod:`dnd_audio.models`, and deliberately: the
    store checks what it fetched, this checks what it is about to *execute*. A file replaced
    between the two would otherwise reach a session, and a model whose bytes we cannot vouch
    for produces a transcript that is slightly wrong rather than an error (ADR-0013). Both
    checks happen before ``onnxruntime`` is imported, so an absent or wrong artifact fails the
    same way on a machine that has never loaded a model.

    Single-threaded on both pools: this runs six times over a session that is hours long, and
    a thread pool buys a fraction of a second per track in exchange for scheduling
    nondeterminism in the one number every downstream decision is made from.

    Args:
        path: The model file. :func:`dnd_audio.models.find_model` is how a caller gets one.
        expected_sha256: The pinned digest, lowercase hex. ``None`` skips the check, which is
            for a caller that has already verified these exact bytes.

    Raises:
        SileroError: The file is absent, or its contents are not what was pinned.
    """
    if not path.is_file():
        message = (
            f"the voice-activity model is not at {path}. Run `dnd-audio models fetch` — "
            f"detection is the one stage that cannot proceed without it."
        )
        raise SileroError(message, code=MODEL_UNAVAILABLE)

    if expected_sha256 is not None:
        found = sha256_file(path)
        if found != expected_sha256:
            message = (
                f"{path} hashes to {found}, not the pinned {expected_sha256}. The pin is what "
                f"makes an answer reproducible, so the file is refused rather than run; "
                f"delete it and re-run `dnd-audio models fetch`."
            )
            raise SileroError(message, code=MODEL_HASH_MISMATCH)

    import onnxruntime  # Lazy on purpose; see the module docstring.

    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session: OnnxSession = onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=[CPU_EXECUTION_PROVIDER]
    )
    return session


def silero_factory(
    session: OnnxSession,
    *,
    identity: SileroIdentity,
    silence_threshold: float,
    interface: DetectorInterface = SILERO_INTERFACE,
) -> Callable[[str], SileroActivityDetector]:
    """A per-track detector builder over one loaded session.

    This is the shape :func:`~dnd_audio.activity.detect.detect_track`'s caller needs: the
    runner loads the model once and calls this once per track, which is what makes "one
    instance per track" the easy thing to do rather than a rule to remember.

    **Sharing one ``InferenceSession`` across the six detectors is safe**, and that is a fact
    about where the state lives, not an assumption about the runtime. Everything recurrent —
    the 128-wide state and the 64 samples of context — is held by the wrapper and passed in
    on every call; the session itself is a loaded graph that carries nothing between ``run``
    calls. It matters because the alternative is six loads of the same 2.3 MB artifact, and
    because a reader who assumed otherwise would "fix" this by rebuilding the session per
    track and conclude that the state was safe when it was merely reset.
    """

    def build(track_id: str) -> SileroActivityDetector:
        return SileroActivityDetector(
            session,
            track_id=track_id,
            identity=identity,
            silence_threshold=silence_threshold,
            interface=interface,
        )

    return build


def detector_identity(
    identity: SileroIdentity, interface: DetectorInterface = SILERO_INTERFACE
) -> DetectorIdentity:
    """The artifact record for a detector built from these two halves.

    One function, so the identity a cache key is computed from *before* any detector exists
    and the identity a detector reports afterwards cannot drift apart (INV-08).
    """
    return DetectorIdentity(
        name=identity.name,
        release=identity.release,
        commit=identity.commit,
        model_sha256=identity.model_sha256,
        runtime=identity.runtime,
        runtime_version=identity.runtime_version,
        execution_provider=identity.execution_provider,
        interface=interface,
        variant_digest=None,
    )


def silero_bundle(*, silence_threshold: float, directory: Path | None = None) -> DetectorBundle:
    """The pinned detector, resolved from the local model store, ready for the runner.

    Everything that can fail does so here, at the point detection starts, rather than at
    import: an absent or wrong-hashed artifact raises
    :class:`~dnd_audio.models.ModelError`, which the runner turns into a failed stage and a
    nonzero exit (INV-13). One session is loaded and shared by the per-track detectors that
    :func:`silero_factory` builds over it — see that function for why sharing is safe.

    Both imports are deferred: ``dnd_audio.models`` because nothing about this module should
    require a model store to exist, and the runner's :class:`DetectorBundle` because the
    runner is this module's *caller*. A top-level import in the other direction would be a
    cycle waiting for someone to add one line to either file.
    """
    from dnd_audio.activity.runner import DetectorBundle
    from dnd_audio.models import SILERO_VAD, require_model

    path = require_model(SILERO_VAD, directory=directory)
    version = onnxruntime_version()
    identity = SileroIdentity(
        release=SILERO_VAD.release,
        commit=SILERO_VAD.commit,
        model_sha256=SILERO_VAD.sha256,
        runtime_version=version,
    )
    # Verified again, against the bytes about to be executed rather than against the bytes
    # the store found. `require_model` already hashed them; this costs 2.3 MB of reading and
    # removes the window between the two.
    session = load_silero_session(path, expected_sha256=SILERO_VAD.sha256)
    return DetectorBundle(
        identity=detector_identity(identity),
        make=silero_factory(session, identity=identity, silence_threshold=silence_threshold),
        runtime_version=version,
    )


class SileroActivityDetector:
    """One track's Silero detector: recurrent, ordered, and loud when it is misused.

    Implements :class:`~dnd_audio.interfaces.ActivityDetector` and the optional
    :class:`~dnd_audio.activity.detect.FrameProbabilities`, so
    :func:`~dnd_audio.activity.detect.detect_track` gets the measured per-frame probabilities
    rather than a reconstruction rasterized from spans.

    ADR-0013, on why the two constraints below raise instead of resetting:

        The model is recurrent: it carries a 128-wide state and 64 samples of context between
        frames. ``ActivityDetector.detect(window)`` looks stateless, which is how independent
        review found this: reusing one instance across tracks leaks one speaker's state into
        another, and rebuilding it per window makes the answer depend on the window
        partitioning. […] A loud failure beats an inferred convention.

    Args:
        session: A loaded ONNX session. May be shared with this track's siblings — see
            :func:`silero_factory`.
        track_id: The one track this instance may ever be handed.
        identity: The pinned artifact and the runtime executing it.
        silence_threshold: Fractional, and the same number
            :class:`~dnd_audio.config.VadConfig` gives the region assembler. Returned spans
            cover frames at or above it, so the spans and the assembler's hysteresis close on
            the same value rather than on two that drifted apart.
        interface: How the model is called. Not a knob — it is the model's protocol, and it is
            a parameter only because it is *used* here as well as recorded, so a test that
            changes it changes both the call and the cache key.
    """

    def __init__(
        self,
        session: OnnxSession,
        *,
        track_id: str,
        identity: SileroIdentity,
        silence_threshold: float,
        interface: DetectorInterface = SILERO_INTERFACE,
    ) -> None:
        if len(interface.input_names) != len(SILERO_INPUT_NAMES):
            message = (
                f"the detector interface names {len(interface.input_names)} model inputs; the "
                f"call protocol has {len(SILERO_INPUT_NAMES)} — the chunk, the state, and the "
                f"sample rate"
            )
            raise SileroError(message, code=DETECTOR_MISUSED)
        if not 0.0 < silence_threshold < 1.0:
            message = (
                f"silence_threshold must be a probability strictly inside (0, 1), got "
                f"{silence_threshold}"
            )
            raise SileroError(message, code=DETECTOR_MISUSED)

        self._session = session
        self._track_id = track_id
        self._identity = identity
        self._interface = interface
        self._frame = interface.frame_samples
        self._context_samples = interface.context_samples
        self._input_name, self._state_name, self._rate_name = interface.input_names
        self._silence_permille = round(silence_threshold * PERMILLE)

        self._state = np.zeros(tuple(interface.state_shape), dtype=np.float32)
        # Zeros before the first frame — the model's own convention for "nothing preceded
        # this", and why frame zero of a track is reproducible at all.
        self._context = np.zeros(self._context_samples, dtype=np.float32)
        self._probabilities: list[int] = []
        self._next_sample = 0
        self._padded = False

    @property
    def track_id(self) -> str:
        return self._track_id

    def detect(self, window: AudioWindow) -> tuple[SpeechSpan, ...]:
        """Run whole frames of ``window``, carrying state and context out the far side.

        Spans are runs of consecutive frames at or above the silence threshold, each carrying
        that run's mean probability. A run that reaches the end of the window is cut there and
        continues as a new span in the next call: the assembler works from the per-frame
        probabilities and merges across the seam, so stitching here would duplicate its job
        and hide a window boundary the caller may want to see.

        A window that is not a whole number of frames is zero-padded to one — the same rule and
        direction as :func:`~dnd_audio.activity.detect.frame_count` and as the resampler's
        ``ceil`` length — and, because that padding shifts every subsequent frame off the
        track's grid, it may only ever happen once, at the end.

        Raises:
            SileroError: The window belongs to another track, is not at the model's sample
                rate, does not start where the previous one ended, or follows a padded final
                window.
        """
        self._check(window)
        samples = np.ascontiguousarray(window.samples, dtype=np.float32)
        length = int(samples.shape[0])
        self._next_sample = window.start_sample + length

        remainder = length % self._frame
        if remainder:
            samples = np.concatenate([samples, np.zeros(self._frame - remainder, dtype=np.float32)])
            self._padded = True

        first_frame = len(self._probabilities)
        for start in range(0, int(samples.shape[0]), self._frame):
            self._probabilities.append(self._run_frame(samples[start : start + self._frame]))

        return self._spans(
            self._probabilities[first_frame:],
            window_start=window.start_sample,
            window_end=window.end_sample,
        )

    def frame_probabilities(self) -> npt.NDArray[np.uint16]:
        """Per-mille per frame, for every frame seen so far.

        Two bytes per 32 ms — under a megabyte for a four-hour track, which is why the whole
        track's worth is kept while the audio behind it is not (INV-07).
        """
        return np.asarray(self._probabilities, dtype=np.uint16)

    def identity(self) -> DetectorIdentity:
        """What this detector is, completely enough to be a cache key (INV-08)."""
        return detector_identity(self._identity, self._interface)

    def _check(self, window: AudioWindow) -> None:
        if window.track_id != self._track_id:
            message = (
                f"a {self._track_id} detector was handed a {window.track_id} window. One "
                f"instance belongs to one track: this model carries recurrent state, so "
                f"reusing it would decide {window.track_id}'s audio partly from "
                f"{self._track_id}'s voice (ADR-0013)."
            )
            raise SileroError(message, code=DETECTOR_MISUSED)

        if window.sample_rate != self._interface.sample_rate:
            message = (
                f"{self._track_id}: the window is at {window.sample_rate} Hz and this model "
                f"is called at {self._interface.sample_rate} Hz. Detection runs on the "
                f"derivative, never on the working audio."
            )
            raise SileroError(message, code=DETECTOR_MISUSED)

        if self._padded:
            message = (
                f"{self._track_id}: a window arrived after a short one was zero-padded to a "
                f"whole frame. Padding shifts every later frame off the track's frame grid, "
                f"so a short window can only be the last one."
            )
            raise SileroError(message, code=DETECTOR_MISUSED)

        if window.start_sample != self._next_sample:
            direction = "skips" if window.start_sample > self._next_sample else "goes back over"
            message = (
                f"{self._track_id}: this window starts at {window.start_sample} and "
                f"{direction} the {self._next_sample} samples already seen. Windows must "
                f"arrive in order and contiguously — the state and the 64 samples of context "
                f"crossing this boundary are only meaningful if nothing between them was "
                f"dropped or replayed (ADR-0013)."
            )
            raise SileroError(message, code=DETECTOR_MISUSED)

    def _run_frame(self, frame: npt.NDArray[np.float32]) -> int:
        """One frame through the model, and the state and context out the other side."""
        chunk = np.concatenate([self._context, frame]).reshape(1, -1)
        feeds: dict[str, npt.NDArray[Any]] = {
            self._input_name: chunk,
            self._state_name: self._state,
            self._rate_name: np.array(self._interface.sample_rate, dtype=np.int64),
        }
        outputs = self._session.run(None, feeds)
        if len(outputs) < _EXPECTED_OUTPUTS:
            message = (
                f"the model returned {len(outputs)} outputs; the pinned interface returns a "
                f"probability and the next state"
            )
            raise SileroError(message)

        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not -_PROBABILITY_TOLERANCE <= probability <= 1.0 + _PROBABILITY_TOLERANCE:
            message = (
                f"the model returned {probability}, which is not a probability. Clamping it "
                f"would turn a wrong artifact into a plausible transcript."
            )
            raise SileroError(message)

        state = np.asarray(outputs[1], dtype=np.float32)
        if state.shape != self._state.shape:
            message = (
                f"the model returned a state of shape {state.shape} where the pinned "
                f"interface declares {self._state.shape}"
            )
            raise SileroError(message)

        self._state = state
        # The *last* 64 samples of this frame, which is what the next frame's leading context
        # is defined to be. Taking the first 64 instead would still run, and would feed the
        # model a 32 ms hole at every frame boundary.
        self._context = (
            frame[-self._context_samples :].copy()
            if self._context_samples
            else np.zeros(0, dtype=np.float32)
        )
        # `rint`, matching `rasterize_spans`: one rounding rule for every per-mille in this
        # package, so a measured probability and a rasterized one round the same way.
        return int(np.clip(np.rint(probability * PERMILLE), 0, PERMILLE))

    def _spans(
        self, values: Sequence[int], *, window_start: int, window_end: int
    ) -> tuple[SpeechSpan, ...]:
        found: list[SpeechSpan] = []
        run: list[int] = []
        run_start = 0
        for index, value in enumerate(values):
            if value >= self._silence_permille:
                if not run:
                    run_start = index
                run.append(value)
                continue
            if run:
                found.append(self._span(run_start, run, window_start, window_end))
                run = []
        if run:
            found.append(self._span(run_start, run, window_start, window_end))
        return tuple(found)

    def _span(
        self, run_start: int, run: Sequence[int], window_start: int, window_end: int
    ) -> SpeechSpan:
        start = window_start + run_start * self._frame
        # Clipped to the window: the final frame of a short window is zero-padded, and a span
        # claiming samples the track does not have would place a candidate past its own end.
        end = min(window_start + (run_start + len(run)) * self._frame, window_end)
        mean = int(np.rint(float(np.mean(run))))
        return SpeechSpan(
            start_sample=start,
            end_sample=end,
            probability=mean / PERMILLE,
            details={"peak_probability": max(run) / PERMILLE, "frames": float(len(run))},
        )
