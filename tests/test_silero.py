"""The recurrent detector, proved offline against a fake session (ADR-0013).

The properties this file exists to hold are properties of *carried state*, and none of them
can be reached with a fake detector: a fake detector replaces the very loop that carries the
state. So every offline test here drives the production adapter — the real frame loop, the
real context arithmetic, the real span assembly — against a fake `OnnxSession`. That seam is
what makes "partition invariance" and "one track's state is its own" assertions about the code
that ships rather than about a stand-in (INV-05, INV-10).

**The fake is stateless and the detector is not, on purpose.** `StatefulFakeSession` is a pure
function of the chunk it is given *and* the state it is handed, and it threads a new state
out. It therefore has no memory of its own, which means every test below that observes memory
is observing the adapter's. A fake that remembered things itself would pass all of this with
the adapter's state carry deleted.

Two failure modes are named repeatedly because they are the ones that produce plausible
output rather than errors: resetting the recurrent state at a window boundary, and dropping
the 64 samples of context between frames. Neither changes a length, a type, or a shape.

The `host_smoke` test is the only one that loads the pinned model. It asserts what is true of
*the model* — probabilities in range, the frame count, a loud broadband burst above digital
silence — and never that synthetic noise reads as speech, which INV-10 forbids.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from dnd_audio.activity import DETECTOR_CONTEXT_SAMPLES, DETECTOR_FRAME_SAMPLES
from dnd_audio.activity.detect import PERMILLE, FrameProbabilities, detect_track, frame_count
from dnd_audio.activity.silero import (
    CPU_EXECUTION_PROVIDER,
    DETECTOR_MISUSED,
    MODEL_HASH_MISMATCH,
    MODEL_UNAVAILABLE,
    ONNX_RUNTIME_NAME,
    SILERO_DETECTOR_NAME,
    SILERO_INTERFACE,
    OnnxSession,
    SileroActivityDetector,
    SileroError,
    SileroIdentity,
    load_silero_session,
    onnxruntime_version,
    silero_bundle,
    silero_factory,
)
from dnd_audio.artifacts.activity import DetectorInterface
from dnd_audio.config import VadConfig
from dnd_audio.interfaces import ActivityDetector, AudioWindow, SpeechSpan
from dnd_audio.timeline import DERIVATIVE_SAMPLE_RATE
from dnd_audio.timeline.wavwrite import WavWriter

FRAME = DETECTOR_FRAME_SAMPLES
CONTEXT = DETECTOR_CONTEXT_SAMPLES
RATE = DERIVATIVE_SAMPLE_RATE
TRACK = "tx-a"

#: A plausible identity, complete enough that every field of the record has a value to fail
#: on. The digest is a real sha256 of nothing in particular — the artifact validates its
#: shape, and no offline test is allowed to depend on the real model's bytes.
IDENTITY = SileroIdentity(
    release="v6.2.1",
    commit="7e30209a3e901f9842f81b225f3e93d8199902b1",
    model_sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    runtime_version="1.28.0",
)


class StatefulFakeSession:
    """A deterministic ONNX session whose answer depends on the state it is handed.

    ``probability = (level + carried) / 2``, where ``level`` is the mean magnitude of the
    576-sample chunk — context included, so a dropped context changes the number as well as
    the feed — and ``carried`` is the first element of the incoming state. The same value goes
    back out as the new state's first element, so the probability of frame *n* depends on
    every frame before it. That is the shape a recurrent model has, reduced to arithmetic a
    test can predict: reset the state and the answer changes; skip a frame and the answer
    changes.

    Every feed is recorded, deep-copied, so a test can assert on what the model was *asked*
    rather than only on what came back — which is the only way to prove the context carry,
    since a wrong context is still a well-formed input of the right shape.
    """

    def __init__(self, *, input_names: Sequence[str] = ("input", "state", "sr")) -> None:
        self._input, self._state, self._rate = input_names
        self.feeds: list[dict[str, npt.NDArray[Any]]] = []

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, npt.NDArray[Any]],
        /,
    ) -> Sequence[npt.NDArray[Any]]:
        expected = {self._input, self._state, self._rate}
        if set(input_feed) != expected:
            message = f"fed {sorted(input_feed)}, this graph takes {sorted(expected)}"
            raise KeyError(message)
        self.feeds.append({name: np.array(value, copy=True) for name, value in input_feed.items()})

        chunk = np.asarray(input_feed[self._input], dtype=np.float64)
        state = np.asarray(input_feed[self._state], dtype=np.float64)
        level = float(np.clip(np.abs(chunk).mean(), 0.0, 1.0))
        carried = float(state.reshape(-1)[0])
        probability = float(np.clip((level + carried) / 2.0, 0.0, 1.0))

        next_state = np.zeros(state.shape, dtype=np.float32)
        next_state.flat[0] = probability
        return [np.array([[probability]], dtype=np.float32), next_state]


class BrokenSession:
    """Returns something that is not the pinned graph's answer. One knob per way to be wrong."""

    def __init__(
        self,
        *,
        probability: float = 0.5,
        outputs: int = 2,
        state_shape: tuple[int, ...] = (2, 1, 128),
    ) -> None:
        self._probability = probability
        self._outputs = outputs
        self._state_shape = state_shape

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, npt.NDArray[Any]],
        /,
    ) -> Sequence[npt.NDArray[Any]]:
        produced = [
            np.array([[self._probability]], dtype=np.float32),
            np.zeros(self._state_shape, dtype=np.float32),
        ]
        return produced[: self._outputs]


def build(
    session: OnnxSession,
    *,
    track_id: str = TRACK,
    silence_threshold: float = 0.35,
    interface: DetectorInterface = SILERO_INTERFACE,
) -> SileroActivityDetector:
    return SileroActivityDetector(
        session,
        track_id=track_id,
        identity=IDENTITY,
        silence_threshold=silence_threshold,
        interface=interface,
    )


def noise(n_samples: int, *, seed: int = 7) -> npt.NDArray[np.float32]:
    """Bounded pseudo-random audio. Content is irrelevant here; reproducibility is not."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.standard_normal(n_samples) * 0.3, -1.0, 1.0).astype(np.float32)


def speech_shaped(n_samples: int) -> npt.NDArray[np.float32]:
    """Loud where the fake will score high, silent where it will decay.

    Named for what it does to the *fake*, never for what a real detector would call it: the
    default suite may not assert that synthetic audio triggers a learned model (INV-10).
    """
    samples = np.zeros(n_samples, dtype=np.float32)
    start = RATE // 2
    length = min(RATE, max(n_samples - start, 0))
    half_period = 32
    square = np.where(np.arange(length) % (half_period * 2) < half_period, 0.9, -0.9)
    samples[start : start + length] = square.astype(np.float32)
    return samples


def push(
    detector: SileroActivityDetector, samples: npt.NDArray[np.float32], *, frames_per_window: int
) -> list[tuple[SpeechSpan, ...]]:
    """Feed ``samples`` to ``detector`` in contiguous windows of whole frames."""
    step = frames_per_window * FRAME
    returned: list[tuple[SpeechSpan, ...]] = []
    for start in range(0, int(samples.shape[0]), step):
        window = AudioWindow(
            track_id=detector.track_id,
            sample_rate=RATE,
            start_sample=start,
            samples=samples[start : start + step],
        )
        returned.append(detector.detect(window))
    return returned


def covered(spans: Sequence[SpeechSpan]) -> set[int]:
    """Every derivative sample any span claims. Partition-independent by construction."""
    claimed: set[int] = set()
    for span in spans:
        claimed.update(range(span.start_sample, span.end_sample))
    return claimed


class TestTheSeam:
    def test_the_detector_is_both_protocols(self) -> None:
        """`detect_track` branches on the second one to keep measurements over rasterization."""
        detector = build(StatefulFakeSession())
        assert isinstance(detector, ActivityDetector)
        assert isinstance(detector, FrameProbabilities)

    def test_the_factory_builds_one_detector_per_track_over_one_session(self) -> None:
        """The runner's shape: load once, and let the per-track instance hold the state."""
        session = StatefulFakeSession()
        factory = silero_factory(session, identity=IDENTITY, silence_threshold=0.35)
        first, second = factory("tx-a"), factory("tx-b")

        assert (first.track_id, second.track_id) == ("tx-a", "tx-b")
        assert first is not second
        push(first, noise(FRAME * 4), frames_per_window=2)
        # The shared session carried nothing: tx-b starts from zeros, as a fresh track must.
        assert second.frame_probabilities().shape == (0,)
        push(second, noise(FRAME * 4), frames_per_window=2)
        assert np.array_equal(first.frame_probabilities(), second.frame_probabilities())


class TestPartitionInvariance:
    @pytest.mark.parametrize("frames_per_window", [1, 2, 5, 7])
    def test_every_window_partitioning_gives_identical_probabilities(
        self, frames_per_window: int
    ) -> None:
        """Byte-identical, not close.

        This is the test a per-window state reset fails. It would still be "close" — the error
        lives at the window boundaries and decays — so equality is what tells the two apart.
        Window sizes 5 and 7 do not divide the 40-frame signal, so the last window is short in
        frames as well.
        """
        samples = noise(FRAME * 40)
        reference = build(StatefulFakeSession())
        push(reference, samples, frames_per_window=40)

        detector = build(StatefulFakeSession())
        push(detector, samples, frames_per_window=frames_per_window)
        assert np.array_equal(detector.frame_probabilities(), reference.frame_probabilities())

    def test_the_spans_of_a_partitioned_run_cover_the_same_frames(self) -> None:
        """Spans are cut at window edges; the frames they cover are not."""
        samples = speech_shaped(RATE * 2)

        whole = build(StatefulFakeSession())
        one_shot = [span for spans in push(whole, samples, frames_per_window=64) for span in spans]
        split = build(StatefulFakeSession())
        pieces = [span for spans in push(split, samples, frames_per_window=3) for span in spans]

        assert covered(one_shot) == covered(pieces)
        assert len(pieces) > len(one_shot)


class TestStateIsThreaded:
    def test_a_reset_between_windows_would_change_the_answer(self) -> None:
        """The falsification, stated as a contrast.

        A detector rebuilt for the second window is exactly the bug ADR-0013 describes. Here
        both answers are computed: the streamed one matches the whole-track one, the rebuilt
        one does not. Without the second half this test would pass against an implementation
        that carried no state at all.
        """
        samples = noise(FRAME * 8)
        streamed = build(StatefulFakeSession())
        push(streamed, samples, frames_per_window=4)

        whole = build(StatefulFakeSession())
        push(whole, samples, frames_per_window=8)

        rebuilt = build(StatefulFakeSession())
        push(rebuilt, samples[: FRAME * 4], frames_per_window=4)
        second = build(StatefulFakeSession())
        push(second, samples[FRAME * 4 :], frames_per_window=4)
        restarted = np.concatenate([rebuilt.frame_probabilities(), second.frame_probabilities()])

        assert np.array_equal(streamed.frame_probabilities(), whole.frame_probabilities())
        assert not np.array_equal(streamed.frame_probabilities(), restarted)

    def test_the_state_handed_to_each_frame_is_the_one_the_last_frame_returned(self) -> None:
        """Asserted on the feeds, because a state that is merely *present* proves nothing."""
        session = StatefulFakeSession()
        detector = build(session)
        push(detector, noise(FRAME * 6), frames_per_window=2)

        assert not session.feeds[0]["state"].any()
        probabilities = detector.frame_probabilities().tolist()
        for index, feed in enumerate(session.feeds[1:]):
            # The fake puts the previous frame's probability in the state's first element.
            carried = float(feed["state"].reshape(-1)[0])
            assert int(np.rint(carried * PERMILLE)) == probabilities[index]

    def test_state_does_not_cross_a_window_boundary_as_zeros(self) -> None:
        """The frame that opens a window is the one a reset would silence."""
        session = StatefulFakeSession()
        detector = build(session)
        push(detector, noise(FRAME * 4), frames_per_window=2)
        assert session.feeds[2]["state"].any()


class TestContextIsCarried:
    def test_each_frame_is_prefixed_with_the_previous_frames_last_64_samples(self) -> None:
        """Asserted on the feeds: a wrong context is a well-formed input of the right shape.

        The tempting wrong version is the *first* 64 samples of the previous frame, or of this
        one. Both run, both produce probabilities, and both feed the model a 32 ms hole at
        every frame boundary.
        """
        samples = noise(FRAME * 5)
        session = StatefulFakeSession()
        detector = build(session)
        push(detector, samples, frames_per_window=2)

        assert len(session.feeds) == 5
        assert not session.feeds[0]["input"][0, :CONTEXT].any()
        for index, feed in enumerate(session.feeds):
            chunk = feed["input"]
            assert chunk.shape == (1, CONTEXT + FRAME)
            assert np.array_equal(chunk[0, CONTEXT:], samples[index * FRAME : (index + 1) * FRAME])
            if index:
                previous = samples[index * FRAME - CONTEXT : index * FRAME]
                assert np.array_equal(chunk[0, :CONTEXT], previous)

    def test_the_context_crosses_a_window_boundary(self) -> None:
        """Stated separately: a per-window context reset passes the test above."""
        samples = noise(FRAME * 4)
        session = StatefulFakeSession()
        detector = build(session)
        push(detector, samples, frames_per_window=2)

        boundary = session.feeds[2]["input"][0, :CONTEXT]
        assert np.array_equal(boundary, samples[2 * FRAME - CONTEXT : 2 * FRAME])
        assert boundary.any()


class TestMisuse:
    def test_a_window_from_another_track_raises(self) -> None:
        """Recurrent state makes this a silent attribution bug, so it is a loud one instead."""
        detector = build(StatefulFakeSession(), track_id="tx-a")
        window = AudioWindow(
            track_id="tx-b", sample_rate=RATE, start_sample=0, samples=noise(FRAME)
        )
        with pytest.raises(SileroError, match="tx-b window") as raised:
            detector.detect(window)
        assert raised.value.code == DETECTOR_MISUSED

    def test_a_window_that_skips_samples_raises(self) -> None:
        detector = build(StatefulFakeSession())
        push(detector, noise(FRAME * 2), frames_per_window=2)
        window = AudioWindow(
            track_id=TRACK, sample_rate=RATE, start_sample=FRAME * 3, samples=noise(FRAME)
        )
        with pytest.raises(SileroError, match="skips") as raised:
            detector.detect(window)
        assert raised.value.code == DETECTOR_MISUSED

    def test_a_window_that_goes_backwards_raises(self) -> None:
        detector = build(StatefulFakeSession())
        push(detector, noise(FRAME * 2), frames_per_window=2)
        window = AudioWindow(
            track_id=TRACK, sample_rate=RATE, start_sample=FRAME, samples=noise(FRAME)
        )
        with pytest.raises(SileroError, match="goes back over") as raised:
            detector.detect(window)
        assert raised.value.code == DETECTOR_MISUSED

    def test_a_window_at_the_wrong_sample_rate_raises(self) -> None:
        """Detection runs on the 16 kHz derivative; the working path is 48 kHz."""
        detector = build(StatefulFakeSession())
        window = AudioWindow(
            track_id=TRACK, sample_rate=48000, start_sample=0, samples=noise(FRAME)
        )
        with pytest.raises(SileroError, match="48000 Hz"):
            detector.detect(window)

    def test_a_window_after_a_padded_one_raises(self) -> None:
        """Padding shifts the grid, so a short window can only ever be the last."""
        detector = build(StatefulFakeSession())
        detector.detect(
            AudioWindow(track_id=TRACK, sample_rate=RATE, start_sample=0, samples=noise(100))
        )
        with pytest.raises(SileroError, match="zero-padded") as raised:
            detector.detect(
                AudioWindow(track_id=TRACK, sample_rate=RATE, start_sample=100, samples=noise(100))
            )
        assert raised.value.code == DETECTOR_MISUSED

    def test_an_interface_naming_the_wrong_number_of_inputs_raises(self) -> None:
        interface = SILERO_INTERFACE.model_copy(update={"input_names": ["input", "state"]})
        with pytest.raises(SileroError, match="call protocol"):
            build(StatefulFakeSession(), interface=interface)

    def test_a_threshold_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(SileroError, match="probability"):
            build(StatefulFakeSession(), silence_threshold=1.5)


class TestAModelThatIsNotTheModel:
    """The artifact is refused rather than believed. Clamping would produce a plausible run."""

    def test_a_probability_outside_the_unit_interval_raises(self) -> None:
        detector = build(BrokenSession(probability=4.2))
        with pytest.raises(SileroError, match="not a probability"):
            push(detector, noise(FRAME), frames_per_window=1)

    def test_a_missing_state_output_raises(self) -> None:
        detector = build(BrokenSession(outputs=1))
        with pytest.raises(SileroError, match="outputs"):
            push(detector, noise(FRAME), frames_per_window=1)

    def test_a_state_of_the_wrong_shape_raises(self) -> None:
        detector = build(BrokenSession(state_shape=(2, 1, 64)))
        with pytest.raises(SileroError, match="state of shape"):
            push(detector, noise(FRAME), frames_per_window=1)


class TestTheFinalWindow:
    @pytest.mark.parametrize("n_samples", [1, 100, FRAME - 1, FRAME + 1, FRAME * 3 + 100])
    def test_a_short_final_window_is_padded_to_a_whole_frame(self, n_samples: int) -> None:
        """`detect_track` rejects any other length, and rightly: see its own message."""
        detector = build(StatefulFakeSession())
        push(detector, noise(n_samples), frames_per_window=4)
        assert detector.frame_probabilities().shape[0] == frame_count(n_samples)

    def test_the_padding_is_zeros_rather_than_a_repeat(self) -> None:
        """A repeated tail would score the same as real audio and extend every last word."""
        session = StatefulFakeSession()
        detector = build(session)
        samples = noise(FRAME + 100)
        push(detector, samples, frames_per_window=4)

        tail = session.feeds[-1]["input"][0, CONTEXT:]
        assert np.array_equal(tail[:100], samples[FRAME:])
        assert not tail[100:].any()

    def test_a_span_never_claims_samples_the_track_does_not_have(self) -> None:
        """The padded frame is 512 samples long; the audio behind it is not."""
        detector = build(StatefulFakeSession(), silence_threshold=0.001)
        samples = speech_shaped(RATE // 2 + 700)
        spans = [span for group in push(detector, samples, frames_per_window=8) for span in group]
        assert spans
        assert max(span.end_sample for span in spans) <= int(samples.shape[0])

    def test_an_empty_window_is_a_no_op(self) -> None:
        detector = build(StatefulFakeSession())
        window = AudioWindow(
            track_id=TRACK, sample_rate=RATE, start_sample=0, samples=np.zeros(0, dtype=np.float32)
        )
        assert detector.detect(window) == ()
        assert detector.frame_probabilities().shape == (0,)


class TestSpans:
    def test_a_span_covers_the_run_of_frames_at_or_above_the_threshold(self) -> None:
        detector = build(BrokenSession(probability=0.6), silence_threshold=0.35)
        spans = push(detector, noise(FRAME * 3), frames_per_window=3)[0]
        assert len(spans) == 1
        assert (spans[0].start_sample, spans[0].end_sample) == (0, FRAME * 3)
        assert spans[0].probability == pytest.approx(0.6)

    def test_frames_below_the_threshold_produce_no_span(self) -> None:
        detector = build(BrokenSession(probability=0.2), silence_threshold=0.35)
        assert push(detector, noise(FRAME * 3), frames_per_window=3)[0] == ()

    def test_the_threshold_is_the_one_it_was_given(self) -> None:
        """The same number the assembler's hysteresis closes on, not a second opinion."""
        lenient = build(BrokenSession(probability=0.2), silence_threshold=0.1)
        strict = build(BrokenSession(probability=0.2), silence_threshold=0.35)
        assert push(lenient, noise(FRAME), frames_per_window=1)[0]
        assert not push(strict, noise(FRAME), frames_per_window=1)[0]


class TestIdentity:
    def test_every_field_is_populated(self) -> None:
        identity = build(StatefulFakeSession()).identity()
        assert identity.name == SILERO_DETECTOR_NAME
        assert identity.release == "v6.2.1"
        assert identity.commit == "7e30209a3e901f9842f81b225f3e93d8199902b1"
        assert identity.model_sha256 == IDENTITY.model_sha256
        assert identity.runtime == ONNX_RUNTIME_NAME
        assert identity.runtime_version == "1.28.0"
        assert identity.execution_provider == CPU_EXECUTION_PROVIDER
        assert identity.variant_digest is None
        assert identity.interface == DetectorInterface(
            frame_samples=FRAME,
            context_samples=CONTEXT,
            state_shape=[2, 1, 128],
            input_names=["input", "state", "sr"],
            sample_rate=RATE,
        )

    @pytest.mark.parametrize(
        "change",
        [
            {"frame_samples": 256},
            {"context_samples": 32},
            {"state_shape": [2, 1, 64]},
            {"input_names": ["chunk", "state", "sr"]},
            {"sample_rate": 8000},
        ],
        ids=["frame", "context", "state", "names", "rate"],
    )
    def test_a_changed_interface_changes_the_identity(self, change: dict[str, Any]) -> None:
        """The whole reason the interface is in the record (INV-08).

        A future release under the same name with a 256-sample frame must be a cache *miss*,
        not the same key over different answers. Every field is varied independently, because
        an identity that responded to only some of them would still let one through.
        """
        interface = SILERO_INTERFACE.model_copy(update=change)
        detector = SileroActivityDetector(
            StatefulFakeSession(),
            track_id=TRACK,
            identity=IDENTITY,
            silence_threshold=0.35,
            interface=interface,
        )
        assert detector.identity() != build(StatefulFakeSession()).identity()

    def test_the_interface_is_used_and_not_merely_recorded(self) -> None:
        """The record would be a lie if the adapter called the model some other way.

        Renaming the inputs and shrinking the frame changes what the session is *fed*: the
        fake refuses any other input names, and the frame count follows the declared size.
        """
        interface = SILERO_INTERFACE.model_copy(
            update={"input_names": ["chunk", "memory", "rate"], "frame_samples": 256}
        )
        session = StatefulFakeSession(input_names=("chunk", "memory", "rate"))
        detector = build(session, interface=interface)
        push(detector, noise(1024), frames_per_window=4)

        assert len(session.feeds) == 1024 // 256
        assert session.feeds[0]["chunk"].shape == (1, CONTEXT + 256)

    def test_two_detectors_over_one_model_have_one_identity(self) -> None:
        """Identity is the model and the call, never the track it happened to run on."""
        factory = silero_factory(StatefulFakeSession(), identity=IDENTITY, silence_threshold=0.35)
        assert factory("tx-a").identity() == factory("tx-b").identity()


class TestLoading:
    """Refusal, proved without a model — which is the point (INV-05)."""

    def test_an_absent_model_raises_naming_the_command_that_fixes_it(self, tmp_path: Path) -> None:
        with pytest.raises(SileroError, match="models fetch") as raised:
            load_silero_session(tmp_path / "silero_vad.onnx")
        assert raised.value.code == MODEL_UNAVAILABLE

    def test_the_refusal_does_not_import_onnxruntime(self, tmp_path: Path) -> None:
        """In a **child process**, because `sys.modules` is not this test's to speak for.

        The claim is a property of the code: `load_silero_session` must check for the file
        before it imports a runtime, so the default suite stays free of one (INV-05). Asserted
        in-process it was a property of the *worker* — true only while nothing else scheduled
        onto the same xdist worker had already imported `onnxruntime`.

        It passed for a year and then failed the day this milestone added eleven tests, which
        changed the distribution and nothing else. That is the same honest boundary
        `conftest.py` records for the socket block and `tests/test_runtime.py` for Torch: a
        fixture cannot see into a subprocess, and a subprocess is the only place a claim about
        an empty `sys.modules` can be made truthfully.
        """
        source = textwrap.dedent(f"""
            import sys
            from pathlib import Path
            from dnd_audio.activity.silero import load_silero_session
            from dnd_audio.errors import DndAudioError

            try:
                load_silero_session(Path({str(tmp_path / "silero_vad.onnx")!r}))
            except DndAudioError:
                pass
            else:  # pragma: no cover - the file does not exist
                raise SystemExit("an absent model did not raise")

            if "onnxruntime" in sys.modules:
                raise SystemExit("onnxruntime was imported before the file was checked")
        """)
        result = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_wrong_hash_raises_before_the_file_is_executed(self, tmp_path: Path) -> None:
        """The bytes are checked, then the graph is loaded. Never the other way round.

        This file is not a valid ONNX graph at all, so an implementation that loaded first
        would fail with a runtime parse error and a different code — which is what the code
        assertion below distinguishes.
        """
        path = tmp_path / "silero_vad.onnx"
        path.write_bytes(b"not an onnx graph")
        with pytest.raises(SileroError, match="pinned") as raised:
            load_silero_session(path, expected_sha256="0" * 64)
        assert raised.value.code == MODEL_HASH_MISMATCH

    def test_the_bundle_fails_loudly_when_the_store_holds_no_model(self, tmp_path: Path) -> None:
        """The runner's entry point, over an empty store.

        `silero_bundle` is what `run_activity` calls when no detector was injected, and its
        failure has to be a structured one the report can carry (INV-13) rather than an
        ONNX Runtime traceback from somewhere inside a graph load.
        """
        from dnd_audio import models

        with pytest.raises(models.ModelError, match="models fetch") as raised:
            silero_bundle(silence_threshold=0.35, directory=tmp_path)
        assert raised.value.code == models.MODEL_UNAVAILABLE

    def test_the_codes_match_the_model_stores_vocabulary(self) -> None:
        """One vocabulary for "the pinned model is not usable", two places that notice it.

        Imported inside the test so the offline suite's *import* of this module does not
        depend on the model store at all.
        """
        from dnd_audio import models

        assert MODEL_UNAVAILABLE == models.MODEL_UNAVAILABLE
        assert MODEL_HASH_MISMATCH == models.MODEL_HASH_MISMATCH


class TestNoTorch:
    def test_importing_and_using_this_module_never_imports_torch(self) -> None:
        """ADR-0013's whole packaging argument, asserted rather than assumed.

        `silero-vad` would have put Torch and torchaudio in this environment. Nothing in the
        adapter's path may reintroduce them — not the module import, not a detection run.
        """
        import dnd_audio.activity.silero  # noqa: F401 - the import under test

        detector = build(StatefulFakeSession())
        push(detector, noise(FRAME * 2), frames_per_window=1)
        assert "torch" not in sys.modules
        assert "torchaudio" not in sys.modules

    def test_the_module_can_be_imported_without_onnxruntime(self) -> None:
        """The lazy import, proved in a process that has never loaded the runtime.

        A subprocess because `sys.modules` cannot be un-rung: the `host_smoke` test in this
        same file loads the runtime, and an in-process assertion would then pass or fail on
        test ordering rather than on the code.
        """
        source = (
            "import sys; import dnd_audio.activity.silero as m; "
            "assert 'onnxruntime' not in sys.modules, 'imported eagerly'; "
            "assert 'torch' not in sys.modules, 'imported torch'; "
            f"assert m.SILERO_INTERFACE.frame_samples == {FRAME}"
        )
        completed = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr


class TestThroughTheAssembler:
    """The adapter and `detect_track` composed, over a real 16 kHz WAV."""

    @staticmethod
    def write_wav(path: Path, samples: npt.NDArray[np.float32]) -> Path:
        with WavWriter(path, sample_rate=RATE, n_samples=int(samples.shape[0])) as writer:
            writer.write(samples)
        return path

    def test_regions_come_from_the_measured_probabilities(self, tmp_path: Path) -> None:
        samples = speech_shaped(RATE * 3)
        path = self.write_wav(tmp_path / "tx-a.wav", samples)
        detector = build(StatefulFakeSession())

        result = detect_track(
            path,
            track_id=TRACK,
            detector=detector,
            settings=VadConfig(),
            window_samples=RATE // 4,
        )

        assert result.from_detector is True
        assert result.frame_probabilities.shape[0] == frame_count(int(samples.shape[0]))
        assert result.frame_probabilities.dtype == np.uint16
        assert len(result.regions) == 1
        region = result.regions[0]
        # The loud second, plus the configured padding at each end and the frame the detector
        # needed to climb through its own threshold.
        assert RATE // 2 - RATE < region.start_sample <= RATE // 2 + FRAME * 4
        assert RATE + RATE // 2 <= region.end_sample <= RATE * 2
        assert region.peak_probability_permille >= region.probability_permille
        assert region.probability_permille > PERMILLE // 2

    def test_the_window_size_does_not_change_the_answer(self, tmp_path: Path) -> None:
        """INV-07's bounded reads must not be visible in the result."""
        samples = speech_shaped(RATE * 2)
        path = self.write_wav(tmp_path / "tx-a.wav", samples)

        results = [
            detect_track(
                path,
                track_id=TRACK,
                detector=build(StatefulFakeSession()),
                settings=VadConfig(),
                window_samples=window,
            )
            for window in (FRAME, FRAME * 3, RATE, RATE * 4)
        ]
        for other in results[1:]:
            assert np.array_equal(other.frame_probabilities, results[0].frame_probabilities)
            assert other.regions == results[0].regions

    def test_silence_produces_no_regions(self, tmp_path: Path) -> None:
        path = self.write_wav(tmp_path / "tx-a.wav", np.zeros(RATE, dtype=np.float32))
        result = detect_track(
            path,
            track_id=TRACK,
            detector=build(StatefulFakeSession()),
            settings=VadConfig(),
            window_samples=RATE,
        )
        assert result.regions == ()
        assert not result.frame_probabilities.any()


@pytest.mark.host_smoke
def test_the_pinned_model_runs_on_real_inference() -> None:
    """The pinned artifact under real ONNX Runtime. The only test here that loads a model.

    What it asserts is true of *the model*: probabilities are probabilities, the frame count
    is the one the track's length predicts, a loud broadband burst scores above digital
    silence, and the recurrence survives a change of window partitioning. What it deliberately
    does **not** assert is that synthetic speech-shaped noise reads as speech — INV-10 forbids
    expecting a particular learned release to fire on audio no human made, and a test that did
    would fail on the next release for no reason anyone could act on.

    One test rather than several, because each one costs a load of the artifact and the thing
    being established is a single claim: this file, through this adapter, answers.
    """
    from dnd_audio.models import SILERO_VAD, find_model

    path = find_model(SILERO_VAD)
    assert path is not None, (
        "the pinned VAD model is not on this machine. Run `dnd-audio models fetch`.\n"
        "This fails rather than skipping on purpose: `host_smoke` already means "
        "'needs the target host', and a host test that quietly skips when the host is not "
        "set up is a check that verifies nothing — which is the failure M1's closeout "
        "records finding in its own suite."
    )

    session = load_silero_session(path, expected_sha256=SILERO_VAD.sha256)
    factory = silero_factory(
        session,
        identity=SileroIdentity(
            release=SILERO_VAD.release,
            commit=SILERO_VAD.commit,
            model_sha256=SILERO_VAD.sha256,
            runtime_version=onnxruntime_version(),
        ),
        silence_threshold=0.35,
    )

    rng = np.random.default_rng(20260802)
    samples = np.zeros(RATE * 2, dtype=np.float32)
    samples[RATE : RATE + RATE // 2] = np.clip(rng.standard_normal(RATE // 2) * 0.8, -1.0, 1.0)

    detector = factory("tx-a")
    detector.detect(AudioWindow(track_id="tx-a", sample_rate=RATE, start_sample=0, samples=samples))
    probabilities = detector.frame_probabilities()

    assert probabilities.shape[0] == frame_count(int(samples.shape[0]))
    assert probabilities.min() >= 0
    assert probabilities.max() <= PERMILLE
    silent = probabilities[: RATE // FRAME]
    loud = probabilities[RATE // FRAME : (RATE + RATE // 2) // FRAME]
    assert loud.max() > silent.max()

    # The same audio through the same shared session, in three-frame windows: the recurrence
    # lives in the detector, so the partitioning must not be visible in the answer.
    split = factory("tx-b")
    push(split, samples, frames_per_window=3)
    assert np.array_equal(split.frame_probabilities(), probabilities)

    # The identity a detection cache key would be built from, populated from the pin itself.
    identity = detector.identity()
    assert identity.model_sha256 == SILERO_VAD.sha256
    assert identity.runtime_version == onnxruntime_version()
    assert identity.interface == SILERO_INTERFACE
