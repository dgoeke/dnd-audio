"""Place synthetic signals on a timeline and write a session directory.

The spec's fixture recipe is a list of properties a test needs to be able to assert:
multiple chunks per transmitter, different start offsets, a real gap, a shared clap,
solo speech that bleeds quietly into the others, and one two-speaker interval.
:func:`canonical_session` is that list, made concrete; :func:`build_session` writes it
and hands back a :class:`FixtureTruth` recording what it wrote.

**The truth record is the point.** A test that inspects the fixture and then asserts
against numbers re-derived from the same fixture proves nothing. Every property the
gate asks for is stated here, in samples, before any file exists.

Rendering is per chunk rather than per track. A whole-track buffer would be simpler and
would make the four-hour soak fixture H2 needs impossible: six session-length float32
arrays is what INV-07 exists to forbid. Events are rendered once at full length and
sliced into whichever chunks they touch, so where a chunk boundary falls cannot change
a sample.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import yaml

from dnd_audio.config import SessionConfig
from dnd_audio.determinism import sha256_file, write_json_atomic
from dnd_audio.fixtures import synth
from dnd_audio.fixtures.wav import (
    BroadcastMetadata,
    ExtraChunk,
    SampleFormatName,
    quantize,
    write_wav,
)
from dnd_audio.interfaces import SpeechSpan
from dnd_audio.timecode import (
    FrameRate,
    frame_index,
    parse_frame_rate,
    parse_timecode,
)

__all__ = [
    "CANONICAL_ORIGIN_DATE",
    "CANONICAL_SESSION_ZERO_TIMECODE",
    "ClapInterval",
    "FixtureChunk",
    "FixtureSession",
    "FixtureTrack",
    "FixtureTruth",
    "SpeechInterval",
    "WrittenChunk",
    "build_session",
    "canonical_session",
    "dji_filename",
]

SAMPLE_RATE: Final = 48000

#: The canonical fixture's calendar day and the wall clock of session sample zero.
#: Both are stated rather than derived: `timecode.origin_date` must never be inferred
#: from a date-shaped `session_id`, and a fixture that encouraged that would be
#: teaching the wrong lesson.
CANONICAL_ORIGIN_DATE: Final = dt.date(2026, 8, 15)
CANONICAL_SESSION_ZERO_TIMECODE: Final = "19:00:00:00"

TimecodeSource = Literal["bext", "info_ismp", "both", "none"]
Variant = Literal["orig", "edit"]

#: 3 ms. Sound crossing a table between two wearers, and far enough outside zero lag
#: that a correlator restricted to zero lag would miss it entirely.
_BLEED_DELAY_SAMPLES: Final = 144
_BLEED_ATTENUATION_DB: Final = 26.0

#: How much quieter a processed `edit` file is than its original. Arbitrary, and only
#: has to be audible to a hash: what matters is that the two files differ.
_EDIT_GAIN: Final = 0.9

#: Samples of noise floor generated per seeded block. One second: small enough that a
#: bounded window never materializes much more than it asked for, large enough that the
#: per-block seeding overhead is irrelevant.
_FLOOR_BLOCK: Final = SAMPLE_RATE


def _seed(*parts: object) -> int:
    """A stable 32-bit seed from any description of an event.

    Python's ``hash()`` is salted per process, so seeding from it would make a fixture
    reproducible only within one interpreter run — which is exactly the kind of
    almost-determinism INV-02 is about.
    """
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


@dataclass(frozen=True, slots=True)
class SpeechInterval:
    """One person speaking, and whose transmitters quietly pick it up.

    ``utterance_id`` and ``text`` are here for milestones M1 does not reach. The spec's
    fixture recipe asks for deterministic fake-VAD ground truth *and* deterministic
    fake-ASR results, and both are nearly free to state alongside the interval that
    produced them. Retrofitting them in M4, once the fixture's sample positions are
    baked into a dozen tests, is not.
    """

    track_id: str
    start_sample: int
    n_samples: int
    #: Tracks that receive an attenuated, delayed copy. Only tracks that are actually
    #: recording during the interval get one — a transmitter that is off hears nothing,
    #: and the fixture would be lying if it said otherwise.
    bleeds_into: tuple[str, ...] = ()
    #: How late and how much quieter the bleed arrives. ``None`` takes the module defaults,
    #: which are one plausible table's worth of geometry; a fixture about the *lag window*
    #: or about mutual bleed during real overlap needs its own numbers.
    bleed_delay_samples: int | None = None
    bleed_attenuation_db: float | None = None
    gain: float = 0.25
    #: Stable identity, derived from track and position rather than from list order.
    utterance_id: str = ""
    #: What a fake ASR should return for this interval. Not what the audio "says" —
    #: nothing here is speech (INV-10).
    text: str = ""

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.n_samples

    @property
    def delay_samples(self) -> int:
        """How late this interval's bleed arrives, its own value or the module default.

        A property rather than three copies of the same conditional: the renderer, the truth
        record, and the declared fake-ASR script all need it, and a fixture whose scripted text
        landed near the bleed rather than on it would be testing nothing in particular.
        """
        return (
            _BLEED_DELAY_SAMPLES if self.bleed_delay_samples is None else self.bleed_delay_samples
        )


@dataclass(frozen=True, slots=True)
class ClapInterval:
    """A shared transient. Every live transmitter hears it at the same sample.

    ``tracks`` restricts it to a subset, which is how a *drift* fixture is built: the same
    physical clap is written to two tracks at slightly different samples, so the only way
    to find the difference is to correlate the audio. Moving the metadata instead would
    let a drift test pass while the correlator was never exercised.
    """

    start_sample: int
    n_samples: int = int(0.8 * SAMPLE_RATE)
    gain: float = 0.7
    #: ``None`` means every track, which is what a real clap in a shared room is.
    tracks: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FixtureChunk:
    """One recorded file on one transmitter's timeline."""

    start_sample: int
    n_samples: int
    #: The DJI filename's ``MIC###`` counter. Monotonic per transmitter, and only ever
    #: a secondary ordering hint (OQ-003).
    sequence: int
    variant: Variant = "orig"
    timecode_source: TimecodeSource = "bext"
    #: Defaults to the session rate. Set it to 44100 to build the nonconforming source
    #: M1 warns about and M2 rejects.
    sample_rate: int | None = None
    #: Floor this chunk's written `bext.time_reference` to a multiple of this many samples,
    #: the way real hardware does. OQ-004 measured every DJI reference as a multiple of
    #: 1600, which means a chunk after a switch-off reports a start *earlier* than where its
    #: audio really begins — and therefore overlapping the chunk before it. Every other
    #: fixture here is sample-exact by construction, which is why nothing caught that.
    quantize_reference_samples: int | None = None
    #: What the transmitter was set to write. `pcm_s24le` is not hypothetical — it is what
    #: two of four kits produced in the 2026-08-02 probe from a per-transmitter setting the
    #: operator had not matched (ADR-0030), and it is per *chunk* because that is how the
    #: mistake happens: one kit, not the session.
    sample_format: SampleFormatName = "pcm_f32le"
    rf64: bool = False
    #: An opaque custom chunk `ffprobe` will not report, so the generic RIFF walk has
    #: something to find. Invented, not observed (OQ-005).
    private_chunk: bool = True
    #: Overrides the generated DJI name, for fixtures about naming itself.
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class FixtureTrack:
    """One transmitter, its wearer, and the receiver channel it was paired to."""

    track_id: str
    speaker_id: str
    speaker_name: str
    receiver_id: str
    receiver_channel: int
    #: DJI's pairing-order label. Deliberately **not** unique across kits in the
    #: canonical fixture: two receivers both produce a ``TX01`` (OQ-002, INV-11).
    tx_label: str
    chunks: tuple[FixtureChunk, ...]
    #: When false the directory is created but left empty, which is how an absent
    #: player is distinguished from a configured one.
    write_files: bool = True
    #: How far this receiver's real-time clock is from the session's, for the `bext`
    #: origination date and time only. Two receivers were measured **48.7 s apart** while
    #: their timecode agreed to under one frame (ADR-0031), so a fixture has to be able to
    #: express a wall clock that disagrees with a timecode that does not. Nothing else in
    #: the fixture moves: the audio, the sample positions and the `time_reference` are
    #: unchanged, which is the whole point.
    wall_clock_offset: dt.timedelta = dt.timedelta()


@dataclass(frozen=True, slots=True)
class FixtureSession:
    """Everything needed to write a session directory."""

    session_id: str
    title: str
    tracks: tuple[FixtureTrack, ...]
    origin_date: dt.date = CANONICAL_ORIGIN_DATE
    #: Where the session's audio actually sits in the day. Always used to place the
    #: fixture; written to `session.yaml` only when ``origin_timecode`` is not overridden.
    session_zero_timecode: str = CANONICAL_SESSION_ZERO_TIMECODE
    #: What `session.yaml` states as `timecode.origin_timecode`. Defaults to the fixture's
    #: own zero; set it to ``None`` to build a session whose origin must be *derived* from
    #: the earliest source, which is the other half of ADR-0009 and has no other coverage.
    origin_timecode: str | None = "="
    frame_rate: str = "30F"
    sample_rate: int = SAMPLE_RATE
    speech: tuple[SpeechInterval, ...] = ()
    claps: tuple[ClapInterval, ...] = ()
    active_tracks: Literal["auto"] | tuple[str, ...] = "auto"
    allow_processed_audio: bool = False
    source_time_overrides: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WrittenChunk:
    """One file the generator actually wrote."""

    track_id: str
    relative_path: str
    start_sample: int
    n_samples: int
    sample_rate: int
    #: Samples since midnight at this file's own rate — what a BWF ``time_reference``
    #: means, and what the strategy chain must recover.
    time_reference: int
    variant: Variant
    sequence: int
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FixtureTruth:
    """What the generator wrote, stated independently of anything that reads it."""

    session_dir: Path
    sample_rate: int
    session_zero_since_midnight: int
    chunks: tuple[WrittenChunk, ...]
    speech: tuple[SpeechInterval, ...]
    claps: tuple[ClapInterval, ...]

    def for_track(self, track_id: str) -> tuple[WrittenChunk, ...]:
        return tuple(c for c in self.chunks if c.track_id == track_id)

    def activity_spans(
        self, *, sample_rate: int | None = None
    ) -> dict[str, tuple[SpeechSpan, ...]]:
        """Ground-truth speech, in the shape :class:`ScriptedActivityDetector` takes.

        M3's detector fake is driven from this rather than from an opinion about the
        synthetic audio: INV-10 forbids expecting speech-shaped noise to trigger a
        particular learned Silero release. Bleed is deliberately absent — it is not
        speech on the track that received it, and a fixture that claimed otherwise
        would make the bleed gate untestable.

        Args:
            sample_rate: The grid to express the spans on. Defaults to the fixture's own
                48 kHz. **Pass the derivative rate when scripting a detector for M3**: the
                detection pass hands out windows of the 16 kHz derivative, so 48 kHz spans
                land past the end of every window, the detector returns nothing, and every
                attribution assertion then passes over an empty graph. That failure is
                silent in exactly the way this project keeps finding — hence the argument
                rather than a convention.
        """
        return self._at_rate(
            {
                interval.track_id: [
                    SpeechSpan(start_sample=i.start_sample, end_sample=i.end_sample)
                    for i in self.speech
                    if i.track_id == interval.track_id
                ]
                for interval in self.speech
            },
            sample_rate,
        )

    def bleed_spans(self, *, sample_rate: int | None = None) -> dict[str, tuple[SpeechSpan, ...]]:
        """Where each track *hears someone else*, in that track's own samples.

        The fixture knows this exactly — it wrote the bleed — so a test never has to guess
        where a leaky detector would fire. A real VAD fires on bleed; :meth:`activity_spans`
        deliberately does not, because it is ground truth about who *spoke*. Combining them
        is :meth:`leaky_activity_spans`.

        The delay matters and is included: bleed arrives late, which is the entire reason
        the similarity measure is lag-tolerant rather than zero-lag.
        """
        spans: dict[str, list[SpeechSpan]] = {}
        for interval in self.speech:
            delay = interval.delay_samples
            for target in interval.bleeds_into:
                spans.setdefault(target, []).append(
                    SpeechSpan(
                        start_sample=interval.start_sample + delay,
                        end_sample=interval.end_sample + delay,
                    )
                )
        return self._at_rate(spans, sample_rate)

    def leaky_activity_spans(
        self, *, sample_rate: int | None = None
    ) -> dict[str, tuple[SpeechSpan, ...]]:
        """What a *real* detector would find: speech and bleed alike, per track.

        Provided here rather than left to each test because the obvious spelling —
        ``activity_spans() | bleed_spans()`` — is a dict union, and a union **replaces** the
        value for any track present in both. In the canonical fixture that is `tx-d` and
        `tx-e`, whose genuine simultaneous speech would vanish and take the overlap case
        with it. Found by independent review of M3's plan; merged per track here so no test
        can spell it the wrong way.
        """
        speech = self.activity_spans(sample_rate=sample_rate)
        bleed = self.bleed_spans(sample_rate=sample_rate)
        merged: dict[str, tuple[SpeechSpan, ...]] = {}
        for track in sorted(set(speech) | set(bleed)):
            items = [*speech.get(track, ()), *bleed.get(track, ())]
            merged[track] = tuple(sorted(items, key=lambda s: (s.start_sample, s.end_sample)))
        return merged

    def _at_rate(
        self, spans: dict[str, list[SpeechSpan]], sample_rate: int | None
    ) -> dict[str, tuple[SpeechSpan, ...]]:
        """Convert to another grid, covering the interval rather than shrinking it.

        The start floors and the end ceils, which is the same rule
        `timeline.resample.to_derivative_interval` states and for the same reason: rounding
        both ends alike loses up to two samples of a word, at the end where the phonemes are.
        """
        if sample_rate is None or sample_rate == self.sample_rate:
            return {
                track: tuple(sorted(items, key=lambda s: (s.start_sample, s.end_sample)))
                for track, items in sorted(spans.items())
            }
        if self.sample_rate % sample_rate:
            message = (
                f"{self.sample_rate} Hz does not divide by {sample_rate} Hz, so a span "
                f"cannot be moved between the two grids exactly"
            )
            raise ValueError(message)
        factor = self.sample_rate // sample_rate
        return {
            track: tuple(
                sorted(
                    (
                        SpeechSpan(
                            start_sample=span.start_sample // factor,
                            end_sample=-(-span.end_sample // factor),
                            probability=span.probability,
                            details=dict(span.details),
                        )
                        for span in items
                    ),
                    key=lambda s: (s.start_sample, s.end_sample),
                )
            )
            for track, items in sorted(spans.items())
        }

    def transcript_script(self) -> dict[str, str]:
        """``{utterance_id: text}`` for M4's :class:`ScriptedTranscriber`.

        The mapping from an utterance to a transcription *request* is M4's to define;
        what the fixture owes it is a stable id and the text that belongs to it.
        """
        return {i.utterance_id: i.text for i in self.speech if i.utterance_id}

    def gaps(self) -> tuple[tuple[str, int, int], ...]:
        """``(track_id, gap_start, gap_end)`` for every real gap in the fixture."""
        found: list[tuple[str, int, int]] = []
        for track_id in sorted({c.track_id for c in self.chunks}):
            ordered = sorted(self.for_track(track_id), key=lambda c: c.start_sample)
            for earlier, later in itertools.pairwise(ordered):
                end = earlier.start_sample + earlier.n_samples
                if later.start_sample > end:
                    found.append((track_id, end, later.start_sample))
        return tuple(found)

    def overlapping_speech(self) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
        """Intervals where two different tracks are speaking at once."""
        found: list[tuple[int, int, tuple[str, ...]]] = []
        for i, first in enumerate(self.speech):
            for second in self.speech[i + 1 :]:
                if first.track_id == second.track_id:
                    continue
                start = max(first.start_sample, second.start_sample)
                end = min(
                    first.start_sample + first.n_samples,
                    second.start_sample + second.n_samples,
                )
                if end > start:
                    found.append((start, end, tuple(sorted((first.track_id, second.track_id)))))
        return tuple(sorted(found))


def dji_filename(tx_label: str, sequence: int, start: dt.datetime, variant: Variant) -> str:
    """The filename grammar the fixtures assume DJI uses (OQ-003).

    Written as an explicit format string rather than shared with
    :mod:`dnd_audio.inspection.naming`, which parses. Sharing one grammar table would
    make the round-trip test prove only that the table is self-consistent — and the
    grammar itself is a guess until H1 lands.
    """
    return f"{tx_label}_MIC{sequence:03d}_{start:%Y%m%d}_{start:%H%M%S}_{variant}.wav"


def canonical_session() -> FixtureSession:
    """The six-transmitter fixture M1's completion gate describes.

    Laid out in whole samples at 48 kHz. The properties, and where each one lives:

    * **Multiple chunks per transmitter** — every track has two.
    * **Different start offsets** — 0.0 s to 3.5 s, none of them equal.
    * **A real gap** — ``tx-c`` stops at 5.0 s and resumes at 8.0 s.
    * **A shared clap** — at 4.0 s, when all six are recording. Deliberately inside
      ``tx-c``'s first chunk and after ``tx-f``'s start, so it is genuinely shared.
    * **Solo speech that bleeds** — ``tx-a`` at 5.2-6.2 s, leaking into ``tx-b``,
      ``tx-d``, ``tx-e``, ``tx-f``. Not into ``tx-c``, which is inside its gap: a
      transmitter that is switched off hears nothing.
    * **Two simultaneous speakers** — ``tx-d`` and ``tx-e`` at 6.8-7.8 s.
    * **Post-gap speech** — ``tx-c`` at 8.5-9.5 s, so a bug that slides later audio
      earlier has something to get wrong.

    ``tx-f`` carries its timecode as an ``INFO``/``ISMP`` tag instead of a ``bext``
    time reference, so the canonical fixture exercises both strategies in the chain.
    Its chunk starts fall on exact 30 fps frame boundaries because a timecode tag
    cannot express anything finer.
    """
    layout: tuple[tuple[str, str, str, str, int, str, tuple[tuple[int, int], ...]], ...] = (
        ("tx-a", "alice", "Alice", "rx-a", 1, "TX01", ((0, 144000), (144000, 192000))),
        ("tx-b", "bob", "Bob", "rx-a", 2, "TX02", ((24000, 144000), (168000, 144000))),
        ("tx-c", "carol", "Carol", "rx-b", 1, "TX01", ((60000, 180000), (384000, 120000))),
        ("tx-d", "dan", "Dan", "rx-b", 2, "TX02", ((96000, 144000), (240000, 144000))),
        ("tx-e", "erin", "Erin", "rx-c", 1, "TX01", ((132000, 144000), (276000, 120000))),
        ("tx-f", "frank", "Frank", "rx-c", 2, "TX02", ((168000, 144000), (312000, 120000))),
    )

    tracks: list[FixtureTrack] = []
    for track_id, speaker_id, name, receiver, channel, tx_label, chunks in layout:
        source: TimecodeSource = "info_ismp" if track_id == "tx-f" else "bext"
        tracks.append(
            FixtureTrack(
                track_id=track_id,
                speaker_id=speaker_id,
                speaker_name=name,
                receiver_id=receiver,
                receiver_channel=channel,
                tx_label=tx_label,
                chunks=tuple(
                    FixtureChunk(
                        start_sample=start,
                        n_samples=length,
                        sequence=index + 1,
                        timecode_source=source,
                    )
                    for index, (start, length) in enumerate(chunks)
                ),
            )
        )

    return FixtureSession(
        session_id="2026-08-15",
        title="Session 01",
        tracks=tuple(tracks),
        claps=(ClapInterval(start_sample=192000),),
        speech=(
            SpeechInterval(
                track_id="tx-a",
                start_sample=249600,
                n_samples=48000,
                bleeds_into=("tx-b", "tx-d", "tx-e", "tx-f"),
                utterance_id="utt_tx-a_000249600",
                text="We should go back to Zephyrine.",
            ),
            SpeechInterval(
                track_id="tx-d",
                start_sample=326400,
                n_samples=48000,
                utterance_id="utt_tx-d_000326400",
                text="Absolutely not.",
            ),
            SpeechInterval(
                track_id="tx-e",
                start_sample=326400,
                n_samples=48000,
                utterance_id="utt_tx-e_000326400",
                text="Wait, say that again?",
            ),
            SpeechInterval(
                track_id="tx-c",
                start_sample=408000,
                n_samples=48000,
                utterance_id="utt_tx-c_000408000",
                text="Sorry, my transmitter was off.",
            ),
        ),
    )


def build_session(spec: FixtureSession, directory: Path) -> FixtureTruth:
    """Write ``spec`` into ``directory`` and return what was written."""
    directory.mkdir(parents=True, exist_ok=True)
    zero = _samples_since_midnight(
        spec.session_zero_timecode, spec.sample_rate, parse_frame_rate(spec.frame_rate)
    )
    events = _build_events(spec)

    written: list[WrittenChunk] = []
    for track in spec.tracks:
        track_dir = directory / "raw" / track.track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        if not track.write_files:
            continue
        for chunk in track.chunks:
            written.append(_write_chunk(spec, track, chunk, directory, zero, events))

    _write_config(spec, directory)
    truth = FixtureTruth(
        session_dir=directory,
        sample_rate=spec.sample_rate,
        session_zero_since_midnight=zero,
        chunks=tuple(sorted(written, key=lambda c: c.relative_path)),
        speech=spec.speech,
        claps=spec.claps,
    )
    _write_fake_models(truth, directory)
    return truth


@dataclass(frozen=True, slots=True)
class _Event:
    """A rendered sound already placed on one track's timeline."""

    track_id: str
    start_sample: int
    samples: npt.NDArray[np.float32]


def _build_events(spec: FixtureSession) -> tuple[_Event, ...]:
    """Render every sound once, at full length, before any chunking."""
    events: list[_Event] = []
    live = {track.track_id: _live_intervals(track) for track in spec.tracks}

    for clap in spec.claps:
        # Seeded by position alone, so a clap written to two tracks at different samples
        # is still recognisably the *same* transient to a correlator.
        samples = synth.clap(
            clap.n_samples,
            spec.sample_rate,
            seed=_seed("clap", clap.start_sample if clap.tracks is None else "shared"),
            gain=clap.gain,
        )
        for track in spec.tracks:
            if clap.tracks is not None and track.track_id not in clap.tracks:
                continue
            events.append(_Event(track.track_id, clap.start_sample, samples))

    for interval in spec.speech:
        voice = synth.speech_shaped(
            interval.n_samples,
            spec.sample_rate,
            seed=_seed("speech", interval.track_id, interval.start_sample),
            gain=interval.gain,
        )
        events.append(_Event(interval.track_id, interval.start_sample, voice))

        bleed = synth.bleed_of(
            voice,
            delay_samples=interval.delay_samples,
            attenuation_db=(
                _BLEED_ATTENUATION_DB
                if interval.bleed_attenuation_db is None
                else interval.bleed_attenuation_db
            ),
        )
        for target in interval.bleeds_into:
            if not _is_live(live.get(target, ()), interval.start_sample, interval.n_samples):
                message = (
                    f"speech on {interval.track_id} at {interval.start_sample} cannot bleed "
                    f"into {target}: that transmitter is not recording then"
                )
                raise ValueError(message)
            events.append(_Event(target, interval.start_sample, bleed))

    return tuple(events)


def _live_intervals(track: FixtureTrack) -> tuple[tuple[int, int], ...]:
    """When a transmitter was recording, with contiguous chunks merged.

    Merging matters: DJI splitting a file mid-sentence does not mean the transmitter
    stopped, and a bleed that straddles a chunk boundary is perfectly ordinary. Only a
    real gap — the thing M2 has to preserve as silence — separates two intervals here.
    """
    ordered = sorted((c.start_sample, c.start_sample + c.n_samples) for c in track.chunks)
    merged: list[tuple[int, int]] = []
    for begin, end in ordered:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return tuple(merged)


def _is_live(intervals: tuple[tuple[int, int], ...], start: int, length: int) -> bool:
    """Whether one interval is wholly inside a single unbroken recording."""
    return any(start >= begin and start + length <= end for begin, end in intervals)


def _floor(track_id: str, start: int, n_samples: int) -> npt.NDArray[np.float32]:
    """The track's self-noise for a window, seeded by **timeline position**.

    Seeding the floor per chunk would be simpler and would quietly break the property
    the whole renderer is built around: two chunkings of the same session would then
    differ in every sample, and "chunk boundaries cannot change a sample" would be true
    only of the parts anyone bothered to look at. Blocks are fixed-size and generated on
    demand, so this stays bounded for a four-hour fixture (INV-07).
    """
    first = start // _FLOOR_BLOCK
    last = (start + n_samples - 1) // _FLOOR_BLOCK
    blocks = [
        synth.noise_floor(_FLOOR_BLOCK, seed=_seed("floor", track_id, index))
        for index in range(first, last + 1)
    ]
    offset = start - first * _FLOOR_BLOCK
    return np.concatenate(blocks)[offset : offset + n_samples]


def _render(
    events: tuple[_Event, ...], track_id: str, start: int, n_samples: int
) -> npt.NDArray[np.float32]:
    """Sum every event overlapping ``[start, start + n_samples)`` on one track."""
    buffer = _floor(track_id, start, n_samples).astype(np.float64)
    for event in events:
        if event.track_id != track_id:
            continue
        event_end = event.start_sample + int(event.samples.shape[0])
        overlap_start = max(event.start_sample, start)
        overlap_end = min(event_end, start + n_samples)
        if overlap_end <= overlap_start:
            continue
        into = slice(overlap_start - start, overlap_end - start)
        source = slice(overlap_start - event.start_sample, overlap_end - event.start_sample)
        buffer[into] += event.samples[source]
    return buffer.astype(np.float32)


def _write_chunk(
    spec: FixtureSession,
    track: FixtureTrack,
    chunk: FixtureChunk,
    directory: Path,
    zero: int,
    events: tuple[_Event, ...],
) -> WrittenChunk:
    rate = chunk.sample_rate or spec.sample_rate
    start_wall = _wall_clock(spec, chunk.start_sample) + track.wall_clock_offset
    name = chunk.filename or dji_filename(track.tx_label, chunk.sequence, start_wall, chunk.variant)
    relative = f"raw/{track.track_id}/{name}"
    path = directory / relative

    samples = _render(events, track.track_id, chunk.start_sample, chunk.n_samples)
    if chunk.variant == "edit":
        # A processed file is not a copy of the original — the recorder applies its own
        # gain and filtering. Writing identical bytes would make every orig/edit fixture
        # accidentally a *duplicate* fixture, and the selection rule under test would
        # never run.
        samples = (samples * _EDIT_GAIN).astype(np.float32)

    # `time_reference` is at the file's own rate, which is why a nonconforming 44.1 kHz
    # chunk cannot simply reuse the session-rate number.
    time_reference = (zero + chunk.start_sample) * rate // spec.sample_rate
    if chunk.quantize_reference_samples is not None:
        # Floored, not rounded: a counter that ticks once a frame reports the last tick it
        # saw, so the reported start is never later than the real one.
        time_reference -= time_reference % chunk.quantize_reference_samples

    broadcast: BroadcastMetadata | None = None
    if chunk.timecode_source in ("bext", "both"):
        broadcast = BroadcastMetadata(
            time_reference=time_reference,
            # Both halves of one instant. Taking the date from the session and the time
            # from the chunk would put 00:00:00 on the day before it happened, which is
            # exactly the shape `wall_clock_skew_session` exists to write deliberately.
            origination_date=start_wall.date(),
            origination_time=start_wall.time(),
            description=f"{track.tx_label} {track.track_id}",
            originator="DJI",
        )

    info: dict[bytes, str] | None = None
    if chunk.timecode_source in ("info_ismp", "both"):
        info = {b"ISMP": _timecode_text(spec, chunk.start_sample), b"INAM": spec.title}

    extra: list[ExtraChunk] = []
    if chunk.private_chunk:
        # An *invented* opaque chunk, not an observed one. No DJI file has been seen
        # yet (OQ-005); what is being tested is that the RIFF walk finds and hashes a
        # chunk nothing can interpret, which is true whatever four bytes name it.
        # Naming it after the vendor would turn a placeholder into a claim about
        # hardware. `ffprobe` reports neither this nor the iXML below.
        extra.append(ExtraChunk(b"XPRV", bytes(range(64))))
        extra.append(
            ExtraChunk(
                b"iXML",
                (
                    "<BWFXML><IXML_VERSION>1.5</IXML_VERSION>"
                    f"<PROJECT>{spec.title}</PROJECT>"
                    f"<TRACK_LABEL>{track.tx_label}</TRACK_LABEL>"
                    "</BWFXML>"
                ).encode("ascii"),
            )
        )

    written: npt.NDArray[np.float32] | npt.NDArray[np.int32] = samples
    if chunk.sample_format != "pcm_f32le":
        written = quantize(samples, chunk.sample_format)

    size = write_wav(
        path,
        written,
        sample_rate=rate,
        sample_format=chunk.sample_format,
        broadcast=broadcast,
        info=info,
        extra=tuple(extra),
        rf64=chunk.rf64,
    )

    return WrittenChunk(
        track_id=track.track_id,
        relative_path=relative,
        start_sample=chunk.start_sample,
        n_samples=chunk.n_samples,
        sample_rate=rate,
        time_reference=time_reference,
        variant=chunk.variant,
        sequence=chunk.sequence,
        sha256=sha256_file(path),
        size_bytes=size,
    )


def _samples_since_midnight(timecode: str, sample_rate: int, frame_rate: FrameRate) -> int:
    """Whole samples since the day origin for a ``HH:MM:SS:FF`` string, at any rate.

    Exact or nothing. The generator is deliberately restricted to positions that land on a
    whole sample: a fixture that had to round would be encoding an approximation as ground
    truth, and every test built on it would inherit the error rather than detect it.

    At 30000/1001 fps one frame is 8008/5 samples at 48 kHz, so only every fifth frame is
    expressible — which is why the drop-frame fixture's offsets are multiples of 8008.
    """
    parsed = parse_timecode(timecode, frame_rate)
    frames = frame_index(parsed)
    exact = Fraction(frames) / frame_rate.rate * sample_rate
    if exact.denominator != 1:
        message = (
            f"fixture timecode {timecode!r} at {frame_rate.label} is {float(exact):.4f} "
            f"samples at {sample_rate} Hz, which is not a whole sample. Choose a frame "
            f"count that divides exactly — at 29.97 that means a multiple of five frames."
        )
        raise ValueError(message)
    return int(exact)


def _timecode_text(spec: FixtureSession, start_sample: int) -> str:
    """``HH:MM:SS:FF`` for a chunk start, at the fixture's configured rate.

    The inverse of :func:`~dnd_audio.timecode.frame_index`, drop-frame included.
    ``tests/test_fixtures.py`` round-trips the two against each other rather than trusting
    either: an inverse that is subtly wrong would put a fixture's declared truth and its
    written metadata into quiet disagreement, which is the one bug a ground-truth fixture
    must not have.
    """
    frame_rate = parse_frame_rate(spec.frame_rate)
    zero = _samples_since_midnight(spec.session_zero_timecode, spec.sample_rate, frame_rate)
    exact = Fraction(zero + start_sample, spec.sample_rate) * frame_rate.rate
    if exact.denominator != 1:
        message = (
            f"chunk start {start_sample} is not on a {frame_rate.label} frame boundary, "
            f"so it cannot be written as a timecode tag; move it, or use bext"
        )
        raise ValueError(message)
    return _frames_to_timecode(int(exact), frame_rate)


def _frames_to_timecode(frames: int, frame_rate: FrameRate) -> str:
    """Turn an exact frame index back into the label a recorder would write."""
    nominal = frame_rate.frames_per_timecode_second
    counted = frames
    if frame_rate.drop_frame:
        dropped = nominal // 15
        per_ten_minutes = nominal * 600 - dropped * 9
        per_minute = nominal * 60 - dropped
        tens, rest = divmod(frames, per_ten_minutes)
        counted = frames + dropped * 9 * tens
        if rest > dropped:
            counted += dropped * ((rest - dropped) // per_minute)

    frame_part = counted % nominal
    total_seconds = counted // nominal
    hours, rest_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest_seconds, 60)
    separator = ";" if frame_rate.drop_frame else ":"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frame_part:02d}"


def _wall_clock(spec: FixtureSession, start_sample: int) -> dt.datetime:
    """The calendar instant a chunk starts at, for the filename and the `bext` fields.

    Counted from the *timecode* day origin, which is real midnight at every integer rate.
    At a fractional non-drop rate the two differ (OQ-015); the fixtures that use those
    rates carry no origination date, so nothing here depends on the difference.
    """
    frame_rate = parse_frame_rate(spec.frame_rate)
    zero = _samples_since_midnight(spec.session_zero_timecode, spec.sample_rate, frame_rate)
    seconds, _ = divmod(zero + start_sample, spec.sample_rate)
    midnight = dt.datetime.combine(spec.origin_date, dt.time(), tzinfo=dt.UTC)
    return midnight + dt.timedelta(seconds=seconds)


def _write_fake_models(truth: FixtureTruth, directory: Path) -> None:
    """Write the declared fake model outputs beside `session.yaml` (ADR-0018).

    Imported here rather than at module scope because it reads a completed
    :class:`FixtureTruth`, which is what this module produces — importing it at the top would
    be a cycle. It writes an *input*: something a run may be told to read, never under `raw/`,
    and never consulted unless `--fake-models` says so.
    """
    from dnd_audio.fixtures.fakemodels import fake_models_document
    from dnd_audio.transcript import FAKE_MODELS_FILENAME

    write_json_atomic(directory / FAKE_MODELS_FILENAME, fake_models_document(truth))


def _write_config(spec: FixtureSession, directory: Path) -> None:
    """Write `session.yaml`, validated through the real model before it is written.

    Building the document and then validating it means a fixture can never describe a
    session the pipeline would reject — which would turn a configuration bug into a
    mysterious inspection failure.
    """
    document: dict[str, object] = {
        "schema_version": 1,
        "session_id": spec.session_id,
        "title": spec.title,
        "active_tracks": ("auto" if spec.active_tracks == "auto" else sorted(spec.active_tracks)),
        "timecode": {
            "frame_rate": spec.frame_rate,
            "origin_date": spec.origin_date.isoformat(),
            "origin_timecode": (
                spec.session_zero_timecode if spec.origin_timecode == "=" else spec.origin_timecode
            ),
            "rollover_policy": "infer_forward",
        },
        "tracks": [
            {
                "track_id": track.track_id,
                "receiver_id": track.receiver_id,
                "receiver_channel": track.receiver_channel,
                "speaker_id": track.speaker_id,
                "speaker_name": track.speaker_name,
                "input": f"raw/{track.track_id}",
            }
            for track in spec.tracks
        ],
        "recovery": {
            "allow_processed_audio": spec.allow_processed_audio,
            "source_time_overrides": spec.source_time_overrides,
        },
    }
    SessionConfig.model_validate(document)
    (directory / "session.yaml").write_text(
        yaml.safe_dump(document, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
