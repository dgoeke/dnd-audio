"""The `session.yaml` contract.

Every model here forbids unknown keys. That is deliberate: a typo in a session file
would otherwise be silently ignored, and the run would use a default the operator
believed they had overridden. It also means a later milestone adding a field is a
visible, deliberate change rather than a quiet drift — M3 and M5 will extend the
``activity`` and ``mix`` sections when they choose their thresholds.

INV-11 is structural here rather than enforced by a check: ``track_id`` is the mapping
key and derives from the configured directory. ``receiver_id`` and ``receiver_channel``
document and validate the physical setup, and there is no code path by which they can
become identity.

:func:`resolved_config` and :func:`config_hash` define what "the configuration" means to
a cache key (INV-08). A session file that omits a default must hash identically to one
that states it, or every default change would silently reuse stale cached work.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.errors import ConfigError, TimecodeError
from dnd_audio.models import QWEN3_ALIGNER, QWEN3_ASR, REVISION_PATTERN
from dnd_audio.timecode import parse_frame_rate, parse_timecode
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE as _CANONICAL_SAMPLE_RATE

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ActivityConfig",
    "AsrConfig",
    "BleedConfig",
    "DuplicateConfig",
    "EncodeConfig",
    "EnvelopeConfig",
    "MixConfig",
    "RecoveryConfig",
    "ScoringConfig",
    "SessionConfig",
    "SourceTimeOverride",
    "StageScope",
    "SyncQaConfig",
    "TimecodeConfig",
    "TrackConfig",
    "TranscriptConfig",
    "VadConfig",
    "config_hash",
    "load_session_config",
    "resolved_config",
    "stage_config",
    "stage_config_hash",
]

#: Bumped when the meaning of an existing field changes. Part of every cache identity
#: built on this configuration (INV-08).
CONFIG_SCHEMA_VERSION: Final = 1

#: Rate labels the DJI receivers can be set to. Kept in step with
#: :data:`dnd_audio.timecode.FRAME_RATE_LABELS` by ``tests/test_timecode.py``.
FrameRateLabel = Literal["23.98F", "24F", "25F", "29.97F", "29.97DF", "30F", "50F", "60F"]

#: ``infer_forward`` may infer a single forward midnight rollover when chunk sequence
#: and session span make it unambiguous. ``reject`` never infers and fails instead;
#: the spec named only the former, so the latter is this project's choice (ADR-0005).
RolloverPolicy = Literal["infer_forward", "reject"]

#: ``reject`` makes a material chunk overlap fatal; ``nudge_later`` places the later chunk
#: immediately after the earlier one and records the shift. The spec requires "an explicit
#: policy rather than silently discarding audio" without naming values, so these are this
#: project's (ADR-0010). There is deliberately no value that trims or drops a chunk.
ChunkOverlapPolicy = Literal["reject", "nudge_later"]

#: MPEG-1 Layer III bitrates. Anything else makes the encoder pick for us.
_MP3_BITRATES: Final = frozenset({32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320})

_IDENTIFIER = r"^[a-z0-9][a-z0-9_-]*$"

Identifier = Annotated[str, Field(pattern=_IDENTIFIER, min_length=1, max_length=64)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _validate_relative_path(value: str) -> str:
    """Reject anything that is not a plain path inside the session directory.

    An absolute path or a ``..`` escape would let a session file read from — and, once
    later milestones write, potentially write to — somewhere the operator did not mean.
    INV-01 depends on output paths being provably inside the session tree.
    """
    if not value or value.strip() != value:
        message = f"path {value!r} is empty or has surrounding whitespace"
        raise ValueError(message)

    path = PurePosixPath(value)
    if path.is_absolute():
        message = f"path {value!r} must be relative to the session directory"
        raise ValueError(message)
    if ".." in path.parts:
        message = f"path {value!r} must not escape the session directory with '..'"
        raise ValueError(message)
    return str(path)


class _Strict(BaseModel):
    """Base for every configuration model: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TimecodeConfig(_Strict):
    """How to interpret the timecode embedded in the source files."""

    frame_rate: FrameRateLabel = "30F"
    #: The ISO calendar date of timecode day zero. Never inferred from a date-looking
    #: ``session_id`` — the spec forbids it, because the two can legitimately differ.
    origin_date: dt.date | None = None
    #: The absolute timecode corresponding to session time zero. When null, session
    #: zero is the earliest normalized valid source time.
    origin_timecode: str | None = None
    rollover_policy: RolloverPolicy = "infer_forward"
    #: What to do when two chunks of one track claim overlapping time. The spec requires
    #: "an explicit policy rather than silently discarding audio" and names no values;
    #: these are ADR-0010's. Overlaps within the quantization tolerance are resolved under
    #: either policy — at 29.97 fps a timecode start is quantized to 1602 samples, so
    #: ordinary contiguous chunks routinely appear to overlap by less than a frame.
    #: Neither value discards a sample.
    chunk_overlap_policy: ChunkOverlapPolicy = "reject"
    #: How coarsely this hardware writes `bext.time_reference`, in samples at the file's own
    #: rate. **Not derivable from the file**: FFprobe does not surface the iXML that declares
    #: the frame rate, and OQ-024 showed the receiver's configured rate does not reach an
    #: `orig` file at all — so it is configuration with a measured default. 1600 samples is
    #: what OQ-004 measured on the DJI Mic 3 at 48 kHz, one frame at 30 fps. A recorder that
    #: really is sample-exact is stated as 1, which restores the pre-M8 behaviour exactly.
    bwf_reference_quantum_samples: int = Field(default=1600, ge=1, le=48000)

    @model_validator(mode="after")
    def _check_origin(self) -> TimecodeConfig:
        if self.origin_timecode is None:
            return self
        if self.origin_date is None:
            message = (
                "timecode.origin_timecode requires timecode.origin_date: an absolute "
                "timecode without a date cannot be placed on a calendar"
            )
            raise ValueError(message)
        try:
            parse_timecode(self.origin_timecode, parse_frame_rate(self.frame_rate))
        except TimecodeError as exc:
            message = f"timecode.origin_timecode: {exc}"
            raise ValueError(message) from exc
        return self


class TrackConfig(_Strict):
    """One physical transmitter, and the person wearing it.

    The roster is durable: a track stays configured whether or not its wearer attended.
    Presence is decided by discovery in M1, not by editing this list.
    """

    track_id: Identifier
    receiver_id: Identifier
    receiver_channel: int = Field(ge=1, le=2)
    speaker_id: Identifier
    speaker_name: str = Field(min_length=1)
    input: str

    @field_validator("input")
    @classmethod
    def _check_input(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def _input_must_be_the_track_directory(self) -> TrackConfig:
        """INV-11, made structural rather than merely asserted.

        Without this, `track_id: tx-a` with `input: raw/tx-f` validates happily and
        every word Frank says is attributed to Alice — a whole session mis-attributed
        by one transposed letter, with nothing downstream able to notice. Tying the two
        together means the directory really is the identity, which is what the
        invariant claims.

        The directory may live anywhere in the session; only its final component is
        constrained.
        """
        directory = PurePosixPath(self.input).name
        if directory != self.track_id:
            message = (
                f"track {self.track_id!r} reads from {self.input!r}, whose directory is "
                f"{directory!r}. The input directory's name is the track's identity "
                f"(INV-11), so these must match — rename the directory or fix the "
                f"track_id, but do not cross them."
            )
            raise ValueError(message)
        return self


class AsrConfig(_Strict):
    """Model identity and inference parameters. Every field affects output."""

    # `model_revision` collides with pydantic's reserved `model_` prefix; the field
    # names come from the spec's session.yaml, so the namespace guard goes instead.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model: str = Field(default="Qwen/Qwen3-ASR-1.7B", min_length=1)
    #: An exact commit, overriding the one pinned in this build. `None` means "use the
    #: pin". Both exist because a mutable Hugging Face branch must never be resolved
    #: during `process` — and the validator below is what makes that structural rather
    #: than a rule to remember: a branch or tag cannot be written here at all (ADR-0027).
    model_revision: str | None = None
    aligner: str = Field(default="Qwen/Qwen3-ForcedAligner-0.6B", min_length=1)
    aligner_revision: str | None = None
    #: Optional local glossary. Its absence must not block a run.
    context_file: str | None = "glossary.txt"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "bfloat16"] = "auto"
    #: Applies to the entire padded waveform submitted to the model, not just the
    #: unpadded ownership interval. Capped at 120 s: the official package's timestamp
    #: path chunks at 180 s, and the advertised five-minute model limit is not the
    #: package limit. See OQ-009 — if that resolves differently, this cap moves.
    max_segment_s: int = Field(default=120, gt=0, le=120)
    #: Explicit rather than inheriting the upstream wrapper's 512. Part of the ASR
    #: cache key, so changing it must re-run the work.
    max_new_tokens: int = Field(default=1024, gt=0)
    #: How close to `max_new_tokens` a retokenized response must land before it is treated
    #: as cut off at the generation ceiling. `qwen-asr` 0.0.6 exposes no finish reason —
    #: its high-level path decodes to strings and discards everything else — so this
    #: heuristic is the whole of truncation detection rather than a fallback (ADR-0028).
    #: Too small and a genuinely truncated response is transcribed as complete; too large
    #: and a merely long one is split and retried for nothing. Both directions are guesses
    #: about *this* model until the smoke test measures them (**OQ-018**). In the ASR cache
    #: key, because it decides whether a retry happened and therefore what the text is.
    truncation_margin_tokens: int = Field(default=16, ge=0, le=512)

    @field_validator("context_file")
    @classmethod
    def _check_context_file(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_path(value)

    @field_validator("model", "aligner")
    @classmethod
    def _check_repository(cls, value: str, info: ValidationInfo) -> str:
        """The two repositories this build has snapshots for, and no others.

        These fields exist because the spec's `session.yaml` has them, and they reach the
        cache key and the report as the identity of what produced a transcript. But
        `_default_transcriber` verifies and loads the *descriptors* — there is no code path
        that fetches an arbitrary repository, and `models fetch` has no way to install one.
        Accepting another name would therefore run Qwen and record something else as having
        produced the result: a cache key and an ingest report that name weights that were
        never loaded, which is worse than a refusal because nothing downstream can detect
        it (INV-08). Refused at configuration load, before any work. Raised by M6b's code
        review.
        """
        expected = QWEN3_ASR.repository if info.field_name == "model" else QWEN3_ALIGNER.repository
        if value != expected:
            message = (
                f"asr {info.field_name} {value!r} is not a repository this build carries. "
                f"It has snapshots for {expected!r} only, pinned by commit, and no command "
                f"can install another — so a run would load {expected!r} and record "
                f"{value!r} as what produced the transcript. Pin a different *revision* of "
                f"the same repository with `{info.field_name}_revision` instead."
            )
            raise ValueError(message)
        return value

    @field_validator("model_revision", "aligner_revision")
    @classmethod
    def _check_revision(cls, value: str | None) -> str | None:
        """A revision is a commit, or it is refused here.

        The spec requires `process` to use the model lock "rather than re-resolving a
        moving branch". Validating the *shape* is what makes that structural: with no
        branch name accepted anywhere in configuration, there is nothing left in the
        system for a run to re-resolve, and an offline `process` cannot be asked to do
        something it cannot do (ADR-0027).
        """
        if value is None:
            return None
        # `fullmatch`, not `match`: Python's `$` also matches immediately before a trailing
        # newline, so `re.match` accepts a 41-character value ending in one — reachable from
        # a YAML block scalar. The directory layout is keyed by this string, so a revision
        # with an invisible newline would name a directory nothing installs into. Raised by
        # M6b's code review.
        if not re.fullmatch(REVISION_PATTERN, value):
            message = (
                f"asr revision {value!r} is not a commit. Give the full 40-character "
                f"lowercase hexadecimal commit sha — a branch or tag moves, and this "
                f"pipeline resolves nothing at run time, so it would have no way to know "
                f"which weights produced a transcript. Leaving it unset uses the revision "
                f"pinned in this build."
            )
            raise ValueError(message)
        return value


class VadConfig(_Strict):
    """Where speech starts and stops on one track.

    Every default here is a number chosen against synthetic audio and **OQ-017** is the
    record of that: a real session is what will move them.
    """

    #: A frame above this is speech. Silero's own documented default, and the value its
    #: authors say is a good "lazy" choice on most material (OQ-017).
    speech_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    #: Speech continues until a frame falls below this. Two thresholds rather than one,
    #: because a single one chops a word in half every time a probability wobbles across
    #: it mid-syllable (OQ-017).
    silence_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    #: Regions shorter than this are noise, not words (OQ-017).
    min_speech_ms: int = Field(default=250, ge=0, le=10_000)
    #: A dip shorter than this is a stop consonant, not the end of a turn (OQ-017).
    min_silence_ms: int = Field(default=100, ge=0, le=10_000)
    #: Regions closer together than this become one candidate. Wider than
    #: `min_silence_ms`: that one is about the detector's own hysteresis, this one is
    #: about not handing M4 a hundred fragments of one sentence (OQ-017).
    merge_gap_ms: int = Field(default=200, ge=0, le=10_000)
    #: Added to both ends so a padded region does not clip the word it contains (OQ-017).
    pad_ms: int = Field(default=30, ge=0, le=1000)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> VadConfig:
        if self.silence_threshold >= self.speech_threshold:
            message = (
                f"vad.silence_threshold ({self.silence_threshold}) must be below "
                f"vad.speech_threshold ({self.speech_threshold}). Equal thresholds are no "
                f"hysteresis at all, and an inverted pair would start speech where it ends."
            )
            raise ValueError(message)
        return self


class BleedConfig(_Strict):
    """When one track's candidate is really another track's voice (ADR-0014).

    Suppression requires all three of a score margin, a correlation, and a level below the
    veto. Any one of them failing keeps the candidate, because losing real overlapped
    speech is worse than spending more ASR compute.
    """

    #: How much better another track's source score must be. Fractional, against scores in
    #: [0, 1] (OQ-017).
    min_score_margin: float = Field(default=0.15, gt=0.0, le=1.0)
    #: Below this normalized peak correlation the two signals are not the same sound, and
    #: nothing is suppressed however loud the other track is (OQ-017).
    min_correlation: float = Field(default=0.5, gt=0.0, le=1.0)
    #: The veto. A candidate whose band-limited level is within this many dB of its own
    #: track's speech reference is never suppressed: a lav hearing its wearer at the
    #: wearer's normal level is not hearing someone else, however correlated the two
    #: tracks are (ADR-0014, OQ-017).
    veto_db: float = Field(default=12.0, gt=0.0, le=60.0)
    #: How much of a long overlap is actually correlated. Bounded, because this is one of
    #: the few places holding a contiguous array and an unbounded value in a session file
    #: would be an INV-07 violation an operator could configure.
    correlation_window_ms: int = Field(default=2000, gt=0, le=30_000)
    #: Candidates needed before a track has a speech reference at all, when the reference is
    #: estimated from an **unclassified mixture** — the bootstrap pass, and the fallback for a
    #: track that won nothing. Below it the veto cannot be evaluated and the graph says so,
    #: rather than treating a reference of zero as a measurement (OQ-017).
    min_reference_candidates: int = Field(default=3, ge=1, le=100)
    #: The same floor for the population that has already *won* attribution (ADR-0029). One,
    #: because a winner is direct evidence that this is the wearer speaking, where three of a
    #: mixture are not — the two populations are not equally good and a single number would be
    #: set for the wrong one. Two floors rather than one knob selecting the same candidates
    #: twice, which ADR-0014's amendment rightly warns against (OQ-017).
    min_attributed_reference_candidates: int = Field(default=1, ge=1, le=100)


class ScoringConfig(_Strict):
    """How four pieces of evidence become one source score.

    Weights are relative and normalized by their sum, so doubling all four changes
    nothing. The spans are the dynamic ranges each term is mapped over; a term is clamped
    at both ends rather than allowed to dominate the total from one outlier.
    """

    #: Track-relative speech level: is this as loud as this wearer normally is? (OQ-017)
    level_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    #: The detector's own confidence.
    confidence_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Cross-track dominance: is this louder than everyone else here? (OQ-017)
    dominance_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Independence: a candidate strongly correlated with another track is more likely a
    #: copy of it than a voice of its own (OQ-017).
    correlation_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    #: dB below its own track's reference at which the level term reaches zero (OQ-017).
    level_span_db: float = Field(default=30.0, gt=0.0, le=120.0)
    #: dB of cross-track level difference at which the dominance term saturates (OQ-017).
    dominance_span_db: float = Field(default=20.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def _check_weights(self) -> ScoringConfig:
        total = (
            self.level_weight
            + self.confidence_weight
            + self.dominance_weight
            + self.correlation_weight
        )
        if total <= 0.0:
            message = (
                "activity.scoring weights sum to zero, so every candidate would score the "
                "same and the bleed gate could never prefer one track over another"
            )
            raise ValueError(message)
        return self


class ActivityConfig(_Strict):
    """Pre-ASR activity and bleed-gate parameters."""

    #: Bleed arrives late. Similarity must be measured over a bounded lag window
    #: rather than at zero lag, or a delayed copy of the same speech looks unrelated.
    correlation_max_lag_ms: int = Field(default=30, gt=0, le=1000)
    vad: VadConfig = Field(default_factory=VadConfig)
    bleed: BleedConfig = Field(default_factory=BleedConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


class DuplicateConfig(_Strict):
    """When two tracks' segments are the same utterance heard twice (spec, Milestone 3).

    Collapse needs **all three** of substantial temporal overlap, strongly similar normalized
    text, and supporting acoustic evidence — and the acoustic half comes from the activity
    graph's own per-pair measurements rather than from a second correlator (ADR-0017).

    The thresholds split by what settles them. The text ones are calibrated against *Qwen's*
    error distribution — how differently one model transcribes the same utterance heard on two
    lavs (OQ-018). The acoustic ones are about a real room, which is OQ-017's question.
    """

    #: How much of the shorter segment the two must share before overlap counts at all.
    #: Compared by integer cross-multiplication against the sample counts, never as a float
    #: ratio at a boundary (OQ-018).
    min_overlap_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    #: Normalized-text similarity, quantized to per-mille before it is compared or recorded.
    #: High by design: a duplicate is the *same words*, and two people saying similar things
    #: is ordinary conversation (OQ-018).
    min_text_similarity: float = Field(default=0.85, gt=0.0, le=1.0)
    #: Below this many words, text similarity is ignored entirely and nothing collapses. The
    #: spec names the case: "yes" and "no" match perfectly and mean two people agreeing
    #: (OQ-018).
    min_text_words: int = Field(default=4, ge=1, le=100)
    #: The same floor in characters, because four short words are still not evidence (OQ-018).
    min_text_chars: int = Field(default=12, ge=1, le=1000)
    #: The graph's peak normalized correlation between the two candidates. Every pair that
    #: exists must reach it, not merely the best one — the conservative direction (OQ-017).
    min_correlation: float = Field(default=0.5, gt=0.0, le=1.0)
    #: Or compelling source dominance: how much better the winner's source score must be
    #: when the correlation alone is not decisive (OQ-017).
    min_score_margin: float = Field(default=0.1, gt=0.0, le=1.0)


class TranscriptConfig(_Strict):
    """How activity candidates become ASR requests, and what happens to the answers.

    Every default here is a guess about a model this milestone deliberately does not have.
    **OQ-018** is the record of that, and it is cited from each one so `rg 'OQ-018'` finds
    them together when M6b can finally measure them.
    """

    #: Audio added to each side of an ownership interval so the model hears the context around
    #: an utterance and does not clip its first or last word. Padding is context, never
    #: content: a word inside it and inside no ownership interval is dropped (ADR-0020,
    #: OQ-018).
    pad_ms: int = Field(default=500, ge=0, le=10_000)
    #: Adjacent candidates on one track closer together than this are merged into one request.
    #: The audio merges; ownership does not (ADR-0017). Wider than `activity.vad.merge_gap_ms`,
    #: which is about not fragmenting a sentence; this one is about not paying a model's
    #: fixed cost per fragment (OQ-018).
    merge_gap_ms: int = Field(default=1500, ge=0, le=60_000)
    #: How much two retained segments of *different* speakers must share before either is
    #: marked `overlap`. The spec defines the flag in terms of "at least the configured
    #: overlap threshold" and leaves the number to us (OQ-018).
    overlap_min_ms: int = Field(default=250, ge=0, le=60_000)
    #: A **global budget of additional submissions** spent resolving one truncated request,
    #: not a recursion depth — depth doubles, and a depth of 3 would be fifteen calls
    #: (ADR-0020, OQ-018).
    max_truncation_retries: int = Field(default=4, ge=0, le=32)
    #: A child of a truncation split shorter than this is not split again. Without it the
    #: recursion produces sub-word requests whose transcription means nothing (OQ-018).
    min_split_core_ms: int = Field(default=2000, gt=0, le=120_000)
    duplicate: DuplicateConfig = Field(default_factory=DuplicateConfig)


class SyncQaConfig(_Strict):
    """Optional clap cross-correlation, as synchronization QA only.

    The spec is explicit that this "should report disagreement with timecode, not override
    valid timecode automatically", so nothing here can move a sample. Off by default: it
    costs a correlation over two windows per track and answers a question most sessions do
    not have.

    Measuring near *both* ends is the point. A constant lag is a constant timecode offset;
    a lag that *changes* between the start and the end is evidence of sample-clock drift
    (OQ-006), which is the thing the MVP assumes is negligible and never corrects.
    """

    enabled: bool = False
    #: Seconds of audio correlated at each end. Bounded, because this window is one of the
    #: few places in the pipeline that holds a contiguous array, and an unbounded value in
    #: a session file would be an INV-07 violation an operator could configure.
    window_s: int = Field(default=30, gt=0, le=300)
    #: How far apart two tracks' transients may be and still be matched. Wider than
    #: `activity.correlation_max_lag_ms`, which is about acoustic bleed across a table;
    #: this one is about receivers whose timecode disagrees.
    max_lag_ms: int = Field(default=100, gt=0, le=5000)
    #: A start-to-end change in measured lag beyond this warns. Integer milliseconds
    #: rather than a float, so a threshold comparison cannot depend on binary rounding.
    drift_warn_ms: int = Field(default=5, gt=0, le=1000)
    #: Below this normalized peak correlation, QA reports that it found no shared
    #: transient instead of reporting a lag. Without it, correlating two independent noise
    #: floors yields a confident-looking number for a clap that was never recorded. The
    #: value is a starting point; H1 and H2 are what will tune it (OQ-006).
    min_correlation: float = Field(default=0.5, gt=0.0, le=1.0)


class EnvelopeConfig(_Strict):
    """How the activity graph becomes a gain per track per moment (ADR-0022).

    Everything here changes the samples in the mix, so this whole section — and only this
    section — enters the render cache identity. The encode settings beside it reach the MP3,
    which is regenerated on every run and never cached (ADR-0023).

    Every default is a number chosen against 10.5 seconds of shaped noise. **OQ-019** is the
    record of that.
    """

    #: Gains are computed per control frame and linearly interpolated to samples, so the
    #: applied gain is continuous by construction. 1 kHz is 48 samples per frame; at 100 Hz a
    #: 10 ms attack would be a single frame, which is no slew limit at all (OQ-019).
    control_rate_hz: int = Field(default=1000, gt=0, le=48_000)
    #: Short, so a word is not clipped. A third of `activity.vad.pad_ms`, so the ramp finishes
    #: inside the padding the candidate already carries (OQ-019).
    attack_ms: int = Field(default=10, gt=0, le=1000)
    #: Longer, so a channel change does not click or pump (OQ-019).
    release_ms: int = Field(default=300, gt=0, le=10_000)
    #: The weight every channel keeps when nobody on it is speaking, so silence blends room
    #: tone rather than muting five lavs. Only its *ratio* to `min_active_share` matters:
    #: during silence every weight is equal and the shares are 1/N whatever this is (OQ-019).
    room_tone_share: float = Field(default=0.005, gt=0.0, lt=1.0)
    #: The weight an active channel keeps however badly it scored. Without this floor,
    #: dominance would scale with the winner's score and the gate criterion below would be a
    #: property of the fixture rather than of the rule (ADR-0022, OQ-019).
    min_active_share: float = Field(default=0.5, gt=0.0, le=1.0)
    #: The spec's "clamp correction to a safe range", in dB either way. Deliberately
    #: conservative: it costs twice this much of the dominance margin, and a wearer whose lav
    #: is further out than this is a capture problem a mixer should not paper over (OQ-019).
    max_level_correction_db: float = Field(default=6.0, ge=0.0, le=24.0)
    #: The gate's "configured attenuation margin": how far a solo speaker's applied
    #: coefficient must sit above every inactive channel's once the attack has finished.
    #: Validated to be *achievable* below, rather than merely asserted by a test (OQ-019).
    solo_attenuation_margin_db: float = Field(default=20.0, gt=0.0, le=120.0)
    #: The gate's "nontrivial audible gain" during genuine overlap, against the applied
    #: coefficient. **Derived, not estimated** — see :meth:`guaranteed_overlap_gain_db`, which
    #: `SessionConfig` validates this against for the session's own track count. The estimate
    #: that used to sit here ("two channels share roughly -6 dB each, and the clamp can take
    #: another 6" → -15) was 0.66 dB optimistic for a six-track session, because the quieter
    #: speaker holds `min_active_share` while the louder holds 1.0 and four room-tone floors
    #: still take a share. Found by M5's code review (OQ-019).
    overlap_min_gain_db: float = Field(default=-16.0, lt=0.0, ge=-60.0)

    def guaranteed_overlap_gain_db(self, track_count: int) -> float:
        """The worst applied coefficient a genuine second speaker can be reduced to, in dB.

        The gate's overlap criterion, stated as a bound rather than as something a fixture
        happens to satisfy — the same treatment `solo_attenuation_margin_db` already gets, and
        the omission M5's code review found.

        The worst two-speaker case: this speaker scored zero, so its weight is
        `min_active_share`; the other scored full, so its weight is 1.0; the remaining
        ``track_count - 2`` channels sit at `room_tone_share`; and this speaker's own level
        correction is the full clamp downward. Three or more simultaneous speakers divide
        further still, but "genuine two-person overlap" is what the criterion says and what
        `overlap_min_gain_db` is compared against.
        """
        others = self.room_tone_share * max(track_count - 2, 0)
        share = self.min_active_share / (1.0 + self.min_active_share + others)
        return 20.0 * math.log10(share) - self.max_level_correction_db

    @model_validator(mode="after")
    def _check_grid(self) -> EnvelopeConfig:
        """The control grid is exact, or it is not a grid.

        Two separate properties, and the second does not follow from the first: 800 Hz divides
        48000 and an 11 ms attack is 8.8 frames of it. Caught by M5's plan review, which was
        right that stating "the rate divides the sample rate" accounts for only half of it.
        """
        if _CANONICAL_SAMPLE_RATE % self.control_rate_hz:
            message = (
                f"mix.envelope.control_rate_hz={self.control_rate_hz} does not divide the "
                f"{_CANONICAL_SAMPLE_RATE} Hz session grid, so a control frame would not be a "
                f"whole number of samples"
            )
            raise ValueError(message)
        for name, milliseconds in (("attack_ms", self.attack_ms), ("release_ms", self.release_ms)):
            if milliseconds * self.control_rate_hz % 1000:
                message = (
                    f"mix.envelope.{name}={milliseconds} is "
                    f"{milliseconds * self.control_rate_hz / 1000} control frames at "
                    f"{self.control_rate_hz} Hz, not a whole number. A slew limit expressed in "
                    f"fractional frames is not a limit anything can check."
                )
                raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _check_margin_is_achievable(self) -> EnvelopeConfig:
        """Refuse a configuration whose own dominance criterion cannot be met (ADR-0022).

        The two floors bound the share ratio at `min_active_share / room_tone_share`; the
        level correction can erode it by twice its clamp, when a quiet track is lifted while a
        loud one is cut. A configuration promising more than that would produce a mix that
        fails its own gate, silently, on the first session where a correction was needed.
        """
        available = 20.0 * math.log10(self.min_active_share / self.room_tone_share)
        guaranteed = available - 2.0 * self.max_level_correction_db
        if guaranteed < self.solo_attenuation_margin_db:
            message = (
                f"mix.envelope.solo_attenuation_margin_db={self.solo_attenuation_margin_db} dB "
                f"is not achievable: min_active_share/room_tone_share gives {available:.2f} dB "
                f"of separation and max_level_correction_db={self.max_level_correction_db} can "
                f"erode {2.0 * self.max_level_correction_db:.2f} of it, leaving "
                f"{guaranteed:.2f} dB. Lower room_tone_share, raise min_active_share, or "
                f"tighten the correction clamp."
            )
            raise ValueError(message)
        return self


class EncodeConfig(_Strict):
    """Everything after the render boundary: what the MP3 must measure, and what to do if it
    does not (ADR-0023).

    None of this enters the render cache identity. The intermediate is written at unity master
    gain, so changing a target or a tolerance re-encodes rather than re-mixing six tracks.
    """

    #: How far the decoded MP3's integrated loudness may sit from `integrated_lufs`. The
    #: spec's own number.
    loudness_tolerance_lu: float = Field(default=1.0, gt=0.0, le=10.0)
    #: The spec's "documented measurement tolerance" on the true-peak ceiling. FFmpeg's
    #: `ebur128` summary reports one decimal place, so 0.1 dB of this is pure quantization and
    #: the rest is margin for measuring a decode rather than the encoder's own model (OQ-020).
    true_peak_tolerance_db: float = Field(default=0.3, gt=0.0, le=6.0)
    #: The spec's "within one MP3 frame (or another documented codec-appropriate tolerance)",
    #: applied to the **decoded sample count**. One MPEG-1 Layer III frame is 1152 samples,
    #: 24 ms at 48 kHz (OQ-020).
    duration_tolerance_frames: int = Field(default=1, ge=0, le=100)
    #: Additional encodes spent walking the gain down under a true-peak overshoot. Exhausting
    #: it **fails the mix stage** rather than claiming a compliance nothing demonstrated
    #: (OQ-020).
    max_retries: int = Field(default=3, ge=0, le=16)
    #: A ceiling on the two-pass loudness gain. A normalizer with no ceiling turns a session
    #: nobody spoke in into 50 dB of amplified noise floor (OQ-019).
    max_master_gain_db: float = Field(default=30.0, gt=0.0, le=60.0)
    #: Below this the mix is left un-normalized, with a warning. Not hypothetical: a session
    #: where the detector found nothing has every track at the room-tone share, and that is a
    #: correct outcome to report rather than an input to amplify (OQ-019).
    silence_floor_lufs: float = Field(default=-50.0, le=0.0, ge=-70.0)


class MixConfig(_Strict):
    """Automix and encode targets.

    The three fields the spec's own `session.yaml` names stay at this level; the rest is split
    by which side of the render boundary it falls on (ADR-0023).
    """

    integrated_lufs: float = Field(default=-16.0, ge=-70.0, le=0.0)
    #: Applies to the decoded MP3, not merely the lossless intermediate.
    true_peak_dbtp: float = Field(default=-1.5, ge=-20.0, le=0.0)
    mp3_bitrate_kbps: int = 128
    envelope: EnvelopeConfig = Field(default_factory=EnvelopeConfig)
    encode: EncodeConfig = Field(default_factory=EncodeConfig)

    @field_validator("mp3_bitrate_kbps")
    @classmethod
    def _check_bitrate(cls, value: int) -> int:
        if value not in _MP3_BITRATES:
            allowed = ", ".join(str(rate) for rate in sorted(_MP3_BITRATES))
            message = f"mp3_bitrate_kbps={value} is not an MPEG-1 Layer III bitrate ({allowed})"
            raise ValueError(message)
        return value


class SourceTimeOverride(_Strict):
    """Exceptional replacement timing for one source file.

    Exists because file presence cannot recover missing timing metadata, and INV-12
    forbids inventing it. An override supplies evidence from outside the file — a field
    log — and says so in ``reason``, which is required so the manifest and report can
    show why a time was not read from the source.
    """

    #: When present, the source's actual hash must match before the override applies.
    sha256: Sha256Hex | None = None
    recording_date: dt.date | None = None
    start_timecode: str | None = None
    #: Signed, at the canonical 48 kHz rate, relative to session time zero.
    start_offset_samples: int | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_timing(self) -> SourceTimeOverride:
        if self.start_timecode is not None and self.start_offset_samples is not None:
            message = "an override may supply start_timecode or start_offset_samples, not both"
            raise ValueError(message)
        if (
            self.recording_date is None
            and self.start_timecode is None
            and self.start_offset_samples is None
        ):
            message = (
                "an override must supply at least one of recording_date, start_timecode, "
                "or start_offset_samples; otherwise it overrides nothing"
            )
            raise ValueError(message)
        return self


class RecoveryConfig(_Strict):
    """Escape hatches, all of them off by default and all of them recorded."""

    #: Consuming a processed `edit` file loses the 32-bit-float original. Off unless
    #: the original is genuinely gone.
    allow_processed_audio: bool = False
    #: Keyed by session-relative source path. An override applies only to the file it
    #: names; every affected chunk needs its own evidence.
    source_time_overrides: dict[str, SourceTimeOverride] = Field(default_factory=dict)

    @field_validator("source_time_overrides")
    @classmethod
    def _normalize_keys(cls, value: dict[str, SourceTimeOverride]) -> dict[str, SourceTimeOverride]:
        """Normalize the keys, and reject two spellings of the same path.

        Validating without keeping the normalized form leaves `raw/tx-a/f.wav` and
        `raw//tx-a/./f.wav` as distinct keys: they hash differently (INV-08) and a
        lookup by the discovered path finds only one of them, so an override written
        the second way silently does nothing.
        """
        normalized: dict[str, SourceTimeOverride] = {}
        for key, override in value.items():
            canonical = _validate_relative_path(key)
            if canonical in normalized:
                message = (
                    f"two source_time_overrides refer to the same file {canonical!r}; "
                    f"only one can apply"
                )
                raise ValueError(message)
            normalized[canonical] = override
        return normalized


class SessionConfig(_Strict):
    """A complete, self-describing session."""

    schema_version: Literal[1] = 1
    session_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = Field(default="English", min_length=1)
    #: ``auto`` derives the active participants from which configured directories hold
    #: a usable original. An explicit list makes every named track required, which is
    #: the only way to tell an intentional absence from a capture failure.
    active_tracks: Literal["auto"] | list[Identifier] = "auto"
    timecode: TimecodeConfig = Field(default_factory=TimecodeConfig)
    tracks: list[TrackConfig] = Field(min_length=1)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    sync_qa: SyncQaConfig = Field(default_factory=SyncQaConfig)
    mix: MixConfig = Field(default_factory=MixConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)

    @model_validator(mode="after")
    def _check_roster(self) -> SessionConfig:
        # No separate duplicate-input check: `TrackConfig` requires the input
        # directory to be named for its track, so distinct track ids already imply
        # distinct inputs. A check nothing can reach is a check nothing tests.
        self._reject_duplicates("track_id", [track.track_id for track in self.tracks])
        self._reject_duplicates(
            "receiver_id/receiver_channel",
            [f"{track.receiver_id}:{track.receiver_channel}" for track in self.tracks],
        )
        return self

    @model_validator(mode="after")
    def _check_overlap_gain_is_achievable(self) -> SessionConfig:
        """Refuse a configuration whose overlap criterion the share rule cannot deliver.

        The twin of `EnvelopeConfig._check_margin_is_achievable`, and it lives here rather
        than beside it because the guarantee depends on how many tracks the gain is divided
        between — a number only the roster knows. Without it the gate's "both active channels
        retain nontrivial gain" was a property of the score combinations the tests happened to
        use: one speaker at 1000 against one at 0, with the permitted correction, lands at
        -15.66 dB. Found by M5's code review.
        """
        envelope = self.mix.envelope
        guaranteed = envelope.guaranteed_overlap_gain_db(len(self.tracks))
        if guaranteed < envelope.overlap_min_gain_db:
            message = (
                f"mix.envelope.overlap_min_gain_db={envelope.overlap_min_gain_db} dB is not "
                f"achievable across {len(self.tracks)} tracks: the quieter of two genuine "
                f"speakers can be reduced to {guaranteed:.2f} dB, because it holds "
                f"min_active_share={envelope.min_active_share} against the other's full "
                f"weight while {max(len(self.tracks) - 2, 0)} room-tone floors still take a "
                f"share, and max_level_correction_db={envelope.max_level_correction_db} can "
                f"cut it further. Raise min_active_share, lower room_tone_share, tighten the "
                f"correction clamp, or lower the promise."
            )
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _check_active_tracks(self) -> SessionConfig:
        if self.active_tracks == "auto":
            return self
        if not self.active_tracks:
            message = "active_tracks must be 'auto' or a non-empty list of configured track ids"
            raise ValueError(message)
        self._reject_duplicates("active_tracks", list(self.active_tracks))
        known = {track.track_id for track in self.tracks}
        unknown = sorted(set(self.active_tracks) - known)
        if unknown:
            message = (
                f"active_tracks names {', '.join(unknown)}, which are not in the roster. "
                f"A track must be configured before it can be required."
            )
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _check_override_timecodes(self) -> SessionConfig:
        frame_rate = parse_frame_rate(self.timecode.frame_rate)
        for source, override in sorted(self.recovery.source_time_overrides.items()):
            if override.start_timecode is None:
                continue
            try:
                parse_timecode(override.start_timecode, frame_rate)
            except TimecodeError as exc:
                message = f"recovery.source_time_overrides[{source!r}].start_timecode: {exc}"
                raise ValueError(message) from exc
        return self

    @staticmethod
    def _reject_duplicates(label: str, values: list[str]) -> None:
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            message = f"duplicate {label}: {', '.join(duplicates)}"
            raise ValueError(message)


def load_session_config(path: Path) -> SessionConfig:
    """Read and validate a `session.yaml`.

    Raises:
        ConfigError: if the file is missing, is not a YAML mapping, or fails
            validation. The pydantic report is included verbatim — it names the exact
            field, which is what an operator needs.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"cannot read session configuration at {path}: {exc}"
        raise ConfigError(message) from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        message = f"{path} is not valid YAML: {exc}"
        raise ConfigError(message) from exc

    if not isinstance(raw, dict):
        message = f"{path} must contain a YAML mapping, got {type(raw).__name__}"
        raise ConfigError(message)

    try:
        return SessionConfig.model_validate(raw)
    except ValidationError as exc:
        message = f"{path} is not a valid session configuration:\n{exc}"
        raise ConfigError(message) from exc


def resolved_config(config: SessionConfig) -> dict[str, Any]:
    """The canonical projection every cache identity is built from (INV-08).

    Defaults are materialized and dates become ISO strings, so a session file that
    omits a default is indistinguishable here from one that states it. The schema
    version travels with it, so changing what a field *means* invalidates caches even
    when no value changed.

    The roster and the active-track list are sorted here, though not in the model: both
    are sets keyed by ``track_id``, so reordering them in the file changes nothing about
    the output and must not invalidate a cache. The file keeps its own order, which is
    what error messages and human readers see.
    """
    session = config.model_dump(mode="json")
    session["tracks"] = sorted(session["tracks"], key=lambda track: str(track["track_id"]))
    if isinstance(session["active_tracks"], list):
        session["active_tracks"] = sorted(session["active_tracks"])
    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "session": session,
    }


def config_hash(config: SessionConfig) -> str:
    """Hash the resolved configuration. Stable across key order and omitted defaults."""
    return sha256_bytes(canonical_json(resolved_config(config)).encode("utf-8"))


#: The cached stages, each with its own view of what "the configuration" means (ADR-0016).
StageScope = Literal["inspection", "derivative", "detection", "attribution", "mix"]

_ALL_STAGES: Final[frozenset[StageScope]] = frozenset(
    ("inspection", "derivative", "detection", "attribution", "mix")
)
_PLACEMENT: Final[frozenset[StageScope]] = frozenset(("inspection", "derivative"))

#: Which stages each configuration field can change the *output bytes* of.
#:
#: Read as data rather than inferred from call sites, because the property that matters is
#: not "which field does this stage read" but "which field can change what this stage
#: writes", and the two differ wherever one value is derived from another.
#:
#: **Generous, not minimal.** A field is excluded only where its exclusion is provable: the
#: failure mode of a too-narrow key is a stale artifact served as current, which is silent,
#: and the failure mode of a too-broad key is recomputation, which is merely slow. When a
#: new field's blast radius is unclear, it belongs in the broader scope.
#:
#: A key may name a nested path. ``tests/test_config.py`` asserts this table is exhaustive
#: over :class:`SessionConfig`, so a new section must be classified deliberately rather than
#: defaulting to "affects nothing".
_FIELD_SCOPES: Final[dict[str, frozenset[StageScope]]] = {
    # Identity travels everywhere: it is what a cached artifact belongs to.
    "schema_version": _ALL_STAGES,
    "session_id": _ALL_STAGES,
    # The roster decides which audio exists at all, so every stage depends on it (INV-11).
    "tracks": _ALL_STAGES,
    # Placement: which files are selected, where their samples land, and what overrides
    # moved them. The 16 kHz derivative is a function of the segment map, so it inherits
    # everything the map depends on.
    #
    # The mix is *not* here even though it plainly depends on placement, and this is the same
    # exception `asr` and `transcript` get below: the render identity carries the timeline's
    # own sha256 and the activity graph's `attribution_cache_key`, each of which is already
    # downstream of every one of these sections. Restating the same facts in a second place
    # buys nothing and creates somewhere for the two to disagree.
    "active_tracks": _PLACEMENT,
    "timecode": _PLACEMENT,
    "recovery": _PLACEMENT,
    # Detection is inference over that audio; attribution consumes detections and compares
    # them, so it depends on the whole activity section including the VAD half.
    "activity.vad": frozenset(("detection", "attribution")),
    "activity.bleed": frozenset(("attribution",)),
    "activity.scoring": frozenset(("attribution",)),
    "activity.correlation_max_lag_ms": frozenset(("attribution",)),
    # The mix, split at the render boundary (ADR-0023). The envelope decides every sample of
    # the lossless intermediate, which is cached. Everything after it decides only the MP3,
    # which is regenerated on every run and never cached — so keying the intermediate on a
    # bitrate or a tolerance would re-mix six four-hour tracks to change a number that cannot
    # reach it.
    "mix.envelope": frozenset(("mix",)),
    "mix.integrated_lufs": frozenset(),
    "mix.true_peak_dbtp": frozenset(),
    "mix.mp3_bitrate_kbps": frozenset(),
    "mix.encode": frozenset(),
    # Reaches none of the stages *this table covers*. `title` and `language` are carried into
    # outputs; `title` in particular reaches the MP3's ID3 tags, which are written every run
    # beside the encode settings above and are likewise never cached. `sync_qa` produces
    # warnings and report decisions but never a sample.
    #
    # `asr` and `transcript` are the subtle entries: they genuinely change what the ASR stage
    # submits and returns, and that stage caches. Its identity is built where it is used
    # rather than from a projection here, because the key is content-addressed on the audio
    # actually submitted plus the inference parameters actually used (ADR-0019) — a
    # projection would restate the same facts in a second place that could disagree.
    "title": frozenset(),
    "language": frozenset(),
    "asr": frozenset(),
    "transcript": frozenset(),
    "sync_qa": frozenset(),
}


def stage_config(config: SessionConfig, stage: StageScope) -> dict[str, Any]:
    """The projection of the resolved configuration one cached stage depends on.

    Built from :func:`resolved_config`, so it inherits the same normalization: defaults are
    materialized and the roster is sorted, and a session file that omits a default projects
    identically to one that states it.
    """
    session = resolved_config(config)["session"]
    projected: dict[str, Any] = {}
    for path, stages in _FIELD_SCOPES.items():
        if stage not in stages:
            continue
        head, _, tail = path.partition(".")
        if tail:
            projected.setdefault(head, {})[tail] = session[head][tail]
        else:
            projected[head] = session[head]
    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "stage": stage,
        "session": projected,
    }


def stage_config_hash(config: SessionConfig, stage: StageScope) -> str:
    """Hash one stage's projection (INV-08, ADR-0016).

    What a cache identity carries instead of the whole configuration. Tuning a bleed
    threshold must not rebuild gigabytes of 16 kHz PCM that provably cannot depend on it —
    and, in the other direction, the projections are tested to *change* for every section
    they include, because a key that quietly stops changing is how this invariant dies.
    """
    return sha256_bytes(canonical_json(stage_config(config, stage)).encode("utf-8"))
