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
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from dnd_audio.determinism import canonical_json, sha256_bytes
from dnd_audio.errors import ConfigError, TimecodeError
from dnd_audio.timecode import parse_frame_rate, parse_timecode

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ActivityConfig",
    "AsrConfig",
    "MixConfig",
    "RecoveryConfig",
    "SessionConfig",
    "SourceTimeOverride",
    "TimecodeConfig",
    "TrackConfig",
    "config_hash",
    "load_session_config",
    "resolved_config",
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
    #: Pinning a revision here overrides the lock `models fetch` writes. Both exist
    #: because a mutable Hugging Face branch must never be resolved during `process`.
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

    @field_validator("context_file")
    @classmethod
    def _check_context_file(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_path(value)


class ActivityConfig(_Strict):
    """Pre-ASR activity and bleed-gate parameters. M3 extends this."""

    #: Bleed arrives late. Similarity must be measured over a bounded lag window
    #: rather than at zero lag, or a delayed copy of the same speech looks unrelated.
    correlation_max_lag_ms: int = Field(default=30, gt=0, le=1000)


class MixConfig(_Strict):
    """Automix and encode targets. M5 extends this."""

    integrated_lufs: float = Field(default=-16.0, ge=-70.0, le=0.0)
    #: Applies to the decoded MP3, not merely the lossless intermediate.
    true_peak_dbtp: float = Field(default=-1.5, ge=-20.0, le=0.0)
    mp3_bitrate_kbps: int = 128

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
