"""The independent event log: what the operator did, recorded outside `session.yaml`.

**Not a `SessionConfig` field, and that is load-bearing rather than tidy.** A new section on
`SessionConfig` moves `config_hash`, which moves every stage projection, which invalidates the
inspection cache, the derivative cache, the detection and attribution caches and every ASR
result keyed on them (ADR-0016). `archive/config.py` records the same reasoning for the
archive's settings. Setting a search window must not cost gigabytes of re-inference.

**Times are integer milliseconds, converted through the one quantizer.** Taking approximate
seconds and rounding each end of an interval separately would be a second quantization path,
which is how INV-04's single-quantizer rule dies. The operator writes milliseconds;
:func:`searched_intervals` converts once through `determinism.to_samples`.

**The geometry ID is what licenses the strongest claim this project makes.** ADR-0040 permits
calling a start-to-end change *recorder drift* only when the source and every compared
transmitter stayed fixed between the two occurrences. Nothing in the audio can establish that,
and nothing in the pipeline can infer it — so it is an assertion the operator makes, in
writing, at capture time. Two events sharing a geometry ID are a claim that nothing moved
between them. An operator who keeps no log gets differential arrival and no drift
classification, which is the correct outcome rather than a limitation.
"""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dnd_audio.determinism import canonical_json, sha256_bytes, to_samples
from dnd_audio.errors import DndAudioError

__all__ = [
    "EVENT_LOG_SCHEMA_VERSION",
    "EventLogError",
    "EventRole",
    "MarkerEvent",
    "MarkerEventLog",
    "load_event_log",
]

EVENT_LOG_SCHEMA_VERSION: Final = 1

Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")]


class EventLogError(DndAudioError):
    """The event log is absent, unparseable, or internally inconsistent."""

    default_code = "invalid_event_log"


class EventRole(StrEnum):
    """What an occurrence was for.

    Roles come from this log or from the one-event-per-default-window rule, and **never from
    peak strength** — a louder detection is not a more start-like one, and choosing by score
    would let a moved-phone diagnostic displace the pair being measured.
    """

    START = "start"
    END = "end"
    #: Deliberately not part of any start/end pair: a moved-phone or spare play, enumerated
    #: and reported so it is visible, and excluded from every comparison.
    DIAGNOSTIC = "diagnostic"


class _Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarkerEvent(_Document):
    """One deliberate playback, as the operator observed it."""

    role: EventRole
    #: Which waveform was played. Checked against the marker being analyzed, so a take
    #: recorded with one candidate cannot be scored as another.
    marker_name: Identifier
    #: Approximate, half-open, and generous: the operator writes down roughly when they
    #: pressed the button, and the detector searches the window. Integer milliseconds on the
    #: session timeline.
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    #: The order the operator played them in, used to break a tie when two events could
    #: claim the same occurrence. Distinct across the log.
    playback_order: int = Field(ge=0)
    #: An operator's written assertion that the phone and every transmitter were in the same
    #: places as they were for every other event sharing this ID. Absent means "unknown",
    #: which is not the same as "unchanged" and never licenses a drift claim (ADR-0040).
    geometry_id: Identifier | None = None
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.end_ms <= self.start_ms:
            message = (
                f"event at playback_order={self.playback_order} has end_ms={self.end_ms} "
                f"at or before start_ms={self.start_ms}; intervals are half-open and nonempty"
            )
            raise ValueError(message)
        return self

    def interval_samples(self, sample_rate: int) -> tuple[int, int]:
        """The half-open ``[start, end)`` on the sample grid, quantized once.

        Both ends go through :func:`~dnd_audio.determinism.to_samples`, which is the
        project's single quantizer (INV-04). Converting to seconds and rounding each end
        independently is the second float path this avoids.
        """
        return (
            to_samples(Fraction(self.start_ms, 1000), sample_rate),
            to_samples(Fraction(self.end_ms, 1000), sample_rate),
        )


class MarkerEventLog(_Document):
    """Every deliberate playback in one session."""

    schema_version: Literal[1] = EVENT_LOG_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    events: list[MarkerEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        orders = [event.playback_order for event in self.events]
        if len(set(orders)) != len(orders):
            message = f"playback_order must be distinct across the log, got {sorted(orders)}"
            raise ValueError(message)

        for role in (EventRole.START, EventRole.END):
            named = [event for event in self.events if event.role is role]
            if len(named) > 1 and len({event.geometry_id for event in named}) > 1:
                # Not fatal in general — an operator may legitimately log two starts — but
                # two with *different* geometry cannot both anchor the same comparison, and
                # saying so here is cheaper than discovering it after a four-hour session.
                message = (
                    f"{len(named)} events are marked `{role.value}` and they do not share a "
                    f"geometry_id. Mark the extras `diagnostic`: a start/end pair is only "
                    f"comparable when its geometry is asserted unchanged (ADR-0040)."
                )
                raise ValueError(message)

        object.__setattr__(
            self, "events", sorted(self.events, key=lambda event: event.playback_order)
        )
        return self

    def for_marker(self, marker_name: str) -> list[MarkerEvent]:
        """Events recorded for one waveform, in playback order."""
        return [event for event in self.events if event.marker_name == marker_name]

    def digest(self) -> str:
        """A canonical digest of the log, for the analysis identity.

        The log decides which intervals are searched and which occurrence carries which
        role, so a changed log is a changed analysis. Hashing the *model* rather than the
        file means reformatting the YAML does not invalidate anything, while changing a
        number does (ADR-0041).
        """
        return sha256_bytes(canonical_json(self.model_dump(mode="json")).encode("utf-8"))


def load_event_log(path: Path) -> MarkerEventLog:
    """Read and validate an event log.

    Raises:
        EventLogError: if the file cannot be read, is not a YAML mapping, or fails
            validation. The pydantic report is included verbatim — unlike the archive
            configuration, an event log holds no secrets, so naming the exact field is
            the whole point (`config.load_session_config` makes the same choice).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"the marker event log at {path} cannot be read: {exc}"
        raise EventLogError(message, code="event_log_missing") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        message = f"the marker event log at {path} is not valid YAML: {exc}"
        raise EventLogError(message) from exc

    if not isinstance(raw, dict):
        message = (
            f"the marker event log at {path} must be a YAML mapping with `session_id` and "
            f"`events`, got {type(raw).__name__}"
        )
        raise EventLogError(message)

    try:
        return MarkerEventLog.model_validate(raw)
    except ValidationError as exc:
        message = f"the marker event log at {path} is invalid:\n{exc}"
        raise EventLogError(message) from exc
