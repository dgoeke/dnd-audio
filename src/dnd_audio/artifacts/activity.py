"""`activity.json` — who was speaking, decided without ever reading a word.

M3 owns this artifact and **freezes it here** (ADR-0012), because both downstream branches
consume it and the spec forbids text-dependent decisions from reaching the mix (INV-09).
After M3 closes: additive optional fields only.

Six shapes are deliberate, and each has a tempting wrong version.

**There are no floats in this document at all.** Not one, for the reason `timeline.json` has
none: this artifact is byte-stable on an unchanged rerun (INV-02), and a float that is the
quotient of two NumPy reductions is not reliably identical across a library upgrade.
Probabilities, scores, and correlations are integer **per-mille**; levels and level
differences are integer **millibels** (decibels scaled by a hundred). Quantization happens
once, at the boundary, through the project's one rounding rule.

**Two grids, and every field says which one it is on.** `*_sample` fields are canonical
48 kHz session samples — what M5 mixes and M4 requests. `derivative_*_sample` fields are the
16 kHz samples the detector actually decided on. `lag_derivative_samples` keeps its own grid
in its name rather than being scaled by three, which would manufacture precision the
measurement does not have.

**Every interval is half-open**, and the 48 kHz interval always *covers* the 16 kHz one:
start floors, end ceils. Rounding both ends alike shrinks a speech region by up to two
derivative samples, which is how a word loses its first phoneme.

**Suppressed candidates stay, and name the candidate that beat them** — not merely the
track, because "tx-a won" does not say which of tx-a's utterances did. A graph listing only
survivors cannot be audited, and the spec requires rejected alternatives to be recorded.

**Evidence is one record per compared pair**, so a candidate two tracks nearly suppressed
shows both. Collapsing to a best-competitor summary hides exactly the marginal case this
milestone is most likely to get wrong.

**Attribution is the retained candidates and nothing else.** For the MVP baseline the spec
permits attributing every retained candidate to the person wearing that transmitter, so
there is no second structure that can disagree with the first. `ActivityTrack` carries the
speaker mapping once; a candidate names only its track.

The two consumer reads this exists to serve:

* **M4** takes retained candidates in order, merges short adjacent ones, and pads them into
  transcription requests. Suppressed candidates are precisely what it must not transcribe.
* **M5** takes each track's retained candidates as that track's active intervals, with
  `score_permille` as the confidence it smooths a gain envelope from, and
  `speech_reference_mbfs` as the per-track voice-level correction it was asked to estimate.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ACTIVITY_SCHEMA_VERSION",
    "ActivityCandidate",
    "ActivityDecision",
    "ActivityGraph",
    "ActivityNote",
    "ActivityProvenance",
    "ActivityTrack",
    "CandidateEvidence",
    "DetectorIdentity",
    "DetectorInterface",
    "EvidenceOutcome",
    "candidate_id",
]

#: Frozen at M3's close. Only additive optional fields from here; anything else bumps the
#: version (ADR-0005). M4 and M5 both index into this document.
ACTIVITY_SCHEMA_VERSION: Final = 1

#: Width of the zero-padded sample position in a candidate id. Twelve digits covers a
#: 24-hour session at 48 kHz, so ids sort lexically in the same order they sort numerically.
_ID_WIDTH: Final = 12

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

#: A probability, a score, or a normalized correlation, as thousandths.
Permille = Annotated[int, Field(ge=0, le=1000)]

#: A signed difference of two per-mille quantities.
SignedPermille = Annotated[int, Field(ge=-1000, le=1000)]

#: Why one comparison did not suppress a candidate — or that it did. A closed vocabulary,
#: so a reader branches on a value rather than parsing prose (ADR-0014).
EvidenceOutcome = Literal[
    "suppresses",
    "insufficient_margin",
    "insufficient_correlation",
    "vetoed_by_track_level",
]

#: What the detector decided. ``suppressed`` means another track's candidate won this
#: interval; ``retained`` includes every ambiguous case, deliberately.
CandidateDecision = Literal["retained", "suppressed"]


def candidate_id(track_id: str, start_sample: int) -> str:
    """The deterministic id of a candidate, from sorted source identity and time.

    Never from completion order (INV-02). A track's candidates are disjoint after merging,
    so the track and the start position identify one uniquely — and the artifact asserts
    that rather than assuming it.
    """
    return f"cand_{track_id}_{start_sample:0{_ID_WIDTH}d}"


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActivityNote(_Artifact):
    """A warning: worth a human's attention, did not stop the run."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None


class ActivityDecision(_Artifact):
    """An attribution choice worth auditing. Deterministic: no counters, no timings."""

    code: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class DetectorInterface(_Artifact):
    """How the detector was *called*, not merely which weights it holds.

    Part of the identity because a future release carrying the same name with a different
    frame protocol would otherwise produce different answers under an unchanged cache key
    (INV-08, ADR-0013).
    """

    frame_samples: int = Field(gt=0)
    context_samples: int = Field(ge=0)
    #: The recurrent state's dimensions. Empty for a detector that carries none.
    state_shape: list[int] = Field(default_factory=list)
    input_names: list[str] = Field(default_factory=list)
    sample_rate: int = Field(gt=0)


class DetectorIdentity(_Artifact):
    """Everything that decides what the detector would answer.

    A fake carries a ``variant_digest`` instead of a model hash — two scripted detectors with
    different scripts are different detectors, and a cache that could not tell them apart
    would serve one test's answer to another.
    """

    name: str = Field(min_length=1)
    #: Upstream release tag and commit, for an artifact that has them.
    release: str | None = None
    commit: str | None = None
    model_sha256: Sha256Hex | None = None
    #: What executed it, and where. ``None`` for a detector that runs no model.
    runtime: str | None = None
    runtime_version: str | None = None
    execution_provider: str | None = None
    interface: DetectorInterface | None = None
    #: Distinguishes two instances of one implementation. Scripted detectors hash their
    #: script into it; the real one leaves it unset because its model hash already differs.
    variant_digest: Sha256Hex | None = None


class ActivityProvenance(_Artifact):
    """What produced this graph. Deterministic, and part of its cache identity."""

    activity_semantics_version: int = Field(ge=1)
    #: The placement this graph is aligned to. A placement fix moves a chunk without changing
    #: a source byte, and a graph aligned to a timeline that has moved is not obviously wrong.
    timeline_semantics_version: int = Field(ge=1)
    inspection_semantics_version: int = Field(ge=1)
    numpy_version: str = Field(min_length=1)
    scipy_version: str = Field(min_length=1)
    #: ``None`` when no ONNX model ran — the default test suite's case (INV-05).
    onnxruntime_version: str | None = None
    detector: DetectorIdentity
    speech_band_filter_name: str = Field(min_length=1)
    speech_band_filter_identity: Sha256Hex


class CandidateEvidence(_Artifact):
    """One candidate compared against one competing candidate on another track.

    Kept per *pair* rather than summarized per candidate: a candidate that two tracks each
    nearly suppressed is the marginal case worth seeing, and a best-competitor summary is
    exactly where it disappears.
    """

    other_candidate_id: str = Field(min_length=1)
    other_track_id: str = Field(min_length=1)
    #: The full temporal overlap of the two candidates, on the session grid.
    overlap_start_sample: int = Field(ge=0)
    overlap_end_sample: int = Field(gt=0)
    #: How much of it was actually correlated. Bounded by configuration, because a
    #: session-length candidate must not pull a session-length array into memory (INV-07).
    compared_derivative_samples: int = Field(ge=0)
    correlation_permille: Permille
    #: Signed, on the **derivative** grid. Positive means this candidate's audio arrives
    #: later than the other's, which is what bleed crossing a room looks like.
    lag_derivative_samples: int
    #: The other candidate's source score minus this one's.
    score_margin_permille: SignedPermille
    #: The other candidate's band-limited level minus this one's, in millibels.
    level_delta_mb: int
    outcome: EvidenceOutcome

    @model_validator(mode="after")
    def _overlap_is_a_real_interval(self) -> Self:
        if self.overlap_end_sample <= self.overlap_start_sample:
            message = (
                f"evidence against {self.other_candidate_id} claims an overlap of "
                f"[{self.overlap_start_sample}, {self.overlap_end_sample}), which is empty. "
                f"Candidates that do not overlap in time are never compared."
            )
            raise ValueError(message)
        return self


class ActivityCandidate(_Artifact):
    """One region of speech on one track, and what was decided about it."""

    candidate_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    #: Canonical 48 kHz session samples, half-open. What M4 requests and M5 mixes.
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    #: The 16 kHz interval the detector decided on, half-open. The pair above covers it.
    derivative_start_sample: int = Field(ge=0)
    derivative_end_sample: int = Field(gt=0)
    #: Mean and peak detector confidence over the span's frames.
    probability_permille: Permille
    peak_probability_permille: Permille
    #: Band-limited RMS in millibels relative to full scale. Negative in every real signal.
    band_level_mbfs: int
    #: This candidate's level against its own track's speech reference. ``None`` when the
    #: track had too little high-confidence speech to establish one — which is recorded
    #: rather than defaulted, because a missing reference is not a reference of zero.
    relative_level_mb: int | None = None
    #: The combined source score and its four terms, each kept so a wrong attribution is
    #: debuggable from the artifact rather than by re-running with print statements.
    score_permille: Permille
    score_level_permille: Permille
    score_confidence_permille: Permille
    score_dominance_permille: Permille
    score_correlation_permille: Permille
    decision: CandidateDecision
    #: Retained despite mixed evidence. The spec's "default to keeping ambiguous candidates",
    #: made visible instead of implicit.
    ambiguous: bool = False
    suppressed_by_candidate_id: str | None = None
    evidence: list[CandidateEvidence] = Field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end_sample <= self.start_sample:
            message = f"candidate {self.candidate_id} spans an empty interval"
            raise ValueError(message)
        if self.derivative_end_sample <= self.derivative_start_sample:
            message = f"candidate {self.candidate_id} spans an empty derivative interval"
            raise ValueError(message)
        if self.candidate_id != candidate_id(self.track_id, self.start_sample):
            message = (
                f"candidate id {self.candidate_id!r} does not derive from its track and "
                f"start sample. Ids come from sorted source identity and time, never from "
                f"completion order (INV-02)."
            )
            raise ValueError(message)
        if self.peak_probability_permille < self.probability_permille:
            message = (
                f"candidate {self.candidate_id} has a peak probability below its mean, "
                f"which no set of frames can produce"
            )
            raise ValueError(message)

        object.__setattr__(
            self, "evidence", sorted(self.evidence, key=lambda item: item.other_candidate_id)
        )
        seen = [item.other_candidate_id for item in self.evidence]
        if len(set(seen)) != len(seen):
            message = f"candidate {self.candidate_id} compares one competitor twice"
            raise ValueError(message)

        return self._check_decision()

    def _check_decision(self) -> Self:
        """A decision must be consistent with the evidence recorded beside it.

        Split out because these four rules are the ones a future change is most likely to
        break: they are what stop the document from claiming a suppression that nothing in
        it supports.
        """
        suppressing = [item for item in self.evidence if item.outcome == "suppresses"]
        if self.decision == "suppressed":
            if self.suppressed_by_candidate_id is None:
                message = (
                    f"candidate {self.candidate_id} is suppressed but names no suppressor. "
                    f"A rejected alternative that cannot be traced to what beat it is not "
                    f"auditable."
                )
                raise ValueError(message)
            if self.suppressed_by_candidate_id not in {
                item.other_candidate_id for item in suppressing
            }:
                message = (
                    f"candidate {self.candidate_id} says {self.suppressed_by_candidate_id} "
                    f"suppressed it, but no evidence record against that candidate has the "
                    f"`suppresses` outcome"
                )
                raise ValueError(message)
            if self.ambiguous:
                message = (
                    f"candidate {self.candidate_id} is both suppressed and ambiguous. "
                    f"Ambiguity is a reason to keep a candidate, never to drop one."
                )
                raise ValueError(message)
            return self

        if self.suppressed_by_candidate_id is not None:
            message = f"candidate {self.candidate_id} is retained but names a suppressor"
            raise ValueError(message)
        if suppressing:
            message = (
                f"candidate {self.candidate_id} is retained while its evidence against "
                f"{suppressing[0].other_candidate_id} claims to suppress it"
            )
            raise ValueError(message)
        return self


class ActivityTrack(_Artifact):
    """One transmitter's detection results and the person wearing it."""

    track_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    #: The per-track detection entry this track's candidates came from (INV-08).
    detection_cache_key: Sha256Hex
    #: Session-relative. Per-frame probabilities as little-endian ``uint16`` per-mille, one
    #: value per frame, so a bad attribution can be read back frame by frame.
    probability_relative_path: str = Field(min_length=1)
    probability_frames: int = Field(ge=0)
    #: Derivative samples per probability. The detector's frame, recorded because it is the
    #: resolution every probability in this document is quoted at.
    frame_samples: int = Field(gt=0)
    #: What this wearer sounds like when this wearer is talking: the robust band-limited
    #: level of the track's own high-confidence speech. ``None`` when there was too little
    #: of it, which disables the veto for this track rather than silently setting it to zero
    #: (ADR-0014). M5 reads this as its per-track voice-level correction.
    speech_reference_mbfs: int | None = None


class ActivityGraph(_Artifact):
    """The deterministic activity and attribution graph, at ``work/activity.json``."""

    schema_version: Literal[1] = ACTIVITY_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    config_hash: Sha256Hex
    #: Hash of the `timeline.json` this was built from. A graph read beside a timeline that
    #: no longer describes it would place every candidate at the wrong time.
    timeline_sha256: Sha256Hex
    #: This graph's own identity, so a consumer can tell whether it matches the configuration
    #: it is being read under without recomputing anything (INV-08).
    attribution_cache_key: Sha256Hex
    provenance: ActivityProvenance
    #: The grid every ``*_sample`` field is counted on.
    sample_rate: int = Field(gt=0)
    #: The grid every ``derivative_*`` field is counted on, lag included.
    derivative_sample_rate: int = Field(gt=0)
    duration_samples: int = Field(ge=0)
    tracks: list[ActivityTrack] = Field(default_factory=list)
    candidates: list[ActivityCandidate] = Field(default_factory=list)
    warnings: list[ActivityNote] = Field(default_factory=list)
    decisions: list[ActivityDecision] = Field(default_factory=list)

    def retained(self, track_id: str | None = None) -> list[ActivityCandidate]:
        """Retained candidates, in document order, optionally for one track.

        The read both consumers start from, provided once here so M4 and M5 cannot disagree
        about what "retained" means or about the order they arrive in.
        """
        return [
            candidate
            for candidate in self.candidates
            if candidate.decision == "retained"
            and (track_id is None or candidate.track_id == track_id)
        ]

    @model_validator(mode="after")
    def _sort_and_check(self) -> Self:
        object.__setattr__(self, "tracks", sorted(self.tracks, key=lambda t: t.track_id))
        object.__setattr__(
            self,
            "candidates",
            sorted(self.candidates, key=lambda c: (c.start_sample, c.track_id)),
        )
        object.__setattr__(
            self, "warnings", sorted(self.warnings, key=lambda w: (w.code, w.path or "", w.message))
        )
        object.__setattr__(
            self, "decisions", sorted(self.decisions, key=lambda d: (d.code, d.subject, d.detail))
        )

        if self.sample_rate % self.derivative_sample_rate:
            message = (
                f"a session grid of {self.sample_rate} Hz is not a whole multiple of the "
                f"detector's {self.derivative_sample_rate} Hz, so the two cannot be mapped "
                f"exactly"
            )
            raise ValueError(message)

        self._check_candidates()
        return self

    def _check_candidates(self) -> None:
        """Ids are unique, tracks exist, intervals fit, and every reference resolves."""
        known_tracks = {track.track_id for track in self.tracks}
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            message = "two candidates share an id, so a suppression reference is ambiguous"
            raise ValueError(message)
        known_ids = set(ids)
        decimation = self.sample_rate // self.derivative_sample_rate

        for candidate in self.candidates:
            if candidate.track_id not in known_tracks:
                message = (
                    f"candidate {candidate.candidate_id} belongs to track "
                    f"{candidate.track_id!r}, which is not in this graph"
                )
                raise ValueError(message)
            if candidate.end_sample > self.duration_samples:
                message = (
                    f"candidate {candidate.candidate_id} ends at {candidate.end_sample}, past "
                    f"the session's {self.duration_samples} aligned samples"
                )
                raise ValueError(message)

            expected = (
                candidate.start_sample // decimation,
                -(-candidate.end_sample // decimation),
            )
            found = (candidate.derivative_start_sample, candidate.derivative_end_sample)
            if found != expected:
                message = (
                    f"candidate {candidate.candidate_id} spans {found} on the derivative grid "
                    f"but {expected} is what covers [{candidate.start_sample}, "
                    f"{candidate.end_sample}) at 1/{decimation} rate. The start floors and "
                    f"the end ceils, so the session interval always covers the detected one."
                )
                raise ValueError(message)

            self._check_references(candidate, known_ids)

    @staticmethod
    def _check_references(candidate: ActivityCandidate, known_ids: set[str]) -> None:
        for item in candidate.evidence:
            if item.other_candidate_id not in known_ids:
                message = (
                    f"candidate {candidate.candidate_id} carries evidence against "
                    f"{item.other_candidate_id}, which is not in this graph"
                )
                raise ValueError(message)
            if item.other_track_id == candidate.track_id:
                message = (
                    f"candidate {candidate.candidate_id} is compared against "
                    f"{item.other_candidate_id} on its own track. Bleed is a cross-track "
                    f"phenomenon; two candidates on one track are two utterances."
                )
                raise ValueError(message)
