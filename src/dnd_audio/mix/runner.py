"""`dnd-audio mix`: the activity graph and six synchronized tracks become `session.mp3`.

The spec's stage boundary: *"run/cache Milestone 3 as needed, then perform the Milestone 5
automix and MP3 encoding. It must never require ASR or `transcribe` outputs."* That last
sentence is INV-09, and this milestone owns enforcing it — nothing in this package imports the
transcript layer, and the intermediate's bytes are proved not to move when the graph's prose
does.

The ordering is load-bearing and is stated rather than left to be inferred:

1. **Snapshot every file under the raw roots, and refuse outputs that would land inside
   them.** Once, around the whole composed run (INV-01).
2. **Perform the activity stages** through `perform_activity`, which leaves every cache
   staged. The preflight is told a mix is coming, so a session that will not fit is refused
   before it is expanded rather than after.
3. **Verify INV-01, commit the activity caches, write the graph.** The first commit point.
4. **Render the intermediate**, or take it from cache. This is the only stage after inspection
   that reads *source* audio, which is what makes its own commit point load-bearing.
5. **Verify INV-01 again, commit the mix cache.** The second commit point (ADR-0021).
6. **Measure, encode, decode, measure again, retry within budget** — all of it against the
   intermediate, so nothing here can invalidate a source hash (ADR-0023).
7. **Write one report covering six stages**, whichever way it went (INV-13), with the same
   carve-out every other command has: when the report's own location resolves inside a source
   directory, nothing is written and INV-01 wins.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Container, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from dnd_audio.activity import ACTIVITY_RELATIVE_PATH
from dnd_audio.activity.runner import (
    DEFAULT_DETECT_WINDOW,
    ActivityWork,
    DetectorBundle,
    activity_outputs,
    perform_activity,
    remove_activity_artifacts,
)
from dnd_audio.artifacts.activity import ActivityGraph
from dnd_audio.artifacts.report import (
    REPORT_FILENAME,
    Decision,
    IngestReport,
    ReportBuilder,
    ReportWarning,
    StageName,
    StructuredError,
)
from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.config import SessionConfig, config_hash, load_session_config, stage_config_hash
from dnd_audio.determinism import write_json_atomic
from dnd_audio.errors import DiscoveryError, DndAudioError, ExitCode
from dnd_audio.inspection import OUTPUT_DIRNAME
from dnd_audio.mix import MIX_CACHE_DIRNAME, MIX_SEMANTICS_VERSION, MP3_RELATIVE_PATH, MixNote
from dnd_audio.mix.cache import MixCache, mix_identity
from dnd_audio.mix.encode import EncodeAttempt, EncodeError, EncodeResult, encode_mp3
from dnd_audio.mix.envelope import EnvelopeStream
from dnd_audio.mix.levels import MILLIBELS_PER_DB, LevelCorrections, level_corrections
from dnd_audio.mix.loudness import Measurement, ffmpeg_version, measure
from dnd_audio.mix.render import DEFAULT_MIX_WINDOW, render_mix
from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw, snapshot, verify_unchanged
from dnd_audio.timeline import TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.reader import DEFAULT_WINDOW_SAMPLES

__all__ = [
    "MixResult",
    "MixWork",
    "encode_deliverable",
    "mix_outputs",
    "perform_mix",
    "record_mix_stage",
    "remove_mix_artifacts",
    "run_mix",
]

#: The stages `mix` does not run, and why (INV-13).
_SKIPPED_STAGES: Final = (
    (StageName.TRANSCRIBE, "`mix` is the audio branch; the transcript is `transcribe` (INV-09)"),
    (StageName.RENDER, "there is no transcript to render"),
)


@dataclass(frozen=True, slots=True)
class MixWork:
    """A rendered intermediate with its cache **staged and uncommitted**.

    The same shape and the same contract as `ActivityWork`: the caller owns the INV-01
    verification and the commit, because an entry may only be published once the sources it was
    computed from have been re-checked — and the mix is the one stage after inspection that
    reads those sources at all.
    """

    intermediate: Path
    n_samples: int
    peak: float
    corrections: LevelCorrections
    cache: MixCache
    cache_key: str
    from_cache: bool
    warnings: tuple[MixNote, ...] = ()


@dataclass(frozen=True, slots=True)
class MixResult:
    """What one `mix` run produced."""

    mp3_path: Path
    graph: ActivityGraph | None
    encode: EncodeResult | None
    report: IngestReport
    report_path: Path
    exit_code: ExitCode
    #: False only when writing the report would itself have violated INV-01.
    report_written: bool = True


def mix_outputs(session_dir: Path) -> dict[str, Path]:
    """Everything `mix` writes, for the INV-01 output check.

    A superset of `activity`'s, because a composed run performs both. Declared as data so that
    adding an output and forgetting to protect it is a visible omission from one list.
    """
    return {
        **activity_outputs(session_dir),
        "the mixed MP3": session_dir / MP3_RELATIVE_PATH,
        "the mix cache": session_dir / MIX_CACHE_DIRNAME,
    }


def run_mix(
    session_dir: Path,
    *,
    detector: DetectorBundle | None = None,
    now: dt.datetime | None = None,
    use_cache: bool = True,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    mix_window_samples: int = DEFAULT_MIX_WINDOW,
    detect_window_samples: int | None = None,
) -> MixResult:
    """Reconstruct, attribute, mix, and encode.

    Never raises for an expected failure: a fatal condition becomes a failed stage, a
    structured error, a written report, and a nonzero exit code (INV-13).
    """
    started_at = now or dt.datetime.now(dt.UTC)
    mp3_path = session_dir / MP3_RELATIVE_PATH
    report_path = session_dir / OUTPUT_DIRNAME / REPORT_FILENAME
    builder = _builder(session_dir.name, None, started_at)
    graph: ActivityGraph | None = None
    encoded: EncodeResult | None = None

    try:
        config = load_session_config(session_dir / "session.yaml")
        builder = _builder(config.session_id, config_hash(config), started_at)
        roots = raw_roots(config)
        before = snapshot(session_dir, roots)
        reject_outputs_inside_raw(session_dir, config, roots, mix_outputs(session_dir))

        work = perform_activity(
            session_dir,
            config,
            builder=builder,
            detector=detector,
            use_cache=use_cache,
            mix=True,
            window_samples=window_samples,
            detect_window_samples=detect_window_samples or DEFAULT_DETECT_WINDOW,
        )

        # First commit point: the activity caches are verified and published here, so a
        # failure in the mix or the encode — neither of which can invalidate them — does not
        # throw away six tracks of inference (ADR-0021).
        verify_unchanged(session_dir, roots, before)
        work.commit()
        write_json_atomic(session_dir / ACTIVITY_RELATIVE_PATH, work.graph.model_dump(mode="json"))
        graph = work.graph
        builder.stage_complete(StageName.RECONSTRUCT, warnings=_notes(work.timeline.warnings))
        builder.add_deliverable(session_dir / TIMELINE_RELATIVE_PATH, relative_to=session_dir)
        builder.stage_complete(StageName.ACTIVITY, warnings=_notes(work.graph.warnings))
        builder.add_deliverable(session_dir / ACTIVITY_RELATIVE_PATH, relative_to=session_dir)

        mixed = perform_mix(
            session_dir,
            config,
            work,
            use_cache=use_cache,
            mix_window_samples=mix_window_samples,
        )

        # Second commit point, and the reason it exists: rendering is the only work after
        # inspection that reads source audio, so its cache entry must not be published until
        # those sources have been re-checked.
        verify_unchanged(session_dir, roots, before)
        mixed.cache.commit()

        encoded = encode_deliverable(session_dir, config, mixed, builder=builder)
        record_mix_stage(builder, session_dir, mixed, encoded)
    except Exception as exc:
        return _failed(exc, session_dir, mp3_path, report_path, builder, graph, now)

    report = builder.write(report_path, dt.datetime.now(dt.UTC) if now is None else now)
    return MixResult(
        mp3_path=mp3_path,
        graph=graph,
        encode=encoded,
        report=report,
        report_path=report_path,
        exit_code=report.exit_code(),
    )


def perform_mix(
    session_dir: Path,
    config: SessionConfig,
    work: ActivityWork,
    *,
    use_cache: bool = True,
    mix_window_samples: int = DEFAULT_MIX_WINDOW,
) -> MixWork:
    """Render the lossless intermediate, or take it from cache. Leaves the cache staged.

    The composable half of `mix`, so `process` performs it exactly the way this command does
    rather than reimplementing the composition beside it (ADR-0015's argument, two milestones
    later).

    What it deliberately does **not** do: snapshot `raw/`, verify it, commit anything, or write
    a report. Those belong to whoever owns the whole run.
    """
    graph = work.graph
    timeline = work.timeline
    track_ids = _mixable(graph, timeline)
    corrections = level_corrections(graph, settings=config.mix.envelope)
    cache = MixCache(session_dir=session_dir, read_enabled=use_cache)
    key = mix_identity(
        graph,
        stage_config_hash=stage_config_hash(config, "mix"),
        corrections=corrections,
        track_ids=track_ids,
    )

    warnings: list[MixNote] = list(corrections.warnings)
    warnings.extend(_absent(graph, track_ids))

    found = cache.get(key, expected_samples=timeline.duration_samples)
    if found is not None:
        return MixWork(
            intermediate=session_dir / found.relative_path,
            n_samples=found.n_samples,
            peak=_peak_of(session_dir / found.relative_path),
            corrections=corrections,
            cache=cache,
            cache_key=key,
            from_cache=True,
            warnings=tuple(warnings),
        )

    summary = render_mix(
        cache.audio_path(key),
        session_dir=session_dir,
        timeline=timeline,
        track_ids=track_ids,
        envelope=EnvelopeStream(
            graph,
            settings=config.mix.envelope,
            corrections=corrections,
            track_ids=track_ids,
        ),
        window_samples=mix_window_samples,
    )
    cache.publish(key, sample_rate=summary.sample_rate, n_samples=summary.n_samples)
    return MixWork(
        intermediate=summary.path,
        n_samples=summary.n_samples,
        peak=summary.peak,
        corrections=corrections,
        cache=cache,
        cache_key=key,
        from_cache=False,
        warnings=tuple(warnings),
    )


def encode_deliverable(
    session_dir: Path, config: SessionConfig, work: MixWork, *, builder: ReportBuilder
) -> EncodeResult:
    """Measure the intermediate, encode the MP3, verify the decode, and record all of it.

    Reads nothing but the intermediate, so it cannot invalidate a source hash — which is what
    lets it run after the mix's commit point rather than before it.
    """
    mp3_path = session_dir / MP3_RELATIVE_PATH
    source = measure(work.intermediate)

    builder.record_cache(hits=work.cache.hits, misses=work.cache.misses)
    builder.record_package_version("dnd_audio.mix", str(MIX_SEMANTICS_VERSION))
    builder.record_tool_version("ffmpeg", ffmpeg_version())
    builder.record_command(source.command)

    try:
        encoded = encode_mp3(
            work.intermediate,
            mp3_path,
            settings=config.mix,
            session_id=config.session_id,
            title=config.title,
            source_measurement=source,
            expected_samples=work.n_samples,
        )
    except EncodeError as exc:
        # "Retain all measurements in the report", and the interesting run is the one that
        # failed. Recording only on the success path left an exhausted retry budget with a
        # single error string and nothing structured to audit — found by M5's code review.
        for command in exc.commands:
            builder.record_command(command)
        _record_intermediate(builder, work, source)
        for attempt in exc.attempts:
            _record_attempt(builder, attempt)
        raise

    for command in encoded.commands:
        builder.record_command(command)
    _record_decisions(builder, work, source, encoded)

    return encoded


def record_mix_stage(
    builder: ReportBuilder, session_dir: Path, work: MixWork, encoded: EncodeResult
) -> None:
    """Mark the mix complete and hash its deliverable.

    Separate from :func:`encode_deliverable` so a caller can perform its **final** INV-01
    verification between the work and the record. `process` does exactly that: with several
    commit points, a branch that fails before its own commit leaves the window after the mix's
    read unchecked, and a stage recorded as complete before that check would have to be
    un-recorded afterwards. Deferring the record is cheaper than making one retractable.
    """
    builder.stage_complete(StageName.MIX, warnings=_notes([*work.warnings, *encoded.warnings]))
    builder.add_deliverable(session_dir / MP3_RELATIVE_PATH, relative_to=session_dir)


def remove_mix_artifacts(session_dir: Path, *, completed: Container[StageName] = ()) -> None:
    """Delete the MP3 a failed run may have left behind.

    Not the intermediate: it lives under `work/cache/` and is inert without its sidecar, which
    a failed run never commits. The MP3 is the deliverable, and a stale one sitting beside a
    report that calls the mix stage failed is worse than none — the file looks current and
    nothing in it says otherwise.

    **Never called before the `output_inside_raw` carve-out.** When an output path resolves
    inside a source directory these unlinks *are* the INV-01 violation (ADR-0021).
    """
    if StageName.MIX not in completed:
        (session_dir / MP3_RELATIVE_PATH).unlink(missing_ok=True)


def _mixable(graph: ActivityGraph, timeline: Timeline) -> tuple[str, ...]:
    """The tracks the mix divides gain between: those with a segment map and a graph entry.

    A track with no working audio contributes nothing but would still take a share, so the
    other five would each be quieter for the sake of a channel that is silence by construction.
    Both documents have to agree, in the timeline's order, because the renderer steps its
    readers in that order.
    """
    known = {track.track_id for track in graph.tracks}
    return tuple(
        track.track_id for track in timeline.tracks if track.segments and track.track_id in known
    )


def _absent(graph: ActivityGraph, track_ids: tuple[str, ...]) -> list[MixNote]:
    """Warn about every configured track the mix could not include."""
    return [
        MixNote(
            code="mix_track_absent",
            message=(
                f"{track.track_id} has no working audio in this session, so it is not in the "
                f"mix and takes no share of the gain. The other tracks are each louder for it."
            ),
            path=track.track_id,
        )
        for track in graph.tracks
        if track.track_id not in track_ids
    ]


def _peak_of(path: Path) -> float:
    """The largest absolute sample in a cached intermediate.

    Read back in bounded windows rather than remembered, because a cache hit did not run the
    renderer and the report records this number for every run alike. It is *not* what the
    first encode's gain is aimed with — that is the true peak `measure` reads back off the
    decode, which accounts for inter-sample overs and this does not.
    """
    import numpy as np

    from dnd_audio.timeline.pcm import PcmReader, open_pcm

    peak = 0.0
    with PcmReader(open_pcm(path)) as reader:
        position = 0
        while position < reader.source.n_samples:
            length = min(DEFAULT_MIX_WINDOW, reader.source.n_samples - position)
            peak = max(peak, float(np.abs(reader.read(position, length)).max(initial=0.0)))
            position += length
    return peak


def _record_decisions(
    builder: ReportBuilder, work: MixWork, source: Measurement, encoded: EncodeResult
) -> None:
    """Put every correction and every measurement in the report.

    The spec asks for all measurements to be retained, and M5 publishes no deterministic
    document of its own (ADR-0022) — so this subsection *is* the audit trail, and INV-02
    already requires it to be semantically stable across an unchanged rerun.
    """
    for item in work.corrections.corrections:
        builder.record_decision(
            Decision(
                code="mix_level_correction",
                subject=item.track_id,
                detail=(
                    f"{item.track_id} is corrected by "
                    f"{item.correction_mb / MILLIBELS_PER_DB:+.2f} dB toward the session's "
                    f"speech reference" + (", clamped" if item.clamped else "") + "."
                ),
                details={
                    "correction_mb": str(item.correction_mb),
                    "reference_mbfs": (
                        "unknown" if item.reference_mbfs is None else str(item.reference_mbfs)
                    ),
                    "clamped": str(item.clamped).lower(),
                },
            )
        )

    _record_intermediate(builder, work, source)
    for attempt in encoded.attempts:
        _record_attempt(builder, attempt)

    checked = (
        "every configured tolerance"
        if encoded.normalized
        else "the true-peak ceiling and the decoded duration; the loudness target was not "
        "aimed at and so was not checked (ADR-0023)"
    )
    builder.record_decision(
        Decision(
            code="mix_encoded",
            subject=MP3_RELATIVE_PATH,
            detail=(
                f"{encoded.facts.codec} {encoded.facts.channels}ch "
                f"{encoded.facts.bit_rate_kbps} kbps, decoded and measured within {checked} "
                f"after {len(encoded.attempts)} attempt(s)."
            ),
            details={
                "channels": str(encoded.facts.channels),
                "sample_rate": str(encoded.facts.sample_rate),
                "bit_rate_kbps": str(encoded.facts.bit_rate_kbps),
                "attempts": str(len(encoded.attempts)),
                "loudness_normalized": str(encoded.normalized).lower(),
            },
        )
    )


def _record_intermediate(builder: ReportBuilder, work: MixWork, source: Measurement) -> None:
    """The intermediate this encode was made from, and what it measured.

    Deliberately says nothing about whether it was rendered or reused: INV-02 requires this
    subsection to be *semantically stable* across an unchanged rerun, and cache state is
    per-run telemetry that `record_cache` already carries. Recording it here made the second
    run's report disagree with the first about a session neither run changed — found by M5's
    code review.
    """
    builder.record_decision(
        Decision(
            code="mix_intermediate",
            subject=work.cache_key,
            detail=(
                f"the lossless intermediate is {work.n_samples} samples at unity master gain "
                f"(ADR-0023)."
            ),
            details={
                "integrated_lufs_mb": _text(source.integrated_lufs_mb),
                "true_peak_dbtp_mb": _text(source.true_peak_dbtp_mb),
                "sample_peak": f"{work.peak:.6f}",
            },
        )
    )


def _record_attempt(builder: ReportBuilder, attempt: EncodeAttempt) -> None:
    """One encode attempt, compliant or not. Recorded either way (the spec asks for both)."""
    builder.record_decision(
        Decision(
            code="mix_encode_attempt",
            subject=f"attempt_{attempt.index}",
            detail=(
                f"encoded at {attempt.gain_mb / MILLIBELS_PER_DB:+.2f} dB master gain; "
                + (
                    "accepted."
                    if attempt.compliant
                    else f"rejected: {', '.join(attempt.failures)}."
                )
            ),
            details={
                "gain_mb": str(attempt.gain_mb),
                "integrated_lufs_mb": _text(attempt.measurement.integrated_lufs_mb),
                "true_peak_dbtp_mb": _text(attempt.measurement.true_peak_dbtp_mb),
                "decoded_samples": str(attempt.measurement.n_samples),
                "failures": ",".join(attempt.failures),
            },
        )
    )


def _text(millibels: int | None) -> str:
    return "unknown" if millibels is None else str(millibels)


def _failed(
    exc: Exception,
    session_dir: Path,
    mp3_path: Path,
    report_path: Path,
    builder: ReportBuilder,
    graph: ActivityGraph | None,
    now: dt.datetime | None,
) -> MixResult:
    """Every failure, not only the ones raised on purpose (INV-13)."""
    error = StructuredError(code=_code_of(exc), message=str(exc) or type(exc).__name__)
    completed = [
        stage
        for stage in (StageName.RECONSTRUCT, StageName.ACTIVITY, StageName.MIX)
        if builder.completed(stage)
    ]
    for stage in (
        StageName.INSPECT,
        StageName.RECONSTRUCT,
        StageName.ACTIVITY,
        StageName.MIX,
    ):
        if not builder.recorded(stage):
            builder.stage_failed(stage, [error])

    finished = dt.datetime.now(dt.UTC) if now is None else now
    if isinstance(exc, DiscoveryError) and exc.code == "output_inside_raw":
        # INV-01 outranks INV-13 here, and this returns **before** the cleanup below: with
        # `work -> raw/tx-a` every artifact path resolves inside a source directory, so
        # unlinking the stale ones is itself a write into `raw/` (ADR-0021).
        return MixResult(
            mp3_path=mp3_path,
            graph=graph,
            encode=None,
            report=builder.build(finished),
            report_path=report_path,
            report_written=False,
            exit_code=ExitCode.FATAL,
        )

    remove_activity_artifacts(session_dir, completed=completed)
    remove_mix_artifacts(session_dir, completed=completed)
    report = builder.write(report_path, finished)
    return MixResult(
        mp3_path=mp3_path,
        graph=graph,
        encode=None,
        report=report,
        report_path=report_path,
        exit_code=report.exit_code(),
    )


def _builder(session_id: str, hash_: str | None, started_at: dt.datetime) -> ReportBuilder:
    builder = ReportBuilder(session_id=session_id, config_hash=hash_, started_at=started_at)
    for stage, reason in _SKIPPED_STAGES:
        builder.stage_skipped(stage, reason)
    return builder


class _Note(Protocol):
    """What the report needs from a warning, whichever artifact it came from."""

    @property
    def code(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def path(self) -> str | None: ...


def _notes(notes: Sequence[_Note]) -> list[ReportWarning]:
    """Flatten artifact warnings for the report, in a stable order."""
    flattened = [
        ReportWarning(code=note.code, message=note.message, path=note.path) for note in notes
    ]
    return sorted(flattened, key=lambda note: (note.code, note.path or "", note.message))


def _code_of(exc: BaseException) -> str:
    if isinstance(exc, DndAudioError):
        return exc.code
    return "internal_error"
