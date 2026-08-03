"""The `dnd-audio` command line.

Every command the spec names is registered here from M0 onward, so the surface is
stable and a caller can discover it before the stages behind it exist. Everything
except `doctor` raises ``NotImplementedError`` with a ``DEFERRED: M<n>`` annotation at
the raise site — deliberately visible to ``scripts/scan_placeholders.py``. Hiding
placeholder work behind a bespoke exception type would defeat the check that exists to
find it.

Stage boundaries, from the spec:

* ``inspect``   — discover and validate sources, write the manifest.
* ``ingest``    — timeline maps, lossless working path, 16 kHz derivatives.
* ``activity``  — speech detection, bleed rejection, and the graph both branches read.
  A stage in the spec's own DAG rather than one of its named commands, exposed here as one
  of the "independently resumable stages for development and recovery" it asks for, and
  called directly by `transcribe`, `mix`, and `process` (ADR-0015).
* ``transcribe``— activity, ASR, alignment, normalized transcript records.
* ``mix``       — automix and MP3. Never requires ASR or `transcribe` output (INV-09).
* ``render``    — regenerate transcript files from cached records. No ASR, no mixer.
* ``process``   — dependency-aware orchestration; the two branches fail independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dnd_audio import __version__
from dnd_audio.activity.runner import ActivityResult, run_activity
from dnd_audio.determinism import canonical_json
from dnd_audio.doctor import CheckStatus, overall_status, run_checks
from dnd_audio.errors import DndAudioError, ExitCode
from dnd_audio.inspection.runner import InspectionResult, run_inspect
from dnd_audio.mix.runner import MixResult, run_mix
from dnd_audio.models import SILERO_VAD, fetch, find_model, lock_path
from dnd_audio.orchestrate import ProcessResult, run_process
from dnd_audio.timeline.runner import IngestResult, run_ingest
from dnd_audio.transcript.runner import (
    RenderResult,
    TranscribeResult,
    run_render,
    run_transcribe,
)

__all__ = ["app", "main"]

SessionDir = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Session directory: the one holding session.yaml and raw/.",
    ),
]

app = typer.Typer(
    name="dnd-audio",
    help=(
        "Local audio ingestion and transcription for long tabletop-RPG sessions. "
        "Nothing here sends audio anywhere."
    ),
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(
    help="Model management. `fetch` is the only command permitted to touch the network.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")


@app.command()
def process(
    session_dir: SessionDir,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Re-run everything, ignoring cached work. Every cache is still written, so "
            "this costs one slow run rather than every run.",
        ),
    ] = False,
    fake_models: Annotated[
        bool,
        typer.Option(
            "--fake-models",
            help="Drive speech detection and transcription from this session's declared "
            "fake-models.json instead of from models. For synthetic fixtures: the real ASR "
            "adapter lands in M6b.",
        ),
    ] = False,
) -> None:
    """Run every applicable stage, then always finalize the report.

    Activity runs once; the mix branch and the transcript branch are then attempted
    independently, so a transcription failure still yields `session.mp3` and a report. A run
    where either branch failed exits nonzero, so automation cannot mistake partial output for
    full success (INV-13).
    """
    result = run_process(session_dir, use_cache=not no_cache, fake_models=fake_models)
    _summarize_process(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@app.command()
def inspect(
    session_dir: SessionDir,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Re-inspect every source, ignoring cached captures. The cache is still "
            "written, so this costs one slow run rather than every run.",
        ),
    ] = False,
) -> None:
    """Discover and validate sources; write the deterministic manifest.

    Always writes `output/ingest-report.json`, including when inspection fails —
    that is what INV-13 means, and it is why this exits through a status rather than
    by raising.
    """
    result = run_inspect(session_dir, use_cache=not no_cache)
    _summarize(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@app.command()
def ingest(
    session_dir: SessionDir,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Re-inspect and re-derive everything, ignoring cached work. Both caches "
            "are still written, so this costs one slow run rather than every run.",
        ),
    ] = False,
    materialize_48k: Annotated[
        bool,
        typer.Option(
            "--materialize-48k",
            help="Also write contiguous 48 kHz float32 RF64 files per track. The segment "
            "map is the working path; these are disposable cache artifacts for debugging, "
            "interoperability, and performance investigation.",
        ),
    ] = False,
) -> None:
    """Build the timeline, the working path, and the 16 kHz derivatives.

    Runs inspection first, every time — a manifest whose configuration hash still matches
    is not evidence that the files it describes are the ones on disk. A warm run costs no
    FFprobe.

    Always writes `output/ingest-report.json`, including when it fails (INV-13).
    """
    result = run_ingest(session_dir, use_cache=not no_cache, materialize_48k=materialize_48k)
    _summarize_ingest(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@app.command()
def activity(
    session_dir: SessionDir,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Re-detect and re-attribute everything, ignoring cached work. Both caches "
            "are still written, so this costs one slow run rather than every run.",
        ),
    ] = False,
) -> None:
    """Detect speech per track, reject bleed, and write the activity graph.

    The spec's stage DAG calls `activity` the shared cached operation that `transcribe`,
    `mix`, and `process` all invoke; this exposes it as one of the independently resumable
    stages the spec asks for, so it can be run and inspected on its own (ADR-0015). It
    reconstructs the timeline first, every time, for the reason `ingest` re-inspects every
    time: an artifact on disk is not evidence that it still describes what is beside it.

    Needs the pinned VAD model — `dnd-audio models fetch` — and says so if it is absent.

    Always writes `output/ingest-report.json`, including when it fails (INV-13).
    """
    result = run_activity(session_dir, use_cache=not no_cache)
    _summarize_activity(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@app.command()
def transcribe(
    session_dir: SessionDir,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Re-run everything, ignoring cached work. Every cache is still written, so "
            "this costs one slow run rather than every run.",
        ),
    ] = False,
    fake_models: Annotated[
        bool,
        typer.Option(
            "--fake-models",
            help="Drive speech detection and transcription from this session's declared "
            "fake-models.json instead of from models. For synthetic fixtures: the real ASR "
            "adapter lands in M6b. The transcript records and the report both say so.",
        ),
    ] = False,
) -> None:
    """Run activity attribution and ASR; write the records and both transcript files.

    The whole transcript branch: inspection, the timeline, activity, ASR, and the render.
    It reconstructs and re-attributes every time, for the reason `ingest` re-inspects every
    time — an artifact on disk is not evidence that it still describes what is beside it.

    Always writes `output/ingest-report.json`, including when it fails (INV-13).
    """
    result = run_transcribe(session_dir, use_cache=not no_cache, fake_models=fake_models)
    _summarize_transcribe(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@app.command()
def mix(
    session_dir: SessionDir,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Re-run everything, ignoring cached work. Every cache is still written, so "
            "this costs one slow run rather than every run.",
        ),
    ] = False,
) -> None:
    """Automix the synchronized tracks and encode session.mp3.

    The whole audio branch: inspection, the timeline, activity, the gain envelopes, the
    streamed mix, and the encode. It never reads a transcript and never runs a model beyond
    the VAD (INV-09), so it is the branch that survives a transcription failure.

    Always writes `output/ingest-report.json`, including when it fails (INV-13).
    """
    result = run_mix(session_dir, use_cache=not no_cache)
    _summarize_mix(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@app.command()
def render(session_dir: SessionDir) -> None:
    """Regenerate transcript.json and transcript.md from cached records.

    Reads `work/transcript-records.json` and nothing else: no model, no activity graph, no
    timeline, no mixer. Absent records are a clear failure rather than an empty transcript.

    Always writes `output/ingest-report.json`, including when it fails (INV-13).
    """
    result = run_render(session_dir)
    _summarize_render(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


@models_app.command("fetch")
def models_fetch() -> None:
    """Download the pinned voice-activity model and record what it resolved to.

    The only command permitted to touch the network (INV-06), and it fetches exactly
    one artifact: Silero VAD, pinned by commit and sha256, verified before it is written
    (ADR-0013). The ASR and alignment models land in M6b, and the lock format is
    provisional until they do.

    Already present and verifying means no download. A file that does not match the pin
    is not a model, so this exits nonzero rather than leaving one behind.
    """
    descriptor = SILERO_VAD
    already_present = find_model(descriptor) is not None
    try:
        path = fetch(descriptor)
    except DndAudioError as exc:
        typer.secho(f"  error  {exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.FATAL) from exc

    typer.echo(f"  {'already present' if already_present else 'fetched'}  {descriptor.key}")
    typer.echo(f"  model     {path}")
    typer.echo(f"  release   {descriptor.release}")
    typer.echo(f"  commit    {descriptor.commit}")
    typer.echo(f"  lock      {lock_path()}")
    typer.echo("  ASR and alignment models land in M6b; this fetches the VAD model only.")


@app.command()
def doctor(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory to check for writability and free space. Defaults to the "
            "working directory.",
        ),
    ] = Path(),
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the results as canonical JSON instead of a table."),
    ] = False,
) -> None:
    """Check this host without touching session audio.

    Reports the pinned VAD model as a warning when it is absent — that is a machine
    that can do everything but activity detection. GPU checks arrive with M6a, the ASR
    models with M6b.
    """
    results = run_checks(path)
    status = overall_status(results)

    if as_json:
        typer.echo(
            canonical_json(
                {
                    "version": __version__,
                    "status": status.value,
                    "checks": [
                        {"name": r.name, "status": r.status.value, "detail": r.detail}
                        for r in results
                    ],
                }
            ),
            nl=False,
        )
    else:
        width = max(len(result.name) for result in results)
        for result in results:
            colour = {
                CheckStatus.OK: typer.colors.GREEN,
                CheckStatus.WARN: typer.colors.YELLOW,
                CheckStatus.FAIL: typer.colors.RED,
            }[result.status]
            marker = typer.style(f"{result.status.value:>4}", fg=colour)
            typer.echo(f"  {marker}  {result.name:<{width}}  {result.detail}")

    if status is CheckStatus.FAIL:
        raise typer.Exit(code=ExitCode.FATAL)


def _summarize(result: InspectionResult) -> None:
    """Human-readable progress, as the spec's observability section asks for.

    Deliberately short. The report holds everything; this is the part someone reads
    while the run is still on screen.
    """
    manifest = result.manifest
    roster = manifest.roster
    if result.exit_code is ExitCode.OK:
        selected = sum(
            1 for track in manifest.tracks for source in track.sources if source.role == "selected"
        )
        typer.echo(
            f"  inspected {selected} source(s) across "
            f"{len(roster.active_tracks)}/{len(roster.known_tracks)} active track(s)"
        )
        for note in manifest.warnings:
            typer.secho(f"  warn  {note.code}: {note.message}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"  manifest  {result.manifest_path}")
    else:
        for stage in result.report.stages:
            for error in stage.errors:
                typer.secho(
                    f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True
                )
    if result.report_written:
        typer.echo(f"  report    {result.report_path}")
    else:
        typer.secho(
            f"  no report written: {result.report_path} would land inside the session's "
            f"own sources, and nothing under them may be written to (INV-01)",
            fg=typer.colors.RED,
            err=True,
        )


def _summarize_ingest(result: IngestResult) -> None:
    """Human-readable progress for `ingest`. The report holds everything."""
    timeline = result.timeline
    if timeline is not None:
        active = [track for track in timeline.tracks if track.segments]
        seconds = timeline.duration_samples / timeline.sample_rate
        typer.echo(
            f"  reconstructed {len(active)}/{len(timeline.tracks)} track(s), "
            f"{seconds:.3f}s aligned ({timeline.duration_samples} samples)"
        )
        for note in timeline.warnings:
            typer.secho(f"  warn  {note.code}: {note.message}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"  timeline  {result.timeline_path}")
    else:
        for stage in result.report.stages:
            for error in stage.errors:
                typer.secho(
                    f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True
                )
    if result.report_written:
        typer.echo(f"  report    {result.report_path}")
    else:
        typer.secho(
            f"  no report written: {result.report_path} would land inside the session's "
            f"own sources, and nothing under them may be written to (INV-01)",
            fg=typer.colors.RED,
            err=True,
        )


def _summarize_activity(result: ActivityResult) -> None:
    """Human-readable progress for `activity`. The graph and the report hold everything."""
    graph = result.graph
    if graph is not None:
        retained = graph.retained()
        suppressed = [c for c in graph.candidates if c.decision == "suppressed"]
        ambiguous = [c for c in retained if c.ambiguous]
        typer.echo(
            f"  {len(retained)} candidate(s) retained across {len(graph.tracks)} track(s), "
            f"{len(suppressed)} suppressed as bleed, {len(ambiguous)} kept despite the evidence"
        )
        for note in graph.warnings:
            typer.secho(f"  warn  {note.code}: {note.message}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"  activity  {result.graph_path}")
    else:
        for stage in result.report.stages:
            for error in stage.errors:
                typer.secho(
                    f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True
                )
    if result.report_written:
        typer.echo(f"  report    {result.report_path}")
    else:
        typer.secho(
            f"  no report written: {result.report_path} would land inside the session's "
            f"own sources, and nothing under them may be written to (INV-01)",
            fg=typer.colors.RED,
            err=True,
        )


def _summarize_transcribe(result: TranscribeResult) -> None:
    """Human-readable progress for `transcribe`. The records and the report hold everything."""
    records = result.records
    if records is not None:
        retained = records.retained()
        collapsed = [s for s in records.segments if s.decision == "duplicate"]
        overlapping = [s for s in retained if s.overlap]
        speakers = {segment.speaker_id for segment in retained}
        typer.echo(
            f"  {len(retained)} segment(s) across {len(speakers)} speaker(s), "
            f"{len(collapsed)} collapsed as duplicates, {len(overlapping)} marked as overlap"
        )
        for note in records.warnings:
            typer.secho(f"  warn  {note.code}: {note.message}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"  records    {result.records_path}")
        typer.echo(f"  transcript {result.transcript_path}")
        typer.echo(f"  markdown   {result.markdown_path}")
    else:
        for stage in result.report.stages:
            for error in stage.errors:
                typer.secho(
                    f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True
                )
    _report_line(result.report_written, result.report_path)


def _summarize_render(result: RenderResult) -> None:
    """Human-readable progress for `render`."""
    records = result.records
    if records is not None:
        typer.echo(f"  rendered {len(records.retained())} segment(s) from cached records")
        typer.echo(f"  transcript {result.transcript_path}")
        typer.echo(f"  markdown   {result.markdown_path}")
    else:
        for stage in result.report.stages:
            for error in stage.errors:
                typer.secho(
                    f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True
                )
    _report_line(result.report_written, result.report_path)


def _summarize_mix(result: MixResult) -> None:
    """Human-readable progress for `mix`. The MP3 and the report hold everything."""
    encoded = result.encode
    if encoded is not None:
        decoded = encoded.accepted.measurement
        loudness = (
            "unmeasurable"
            if decoded.integrated_lufs_mb is None
            else f"{decoded.integrated_lufs_mb / 100:.1f} LUFS"
        )
        peak = (
            "unmeasurable"
            if decoded.true_peak_dbtp_mb is None
            else f"{decoded.true_peak_dbtp_mb / 100:.1f} dBTP"
        )
        typer.echo(
            f"  mixed {decoded.n_samples / encoded.facts.sample_rate:.3f}s to "
            f"{encoded.facts.channels}-channel {encoded.facts.bit_rate_kbps} kbps MP3: "
            f"{loudness}, {peak}, {len(encoded.attempts)} encode attempt(s)"
        )
        for note in encoded.warnings:
            typer.secho(f"  warn  {note.code}: {note.message}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"  mp3        {result.mp3_path}")
    else:
        for stage in result.report.stages:
            for error in stage.errors:
                typer.secho(
                    f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True
                )
    _report_line(result.report_written, result.report_path)


def _summarize_process(result: ProcessResult) -> None:
    """Human-readable progress for `process`: one line per branch, whichever way each went."""
    encoded = result.encode
    if encoded is not None:
        typer.echo(f"  mix        {result.mp3_path}")
    else:
        typer.secho("  mix        failed", fg=typer.colors.RED, err=True)

    records = result.records
    if records is not None:
        typer.echo(f"  transcript {len(records.retained())} segment(s)")
    else:
        typer.secho("  transcript failed", fg=typer.colors.RED, err=True)

    for stage in result.report.stages:
        for error in stage.errors:
            typer.secho(f"  error  {error.code}: {error.message}", fg=typer.colors.RED, err=True)
    _report_line(result.report_written, result.report_path)


def _report_line(written: bool, path: Path) -> None:
    """Where the report went, or why it deliberately did not (INV-01 outranks INV-13)."""
    if written:
        typer.echo(f"  report     {path}")
    else:
        typer.secho(
            f"  no report written: {path} would land inside the session's own sources, and "
            f"nothing under them may be written to (INV-01)",
            fg=typer.colors.RED,
            err=True,
        )


def main() -> None:
    """Console-script entry point.

    Turns the two failure shapes into distinct exit codes so a caller never has to
    parse text to tell "not built yet" from "your session is broken" (INV-13).
    """
    try:
        app()
    except NotImplementedError as exc:
        typer.secho(f"not implemented yet: {exc}", fg=typer.colors.YELLOW, err=True)
        raise SystemExit(ExitCode.NOT_IMPLEMENTED) from exc
    except DndAudioError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(ExitCode.FATAL) from exc
