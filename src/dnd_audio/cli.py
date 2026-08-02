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
from dnd_audio.determinism import canonical_json
from dnd_audio.doctor import CheckStatus, overall_status, run_checks
from dnd_audio.errors import DndAudioError, ExitCode
from dnd_audio.inspection.runner import InspectionResult, run_inspect
from dnd_audio.timeline.runner import IngestResult, run_ingest

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
def process(session_dir: SessionDir) -> None:
    """Run every applicable stage, then always finalize the report."""
    # DEFERRED: M5
    raise NotImplementedError(f"`process` orchestrates both branches from M5 ({session_dir})")


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
def transcribe(session_dir: SessionDir) -> None:
    """Run activity attribution and ASR; write normalized transcript records."""
    # DEFERRED: M4
    raise NotImplementedError(f"`transcribe` lands in M4 ({session_dir})")


@app.command()
def mix(session_dir: SessionDir) -> None:
    """Automix the synchronized tracks and encode session.mp3."""
    # DEFERRED: M5
    raise NotImplementedError(f"`mix` lands in M5 ({session_dir})")


@app.command()
def render(session_dir: SessionDir) -> None:
    """Regenerate transcript.json and transcript.md from cached records."""
    # DEFERRED: M4
    raise NotImplementedError(f"`render` lands in M4 ({session_dir})")


@models_app.command("fetch")
def models_fetch() -> None:
    """Download models and record their resolved snapshot revisions."""
    # DEFERRED: M6b
    raise NotImplementedError("`models fetch` lands in M6b")


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

    GPU and model-availability checks arrive with M6a and M6b.
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
