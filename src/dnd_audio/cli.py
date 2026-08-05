"""The `dnd-audio` command line.

Every command the spec names is registered here from M0 onward, so the surface is
stable and a caller can discover it before the stages behind it exist. Everything
except `doctor` raised ``NotImplementedError`` with a ``DEFERRED: M<n>`` annotation at
the raise site — deliberately visible to ``scripts/scan_placeholders.py``, because hiding
placeholder work behind a bespoke exception type would defeat the check that exists to
find it. **None are left as of M6b.** A machine that cannot run a stage now fails like
any other environment problem: a failed stage, a written report, and a nonzero exit.

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

import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

if TYPE_CHECKING:
    # Imported for the annotation only. The runtime import sits inside `marker_analyze`,
    # the same shape the archive modules use below: a process that will only ever run
    # `inspect` should not construct the detector's imports to satisfy a type hint.
    from dnd_audio.marker.runner import MarkerAnalysisResult

from dnd_audio import __version__
from dnd_audio.activity.runner import ActivityResult, run_activity

# The enum only; `dnd_audio.archive.report` imports no provider code, so registering the
# subcommands does not pull an S3 SDK into a process that will only ever run `mix`. The
# archive modules that do are imported inside `_run_archive` (INV-06, ADR-0035).
from dnd_audio.archive.report import ArchiveOperation
from dnd_audio.determinism import canonical_json
from dnd_audio.doctor import CheckStatus, overall_status, run_checks
from dnd_audio.errors import DndAudioError, ExitCode
from dnd_audio.inspection.runner import InspectionResult, run_inspect
from dnd_audio.mix.runner import MixResult, run_mix
from dnd_audio.models import (
    QWEN3_ALIGNER,
    QWEN3_ASR,
    QWEN_SNAPSHOTS,
    REVISION_PATTERN,
    SILERO_VAD,
    SNAPSHOT_FETCH_COMMAND,
    fetch,
    find_model,
    install_snapshot,
    lock_path,
    model_path,
    snapshot_dir,
    snapshot_present,
)
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

DeviceChoice = Literal["auto", "cpu", "cuda"]
DtypeChoice = Literal["auto", "float32", "bfloat16"]

#: Validated by hand rather than through a `StrEnum` parameter type, because these are the
#: `asr.device`/`asr.dtype` vocabularies from `session.yaml` and they are spelled there as
#: plain strings. An enum here would be a second place for that vocabulary to live, and a
#: second place is how two spellings of `bfloat16` end up in one project (ADR-0005).
_DEVICE_CHOICES: tuple[DeviceChoice, ...] = ("auto", "cpu", "cuda")
_DTYPE_CHOICES: tuple[DtypeChoice, ...] = ("auto", "float32", "bfloat16")

app = typer.Typer(
    name="dnd-audio",
    help=(
        "Local audio ingestion and transcription for long tabletop-RPG sessions. "
        "Nothing here sends audio to anything that processes it; `archive` is the one "
        "command group that sends anything at all, to your own private storage."
    ),
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(
    help="Model management. `fetch` is one of the two commands permitted to touch the "
    "network; `archive` is the other.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")

archive_app = typer.Typer(
    help="Verified off-site backup of a session's raw sources. The only command group "
    "that sends session audio anywhere, and it never publishes or deletes (INV-06).",
    no_args_is_help=True,
)
app.add_typer(archive_app, name="archive")

marker_app = typer.Typer(
    help="The acoustic synchronization marker: build one to play, and find it afterwards. "
    "Verifies the LTC jam and measures differential acoustic arrival; it never places a "
    "file and never corrects a timeline (ADR-0040).",
    no_args_is_help=True,
)
app.add_typer(marker_app, name="marker")


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
def models_fetch(
    qwen: Annotated[
        bool,
        typer.Option(
            "--qwen",
            help="Also install the ASR and alignment snapshots — about 6 GB, and it needs "
            "the `hf` CLI, which lives in the ROCm environment. `./scripts/fetch-models.sh` "
            "runs this for you from there.",
        ),
    ] = False,
    asr_revision: Annotated[
        str | None,
        typer.Option(
            "--asr-revision",
            help="Install this commit of the ASR repository instead of the pinned one — "
            "the counterpart of `asr.model_revision` in session.yaml.",
        ),
    ] = None,
    aligner_revision: Annotated[
        str | None,
        typer.Option(
            "--aligner-revision",
            help="Install this commit of the aligner repository instead of the pinned one "
            "— the counterpart of `asr.aligner_revision` in session.yaml.",
        ),
    ] = None,
) -> None:
    """Install the pinned models and record what they resolved to.

    One of the two commands permitted to touch the network (INV-06); `archive` is the
    other. Without `--qwen` it fetches
    exactly one artifact: Silero VAD, pinned by commit and sha256, verified in memory
    before it is written (ADR-0013). With it, the Qwen ASR model and forced aligner are
    installed too, each pinned to a commit with a per-file digest manifest and downloaded
    by the `hf` CLI (ADR-0027).

    Not on by default because the two snapshots are about six gigabytes and most reasons
    to run this are not about them. The Qwen half is reported either way, so a run that
    does not install them still says where they stand.

    Already present and verifying means no download, so this is safe — and cheap — to
    re-run as an "am I set up?" check. Anything that does not match its pin is not a
    model, so this exits nonzero rather than leaving one behind.
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

    overrides = _snapshot_revisions(asr_revision, aligner_revision)
    for snapshot in QWEN_SNAPSHOTS:
        revision = overrides[snapshot.key]
        if not qwen:
            state = "present" if snapshot_present(snapshot, revision=revision) else "absent"
            typer.echo(f"  {state:<15} {snapshot.key} ({snapshot.repository})")
            continue
        try:
            target, downloaded = install_snapshot(snapshot, revision=revision)
        except DndAudioError as exc:
            typer.secho(f"  error  {exc.code}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=ExitCode.FATAL) from exc
        typer.echo(f"  {'installed' if downloaded else 'already present'}  {snapshot.key}")
        typer.echo(f"  model     {target}")
        typer.echo(f"  commit    {revision}")

    typer.echo(f"  lock      {lock_path()}")
    if not qwen:
        typer.echo(
            "  The ASR and alignment snapshots are about 6 GB and are not installed by "
            f"this command unless asked: run `{SNAPSHOT_FETCH_COMMAND}`."
        )


def _snapshot_revisions(
    asr_revision: str | None, aligner_revision: str | None
) -> dict[str, str | None]:
    """Map each snapshot key to the revision asked for, refusing anything not a commit.

    `models fetch` has to accept these because `session.yaml` accepts them. Without it an
    operator could *configure* `asr.model_revision` — the completion gate requires that —
    and then have no command able to install it: `process` would report "revision not
    installed" forever, and the only permitted network command would keep fetching the
    revision the build was pinned to. Found by M6b's code review.

    The shape check is `AsrConfig`'s, restated here because this entry point does not go
    through a config file. Both refuse for the same reason (ADR-0027): a directory is keyed
    by this string, so a branch name would install into a directory `process` never looks in.
    """
    chosen = {QWEN3_ASR.key: asr_revision, QWEN3_ALIGNER.key: aligner_revision}
    for key, revision in chosen.items():
        if revision is not None and not re.fullmatch(REVISION_PATTERN, revision):
            typer.secho(
                f"  error  {revision!r} is not a commit. Give the full 40-character "
                f"lowercase hexadecimal commit sha for {key} — a branch or tag moves, and "
                f"the snapshot directory is keyed by this string.",
                fg=typer.colors.RED,
                err=True,
            )
            # Click's usage code, and `ExitCode` deliberately does not define 2 for exactly
            # this reason: a malformed option is a typo, not a pipeline failure.
            raise typer.Exit(code=2)
    return chosen


@models_app.command("plan")
def models_plan(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the plan as canonical JSON instead of a table."),
    ] = False,
    asr_revision: Annotated[
        str | None,
        typer.Option("--asr-revision", help="Plan for this commit of the ASR repository."),
    ] = None,
    aligner_revision: Annotated[
        str | None,
        typer.Option("--aligner-revision", help="Plan for this commit of the aligner."),
    ] = None,
) -> None:
    """Print what `models fetch` would install, and where. Touches nothing.

    The single statement of the pin, in a form another program can read. This exists so
    `scripts/fetch-models.sh` does not have to repeat a repository, a commit, or a target
    directory — a wrapper that restated any of them would be a second place for the pin to
    live, and the one that drifts is always the one nobody is looking at.
    """
    overrides = _snapshot_revisions(asr_revision, aligner_revision)
    rows = [
        {
            "key": snapshot.key,
            "kind": "snapshot",
            "present": snapshot_present(snapshot, revision=overrides[snapshot.key]),
            "repository": snapshot.repository,
            "revision": overrides[snapshot.key] or snapshot.revision,
            "target": str(snapshot_dir(snapshot, revision=overrides[snapshot.key])),
        }
        for snapshot in QWEN_SNAPSHOTS
    ]
    rows.append(
        {
            "key": SILERO_VAD.key,
            "kind": "file",
            "present": find_model(SILERO_VAD) is not None,
            "repository": SILERO_VAD.repository,
            "revision": SILERO_VAD.commit,
            "target": str(model_path(SILERO_VAD)),
        }
    )

    if as_json:
        typer.echo(canonical_json({"lock": str(lock_path()), "models": rows}))
        return

    for row in rows:
        typer.echo(f"  {'present' if row['present'] else 'absent':<8} {row['key']}")
        typer.echo(f"    repository  {row['repository']}")
        typer.echo(f"    revision    {row['revision']}")
        typer.echo(f"    target      {row['target']}")
    typer.echo(f"  lock          {lock_path()}")


@archive_app.command("upload")
def archive_upload(session_dir: SessionDir) -> None:
    """Compress, upload, and verify every raw source; commit the manifest last.

    Requires a current `manifest.json`, so `dnd-audio inspect` runs first. Nothing under
    the session's sources is written, renamed, or deleted (INV-01), and no output or
    transcript is published — that is M7b and does not exist yet.

    Every object is downloaded and decompressed again before the manifest goes up, which
    roughly doubles the network cost and is the entire difference between a backup and a
    belief (ADR-0038).
    """
    _run_archive(ArchiveOperation.UPLOAD, session_dir=session_dir)


@archive_app.command("status")
def archive_status(session_dir: SessionDir) -> None:
    """Compare a local session against the archive. Cheap, and never authoritative.

    Reports `absent`, `pending`, `committed`, `previously_verified_at_commit` or
    `divergent`. It structurally cannot report `verified`: only a current full download
    establishes that, and saying it from provider metadata would be the one lie this
    design exists to prevent (ADR-0039).
    """
    _run_archive(ArchiveOperation.STATUS, session_dir=session_dir)


@archive_app.command("list")
def archive_list() -> None:
    """Every committed session id, without needing a local session directory.

    What makes the recovery drill possible when nobody remembers what the session was
    called. Follows pagination to exhaustion; a partial listing is an error, never a
    shorter answer.
    """
    _run_archive(ArchiveOperation.LIST)


@archive_app.command("verify")
def archive_verify(
    session_id: Annotated[str, typer.Option("--session-id", help="The archived session to check.")],
    track: Annotated[
        str | None,
        typer.Option("--track", help="Check only this track. Omit for the whole session."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Where to write the operation report."),
    ] = None,
) -> None:
    """Download every selected object and prove it still restores. The real check.

    Needs no local session directory — it is built for the case where there isn't one.
    Expensive by design: Cold Storage charges for retrieval, and anything cheaper would
    not be a verification.
    """
    _run_archive(ArchiveOperation.VERIFY, session_id=session_id, track=track, report_path=report)


@archive_app.command("restore")
def archive_restore(
    session_id: Annotated[
        str, typer.Option("--session-id", help="The archived session to restore.")
    ],
    to: Annotated[
        Path,
        typer.Option("--to", help="An existing empty directory to rebuild the session in."),
    ],
    track: Annotated[
        str | None,
        typer.Option("--track", help="Restore only this track. Omit for everything."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Where to write the operation report."),
    ] = None,
) -> None:
    """Rebuild a session's files from the archive alone, into an empty directory.

    Transactional: the whole tree is staged beside the destination and moved in at the
    end, so a failure leaves the destination untouched and the retry is just a retry.

    A track scope recovers only files attributed to that track. Nested notes and
    unassigned audio come back from a whole-session restore, because attributing them to
    a track would be inventing identity (INV-11).
    """
    _run_archive(
        ArchiveOperation.RESTORE,
        session_id=session_id,
        track=track,
        destination=to,
        report_path=report,
    )


def _sessions_above(destination: Path) -> list[Path]:
    """Every session directory the restore destination sits inside.

    Walks up looking for a `session.yaml`, because the CLI has no roster of sessions and a
    destination inside one is the case INV-01 forbids. Returns a list so the runner can
    resolve each one's configured source roots — a session directory is not itself
    protected, only the source roots within it are, and `restore --to SESSION/recovered`
    is a perfectly reasonable thing to want.

    Resolves first, so a symlinked component cannot hide the session it lands in.
    """
    found: list[Path] = []
    current = destination.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "session.yaml").is_file():
            found.append(candidate)
    return found


def _reject_path_inside_any_session(label: str, target: Path) -> None:
    """Refuse to write ``target`` under any session's configured sources (INV-01).

    Driven by the **path itself** rather than by whether the command was given a session
    directory, and that distinction is the whole point. The first version of this guarded
    only `archive upload` and `status`, so `archive verify --report
    SESSION/raw/tx-a/DJI_01.WAV` — a remote-only operation, which never has a session
    directory — replaced an irreplaceable recording with JSON. Found by M7a's second code
    review.

    `marker build` is the second command with the same shape: an arbitrary destination and
    no session argument. It reuses this rather than growing a parallel check, because the
    lesson of the first occurrence was that the *condition* was wrong, not the code.

    Every session above the resolved path is consulted, which covers both an explicit
    `--report`/destination and the `work -> raw/tx-a` symlink M1's verify phase found:
    resolving `SESSION/work/archive-upload-report.json` lands inside `SESSION/raw/tx-a`,
    whose session is still an ancestor.

    Args:
        label: What is being written, for the diagnostic. Reaches the operator's terminal.
        target: Where it would go. Need not exist; resolution is what decides.

    Raises:
        DndAudioError: with code ``output_inside_raw``, before the operation runs. Checked
            first rather than last so an expensive `verify` is not paid for and then
            refused — and, for `marker build`, so that nothing is created or unlinked on a
            path where the creation would itself be the violation.
    """
    from dnd_audio.config import load_session_config
    from dnd_audio.errors import DiscoveryError
    from dnd_audio.raw_guard import raw_roots, reject_outputs_inside_raw

    for session_dir in _sessions_above(target):
        try:
            config = load_session_config(session_dir / "session.yaml")
        except DndAudioError as exc:
            # **Refused, not skipped.** The first version of this guard wrote `continue`
            # under a comment claiming unknown roots are "not permissive", which is exactly
            # what continuing makes them: a session whose `session.yaml` does not parse has
            # source roots this process cannot enumerate, and the report path is *inside*
            # that session — so writing there might land in `raw/`. The only session that
            # reaches this branch is one containing the report, so refusing blocks nothing
            # unrelated. Found by M7a's third code review, in the fix for the second's P0.
            message = (
                f"the {label} would be written inside a session whose session.yaml cannot "
                f"be read, so its source directories are unknown and nothing may be "
                f"written under them (INV-01): {exc}"
            )
            raise DiscoveryError(message, code="output_inside_raw") from exc
        reject_outputs_inside_raw(session_dir, config, raw_roots(config), {label: target})


def _run_archive(
    operation: ArchiveOperation,
    *,
    session_dir: Path | None = None,
    session_id: str | None = None,
    track: str | None = None,
    destination: Path | None = None,
    report_path: Path | None = None,
) -> None:
    """Resolve configuration, build the client, run the operation, write the report.

    One place, because five commands share every step of it and the interesting failure —
    an unconfigured machine — must produce the same message from all of them.
    """
    from dnd_audio.archive.config import ArchiveConfigError, default_report_dir, load_archive_config
    from dnd_audio.archive.report import write_report
    from dnd_audio.archive.runner import run_list, run_restore, run_status, run_upload, run_verify
    from dnd_audio.archive.spaces import build_storage

    try:
        settings = load_archive_config()
    except ArchiveConfigError as exc:
        typer.secho(f"  error  {exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.FATAL) from exc

    # Resolved and checked **before** the operation runs. INV-01 outranks INV-13 when the
    # report's own location is the violation, exactly as `inspect` has done since M1: a
    # report is regenerable and a source directory written into is not. Doing it first also
    # means a full `verify` download is not paid for and then thrown away.
    if report_path is None:
        report_path = (
            session_dir / "work" / f"archive-{operation.value}-report.json"
            if session_dir is not None
            else default_report_dir() / f"archive-{operation.value}-report.json"
        )
    try:
        _reject_path_inside_any_session("archive report", report_path)
    except DndAudioError as exc:
        typer.secho(
            f"  no report written: {report_path} would land inside a session's own "
            f"sources, and nothing under them may be written to (INV-01)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=ExitCode.FATAL) from exc

    storage = build_storage(settings)
    listed: list[str] = []

    if operation is ArchiveOperation.UPLOAD:
        assert session_dir is not None
        result = run_upload(session_dir, storage=storage)
    elif operation is ArchiveOperation.STATUS:
        assert session_dir is not None
        result = run_status(session_dir, storage=storage)
    elif operation is ArchiveOperation.LIST:
        listed, result = run_list(storage=storage)
    elif operation is ArchiveOperation.VERIFY:
        assert session_id is not None
        result = run_verify(session_id, storage=storage, track_id=track)
    else:
        assert session_id is not None
        assert destination is not None
        # **Protected roots must be resolved here**, not left empty. The runner refuses a
        # destination inside a session's sources, and the CLI passing nothing made that
        # refusal unreachable from the actual command — so `archive restore --to
        # SESSION/raw/anywhere` would have written into protected sources (INV-01). The one
        # test that appeared to prove otherwise passed the list by hand. Found by M7a's
        # code review.
        result = run_restore(
            session_id,
            destination,
            storage=storage,
            track_id=track,
            protected_session_dirs=_sessions_above(destination),
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(result, report_path)

    for session in listed:
        typer.echo(f"  session   {session}")
    typer.echo(f"  {result.operation.value:<9} {result.status.value}")
    typer.echo(f"  archive   {result.verification.value}")
    if result.scope.entries_in_scope:
        typer.echo(f"  scope     {result.scope.entries_in_scope} entry/entries")
    for note in result.notes:
        typer.secho(f"  note      {note}", fg=typer.colors.YELLOW)
    for error in result.errors:
        typer.secho(f"  error     {error.code}: {error.message}", fg=typer.colors.RED, err=True)
    typer.echo(f"  report    {report_path}")

    if result.exit_code() is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code())


@marker_app.command("build")
def marker_build(
    output_directory: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            dir_okay=True,
            help="Where to write the WAV, the standalone page, and the manifest. Created if "
            "it does not exist.",
        ),
    ],
    marker: Annotated[
        str | None,
        typer.Option(
            "--marker",
            # Hidden rather than absent: the bench must drive the shipped command through its
            # real guards — the CLI wiring is where M7a's P0 lived — but this charter's
            # non-goals exclude a public candidate-management interface, and after v1 is
            # frozen the discoverable surface is `marker build OUTPUT_DIRECTORY` alone
            # (ADR-0041). Documented in docs/M10-marker-bench-protocol.md.
            hidden=True,
            help="Build a named bench candidate instead of the frozen v1.",
        ),
    ] = None,
) -> None:
    """Write the marker WAV, the offline phone page, and the manifest.

    Deterministic: the same marker produces byte-identical artifacts every time, and the WAV
    embedded in the page is the same bytes as the `.wav` beside it rather than a second
    encoding of the same samples.

    Until the phone/DJI bench selects a waveform there is no `v1`, and this refuses rather
    than defaulting to a candidate — an operator who recorded Session Zero against an
    unvalidated marker would have no way to find out (ADR-0042).
    """
    from dnd_audio.marker.builder import build_marker
    from dnd_audio.marker.spec import resolve

    try:
        spec = resolve(marker)
    except DndAudioError as exc:
        typer.secho(f"  error  {exc.code}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=ExitCode.FATAL) from exc

    # Before `mkdir`, before a byte is written, and before the previous manifest is
    # unlinked — on this path the unlink would itself be the violation (INV-01, ADR-0021).
    try:
        _reject_path_inside_any_session("marker artifacts", output_directory)
    except DndAudioError as exc:
        typer.secho(
            f"  nothing written: {output_directory} resolves inside a session's own "
            f"sources, and nothing under them may be written to (INV-01)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=ExitCode.FATAL) from exc

    built = build_marker(spec, output_directory)
    typer.echo(f"  marker    {built.manifest.marker_name} ({built.manifest.rationale})")
    typer.echo(f"  duration  {built.manifest.duration_seconds:.3f}s, {spec.total_samples} samples")
    typer.echo(f"  wav       {built.wav_path}  sha256 {built.manifest.wav.sha256}")
    typer.echo(f"  page      {built.page_path}")
    typer.echo(f"  manifest  {built.manifest_path}")


@marker_app.command("analyze")
def marker_analyze(
    session_dir: SessionDir,
    marker: Annotated[
        str | None,
        typer.Option("--marker", hidden=True, help="Analyze for a named bench candidate."),
    ] = None,
    reference_track: Annotated[
        str | None,
        typer.Option(
            "--reference-track",
            help="Which track anchors every occurrence group. Defaults to the track with the "
            "most accepted occurrences, tie-broken lexically.",
        ),
    ] = None,
    event_log: Annotated[
        Path | None,
        typer.Option(
            "--event-log",
            exists=True,
            dir_okay=False,
            help="The operator's independent log of what was played and when. Without it, "
            "roles are assigned only when each default window holds exactly one occurrence, "
            "and no drift classification is possible.",
        ),
    ] = None,
) -> None:
    """Find the marker on every track and write the analysis and the report.

    Reads the session's existing `manifest.json` and `timeline.json` and **never rebuilds or
    rewrites them** — a stale one is refused with a code naming which component disagrees,
    rather than silently trusted. `ingest` must have run.

    It verifies the jam and measures differential acoustic arrival. It never places a file,
    never overrides valid timecode, and calls a start-to-end change recorder drift only when
    the event log asserts that the phone and every compared transmitter stayed fixed
    (ADR-0040).
    """
    from dnd_audio.marker.runner import run_marker_analyze

    result = run_marker_analyze(
        session_dir,
        marker=marker,
        reference_track=reference_track,
        event_log=event_log,
    )
    _summarize_marker(result)
    if result.exit_code is not ExitCode.OK:
        raise typer.Exit(code=result.exit_code)


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
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Which device to check, as `asr.device` would request it: auto, cpu, or "
            "cuda. `cuda` fails when this machine cannot deliver it, which is the point.",
        ),
    ] = "auto",
    dtype: Annotated[
        str,
        typer.Option(
            "--dtype",
            help="Which precision to check, as `asr.dtype` would request it: auto, "
            "float32, or bfloat16. Checked on whatever device the run resolves to.",
        ),
    ] = "auto",
) -> None:
    """Check this host without touching session audio.

    Reports each pinned model as a warning when it is absent rather than as a failure —
    that is a machine that can still do everything the missing model is not needed for.
    A host with no ASR snapshots mixes and detects; a host with no VAD model does neither.

    `--device` and `--dtype` answer "will *my* configuration work here" before a session
    starts rather than during it. An explicitly requested combination this machine cannot
    deliver is a failure with a diagnostic naming what is wrong, never a quiet downgrade
    to something that would have produced different numbers (ADR-0026).
    """
    if device not in _DEVICE_CHOICES:
        message = f"--device must be one of {', '.join(_DEVICE_CHOICES)}, not {device!r}"
        raise typer.BadParameter(message)
    if dtype not in _DTYPE_CHOICES:
        message = f"--dtype must be one of {', '.join(_DTYPE_CHOICES)}, not {dtype!r}"
        raise typer.BadParameter(message)

    # No cast: the membership tests above narrow both to their Literal types.
    results = run_checks(path, device=device, dtype=dtype)
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


def _summarize_marker(result: MarkerAnalysisResult) -> None:
    """Human-readable progress for `marker analyze`. The analysis holds everything."""
    analysis = result.analysis
    if analysis is not None:
        typer.echo(
            f"  {len(analysis.occurrences)} occurrence(s) across "
            f"{len(analysis.groups)} group(s), reference {analysis.identity.reference_track}"
        )
        for comparison in analysis.arrival:
            typer.echo(
                f"  arrival    {comparison.track_id}: {comparison.outcome.value}"
                + (f" ({comparison.change_ms:+d} ms)" if comparison.change_ms is not None else "")
            )
        if result.report.inconclusive:
            typer.secho(
                "  inconclusive: the command ran and the evidence settled nothing. That is a "
                "result about the room, not a failure.",
                fg=typer.colors.YELLOW,
            )
        typer.echo(f"  analysis   {result.analysis_path}")
    for warning in result.report.warnings:
        typer.secho(f"  warn  {warning.code}: {warning.message}", fg=typer.colors.YELLOW, err=True)
    for error in result.report.errors:
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
