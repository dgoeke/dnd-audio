"""Which files exist, which are used, and which person each one belongs to.

The rules come from the spec's session input contract and its selection list. What is
worth stating up front is the shape they produce, because the obvious shape is wrong.

**Every candidate is recorded, including the ones nothing will read.** An ignored `edit`
file, a duplicate, a file in a directory nobody configured — all of them appear, each
with a role and a stable reason code. The alternative is worse in both directions: drop
them and the per-file capture the gate requires is incomplete; attach them to a track and
an unconfigured directory has just been assigned to a speaker, which is the INV-11
violation the whole roster design exists to prevent. So there is a track-independent
:attr:`Discovery.unassigned` list, and nothing in it has a ``track_id``.

**Identity comes from the configured directory and from nothing else** (INV-11). A
filename's ``TX01`` is a hint used to *warn* — a directory holding two apparent
transmitters is worth knowing about — and it is never consulted to decide whose voice a
file holds.

**Presence is not attendance.** Under ``active_tracks: auto`` a configured track with no
usable original is inactive with a warning, never silently dropped. Under an explicit
list, every listed track is required and a missing one is fatal, because that is the only
way an operator can distinguish "Erin did not come" from "Erin's recorder failed".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from dnd_audio.config import SessionConfig, TrackConfig
from dnd_audio.determinism import sha256_file
from dnd_audio.errors import DiscoveryError, RecoveryError
from dnd_audio.inspection.naming import (
    AUDIO_SUFFIXES,
    FilenameHints,
    parse_filename,
    sequence_discontinuities,
)

__all__ = [
    "DiscoveredFile",
    "DiscoveredSource",
    "Discovery",
    "DiscoveryDecision",
    "DiscoveryWarning",
    "SourceRole",
    "TrackDiscovery",
    "discover",
]

SourceRole = Literal["selected", "associated_edit", "duplicate", "unassigned", "unusable"]

#: Where an `orig` and an `edit` of the same recording are paired. Keyed on the parsed
#: transmitter label and counter when the name is recognized, and on the variant-stripped
#: stem otherwise — so association still works for names the grammar does not know
#: (OQ-003, OQ-007).
_VARIANT_SUFFIXES: Final = ("_orig", "_edit")


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """One file on disk, hashed, before any rule has been applied to it."""

    relative_path: str
    size_bytes: int
    sha256: str
    hints: FilenameHints


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    """A file and what discovery decided about it."""

    file: DiscoveredFile
    role: SourceRole
    #: Stable and machine-readable. Reworded prose in a report is not something a
    #: consumer can branch on.
    reason_code: str
    detail: str
    #: For an `edit`, the `orig` it belongs to. For a duplicate, the copy that was kept.
    associated_with: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryWarning:
    """Something the operator should look at that did not stop the run."""

    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryDecision:
    """A choice worth auditing later. Deterministic — no counts, no timings."""

    code: str
    subject: str
    detail: str


@dataclass(frozen=True, slots=True)
class TrackDiscovery:
    """One roster track and everything found in its directory."""

    track_id: str
    speaker_id: str
    speaker_name: str
    input_path: str
    active: bool
    #: Why the track is inactive. ``None`` when it is active — an inactive track with no
    #: reason is indistinguishable from one nobody looked at.
    inactive_reason: str | None
    sources: tuple[DiscoveredSource, ...]

    @property
    def selected(self) -> tuple[DiscoveredSource, ...]:
        return tuple(source for source in self.sources if source.role == "selected")


@dataclass(frozen=True, slots=True)
class Discovery:
    """The whole session's file layout, decided."""

    tracks: tuple[TrackDiscovery, ...]
    #: Candidates found where no track was configured. Captured in full, attributed to
    #: nobody (INV-11).
    unassigned: tuple[DiscoveredSource, ...]
    missing_directories: tuple[str, ...]
    empty_directories: tuple[str, ...]
    extra_directories: tuple[str, ...]
    warnings: tuple[DiscoveryWarning, ...]
    decisions: tuple[DiscoveryDecision, ...]

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(track.track_id for track in self.tracks if track.active)

    def all_sources(self) -> tuple[DiscoveredSource, ...]:
        """Every candidate in the session, assigned or not, in path order."""
        found = [source for track in self.tracks for source in track.sources]
        found.extend(self.unassigned)
        return tuple(sorted(found, key=lambda source: source.file.relative_path))


def discover(session_dir: Path, config: SessionConfig) -> Discovery:
    """Apply the selection and roster rules to what is on disk.

    Raises:
        DiscoveryError: if a track has only processed audio and
            ``recovery.allow_processed_audio`` is off, or if a track named in an
            explicit ``active_tracks`` list has no usable original.
        RecoveryError: if a configured ``source_time_overrides`` key matches no
            discovered file (ADR-0007).
    """
    warnings: list[DiscoveryWarning] = []
    decisions: list[DiscoveryDecision] = []

    found: dict[str, list[DiscoveredFile]] = {}
    missing: list[str] = []
    empty: list[str] = []
    for track in config.tracks:
        directory = session_dir / track.input
        if not directory.is_dir():
            missing.append(track.input)
            found[track.track_id] = []
            continue
        files = _scan(session_dir, directory, warnings)
        if not files:
            empty.append(track.input)
        found[track.track_id] = files

    extra_dirs, unassigned_files = _scan_unconfigured(session_dir, config, warnings)

    duplicates = _duplicate_map(
        [item for items in found.values() for item in items] + unassigned_files
    )

    tracks: list[TrackDiscovery] = []
    for track in config.tracks:
        tracks.append(
            _decide_track(track, found[track.track_id], duplicates, warnings, decisions, config)
        )

    tracks = _apply_roster(tracks, config, warnings, decisions)

    unassigned = tuple(
        DiscoveredSource(
            file=item,
            role="unassigned",
            reason_code="directory_not_configured",
            detail=(
                "found outside every configured track directory, so it is recorded but "
                "attributed to nobody (INV-11)"
            ),
        )
        for item in sorted(unassigned_files, key=lambda item: item.relative_path)
    )

    discovery = Discovery(
        tracks=tuple(tracks),
        unassigned=unassigned,
        missing_directories=tuple(sorted(missing)),
        empty_directories=tuple(sorted(empty)),
        extra_directories=tuple(sorted(extra_dirs)),
        warnings=tuple(
            sorted(warnings, key=lambda item: (item.code, item.path or "", item.message))
        ),
        decisions=tuple(sorted(decisions, key=lambda item: (item.code, item.subject))),
    )
    _verify_overrides_match(discovery, config)
    return discovery


def _scan(
    session_dir: Path, directory: Path, warnings: list[DiscoveryWarning]
) -> list[DiscoveredFile]:
    """Hash every audio candidate in one directory.

    Candidacy is by file extension, never by name shape. The DJI grammar is still a
    guess (OQ-003), and a guess used as a filter silently hides real hardware.
    """
    files: list[DiscoveredFile] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(session_dir).as_posix()
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            warnings.append(
                DiscoveryWarning(
                    code="unexpected_file_type",
                    message=f"{path.suffix or 'a file with no extension'} is not an audio "
                    f"format this pipeline reads; ignored",
                    path=relative,
                )
            )
            continue
        files.append(
            DiscoveredFile(
                relative_path=relative,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                hints=parse_filename(path.name),
            )
        )
    return files


def _scan_unconfigured(
    session_dir: Path, config: SessionConfig, warnings: list[DiscoveryWarning]
) -> tuple[list[str], list[DiscoveredFile]]:
    """Directories beside the configured ones, and the candidates inside them."""
    configured = {track.input for track in config.tracks}
    roots = {str(PurePosixPath(track.input).parent) for track in config.tracks}

    extra: list[str] = []
    files: list[DiscoveredFile] = []
    for root in sorted(roots):
        directory = session_dir / root if root not in (".", "") else session_dir
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(session_dir).as_posix()
            if path.is_dir():
                if relative in configured:
                    continue
                extra.append(relative)
                warnings.append(
                    DiscoveryWarning(
                        code="extra_directory",
                        message="no track is configured for this directory; its files are "
                        "recorded but never attributed to a speaker (INV-11)",
                        path=relative,
                    )
                )
                files.extend(_scan(session_dir, path, warnings))
            elif path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                warnings.append(
                    DiscoveryWarning(
                        code="stray_audio_file",
                        message="audio outside any track directory; recorded but not "
                        "attributed to a speaker (INV-11)",
                        path=relative,
                    )
                )
                files.append(
                    DiscoveredFile(
                        relative_path=relative,
                        size_bytes=path.stat().st_size,
                        sha256=sha256_file(path),
                        hints=parse_filename(path.name),
                    )
                )
    return extra, files


def _duplicate_map(files: Sequence[DiscoveredFile]) -> dict[str, str]:
    """``{relative_path: the copy that was kept}`` for every byte-identical duplicate.

    Ordering is ``(is the file processed, path)``. The path breaks ties so that which
    copy survives does not depend on directory iteration order (INV-02); the variant
    comes first so a processed file can never displace an original it happens to sort
    before. Getting that backwards turns "the edit is ignored" into "the original is
    ignored and the track has only processed audio", which is a fatal error two rules
    downstream and gives no hint where it came from.
    """
    by_hash: dict[str, list[DiscoveredFile]] = {}
    for item in files:
        by_hash.setdefault(item.sha256, []).append(item)

    duplicates: dict[str, str] = {}
    for group in by_hash.values():
        if len(group) < 2:
            continue
        keep, *rest = sorted(
            group, key=lambda item: (item.hints.variant == "edit", item.relative_path)
        )
        for item in rest:
            duplicates[item.relative_path] = keep.relative_path
    return duplicates


def _decide_track(
    track: TrackConfig,
    files: Sequence[DiscoveredFile],
    duplicates: Mapping[str, str],
    warnings: list[DiscoveryWarning],
    decisions: list[DiscoveryDecision],
    config: SessionConfig,
) -> TrackDiscovery:
    """Apply the selection rules to one track's directory."""
    _warn_about_labels(track, files, warnings)

    originals = [item for item in files if item.hints.variant in ("orig", "unknown")]
    edits = [item for item in files if item.hints.variant == "edit"]
    live_originals = [item for item in originals if item.relative_path not in duplicates]

    use_edits = not live_originals and bool(edits)
    if use_edits and not config.recovery.allow_processed_audio:
        message = (
            f"track {track.track_id!r} has only processed audio "
            f"({', '.join(sorted(item.relative_path for item in edits))}). The 32-bit "
            f"float original is what this pipeline mixes from. Set "
            f"recovery.allow_processed_audio: true to consume the processed file anyway, "
            f"and expect a worse mix."
        )
        raise DiscoveryError(message, code="processed_audio_only")

    sources: list[DiscoveredSource] = []
    pairing = {_pairing_key(item.hints, item.relative_path): item for item in originals}

    for item in files:
        duplicate_of = duplicates.get(item.relative_path)
        if duplicate_of is not None:
            sources.append(_as_duplicate(item, duplicate_of, warnings))
            continue
        if item.hints.variant == "edit" and not use_edits:
            partner = pairing.get(_pairing_key(item.hints, item.relative_path))
            sources.append(
                DiscoveredSource(
                    file=item,
                    role="associated_edit",
                    reason_code="processed_variant_ignored",
                    detail="the original was selected instead",
                    associated_with=partner.relative_path if partner else None,
                )
            )
            if partner is not None:
                decisions.append(
                    DiscoveryDecision(
                        code="orig_selected",
                        subject=partner.relative_path,
                        detail=f"both orig and edit present; ignored {item.relative_path}",
                    )
                )
            continue
        sources.append(_as_selected(item, use_edits=use_edits, warnings=warnings))

    if use_edits:
        decisions.append(
            DiscoveryDecision(
                code="processed_audio_allowed",
                subject=track.track_id,
                detail="no original exists; recovery.allow_processed_audio permitted the edit",
            )
        )

    _warn_about_sequences(track, files, warnings)

    return TrackDiscovery(
        track_id=track.track_id,
        speaker_id=track.speaker_id,
        speaker_name=track.speaker_name,
        input_path=track.input,
        active=any(source.role == "selected" for source in sources),
        inactive_reason=None if sources else "no usable original recording was found",
        sources=tuple(sorted(sources, key=lambda source: source.file.relative_path)),
    )


def _as_selected(
    item: DiscoveredFile, *, use_edits: bool, warnings: list[DiscoveryWarning]
) -> DiscoveredSource:
    if item.hints.variant == "unknown":
        warnings.append(
            DiscoveryWarning(
                code="variant_not_determined",
                message="the filename carries no orig/edit marker, so it is treated as an "
                "original (OQ-007)",
                path=item.relative_path,
            )
        )
    if not item.hints.recognized:
        warnings.append(
            DiscoveryWarning(
                code="unrecognized_filename",
                message="the filename does not match the assumed DJI grammar (OQ-003); it is "
                "still inspected, but its label and counter hints are unavailable",
                path=item.relative_path,
            )
        )
    return DiscoveredSource(
        file=item,
        role="selected",
        reason_code="processed_audio_selected" if use_edits else "original_selected",
        detail=(
            "consumed in place of a missing original"
            if use_edits
            else "the original recording for this chunk"
        ),
    )


def _as_duplicate(
    item: DiscoveredFile, keeps: str, warnings: list[DiscoveryWarning]
) -> DiscoveredSource:
    across_tracks = PurePosixPath(item.relative_path).parent != PurePosixPath(keeps).parent
    warnings.append(
        DiscoveryWarning(
            code="duplicate_across_tracks" if across_tracks else "duplicate_source",
            message=(
                f"byte-identical to {keeps}"
                + (
                    ", in a different track directory — one of the two is attributed to the "
                    "wrong person"
                    if across_tracks
                    else "; ignored"
                )
            ),
            path=item.relative_path,
        )
    )
    return DiscoveredSource(
        file=item,
        role="duplicate",
        reason_code="duplicate_content",
        detail=f"byte-identical to {keeps}, which was kept",
        associated_with=keeps,
    )


def _warn_about_labels(
    track: TrackConfig, files: Iterable[DiscoveredFile], warnings: list[DiscoveryWarning]
) -> None:
    """The spec's "files belonging to more than one apparent transmitter" warning.

    A warning and never an error: OQ-002 says the label is not unique across kits, so it
    cannot be trusted enough to reject a file. It is still the cheapest available signal
    that files were copied into the wrong directory.
    """
    labels = sorted({item.hints.tx_label for item in files if item.hints.tx_label is not None})
    if len(labels) > 1:
        warnings.append(
            DiscoveryWarning(
                code="mixed_transmitter_labels",
                message=f"holds files labelled {', '.join(labels)}. The directory is the "
                f"identity (INV-11), so all of them are attributed to "
                f"{track.speaker_name}; check the labels are not telling you something.",
                path=track.input,
            )
        )


def _warn_about_sequences(
    track: TrackConfig, files: Sequence[DiscoveredFile], warnings: list[DiscoveryWarning]
) -> None:
    gaps = sequence_discontinuities([item.hints for item in files])
    for earlier, later in gaps:
        warnings.append(
            DiscoveryWarning(
                code="sequence_discontinuity",
                message=f"the file counter jumps from {earlier} to {later}. A power cycle "
                f"explains this; so does a file that never made it off the recorder.",
                path=track.input,
            )
        )


def _apply_roster(
    tracks: Sequence[TrackDiscovery],
    config: SessionConfig,
    warnings: list[DiscoveryWarning],
    decisions: list[DiscoveryDecision],
) -> list[TrackDiscovery]:
    """Turn "has a usable original" into "was in this session", per `active_tracks`.

    Raises:
        DiscoveryError: if an explicitly listed track has nothing usable.
    """
    if config.active_tracks == "auto":
        resolved: list[TrackDiscovery] = []
        for track in tracks:
            if not track.active:
                warnings.append(
                    DiscoveryWarning(
                        code="track_inactive",
                        message=f"{track.speaker_name} is on the roster but has no usable "
                        f"original recording. Under active_tracks: auto this is treated as "
                        f"an absence; list the track explicitly to make it required.",
                        path=track.input_path,
                    )
                )
            resolved.append(track)
        decisions.append(
            DiscoveryDecision(
                code="active_tracks_derived",
                subject="auto",
                detail="active participants derived from directories holding a usable original: "
                + (", ".join(t.track_id for t in resolved if t.active) or "none"),
            )
        )
        return resolved

    required = set(config.active_tracks)
    absent = sorted(
        track.track_id for track in tracks if track.track_id in required and not track.active
    )
    if absent:
        message = (
            f"active_tracks requires {', '.join(sorted(required))}, but "
            f"{', '.join(absent)} has no usable original recording. An explicit list means "
            f"attendance is known in advance, so a missing recording is a capture failure "
            f"rather than an absence."
        )
        raise DiscoveryError(message, code="required_track_missing")

    resolved = []
    for track in tracks:
        if track.track_id in required:
            resolved.append(track)
            continue
        resolved.append(
            TrackDiscovery(
                track_id=track.track_id,
                speaker_id=track.speaker_id,
                speaker_name=track.speaker_name,
                input_path=track.input_path,
                active=False,
                inactive_reason="not named in the explicit active_tracks list",
                sources=track.sources,
            )
        )
    decisions.append(
        DiscoveryDecision(
            code="active_tracks_explicit",
            subject="explicit",
            detail=f"required by configuration: {', '.join(sorted(required))}",
        )
    )
    return resolved


def _verify_overrides_match(discovery: Discovery, config: SessionConfig) -> None:
    """ADR-0007: an override aimed at a path that does not exist is fatal."""
    known = {source.file.relative_path for source in discovery.all_sources()}
    for key in sorted(config.recovery.source_time_overrides):
        if key in known:
            continue
        directory = str(PurePosixPath(key).parent)
        neighbours = sorted(path for path in known if str(PurePosixPath(path).parent) == directory)
        found = "\n  ".join(neighbours) if neighbours else "(nothing was found there)"
        message = (
            f"recovery.source_time_overrides names {key!r}, which no discovered source "
            f"matches. An override that silently applies to nothing is the failure the "
            f"recovery mechanism exists to prevent (ADR-0007). Found in {directory}:\n"
            f"  {found}"
        )
        raise RecoveryError(message, code="recovery_override_unmatched")


def _pairing_key(hints: FilenameHints, relative_path: str) -> str:
    """What makes an ``orig`` and an ``edit`` two views of the same recording.

    The parsed label and counter when the name is recognized; otherwise the stem with
    its variant suffix removed, which still pairs ``foo_orig.wav`` with ``foo_edit.wav``
    for a grammar this code does not know (OQ-007).
    """
    if hints.recognized and hints.tx_label is not None and hints.sequence is not None:
        return f"{hints.tx_label}:{hints.sequence:06d}"
    stem = PurePosixPath(relative_path).stem
    for suffix in _VARIANT_SUFFIXES:
        if stem.lower().endswith(suffix):
            return f"stem:{PurePosixPath(relative_path).parent}/{stem[: -len(suffix)]}"
    return f"stem:{PurePosixPath(relative_path).parent}/{stem}"
