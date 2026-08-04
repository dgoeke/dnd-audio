"""Hand-built manifests, for testing placement without generating audio.

The timeline is built from `manifest.json`, so a placement test needs a manifest and
nothing else. Synthesizing six WAV files to exercise one rollover rule would be slow and,
worse, would hide the rule under everything else the fixture generator does.

These builders are deliberately *not* the fixture generator. The end-to-end tests use the
real generator and assert against :class:`~dnd_audio.fixtures.FixtureTruth`; these build
the exact evidence a case needs, including shapes the generator refuses to write — a
chunk whose timecode does not land on a whole sample, a source that is 44.1 kHz, a track
whose chunks disagree.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from dnd_audio.artifacts.manifest import (
    BwfSampleReferenceRecord,
    ContainerRecord,
    FilenameHintsRecord,
    InspectionProvenance,
    Manifest,
    ManifestSource,
    ManifestTrack,
    RationalRate,
    SessionOffsetRecord,
    StartEvidenceRecord,
    StartTimeRecord,
    TimecodeRecord,
)
from dnd_audio.artifacts.roster import RosterSummary
from dnd_audio.config import SessionConfig
from dnd_audio.determinism import sha256_bytes
from dnd_audio.timecode import FRAME_RATES, frame_index, parse_timecode
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE


def digest(seed: str) -> str:
    """A stable stand-in hash for a file that does not exist.

    From SHA-256 rather than Python's `hash()`, which is salted per process: a manifest
    that differed between interpreter runs is exactly the almost-determinism INV-02 is
    about, and a test asserting on byte-stability would then be asserting on luck.
    """
    return sha256_bytes(seed.encode("utf-8"))


def bwf(
    samples: int,
    *,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    date: dt.date | None = None,
) -> BwfSampleReferenceRecord:
    """Samples from the recorder's own origin, at the file's own rate (ADR-0031)."""
    return BwfSampleReferenceRecord(samples=samples, sample_rate=sample_rate, origination_date=date)


def timecode(text: str, label: str = "30F", *, date: dt.date | None = None) -> TimecodeRecord:
    """A timecode tag, as an exact frame index plus its rational rate."""
    rate = FRAME_RATES[label]
    return TimecodeRecord(
        text=text,
        frames=frame_index(parse_timecode(text, rate)),
        frame_rate_label=label,
        frame_rate=RationalRate(numerator=rate.rate.numerator, denominator=rate.rate.denominator),
        drop_frame=rate.drop_frame,
        recording_date=date,
    )


def offset(samples: int, *, date: dt.date | None = None) -> SessionOffsetRecord:
    """A signed operator-supplied offset at 48 kHz, relative to session zero."""
    return SessionOffsetRecord(
        samples=samples, sample_rate=CANONICAL_SAMPLE_RATE, recording_date=date
    )


def source(
    path: str,
    evidence: StartEvidenceRecord,
    *,
    n_samples: int = 48000,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    codec: str = "pcm_f32le",
    channels: int = 1,
    role: str = "selected",
    with_container: bool = True,
) -> ManifestSource:
    """One selected source, described exactly as M1 would describe it."""
    container = (
        ContainerRecord(
            codec_name=codec,
            sample_format="flt",
            bits_per_sample=32,
            sample_rate=sample_rate,
            channels=channels,
            sample_count=n_samples,
            sample_count_source="data_chunk",
        )
        if with_container
        else None
    )
    return ManifestSource(
        relative_path=path,
        sha256=digest(path),
        size_bytes=n_samples * 4 + 44,
        role=role,  # type: ignore[arg-type]
        reason_code="selected_original",
        detail="built by tests/manifests.py",
        filename=FilenameHintsRecord(recognized=True, variant="orig"),
        container=container,
        start_time=StartTimeRecord(strategy="test", evidence=evidence),
    )


def manifest(tracks: dict[str, list[ManifestSource]], *, session_id: str = "test") -> Manifest:
    """A manifest holding exactly these tracks and sources."""
    return Manifest(
        session_id=session_id,
        config_hash="0" * 64,
        inspection=InspectionProvenance(
            ffmpeg_version="test", ffprobe_version="test", semantics_version=1
        ),
        roster=RosterSummary(
            known_tracks=sorted(tracks),
            active_tracks=sorted(track for track, items in tracks.items() if items),
            inactive_tracks=sorted(track for track, items in tracks.items() if not items),
        ),
        tracks=[
            ManifestTrack(
                track_id=track_id,
                speaker_id=track_id.replace("-", ""),
                speaker_name=track_id.upper(),
                input_path=f"raw/{track_id}",
                active=bool(sources),
                inactive_reason=None if sources else "no usable original",
                sources=sources,
            )
            for track_id, sources in sorted(tracks.items())
        ],
    )


def config(**timecode_fields: Any) -> SessionConfig:
    """A session configuration over tracks `tx-a` and `tx-b`, with timecode overrides."""
    return config_for(("tx-a", "tx-b"), **timecode_fields)


def config_for(track_ids: tuple[str, ...], **timecode_fields: Any) -> SessionConfig:
    return SessionConfig.model_validate(
        {
            "session_id": "test",
            "title": "Test",
            "timecode": timecode_fields,
            "tracks": [
                {
                    "track_id": track_id,
                    "receiver_id": f"rx-{index}",
                    "receiver_channel": 1,
                    "speaker_id": track_id.replace("-", ""),
                    "speaker_name": track_id.upper(),
                    "input": f"raw/{track_id}",
                }
                for index, track_id in enumerate(track_ids)
            ],
        }
    )
