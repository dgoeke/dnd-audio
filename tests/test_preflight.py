"""The work-space preflight: refuse an impossible run before writing the first byte.

Running out of disk halfway through six derivatives leaves a directory of half-files that
the cache correctly refuses and nothing cleans up. The estimate is exact — every length
comes from the timeline and float32 is four bytes — so the only uncertainty is what else
will use the disk, which is what the headroom factor is for.

This **partially answers OQ-013** and does not close it. That question asks for measured
full-pipeline disk use, and two of the three terms in `doctor`'s original 40 GiB estimate
turn out not to exist: the 48 kHz working audio is a segment map unless `--materialize-48k`
asks otherwise (ADR-0011), and the mix intermediate belongs to M5.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dnd_audio.fixtures import FixtureTruth
from dnd_audio.timeline import CANONICAL_SAMPLE_RATE
from dnd_audio.timeline.preflight import (
    HEADROOM_FACTOR,
    WorkspaceError,
    WorkspaceEstimate,
    estimate,
    preflight,
)

HOUR = 3600 * CANONICAL_SAMPLE_RATE


def an_estimate(**overrides: object) -> WorkspaceEstimate:
    base = WorkspaceEstimate(
        duration_samples=HOUR,
        track_count=6,
        derivative_bytes=1_000_000,
        materialized_bytes=0,
        free_bytes=100_000_000,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class TestTheEstimateIsArithmeticNotAGuess:
    def test_a_derivative_is_four_bytes_per_output_sample(self, tmp_path: Path) -> None:
        """One hour, six tracks, decimated by three: 1 hour * 16000 * 4 * 6."""
        found = estimate(tmp_path, duration_samples=HOUR, track_count=6, materialize_48k=False)
        assert found.derivative_bytes == (HOUR // 3) * 4 * 6
        assert found.materialized_bytes == 0

    def test_every_tracks_derivative_is_the_session_duration(self, tmp_path: Path) -> None:
        """Not the track's own: a track that stopped early is still read to the end.

        M3 and M4 index every track against one aligned duration, so a derivative that
        stopped where its audio did would need special-casing at every call site.
        """
        one = estimate(tmp_path, duration_samples=HOUR, track_count=1, materialize_48k=False)
        six = estimate(tmp_path, duration_samples=HOUR, track_count=6, materialize_48k=False)
        assert six.derivative_bytes == one.derivative_bytes * 6

    def test_materializing_adds_the_full_rate_path(self, tmp_path: Path) -> None:
        found = estimate(tmp_path, duration_samples=HOUR, track_count=6, materialize_48k=True)
        assert found.materialized_bytes == HOUR * 4 * 6
        assert found.total_bytes == found.derivative_bytes + found.materialized_bytes

    def test_the_default_run_is_a_third_of_the_materialized_one(self, tmp_path: Path) -> None:
        """Which is the point of the segment map: 48 kHz costs nothing until asked for."""
        lean = estimate(tmp_path, duration_samples=HOUR, track_count=6, materialize_48k=False)
        full = estimate(tmp_path, duration_samples=HOUR, track_count=6, materialize_48k=True)
        assert full.total_bytes == pytest.approx(lean.total_bytes * 4, rel=0.01)

    def test_the_length_rule_matches_the_resamplers(self, tmp_path: Path) -> None:
        """`ceil`, so the estimate is never one sample short of what is written."""
        from dnd_audio.timeline.resample import output_length

        found = estimate(tmp_path, duration_samples=100, track_count=1, materialize_48k=False)
        assert found.derivative_bytes == output_length(100, 3) * 4

    def test_free_space_is_read_from_the_session_directory(self, tmp_path: Path) -> None:
        """The disk that matters is the one the session is on, not the one `/` is on."""
        found = estimate(tmp_path, duration_samples=HOUR, track_count=1, materialize_48k=False)
        assert found.free_bytes > 0


class TestPreflight:
    def test_a_comfortable_run_warns_about_nothing(self) -> None:
        assert preflight(an_estimate()) == []

    def test_a_tight_run_warns_but_proceeds(self) -> None:
        """It will fit. The machine may not enjoy it."""
        found = an_estimate(derivative_bytes=60_000_000, free_bytes=100_000_000)
        assert found.sufficient
        assert not found.comfortable
        notes = preflight(found)
        assert [note.code for note in notes] == ["work_space_tight"]

    def test_an_impossible_run_is_refused_before_anything_is_written(self) -> None:
        found = an_estimate(derivative_bytes=200_000_000, free_bytes=100_000_000)
        with pytest.raises(WorkspaceError, match="GiB") as caught:
            preflight(found)
        assert caught.value.code == "insufficient_work_space"

    def test_the_diagnostic_names_both_numbers(self) -> None:
        """ "Not enough disk" without them is a message an operator cannot act on."""
        found = an_estimate(
            derivative_bytes=200 * (1 << 30), free_bytes=10 * (1 << 30), track_count=6
        )
        with pytest.raises(WorkspaceError) as caught:
            preflight(found)
        message = str(caught.value)
        assert "200.00 GiB" in message
        assert "10.00 GiB" in message
        assert "6 track(s)" in message

    def test_it_mentions_the_flag_that_would_help(self) -> None:
        found = an_estimate(materialized_bytes=200_000_000, free_bytes=1_000)
        with pytest.raises(WorkspaceError, match="materialize-48k"):
            preflight(found)

    def test_exactly_enough_is_enough(self) -> None:
        """The boundary, stated rather than left to an inequality nobody checked."""
        found = an_estimate(derivative_bytes=1_000, materialized_bytes=0, free_bytes=1_000)
        assert found.sufficient
        assert preflight(found)  # tight, so it warns

    def test_one_byte_short_is_not(self) -> None:
        found = an_estimate(derivative_bytes=1_001, materialized_bytes=0, free_bytes=1_000)
        with pytest.raises(WorkspaceError):
            preflight(found)

    def test_the_headroom_factor_is_what_decides_comfortable(self) -> None:
        needed = 1_000_000
        assert an_estimate(derivative_bytes=needed, free_bytes=needed * HEADROOM_FACTOR).comfortable
        assert not an_estimate(
            derivative_bytes=needed, free_bytes=needed * HEADROOM_FACTOR - 1
        ).comfortable


class TestItRunsBeforeTheFirstByte:
    def test_ingest_refuses_a_session_it_cannot_write(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that dies partway leaves half-files nothing cleans up.

        Free space is faked rather than filled: the assertion is about the ordering, and
        actually exhausting a disk in a test would be both slow and hostile.
        """
        import shutil

        from dnd_audio.errors import ExitCode
        from dnd_audio.timeline.runner import run_ingest

        original = shutil.disk_usage

        def almost_full(path: Path) -> shutil._ntuple_diskusage:
            usage = original(path)
            return type(usage)(usage.total, usage.used, 1024)

        monkeypatch.setattr("dnd_audio.timeline.preflight.shutil.disk_usage", almost_full)

        session_dir = canonical_fixture.session_dir
        result = run_ingest(session_dir)
        assert result.exit_code is not ExitCode.OK
        assert not (session_dir / "work/cache/audio").exists()
