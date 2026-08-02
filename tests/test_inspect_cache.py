"""INV-08: the cache reuses work only when reusing it is provably safe.

The identity tests vary one component at a time. That matters more than it looks: a
cache key that happened to include everything today, assembled by hand, is one refactor
away from silently dropping a component, and the failure mode is a stale answer rather
than a crash.

The integration tests then prove the key is actually *used* — a perfect key that the
runner does not consult would pass every test above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dnd_audio.determinism import write_json_atomic
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection import cache as cache_module
from dnd_audio.inspection.cache import InspectionCache, cache_key
from dnd_audio.inspection.probe import ProbeResult, ToolVersions
from dnd_audio.inspection.runner import run_inspect

TOOLS = ToolVersions(ffmpeg="ffmpeg version 8.0", ffprobe="ffprobe version 8.0")

BASE: dict[str, Any] = {
    "relative_path": "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav",
    "source_sha256": "a" * 64,
    "config_hash": "b" * 64,
    "tools": TOOLS,
    "ffprobe_args": ("-show_format", "-show_streams"),
}


class TestIdentity:
    def test_the_same_inputs_give_the_same_key(self) -> None:
        assert cache_key(**BASE) == cache_key(**BASE)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("relative_path", "raw/tx-b/TX02_MIC001_20260815_190000_orig.wav"),
            ("source_sha256", "c" * 64),
            ("config_hash", "d" * 64),
            ("tools", ToolVersions(ffmpeg="ffmpeg version 8.1", ffprobe="ffprobe version 8.0")),
            ("tools", ToolVersions(ffmpeg="ffmpeg version 8.0", ffprobe="ffprobe version 8.1")),
            ("ffprobe_args", ("-show_format",)),
        ],
    )
    def test_changing_any_component_changes_the_key(self, field: str, value: object) -> None:
        assert cache_key(**{**BASE, field: value}) != cache_key(**BASE)

    def test_the_path_is_part_of_the_identity(self) -> None:
        """Two byte-identical files at two paths have genuinely different captures.

        FFprobe echoes the filename into its own output, so the sidecar differs; and
        which recovery override applies is keyed by path, so the start-time evidence can
        differ too. Keying on content alone would serve one file's capture for the other.
        """
        elsewhere = cache_key(**{**BASE, "relative_path": "raw/tx-b/same-bytes.wav"})
        assert elsewhere != cache_key(**BASE)

    def test_ffmpeg_and_ffprobe_are_tracked_separately(self) -> None:
        """They are separate binaries and upgrade independently."""
        only_ffmpeg = cache_key(
            **{**BASE, "tools": ToolVersions(ffmpeg="new", ffprobe=TOOLS.ffprobe)}
        )
        only_ffprobe = cache_key(
            **{**BASE, "tools": ToolVersions(ffmpeg=TOOLS.ffmpeg, ffprobe="new")}
        )
        assert len({cache_key(**BASE), only_ffmpeg, only_ffprobe}) == 3

    def test_a_semantics_bump_invalidates_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The finding this exists for: a cache that varied only the RIFF-parser
        version would keep serving the answer a fixed strategy-chain bug produced."""
        before = cache_key(**BASE)
        monkeypatch.setattr(cache_module, "INSPECTION_SEMANTICS_VERSION", 999)
        assert cache_key(**BASE) != before

    def test_a_manifest_schema_bump_invalidates_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record is stored in the manifest's shape, so a new shape cannot read it."""
        before = cache_key(**BASE)
        monkeypatch.setattr(cache_module, "MANIFEST_SCHEMA_VERSION", 999)
        assert cache_key(**BASE) != before


class TestStore:
    def test_a_staged_record_is_not_readable_until_it_commits(self, tmp_path: Path) -> None:
        cache = InspectionCache(directory=tmp_path)
        cache.stage("k", {"container": {"sample_rate": 48000}})
        assert cache.get("k") is None

        cache.commit()
        assert InspectionCache(directory=tmp_path).get("k") == {"container": {"sample_rate": 48000}}

    def test_discarding_writes_nothing(self, tmp_path: Path) -> None:
        """A run that failed after inspecting must leave no trace of what it saw."""
        cache = InspectionCache(directory=tmp_path)
        cache.stage("k", {"container": {}})
        cache.discard()
        cache.commit()
        assert not list(tmp_path.glob("*.json"))

    def test_an_interrupted_write_is_never_a_hit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """INV-08: an incomplete entry is never a hit.

        The atomic writer creates a temp file in the destination directory and renames
        it. Failing the rename is the closest reachable analogue of a crash mid-write,
        and it must leave the directory with nothing a later run would read.
        """

        def explode(self: Path, target: Path) -> None:
            message = "interrupted before the rename"
            raise OSError(message)

        monkeypatch.setattr(Path, "replace", explode)

        cache = InspectionCache(directory=tmp_path)
        cache.stage("k", {"container": {}})
        with pytest.raises(OSError, match="interrupted"):
            cache.commit()

        monkeypatch.undo()
        assert InspectionCache(directory=tmp_path).get("k") is None
        assert not list(tmp_path.glob("*.json"))

    def test_a_corrupted_entry_is_a_miss_rather_than_a_crash(self, tmp_path: Path) -> None:
        """A damaged cache should cost time, not a session."""
        (tmp_path / f"{'e' * 64}.json").write_text("{not json", encoding="utf-8")
        cache = InspectionCache(directory=tmp_path)
        assert cache.get("e" * 64) is None
        assert cache.misses == 1

    def test_reads_can_be_disabled_while_writes_continue(self, tmp_path: Path) -> None:
        """`--no-cache` distrusts what is stored; it does not refuse to store.

        Making it do both would turn "one slow run" into "every run slow", which is not
        what anyone reaches for it to do.
        """
        write_json_atomic(tmp_path / "k.json", {"key": "k", "payload": {"container": {}}})
        cache = InspectionCache(directory=tmp_path, read_enabled=False)
        assert cache.get("k") is None
        assert cache.hits == 0

        cache.stage("fresh", {"container": {"sample_rate": 48000}})
        assert cache.commit() == 1
        assert InspectionCache(directory=tmp_path).get("fresh") is not None


class TestTheRunnerActuallyUsesIt:
    """A perfect key the runner never consults would pass every test above."""

    def test_a_second_run_probes_nothing(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_inspect(canonical_fixture.session_dir)

        calls = _count_probes(monkeypatch)
        second = run_inspect(canonical_fixture.session_dir)

        assert calls == [], f"expected no probes on the second run, got {len(calls)}"
        assert second.report.telemetry.cache_hits == len(canonical_fixture.chunks)
        assert second.report.telemetry.cache_misses == 0

    def test_the_first_run_probes_every_selected_source(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _count_probes(monkeypatch)
        run_inspect(canonical_fixture.session_dir)
        assert len(calls) == len(canonical_fixture.chunks)

    def test_a_tool_version_bump_forces_re_inspection(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scenario INV-08 is written for: the bytes did not change, the parser did.

        Simulated by reporting a different FFprobe version rather than by installing
        one, which is the same input to the cache identity and does not need a second
        FFmpeg on the machine.
        """
        run_inspect(canonical_fixture.session_dir)

        real = ToolVersions(ffmpeg="ffmpeg version 8.0", ffprobe="ffprobe version 9.9-fake")
        monkeypatch.setattr("dnd_audio.inspection.runner.tool_versions", lambda: real)
        calls = _count_probes(monkeypatch)
        result = run_inspect(canonical_fixture.session_dir)

        assert len(calls) == len(canonical_fixture.chunks)
        assert result.report.telemetry.cache_hits == 0
        assert result.manifest.inspection.ffprobe_version == "ffprobe version 9.9-fake"

    def test_a_changed_source_forces_re_inspection_of_that_source_only(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_inspect(canonical_fixture.session_dir)

        target = canonical_fixture.for_track("tx-a")[0]
        path = canonical_fixture.session_dir / target.relative_path
        path.write_bytes(path.read_bytes() + b"\x00\x00\x00\x00")

        calls = _count_probes(monkeypatch)
        run_inspect(canonical_fixture.session_dir)
        assert calls == [target.relative_path]

    def test_no_cache_re_probes_but_still_writes_entries(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--no-cache` should cost one slow run, not every run.

        Deliberately starting from a *cold* cache: running it after a warm one would
        pass even if `--no-cache` wrote nothing at all, because the earlier run had
        already populated everything.
        """
        forced = _count_probes(monkeypatch)
        run_inspect(canonical_fixture.session_dir, use_cache=False)
        assert len(forced) == len(canonical_fixture.chunks)

        monkeypatch.undo()
        after = _count_probes(monkeypatch)
        result = run_inspect(canonical_fixture.session_dir)
        assert after == []
        assert result.report.telemetry.cache_hits == len(canonical_fixture.chunks)


def _count_probes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every FFprobe invocation the runner makes, and still perform it."""
    seen: list[str] = []
    real = run_inspect.__globals__["run_ffprobe"]

    def spy(session_dir: Path, relative_path: str) -> ProbeResult:
        seen.append(relative_path)
        result: ProbeResult = real(session_dir, relative_path)
        return result

    monkeypatch.setattr("dnd_audio.inspection.runner.run_ffprobe", spy)
    return seen
