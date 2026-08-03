"""INV-01's machinery, tested where it now lives.

M1 built these helpers inside `inspection/runner.py`, and its verify phase found two ways
they could pass while verifying nothing. Both are preserved here as tests, because the
helpers now serve two stages and a regression would be twice as expensive:

* **`"."` must stay in the roots.** Dropping it looks reasonable — every relative path is
  under `"."` — and for a session configured as `input: "tx-a"` it empties the snapshot, so
  `verify_unchanged` compares two empty dicts and passes no matter what happened.
* **Paths must be compared after resolution.** With `output -> raw/tx-a`, a lexical check
  sees nothing wrong and the run writes into a track's source directory.

The point of both is that a check which cannot fail is not a check, so each is driven with
the exact input that makes the thing it verifies actually change.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from dnd_audio.activity.runner import run_activity
from dnd_audio.config import SessionConfig
from dnd_audio.errors import DiscoveryError, ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.runner import inspect_outputs
from dnd_audio.mix.runner import run_mix
from dnd_audio.raw_guard import (
    raw_roots,
    reject_outputs_inside_raw,
    snapshot,
    verify_unchanged,
)
from dnd_audio.timeline.runner import ingest_outputs, run_ingest
from dnd_audio.transcript.runner import run_transcribe
from tests.manifests import config_for


def _scripted(session_dir: Path) -> Any:
    """The session's own declared fake detector, so a composed run needs no Silero (INV-05)."""
    from dnd_audio.transcript.fakemodels import load_fake_models

    return load_fake_models(session_dir).detector


#: Every command that composes more than one stage, in one place.
#:
#: M2, M3 and M4 each wrote an INV-01 regression test naming only the runner that milestone
#: had added, and all three carried the same bug for five milestones. So the composed commands
#: are enumerated here once and every INV-01 property below is parametrized over them: adding a
#: runner is then one missing entry in one list, which is visible in review.
COMPOSED: Any = [
    pytest.param(lambda d: run_ingest(d), id="ingest"),
    pytest.param(lambda d: run_activity(d, detector=_scripted(d)), id="activity"),
    pytest.param(lambda d: run_transcribe(d, fake_models=True), id="transcribe"),
    pytest.param(lambda d: run_mix(d, detector=_scripted(d)), id="mix"),
]


def a_session(root: Path, *, input_template: str = "raw/{track}") -> Path:
    """A minimal two-track session directory with real files under its sources."""
    for track in ("tx-a", "tx-b"):
        directory = root / input_template.format(track=track)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "one.wav").write_bytes(b"RIFF" + track.encode() + b"WAVEdata")
    return root


def a_config(input_template: str = "raw/{track}") -> SessionConfig:
    session = config_for(("tx-a", "tx-b"))
    tracks = [
        track.model_copy(update={"input": input_template.format(track=track.track_id)})
        for track in session.tracks
    ]
    return session.model_copy(update={"tracks": tracks})


class TestRawRoots:
    def test_the_canonical_layout_protects_raw(self) -> None:
        assert raw_roots(a_config()) == ("raw",)

    def test_a_track_directory_in_the_session_root_yields_dot(self) -> None:
        """The shape that emptied the snapshot in M1.

        `input: "tx-a"` has parent `"."`, and dropping it is what made INV-01's
        verification compare two empty dictionaries.
        """
        assert raw_roots(a_config("{track}")) == (".",)


class TestSnapshotAndVerify:
    def test_an_unchanged_tree_verifies(self, tmp_path: Path) -> None:
        session = a_session(tmp_path)
        roots = raw_roots(a_config())
        verify_unchanged(session, roots, snapshot(session, roots))

    def test_a_modified_file_is_caught(self, tmp_path: Path) -> None:
        session = a_session(tmp_path)
        roots = raw_roots(a_config())
        before = snapshot(session, roots)
        (session / "raw/tx-a/one.wav").write_bytes(b"RIFFchangedWAVEdata")
        with pytest.raises(DiscoveryError, match="modified") as caught:
            verify_unchanged(session, roots, before)
        assert caught.value.code == "raw_sources_modified"

    def test_a_removed_file_is_caught(self, tmp_path: Path) -> None:
        session = a_session(tmp_path)
        roots = raw_roots(a_config())
        before = snapshot(session, roots)
        (session / "raw/tx-b/one.wav").unlink()
        with pytest.raises(DiscoveryError, match="removed"):
            verify_unchanged(session, roots, before)

    def test_an_added_file_is_caught(self, tmp_path: Path) -> None:
        """A stage writing a normalized copy beside a source is what this forbids."""
        session = a_session(tmp_path)
        roots = raw_roots(a_config())
        before = snapshot(session, roots)
        (session / "raw/tx-a/one-normalized.wav").write_bytes(b"new")
        with pytest.raises(DiscoveryError, match="appeared"):
            verify_unchanged(session, roots, before)

    def test_a_root_level_layout_still_snapshots_its_sources(self, tmp_path: Path) -> None:
        """M1's defect, as a test that would have caught it.

        With `input: "tx-a"` the root is `"."`. If the snapshot were empty here, every
        assertion below would pass while INV-01 verified nothing at all.
        """
        session = a_session(tmp_path, input_template="{track}")
        roots = raw_roots(a_config("{track}"))
        before = snapshot(session, roots)
        assert before, "the snapshot is empty, so verification would pass unconditionally"
        assert set(before) == {"tx-a/one.wav", "tx-b/one.wav"}

        (session / "tx-a/one.wav").write_bytes(b"tampered")
        with pytest.raises(DiscoveryError):
            verify_unchanged(session, roots, before)

    def test_a_source_under_a_directory_named_work_is_still_hashed(self, tmp_path: Path) -> None:
        """The exclusion is the session's own `work/`, not the name at any depth.

        An earlier version matched the component anywhere in the path, so every file under
        `raw/tx-a/work/` — or under a source root called `archive/work` — was silently
        dropped from the snapshot and could be mutated without verification noticing. A
        check that is present, looks right, and verifies nothing.
        """
        session = a_session(tmp_path)
        (session / "raw/tx-a/work").mkdir()
        (session / "raw/tx-a/work/notes.txt").write_bytes(b"field notes")
        (session / "raw/output").mkdir()
        (session / "raw/output/take2.wav").write_bytes(b"a second take")

        roots = raw_roots(a_config())
        before = snapshot(session, roots)
        assert "raw/tx-a/work/notes.txt" in before
        assert "raw/output/take2.wav" in before

        (session / "raw/tx-a/work/notes.txt").write_bytes(b"tampered")
        with pytest.raises(DiscoveryError, match="modified"):
            verify_unchanged(session, roots, before)

    def test_generated_directories_are_excluded(self, tmp_path: Path) -> None:
        """`work/` and `output/` are inside the root when a track sits in the session
        directory, and they are the two places a run is *supposed* to write."""
        session = a_session(tmp_path, input_template="{track}")
        roots = raw_roots(a_config("{track}"))
        before = snapshot(session, roots)

        (session / "work").mkdir()
        (session / "work/manifest.json").write_text("{}")
        (session / "output").mkdir()
        (session / "output/ingest-report.json").write_text("{}")
        verify_unchanged(session, roots, before)


class TestOutputsInsideRaw:
    def test_ordinary_outputs_are_allowed(self, tmp_path: Path) -> None:
        session = a_session(tmp_path)
        config = a_config()
        reject_outputs_inside_raw(
            session,
            config,
            raw_roots(config),
            ingest_outputs(session),
        )

    def test_a_symlinked_output_is_caught(self, tmp_path: Path) -> None:
        """The defect a lexical comparison lets through.

        `output/ingest-report.json` does not *look* like it is inside `raw/`. The snapshot
        cannot catch it either, because outputs are written after the snapshot is verified.
        """
        session = a_session(tmp_path)
        (session / "output").symlink_to(session / "raw" / "tx-a")
        config = a_config()
        with pytest.raises(DiscoveryError, match="If a symlink put it there") as caught:
            reject_outputs_inside_raw(
                session,
                config,
                raw_roots(config),
                ingest_outputs(session),
            )
        assert caught.value.code == "output_inside_raw"

    def test_a_symlinked_work_directory_is_caught(self, tmp_path: Path) -> None:
        """`ingest` writes far more under `work/` than `inspect` does.

        Which of the outputs trips first is not asserted — they are checked in sorted
        order, so pinning one would be pinning an alphabetical accident. What matters is
        that a redirected `work/` cannot get past this at all.
        """
        session = a_session(tmp_path)
        (session / "work").symlink_to(session / "raw" / "tx-b")
        config = a_config()
        with pytest.raises(DiscoveryError, match="inside the source directory"):
            reject_outputs_inside_raw(
                session,
                config,
                raw_roots(config),
                ingest_outputs(session),
            )

    def test_each_stage_declares_its_own_outputs(self, tmp_path: Path) -> None:
        """The parameter exists so an undeclared output is a visible omission.

        `ingest` writes everything `inspect` does plus the timeline and the working-audio
        cache; if the two sets ever diverge the wrong way, this is what says so.
        """
        session = a_session(tmp_path)
        inspect_set = set(inspect_outputs(session).values())
        ingest_set = set(ingest_outputs(session).values())
        assert inspect_set < ingest_set
        assert session / "work/timeline.json" in ingest_set
        assert session / "work/cache/audio" in ingest_set

    def test_the_working_audio_cache_is_protected(self, tmp_path: Path) -> None:
        """A derivative written into a source directory is the same violation as a report."""
        session = a_session(tmp_path)
        cache = session / "work/cache"
        cache.mkdir(parents=True)
        (cache / "audio").symlink_to(session / "raw" / "tx-a")
        config = a_config()
        with pytest.raises(DiscoveryError, match="working-audio cache"):
            reject_outputs_inside_raw(
                session,
                config,
                raw_roots(config),
                ingest_outputs(session),
            )


class TestCleanupNeverWritesIntoRaw:
    """INV-01 outranks the stale-artifact cleanup, not only the report.

    Every runner deletes the artifacts a failed run may have left behind, so that a stale
    file cannot sit beside a report calling its stage failed. When `work` resolves inside a
    source directory, **those unlinks are the violation** — the run that correctly detected
    it commits it on the way out, which is worse than not detecting it at all.

    Driven through all three composed entry points from one place on purpose. M4's verify
    phase found this in `transcribe`, and it was in `activity` and `ingest` too, inherited
    unchanged from M2: a test naming only the runner that milestone happened to add would
    have found one of three (the lesson INV-08 already records about caches).
    """

    @staticmethod
    def _rig(session_dir: Path) -> Path:
        """`work -> raw/tx-a`, with a source file the cleanup would delete."""
        if (session_dir / "work").exists():
            shutil.rmtree(session_dir / "work")
        (session_dir / "work").symlink_to(session_dir / "raw" / "tx-a")
        victim = session_dir / "raw" / "tx-a" / "timeline.json"
        victim.write_text("a real file that lives in a source directory", encoding="utf-8")
        return victim

    @pytest.mark.parametrize("command", COMPOSED)
    def test_a_failed_run_deletes_nothing_under_raw(
        self, canonical_fixture: FixtureTruth, command: Any
    ) -> None:
        session_dir = canonical_fixture.session_dir
        victim = self._rig(session_dir)
        before = victim.read_bytes()

        result = command(session_dir)

        assert result.exit_code is ExitCode.FATAL
        assert result.report_written is False
        assert victim.exists(), "the failure cleanup deleted a file under raw/ (INV-01)"
        assert victim.read_bytes() == before


class TestEveryComposedRunVerifiesItsSources:
    """INV-01's two other halves, over every composed command rather than one each.

    The invariant is enforced by *three* mechanisms — refuse outputs inside `raw/`, hash every
    source before and after, and prove the check can fail — and until now only the first was
    parametrized. The other two were tested once per milestone, in the file belonging to
    whichever runner that milestone happened to add, which is exactly the shape that let one
    cleanup bug live in all three runners at once.
    """

    @pytest.mark.parametrize("command", COMPOSED)
    def test_a_complete_run_leaves_every_source_byte_identical(
        self, canonical_fixture: FixtureTruth, command: Any
    ) -> None:
        """Acceptance criterion 10, for each branch of the DAG separately."""
        session_dir = canonical_fixture.session_dir
        config = SessionConfig.model_validate(
            __import__("yaml").safe_load((session_dir / "session.yaml").read_text())
        )
        roots = raw_roots(config)
        before = snapshot(session_dir, roots)
        assert before, "the snapshot is empty, so comparing it proves nothing"

        result = command(session_dir)

        assert result.exit_code is ExitCode.OK, [
            f"{e.code}: {e.message}" for s in result.report.stages for e in s.errors
        ]
        assert snapshot(session_dir, roots) == before

    @pytest.mark.parametrize("command", COMPOSED)
    def test_a_source_corrupted_mid_run_fails_the_run(
        self, canonical_fixture: FixtureTruth, monkeypatch: pytest.MonkeyPatch, command: Any
    ) -> None:
        """The check has to be able to fail, or it proves nothing.

        A source is corrupted from inside the run — after the snapshot and before the
        verification — which is the only window the invariant is about. The seam is a function
        every composed runner reaches through `build_timeline`, so one patch drives all of
        them and none of them can be quietly exempt.
        """
        from dnd_audio.artifacts.manifest import Manifest
        from dnd_audio.timeline import layout

        victim = canonical_fixture.session_dir / canonical_fixture.chunks[0].relative_path
        original = layout.reject_unusable_sources

        def corrupting(manifest: Manifest) -> None:
            original(manifest)
            with victim.open("r+b") as handle:
                handle.seek(0, 2)
                handle.write(b"\x00" * 16)

        monkeypatch.setattr("dnd_audio.timeline.runner.reject_unusable_sources", corrupting)

        result = command(canonical_fixture.session_dir)

        assert result.exit_code is not ExitCode.OK
        codes = {e.code for s in result.report.stages for e in s.errors}
        assert "raw_sources_modified" in codes
