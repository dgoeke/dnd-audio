"""`marker build`, driven through the command an operator actually types.

M7a's closeout is blunt about why: nine test files and complete runner coverage did not
compensate for the fact that *nothing ran a command*, and a P0 that overwrote a source
recording sat in exactly that gap. So the INV-01 tests here go through `CliRunner`, and the
destination they aim at is a real recording inside a real session.

The other thing being guarded is publication order. "Manifest last" is a completeness marker
for a first build and **not** for a rebuild: interrupted between replacing the WAV and
replacing the page, it would leave the previous manifest describing bytes that are gone, and
the set would look complete. Removing it first makes every interrupted state detectable.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnd_audio.cli import app
from dnd_audio.determinism import sha256_bytes, write_atomic, write_json_atomic
from dnd_audio.errors import ExitCode
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.riff import read_inventory
from dnd_audio.marker import MARKER_MANIFEST_FILENAME, artifact_stem
from dnd_audio.marker.builder import build_marker
from dnd_audio.marker.manifest import MarkerManifest
from dnd_audio.marker.page import payload_from_html
from dnd_audio.marker.spec import MARKER_SPECS, MarkerSpec
from dnd_audio.timeline.pcm import open_pcm

ALL_SPECS = pytest.mark.parametrize("spec", MARKER_SPECS.values(), ids=list(MARKER_SPECS))

runner = CliRunner()


def test_no_generated_marker_artifact_is_tracked() -> None:
    """The generator is the source; WAV/HTML/manifests are operator-local outputs."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert not [
        path
        for path in tracked
        if path.startswith("bench-markers/")
        or (path.endswith(".wav") and "dnd-audio-sync-marker-" in path)
        or (path.endswith(".html") and "dnd-audio-sync-marker-" in path)
    ]


def build(spec: MarkerSpec, destination: Path) -> MarkerManifest:
    return build_marker(spec, destination).manifest


class TestTheArtifacts:
    """What lands on disk, and whether it is what the manifest says."""

    @ALL_SPECS
    def test_three_files_appear_under_the_marker_name(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        build(spec, tmp_path)
        stem = artifact_stem(spec.name)
        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
            [f"{stem}.wav", f"{stem}.html", MARKER_MANIFEST_FILENAME]
        )

    @ALL_SPECS
    def test_the_manifest_digests_match_the_files_on_disk(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        """The claim an operator relies on when copying a file to a phone."""
        manifest = build(spec, tmp_path)
        for artifact in (manifest.wav, manifest.page):
            payload = (tmp_path / artifact.filename).read_bytes()
            assert sha256_bytes(payload) == artifact.sha256
            assert len(payload) == artifact.size_bytes

    @ALL_SPECS
    def test_the_page_and_the_wav_hold_the_same_bytes(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        manifest = build(spec, tmp_path)
        wav = (tmp_path / manifest.wav.filename).read_bytes()
        page = (tmp_path / manifest.page.filename).read_text(encoding="utf-8")
        assert payload_from_html(page) == wav

    @ALL_SPECS
    def test_the_wav_reads_back_through_this_projects_own_parsers(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        """Three independent implementations agreeing beats sharing a table with the writer."""
        manifest = build(spec, tmp_path)
        path = tmp_path / manifest.wav.filename

        inventory = read_inventory(path)
        assert [chunk.chunk_id for chunk in inventory.chunks] == ["fmt ", "data"]

        source = open_pcm(path)
        assert source.n_samples == spec.total_samples
        assert source.sample_rate == manifest.sample_rate
        assert source.sample_format.codec_name == manifest.sample_format

    @ALL_SPECS
    def test_the_manifest_validates_against_its_checked_in_schema(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        import jsonschema

        build(spec, tmp_path)
        document = json.loads((tmp_path / MARKER_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        schema = json.loads(Path("schemas/marker-manifest.schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(document, schema)

    @ALL_SPECS
    def test_the_manifest_does_not_hash_itself(self, spec: MarkerSpec, tmp_path: Path) -> None:
        """ADR-0003's fixed point: writing the hash would change the bytes it describes."""
        build(spec, tmp_path)
        document = json.loads((tmp_path / MARKER_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        named = {document["wav"]["filename"], document["page"]["filename"]}
        assert MARKER_MANIFEST_FILENAME not in named

    @ALL_SPECS
    def test_the_manifest_names_no_machine(self, spec: MarkerSpec, tmp_path: Path) -> None:
        """Deterministic means comparable: no clock, no host, no absolute path."""
        build(spec, tmp_path)
        text = (tmp_path / MARKER_MANIFEST_FILENAME).read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "/" not in json.loads(text)["wav"]["filename"]


class TestDeterminism:
    """INV-02 over the published set."""

    @ALL_SPECS
    def test_two_builds_into_different_directories_are_byte_identical(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        build(spec, first)
        build(spec, second)
        for name in sorted(path.name for path in first.iterdir()):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    @ALL_SPECS
    def test_rebuilding_in_place_reproduces_the_same_bytes(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        build(spec, tmp_path)
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        build(spec, tmp_path)
        after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        assert before == after

    def test_two_candidates_do_not_collide(self, tmp_path: Path) -> None:
        """They share a directory at the bench, so the filenames must differ."""
        names = set()
        for spec in MARKER_SPECS.values():
            manifest = build(spec, tmp_path)
            names.update({manifest.wav.filename, manifest.page.filename})
        assert len(names) == 2 * len(MARKER_SPECS)


class TestPublicationOrder:
    """The manifest is the completeness marker, so an interrupted set must not have one."""

    @ALL_SPECS
    def test_a_stale_manifest_is_removed_before_anything_is_written(
        self, spec: MarkerSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interrupt the rebuild after the manifest is gone and before the WAV lands.

        Without the unlink-first ordering this leaves the *previous* manifest beside a
        half-replaced pair — a set that looks complete and is not.
        """
        build(spec, tmp_path)
        manifest_path = tmp_path / MARKER_MANIFEST_FILENAME
        assert manifest_path.exists()

        def explode(*_args: object, **_kwargs: object) -> None:
            message = "interrupted"
            raise OSError(message)

        monkeypatch.setattr("dnd_audio.marker.builder.write_atomic", explode)
        with pytest.raises(OSError, match="interrupted"):
            build(spec, tmp_path)

        assert not manifest_path.exists(), (
            "a manifest survived an interrupted rebuild, so an incomplete set would look complete"
        )

    @ALL_SPECS
    def test_the_manifest_is_written_after_both_artifacts(
        self, spec: MarkerSpec, tmp_path: Path
    ) -> None:
        """Recorded as an ordering, not inferred from timestamps."""
        written: list[str] = []
        stem = artifact_stem(spec.name)

        import dnd_audio.marker.builder as builder

        def note_atomic(path: Path, data: str | bytes) -> None:
            written.append(path.name)
            write_atomic(path, data)

        def note_json(path: Path, value: object) -> None:
            written.append(path.name)
            write_json_atomic(path, value)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "write_atomic", note_atomic)
            patch.setattr(builder, "write_json_atomic", note_json)
            build(spec, tmp_path)

        assert written == [f"{stem}.wav", f"{stem}.html", MARKER_MANIFEST_FILENAME]


class TestTheCommand:
    """Through the CLI, because the wiring is what carries the guards."""

    def test_building_without_a_marker_produces_frozen_v1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["marker", "build", str(tmp_path)])
        assert result.exit_code == 0, result.output
        manifest = json.loads((tmp_path / MARKER_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["marker_name"] == "v1"
        assert manifest["wav"]["sha256"] == (
            "70355baad6bb72b38e0b606cddbbaa3428c11429bec74cd127aa6f8935ecdf6f"
        )

    def test_building_a_candidate_succeeds_and_reports_the_digest(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["marker", "build", str(tmp_path), "--marker", "cand-a"])
        assert result.exit_code == 0, result.output
        manifest = json.loads((tmp_path / MARKER_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["wav"]["sha256"] in result.output

    def test_an_unknown_marker_lists_what_is_available(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["marker", "build", str(tmp_path), "--marker", "nope"])
        assert result.exit_code == ExitCode.FATAL
        assert "cand-a" in result.output

    def test_the_candidate_option_is_not_advertised(self) -> None:
        """The charter's non-goal: no public candidate-management interface (ADR-0041)."""
        result = runner.invoke(app, ["marker", "build", "--help"])
        assert result.exit_code == 0
        assert "--marker" not in result.output

    def test_a_missing_directory_is_created(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "deeper"
        result = runner.invoke(app, ["marker", "build", str(destination), "--marker", "cand-c"])
        assert result.exit_code == 0, result.output
        assert (destination / MARKER_MANIFEST_FILENAME).exists()


class TestTheDestinationIsGuarded:
    """INV-01. `marker build` has no session argument, so it is outside the COMPOSED matrix."""

    def test_a_destination_inside_a_sessions_sources_is_refused(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The P0 shape M7a's second review found, in a new command with the same signature."""
        target = canonical_fixture.session_dir / "raw" / "tx-a"
        before = {path: path.read_bytes() for path in sorted(target.iterdir()) if path.is_file()}
        assert before, "the fixture must have a recording to protect"

        result = runner.invoke(app, ["marker", "build", str(target), "--marker", "cand-a"])

        assert result.exit_code == ExitCode.FATAL
        assert "INV-01" in result.output
        after = {path: path.read_bytes() for path in sorted(target.iterdir()) if path.is_file()}
        assert after == before, "a source file changed"
        assert not (target / MARKER_MANIFEST_FILENAME).exists()

    def test_a_symlink_into_the_sources_is_refused_too(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Lexical comparison is not a boundary; resolution is (M1's verify phase)."""
        session = canonical_fixture.session_dir
        link = session / "elsewhere"
        link.symlink_to(session / "raw" / "tx-b")
        before = sorted(
            (path.name, path.read_bytes())
            for path in (session / "raw" / "tx-b").iterdir()
            if path.is_file()
        )

        result = runner.invoke(app, ["marker", "build", str(link), "--marker", "cand-a"])

        assert result.exit_code == ExitCode.FATAL
        after = sorted(
            (path.name, path.read_bytes())
            for path in (session / "raw" / "tx-b").iterdir()
            if path.is_file()
        )
        assert after == before

    def test_nothing_is_created_on_the_refused_path(self, canonical_fixture: FixtureTruth) -> None:
        """The refusal must precede `mkdir`, or the check has already half-run."""
        target = canonical_fixture.session_dir / "raw" / "tx-a" / "deeper" / "still-deeper"
        result = runner.invoke(app, ["marker", "build", str(target), "--marker", "cand-a"])
        assert result.exit_code == ExitCode.FATAL
        assert not target.exists()
        assert not target.parent.exists()

    def test_a_destination_beside_the_sources_is_allowed(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The contrast that keeps the refusals from being a blanket ban on session paths.

        Writing into `SESSION/markers/` is a perfectly reasonable thing to want, and a guard
        that refused it would be measuring "is this path inside a session" rather than "is
        this path inside a session's *sources*".
        """
        target = canonical_fixture.session_dir / "markers"
        result = runner.invoke(app, ["marker", "build", str(target), "--marker", "cand-a"])
        assert result.exit_code == 0, result.output
        assert (target / MARKER_MANIFEST_FILENAME).exists()
