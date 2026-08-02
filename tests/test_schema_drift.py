"""The checked-in schemas must match the models, and must not depend on hash order.

This is the rail that makes every later "output validates against its schema" claim
mean something. Without it, a model change would silently leave the committed schema
describing the previous shape, and the validation tests would keep passing against it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from dnd_audio.schema_export import (
    JSON_SCHEMA_DIALECT,
    SCHEMA_DIRNAME,
    schema_documents,
    write_schemas,
)


@pytest.fixture
def schema_dir(repo_root: Path) -> Path:
    return repo_root / SCHEMA_DIRNAME


def _assert_matches(directory: Path, filename: str) -> None:
    """The one comparison. Shared so the test below exercises the real thing."""
    expected = schema_documents()[filename]
    actual = (directory / filename).read_text(encoding="utf-8")
    assert actual == expected, (
        f"{filename} is stale. Run: uv run --no-sync python scripts/gen_schemas.py"
    )


def test_every_document_is_checked_in(schema_dir: Path) -> None:
    on_disk = {path.name for path in schema_dir.glob("*.schema.json")}
    assert on_disk == set(schema_documents())


@pytest.mark.parametrize("filename", sorted(schema_documents()))
def test_checked_in_bytes_match_the_models(schema_dir: Path, filename: str) -> None:
    """Fails when a model changed and `scripts/gen_schemas.py` was not re-run."""
    _assert_matches(schema_dir, filename)


def test_drift_is_actually_detected(tmp_path: Path) -> None:
    """The comparison above must be able to fail.

    Runs `_assert_matches` — the same function the real test uses — against a copy
    that has drifted. Asserting that a mutated dict differs from an unmutated one
    would prove nothing about whether the drift check notices.
    """
    write_schemas(tmp_path)
    _assert_matches(tmp_path, "manifest.schema.json")

    target = tmp_path / "manifest.schema.json"
    target.write_text(
        target.read_text(encoding="utf-8").replace('"session_id"', '"session_identifier"'),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="is stale"):
        _assert_matches(tmp_path, "manifest.schema.json")


def test_declares_its_dialect(schema_dir: Path) -> None:
    for path in sorted(schema_dir.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == JSON_SCHEMA_DIALECT


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_generation_does_not_depend_on_hash_order(repo_root: Path, hash_seed: str) -> None:
    """INV-02, from a fresh interpreter with a different `PYTHONHASHSEED`.

    Repeated generation inside one process shares that process's hash randomization, so
    it cannot detect a set or dict iteration order leaking into the output. A subprocess
    with a different seed can.
    """
    program = textwrap.dedent(
        """
        import hashlib, sys
        from dnd_audio.schema_export import schema_documents
        joined = "".join(f"{k}{v}" for k, v in sorted(schema_documents().items()))
        sys.stdout.write(hashlib.sha256(joined.encode("utf-8")).hexdigest())
        """
    )
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)

    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
        env=environment,
        timeout=120,
    )

    joined = "".join(f"{k}{v}" for k, v in sorted(schema_documents().items()))
    import hashlib

    assert completed.stdout == hashlib.sha256(joined.encode("utf-8")).hexdigest()


def test_regeneration_is_idempotent(tmp_path: Path) -> None:
    from dnd_audio.schema_export import write_schemas

    write_schemas(tmp_path)
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())}
    write_schemas(tmp_path)
    second = {path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())}

    assert first == second
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]
