"""INV-02, INV-04, INV-07: the helpers every deterministic artifact is built on."""

from __future__ import annotations

import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from dnd_audio.determinism import (
    canonical_json,
    public_seconds,
    sha256_bytes,
    sha256_file,
    sha256_stream,
    to_milliseconds,
    write_atomic,
    write_json_atomic,
)


class TestCanonicalJson:
    def test_key_order_does_not_depend_on_insertion_order(self) -> None:
        """INV-02: two dicts with the same contents serialize to the same bytes."""
        first = canonical_json({"b": 1, "a": 2, "c": {"z": 1, "y": 2}})
        second = canonical_json({"c": {"y": 2, "z": 1}, "a": 2, "b": 1})
        assert first == second

    def test_ends_with_a_newline(self) -> None:
        assert canonical_json({"a": 1}).endswith("}\n")

    def test_unicode_is_preserved_not_escaped(self) -> None:
        # A curly apostrophe is exactly what ASR returns and what a transcript must
        # round-trip; ruff's ambiguous-character rule is about source identifiers.
        curly = "Zephyrine’s"
        assert curly in canonical_json({"text": curly})
        assert "\\u2019" not in canonical_json({"text": curly})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_are_rejected(self, value: float) -> None:
        """`NaN` and `Infinity` are not JSON. Emitting them produces an unparseable file."""
        with pytest.raises(ValueError, match="Out of range"):
            canonical_json({"value": value})

    def test_output_is_parseable(self) -> None:
        payload = {"a": [1, 2, {"b": "c"}], "d": None, "e": True}
        assert json.loads(canonical_json(payload)) == payload


class TestWriteAtomic:
    def test_repeated_writes_are_byte_identical(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.json"
        payload: dict[str, Any] = {"b": [3, 1, 2], "a": "x"}

        write_json_atomic(target, payload)
        first = target.read_bytes()
        write_json_atomic(target, payload)

        assert target.read_bytes() == first

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "work" / "nested" / "artifact.json"
        write_json_atomic(target, {"a": 1})
        assert target.is_file()

    def test_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.json"
        write_atomic(target, "content")
        assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path: Path) -> None:
        """INV-13: the report is written even on partial failure — but never half-written."""
        target = tmp_path / "artifact.json"
        write_atomic(target, "original")

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            write_json_atomic(target, {"bad": Unserializable()})

        assert target.read_text(encoding="utf-8") == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]

    def test_accepts_bytes_and_text_alike(self, tmp_path: Path) -> None:
        text_target = tmp_path / "text"
        bytes_target = tmp_path / "bytes"
        write_atomic(text_target, "samé")
        write_atomic(bytes_target, "samé".encode())
        assert text_target.read_bytes() == bytes_target.read_bytes()


class _RecordingStream:
    """A stream that remembers how much each read asked for."""

    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)
        self.requested: list[int] = []

    def read(self, size: int = -1, /) -> bytes:
        self.requested.append(size)
        return self._buffer.read(size)


class TestHashing:
    def test_matches_a_known_digest(self) -> None:
        assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()

    def test_file_hash_matches_the_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / "source.bin"
        payload = b"\x00\x01\x02" * 5000
        target.write_bytes(payload)
        assert sha256_file(target) == hashlib.sha256(payload).hexdigest()

    def test_reads_are_bounded(self) -> None:
        """INV-07: from M1 this is pointed at multi-gigabyte recordings.

        An implementation based on `read()` or `Path.read_bytes()` would produce the
        right digest and still load the whole file, so the digest alone proves nothing.
        """
        payload = b"x" * (4 * 1024 + 7)
        stream = _RecordingStream(payload)

        digest = sha256_stream(stream, chunk_bytes=1024)

        assert digest == hashlib.sha256(payload).hexdigest()
        assert stream.requested, "the stream was never read"
        assert max(stream.requested) == 1024
        assert len(stream.requested) >= 5

    def test_rejects_an_unbounded_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            sha256_stream(io.BytesIO(b"x"), chunk_bytes=-1)


class TestMillisecondQuantization:
    """INV-04: exact rationals in, an explicit tie rule out."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (Fraction(0), 0),
            (Fraction(1), 1000),
            (Fraction(1, 1000), 1),
            # Exactly half a millisecond, both signs: away from zero, not to even.
            (Fraction(1, 2000), 1),
            (Fraction(3, 2000), 2),
            (Fraction(-1, 2000), -1),
            (Fraction(-3, 2000), -2),
            # Just below a half stays down.
            (Fraction(4999, 10_000_000), 0),
            # A 48 kHz sample count that does not land on a millisecond.
            (Fraction(4821440 * 48, 48_000), 4821440),
        ],
    )
    def test_tie_rule_is_half_away_from_zero(self, seconds: Fraction, expected: int) -> None:
        assert to_milliseconds(seconds) == expected

    def test_differs_from_bankers_rounding(self) -> None:
        """The reason the rule is stated rather than inherited.

        `round()` would send 0.5 ms to 0 and 1.5 ms to 2 — a tie rule that depends on
        the neighbouring integer's parity is not something an artifact should encode.
        """
        assert to_milliseconds(Fraction(1, 2000)) == 1
        assert round(0.5) == 0

    def test_public_seconds_round_trips_through_json(self) -> None:
        """The serialized float must read back as the same value (INV-02)."""
        exact = Fraction(4821440, 1000)
        value = public_seconds(exact)
        assert value == 4821.44
        assert json.loads(canonical_json({"start_s": value}))["start_s"] == value

    def test_public_seconds_is_exactly_the_quantizer(self) -> None:
        for numerator in range(0, 5000, 37):
            seconds = Fraction(numerator, 48_000)
            assert public_seconds(seconds) == to_milliseconds(seconds) / 1000

    def test_accepts_whole_seconds_as_int(self) -> None:
        assert to_milliseconds(7) == 7000
