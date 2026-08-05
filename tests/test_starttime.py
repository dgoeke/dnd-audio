"""The strategy chain finds evidence, records what declined, or fails saying why.

Two properties are load-bearing here and both are invariants. Evidence keeps its own
units and origin (ADR-0006, INV-04) — a signed 48 kHz offset from session zero is not a
sample count from a recorder's own epoch, and nothing in this module pretends otherwise.
And a source with no reliable timing is fatal (INV-12), including when a filename and a
modification time both look perfectly plausible.
"""

from __future__ import annotations

import datetime as dt
import os
from fractions import Fraction
from pathlib import Path

import pytest

from dnd_audio.config import SourceTimeOverride
from dnd_audio.errors import RecoveryError, TimecodeError
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.inspection.probe import format_tags, parse_probe, run_ffprobe
from dnd_audio.inspection.starttime import (
    BwfSampleReference,
    SessionOffset,
    SourceContext,
    TimecodeReference,
    extract_start_time,
    strategy_names,
)
from dnd_audio.timecode import parse_frame_rate

RATE_30 = parse_frame_rate("30F")


def context(**overrides: object) -> SourceContext:
    """A source with no metadata at all, so each test adds only what it is about."""
    base: dict[str, object] = {
        "relative_path": "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav",
        "sha256": "a" * 64,
        "sample_rate": 48000,
        "tags": {},
        "frame_rate": RATE_30,
        "override": None,
    }
    base.update(overrides)
    return SourceContext(**base)  # type: ignore[arg-type]


class TestStrategyOrder:
    def test_the_chain_is_what_the_charter_says(self) -> None:
        assert strategy_names() == (
            "recovery_override_offset",
            "recovery_override_timecode",
            "bwf_time_reference",
            "timecode_tag",
        )

    def test_an_override_outranks_the_files_own_metadata(self) -> None:
        """An override exists for the case where the file's metadata is wrong, so a
        chain that preferred the file would make the escape hatch unreachable."""
        found = extract_start_time(
            context(
                tags={"time_reference": "3283200000"},
                override=SourceTimeOverride(
                    start_offset_samples=-4800, reason="clap-measured, bext was damaged"
                ),
            )
        )
        assert found.strategy == "recovery_override_offset"
        assert found.evidence == SessionOffset(samples=-4800)

    def test_a_sample_reference_outranks_a_timecode_tag(self) -> None:
        """It is finer than a frame and needs no configured rate to interpret."""
        found = extract_start_time(
            context(tags={"time_reference": "3283200000", "timecode": "19:00:00:00"})
        )
        assert found.strategy == "bwf_time_reference"

    def test_what_declined_and_why_is_recorded(self) -> None:
        """Real capture checks get cheaper the more this says: settling OQ-001 becomes reading the
        recorded reasons rather than re-running an investigation."""
        found = extract_start_time(context(tags={"timecode": "19:00:00:00"}))
        assert found.strategy == "timecode_tag"
        assert [item.name for item in found.declined] == [
            "recovery_override_offset",
            "recovery_override_timecode",
            "bwf_time_reference",
        ]
        assert "no time_reference tag" in found.declined[-1].reason


class TestEvidenceKeepsItsUnits:
    def test_a_bwf_reference_stays_an_integer_at_the_files_rate(self) -> None:
        found = extract_start_time(
            context(tags={"time_reference": "3283200000", "date": "2026-08-15"}, sample_rate=48000)
        )
        assert found.evidence == BwfSampleReference(
            samples=3283200000, sample_rate=48000, origination_date=dt.date(2026, 8, 15)
        )

    def test_a_files_own_rate_is_carried_not_assumed(self) -> None:
        """A 44.1 kHz source's reference counts 44100ths of a second, not 48000ths.
        Recording the session rate here would misplace it by 8.75%."""
        found = extract_start_time(
            context(tags={"time_reference": "3018150000"}, sample_rate=44100)
        )
        assert isinstance(found.evidence, BwfSampleReference)
        assert found.evidence.sample_rate == 44100

    def test_a_timecode_becomes_an_exact_frame_index(self) -> None:
        found = extract_start_time(context(tags={"timecode": "19:00:03:15"}))
        assert found.evidence == TimecodeReference(
            text="19:00:03:15",
            frames=19 * 3600 * 30 + 3 * 30 + 15,
            frame_rate_label="30F",
            frame_rate=Fraction(30),
            drop_frame=False,
        )

    def test_a_fractional_rate_stays_rational(self) -> None:
        """INV-04: 30000/1001 never becomes 29.97, and the frame index never becomes
        a sample position — 8008/5 samples per frame is M2's rounding to define."""
        found = extract_start_time(
            context(tags={"timecode": "01:00:00:00"}, frame_rate=parse_frame_rate("29.97F"))
        )
        assert isinstance(found.evidence, TimecodeReference)
        assert found.evidence.frame_rate == Fraction(30000, 1001)
        assert isinstance(found.evidence.frames, int)

    def test_a_session_offset_is_signed_and_at_48k(self) -> None:
        """The distinction ADR-0006 exists for. Collapsing this into samples-since-
        midnight needs session zero, which M1 does not have and must not invent."""
        found = extract_start_time(
            context(override=SourceTimeOverride(start_offset_samples=-96000, reason="field log"))
        )
        assert found.evidence == SessionOffset(samples=-96000, sample_rate=48000)

    def test_the_three_evidence_kinds_are_distinct_types(self) -> None:
        """So a consumer must handle each rather than reading one number that means
        three different things depending on where it came from."""
        kinds = {
            type(extract_start_time(context(tags={"time_reference": "1"})).evidence),
            type(extract_start_time(context(tags={"timecode": "19:00:00:00"})).evidence),
            type(
                extract_start_time(
                    context(override=SourceTimeOverride(start_offset_samples=1, reason="r"))
                ).evidence
            ),
        }
        assert kinds == {BwfSampleReference, TimecodeReference, SessionOffset}


class TestAssumptionsAreTagged:
    def test_every_file_metadata_strategy_names_its_open_question(self) -> None:
        """`rg OQ-004` has to find every place that changed when the real evidence answered it."""
        bwf = extract_start_time(context(tags={"time_reference": "1"}))
        assert any("OQ-004" in note for note in bwf.assumptions)
        assert any("OQ-001" in note for note in bwf.assumptions)

        # And it states what OQ-004 actually measured, not what EBU Tech 3285 says the
        # field means. Every manifest carries this sentence; a stale one would tell the
        # next reader the reference is wall clock (ADR-0031).
        stamped = " ".join(bwf.assumptions)
        assert "which is not midnight" in stamped
        assert "since midnight" not in stamped

        tagged = extract_start_time(context(tags={"timecode": "19:00:00:00"}))
        assert any("OQ-001" in note for note in tagged.assumptions)

    def test_an_override_records_the_operators_reason(self) -> None:
        """The spec requires overrides to be recorded prominently: the manifest has to
        be able to say why a time was not read from the file."""
        found = extract_start_time(
            context(
                override=SourceTimeOverride(
                    start_timecode="19:00:00:00",
                    reason="bext was damaged; value from the field log",
                )
            )
        )
        assert found.override_reason == "bext was damaged; value from the field log"

    def test_a_strategy_reading_the_file_claims_no_operator_reason(self) -> None:
        assert extract_start_time(context(tags={"time_reference": "1"})).override_reason is None


class TestNoTimingIsFatal:
    def test_a_source_with_no_evidence_raises(self) -> None:
        with pytest.raises(TimecodeError, match="no reliable start time"):
            extract_start_time(context())

    def test_the_message_names_every_strategy_and_the_fix(self) -> None:
        """An unactionable failure at this point costs an operator their session."""
        with pytest.raises(TimecodeError) as raised:
            extract_start_time(context())
        message = str(raised.value)
        for name in strategy_names():
            assert name in message
        assert "source_time_overrides" in message
        assert "start_timecode" in message
        assert "raw/tx-a/TX01_MIC001_20260815_190000_orig.wav" in message

    def test_an_unparseable_reference_declines_rather_than_crashing(self) -> None:
        with pytest.raises(TimecodeError) as raised:
            extract_start_time(context(tags={"time_reference": "not-a-number"}))
        assert "is not an integer" in str(raised.value)

    def test_a_timecode_in_the_wrong_rate_declines_with_its_reason(self) -> None:
        with pytest.raises(TimecodeError) as raised:
            extract_start_time(context(tags={"timecode": "19:00:00;15"}))
        assert "drop-frame notation" in str(raised.value)

    def test_a_date_alone_is_not_a_time(self) -> None:
        """A `bext` origination date with no reference says which day, not when."""
        with pytest.raises(TimecodeError):
            extract_start_time(context(tags={"date": "2026-08-15"}))

    def test_a_recording_date_only_override_supplies_no_timing(self) -> None:
        """It supplies the calendar day and lets a later strategy supply the time; on
        its own it must not rescue a file with no timing at all."""
        with pytest.raises(TimecodeError):
            extract_start_time(
                context(
                    override=SourceTimeOverride(
                        recording_date=dt.date(2026, 8, 15), reason="from the field log"
                    )
                )
            )


class TestOverrides:
    def test_a_matching_hash_lets_the_override_apply(self) -> None:
        found = extract_start_time(
            context(
                sha256="b" * 64,
                override=SourceTimeOverride(
                    sha256="b" * 64, start_offset_samples=1200, reason="measured"
                ),
            )
        )
        assert found.evidence == SessionOffset(samples=1200)

    def test_a_mismatched_hash_is_fatal(self) -> None:
        """Applying it anyway attaches a field-log time to the wrong recording, which
        is worse than the missing metadata the override was written for."""
        with pytest.raises(RecoveryError, match="expects sha256"):
            extract_start_time(
                context(
                    sha256="b" * 64,
                    tags={"time_reference": "1"},
                    override=SourceTimeOverride(
                        sha256="c" * 64, start_offset_samples=1200, reason="measured"
                    ),
                )
            )

    def test_the_hash_is_checked_before_any_strategy_runs(self) -> None:
        """Otherwise a mismatched override on a file with good metadata would pass
        silently, which is the case where nobody would ever notice."""
        with pytest.raises(RecoveryError):
            extract_start_time(
                context(
                    sha256="b" * 64,
                    tags={"time_reference": "3283200000"},
                    override=SourceTimeOverride(
                        sha256="c" * 64,
                        recording_date=dt.date(2026, 8, 15),
                        reason="the day, from the field log",
                    ),
                )
            )

    def test_an_override_with_no_hash_still_applies(self) -> None:
        """The spec makes the hash optional; requiring it would break the recovery
        path for exactly the damaged files it exists to rescue."""
        found = extract_start_time(
            context(override=SourceTimeOverride(start_timecode="19:00:00:00", reason="log"))
        )
        assert found.strategy == "recovery_override_timecode"


class TestTimingIsNeverInvented:
    def test_the_context_offers_no_filename_or_stat(self) -> None:
        """INV-12 structurally: a strategy cannot reach for a modification time because
        there is nowhere in its input to find one."""
        fields = set(SourceContext.__slots__)
        assert "mtime" not in fields
        assert "modified" not in fields
        assert "path" not in fields
        assert fields == {
            "relative_path",
            "sha256",
            "sample_rate",
            "tags",
            "frame_rate",
            "override",
        }

    def test_a_plausible_filename_and_mtime_do_not_rescue_a_bare_file(self, tmp_path: Path) -> None:
        """The behavioural half. A file named as if it started at 19:00:00, whose
        modification time agrees, and which carries no embedded timing, is still fatal.
        """
        source = tmp_path / "TX01_MIC001_20260815_190000_orig.wav"
        source.write_bytes(b"")
        stamp = dt.datetime(2026, 8, 15, 19, 0, 0, tzinfo=dt.UTC).timestamp()
        os.utime(source, (stamp, stamp))

        with pytest.raises(TimecodeError):
            extract_start_time(context(relative_path=source.name, tags={}))

    def test_touching_a_file_does_not_change_the_evidence(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The other behavioural half: mtime cannot influence a decision it is not an
        input to. Proved by changing it and seeing nothing move."""
        chunk = canonical_fixture.for_track("tx-a")[0]
        path = canonical_fixture.session_dir / chunk.relative_path
        tags = format_tags(
            parse_probe(run_ffprobe(canonical_fixture.session_dir, chunk.relative_path).raw)
        )
        before = extract_start_time(context(tags=tags, sha256=chunk.sha256))

        os.utime(path, (0, 0))
        after = extract_start_time(context(tags=tags, sha256=chunk.sha256))
        assert before == after


class TestAgainstTheRealFixture:
    def test_five_tracks_resolve_through_the_bwf_reference(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        for chunk in canonical_fixture.chunks:
            if chunk.track_id == "tx-f":
                continue
            tags = format_tags(
                parse_probe(run_ffprobe(canonical_fixture.session_dir, chunk.relative_path).raw)
            )
            found = extract_start_time(context(tags=tags, sha256=chunk.sha256))
            assert found.strategy == "bwf_time_reference"
            assert isinstance(found.evidence, BwfSampleReference)
            assert found.evidence.samples == chunk.time_reference

    def test_the_sixth_falls_through_to_the_timecode_tag(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Both branches of the chain are exercised by the canonical fixture, so
        neither can rot unnoticed."""
        chunk = canonical_fixture.for_track("tx-f")[0]
        tags = format_tags(
            parse_probe(run_ffprobe(canonical_fixture.session_dir, chunk.relative_path).raw)
        )
        found = extract_start_time(context(tags=tags, sha256=chunk.sha256))

        assert found.strategy == "timecode_tag"
        assert isinstance(found.evidence, TimecodeReference)
        # 19:00:03:15 at 30 fps, and the same instant the bext reference would encode.
        assert found.evidence.frames * 48000 // 30 == chunk.time_reference
