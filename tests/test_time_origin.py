"""INV-14 — the mix and the transcript are on the same clock, from the same zero.

Every other timing test in this suite checks one artifact against the timeline. This one
checks two artifacts against *each other*, because that is the property an external consumer
depends on and the one nothing else would catch.

ADR-0044 publishes `session.mp3` as an attachment and the transcript as document text whose
every turn carries a `#t=<seconds>` link into that audio. So a constant offset introduced on
either side — a mix lead-in, a trimmed head, a transcript time rebased onto something other
than session zero — mislabels every published turn by the same amount while leaving both
artifacts internally consistent, every schema valid, and every existing test green. It would
be found by a person listening to a four-hour recording and noticing the words are early.

**The expected values are computed here, from the timeline, in plain arithmetic.** Not
through `public_seconds`, not through `presentation_turns`. A test that re-derived the
transcript's times with the transcript's own converter would agree with itself no matter what
either side did. The tolerance below is half a millisecond because INV-04 permits exactly one
difference between the sample domain and the public one — quantization to whole milliseconds
at the serialization boundary — and nothing else.

The mix's *positional* correspondence (that mix sample N carries session sample N, not merely
that the file is the right length) is proved in `test_mix_render.py`, which reads the tracks
independently at an interior offset. A lead-in fails there. This file does not repeat it.
"""

from __future__ import annotations

from pathlib import Path

from dnd_audio.artifacts.records import TranscriptRecords
from dnd_audio.artifacts.timeline import Timeline
from dnd_audio.artifacts.transcript import Transcript
from dnd_audio.fixtures import FixtureTruth
from dnd_audio.mix import MIX_CACHE_DIRNAME
from dnd_audio.orchestrate import run_process
from dnd_audio.timeline import TIMELINE_RELATIVE_PATH
from dnd_audio.timeline.pcm import PcmReader, open_pcm
from dnd_audio.transcript import RECORDS_RELATIVE_PATH, TRANSCRIPT_JSON_RELATIVE_PATH

#: INV-04's public boundary rounds to whole milliseconds, so a public time may differ from the
#: sample it denotes by at most half of one. Any larger disagreement is an offset, not rounding.
MILLISECOND = 0.0005


class TestTheMixAndTheTranscriptShareOneOrigin:
    def _artifacts(
        self, fixture: FixtureTruth
    ) -> tuple[Timeline, TranscriptRecords, Transcript, Path]:
        result = run_process(fixture.session_dir, fake_models=True)
        assert result.records is not None, "both branches must have run for this comparison"
        session_dir = fixture.session_dir

        timeline = Timeline.model_validate_json((session_dir / TIMELINE_RELATIVE_PATH).read_text())
        records = TranscriptRecords.model_validate_json(
            (session_dir / RECORDS_RELATIVE_PATH).read_text()
        )
        transcript = Transcript.model_validate_json(
            (session_dir / TRANSCRIPT_JSON_RELATIVE_PATH).read_text()
        )
        mixes = sorted((session_dir / MIX_CACHE_DIRNAME).glob("*.wav"))
        assert len(mixes) == 1, (
            f"expected exactly one committed mix intermediate, found {[m.name for m in mixes]}"
        )
        return timeline, records, transcript, mixes[0]

    def test_the_mix_spans_the_session_and_starts_at_its_zero(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The rendered audio is the whole session and nothing but the session.

        Length alone carries the origin here: the renderer writes one contiguous stream from
        session sample zero, so a file of exactly the session's length cannot also have been
        shifted without losing samples off the end.
        """
        timeline, _, _, mix = self._artifacts(canonical_fixture)

        with PcmReader(open_pcm(mix)) as reader:
            assert reader.source.n_samples == timeline.duration_samples
            assert reader.source.sample_rate == timeline.sample_rate

    def test_the_transcripts_duration_is_the_timelines_duration(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """Both artifacts describe the same span, in their own units."""
        timeline, _, transcript, _ = self._artifacts(canonical_fixture)

        expected = timeline.duration_samples / timeline.sample_rate
        assert abs(transcript.duration_s - expected) <= MILLISECOND

    def test_every_published_time_lands_on_the_sample_its_record_names(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """A published `start_s` is its record's `start_sample`, on the session grid.

        This is the assertion the wiki's timestamp links rest on. Each public turn is matched
        to the record boundary it was built from by searching the records for one that agrees
        within the millisecond quantization — so a systematic offset on either side leaves
        some turn with no match at all, rather than shifting both sides together.
        """
        timeline, records, transcript, _ = self._artifacts(canonical_fixture)
        rate = timeline.sample_rate
        assert records.sample_rate == rate, (
            "the records are counted on a different grid than the timeline, so nothing below "
            "compares what it claims to"
        )

        retained = records.retained()
        assert transcript.segments, "a transcript with no segments proves nothing here"
        starts = {record.start_sample / rate for record in retained}
        ends = {record.end_sample / rate for record in retained}

        for segment in transcript.segments:
            assert any(abs(segment.start_s - start) <= MILLISECOND for start in starts), (
                f"published segment {segment.segment_id} starts at {segment.start_s}s, which "
                f"is not within a millisecond of any record's start sample. Either the "
                f"transcript's times or the records' samples have moved off session zero."
            )
            assert any(abs(segment.end_s - end) <= MILLISECOND for end in ends), (
                f"published segment {segment.segment_id} ends at {segment.end_s}s, which is "
                f"not within a millisecond of any record's end sample."
            )

    def test_no_published_time_falls_outside_the_audio(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """A timestamp link past the end of the MP3 seeks nowhere and plays nothing."""
        _, _, transcript, mix = self._artifacts(canonical_fixture)

        with PcmReader(open_pcm(mix)) as reader:
            playable = reader.source.n_samples / reader.source.sample_rate

        for segment in transcript.segments:
            assert 0.0 <= segment.start_s <= playable + MILLISECOND, (
                f"published segment {segment.segment_id} starts at {segment.start_s}s in a "
                f"{playable}s recording"
            )
            assert segment.end_s <= playable + MILLISECOND
            for word in segment.words:
                assert 0.0 <= word.start_s <= playable + MILLISECOND

    def test_the_matching_above_would_reject_a_shifted_transcript(
        self, canonical_fixture: FixtureTruth
    ) -> None:
        """The bridge assertion has teeth, checked by breaking it deliberately.

        Matching each published time against *any* record boundary is only meaningful if some
        offsets fail to match. A fixture dense enough in boundaries, or a tolerance wide
        enough, would accept a shifted transcript and the guard would be decorative. So shift
        every published time by a second and require the same matching to reject it.
        """
        timeline, records, transcript, _ = self._artifacts(canonical_fixture)
        rate = timeline.sample_rate
        starts = {record.start_sample / rate for record in records.retained()}

        shifted = [segment.start_s + 1.0 for segment in transcript.segments]
        unmatched = [
            start
            for start in shifted
            if not any(abs(start - known) <= MILLISECOND for known in starts)
        ]

        assert unmatched, (
            "every published time shifted by a whole second still matched some record "
            "boundary, so the assertion in the test above cannot detect an offset at all"
        )
