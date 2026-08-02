# Open questions

Assumptions about the real world that evidence could overturn. Each has a stable
ID. **Cite the ID in the code comment or test that depends on the assumption**
(`# assumes OQ-002`) so that when the answer arrives, `rg 'OQ-002'` finds every
place that must change.

Status: `open` → `answered` (record the answer and the evidence) or `dropped`
(record why it stopped mattering). Never delete an entry.

Fields: **Assumption** is what the code does today. **Evidence** is what would
settle it. **Needs** is the milestone or fixture that can produce that evidence.

---

## OQ-001 — What metadata does the DJI Mic 3 actually embed in transmitter WAVs?
**Assumption:** Standard BWF `bext` time reference and/or a standard timecode tag,
reachable through `ffprobe -show_format -show_streams`.
**Why it matters:** The entire timecode strategy chain in M1/M2 rests on it. The
spec explicitly forbids inventing a layout.
**Evidence:** Raw `ffprobe` JSON + RIFF chunk inventory from a real file.
**Needs:** H1 · **Blocks:** M1, M2 · **Status:** open

**What M1 built while waiting.** Both halves of the assumption are reachable through
FFprobe on a hand-built file: a `bext` time reference surfaces as `format.tags
.time_reference`, and an `INFO`/`ISMP` entry surfaces as `format.tags.timecode`. Each is
a *named strategy* in `dnd_audio.inspection.starttime`, and every source's manifest entry
records which strategy fired, which declined, why, and the assumptions the winner rests
on. Answering this becomes reading one real manifest: if neither tag appears, the
declined list says so in the file's own words, and the RIFF inventory beside it shows
what the file does contain.

## OQ-002 — Is the `TX01`/`TX02` filename component unique across three kits?
**Assumption:** No. It is a receiver-assigned pairing-order identifier, so two kits
can both produce `TX01`. Directory identity is authoritative (INV-11).
**Why it matters:** If it were unique it would be a useful cross-check; if it is
not, treating it as identity would silently mis-attribute a speaker.
**Evidence:** Filenames from six transmitters across three kits recorded together.
**Needs:** H1 · **Blocks:** M1 (validation hints only) · **Status:** open

## OQ-003 — What is the exact DJI filename grammar, including the sequence field?
**Assumption:** Roughly `TX##_MIC###_YYYYMMDD_HHMMSS_orig.wav`, with a
monotonically increasing counter usable only as a secondary chunk-order hint.
**Why it matters:** Chunk discontinuity warnings and `orig`/`edit` pairing in M1.
**Evidence:** A full directory listing from the fixture, including a
power-cycle-induced discontinuity.
**Needs:** H1 · **Blocks:** M1 · **Status:** open

## OQ-004 — Is `time_reference` present, midnight-relative, and at the file rate?
**Assumption:** Yes: integer sample count since midnight at the file's own sample
rate, kept as an integer and never rounded through frames (INV-04).
**Why it matters:** It is the preferred source in the strategy chain; the fallback
is a timecode tag plus configured frame rate.
**Evidence:** `bext` chunk contents from a file whose wall-clock start is known.
**Needs:** H1 · **Blocks:** M1, M2 · **Status:** open

## OQ-005 — Are there DJI-private or iXML chunks, and do they carry timing?
**Assumption:** There may be; the generic RIFF inventory captures ID/offset/size
regardless, with bounded textual payloads retained and every payload hashed in full.
**Why it matters:** If timing lives only in a private chunk, the strategy chain
needs a DJI-specific parser rather than standard BWF handling.
**Evidence:** RIFF chunk inventory from the fixture.
**Also matters for:** M7 — a compressor that cannot reproduce an unknown private
chunk byte-for-byte fails the archival hash check.
**Needs:** H1 · **Blocks:** M1, M7 · **Status:** open

**Half-answered in M1, about FFprobe rather than about DJI.** Measured against FFmpeg
8.0: a WAV carrying both an `iXML` chunk and a four-byte-named private chunk produces
`ffprobe -show_format -show_streams` output mentioning **neither**. So whatever DJI
writes, FFprobe is not the thing that will surface it. `dnd_audio.inspection.riff` walks
the container itself and records every chunk's id, header offset, size, and a SHA-256 of
its complete payload, so when a real file arrives the private chunk is already visible
and the remaining question is only what its bytes *mean*.
`tests/test_riff.py::TestIndependenceFromFfprobe` runs both tools over the same file and
asserts the asymmetry, so this stops being true loudly rather than quietly.

## OQ-006 — How much do the three kits' sample clocks drift over a full session?
**Assumption:** Small enough that timeline sync without affine drift correction is
acceptable for the MVP. Jammed timecode is timeline sync, not a shared word clock.
**Why it matters:** If drift is material, the transcript's word times and any
future coherent processing degrade over four hours.
**Evidence:** Differential clap lag measured near the start and near the end of a
~4-hour recording.
**Needs:** H2 · **Blocks:** nothing (warning threshold tuning) · **Status:** open

## OQ-007 — Does dual-file mode produce `orig`/`edit` pairs as assumed?
**Assumption:** Yes, distinguishable by filename suffix; `orig` is 32-bit float and
is the only file consumed.
**Why it matters:** Selection rules, duplicate detection, and the
`allow_processed_audio` recovery path in M1.
**Evidence:** A fixture recorded with dual-file mode enabled.
**Needs:** H1 · **Blocks:** M1 · **Status:** open

## OQ-008 — Does AMD's stable `gfx1151` index yield a working Torch under uv + FHS?
**Assumption:** Yes, with the `rocm[libraries]` sdist building at install time
given the FHS compiler toolchain and setuptools ≥ 70.2.
**Why it matters:** All of M6a. Failure means finding another tested gfx1151 build.
**Evidence:** A successful locked install plus a BF16 op on the real device.
**Needs:** M6a · **Blocks:** M6a, M6b · **Status:** open

## OQ-009 — Where does `qwen-asr`'s timestamp path actually chunk?
**Assumption:** 180 s in 0.0.6, which is why `max_segment_s` defaults to 120 and
the advertised five-minute model limit is not trusted.
**Why it matters:** Segment construction in M4 and request sizing in M6b.
**Evidence:** Reading the installed package plus a long-segment experiment.
**Needs:** M6b · **Blocks:** M6b · **Status:** open

## OQ-010 — How is Silero pinned and loaded without a runtime `torch.hub` fetch?
**Assumption:** A pinned package plus a locally vendored/cached model artifact with
a recorded revision, running on CPU or ONNX.
**Why it matters:** INV-05 (offline default suite) and cache-key identity (INV-08).
**Evidence:** A working offline load path in M3.
**Needs:** M3 · **Blocks:** M3 · **Status:** open

## OQ-011 — Does `ffprobe` expose an exact PCM sample count for these files?
**Assumption:** `duration_ts` plus time base is exact for PCM; a decode pass is the
fallback when it is not.
**Why it matters:** Sample-exact chunk end computation in M2, and the
one-sample duration tolerance in M2's gate.
**Evidence:** Compare `ffprobe` output against a decoded sample count on a real
file and on synthetic fixtures.
**Needs:** M1 (synthetic), H1 (real) · **Blocks:** M2 · **Status:** open

**Synthetic half answered in M1, and the approach changed as a result.** No decode is
needed for either half: the RIFF `data` chunk size divided by the block alignment is
exact by construction for PCM, and M1 already walks the container. So the *data chunk*
is the source and `duration_ts` is the cross-check, rather than the other way round.
Across all twelve canonical-fixture files the two agree exactly
(`tests/test_probe.py::TestExactSampleCount`). Their agreement is recorded per source in
the manifest as `container.sample_count_agrees`, and a disagreement raises a
`sample_count_disagreement` warning — so H1 answers the real half by reading the
manifest rather than by running an experiment. A `data` size that is not a whole number
of frames falls back to `duration_ts` instead of flooring, which would invent a sample.

## OQ-012 — Do all three receivers hold identical timecode after the LTC jam?
**Assumption:** Yes, and they stay matched for the session while powered.
**Why it matters:** It is the premise of cross-kit synchronization. A mismatch at
the start is a capture-procedure problem the pipeline should detect and warn about.
**Evidence:** Displayed timecode/rate on all three receivers recorded after the
jam procedure, cross-checked against the files' embedded timecode.
**Needs:** H1 · **Blocks:** nothing directly · **Status:** open

## OQ-013 — How much working disk does a full session actually consume?
**Assumption:** Roughly 25 GiB for a four-hour six-transmitter session — about 15 GiB of
48 kHz float32 working audio, 5 GiB of 16 kHz derivatives, and 3 GiB of mix
intermediate — so `doctor` warns below 40 GiB free. The arithmetic is in
`src/dnd_audio/doctor.py`; how many intermediates actually survive on disk is a guess
about a pipeline that does not exist yet.
**Why it matters:** `doctor` runs before a session, and a threshold set too low turns
the warning into noise that fires once the disk is already gone. M2 owns the real
preflight, which knows the actual session length instead of assuming four hours.
**Evidence:** Measure `work/` after the first complete run.
**Needs:** M2 (preflight), H2 or the first real session (real numbers) ·
**Blocks:** nothing · **Status:** open

**Partially answered in M2, and the arithmetic's premise changed.** M2 builds a preflight
that estimates work-space from the session's *actual* length rather than an assumed four
hours, and from the artifacts actually requested. Two of the three terms in the original
estimate are now wrong by construction: the 48 kHz working audio is a segment map rather
than 15 GiB of materialized float32 (ADR-0011), and the mix intermediate belongs to M5.
What M2 can measure is its own footprint; the full-pipeline number this question asks for
still needs H2 or a real session, so this stays **open**.

## OQ-014 — How long is a real session, and when is an inferred span implausible?
**Assumption:** Under 12 hours. A span longer than that is unambiguous arithmetically —
midnight rollover is unique within one 24-hour cycle — but implausible enough to be worth
a human's attention, so it warns rather than failing (ADR-0009).
**Why it matters:** Only whether a warning fires. It changes no placement and no artifact.
Set too low it is noise; set too high it never fires and the operator learns nothing from
a session whose timecode is a day out.
**Evidence:** The wall-clock length of real sessions, and whether the warning ever fires
on one.
**Needs:** H2 or the first real session · **Blocks:** nothing · **Status:** open

## OQ-016 — Is a session always the shortest arc through its chunk start times?
**Assumption:** Yes. With no configured origin, M2 infers which chunks fall after midnight
by treating the widest quiet stretch in the sources' start times as the one containing
midnight — which is the same as assuming the session is the *shortest* arc that contains
every start (ADR-0009, `timeline/origin.py::_cycles_by_largest_gap`).
**Why it matters:** Starts at 23:00 and 01:00 admit two readings: a two-hour session across
midnight, or a twenty-two-hour session within one day. The evidence does not distinguish
them; this assumption picks the first. A session that genuinely ran longer than half a day
without a configured origin would be reconstructed with its chunks on the wrong days, which
moves audio by hours. Every session relying on the inference is warned
(`midnight_rollover_inferred`), and a recorded `origin_date` plus `origin_timecode` removes
the question entirely.
**Evidence:** The wall-clock span of real sessions, and whether any is ever run without a
configured origin. Overlaps with OQ-014, which asks the same thing from the other side.
**Needs:** H2 or the first real session · **Blocks:** nothing · **Status:** open

## OQ-015 — Where is the DJI receivers' timecode zero relative to real midnight?
**Assumption:** `00:00:00:00` is jammed to real midnight, so a timecode and a BWF sample
reference in the same session share a day origin.
**Why it matters:** At a fractional non-drop rate a timecode day is not a real day —
2 592 000 frames at 30000/1001 fps is 86 486.4 seconds, 86.4 seconds longer than a
calendar day. Within a session that costs nothing, because elapsed time converts exactly;
it matters only where the two domains are anchored to each other. A session mixing BWF and
timecode evidence at 23.98F or 29.97F therefore rests on this assumption, and M2 warns when
one does (ADR-0009). The canonical fixture mixes exactly these domains, at 30F, where a
timecode day *is* 86 400 seconds and the question does not arise.
**Evidence:** The displayed timecode on all three receivers after the LTC jam, recorded
against wall-clock time, cross-checked with the `bext` origination time in the files.
**Needs:** H1 · **Blocks:** nothing directly · **Status:** open
