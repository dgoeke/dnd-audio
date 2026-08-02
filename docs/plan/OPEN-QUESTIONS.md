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
regardless, with bounded textual payloads retained and larger ones hashed.
**Why it matters:** If timing lives only in a private chunk, the strategy chain
needs a DJI-specific parser rather than standard BWF handling.
**Evidence:** RIFF chunk inventory from the fixture.
**Also matters for:** M7 — a compressor that cannot reproduce an unknown private
chunk byte-for-byte fails the archival hash check.
**Needs:** H1 · **Blocks:** M1, M7 · **Status:** open

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
