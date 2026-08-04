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
**Needs:** H1 · **Blocks:** M1, M2 · **Status:** **answered** (sample probe, 2026-08-02)

**Answer — `bext` and `iXML`, both present, and no `INFO`/`ISMP` timecode.** Four real
DJI Mic 3 transmitter files (firmware `ver:02.00.06.01`) carry, in order: `fmt `(16),
`bext`(602), `iXML`(1088), `cue`(28, **zero cue points**), `PAD`(30982 — audio starts at a
32768-byte boundary), `data`. `bext.originator` is `MIC 3`, `bext.description` is the
firmware string, `originator_reference` and `coding_history` are **empty**, and
`bext.version` is 0 even though the iXML claims `BWF_VERSION 02.00`. FFprobe surfaces the
sample reference as `format.tags.time_reference` exactly as assumed — so the *plumbing* is
right. What it means is not: see **OQ-004**.

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
**Needs:** H1 · **Blocks:** M1 (validation hints only) · **Status:** **answered**
(sample probe, 2026-08-02)

**Answer — no, it is not unique, and the assumption was right.** Two receivers recording
together each produced a `TX01` and a `TX02`; the operator renamed them by hand afterwards
so the four files could sit in one directory at all. That is the collision this entry
predicted, observed directly at two kits rather than three. **INV-11 stands: the configured
directory is the only identity**, and `TX##` is a validation hint that two kits will happily
duplicate. Note for whoever reads the sample files: their `TX##` components are the
operator's renames, not DJI's — only the `MIC###`, date, and time components are original.

## OQ-003 — What is the exact DJI filename grammar, including the sequence field?
**Assumption:** Roughly `TX##_MIC###_YYYYMMDD_HHMMSS_orig.wav`, with a
monotonically increasing counter usable only as a secondary chunk-order hint.
**Why it matters:** Chunk discontinuity warnings and `orig`/`edit` pairing in M1.
**Evidence:** A full directory listing from the fixture, including a
power-cycle-induced discontinuity.
**Needs:** H1 · **Blocks:** M1 · **Status:** open — **grammar confirmed, counter not**

**Partial evidence (sample probe, 2026-08-02).** The grammar is exactly as assumed:
`TX##_MIC###_YYYYMMDD_HHMMSS_orig.wav`, and `inspection`'s parser recognized all four real
files without a change. Two observations about the middle field, which M1 reads as
`sequence`:

- **It is not a per-transmitter serial.** Both transmitters on one receiver produced
  `MIC001`, while the other receiver's two produced `MIC004` and `MIC002`. A serial would
  not repeat within a kit.
- **It is consistent with a per-transmitter recording counter**, which is what this entry
  assumed: the operator had recorded with one kit several times before and with the other
  not at all. Consistent with, not evidence for — two files at `001` prove very little.

The date and time components match `bext.origination_date`/`_time` exactly in all four.
**Still unanswered:** what the counter does across a power cycle, which is the only reason
M1 wants it, and whether it ever wraps. The sample has no power cycle in it.

## OQ-004 — Is `time_reference` present, midnight-relative, and at the file rate?
**Assumption:** Yes: integer sample count since midnight at the file's own sample
rate, kept as an integer and never rounded through frames (INV-04).
**Why it matters:** It is the preferred source in the strategy chain; the fallback
is a timecode tag plus configured frame rate.
**Evidence:** `bext` chunk contents from a file whose wall-clock start is known.
**Needs:** H1 · **Blocks:** M1, M2 · **Status:** **answered — and the answer is no, on
both halves** (sample probe, 2026-08-02). **The implementation was corrected in M8**
(2026-08-03, ADR-0031); see the consequences below.

**Answer — it is neither midnight-relative nor sample-accurate.**

| file | `bext.origin_time` | `time_reference` | ÷48000 | as a clock | ÷1600 |
| --- | --- | --- | --- | --- | --- |
| rx-a tx1 | 19:26:55 | 18628800 | 388.100 s | 00:06:28 | 11643 |
| rx-a tx2 | 19:26:55 | 18627200 | 388.067 s | 00:06:28 | 11642 |
| rx-b tx1 | 19:27:39 | 18347200 | 382.233 s | 00:06:22 | 11467 |
| rx-b tx2 | 19:27:39 | 18347200 | 382.233 s | 00:06:22 | 11467 |

1. **Not since midnight.** A 19:26:55 file would carry 3 360 720 000 samples; it carries
   18 628 800, a factor of 180 out. The files also contradict each other: the pair created
   *44 seconds later* has the *smaller* reference. Subtracting gives per-receiver epochs of
   19:20:26.9 and 19:21:16.8 — consistent with **elapsed since that receiver powered on**,
   the alternative this entry named. One recording supports that reading; a second from the
   same power-on cycle would confirm it, and nothing here rules out another epoch entirely.
2. **Not sample-accurate.** Every value is an exact multiple of **1600**, which is one frame
   at the 30/1 rate the iXML declares. The timestamp is frame-quantized: **resolution is
   33.3 ms**, not one sample. So "kept as an integer and never rounded through a frame
   count" describes what M2 does with the number, not what DJI did to it before writing it.

DJI copies the same value into iXML's `TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO` beside
`TIMESTAMP_SAMPLE_RATE 48000`, so the misleading label appears twice and agreeing with
itself is not evidence.

**Consequence, acted on in M8 (2026-08-03).** Absolute wall-clock placement from a BWF
reference alone is **not available** on this hardware; within one receiver, and across
receivers that were jammed (**OQ-023**), the value is usable for relative placement to
±33 ms. **ADR-0031** records the reframe and what moved with it: the assumption string M1
stamps into every manifest now says the origin is the recorder's and not midnight,
`rasterize`'s docstrings and constants say the same, `ZeroDomain`'s `real_time` became
`recorder_epoch`, `since_day_origin_samples` became `since_domain_origin_samples`, and
`timeline.json` is therefore **schema version 2**. The arithmetic is unchanged throughout —
it never depended on what the origin meant — which is exactly why prose was the only place
the false claim could survive, and why the test for it reads the artifacts on disk rather
than the source. The spec is amended in the same change.

Two things that did *not* change, deliberately. The 24-hour unwrap stays, because it is
spec-required and tested and a genuinely midnight-relative recorder needs it; what it lost
was its justification, now registered as **OQ-026** and cited from `cycle_units`. And
mixing a BWF reference with a timecode tag stays permitted — under this reading they are
the same clock in two units, which is weaker grounds for refusing to relate them, not
stronger.

**The quantization half was acted on in M8 (2026-08-03), and it was not cosmetic.**
`rasterize` treated a BWF reference as sample-*exact* because it is expressed in samples,
giving two chunks of one track a **one-sample** overlap tolerance. A second chunk whose
reference floors backward by up to 1599 samples — which is what a counter that ticks once a
frame does at every switch-off — was therefore a *material* overlap, and
`timecode.chunk_overlap_policy` defaults to `reject`. An ordinary four-hour session with
several chunks per track would have failed outright, on correct evidence. The tolerance now
comes from `timecode.bwf_reference_quantum_samples`, default 1600; the value cannot be read
from the file (**OQ-024**), so it is configuration with a measured default, and stating 1
restores the old behaviour exactly. Nothing caught this because every fixture in the
repository wrote sample-exact references and the jam capture has one chunk per track —
`quantized_reference_session` now writes floored ones.

---

**Reframed 2026-08-03, and the reframe shrinks the work considerably. A *shared* origin is
sufficient; wall clock was never the requirement.**

Placing six tracks on one timeline is `rasterize.session_position(source_time,
session_zero_time)` — a **subtraction**. A common epoch cancels out of it no matter what that
epoch means. Wall-clock time is useful for archival naming and for a human reading a report;
it is not needed to align anything. Earlier phrasing here treated "not midnight-relative" as
though it invalidated the quantity, when what it invalidates is the *label*.

So the requirement is not "recover real time" but **"make every file share one origin"**, and
the hardware already has a mechanism for that: jamming receivers L-OUT → L-IN, which the
operator can perform and watch succeed on the displays (**OQ-012**). Whether the jammed value
reaches `bext.time_reference` is the thing that actually decides this, and it is **OQ-023** —
**answered 2026-08-03: it does, to within one frame.** Cross-receiver alignment therefore
comes free from the metadata, to ±33 ms, and this reframe holds in full: a shared origin was
the whole requirement, and the hardware supplies one.

**What genuinely breaks, narrowly.** Only the case where session zero comes from a configured
wall-clock `timecode.origin_timecode` while the files carry a power-on-relative count: the
subtraction then spans two unrelated origins, and the day-origin and 24-hour-wrap logic in
`timeline/origin.py` assumes a real day that these files do not have. Inferring session zero
from the earliest source instead sidesteps all of it, and that path already exists.

**Two corrections to what was written above, from DJI's documentation** ([FAQ](https://www.dji.com/mic-3/faq),
[Introduction to the Use of Timecode](https://support.dji.com/help/content?customId=01700007306&spaceId=17&re=US&lang=en&documentType=&paperDocType=ARTICLE)):

- **There is no time-of-day mode to misconfigure.** Timecode is "a frame counter relative to
  recording duration"; it "resets to zero and restarts"; users cannot set its value. The
  per-receiver epochs are the hardware behaving as designed, not a settings mistake — so
  there is nothing to fix at capture on that axis, only something to *share* via a jam.
- **33.3 ms is not the floor.** Supported rates are 23.98, 24, 25, 29.97, 29.97DF, 30, **50
  and 60**. The sample files declare 30/1. At **60 fps the quantum halves to 800 samples —
  16.7 ms**. That is a menu setting and should be applied before the next capture regardless
  of how OQ-023 resolves.
  **Retracted 2026-08-03 — the setting does not reach the file.** A receiver set to 60 fps
  wrote `TIMECODE_RATE 30/1` and a `time_reference` on a 1600-sample boundary exactly like
  the 30 fps receiver beside it. See **OQ-024**. 33.3 ms *is* the floor on this hardware for
  the files this project consumes, and H1's recipe no longer asks for 60 fps.

**An unused signal worth remembering — and now known unusable across receivers.**
`bext.origination_date`/`origination_time` carry real wall clock — 19:26:55 on the rx-a pair,
19:27:39 on rx-b — consistent within each receiver. The strategy chain in
`inspection/starttime.py` reads `time_reference` and a timecode tag and **nothing reads
these**. One-second resolution cannot place a session precisely, but it looked like it could
bound a cross-receiver offset to ±1 s, which is exactly the search window a correlator needs
and does not currently have. The caveat attached to that — two receivers' real-time clocks
are independent and may not agree (INV-12) — **was measured on 2026-08-03 and is fatal to the
idea.** The two receivers' implied epochs differed by **48.7 s** while their timecode agreed
to under one frame, so the wall clock was wrong by nearly a minute and the timecode was
right. Wall clock is for archival naming and for a human reading a report. It must never
anchor a cross-receiver offset, and **nothing in the code currently prevents that** — a guard
belongs with OQ-004's other scoped M1/M2 work.

**What M2 does with it.** `timeline/rasterize.py` treats a `bwf_sample_reference` as
unsigned samples since **midnight at the file's own rate**, converts it to exact rational
seconds in that domain, and unwraps a 24-hour cycle by adding whole samples-per-day before
any conversion. If the real answer is "midnight-relative but at 48 kHz regardless of the
file's rate", or "relative to power-on rather than midnight", the change is localized to
that one conversion — but every placement in every session moves. `rg 'OQ-004'` finds it.

## OQ-005 — Are there DJI-private or iXML chunks, and do they carry timing?
**Assumption:** There may be; the generic RIFF inventory captures ID/offset/size
regardless, with bounded textual payloads retained and every payload hashed in full.
**Why it matters:** If timing lives only in a private chunk, the strategy chain
needs a DJI-specific parser rather than standard BWF handling.
**Evidence:** RIFF chunk inventory from the fixture.
**Also matters for:** M7 — a compressor that cannot reproduce an unknown private
chunk byte-for-byte fails the archival hash check.
**Needs:** H1 · **Blocks:** M1, M7 · **Status:** **answered** (sample probe, 2026-08-02)

**Answer — no private chunks, and the iXML carries nothing new.** The inventory is
`fmt `, `bext`, `iXML`, `cue`, `PAD`, `data` — all standard. The iXML is a `<BWFXML>`
document whose `<BEXT>` block restates the `bext` chunk field for field, and whose
`<SPEED>` block adds `MASTER_SPEED`/`TIMECODE_RATE` `30/1`, `TIMECODE_FLAG NDF`,
`FILE_SAMPLE_RATE`/`DIGITIZER_SAMPLE_RATE` 48000, `AUDIO_BIT_DEPTH`, and a
`TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_*` pair duplicating `time_reference` (**with the same
wrong semantics — OQ-004**). The declared 30/1 timecode rate is itself useful: it is where
the 1600-sample quantization in OQ-004 comes from. `cue` declares **zero** cue points, and
`PAD` is alignment padding that puts `data` at offset 32768. **No DJI-specific parser is
needed**, and M7's archival hash check has no unknown chunk to reproduce.

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
**Needs:** H2 · **Blocks:** nothing (warning threshold tuning) · **Status:** **partially
answered — bounded at ≈1 ppm, catastrophic case ruled out** (jam verification capture,
2026-08-03); the long-baseline confirmation is still H2

**First measurement — the clocks are far better than consumer tolerance.** This is the
question that decided whether the architecture works at all, and it is not really about
timecode. Each `orig` file carries **one** `time_reference`, stamped at the start; from there
the transmitter's own crystal defines the timeline. At typical consumer tolerance (±20–50 ppm)
a four-hour session would diverge by 288–720 ms — an order of magnitude worse than any
quantization, growing without bound with session length, and fatal to treating each
transmitter as an independent recorder.

Measuring the residual lag in the first third of each overlap against the last third, across
all six transmitter pairs (`docs/fixtures/2026-08-03-jam-verification.md`):

```
drift:  +1.0  −0.2  +2.4  −0.3  +0.9  +1.1  ppm
lag change over ~30 s:  0.00 to 0.07 ms
```

Consistent with zero inside a noise floor of about **±3 ppm** on this 30 s baseline. At the
pessimistic end, 3 ppm over 4 hours is 43 ms and over 6 hours 65 ms; at the likely ~1 ppm,
14 ms and 22 ms. **Drift over a full session is the same size as or smaller than the 33 ms
quantization already present at file start** (**OQ-004**) — so the cross-track error budget
does not grow materially with session length, and the MVP's decision to skip affine drift
correction is supported rather than merely assumed.

**Why this stays open.** A 30 s baseline cannot distinguish 1 ppm from 3 ppm, and it contains
no thermal excursion, no battery swap, and no power cycle — all of which H2 will. It rules
out the catastrophic case; it does not set the warning threshold.

**M2 built the instrument.** `session.sync_qa` (off by default) correlates each track
against a reference near both ends and reports the lag at each, never a correction — a
constant lag is a timecode disagreement, a lag that changes between the ends is drift.
Answering this is now enabling a config flag on a long recording and reading the warnings,
not writing measurement code. Proven on a synthetic drift fixture whose tracks carry
identical metadata and differ only in where the end transient sits in the audio: +0.000 ms
at the start, +20.000 ms at the end.

## OQ-007 — Does dual-file mode produce `orig`/`edit` pairs as assumed?
**Assumption:** Yes, distinguishable by filename suffix; `orig` is 32-bit float and
is the only file consumed.
**Why it matters:** Selection rules, duplicate detection, and the
`allow_processed_audio` recovery path in M1.
**Evidence:** A fixture recorded with dual-file mode enabled.
**Needs:** H1 · **Blocks:** M1 · **Status:** open — **but half the assumption is already
known false**

**`orig` is not always 32-bit float (sample probe, 2026-08-02).** Two of four real
transmitters wrote `pcm_s24le` `_orig` files and two wrote `pcm_f32le`, same firmware,
same session — a per-transmitter setting the operator had not matched across kits, which
is exactly the mistake H1's recipe already tells the owner to check for. It will be made
again, so the pipeline has to survive it.

**It currently does not.** `ingest` refuses the 24-bit files:

> `undecodable_source: ... is 1-channel pcm_s24le, and the working path reads mono
> pcm_f32le — dual-file mode's `orig`. An integer format cannot be converted to float32
> exactly, so it is refused rather than quietly rounded (ADR-0011).`

**The stated reason is wrong for 24-bit.** A 24-bit signed integer is exactly
representable in float32 — the significand is 24 bits — so `s24 → f32` is lossless.
Verified empirically rather than argued: 2 000 000 random values plus the range edges
round-trip with zero error, and the same test on 32-bit integers fails, so the guard is
right to refuse `s32` and wrong to refuse `s24` with it. ADR-0011's principle — never
quietly round — is intact; only its implementation is too broad.

**Fixed in M8 (2026-08-03), as its own scoped piece of work.** The allowlist became the
principle it was always meant to be: a format is accepted when it is a signed
little-endian integer or IEEE float **and** converts to float32 with zero error
(**ADR-0030**). That is `pcm_f32le`, `pcm_s24le` and `pcm_s16le`. `pcm_s32le` is still
refused, and now with the sentence that is true of it; `pcm_u8` is refused as *untested*
rather than as unrepresentable, because it would convert exactly and only its
offset-binary convention is unimplemented. The round-trip is measured over 2 000 000
values per width and cross-checked against FFmpeg's own decode, so the `2**(bits-1)`
scaling convention is agreed with another implementation rather than asserted. A
mixed-format session ingests end to end, and the operator still gets an
`unexpected_codec` warning naming the settings mismatch — readable is not the same as
intended. The spec's "32-bit float" input rule is amended in the same change.

**The remaining half of this entry is untouched:** whether `orig`/`edit` pairs appear as
assumed still needs a fixture recorded in dual-file mode. The sample captures have no
`edit` files at all, so this entry stays **open**.

## OQ-008 — Does AMD's stable `gfx1151` index yield a working Torch under uv + FHS?
**Assumption:** Yes, with the `rocm[libraries]` sdist building at install time
given the FHS compiler toolchain and setuptools ≥ 70.2.
**Why it matters:** All of M6a. Failure means finding another tested gfx1151 build.
**Evidence:** A successful locked install plus a BF16 op on the real device.
**Needs:** M6a · **Blocks:** M6a, M6b · **Status:** **answered — yes, and the sdist needed
nothing the FHS shell did not already have** (M6a, 2026-08-03)

**Answer — it works, and the assumption was right about the build and wrong about the
routing.** `torch 2.9.1+rocm7.13.0` resolves, installs, and computes correctly:

| | |
| --- | --- |
| torch | `2.9.1+rocm7.13.0`, from `https://repo.amd.com/rocm/whl/gfx1151/` |
| HIP runtime | `7.13.99004-3309c6114a` |
| device | `Radeon 8060S Graphics`, `gcnArchName` **`gfx1151`** |
| GPU bfloat16 | exact |
| GPU float32 | exact |
| CPU bfloat16 | exact |

The `rocm==7.13.0` sdist **built first time** inside `nix develop`'s FHS sibling with the
`targetPkgs` list M0 guessed at from the host's ComfyUI module — no additions, no compiler
errors. setuptools was left unconstrained as the spec instructs. So the half of this
question about the *build* was right, and the FHS package list is now a tested set rather
than a starting point.

**What was wrong is the half nobody asked about: the routing.** The spec's configuration
sketch routes `torch` alone, and that does not resolve. `[tool.uv.sources]` only applies to
packages that are also **direct members of a dependency list**; a requirement discovered
inside another package's metadata is looked up on PyPI regardless — silently, with the
wrong registry simply recorded in the lock. Torch needs `rocm[libraries]==7.13.0` and
`triton==3.5.1+rocm7.13.0`, and `rocm[libraries]` in turn needs `rocm-sdk-core` and
`rocm-sdk-libraries-gfx1151`; all four are now listed in the `asr-qwen` group so the
routing can reach them. Five packages resolve from the AMD index, no `nvidia-*` wheel
appears, and `accelerate 1.12.0` sits in the same lock wanting `torch>=2.0.0` and not
getting a CUDA build. See [ADR-0025](decisions/0025-provisioning-amd-gfx1151-torch.md).

**Not answered here, because M6a does not run a model:** whether this stack is *fast*
enough, and whether the AOTriton flag measurably changes SDPA. Both are M6b's smoke test.

## OQ-021 — Which render node backs the ROCm compute device on a multi-GPU host?
**Assumption:** It does not matter yet, because there is one. `runtime.render_nodes()`
globs `/dev/dri/renderD*`, opens every match, and treats one that opens as sufficient; the
spec names `/dev/dri/renderD128` and qualifies it with the word *currently*, which is a
numbering that shifts with how many DRM devices the kernel enumerated first.
**Why it matters:** Only for the accuracy of a `doctor` check, and only on a host with more
than one GPU. Two failure directions, both mild: a second, unrelated GPU whose node is
restricted would be reported as a refused node beside a working one (noise, not a false
failure — one openable node is a pass), and a host where the *compute* node is the
restricted one would be reported as healthy while Torch found no device. The second is the
one worth closing, and on that host `torch.cuda.is_available()` is already false, so the
`gpu` check catches it and only the *explanation* is wrong.
**Evidence:** A host with two DRM render nodes, and a way to tie one to Torch's device 0 —
`amdgpu`'s sysfs topology under `/sys/class/kfd/kfd/topology/nodes/` carries the DRM render
minor per agent, which is the obvious source if this ever needs answering.
**Needs:** a second GPU on a target host · **Blocks:** nothing · **Status:** open

## OQ-022 — Is Qwen inference on ROCm reproducible across cold runs?
**Assumption:** Yes. `transcript.json`, `transcript.md` and `work/transcript-records.json`
are declared byte-stable on unchanged inputs and unchanged configuration (INV-02), and from
M6b onward the text and word times in them come out of a GPU kernel. The adapter disables
sampling explicitly rather than inheriting whatever generation configuration the snapshot
ships, so the remaining question is whether *greedy* decoding through ROCm SDPA on gfx1151
returns bit-identical logits — and therefore identical token choices and identical aligner
argmaxes — on two separate cold executions of the same request.
**Why it matters:** The ASR cache hides this completely. A warm run replays stored text and
stored word times, so every byte-stability test the project has ever run would pass on a
model that answers differently every time. It only surfaces when a cache is cleared, a cache
key legitimately changes, or the work is re-run on another machine — and it surfaces as two
transcripts of one session that disagree, with nothing in either saying which is right.
Reduction-order nondeterminism in an attention kernel is the ordinary cause; it is not
exotic, and it costs nothing until the moment it costs a byte-stability claim two milestones
old.
**Evidence:** The same request submitted twice with the cache bypassed, text and word times
compared **exactly** rather than within a tolerance — a tolerance would pass on precisely the
wobble this asks about. `tests/test_qwen_smoke.py` does that on the target host, and the
cheap extension if it ever fails is to compare across a process restart as well, since a
warm HIP context can be reproducible where a cold one is not.
**Needs:** M6b · **Blocks:** nothing — INV-02 is a claim about artifacts, and if the answer
is no the fix is to say so in the invariant rather than to change the pipeline ·
**Status:** **answered — yes, on this stack, in process and across processes** (M6b,
2026-08-03)

**Raised by M6b's plan review**, which noticed that a rule two milestones old had quietly
acquired a new dependency nobody had written down.

**Answer — reproducible, and compared exactly rather than within a tolerance.** Two
measurements on `gfx1151` in bfloat16 through ROCm SDPA:

- **In process**, the same request submitted twice with the cache bypassed returns identical
  text, identical language, identical truncation verdict, and **identical word times** — done
  for two different pieces of real audio, because one short utterance agreeing with itself
  could be a coincidence of having little to disagree about
  (`tests/test_qwen_smoke.py::TestOq022Determinism`).
- **Across processes**, which is the stronger claim and the one the review specifically asked
  for, since a warm HIP context can be reproducible where a cold one is not: two separate
  interpreter runs, each loading the models from scratch, produced byte-identical text and
  the same fifteen word times.

Sampling is disabled explicitly rather than inherited from the snapshot's
`generation_config.json` (`qwen.py::_force_greedy`), so this is a statement about greedy
decoding and not about a temperature that happened to be zero.

**INV-02 therefore stands as written**, and needs no amendment naming a model boundary. What
would overturn this is a different GPU, a different ROCm or Torch version, or a batch size
above one — none of which this project uses today, and all of which are in the ASR cache key
(INV-08), so a change to any of them re-runs the work rather than serving an answer produced
under different arithmetic. See [ADR-0028](decisions/0028-the-qwen-adapter-seam.md).

## OQ-009 — Where does `qwen-asr`'s timestamp path actually chunk?
**Assumption:** 180 s in 0.0.6, which is why `max_segment_s` defaults to 120 and
the advertised five-minute model limit is not trusted.
**Why it matters:** Segment construction in M4 and request sizing in M6b.
**Evidence:** Reading the installed package plus a long-segment experiment.
**Needs:** M6b · **Blocks:** M6b · **Status:** **answered — 180 s, and it is not on this
project's route at all** (M6b, 2026-08-03)

**Answer — the assumption was exactly right about the number.**
`qwen_asr/inference/utils.py` declares `MAX_FORCE_ALIGN_INPUT_SECONDS = 180` against
`MAX_ASR_INPUT_SECONDS = 1200`, and `Qwen3ASRModel.transcribe` picks between them on
`return_time_stamps`. So the timestamp path chunks at 180 s, the advertised five-minute
model limit is indeed not the package limit, and `max_segment_s`'s 120 s cap sits under
both with room to spare.

**What the reading alone does not show is that neither limit binds here.** The chunking
lives in `Qwen3ASRModel.transcribe`, and M6b's adapter never calls it: the package's
combined `transcribe(return_time_stamps=True)` runs alignment *after* ASR and destroys
already-generated text when the aligner raises, which makes the gate's "alignment failure
retains the segment text" unimplementable through it. The adapter therefore drives
`transcribe(return_time_stamps=False)` and `Qwen3ForcedAligner.align` separately
(**ADR-0028**) — and `align` does no chunking at all. So on this route the ASR call would
chunk at 1200 s and alignment never chunks, while `max_segment_s` holds every padded window
at or under 120 s regardless.

Both halves are asserted in `tests/test_qwen_smoke.py::TestOq009WhereThePackageChunks`: the
constants, and that `align`'s source contains neither `split_audio_into_chunks` nor
`return_time_stamps`. **The 120 s cap stays**, because it is the number that actually bounds
a request's memory and latency, and because a limit that is currently unreachable is a poor
thing to rely on remaining unreachable.

## OQ-010 — How is Silero pinned and loaded without a runtime `torch.hub` fetch?
**Assumption:** A pinned package plus a locally vendored/cached model artifact with
a recorded revision, running on CPU or ONNX.
**Why it matters:** INV-05 (offline default suite) and cache-key identity (INV-08).
**Evidence:** A working offline load path in M3.
**Needs:** M3 · **Blocks:** M3 · **Status:** **answered** (M3)

**Answer — the artifact, not the package.** The `silero-vad` distribution hard-depends on
`torch` and `torchaudio`, which is unacceptable in the environment the default suite runs in
(INV-05) and would pre-empt M6a's AMD wheel index and per-package sourcing. Its **ONNX**
protocol needs neither: inputs are `input` (1, 64 context + 512 samples), `state` (2, 1, 128),
and `sr` (int64); outputs are a probability and the next state. So the pin is a commit-pinned
model *file* driven by `onnxruntime` on CPU from a plain NumPy loop, never the package.

Pinned by upstream release `v6.2.1`, commit `7e30209a3e901f9842f81b225f3e93d8199902b1`,
and sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` — verified
against **two independent sources of the same file**, the repository at that commit and the
published 6.2.1 wheel's copy, which are byte-identical. That equality is what makes "we did
not install the package" a packaging decision rather than a change of artifact.

The runtime and the *calling interface* are pinned too (frame size, context size, state
shape, input names, rate) and all of it enters the detection cache key: a runtime upgrade
that changed a kernel's rounding must re-run the work. `models fetch` verifies the digest
before moving the file into place and is the only command permitted to touch the network
(INV-06). Answering this required amending the spec as well as M3's charter — see
[ADR-0013](decisions/0013-silero-through-onnx-runtime.md).

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
**Needs:** H1 · **Blocks:** nothing directly · **Status:** **answered for two receivers**
(jam verification capture, 2026-08-03); the third receiver and "stays matched for the
session" are still H1/H2

**Answered — the jam holds, and it reaches the files.** Two receivers jammed L-OUT → L-IN
produced files whose `bext.time_reference` values agree on the true inter-receiver offset to
**17–30 ms**, inside one 30 fps frame, measured against audio cross-correlation
(`docs/fixtures/2026-08-03-jam-verification.md`). The follow-on assumption this entry
deferred to **OQ-023** — that a matched display propagates into the written file — is
answered affirmatively there.

**Two caveats keep this from being fully closed.** Only two of the three receivers were
exercised, and the capture is 47 seconds, so "they stay matched for the session" is
untested — that is OQ-006 and H2. Note also that the two receivers were on **different frame
rates** (**OQ-024**) and the jam held anyway.

**The capability is not in doubt (operator, 2026-08-03).** Connecting the receivers
L-OUT → L-IN and pressing SYNC visibly aligns the timecodes on their displays. This entry
previously read as though the jam were unreliable equipment; it is not. What follows below
is about one specific capture that predates a jam, and the earlier phrasing over-generalized
from it.

**About the 2026-08-02 sample probe specifically.** Within each pair the two transmitters
are timecode-synced — TX01 with TX02, TX03 with TX04 — but the four files are not mutually
aligned. No displayed timecode was recorded at capture, which is the evidence this entry
actually asks for, so this is inference from the files rather than measurement.

It is consistent with what the files say. **OQ-004** derived per-receiver epochs of
19:20:26.9 and 19:21:16.8 from the `bext` references — about fifty seconds apart, which is
what two independently-started receivers look like and not what a successful jam does.

**Two consequences.** For M6b, none: every measurement in this milestone used TX03 and TX04
only — one receiver, one pair — so nothing here rests on cross-receiver alignment. For H1,
the lesson survives the correction above even though the diagnosis changed: the jam is a step
whose *outcome is not visible in the output*, so the recipe has to capture evidence of it at
the time. `session.sync_qa` (M2, off by default) is the instrument that would settle it from
the audio, and running it on the sample pair is a cheap way to turn this inference into a
measurement.

**What this entry is actually asking, restated.** Two receivers holding the same displayed
timecode is necessary and not sufficient. The pipeline never sees a display — it reads
`bext.time_reference`. Whether a jammed display propagates into the written file is a
separate assumption and is now **OQ-023**, which is the one to test first, because if it
fails then holding identical timecode on the displays buys this project nothing.

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

## OQ-017 — What separates real speech from lav bleed at a real table?
**Assumption:** Bleed arriving at another wearer's lav is both *much quieter* than that
wearer's own voice and *strongly correlated* with the speaker's own track, so a candidate is
obvious bleed only when a competing track's source score exceeds it by
`activity.bleed.min_score_margin` **and** their normalized speech-band cross-correlation
reaches `activity.bleed.min_correlation` within `activity.correlation_max_lag_ms` — with a
veto: a candidate whose band-limited level is within `activity.bleed.veto_db` of its own
track's speech reference is never suppressed, because a lav hearing its wearer at the
wearer's normal level is not hearing someone else.
It also covers **how a track's speech reference is estimated** — the 75th percentile of that
track's own candidate levels (`activity/bleed.py::REFERENCE_PERCENTILE`), rather than the
median ADR-0014 first specified. Including a track's bleed candidates drags that reference
down and the upper quartile pushes it up; which effect dominates is a property of a real
room, and the veto's usefulness depends on the answer.
**Why it matters:** Every default in `activity.vad`, `activity.bleed`, and
`activity.scoring` is a number chosen against synthetic audio whose bleed is a delayed,
attenuated copy of the same signal — which is the *easy* case. Real bleed crosses a room,
reflects, and arrives filtered, so its correlation against the source track is lower and its
level depends on where two people are sitting. Set the thresholds too aggressively and real
overlapped speech disappears, which the spec says is worse than extra ASR compute; set them
too leniently and every utterance is transcribed six times.
**Evidence:** Measured, on a real recording: the distribution of band-limited level
difference between a speaker's own lav and the others during solo speech, and the
distribution of peak normalized correlation and its lag for those same intervals. The
pipeline already records exactly these numbers for every candidate pair in
`work/activity.json` and in the report, so answering this is reading one real session's
graph rather than running an experiment.
**Needs:** H2 or the first real session · **Blocks:** nothing (threshold tuning) ·
**Status:** open — **first real measurements taken, from a deliberately hard geometry**

**First real numbers (sample probe, 2026-08-02).** 47 s, two transmitters, one operator
holding one mic at a time in front of their mouth with the others roughly two feet away —
so the "bleed" is far closer and louder than anyone sitting across a table. The real Silero
release fired on real speech for the first time in this project: 14 candidates, **0
suppressed**, 5 kept by the track-level veto.

| population | band-limited level delta | peak correlation |
| --- | --- | --- |
| one mic held at the mouth, others nearby (bleed) | 1796–2201 mb (**18–22 dB**) | 814–913‰ |
| both mics hearing the operator similarly | 94–149 mb (**~1 dB**) | 866–901‰ |

Three things follow, and the geometry makes all three conservative — a real table can only
separate further:

- **The two populations are an order of magnitude apart in level**, which is the separation
  this entry exists to establish. `min_score_margin` has a great deal of room.
- **Correlation does not discriminate.** It is 814–913‰ for bleed and 866–901‰ for genuine
  co-incidence — *overlapping ranges*. Correlation confirms two tracks heard the same room;
  the **level delta** is what says which lav belongs to the speaker. A gate leaning on
  correlation alone would be reading noise, and ADR-0014's insistence on margin **and**
  correlation **and** a veto is what stops that.
- **Peak lag was 115–174 derivative samples (7.2–10.9 ms)** — and at two feet, air accounts
  for under 2 ms. Most of it is device or clock offset, consistent with OQ-004's 33 ms
  quantization. **A zero-lag correlator would have found none of this**, which retroactively
  justifies M3's lag-tolerant one. It also means two tracks summed during genuine overlap
  are ~9 ms apart, which is **M5's** problem, not M3's.

The veto kept 5 bleed copies rather than suppressing them — the conservative direction the
spec asks for, with M4's post-ASR duplicate collapse as the thing that actually removes
them once text confirms. That handoff behaving as designed on real audio is itself a result.
**Still needed:** a real table, real spacing, more than one pair of speakers, and long
enough to see a distribution rather than 14 candidates.

## OQ-018 — What do Qwen3-ASR and its aligner need at a request boundary?
**Assumption:** Four guesses, each of which M4 has to make a number out of before any model
exists to check it against:

1. **Padding.** `transcript.pad_ms` of audio on each side of an ownership interval is enough
   context for the model to recover the first and last word intact. A word clipped by the
   segment boundary is the failure this pays for.
2. **Timestamp stability.** Two overlapping requests covering the same audio return that
   audio's words at *close enough* times that a duplicate at a stitch boundary can be
   recognized by text equality plus interval overlap. If the model's times wander by more
   than a word's length between requests, the stitch rule stops recognizing the duplicate and
   emits it twice.
3. **Truncation.** A response that stopped at the generation ceiling is worth retrying as two
   halves split at the quietest interior point, and `transcript.max_truncation_retries`
   additional submissions are enough to resolve one. Both halves of that are guesses: that a
   low-energy point is a better split than the midpoint at all, and that the recursion
   terminates in practice rather than at the budget.
4. **Duplicate text.** `transcript.duplicate.min_text_similarity` and the minimum-length
   floors below which similarity is ignored are calibrated against *Qwen's* error
   distribution — how differently it transcribes the same utterance heard on two lavs. Set
   too high, real duplicates survive; too low, two people who happened to say the same thing
   are collapsed into one.

**Why it matters:** Every one of these is a property of a model this milestone deliberately
does not have. They shape what is submitted, what is stitched, and what is deleted — and (4)
is the one that can silently destroy speech, which the spec says is the worse failure. M4 is
correct under whatever the configured values are; nothing here changes that. What is unknown
is whether the *defaults* are sensible.
**Evidence:** M6b's smoke test against the real adapter answers (1), (2) and (3) directly:
submit overlapping requests over the same audio and compare returned word times; force a
truncated response and observe both the finish signal and whether a split resolves it. (4)
needs a real session, or at minimum a real recording of one utterance heard on two
transmitters — the same evidence OQ-017 waits on, from the text side rather than the acoustic
one.
**Needs:** H1/H2 or the first real session · **Blocks:** nothing — M4 is correct under the
configured values · **Status:** **items 1–3 answered** (M6b, 2026-08-03, measured against the
real model); **item 4 open**, and so is whether a low-energy split beats a midpoint — both
need speech this capture does not contain

**Raised before the constants landed, by M4's plan review.** The plan promised that every
default would cite an open question and then had none to cite for any of these: OQ-009 covers
only the package's segment limit and OQ-017 only the acoustic side of bleed. Every
request-shaping and text-similarity default in `TranscriptConfig` cites this entry, so
`rg 'OQ-018'` finds all of them at once.

---

**Items (1)–(3) measured against the real model** (M6b, 2026-08-03). 47 s of real speech from
the sample probe, on `gfx1151` in bfloat16; `tests/test_qwen_smoke.py` is the instrument and
prints every number below. Item (4) is untouched and still needs a real session.

**(2) Timestamp stability — answered, and the rule's hit rate is 96%.** Two requests over the
same audio, offset by four seconds so they share twelve. The measurement counts every word of
the shared span whose text appears in both — the duplicates M4's stitch rule *ought* to
recognize — and then reports how many of them it actually paired, which needs both the same
`comparison_key` and an overlap in time. Measured 2026-08-03, all four recordings:

| recording | paired / shared | worst disagreement among the paired |
| --- | --- | --- |
| TX01 (`pcm_s24le`, receiver A) | 20 / 22 | **0 ms** |
| TX02 (`pcm_s24le`, receiver A) | 21 / 22 | **80 ms** |
| TX03 (`pcm_f32le`, receiver B) | 18 / 18 | **400 ms** |
| TX04 (`pcm_f32le`, receiver B) | 18 / 18 | **320 ms** |

77 of 80, across two receivers and two sample formats, which is what makes this a property of
the model rather than of one file. The failure this item was raised against — times wandering
by more than a word's length, so the duplicate is emitted twice — is not happening at a rate
that threatens the rule.

**The three misses are worth naming, because two of them are the measurement's fault and one
is real.** Both `'transmitter'` misses have a nearest counterpart 1.1–1.2 s away: the recording
says the word repeatedly, so these are almost certainly *different* occurrences that the
text-keyed candidate count could not tell apart — the same repeated-word confound described
below, now confined to the denominator instead of the deltas. The real one is TX01's `'a'`,
whose two placements sit exactly 80 ms apart: one step of the aligner's `timestamp_segment_time`
quantization, which for a word shorter than one step is enough to stop it overlapping itself.
So the residual risk is specifically **very short words at a stitch boundary**, and the cost of
one is that a single word appears twice — visible, harmless, and not worth machinery.

*Two notes on how these numbers were obtained, because the first two attempts were both
wrong, in opposite directions.* Matching shared words by text alone reported five outliers of
2–9 seconds — the test's fault, since a text-only key paired the first `"testing"` in one
window with the second in the other. Pairing the way the rule under test pairs fixed that and
introduced the opposite error: selecting only words that already overlap and then reporting how
closely they agree measures nothing, because a word that drifted far enough to stop overlapping
— which is precisely the failure under investigation — simply leaves the sample. "20 paired,
worst 0 ms" was true and would have been equally true of a model that got half of them badly
wrong. Codex's code review caught it. **A measurement of a rule needs the rule's own notion of
"the same thing" for its numerator and something independent of the rule for its denominator.**

**(3) Truncation — answered.** With `max_new_tokens` forced to 8, the model returned
`'Testing a first transmitter. Hello, one'` — visibly cut off mid-utterance — and the
retokenized-length heuristic flagged it. At the default 1024 the same audio is not flagged, so
the heuristic is not trivially satisfiable. That is the whole of truncation detection, because
0.0.6 exposes no finish reason (**ADR-0028**). Whether a low-energy split *resolves* a
truncation better than the midpoint is still unmeasured: an eight-token ceiling truncates
everything, and a natural truncation needs an utterance long enough to exhaust 1024 tokens,
which 47 seconds of one person testing microphones does not contain.

**(1) Padding — answered, and it found something the question was not asking about.**
The padding does its job: submitted a window padded by `transcript.pad_ms` and the same speech
clipped hard at its first and last aligned words, the model returned *identical text* both
times. `pad_ms` = 500 is not the constraint.

**What is** is the ownership boundary, and it is costing real words. On the real end-to-end
run, the model heard `'Testing a first transmitter. Hello. One two three. Here we go.'` and the
transcript recorded `'a first transmitter Hello One two three Here we go'`. The aligner places
"Testing" at 10.480 s; the VAD candidate's ownership interval begins at 10.530 s; M4's rule —
a word belongs to the interval containing its **start** — correctly drops it. Five of eleven
retained segments lost their opening word this way, always the word the utterance starts with.

Two things follow, and neither is M6b's to act on:

- **It is a threshold, not a defect.** `activity.vad.pad_ms` is 30 ms, and a stop consonant
  like the /t/ in "Testing" is exactly the onset a VAD is late on. M4's ownership rule and the
  transcript padding are both behaving as designed. The number that is wrong is M3's, it is
  registered here and under **OQ-017** as a value chosen against synthetic audio, and it is
  the kind of thing only a real model on real speech could have revealed.
- **This recording is not enough to retune on.** It is one operator holding one mic at a time
  — the deliberately hard geometry OQ-017 already describes — and moving a detection threshold
  on 47 seconds of it would be over-fitting to a microphone test. A real table is what should
  move it, and when it does, the symptom to look for is a transcript quietly missing the first
  word of an utterance rather than anything that raises.

## OQ-019 — What do the automix constants need to be at a real table?
**Assumption:** Six numbers, each of which M5 has to choose before anyone has heard a mix:

1. **`mix.envelope.attack_ms` = 10 and `release_ms` = 300.** Short attack so a word is not
   clipped, longer release so a channel change does not click or pump. 10 ms is a third of
   `activity.vad.pad_ms`, so the ramp completes inside the padding the candidate already
   carries and the channel is open before the word starts — a relationship a test pins,
   because raising the attack past the pad is how a first phoneme is lost.
2. **`room_tone_share` = 0.005 and `min_active_share` = 0.5.** The two floors that make the
   gate's dominance criterion provable rather than incidental: an inactive channel never
   falls below the first, an active one never below the second, so worst-case solo dominance
   is `20·log10(0.5/0.005) = 40 dB` before the correction clamp erodes it. Their *ratio* is
   what matters; their absolute values do not, because during silence every weight is equal
   and the shares are `1/N` regardless.
3. **`max_level_correction_db` = 6.** The spec says "clamp correction to a safe range" and
   names no number. 6 dB is deliberately conservative: it costs 12 dB of the dominance
   margin (a quiet track lifted while a loud one is cut), and a wearer whose lav is more than
   6 dB out is a capture problem a mixer should not paper over.
4. **`solo_attenuation_margin_db` = 20 and `overlap_min_gain_db` = −16.** The gate's
   "configured attenuation margin" and "nontrivial audible gain". **Both** are validated to be
   *achievable* from (2) and (3) at configuration load, and both are asserted against the
   applied coefficient, correction included. The second started at −15 on the estimate "two
   channels share roughly −6 dB each, and the clamp can take another 6"; that is 0.66 dB
   optimistic, because the quieter of two speakers holds `min_active_share` while the louder
   holds full weight and four room-tone floors still take a share. The real bound for six
   tracks is −15.66 dB, and `EnvelopeConfig.guaranteed_overlap_gain_db` computes it from the
   roster rather than from a worked example. Corrected in M5's verify phase.

**Why it matters:** Every one is a number chosen against a 10.5-second synthetic fixture
whose speech is shaped noise and whose bleed is a delayed attenuated copy of its source
(INV-10). None of them can make the mix *wrong* — the bounded-gain invariant holds for any
admissible values and the config validator refuses an unachievable combination — but they
decide whether the result is pleasant to listen to, which no test can assert.
**Evidence:** The first real session's mix, listened to, against the graph that produced it.
The pipeline already records each track's correction and every measurement in the report, so
answering this is reading one report and one MP3 rather than running an experiment.
**Needs:** H2 or the first real session · **Blocks:** nothing — the mix is correct under
whatever the configured values are · **Status:** open

## OQ-020 — What does a real 128 kbps mono MP3 encode actually do to peak and duration?
**Assumption:** Two guesses that shape the encode/verify loop:

1. **True-peak overshoot is small enough that `mix.encode.max_retries` = 3 resolves it**, and
   that pre-emptively targeting the ceiling from the intermediate's own measured true peak
   makes the first encode compliant in the ordinary case. Lossy encoding provably introduces
   overshoot; how much, at this bitrate, on this material, is not known.
2. **The decoded duration lands within one MP3 frame** (`duration_tolerance_frames` = 1;
   1152 samples at 48 kHz is 24 ms). LAME writes encoder delay and padding into its Xing/LAME
   header and FFmpeg's decoder trims accordingly, so the count should be exact — but "should"
   is the word this entry exists for, and the tolerance is the spec's own "or another
   documented codec-appropriate tolerance".

Also here: `true_peak_tolerance_db` = 0.3, the "documented measurement tolerance" the gate
asks for. FFmpeg's `ebur128` summary reports one decimal place, so 0.1 dB of it is pure
quantization; the rest is margin for a measurement taken on a decode rather than on the
encoder's own model.
3. **An overshoot large enough to matter is smaller than `loudness_tolerance_lu`.** A retry
   reduces the master gain by exactly the overshoot, so on a run that *was* aiming at the
   loudness target an overshoot above 1 dB makes the retry land outside the loudness tolerance
   and fail the stage rather than resolve it. The ceiling-limited case is unaffected, because
   there the loudness comparison is waived (ADR-0023). Noticed in M5's verify phase; no
   evidence either way, and the fix if it bites is a larger pre-encode ceiling margin rather
   than a wider tolerance.

**Why it matters:** Set the retry budget too low and a compliant mix is reported as a failed
stage; set the tolerances too loose and the pipeline claims a compliance it did not verify,
which the spec forbids in as many words. The failure direction is safe either way — the stage
**fails** rather than asserting compliance it cannot demonstrate.
**Evidence:** The measurements are already retained in the report for every attempt, so a
first real session answers both halves by being encoded once. A 10.5-second fixture answers
neither: overshoot is a property of material with real dynamics over a real duration.
**Needs:** H2 or the first real session · **Blocks:** nothing · **Status:** open

## OQ-023 — Does the receiver's displayed timecode reach `bext.time_reference` in the WAV?
**Assumption:** Yes. The receiver holds one timecode, shows it on the display, and writes
that same value into every file its transmitters record — so jamming two receivers to a
common count makes their files mutually placeable.
**Why it matters:** **This is the assumption the whole cross-receiver synchronization
strategy rests on, and it is the one nobody has checked.** The operator can connect
L-OUT → L-IN, press SYNC, and watch two displays align (**OQ-012**) — but the pipeline never
sees a display. It reads `bext.time_reference` through FFprobe. If the displayed timecode and
the written reference have different origins, then a jam that visibly succeeds produces files
that are no more alignable than unjammed ones, and the failure is invisible at capture time
*and* at ingest time. That is the worst shape a failure can have here: a procedure the
operator correctly believes worked.

There is a specific reason to doubt it rather than merely to check it. DJI's documentation
describes timecode as "a frame counter relative to recording duration" that "resets to zero
and restarts", with no facility for the user to set its value — while the 2026-08-02 sample
files carry references implying **per-receiver power-on epochs about fifty seconds apart**
(**OQ-004**). Both are consistent with a device-local counter. Neither tells us what a jam
does to the number that gets written.

**Evidence:** Jam the receivers, confirm the displays match, then record ten seconds on both
simultaneously and run `dnd-audio inspect`. The discriminator is whether the two receivers'
files now imply the **same** epoch. Do this before a real session, not during one — it costs
five minutes and it decides whether the capture is usable.

Three outcomes worth telling apart:

1. **Epochs match.** The jam propagates. Cross-receiver alignment is solved to one frame
   (16.7 ms at 60 fps — see **OQ-004**), and `session.sync_qa` refines from there.
2. **Epochs still differ by the pre-jam amount.** The written reference ignores the jam
   entirely and is a device-local counter. Cross-receiver alignment must then come from the
   audio — `session.sync_qa` and a shared transient — and the timecode is per-receiver only.
3. **Epochs differ by something new.** The jam reaches the file partially or with an offset,
   which is the case most likely to be silently wrong in production and most deserving of a
   warning.

**A warning is wanted in every case except (1).** The pipeline can compute the implied epoch
per source at ingest and compare across receivers; disagreement is exactly the
"capture-procedure problem the pipeline should detect and warn about" OQ-012 names, and
unlike the display it is checkable after the fact, on every session, for free.
**Needs:** H1, or five minutes with the receivers · **Blocks:** nothing yet — but it decides
what OQ-004's rework has to achieve · **Status:** **answered — outcome (1), the jam
propagates** (jam verification capture, 2026-08-03)

**Answer — yes, and to within one frame.** Full evidence in
`docs/fixtures/2026-08-03-jam-verification.md`. Two receivers were jammed L-OUT → L-IN,
their displays confirmed matching, and each receiver's pair started a few seconds after the
other. Cross-correlating the audio — which is the only arbiter, because metadata is
self-consistent under both readings — measures the true offset between the two receivers'
recordings at **5.28 s**. `bext.time_reference` alone predicts 5.267–5.300 s. All four
cross-receiver pairs agree to **17–30 ms**, inside the 33.3 ms frame quantum; the worst pair
overall is 47 ms.

The correlator never saw the metadata. Four independent pairs landing within 30 ms of a
metadata-only prediction across a ±47 s search range is not chance.

**What this settles.** Cross-receiver alignment is solved to one frame with no audio
processing at all, which is what OQ-004's reframe needed and what OQ-012 could not establish
from the displays. **The absolute value is still meaningless** — `time_reference` here is
~284 s, nowhere near the `01:51:2X` the displays showed before the capture — but the *origin*
is shared, and `session_position` is a subtraction, so only the origin was ever required.

**The warning is still wanted, and is now the highest-value piece of work left.** A failed
jam produces files that look perfectly normal; the 2026-08-02 probe is proof, and nothing at
capture time or ingest time flags it. The check is exactly the measurement above: correlate a
shared transient across tracks and compare the measured offset to the timecode prediction.
`session.sync_qa` already correlates; what it does not do is compare its result against
`time_reference` and fail loudly when they disagree. That converts an operator ritual whose
outcome is invisible into a pipeline assertion. See **OQ-025** for whether a deliberate
acoustic sync signal should feed it.

## OQ-024 — Does the receiver's timecode frame-rate setting reach the transmitter's file?
**Assumption:** Yes. The receiver's configured rate is written to `iXML TIMECODE_RATE` and
sets the quantization of `bext.time_reference`, so choosing a finer rate buys finer
cross-track resolution.
**Why it matters:** **OQ-004** concluded from DJI's documentation that 50 and 60 fps are
supported and that 60 fps would halve the quantum from 1600 samples (33.3 ms) to 800
(16.7 ms), and H1's recipe was amended to require 60 fps on all three receivers. If the
setting does not reach the file, that instruction is an unverified ritual and 33.3 ms is the
floor.
**Evidence:** Record simultaneously on two receivers set to different rates and diff the
written metadata.
**Needs:** nothing further for `orig` · **Blocks:** nothing · **Status:** **answered — no,
not for `orig` files** (jam verification capture, 2026-08-03)

**Answer — the setting changes nothing in the file.** One receiver was set to 60 fps and the
other to 30. The two groups' `bext` chunks differ in **exactly five bytes** —
`origination_time` and `time_reference` — and every rate field is byte-identical across all
four files: `TIMECODE_RATE 30/1`, `MASTER_SPEED 30/1`, `CURRENT_SPEED 30/1`,
`TIMECODE_FLAG NDF`. Every `time_reference` is an exact multiple of **1600 samples**. No file
shows finer resolution. (Testing for "exact at 60F" is vacuous: anything divisible by 1600 is
divisible by 800.)

**Consequences.** OQ-004's 60 fps recommendation is retracted, H1's recipe no longer asks for
it, and **33.3 ms is the cross-track quantization floor on this hardware.** That is
acceptable — see the error budget in **OQ-025**.

**Scope, and why the untested half does not matter.** These are `orig` files: the
transmitter's own internal recording, which may be handed a timecode value without being told
a rate. A receiver-side `edit` file might well carry 60/1. That is untested and uninteresting
here, because `orig` is the only file this project consumes (**OQ-007**) — the entire point is
to treat each transmitter as an independent 32-bit-float recorder, immune to wireless dropouts
and receiver-side processing, and merely *placed* by timecode.

**A useful accident.** The two receivers were on different rates and **the jam held anyway**,
to within one frame across a real 5.28 s offset (**OQ-023**). The spec's owner note asks for a
consistent rate across all three kits; this capture violated that and nothing downstream
degraded. Keep the rates consistent as hygiene, but no known behaviour depends on it.

## OQ-025 — Should the capture include a deliberate acoustic sync signal?
**Assumption:** No. Jammed timecode places the tracks, and `session.sync_qa` correlates
ordinary speech well enough to verify it. H1's recipe asks for a three-clap pattern purely as
a human-checkable landmark, not as the alignment mechanism.
**Why it matters:** The LTC jam is accurate — 17–30 ms cross-receiver, better than one frame
(**OQ-023**) — but it is a tedious manual ritual at the start of every session: cable up
A → B, SYNC, disconnect, A → C, SYNC, disconnect, and confirm three displays. An acoustic
alignment signal recorded once at the top of the session would be detectable automatically
and would need no cables. The question is whether it could **replace** the jam, or only
**verify** it.
**Evidence:** Measured alignment accuracy achievable from an acoustic signal alone, against
the 33 ms the jam already delivers; and whether a single anchor is sufficient given measured
drift.
**Needs:** a bench test; no session required · **Blocks:** nothing — the jam works today ·
**Status:** open

**What the error budget says.** Cross-track error is 33.3 ms of fixed quantization
(**OQ-024**) plus ~15–45 ms of drift over a long session (**OQ-006**) — call it 80 ms worst
case. Against the consumers: speaker attribution compares energy across tracks where syllables
are 150–250 ms; duplicate collapse already sees 16–48 ms of spread from lav geometry alone;
transcript ordering is indifferent. The one place tighter alignment would matter is any mix
stage that **sums** correlated tracks, where 80 ms is an audible slap — M5's automix ducks
rather than sums, so it does not arise, and that is worth not breaking.

So the honest position is that **an acoustic method does not need to beat the jam; it needs to
be good enough and cheaper.** Anything reaching ~10 ms would be strictly better than what the
jam delivers.

**The distinction that decides this.** Timecode supplies an *origin per file*, including for
files with no overlap — a transmitter switched off and back on mid-session produces a fresh
file whose `time_reference` places it with no audio evidence at all. An acoustic anchor at the
top of the session cannot place that file. So an acoustic signal can replace the jam only if
every transmitter records one continuous file per session, and must otherwise supplement it.
The recipe deliberately includes a power cycle for exactly this reason (**OQ-003**).

**Signal choice, if this is pursued.** A clap is broadband and gives a sharp correlation peak,
but its onset is operator-dependent and its level clips easily on a lav at close range. The
alternatives worth bench-testing, in rough order of promise:

1. **A linear chirp** (e.g. 500 Hz → 8 kHz over 1 s). Matched-filter compression gives
   sub-millisecond peak resolution, it is robust to room reverberation and to the band
   limiting of a lav capsule, and it is trivially distinguishable from speech so it cannot be
   mistaken for content. This is the standard answer.
2. **A short broadband burst** (clap, slate, or a synthesized click train). Simple, no
   playback equipment, but a single click's peak is smeared by reverberation.
3. **A pure sine tone.** Poor — a continuous tone has no time structure, so its correlation
   peak is ambiguous by whole cycles. Only useful for measuring *drift* between two long
   recordings, not for establishing an origin.

**The confound that limits all of them.** Every method above measures *acoustic arrival*, not
recording time. Six lavs at a table are 0.5–3 m from any one sound source, which is
1.5–9 ms of propagation spread, and it is not a constant — it depends where each person is
sitting. So an acoustic anchor has a floor of several milliseconds that the jam does not have,
and driving it below that would require a per-transmitter electrical injection rather than a
sound in the room.

**Recommendation as of 2026-08-03: keep the jam, and spend the effort on the verifier
instead.** The jam is already better than the error budget requires, and the real gap is not
accuracy — it is that a *failed* jam is invisible (**OQ-023**). A chirp would make automatic
verification robust and cheap, and that is worth doing on its own merits even while the jam
remains the alignment mechanism. Revisit replacing the jam only if H2 shows the ritual failing
in practice, or if the power-cycle case turns out never to occur.

## OQ-026 — Does a DJI receiver's timecode counter wrap, and with what period?
**Assumption:** Yes, at 24 hours — `rasterize.SECONDS_PER_DAY` adds `86400 * sample_rate`
whole samples to unwrap a `bwf_sample_reference`, and `cycle_units` reports that as the
evidence's cycle.
**Why it matters:** Only for a session whose sources' references appear to wrap. Until M8 this
assumption was implied by a *different* one — that a BWF reference counts samples since real
midnight, which **OQ-004** disproved. "Device-local counter" does not imply a 24-hour period,
so the wrap arithmetic lost the reason that used to make it obvious and now stands on its own.

DJI's documentation describes timecode as "a frame counter relative to recording duration"
that "resets to zero and restarts". A counter with that description might roll at 24 hours
like SMPTE timecode, at some device-specific width, or never within a battery's life. Nothing
observed so far distinguishes them: the four jam-capture references sit at ~284 s and ~290 s,
six orders of magnitude from any plausible wrap.

**Why it is not fixed by removing the wrap.** The inference is spec-required and tested
(`rollover_session`, M2's completion gate), and a recorder whose reference genuinely *is*
midnight-relative — which the BWF standard specifies — needs it. Deleting a tested capability
on a hypothesis about one vendor is the larger risk. What was missing was the registration,
which is this entry. INV-12 keeps it safe meanwhile: the inference warns
(`midnight_rollover_inferred`), refuses a tie rather than guessing, and a real DJI session
never reaches it, because its counters are a few hundred seconds apart and nothing appears to
wrap.
**Evidence:** A receiver left running long enough for its counter to approach 24 hours, or
vendor documentation stating the counter's width. Either settles it; neither is urgent.
**Needs:** H2, or DJI documentation · **Blocks:** nothing · **Status:** open

**Raised by M8's plan review** (`reviews/M8-plan-20260803-1729.md`, finding 2), which noticed
that OQ-004's answer had quietly orphaned an assumption nobody had written down. The reviewer
argued for removing the wrap; the narrower remedy — register it and cite it — is recorded there
with the reason.
