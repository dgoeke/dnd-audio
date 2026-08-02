# ADR-0011 — The working-audio path: a virtual track, read directly, decimated once

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M2

## Context

The spec asks for "a lossless 48 kHz floating-point working path for the mix" and "cached
16 kHz mono working audio for VAD and ASR", and immediately constrains how:

> "Virtual track" does not require loading or materializing a session-length NumPy array.
> Prefer a segment map plus streamed/windowed reads and writes. If a contiguous
> floating-point WAV intermediate is useful, use RF64.

INV-07 is the same sentence as an invariant, on a UMA host where memory pressure kills
processes. Four hours × 48 kHz × float32 × six transmitters is roughly 16 GB of audio that
must never be resident and, by the same arithmetic, is expensive to write down at all.

Three decisions follow from that, and one from independent review
(`docs/plan/reviews/M2-plan-20260802-1241.md`), which noted that the reader needs *random*
access — a mix pass asks for a window, not a stream from the beginning — and that the
first draft's plan to accept integer PCM rested on a false claim.

## Decision

### The segment map is the working path; contiguous 48 kHz files are cache

`work/timeline.json` is authoritative. A `TrackReader` answers "give me
`[start, start + n)` of this track" by reading the underlying sources and returning
silence for gaps, in bounded windows. Nothing materializes a session-length array, and
nothing needs to.

`ingest --materialize-48k` writes contiguous float32 **RF64** files for debugging,
interoperability, and performance investigation. They are **disposable content-addressed
cache artifacts, not pipeline truth**: they live under `work/cache/`, they carry the same
cache identity discipline as the 16 kHz derivatives, and deleting them costs a rebuild and
nothing else. Every test of aligned duration and sample position reads through the virtual
reader, so no test can come to depend on their existence.

### Sources are read directly, not through an FFmpeg pipe

M1's RIFF walk already records each source's `data` chunk offset and size. For mono PCM
that makes sample *i* a seek to a computed byte offset, which is what windowed random
access requires; a subprocess per window is not that, and a subprocess per *track* is a
stream, not a reader.

The supported input is the one the session contract specifies: **mono 32-bit float
RIFF/RF64**, which is DJI dual-file mode's `orig`. Any other encoding is the spec's "a
source file cannot be decoded" — fatal, naming the codec and the file. Integer PCM support
arrives when a real recovery need justifies it, not speculatively; in particular
`pcm_s32le` cannot be converted to float32 exactly (float32 has 24 mantissa bits;
`2147483647` becomes `2147483648.0`), so a "lossless integer path" is not available to be
built anyway.

### One canonical 3:1 decimator, applied across the whole virtual track

The 16 kHz derivative is produced by a single fixed linear-phase FIR whose coefficients and
design metadata are **checked in** (`timeline/data/fir_48k_16k.json`), driven through
`scipy.signal.upfirdn(up=1, down=3)` behind a thin wrapper that carries filter state and
decimation phase across window boundaries. Never by taking every third sample.

The design is held to a **declared frequency response**, not merely to producing the
expected number of samples: passband edge 7000 Hz with ≤ 0.1 dB ripple, stopband beginning
at 8000 Hz — 16 kHz's Nyquist, so everything above it aliases — with ≥ 80 dB attenuation,
exact coefficient symmetry, and unity DC gain. Length 259 with group delay 129 samples at
48 kHz, chosen because 129 divides by 3: the delay is then a whole number of samples in
**both** grids (43 at 16 kHz), which is what makes the mapping exact rather than
approximate.

The mapping is `sample16 = sample48 // 3` after delay compensation, output length is
`ceil(n48 / 3)` with the tail zero-padded, and **the filter runs across the entire virtual
track without resetting at a chunk or gap boundary** — a reset would put a transient at
every boundary and make the derivative depend on how the source happened to be split.

FFmpeg and SoX may be used as QA comparisons. They are not the canonical derivative: their
delay and boundary behaviour are less explicit and more version-dependent, and the
derivative is a cached artifact whose identity must not move when a tool is upgraded for
unrelated reasons.

## Alternatives considered

- **Decode each source through `ffmpeg -f f32le -`.** Handles every codec and makes "cannot
  be decoded" someone else's problem. Rejected: it is a stream, not a reader, so windowed
  random access means either one subprocess per window or buffering the track — and the
  latter is the INV-07 violation this whole design exists to avoid.
- **Always materialize the 48 kHz files.** Simplest for downstream code, and about 16 GB
  written on every run of a four-hour session, on the host whose free-space warning
  threshold this milestone is supposed to *reduce*.
- **`scipy.signal.decimate` or `resample_poly`.** Convenient one-shot calls with no
  documented way to carry state across windows, so a streamed implementation would have to
  reimplement the boundary handling anyway — with the filter design now implicit and
  version-dependent.
- **Design the filter at import time.** Rejected: a SciPy upgrade would then silently
  change what every cached derivative in every session was built with. Checked-in
  coefficients make that a commit.
- **Skip the anti-aliasing filter and take every third sample.** Fast, and it folds
  everything from 8 kHz to 24 kHz back into the speech band, corrupting exactly the signal
  the VAD and the ASR are about to read.
- **A 16-bit integer derivative**, halving its size. Rejected for now: it is a lossy
  choice made on behalf of two milestones that have not stated their needs, and the
  derivative is regenerable, so the disk it saves is the cheapest disk in the pipeline.

## Consequences

- The 16 kHz derivative's cache identity has to carry the FIR identity, the SciPy and
  NumPy versions, the PCM-reading and timeline semantics versions, and the track's segment
  map — a parser fix that moves timing evidence must not serve an aligned derivative at
  its old position (INV-08).
- Because the filter never resets, a derivative cannot be built per chunk and concatenated.
  That is deliberate and is tested: streamed output must be byte-identical to a one-shot
  run at every window partitioning.
- `doctor`'s free-space arithmetic (OQ-013) assumed the 48 kHz working audio was always
  written. It is not, by default, so this milestone's preflight replaces that estimate with
  one derived from the session's actual length and the artifacts actually requested.
- A non-float32 source is now fatal in M2 where M1 only warned. The manifest still
  describes it fully, so the diagnostic explains itself.
