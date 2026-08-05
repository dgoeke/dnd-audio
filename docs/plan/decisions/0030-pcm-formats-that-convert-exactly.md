# ADR-0030 — The working path accepts a format when the conversion is exact

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M8
**Amends:** [ADR-0011](0011-the-working-audio-path.md)

## Context

ADR-0011 restricted the working path to mono 32-bit float — DJI dual-file mode's `orig` — and
gave a reason:

> in particular `pcm_s32le` cannot be converted to float32 exactly (float32 has 24 mantissa
> bits; `2147483647` becomes `2147483648.0`), so a "lossless integer path" is not available to
> be built anyway.

The reason is correct. The implementation generalized it into a refusal of **every** integer
format, and `timeline/layout.py` says so in the operator's own diagnostic:

> `undecodable_source: … is 1-channel pcm_s24le, and the working path reads mono pcm_f32le —
> dual-file mode's 'orig'. An integer format cannot be converted to float32 exactly, so it is
> refused rather than quietly rounded (ADR-0011).`

**That sentence is false for 24-bit.** A signed 24-bit integer's whole range fits in float32's
24-bit significand, and the scaling divisor is a power of two, so `s24 → f32` is lossless.
Verified empirically rather than argued (**OQ-007**): 2 000 000 random values plus the range
edges round-trip with zero error, and the identical test on 32-bit integers fails.

This is not hypothetical. Two of four transmitters in the 2026-08-02 probe wrote `pcm_s24le`
`_orig` files, from a per-transmitter setting the operator had not matched across kits — the
exact mistake the capture guide warns about, which means it will be made again. **It is the
item on M8's list that can cost a whole session**, and it would be discovered after the
recording rather than during it.

## Decision

### The rule is a principle, in two parts, not an allowlist

The working path accepts a source when **both** hold:

1. its sample format is a **signed little-endian integer or IEEE float** — a family this
   reader decodes without a convention to guess at; and
2. the conversion to float32 is **exact** — the format carries at most 24 significant bits, and
   the scaling divisor `2**(bits-1)` is a power of two.

Today that is `pcm_f32le`, `pcm_s24le` and `pcm_s16le`. Each refusal now names the half that
failed:

- **`pcm_s32le`** fails (2). Its diagnostic keeps ADR-0011's original sentence, which was
  always true of it.
- **`u8`** fails (1). WAV 8-bit PCM is *unsigned with an offset of 128*, so it converts exactly
  but through a different convention. Refused as untested rather than as unrepresentable,
  because those are different facts and the whole point of this ADR is that a refusal must give
  the reason that is true.

A principle rather than "add `pcm_s24le`" because an allowlist of two would have left
`pcm_s16le` refused with a sentence that is false for it — the same defect, one format over.

### The reader carries the format; nothing else changes

`PcmSource` gains its sample format and `bytes_per_sample`, and `open_pcm`'s validation, the
sample count, and `PcmReader.read`'s seek arithmetic derive from it instead of a module
constant. `BYTES_PER_SAMPLE` survives as the float32 case for `wavwrite`, `preflight` and
`loudness`, which write and measure float32 only.

Integer decode is **windowed like everything else** and unpacks into an array the size of the
requested window. NumPy has no packed 24-bit dtype, so the unpacking step is exactly where an
implementation accidentally expands a whole source; INV-07 is therefore proved over the
composed path in `tests/test_memory.py`'s ordered event log, not by a helper assertion. Raised
by M8's plan review (`../reviews/M8-plan-20260803-1729.md`, finding 7).

### The spec moves with it

`dnd-audio-ingestion-agent-spec.md` says `orig` input is 32-bit float. Real hardware writes
24-bit under an ordinary settings mismatch, so the spec is amended in the same commit, per
AGENTS.md.

## Alternatives considered

- **Accept `pcm_s24le` only.** The narrowest change, and what the completion gate literally
  asks for. Rejected by the operator in favour of the principle: it leaves `pcm_s16le` refused
  with a reason that is false for it, which is this ADR's own defect preserved.
- **Accept every PCM format and round where necessary.** Directly contradicts ADR-0011's
  principle, which stands: never quietly round a lossless path.
- **Convert 24-bit sources to float32 once, on ingest, and cache the result.** Turns an exact
  read into a derived artifact with its own cache identity, for a conversion cheap enough to do
  per window. It would also write a copy of the session's audio to disk, which is what
  ADR-0011's segment map exists to avoid.
- **Decode through FFmpeg**, which handles every format. Rejected in ADR-0011 for the reason
  that still holds: a pipe is a stream, and windowed random access over a stream is either one
  subprocess per window or a buffered track.
- **Scale by `2**(bits-1) - 1`** so that the most negative sample maps to exactly −1.0.
  Rejected: it is not a power of two, so the division stops being exact and the format
  stops satisfying part (2) of the rule it is being admitted under. `2**(bits-1)` is also what
  libsndfile and FFmpeg use, and the reader is cross-checked against FFmpeg's own decode.

## Consequences

- A mixed-format session — some transmitters `f32`, some `s24` — now ingests, which is the
  capture mistake this exists for. Nothing downstream sees the difference: the working path
  hands out float32 windows either way.
- `pcm_s16le` becomes reachable without any hardware that produces it. It is covered by the
  same parametrized bit-exactness test as 24-bit, so it is tested rather than merely permitted.
- **OQ-007's other half is untouched.** Whether `orig`/`edit` pairs appear as assumed still
  needs a fixture recorded in dual-file mode; the sample captures contain no `edit` files at
  all.
- The manifest already records `codec_name`, `bits_per_sample` and the exact sample count from
  the RIFF `data` size over the block alignment, so nothing new has to be probed — M1's
  arithmetic was format-general before there was a format to use it on.
