# ADR-0023 — Loudness measurement is FFmpeg's, and the intermediate is unity gain

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M5

## Context

The spec asks for "final two-pass loudness normalization toward `-16 LUFS` integrated by
default", a `-1.5 dBTP` true-peak ceiling that "applies to the decoded final MP3 deliverable,
not merely the lossless pre-encode intermediate", and then:

> Encode a 128 kbps mono MP3 with metadata containing the session ID/title, decode it, and
> measure integrated loudness and true peak. Because lossy encoding can introduce peak
> overshoot, reduce the pre-encode gain or true-peak target and re-encode from the lossless
> intermediate when necessary. Bound the retry count, retain all measurements in the report,
> and fail the mix stage rather than claim compliance if the decoded MP3 remains outside
> configured tolerances.

It also asks for a "lossless mix intermediate in `work/` for debugging/cache reuse".

Two questions follow. Who measures loudness — and where does the master gain live, given that
the intermediate is supposed to be reusable and a retry is supposed to "re-encode from the
lossless intermediate"?

ADR-0011 is the awkward precedent: it rejected FFmpeg for the canonical 16 kHz derivative,
partly because "their delay and boundary behaviour are less explicit and more
version-dependent, and the derivative is a cached artifact whose identity must not move when
a tool is upgraded for unrelated reasons".

## Decision

### The intermediate is written at unity master gain; the master gain is an encode parameter

`work/cache/mix/<key>.wav` holds the mono float32 mix at unity — the envelope and the
per-track corrections applied, and nothing else. The two-pass loudness gain and any
subsequent true-peak reduction are applied by the **encoder** (`-af volume=…dB`).

This is what makes the spec's own sentences cheap. "Re-encode from the lossless intermediate"
costs one encode rather than one re-mix of six four-hour tracks. Changing
`mix.integrated_lufs` reuses the intermediate outright. And the retry loop has exactly one
variable in it.

It also decides the cache scope: **the render identity carries `mix.envelope` and nothing
else from the mix section.** The loudness target, the bitrate, the tolerances and the retry
budget sit after the render boundary and reach only the MP3, which is regenerated on every
run and never cached. This is the same split, for the same reason, that ADR-0016 makes
between `activity.vad` and `activity.bleed`: tuning a threshold must not rebuild gigabytes of
PCM that provably cannot depend on it.

### FFmpeg measures; the version is recorded; one decode serves everything

Integrated loudness and true peak come from `ffmpeg -af ebur128=peak=true`, parsed from its
`Summary:` block. Decoded duration is the **decoded sample count**, not a container field:

```
ffmpeg -i session.mp3 -af ebur128=peak=true -f f32le -
```

puts the R128 summary on stderr and the decoded samples on stdout. The samples are counted in
bounded chunks and discarded (INV-07), a clean exit status is required, and that exact integer
count is what `duration_tolerance_frames` is applied to. Taking duration from `ffprobe`
instead was the first draft's plan and is wrong for a reason the plan review named: a
container or header duration can stay entirely plausible while decoding yields fewer samples,
and the gate says *decoded*.

Every command is recorded verbatim in the report, as the spec's observability section
requires ("the exact commands/parameters used for FFmpeg outputs"), and the FFmpeg version
enters provenance and the render cache is *not* keyed on it — the intermediate is produced by
this project's own code and FFmpeg never touches it.

**This is not a reversal of ADR-0011.** That decision is about a cached artifact whose bytes
every downstream stage depends on. A measurement is not that: it is read, acted on once, and
recorded alongside the tool that produced it. The alternative — implementing ITU-R BS.1770-4
K-weighting and 4× true-peak oversampling in NumPy — is a real option and is rejected below.

### Three guards, because a normalizer with no floor is a hazard

- The master gain is clamped to `mix.encode.max_master_gain_db` (`mix_master_gain_clamped`).
- A mix whose integrated loudness measures below `mix.encode.silence_floor_lufs` is left
  **un-normalized**, with a warning, rather than amplified toward the target
  (`mix_not_normalized`).
- **Where the true-peak ceiling forbids the gain the target wants, the ceiling wins**, and the
  MP3 lands quieter than asked for (`mix_loudness_target_unreachable`).

The second is not hypothetical. The canonical fixture through the real Silero release yields
zero candidates (M3's closeout says so, and it is the correct answer for synthetic noise), so
every track sits at the room-tone share and the mix is a quiet blend. Normalizing that to
−16 LUFS means roughly 50 dB of gain on six noise floors. The guard makes the outcome a
warning about a session with no detected speech, which is what it is.

The third is the one the canonical fixture actually reaches: peaky material 31 dB down wants
+15.6 dB and the ceiling allows +1.6, so the MP3 lands about 14 LU below target. The ceiling is
a hard limit on clipping and the loudness figure is a target; honouring the first and warning
about the second is the only reading that does not throw away a good mix.

**A run that did not aim at the target is not then failed for missing it**, in any of the three
cases. That is a real amendment to the spec's acceptance criterion 8 rather than an
implementation detail, and the spec is amended in the same commit — M5's code review was right
that "ceiling wins with a warning" is defensible product behaviour but cannot be adopted
silently while the spec says otherwise. Two things keep it honest:

- the true-peak and duration checks still apply, because those are claims about the file rather
  than about a target;
- the report's `mix_encoded` decision says **which** tolerances were checked, and carries
  `loudness_normalized`, so a reader is never told "within every configured tolerance" about a
  comparison nothing performed.

**A measurement nobody took is never a pass.** `-inf` loudness on a run that *was* aiming at
the target is the loudest possible miss and fails; a summary carrying no true-peak line at all
means `peak=true` did not take effect and fails as `true_peak_unmeasured`. Neither is the same
as `Peak: -inf dBFS`, which FFmpeg prints for digital silence and which is a real measurement
infinitely below any ceiling. Also M5's code review.

### The retry loop fails rather than claiming compliance

Encode → decode → measure → compare against `integrated_lufs ± loudness_tolerance_lu` and
`true_peak_dbtp + true_peak_tolerance_db`. On a true-peak overshoot, reduce the master gain by
the overshoot and re-encode, up to `mix.encode.max_retries`. Every attempt's measurements are
retained in the report. Exhausting the budget **fails the mix stage** — the spec says so in as
many words, and INV-13 turns a failed stage into a nonzero exit.

The first attempt's gain already targets the ceiling from the intermediate's own measured true
peak, so the ordinary case needs no retry at all. That is the spec's "reduce the pre-encode
gain or true-peak target", applied before the first encode rather than after the first
failure.

## Alternatives considered

- **Implement BS.1770-4 in NumPy/SciPy** — K-weighting biquads, 400 ms gated blocks, a 4×
  oversampling true-peak filter — and check the coefficients in like `fir_48k_16k.json`.
  Genuinely attractive: no subprocess, deterministic across tool versions, unit-testable, and
  cross-validatable against FFmpeg the way the Silero artifact was validated against two
  sources. Rejected for this milestone as scope: it is a second checked-in filter design with
  its own frequency-response contract, in a milestone that already owns the envelope, the
  streamed mix, the encode loop, a cache, and `process`. The seam is narrow — `loudness.py`
  exposes a measurement function — so it can be replaced later without touching anything else.
- **`loudnorm`'s two-pass mode**, which measures and normalizes in one filter chain. Rejected:
  its second pass applies a limiter and a dynamic mode whose behaviour is version-dependent
  and not what "apply a constant gain toward a target" means. `volume` plus an independent
  measurement keeps the gain a number this project computed and recorded.
- **Normalize into the intermediate** and re-mix on a true-peak retry. Simplest control flow,
  and it re-reads six four-hour tracks per retry while making the intermediate depend on the
  loudness target — so changing the target invalidates the most expensive artifact in the
  pipeline.
- **`ffprobe` for the decoded duration.** One fewer decode. Rejected: see above; it measures
  the container's claim, not the decode.
- **Trust the encoder's own reported peak.** LAME does not expose a true-peak measurement of
  its output, and the ceiling applies to the decode.

## Consequences

- The report gains the exact FFmpeg commands, the FFmpeg version, and every attempt's
  measurements. An operator asking "why is my MP3 −17.2 LUFS" reads the attempt log.
- The MP3 is not required to be byte-stable and is not claimed to be. It is deterministic for
  a fixed FFmpeg version and fixed inputs; a version bump can move its bytes, which is why it
  is a deliverable rather than a cache entry. The intermediate, which *is* cached, is
  byte-stable and produced by no external tool.
- `mix.encode.max_retries`, `duration_tolerance_frames` and `true_peak_tolerance_db` are
  guesses about material nobody has encoded yet — **OQ-020**, cited from each.
  `ebur128`'s summary reports one decimal place, so 0.1 dB of the true-peak tolerance is pure
  quantization and the rest is margin.
- A host without `libmp3lame` fails the mix stage with a named error rather than producing a
  file in some other codec. The flake already carries it, and `doctor` reports the FFmpeg
  version.
- If OQ-020 shows real overshoot routinely exceeding the retry budget, the fix is a larger
  pre-encode ceiling margin rather than a larger budget — the budget existing to bound a
  pathological case, not to walk a gain down in steps.
