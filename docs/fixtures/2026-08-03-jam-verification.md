# Jam verification capture — 2026-08-03

Not the H1 fixture. A four-file probe recorded specifically to settle **OQ-023**: does a
timecode jam the operator can see on the receiver displays reach `bext.time_reference` in
the written WAVs? It also produced the project's first measurement of **OQ-006**, relative
sample-clock drift, and refuted a recommendation this ledger had made about frame rate.

**It answers OQ-023 affirmatively.** That is the assumption the entire cross-receiver
synchronization strategy rests on, and it had never been checked.

## What was recorded

Two receivers, two transmitters each, four `_orig` files, `pcm_f32le` 48 kHz mono, ~47 s.
Firmware `ver:02.00.06.01`. The operator jammed the receivers L-OUT → L-IN, confirmed the
displays matched, then started each receiver's pair a few seconds after the other.

| file | group | `origination_time` | `time_reference` | ÷48000 | ÷1600 |
| --- | --- | --- | --- | --- | --- |
| `TX01_MIC005_…145028_orig` | A | 14:50:28 | 13657600 | 284.533 s | 8536 |
| `TX02_MIC003_…145028_orig` | A | 14:50:28 | 13657600 | 284.533 s | 8536 |
| `TX01_MIC002_…145122_orig` | B | 14:51:22 | 13912000 | 289.833 s | 8695 |
| `TX02_MIC002_…145122_orig` | B | 14:51:22 | 13910400 | 289.800 s | 8694 |

`MIC###` differs per transmitter within one receiver group (005 vs 003), so the counter is
**per transmitter**, not per receiver — evidence toward **OQ-003**. `TX##` repeats across
groups, confirming **OQ-002**/INV-11 again: the label is not globally unique and the
directory is identity.

## Method — why the audio is the arbiter

Metadata alone cannot answer OQ-023, because both readings of it are self-consistent. The
discriminator is the *true* offset between the two receivers' recordings, which only the
audio knows. All four transmitters captured the same room, so cross-correlating them
measures that offset independently of any metadata, and it can then be compared against
what `time_reference` predicts.

Correlation is on a 1 ms log-RMS envelope, normalized over the valid overlap only, with the
peak parabolically interpolated. The speech in this capture is repetitive ("testing a first
transmitter…", "testing a second…"), which makes a naive envelope correlation lock onto the
wrong repeat — an unnormalized first attempt put two files from the *same* receiver 8 s
apart. **The method is validated by injecting known shifts into a single file and
confirming they are recovered exactly** (±3.000 s and −1.500 s both returned to 3 decimal
places, ncc 1.000). Any repeat of this measurement should keep that control.

## Result 1 — the jam reaches the files (OQ-023, outcome 1)

Measured offset versus the offset `bext.time_reference` predicts, for all six pairs:

| pair | groups | audio | timecode | error |
| --- | --- | --- | --- | --- |
| TX01_MIC005 → TX02_MIC003 | A→A | −0.011 s | 0.000 s | 11 ms |
| TX01_MIC002 → TX02_MIC002 | B→B | +0.014 s | −0.033 s | 47 ms |
| TX01_MIC005 → TX01_MIC002 | **A→B** | +5.270 s | +5.300 s | 30 ms |
| TX01_MIC005 → TX02_MIC002 | **A→B** | +5.284 s | +5.267 s | 17 ms |
| TX02_MIC003 → TX01_MIC002 | **A→B** | +5.281 s | +5.300 s | 19 ms |
| TX02_MIC003 → TX02_MIC002 | **A→B** | +5.295 s | +5.267 s | 28 ms |

Worst case 47 ms — 1.4 frames at 30 fps. **Every cross-receiver pair is inside one frame.**
The correlator never saw the metadata, so four independent pairs landing within 30 ms of a
metadata-only prediction across a ±47 s search range is not chance. The residual is the
33.3 ms frame quantum plus real acoustic path differences between lav positions.

The two receivers were started 5.28 s apart and their independently-written timecodes agree
on 5.28 s. The jam propagated.

## Result 2 — wall clock does *not* agree across receivers

Subtracting `time_reference` from each file's `origination_time` gives the implied epoch:

```
group A   53143.467 s      group B   53192.167 s      difference 48.7 s
```

Within a group it agrees to the second; across groups it is **48.7 s out**. The audio proves
the true gap is 5.28 s, so the timecode is right and the receivers' real-time clocks are
wrong relative to each other.

OQ-004 previously noted `origination_date`/`origination_time` as "an unused signal worth
remembering" that could bound a cross-receiver offset to ±1 s, with the caveat that two
receivers' clocks are independent and might not agree. **That caveat is now measured, and it
is fatal to that use.** Wall clock is usable for archival naming and for a human reading a
report. It must never anchor a cross-receiver offset, and nothing in the code currently
stops it from being used that way.

## Result 3 — first drift measurement (OQ-006)

Each `orig` file carries **one** `time_reference`, stamped at the start; from there the
transmitter's own crystal defines the timeline. If those crystals ran at typical consumer
tolerance (±20–50 ppm) a four-hour session would diverge by 288–720 ms, which would make the
"independent recorders synced by timecode" architecture untenable regardless of the jam.

Measuring the residual lag in the first third of each overlap against the last third:

```
drift:  +1.0  −0.2  +2.4  −0.3  +0.9  +1.1  ppm
lag change over ~30 s:  0.00 to 0.07 ms
```

Consistent with zero inside a noise floor of roughly **±3 ppm** on this 30 s baseline. At the
pessimistic end, 3 ppm over 4 hours is 43 ms and over 6 hours is 65 ms; at the likely ~1 ppm
it is 14 ms and 22 ms. **Drift across a full session is the same size as, or smaller than,
the 33 ms quantization already present at file start** — the error budget does not grow
materially with session length.

This does not close OQ-006. A 30 s baseline cannot distinguish 1 ppm from 3 ppm, and it
observes no thermal excursion, no battery swap and no power cycle. It rules out the
catastrophic case; H2 must confirm the bound over hours.

## Result 4 — the frame-rate setting did not reach the files (OQ-024)

One of the two receivers was set to **60 fps** and the other to 30 fps. The two groups'
`bext` chunks differ in **exactly five bytes** — `origination_time` and `time_reference` —
and every rate field is byte-identical across all four files:

```
TIMECODE_RATE 30/1    MASTER_SPEED 30/1    CURRENT_SPEED 30/1    TIMECODE_FLAG NDF
```

Every `time_reference` is an exact multiple of **1600 samples**, the 30 fps quantum. No file
shows finer resolution. (A check for "exact at 60F" is vacuous — anything divisible by 1600
is divisible by 800.)

This refutes a recommendation the ledger had already adopted. OQ-004 concluded from DJI's
documentation that 50 and 60 fps are supported and that "at 60 fps the quantum halves to 800
samples — 16.7 ms. That is a menu setting and should be applied before the next capture." H1's
recipe was amended to require 60 fps on all three receivers. **On the evidence here that
setting changes nothing in an `orig` file**, and the instruction has been removed rather than
left as an unverified ritual.

Scope: `orig` files only. These are the transmitter's internal recordings, and the
transmitter may be handed a timecode value without being told a rate. A receiver-side `edit`
file might carry 60/1. That is untested and, for this project, uninteresting — `orig` is the
only file consumed (**OQ-007**), because the point is to treat each transmitter as an
independent high-quality recorder.

The accidental experiment is more useful than the intended one: **the two receivers were on
different frame rates and the jam held anyway**, to within one frame across a real 5.28 s
offset. The spec's owner note asks for a consistent rate across kits; this capture violated
that and cross-receiver agreement was unaffected. Keep the rates consistent as hygiene, but
nothing downstream is known to depend on it.

## Reproducing

The measurement scripts are not committed — they are twenty lines of `soundfile` plus a
normalized cross-correlation, and the numbers above are the artifact. What must be kept if
this is repeated:

- Normalize the correlation over the **valid overlap only**; an unnormalized correlation
  fails on repetitive speech.
- Keep the **injected-shift control**. The first version of this measurement had an inverted
  sign convention and would have reported that the jam failed.
- Compare against the timecode prediction rather than against zero. Agreement between two
  independent measurements is the evidence; either alone is not.

`session.sync_qa` (M2, off by default) is the pipeline's version of this and should grow the
same comparison — see OQ-023's closing note.
