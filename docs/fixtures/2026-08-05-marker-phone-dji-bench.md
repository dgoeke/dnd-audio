# M10 intended-phone/DJI marker bench — 2026-08-05

This is the sanitized evidence record for M10's physical positive-path bench. The source take,
generated marker assets, standalone page, layout image and analysis artifacts remain local and
untracked. No audio is committed.

## Capture and evidence handling

One continuous 879.018-second take used six DJI Mic 3 transmitters around a table and three
timecode-jammed receivers. Receiver channels map to positions A1/A2, B1/B2 and C1/C2. A phone
played from p1 for every fixed-position block, then p2 and p3 for the final two cand-a
diagnostics. Media volume was approximately 90%.

The take contained, in order, nine opening plays (three of each candidate), an unattended gap,
three closing plays (one of each), and two moved-phone cand-a plays. Spoken slates confirmed
the order, including that the phone had not moved before the closing block; local transcription
was used only to confirm those slates. There was no event log, by amendment A5. Every analyzer
run therefore completed with the expected `marker_roles_unassigned` warning, and roles below
were assigned arithmetically from the fixed block order rather than fabricated with a log.

Each original source was copied byte-exactly into a conventional session `raw/` tree for
inspection and ingest. Independent SHA-256 checks before analysis, after analysis, and after
the final v1 pass agreed:

| track | SHA-256 |
| --- | --- |
| tx-a | `1afd1bf83db8a11dfd4b0b7a0b24f5a1b458bd432b12e2b6227da676f64d62e5` |
| tx-b | `7b86513b0bf4285eacb25f4fc374942033fb337aaf3bebf33551e418b459c0df` |
| tx-c | `5eb7dfaa90c0980a76d0b36bf1aecd1974e585cacb88c1006aef48597a7eae40` |
| tx-d | `ed3d4fc7f4590a5fb8f14801ed92a391cf913140d9a813d3710ef2ac26745077` |
| tx-e | `7807ad5131ecd19bc43ab919ba1983c3c7470a3440a004c4f8598600fd456f67` |
| tx-f | `225271f8aed06c1928689f1a08c54a735962a82df79a790f145628927ce86be5` |

The scoring command, once per candidate, was:

```text
dnd-audio marker analyze SESSION --marker CANDIDATE --start-window-s 1200 --end-window-s 1200
```

The first pass used Phase A's provisional 550/600-permille thresholds and showed only partial
reach. A diagnostic pass lowered both acceptance floors to 100 permille—the floor already
proved clean on 13.7 minutes of real DJI speech—to expose every real playback without choosing
a production threshold from this take. It found exactly the block-structure count for every
candidate and no extras: cand-a 36 track-occurrences (6 plays × 6 tracks), cand-b 24 (4 × 6),
and cand-c 24 (4 × 6). No occurrence was clipped or weak.

## Candidate comparison

The score is the weakest chirp in a three-chirp sequence. Fixed-position results include the
three opening repeats and the closing play.

| candidate | fixed score range, all tracks | worst gap error | opening lag repeatability | conclusion |
| --- | ---: | ---: | ---: | --- |
| cand-a | 252–650‰ | 257 samples | up to 240 samples | Rejected: weak worst seat and multipath-sensitive timing |
| cand-b | **404–634‰** | 29 samples | **0–1 sample** | Selected: strongest worst seat and stable arrival |
| cand-c | 323–630‰ | **25 samples** | up to 26 samples | Rejected: lower worst-seat margin and less stable arrival |

Cand-b is the objective winner. Its worst fixed-position score is 404 permille, 81 above
cand-c and 152 above cand-a. Its three opening arrivals repeat within one sample on every
track. The four fixed plays all survive on every track without clipping at the nearest seat or
weak-signal classification at the farthest.

The original Phase A `runner_up_permille` calculation was not usable evidence: it compared
later chirps and other legitimate plays against the first-chirp anchor, so a valid repeat could
score as its own runner-up. The bench exposed this detector-shape defect. M10 corrected the
diagnostic to compare only unclaimed, same-chirp alternatives local to one occurrence, added a
50-permille decisiveness boundary, and separated sequence-level echo suppression from chirp-
peak suppression. The final v1 run reported zero local runner-up and zero ambiguous
occurrences on all 24 track-occurrences.

## Fixed-position arrival and timecode observations

Cand-b's closing lag minus the median of its three opening lags, relative to tx-a, was:

| track | opening median lag | closing lag | change |
| --- | ---: | ---: | ---: |
| tx-a | 0 | 0 | 0 samples |
| tx-b | 869 | 884 | +15 samples |
| tx-c | 1612 | 1628 | +16 samples |
| tx-d | 1861 | 1878 | +17 samples |
| tx-e | 1424 | 1437 | +13 samples |
| tx-f | 771 | 784 | +13 samples |

The pair spans about 11.8 minutes. The largest same-geometry change is 17 samples (0.35 ms),
which establishes the measured repeat floor for this phone/room/take but is not a four-hour
clock-drift estimate. ADR-0042 sets the material warning at 48 samples (1 ms), leaving a
31-sample margin above this bench while remaining far below DJI's 1600-sample timecode quantum.

At the first cand-b play, the largest relative acoustic arrival was 1860 samples. Against the
1600-sample metadata quantum, tx-c exceeded the nominal floor by 11 samples and tx-d by 260
samples; the other four tracks were within it. This is not evidence of a gross failed jam:
the comparison necessarily includes table-scale acoustic propagation, while the metadata is
frame-quantized. The analyzer correctly reports the two `beyond_quantum` facts without moving
the timeline.

## Moved-phone diagnostic

Only cand-a was played at p2 and p3. Relative to the median p1 opening lag, its changes were:

| track/position | p2 change | p3 change |
| --- | ---: | ---: |
| A2 / tx-b | +14 | +31 samples |
| B1 / tx-c | -108 | -139 samples |
| B2 / tx-d | -21 | +307 samples |
| C1 / tx-e | +231 | -251 samples |
| C2 / tx-f | -91 | -91 samples |

The changes are only partly consistent with the table layout; notably A1/A2 barely changes
from p1 to p2, C1 moves in the wrong relative direction at p2, and B2 moves in the wrong
relative direction at p3. Combined with cand-a's 240-sample fixed-repeat spread and 257-sample
gap error, this is evidence that its short wide-band chirps select room paths inconsistently.
These moved-phone numbers are geometry diagnostics, never recorder drift.

## Frozen production pass

With cand-b copied to the separate public `v1` registry entry and both acceptance floors set
to 300 permille, the public default command (no hidden `--marker`) found exactly 24 occurrences
in four groups, all six tracks per play, with no extras, clipping, weak signal, or ambiguity.
The weakest score was 404 permille, the largest gap error 29 samples, and the expected
`marker_roles_unassigned` warning remained. Raw hashes still matched afterwards.

The same pass was repeated after the independent code review tightened consecutive-gap
enforcement and made both candidate and occurrence ceilings genuinely streaming. Detector
semantics v3 / analysis semantics v2 / analysis schema v3 produced the same 24 occurrences,
four groups, 404-permille weakest score, 29-sample maximum gap error, and zero runner-up,
ambiguous, clipped, or weak outcomes. The review repair therefore changed rejection and safety
boundaries, not the bench-selected result.

False-positive evidence is separate and stronger than this take could provide: the real-DJI
speech sweep accepted zero sequences at a 100-permille floor. See
`2026-08-05-marker-false-positive-sweep.md`.

## Decision

Freeze cand-b's recipe as marker v1. The marker supplements and verifies the LTC jam; it does
not replace timecode, place a restarted file that missed the sound, correct the timeline, or
turn moved-wearer measurements into recorder drift. The three-clap procedure remains the
human-readable fallback.
