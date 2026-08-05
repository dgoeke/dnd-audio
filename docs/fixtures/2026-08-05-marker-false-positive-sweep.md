# Marker false-positive sweep — 2026-08-05

Not the M10 bench. A software-only measurement against recordings this project already had,
made to remove one block from the bench protocol before the operator ever enters the room.

**Question.** Does ordinary speech, recorded through the real DJI capture chain, produce an
accepted marker sequence?

**Answer: no, and not remotely.** Zero accepted sequences for all three candidates across 13.7
minutes of real audio. The strongest *single-chirp* correlation anywhere was 186 permille
against a 550 acceptance threshold — and a single chirp is not a detection, because a sequence
requires all three in order with asymmetric gaps inside tolerance.

## What was analyzed

Every real DJI recording on the project machine, from two unrelated captures. Both are
gitignored session audio; neither contains a marker, since both predate it.

| capture | files | duration | content |
| --- | --- | --- | --- |
| jam verification (2026-08-03) | 4 | 3.1 min | one person announcing microphones while holding one at a time |
| minimal direction check (2026-08-04) | 4 | 10.6 min | two people, deliberately overlapping speech, exact-simultaneous short words, hand claps at both ends |

All eight are `pcm_f32le` 48 kHz mono `_orig` transmitter files — the same format and the same
capture path a real session produces.

**The second capture is the one that matters.** It was designed to be acoustically difficult:
two voices talking over each other on purpose, plus a three-clap alignment pattern at each end.
A clap is broadband with a sharp onset, which makes it the single most plausible thing in a
tabletop recording to be mistaken for a chirp.

## Method

`detect_occurrences` was run over each whole file at the shipped `DetectorThresholds()`, through
the project's own `PcmReader` rather than a second decoder — so the decode path is the pipeline's
and a false positive here would be a false positive in production.

Because a count of zero says only that a line was not crossed and never by how much, a second
pass measured the **margin**: the strongest normalized correlation of any single chirp template
at any position in any file, via the detector's own `_normalized_scores`.

Both are checked in as `tests/test_marker_false_positives.py`, marked `host_smoke` so the gate
stays offline and audio-free. Recordings are **discovered rather than named**, the convention
`test_qwen_smoke.py` established: pinning a filename would turn a replaced corpus into a silent
skip.

## Result

| candidate | accepted sequences | strongest single chirp | threshold | headroom |
| --- | --- | --- | --- | --- |
| `cand-a` | 0 | 114 / 1000 | 550 | +436 |
| `cand-b` | 0 | 106 / 1000 | 550 | +444 |
| `cand-c` | 0 | 186 / 1000 | 550 | +364 |

`cand-c` is the loosest, which is what its design predicts: the shortest chirps at the widest
band have the lowest time-bandwidth product, so they are the least selective. It still sits at a
third of the threshold.

### How far the threshold could actually fall

The per-chirp margin understates the headroom badly, because a sequence needs three chirps *in
order at asymmetric gaps* — and it is that structure, not the score, doing most of the
rejecting. Re-running the hard corpus with the acceptance thresholds forced downward:

| threshold (both) | `cand-a` | `cand-b` | `cand-c` |
| --- | --- | --- | --- |
| 550 (shipped) | 0 | 0 | 0 |
| 300 | 0 | 0 | 0 |
| 150 | 0 | 0 | 0 |
| **100** | **0** | **0** | **0** |
| 50 | 0 | 0 | 4 |
| 25 | 114 | 28 | 237 |

**All three stay clean down to 100 permille** — a fifth of the shipped threshold — and `cand-c`
is the first to break, at 50. Between 50 and 25 all three collapse, which is the gap structure
finally being satisfiable by chance once almost any peak qualifies as a chirp.

This is the number that matters for the bench: if the farthest lav correlates weakly, there is
roughly a factor of five to spend before false positives become a concern at all.

## What this licenses, and what it does not

**Licensed.** The false-positive half of the charter's bench criterion 4, on real hardware over
more audio than the planned in-room block would have supplied. The bench protocol no longer asks
the operator to record speech and media, which is what made a solo bench practical.

**Not licensed, and the distinction is the whole point.** This says a marker is not *invented*
from speech. It says nothing about whether a marker played from a phone speaker is *found* on a
lav across a table — the opposite failure direction, and the one the bench exists for. A
detector that accepted nothing at all would score perfectly here.

Nor did it set the thresholds by itself. It bounds one side, and generously: any threshold at
or above **100 permille** keeps this result on the hard corpus. The later positive bench found
every v1 playback at a 300-permille production floor, with a weakest fixed score of 404;
ADR-0042 freezes the two sides together.

## Notes for the bench

**If the farthest lav is weak, lower the threshold rather than abandoning a candidate.** A phone
speaker is a poor radiator below about 700 Hz and a lav capsule rolls off at the top, so a
correlation well under 550 at the far seat is a plausible outcome — and it is not the failure it
would look like. There is a factor of five to spend before the false-positive result is at risk.
Below 100 permille, re-run this sweep before trusting it.

**The asymmetric gaps are load-bearing, more than the score threshold is.** That is the real
lesson of the sweep-down table: at 100 permille nearly every transient in a two-voice recording
qualifies as a chirp, and still no sequence assembles. Any future change that relaxes gap
tolerance or drops the required chirp count is spending the margin this measurement found, and
should re-run it.

Referenced from answered **OQ-025**, ADR-0042, and `DetectorThresholds` in
`src/dnd_audio/marker/detect.py`.
