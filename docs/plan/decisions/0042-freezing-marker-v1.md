# ADR-0042 — Freezing marker v1

**Status:** accepted
**Date:** 2026-08-05
**Milestone:** M10

## Context

ADR-0041 decided that `MARKER_SPECS` would hold named candidates and have **no `v1` key** until
physical evidence selected one. That intended-phone/six-DJI bench has now run; its sanitized
measurements are in `docs/fixtures/2026-08-05-marker-phone-dji-bench.md`.

The charter is explicit about why theory is not enough. Exact frequencies, chirp durations,
directions, gaps, sample format and peak level are all properties of what a phone speaker
radiates, what a lav capsule accepts, and what a room does in between — none of which a
synthetic fixture can answer. The provisional three-chirp 500 Hz–8 kHz design in the charter
is a *candidate*, described there as "not frozen by this planning document".

## Decision

**Cand-b becomes the separate public `v1` entry.** It won on the two properties that matter
most at opposite ends of the table: the strongest worst-track fixed-position score (404
permille, versus cand-c's 323 and cand-a's 252) and the most repeatable arrival (0–1 sample
over three opening plays, versus 26 and 240). All four fixed-position cand-b plays were found
on all six tracks at approximately 90% phone volume, with no clipping, weak signal, ambiguity,
or extra occurrence. The real-DJI negative sweep accepted no sequence at a 100-permille floor.

The human-readable v1 recipe is mono 48 kHz signed 16-bit PCM; three 250 ms linear rising
chirps from 800 Hz to 6 kHz; 10 ms raised-cosine fades; asymmetric 200 ms then 320 ms gaps;
100 ms leading and trailing silence; and 0.5 full-scale peak amplitude. The frozen anchor is
sample **4800**, the first sample of the first chirp. The canonical WAV is 141,164 bytes and
has SHA-256:

```text
70355baad6bb72b38e0b606cddbbaa3428c11429bec74cd127aa6f8935ecdf6f
```

The detector constants are frozen as follows. These are engineering bounds with stated
margin, not copies of one bench's extrema:

| constant | v1 value | evidence and margin |
| --- | ---: | --- |
| chirp score floor | 300‰ | 104 below the weakest fixed v1 sequence; 3× the floor proved sequence-clean on real speech |
| sequence score floor | 300‰ | Same weakest-link domain; keeping the floors equal avoids an acceptance region the sequence test cannot use |
| runner-up separation | 50‰ | The final bench had no unclaimed local alternative; 50 rejects an echo nearly as persuasive as the selected path without tying the value to a nonexistent bench runner-up |
| chirp peak NMS radius | 2400 samples (50 ms) | Suppresses one correlation lobe/reflection family while preserving a distinct local competitor for ambiguity reporting |
| sequence NMS radius | 7200 samples (150 ms) | The synthetic room response produced a coherent 106 ms echo; v1 lasts 1.47 s and cannot legitimately be replayed this close |
| gap tolerance | 1440 samples (30 ms) | Largest v1 error was 29 samples, leaving 1411; synthetic stretching shows chirp correlation fails before this bound |
| association lag | 4800 samples (100 ms) | Largest bench lag was 1878, leaving 2922; still covers table propagation plus one 33.3 ms metadata quantum |
| clipping ratio | 10‰ at ≥0.99 full scale | No bench occurrence crossed it at approximately 90% phone volume; it classifies score trust, not acceptance |
| weak RMS | 1‰ full scale | No bench occurrence crossed it; below this is effectively no usable signal |
| material fixed-geometry change | 48 samples (1 ms) | Bench repeat maximum 17, leaving 31; 1 ms is below the 1600-sample timecode quantum yet above measured acoustic repeat noise |
| occurrence ceiling | 32 per track | Eight were planned at most and six occurred; 32 is operational headroom and is checked while each interval is scanned and across the canonical interval set |
| peak-candidate ceiling | 256 per chirp/track interval | Eight times the occurrence ceiling leaves room for reflections and incomplete sequences while bounding dense non-marker peaks before the final read |

Three chirps remain required by the waveform and sequence assembler. The fixed-position
closing lag changed by 13–17 samples relative to the opening median over about 11.8 minutes.
That is the measured same-position tolerance for this bench, not a four-hour drift estimate.

The bench also found a detector-shape defect before schemas froze: `runner_up_permille` had
compared later chirps and other valid plays against the first-chirp anchor. It therefore made
legitimate repeats look like ambiguity. Detector semantics v2 compares only unclaimed,
same-chirp alternatives local to the occurrence, after every accepted occurrence is excluded;
analysis schema v2 records the resulting `ambiguous` fact. Sequence-level NMS is separate so
a coherent room echo remains one event without erasing a distinct chirp-level competitor.

The independent code review then found two further shape defects in that first repair. The
occurrence ceiling was checked only after all per-chirp peaks had accumulated, and the
assembler constrained each chirp against the first anchor without enforcing the *consecutive*
gap errors it reported. Detector semantics v3 checks both peak-candidate and occurrence
ceilings after every fixed-size block, enforces the 1440-sample limit on each actual gap, and
applies the 32-occurrence ceiling across the full interval set. Analysis semantics v2 also
refuses overlapping logged roles, a log naming another session, and an arbitrary first pair
when multiple start/end groups exist. Analysis schema v3 adds the consumed manifest schema
version to identity. The re-baselined default v1 pass was unchanged: 24 occurrences, four
groups, weakest score 404, maximum gap error 29, and no ambiguity, clipping, or weak signal.

A future marker change takes a **new** semantic version and a new versioned filename. It does
not silently replace v1, and v1's frozen hash stays in this document as history.

## Alternatives considered

**Freeze the charter's provisional candidate now** and treat the bench as confirmation that
could retire it later. Rejected by the charter and by the operator on 2026-08-05. It would put
a golden SHA-256 and a full synthetic regression battery behind a guess, and the two bench
outcomes that would invalidate it — no candidate surviving at the farthest lav, or playback
warping enough to break sequence detection — are exactly the ones theory cannot rule out.

**Never freeze; always pass a spec name.** Rejected: an operator recording Session Zero should
not be choosing a waveform, and a marker whose identity is not frozen cannot be compared
across sessions, which is the entire point of measuring drift between two of them.

## Consequences

`marker build OUTPUT_DIRECTORY` now resolves v1. The hidden `--marker` option remains only for
reproducing the three bench candidates; it is not a public candidate-management interface.

Once accepted, the golden test that pins v1's SHA-256 also pins everything underneath it —
the sine table, the integer phase arithmetic, and the RIFF layout in `marker/wav.py`. Any
change to any of those turns that test red, which is the desired behaviour: they are all part
of what the frozen bytes mean.

What would make us revisit: a materially better phone, a receiver firmware change altering the
recorded band, or live-session evidence that the marker is failing in practice. Any of those is a new
version, never an edit to this one.
