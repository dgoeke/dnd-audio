# M10 marker bench protocol

This is the executable capture protocol for the phone/DJI bench that
[M10 — Acoustic synchronization marker](plan/milestones/M10-acoustic-sync-marker.md) is built
around. It is the charter's `## Bench protocol` section made step-by-step.

**One person, alone, no talking required.** Set up six transmitters, press a button on a phone
fourteen times, and leave the room recording for a while. Expect 20 minutes of setup and about
fifteen minutes of take, nearly all of it unattended.

The protocol used to ask for ten minutes of conversation and ordinary media as a false-positive
control. **It no longer does** — that measurement has already been made against real recordings
this project already has, and is [written up separately](fixtures/2026-08-05-marker-false-positive-sweep.md).
Nothing is lost by dropping it; the sweep covers more real audio than the block would have.

Public evidence must not contain a hostname, username, absolute home-directory path, or audio.
Never modify anything after it has been copied under a session's `raw/`.

## What this bench decides

Three candidate waveforms exist in the code. **None of them is `v1`, and `marker build` with no
`--marker` refuses to build anything** — deliberately, so nobody can record a real session
against a waveform the hardware has never heard. This bench is what turns one of them into `v1`.

It answers four questions that no synthetic test can, and one is by far the most important:

1. **Does a phone speaker at a normal media volume reach the farthest lav decisively?** This is
   the question. It is what a candidate wins or loses on.
2. Does it **clip** the nearest one?
3. Does a phone browser's playback preserve the marker's internal timing well enough for the
   sequence check ([OQ-029](plan/OPEN-QUESTIONS.md))?
4. Is a repeated same-position lag stable, so that a genuine start-to-end change means
   something?

A fifth question — whether ordinary speech produces a false marker — **is already answered**,
below.

### Already answered: false positives

The detector was run over every real DJI recording on the project machine: 13.7 minutes across
eight files, two different captures, real rooms, real voices, deliberate overlapping speech, and
hand claps at both ends of one of them. A clap is broadband, which makes it the single most
plausible thing to be mistaken for a chirp.

| | accepted sequences | strongest single-chirp match | acceptance threshold |
| --- | --- | --- | --- |
| `cand-a` | **0** | 114 / 1000 | 550 |
| `cand-b` | **0** | 106 / 1000 | 550 |
| `cand-c` | **0** | 186 / 1000 | 550 |

Real speech does not come close, and a *single* chirp is not a detection anyway — a sequence
needs all three, in order, with the asymmetric gaps inside tolerance. Forcing the threshold down
to **100** produces zero false positives too; it is the gap structure doing the rejecting, not
the score. The measurement is repeatable as `tests/test_marker_false_positives.py` (marked
`host_smoke`, since the recordings are gitignored) and the detail is in
[the sweep write-up](fixtures/2026-08-05-marker-false-positive-sweep.md).

**Why that matters in the room:** if the farthest lav turns out to correlate weakly, that is not
a candidate failing. There is roughly a factor of five of threshold to spend before false
positives become a concern, and spending it is the right call.

**This does not answer question 1.** Not producing a false positive on speech and being reliably
found across a table are opposite failure directions, and only the room settles the second.

### The three candidates

| name | band | chirps | gaps | length | asks |
| --- | --- | --- | --- | --- | --- |
| `cand-a` | 500 Hz → 8 kHz | 3 × 180 ms, all rising | 150 / 250 ms | 1.140 s | the charter's provisional design — the reference the other two are measured against |
| `cand-b` | 800 Hz → 6 kHz | 3 × 250 ms, all rising | 200 / 320 ms | 1.470 s | trades bandwidth for time-bandwidth product, in the band a phone speaker actually radiates. **The candidate to beat at the farthest seat** |
| `cand-c` | 400 Hz → 10 kHz | 3 × 120 ms, up/down/up | 90 / 160 ms | 0.810 s | whether a short chirp survives the room, and whether direction asymmetry buys rejection that gap asymmetry alone does not. The least intrusive of the three if it works |

All three are mono, 48 kHz, `pcm_s16le`, peaking at half of full scale (−6 dBFS) with 100 ms of
leading and trailing silence. In all three the **anchor** — the sample every measurement is
relative to — is the first sample of the first chirp, 4800 samples (100 ms) after the file
starts.

**Record all three in one take.** Comparing candidates across separate takes would compare rooms
and phone positions as much as waveforms.

## Materials

- Six DJI transmitters, labelled durably `tx-a` … `tx-f`, and their receivers.
- The intended phone, and the browser you actually plan to use in a session.
- Something to hold the phone in one repeatable place — the middle of the table, screen up.
- A stopwatch, or the phone's own clock. Approximate times are enough; see
  [The event log](#the-event-log).
- A paper or text log. Its columns are fixed:

  | # | candidate | role | geometry | stopwatch | note |
  | --- | --- | --- | --- | --- | --- |
  | 1 | `cand-a` | `start` | `g1` | 0:35 | |

That is the whole list. No second person, no media, no script.

## Step 1 — build the three candidates

On the project machine:

```bash
dnd-audio marker build ./bench-markers/cand-a --marker cand-a
dnd-audio marker build ./bench-markers/cand-b --marker cand-b
dnd-audio marker build ./bench-markers/cand-c --marker cand-c
```

One directory each, because the manifest is named `marker-manifest.json` and describes the pair
beside it — three builds into one directory would leave one manifest describing one candidate
and two orphaned pairs.

`--marker` is hidden from `--help` on purpose: this charter's non-goals exclude a public
candidate-management interface, and after v1 is frozen the discoverable command is
`marker build OUTPUT_DIRECTORY` alone. It is documented here because the bench is the one
context that needs it.

Each build prints the WAV's SHA-256. **Write those three hashes into the bench notes now**, from
the terminal — they are what proves later that the file on the phone is the file the repository
describes.

Do **not** build into a directory under any session's `raw/`. The command refuses, and a refusal
is a clean stop, not a problem to work around.

Three files per candidate:

```text
bench-markers/cand-a/dnd-audio-sync-marker-cand-a.wav     the canonical bytes
bench-markers/cand-a/dnd-audio-sync-marker-cand-a.html    the standalone player, with those exact bytes inside it
bench-markers/cand-a/marker-manifest.json                 published last, as the completeness marker
```

## Step 2 — get the pages onto the phone

Copy the three `.html` files to the phone by whatever means you would normally move a file —
cable, a local file share, your own storage. **Nothing here needs the network to work**, and
that is worth confirming rather than assuming: after the transfer, put the phone in airplane
mode for the rest of the bench. A page that still plays is a page that carried its audio with
it.

**Open all three pages now, and leave them open in three tabs.** Switching tabs mid-take is much
less fiddly than opening files from a file manager between candidates, and it is the only reason
this take needs any dexterity at all.

For each one, before the transmitters are recording:

- The page shows the marker's name and the WAV's SHA-256. **Check it against what Step 1
  printed.** If they differ, the file did not survive the transfer and nothing after this point
  is meaningful.
- Press `Play marker` once and confirm you hear it, that the button cannot be pressed twice into
  overlapping playback, and that the page returns to a resting state afterwards.
- The page also offers the WAV as a **Download**, built from the same bytes it just played. A
  download whose SHA-256 matches Step 1 is direct evidence that what reached the speaker was the
  canonical file. Worth doing once, on one candidate.

This is also the moment to set the **media volume**. Turn it up until the marker is clearly
audible across the table but not unpleasant at arm's length, then stop touching it. Write the
step down. Volume is part of the instrument.

Switch Bluetooth off — a marker played through a speaker you did not intend measures that
speaker's latency, not the room's geometry.

## Step 3 — geometry

Place the six transmitters as they will be worn or seated in a real session, spread across the
table's real span. **The nearest and the farthest from the phone are the two that decide this
bench** — note which they are, and make the spread realistic rather than convenient. A bench
where every lav is a metre from the phone answers nothing.

Put the phone in the **middle**, screen up, in a position you can leave alone for fifteen
minutes. Keep the screen awake — the page does not request a wake lock, and that is deliberate
(it is outside M10).

Jam the receivers as you normally would. The marker measurements are relative and do not depend
on the jam being perfect, but they do depend on `ingest` placing the six tracks within about
100 ms of each other; unjammed files placed seconds apart would show up as *undetected* rather
than as a placement problem, which is exactly the wrong diagnosis to hand yourself.

Give this arrangement the geometry ID **`g1`** in the log. A geometry ID is a written assertion
that nothing moved: two events sharing one are a claim that the phone and every transmitter were
in the same places for both. Nothing in the audio can establish that, which is why it is the
operator's signature and not an inference. Without it the analysis reports differential acoustic
arrival and refuses to call anything drift — the correct outcome, not a limitation.

## Step 4 — the take

Start every transmitter, confirm every recording indicator, and start the stopwatch. **One
continuous take.** Leave several seconds of silence between every play so occurrences never
overlap.

### Block 1 — opening, geometry `g1` — nine plays

Say the block aloud so the audio carries its own slate: *"Opening block, geometry one."*

Then, tab by tab, logging each play as you go:

| order | candidate | role | geometry |
| --- | --- | --- | --- |
| 1–3 | `cand-a` × 3 | `start` on the first, `diagnostic` on the other two | `g1` |
| 4–6 | `cand-b` × 3 | `start` on the first, `diagnostic` on the other two | `g1` |
| 7–9 | `cand-c` × 3 | `start` on the first, `diagnostic` on the other two | `g1` |

**This block is the bench.** Everything after it is a bonus that costs almost nothing. Three
plays of each is what makes same-position repeatability measurable, and repeatability is what
tells us whether a later change means anything.

Only one of the three carries the `start` role, because a start/end pair is a comparison between
two occurrences and a log with three starts cannot say which. The other two are still fully
analyzed — `diagnostic` means *deliberately outside the pair*, not *ignored*.

### Block 2 — leave it running

Ten minutes or so. **Do nothing.** Leave the room, make coffee, work on something else. Do not
touch the phone or the transmitters.

This gap exists so the closing block is a genuinely later measurement. Three minutes is enough
to select a candidate; ten gives a usable drift figure as well, since the ≈1 ppm this project
measured (**OQ-006**) accumulates to about 29 samples over ten minutes and the detector reports
integer samples. Longer is better and costs only patience.

### Block 3 — closing, geometry `g1` — three plays

**The phone has not moved and must not have been rotated, picked up, or nudged.** If it was, say
so in the log and give this block a new geometry ID — an unlogged move is the one mistake that
silently corrupts the measurement this bench exists to make.

Say: *"Closing block, geometry one."* Then **one** play of each candidate, same order, role
`end`, geometry `g1`.

One is enough here: the opening block already established repeatability, and this block only has
to anchor the other end of the comparison.

### Block 4 — the moved-phone diagnostic — two plays

**This block comes last, and the ordering is not negotiable.** Moving the phone before the
closing block would put a real geometry change inside the start/end pair, and the analyzer would
have no way to know.

Say: *"Diagnostic block."* Then:

1. Move the phone somewhere clearly different — one end of the table, or off to a side. Note
   roughly where. Geometry ID **`g2`**. Play `cand-a` once, role `diagnostic`.
2. Move it again, somewhere else. Geometry ID **`g3`**. Play `cand-a` once, role `diagnostic`.

One candidate is enough: this proves the analyzer *enumerates* a differing lag without calling
it drift, which is a claim about software, not about the waveform.

### Stop

Leave five seconds of room tone, then stop every transmitter. **Fourteen plays total.**

## Step 5 — copy, and hash before anything reads it

Copy each transmitter's original files into per-track directories, preserving the DJI filenames
and file bytes exactly. Do not trim, rename, normalize, or convert anything.

```text
<session>/
  raw/tx-a/<original filenames>
  ...
  raw/tx-f/<original filenames>
```

Then, **before running any command against the session**, record the hashes:

```bash
find raw -type f -print0 | sort -z | xargs -0 sha256sum > ../marker-bench-raw.before
```

Repeat it after every analysis run and compare. `dnd-audio` verifies this itself on every run —
that is INV-01 and there is a regression matrix behind it — but a bench is exactly the occasion
to check the checker independently.

Write a `session.yaml` beside `raw/` in the usual shape (the
[H1 runbook](H1-two-person-recording-runbook.md) has a complete example; six tracks, the real
receiver/channel map, and the timecode section matching the receivers).

## The event log

One YAML file, separate from `session.yaml` on purpose: a new configuration field would change
`config_hash`, which invalidates every cache downstream of it, including gigabytes of ASR
(ADR-0016). Setting a search window must not cost that.

```yaml
schema_version: 1
session_id: "YYYY-MM-DD-marker-bench"
events:
  - {role: start,      marker_name: cand-a, start_ms:    20000, end_ms:    50000, playback_order: 0, geometry_id: g1}
  - {role: diagnostic, marker_name: cand-a, start_ms:    28000, end_ms:    58000, playback_order: 1, geometry_id: g1}
  - {role: diagnostic, marker_name: cand-a, start_ms:    36000, end_ms:    66000, playback_order: 2, geometry_id: g1}
  - {role: start,      marker_name: cand-b, start_ms:    44000, end_ms:    74000, playback_order: 3, geometry_id: g1}
  # ... one entry per logged play, fourteen in all ...
  - {role: end,        marker_name: cand-a, start_ms:   700000, end_ms:   730000, playback_order: 9,  geometry_id: g1}
  - {role: diagnostic, marker_name: cand-a, start_ms:  1000000, end_ms:  1030000, playback_order: 12, geometry_id: g2}
  - {role: diagnostic, marker_name: cand-a, start_ms:  1060000, end_ms:  1090000, playback_order: 13, geometry_id: g3}
```

Rules the loader enforces, so a mistake is a refusal rather than a wrong number:

- **`start_ms` and `end_ms` are integer milliseconds on the session timeline**, half-open, and
  generous. They are a search window, not a measurement — being 20 seconds early costs nothing.
- `playback_order` is distinct across the whole log and is what breaks a tie if two events could
  claim the same occurrence.
- At most one `start` and one `end` per geometry: two events marked `start` that do *not* share a
  `geometry_id` are refused at load, because they cannot both anchor the same comparison. Extras
  are `diagnostic`.
- `geometry_id` absent means *unknown*, which is not the same as *unchanged*, and never licenses
  a drift claim.
- `marker_name` is checked against the marker being analyzed, so a take recorded with one
  candidate cannot be scored as another.

### Getting the times right without a synchronized clock

Session time zero is where the *earliest* source starts, which is not where your stopwatch
started. Rather than guess, do it in two passes:

**Pass 1 — find the occurrences.** Analyze with no event log and windows wide enough to cover
the take:

```bash
dnd-audio inspect  <session>
dnd-audio ingest   <session>
dnd-audio marker analyze <session> --marker cand-a \
  --start-window-s 1200 --end-window-s 1200
```

`work/sync-marker-analysis.json` lists every occurrence with an `anchor_ms` — which is all this
pass is for. Without a log the analysis falls back to naming a start and an end only if each
default window holds *exactly one* occurrence; three plays per block means it will not, so the
groups come back unlabelled. It classifies nothing as drift either way, since that needs a
geometry ID. Both are the expected outcome of this pass rather than failures, and it exits zero.

**Pass 2 — write the log against them.** Match the Nth occurrence to the Nth line of your paper
log, take a window of roughly ±15 s around each `anchor_ms`, and fill in the roles and geometry
IDs **from the paper log**. Then:

```bash
dnd-audio marker analyze <session> --marker cand-a --event-log marker-events.yaml
dnd-audio marker analyze <session> --marker cand-b --event-log marker-events.yaml
dnd-audio marker analyze <session> --marker cand-c --event-log marker-events.yaml
```

Taking the *times* from the first pass is not circular — the roles, the ordering, and the
geometry all come from what you wrote down at capture time, and none of them can be inferred
from audio. **If the counts disagree** — nine plays logged, seven occurrences found — that is a
finding. Record it. Do not adjust the log to match.

Each run overwrites `work/sync-marker-analysis.json` and `output/marker-report.json`, so copy
each candidate's pair somewhere before running the next.

## Step 6 — before you pack up

Run pass 1 above for one candidate while the transmitters are still in place. It takes seconds.
Read `work/sync-marker-analysis.json` and check:

| look at | what it should say | if it does not |
| --- | --- | --- |
| `occurrences` | one entry per track per play, at the times you logged | if a track is absent everywhere, its recording, its placement, or the volume step is wrong |
| `occurrences[].clipped` | `false`, especially on the nearest transmitter | lower the media volume one step and redo Blocks 1 and 3 |
| `occurrences[].weak` | `false`, especially on the farthest | raise it one step and redo Blocks 1 and 3 |
| `occurrences[].score_permille` | above 600 on every track | a low value at the farthest seat is **not** necessarily a candidate failing — see the false-positive headroom above — but it is worth knowing while the room is still set up |
| `notes` and the report's `warnings` | read them | they name what the analysis could not conclude and why |

The volume-step retakes are the only ones worth redoing on the spot. Everything else can be
decided at a desk.

**If all three candidates score low at the farthest seat**, do not conclude they failed. The
threshold has room to move down and the sweep says so. What would be a real failure is a marker
the detector cannot find *at any threshold* — no accepted sequence and no strong chirps either —
which means the sound is not reaching that lav at all. That is a design finding (a longer or
narrower chirp buys processing gain) rather than a retake. Say so and stop; do not improvise a
waveform in the room.

## Step 7 — hand over

Leave on the project machine:

- the session directory, with `raw/` untouched;
- `marker-bench-raw.before` and `.after`;
- the event log;
- each candidate's `sync-marker-analysis.json` and `marker-report.json`;
- the paper log, transcribed;
- the phone model, browser and version, and the media-volume step;
- the three WAV SHA-256 values from Step 1;
- a sketch or note of the table layout, and which transmitter was nearest and farthest.

The scoring, the candidate selection, and ADR-0042 follow from those. Sanitized measurements,
commands, hashes and conclusions get written up under `docs/fixtures/` — **never the takes, and
no audio in the repository**.

## What not to do

- **Do not improvise a waveform.** If all three candidates fail, that is the finding, and the
  three-clap procedure in the H1/H2 runbooks is what a real session falls back to.
- **Do not move the phone between the opening and closing blocks.** It is the single mistake
  that produces a confident wrong number instead of an honest inconclusive one.
- **Do not change the media volume mid-take.** Note it once, leave it.
- **Do not edit anything under `raw/`** — not to trim the silence, not to rename a file to
  something tidier.
- **Do not treat a missing marker as a software failure.** A quiet room, a transmitter that
  started late, or a phone too far away are all measurements. The analyzer reports them and
  exits zero on purpose.
- **Do not commit the recordings, the built WAV/HTML files, or the manifest.**
