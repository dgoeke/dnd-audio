# M10 marker bench protocol

**Bench completed 2026-08-05.** Cand-b won and is frozen as marker v1 by ADR-0042. The
sanitized result is in `fixtures/2026-08-05-marker-phone-dji-bench.md`. The procedure below is
retained as the reproducible capture record; production operators use
`dnd-audio marker build OUTPUT_DIRECTORY` without the hidden candidate option.

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

---

## The short version

Everything below this card is reasoning. This is the whole procedure, assuming transmitters and
receivers are already placed and timecode-jammed.

**Before recording**

1. Three pages on the phone, one per candidate, open in three tabs. Airplane mode on, Bluetooth
   off.
2. Play one marker. Set media volume: clearly audible across the table, not unpleasant at arm's
   length. **Then stop touching the volume.**
3. Phone in the middle of the table, screen up, screen kept awake. Note which transmitter is
   nearest it and which is farthest.

**The take — one continuous recording, ~14 minutes**

Leave about five seconds between plays. Say the bold lines out loud; they are the only record of
what happened.

| | say | do |
| --- | --- | --- |
| 1 | **"Opening block."** | `cand-a` ×3, `cand-b` ×3, `cand-c` ×3 |
| 2 | — | **wait ~10 minutes.** Leave the room. Touch nothing |
| 3 | **"Closing block, phone has not moved."** | `cand-a` ×1, `cand-b` ×1, `cand-c` ×1 |
| 4 | **"Diagnostic block, moving the phone now."** | move the phone, say roughly where, `cand-a` ×1 — then move again, say where, `cand-a` ×1 |
| 5 | — | five seconds of silence, stop recording |

Fourteen plays. If you bump the phone at any point, **say so out loud immediately** — that costs
nothing and saves the take.

**Afterwards**

6. Copy the originals into `raw/tx-a/` … `raw/tx-f/`, bytes and filenames unchanged.
7. Tell me which transmitter was nearest the phone and which was farthest.

That is everything. Steps 5–8 below are mine, not yours.

---

## What this bench decides

Three candidate waveforms remain in the code as history. Before this bench, **none was `v1` and
`marker build` with no `--marker` refused** so nobody could record against an untested guess.
This bench selected cand-b and added a separate public `v1` entry with the same waveform bytes.

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

That is the whole list. No second person, no media, no script, **no written log and no
stopwatch** — see [Why there is no event log](#why-there-is-no-event-log).

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

**Once everything is placed, do not touch any of it until Block 4.** That — the claim that the
phone and all six transmitters were in the same places for the opening and closing blocks — is
the one thing about this bench that no recording and no analysis can establish on its own. It is
the operator's assertion, and here it is made by saying so on the take. Without it, a start-to-
end change is differential acoustic arrival and nothing stronger, which is the correct outcome
rather than a limitation.

## Step 4 — the take

Start every transmitter and confirm every recording indicator. **One continuous take.** Leave
several seconds of silence between every play so occurrences never overlap.

**The order of the blocks is the record.** Nothing needs writing down, but the sequence below
has to be followed as written, because the analysis afterwards reads the structure rather than
a log.

### Block 1 — opening — nine plays

Say the block aloud so the audio carries its own slate: *"Opening block."*

Then, tab by tab:

| | | |
| --- | --- | --- |
| plays 1–3 | `cand-a`, three times | |
| plays 4–6 | `cand-b`, three times | |
| plays 7–9 | `cand-c`, three times | |

**This block is the bench.** Everything after it is a bonus that costs almost nothing. Three
plays of each is what makes same-position repeatability measurable, and repeatability is what
tells us whether a later change means anything.

### Block 2 — leave it running

Ten minutes or so. **Do nothing.** Leave the room, make coffee, work on something else. Do not
touch the phone or the transmitters.

This gap exists so the closing block is a genuinely later measurement. Three minutes is enough
to select a candidate; ten gives a usable drift figure as well, since the ≈1 ppm this project
measured (**OQ-006**) accumulates to about 29 samples over ten minutes and the detector reports
integer samples. Longer is better and costs only patience.

### Block 3 — closing — three plays

**The phone has not moved and must not have been rotated, picked up, or nudged.** If it was, say
so out loud on the recording — an unrecorded move is the one mistake that silently corrupts the
measurement this bench exists to make, and it is the single thing here that no amount of
analysis can recover afterwards.

Say: *"Closing block, phone has not moved."* Then **one** play of each candidate, same order.

One is enough here: the opening block already established repeatability, and this block only has
to anchor the other end of the comparison.

### Block 4 — the moved-phone diagnostic — two plays

**This block comes last, and the ordering is not negotiable.** Moving the phone before the
closing block would put a real geometry change inside the start-to-end comparison, and nothing
downstream could tell.

Say: *"Diagnostic block, moving the phone now."* Then:

1. Move the phone somewhere clearly different — one end of the table, or off to a side. Say
   roughly where, out loud. Play `cand-a` once.
2. Move it again, somewhere else. Say where. Play `cand-a` once.

One candidate is enough: this proves the analyzer *enumerates* a differing lag without calling
it drift, which is a claim about software, not about the waveform.

Saying it aloud rather than writing it down is the whole trick: **the slate is on the recording,
on all six tracks, timestamped by construction, and impossible to lose.**

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

## Why there is no event log

`marker analyze` accepts an optional `--event-log`: a YAML file naming what was played, when,
in what order, and under which asserted geometry. **This bench does not write one**, and that is
a deliberate trade rather than a corner cut.

What the log would supply, and where it comes from instead:

| the log would say | without it |
| --- | --- |
| which times to search | search the whole take — `--start-window-s`/`--end-window-s` wide open |
| which waveform was played | the occurrence itself; each detector finds only its own marker |
| which play was the start and which the end | **the block order**, which the protocol fixes and the spoken slates mark |
| that the phone did not move between them | **the spoken slate**, in your own voice, on all six tracks |

So the two things no audio can establish — the pairing and the geometry assertion — are still
operator testimony. They just arrive as a sentence said out loud onto the recording instead of a
YAML file typed afterwards, which is more reliable rather than less: it is timestamped by
construction, captured six times over, and cannot be reconstructed wrongly from memory a day
later.

**What it costs.** Run without a log, `marker analyze` will not name a start/end pair — its
fallback rule needs exactly one occurrence in each default window, and three plays per block
means it will not fire. So it emits no `differential_arrival` or `clock_drift_evidence`
classification. That comparison is arithmetic over two groups' per-track lags, and it gets done
by hand against the block structure; the classification is a label on a subtraction, not a
measurement that is lost.

**What it does not cost.** Everything the bench actually exists to measure is untouched: every
occurrence on every track, per-track scores, clipping and weak-signal flags, exact integer
relative lags within each group, source coordinates, and the timecode cross-check — which uses
the first group whether or not it carries a role. Analysis runs clean and exits zero.

If a later real session wants the automatic drift classification, that is the moment to write a
log. A bench that selects a waveform does not need one.

## Step 6 — analyze

```bash
dnd-audio inspect  <session>
dnd-audio ingest   <session>
dnd-audio marker analyze <session> --marker cand-a --start-window-s 1200 --end-window-s 1200
dnd-audio marker analyze <session> --marker cand-b --start-window-s 1200 --end-window-s 1200
dnd-audio marker analyze <session> --marker cand-c --start-window-s 1200 --end-window-s 1200
```

The windows are clamped to half the session each, so those two numbers mean "search all of it".

Each run overwrites `work/sync-marker-analysis.json` and `output/marker-report.json`, so **copy
each candidate's pair aside before running the next**:

```bash
cp work/sync-marker-analysis.json  ../marker-cand-a-analysis.json
cp output/marker-report.json       ../marker-cand-a-report.json
```

Each run also prints a `marker_roles_unassigned` warning naming how many occurrences it found
and why it labelled none of them. That is the expected outcome without a log, not a problem.

`occurrences` is a flat list of `(track, play)` pairs, so on six continuously recording
transmitters expect **6 × 4 = 24** for `cand-b` and `cand-c` (three opening plays, one closing)
and **6 × 6 = 36** for `cand-a` (two more from Block 4). Fewer means some track did not hear
some play, which is exactly the measurement — read `groups[].members[].outcome` to see which.

**Anything you did not play is a finding.** Each detector matches only its own waveform — three
candidates in one take were checked and produced zero cross-detections — so an occurrence at a
time you were silent means something. Write down what you got; do not reconcile it.
## Step 7 — before you pack up

Run Step 6 for **one** candidate while the transmitters are still in place. It takes seconds.
Read `work/sync-marker-analysis.json` and check:

| look at | what it should say | if it does not |
| --- | --- | --- |
| `occurrences` | one entry per track per play — six tracks × six plays for `cand-a` | if a track is absent everywhere, its recording, its placement, or the volume step is wrong |
| `occurrences[].clipped` | `false`, especially on the nearest transmitter | lower the media volume one step and redo Blocks 1 and 3 |
| `occurrences[].weak` | `false`, especially on the farthest | raise it one step and redo Blocks 1 and 3 |
| `occurrences[].score_permille` | record the value on every track; frozen v1 later uses 300 | a low value at the farthest seat is **not** necessarily a candidate failing — see the false-positive headroom above — but it is worth knowing while the room is still set up |
| `notes` and the report's `warnings` | read them | they name what the analysis could not conclude and why |

The volume-step retakes are the only ones worth redoing on the spot. Everything else can be
decided at a desk.

**If all three candidates score low at the farthest seat**, do not conclude they failed. The
threshold has room to move down and the sweep says so. What would be a real failure is a marker
the detector cannot find *at any threshold* — no accepted sequence and no strong chirps either —
which means the sound is not reaching that lav at all. That is a design finding (a longer or
narrower chirp buys processing gain) rather than a retake. Say so and stop; do not improvise a
waveform in the room.

## Step 8 — hand over

Leave on the project machine:

- the session directory, with `raw/` untouched;
- `marker-bench-raw.before` and `.after`;
- each candidate's `sync-marker-analysis.json` and `marker-report.json`;
- the phone model, browser and version, and the media-volume step;
- the three WAV SHA-256 values from Step 1;
- which transmitter ended up nearest the phone and which farthest.

That last line is the only thing here that is not already on disk or on the recording, and it is
what turns "`tx-d` scored lowest" into "the farthest seat scored lowest". A rough sketch of the
table is better still.

Everything else — what was played, in what order, and whether the phone moved — is in the audio,
in your own voice. The scoring, the candidate selection, and ADR-0042 follow from those.
Sanitized measurements, commands, hashes and conclusions get written up under `docs/fixtures/` —
**never the takes, and no audio in the repository**.

## What not to do

- **Do not improvise a waveform.** If all three candidates fail, that is the finding, and the
  three-clap procedure in the H1/H2 runbooks is what a real session falls back to.
- **Do not move the phone between the opening and closing blocks.** It is the single mistake
  that produces a confident wrong number instead of an honest inconclusive one — and with no
  written log, the spoken slate is the only thing that would reveal it. If it happens, say so
  out loud immediately; a recorded "I just bumped the phone" costs nothing and saves the take.
- **Do not change the media volume mid-take.** Set it once, say the step aloud, leave it.
- **Do not edit anything under `raw/`** — not to trim the silence, not to rename a file to
  something tidier.
- **Do not treat a missing marker as a software failure.** A quiet room, a transmitter that
  started late, or a phone too far away are all measurements. The analyzer reports them and
  exits zero on purpose.
- **Do not commit the recordings, the built WAV/HTML files, or the manifest.**
