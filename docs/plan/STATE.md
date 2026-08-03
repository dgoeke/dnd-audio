# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M6a — ROCm environment (not started)
- **Branch:** `main`
- **Last closed milestone:** M5 — Automix
- **Gate status at HEAD:** passes, zero skips (8 checks, 2028 tests)
- **Blocked on:** nothing for M6a — it is environment work whose dependencies are all closed.
  **The spec changed in M5** and this is the one place it is easy to miss: the true-peak
  ceiling now outranks the `-16 LUFS` loudness target where the two are mutually unreachable
  (acceptance criterion 8, ADR-0023). The code already behaved that way and the spec did not
  say so. On the canonical fixture through real Silero this is not hypothetical — the mix
  measures ~40 LUFS below target and the run says so and exits zero.

  **H1 is still the oldest outstanding item**, but it is no
  longer the whole of the problem it was. A 2026-08-02 sample probe — four real DJI Mic 3
  transmitters, ~47 s, not the H1 fixture — answered **OQ-001, OQ-002, OQ-004 and OQ-005** from
  metadata alone, and the design that made that cheap is M1's: the manifest names the strategy,
  the evidence, and the assumption by `OQ-` id, so settling them was reading one manifest.

  **One of those answers is bad news and is not yet acted on. OQ-004's assumption is false on
  both halves**: DJI's `bext.time_reference` is *not* samples since midnight (a 19:26:55 file
  carries 388 seconds' worth, and the later-created pair carries a *smaller* value), and it is
  frame-quantized to **33.3 ms** rather than sample-accurate. M1 and M2 both encode the old
  reading. Absolute wall-clock placement from a BWF reference alone is not available on this
  hardware, and cross-receiver alignment from it is meaningless because the epochs differ.
  Nothing is corrupted meanwhile — INV-12 forbids inventing timing, a wrong epoch shifts a
  session uniformly rather than scrambling it, and the clap-sync QA exists for the
  cross-receiver case — but **reworking the timing model is now the largest known piece of
  outstanding work in the project**, and it is deliberately unscheduled rather than folded into
  a milestone that did not ask for it. `rg 'OQ-004'` finds every site.

  Also from the probe: **`ingest` refuses real 24-bit `_orig` files** and the reason it gives is
  wrong for 24-bit specifically (`s24 → f32` is lossless — verified over 2M values). ADR-0011's
  principle is intact, its guard is too broad. Recorded under **OQ-007**, not fixed.

  What H1 still owes: OQ-003's counter across a power cycle, OQ-007's `orig`/`edit` pairing,
  **OQ-012 and OQ-015** (receiver displays against wall clock — still unrecoverable afterwards),
  and a second recording from one power-on cycle to confirm OQ-004's epoch reading.

  M3 added a second kind of waiting, and it is not H1's kind. **OQ-017** — what separates
  real speech from lav bleed at a real table — needs H2 or a first real session, because a
  two-minute metadata fixture cannot tune a bleed threshold. Every VAD, bleed, and scoring
  default cites it, and the pipeline already records the numbers that answer it, so this is
  reading one session's graph rather than running an experiment. Nothing is blocked on it:
  the thresholds work on synthetic audio and the gate is conservative by construction.

  The sample probe took the **first real measurements** against it, from a deliberately harder
  geometry than a table: real bleed sat 18–22 dB below the held mic while two mics hearing the
  same voice sat ~1 dB apart — an order of magnitude, with room to spare. The surprise is that
  **correlation does not discriminate** (814–913‰ for bleed, 866–901‰ for genuine co-incidence,
  overlapping ranges): it confirms two lavs heard one room, and *level* is what says whose lav
  it is. ADR-0014's margin **and** correlation **and** veto is what keeps that from being read
  as noise. Peak lag was 7–11 ms where air explains under 2 — a zero-lag correlator would have
  found none of it.

  M4 added a third, and it is M6b's rather than a room's. **OQ-018** — what Qwen3-ASR and its
  aligner need at a request boundary — covers padding, timestamp stability across two
  overlapping requests, whether a low-energy split beats the midpoint, the retry budget, and
  the text-similarity thresholds. M6b's smoke test settles the first three directly; the last
  needs a real session, or one utterance genuinely heard on two transmitters. Nothing is
  blocked: M4 is correct under whatever the configured values are, and only the *defaults*
  are guesses. `rg 'OQ-018'` finds all twelve sites at once.

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `67b70ed` |
| M1  | Inspection                 | closed      | `fd16931` |
| M2  | Timeline                   | closed      | `f33ad6d` |
| M3  | Activity                   | closed      | `38bc989` |
| M4  | Fake transcript            | closed      | `8556f43` |
| M5  | Automix                    | closed      | `PENDING` |
| M6a | ROCm environment           | not started | —         |
| M6b | Qwen adapter               | not started | —         |
| H1  | Hardware fixture (2 min)   | not started | —         |
| H2  | Drift soak / first session | not started | —         |
| M7  | Archival (sketch)          | sketch      | —         |

**Closed at** is the milestone's close commit, and it is recorded by a small follow-up
commit — a commit cannot contain its own hash (the same limit ADR-0003 names for the report).
M0–M3 each wrote theirs by amending instead, so those four SHAs are the pre-amend close commit
and do not resolve in a fresh clone. Left as they are rather than rewritten history; from M4
on the column is reachable.

Status values: `not started` → `in progress` → `verified` → `closed`.
`blocked` is also valid; say what on. `sketch` means a charter exists to hold the
idea but the work is deliberately unplanned.

## What works end to end

`uv run dnd-audio process /path/to/session --fake-models` — **every applicable stage**, which
is the whole of the spec's stage DAG on synthetic input.

One snapshot of `raw/`, activity performed once, then the mix branch and the transcript branch
attempted **independently** — each in its own handler, so a failure in either collects an
error rather than short-circuiting the other — then one unconditional `verify_unchanged`
before the report is finalized. On the canonical fixture: six stages complete, four transcript
segments, `session.mp3`, both transcript deliverables, 18 cache hits and 12 misses. A
transcription failure still yields the MP3 and the report with `transcribe` and `render`
marked failed and exit 4; a *mix* failure still yields the transcript, because independence is
a property of the control flow rather than of the ordering.

Without `--fake-models` it raises the `DEFERRED: M6b` `NotImplementedError` **before any
work** (ADR-0005). An operator who wants the audio branch on such a host runs `mix`, which
needs no ASR adapter at all.

`uv run dnd-audio mix /path/to/session` — the right branch of the DAG, and the one that must
survive a transcription failure (INV-09, whose enforcement M5 owns).

It does everything `activity` does, then estimates a per-track voice-level correction from
each track's own `speech_reference_mbfs` — median target, clamped, a missing reference
corrected by **zero** and warned about — turns the graph's **retained** candidates into a gain
per track per 1 kHz control frame (two weight floors, a slew-limited linear ramp, a
Dugan-style normalized share), interpolates that onto samples, steps six `TrackReader`s and
the envelope over one window range into a streamed mono float32 intermediate under
`work/cache/mix/`, verifies INV-01 a second time, commits the mix cache, then measures,
encodes at 128 kbps mono, decodes, measures again, and retries within a bounded budget.

On the canonical fixture through the **real** Silero release: 10.500 s, mono, 128 kbps,
−39.7 LUFS, −3.0 dBTP, one encode attempt, exit 0 — with
`mix_loudness_target_unreachable` warning that reaching −16 LUFS would need +24.5 dB and the
ceiling allows +1.6. That is the correct answer: Silero finds no candidates in synthetic noise
(INV-10), so every track sits at the room-tone share and the mix is a quiet six-way blend. The
intermediate is byte-identical on rerun and reused from cache; the MP3 is regenerated every
run and is not claimed to be byte-stable.

The mixer imports nothing from the transcript layer, proved over the **transitive** import
closure in a subprocess; the intermediate's bytes do not move when `transcribe` runs, and do
not move when every `ActivityDecision.detail` and `ActivityNote.message` in the graph is
rewritten.

`uv run dnd-audio transcribe /path/to/session --fake-models` — the left branch of the
spec's stage DAG: inspection, the timeline, who was speaking, what they said, and both
transcript deliverables.

It does everything `activity` does, then plans ASR requests from the graph's **retained**
candidates only — merging adjacent regions, padding each core, and capping the *padded*
waveform at `max_segment_s` — submits one window at a time (INV-07), resolves a truncated
response by splitting the unpadded core at its quietest interior frame within a global
submission budget, assigns each word to the ownership interval containing its start, collapses
duplicates on overlap **and** similar text **and** the graph's own acoustic evidence, verifies
INV-01 a second time, commits the ASR cache, and writes `work/transcript-records.json`,
`output/transcript.json`, `output/transcript.md` and one report covering five stages.

On the canonical fixture: 4 segments across 4 speakers, 0 collapsed, 2 marked as overlap, all
three deterministic artifacts byte-identical on rerun, 29 ASR cache hits and 0 misses warm.
Alice's line bleeds into four tracks and the scripted ASR is told to transcribe it there;
every copy is gone before a word is submitted, because M3's gate suppressed the candidate.

`uv run dnd-audio render /path/to/session` regenerates both deliverables from the records
alone — proved by deleting the graph, the timeline and the whole cache tree first, with a spy
asserting no model is constructed. Absent records exit nonzero naming
`transcript_records_missing`, and still write a report.

Without `--fake-models`, `transcribe` raises the `DEFERRED: M6b` `NotImplementedError` naming
the missing adapter rather than writing a report that says the session is broken (ADR-0005).

`uv run dnd-audio activity /path/to/session` — inspection, the timeline, then who was
speaking.

It does everything `ingest` does, then runs a VAD per track over the cached 16 kHz
derivative behind an `ActivityDetector` protocol, measures every overlapping candidate pair
with a lag-tolerant normalized speech-band cross-correlation, scores each candidate on four
terms (track-relative level, VAD confidence, cross-track dominance, correlation as
*independence*), applies a bleed gate that suppresses only on margin **and** correlation
**and** a track-relative veto, verifies INV-01, commits four caches at one moment, and
writes `work/activity.json` plus one `output/ingest-report.json` covering three stages.

On the canonical fixture through the real pinned model: 6/6 tracks, 0 candidates — the
correct answer, because the fixture's speech is synthetic noise and INV-10 forbids expecting
a learned release to fire on audio no human made. Attribution is proved against the
deterministic scripted detector over the fixture's declared truth; the real model is
exercised once under `host_smoke` on claims true of any release.

`uv run dnd-audio ingest /path/to/session` — inspection, then the timeline.

It discovers and hashes the session's sources every run (served warm from M1's content
cache), rasterizes each source's timing evidence onto the 48 kHz grid with exact rational
arithmetic, decides where session zero is and which day each chunk belongs to, lays out
each track's chunks with real gaps preserved as explicit silence, builds cached 16 kHz
derivatives through one checked-in FIR, and writes `work/timeline.json` plus
`output/ingest-report.json`. On the canonical fixture: 6/6 tracks, 10.500 s aligned
(504000 samples), byte-identical on rerun, 18 cache hits and 0 misses.

The 48 kHz working path is the **segment map**, not files — `TrackReader` serves bounded
windows over it, and nothing in the pipeline may depend on `--materialize-48k` having run.

`uv run dnd-audio inspect /path/to/session` still does the first half alone, writing the
manifest and the content-hash-addressed sidecars holding exactly the bytes FFprobe wrote.

`uv run dnd-audio models fetch` downloads the Silero VAD model, pinned by release `v6.2.1`,
commit `7e30209a`, and sha256, verified before the file is moved into place. It is the only
command permitted to touch the network (INV-06). `doctor` reports its availability alongside
tool versions, writable paths, and free disk.

`uv run python scripts/make_fixture.py <dir>` materializes the canonical six-transmitter
synthetic session, and nine variants exercise one deviation each: a 44.1 kHz source, a
track whose chunks disagree about their rate, a material overlap, no configured origin,
midnight rollover, 29.97 drop-frame, drift, bleed delayed 25 ms, and two genuine
simultaneous speakers at unequal levels each carrying the other's bleed. No audio binaries
are in the repository.

**No command is a stub any more.** The only remaining exit-3 path is the missing ASR adapter
that `transcribe` and `process` need without `--fake-models`, and it names M6b.

Underneath, from M0: validated `session.yaml` models, checked-in JSON Schema artifacts
with a drift test, exact rational frame rates, model seams with scripted fakes, a report
writer that cannot lose a stage, and a test suite that is provably offline.

## Next smallest step

Begin **M6a — ROCm environment**. It is the only milestone whose dependencies are all closed,
and it is pure environment work: the AMD `gfx1151` Torch wheel index wired into uv with
per-package sourcing, the separate FHS shell for the `rocm[libraries]` sdist (ADR-0002),
locked versions, and `doctor` device checks. Nothing in M0–M5 depends on it and M6b cannot
start without it. (Claude Code: `/ms-start 6a`.)

Start with `doctor`'s device checks rather than with the wheel index. The checks are what tell
you whether the index worked, and writing them second means debugging two unknowns at once.
They must **open** `/dev/kfd` and the render node rather than testing that the paths exist —
the charter says so and it is the whole difference between a check and a guess.

Read M6b's new "What M5 already provides" section when you get there. The short version:
`process` composes the transcript branch through `perform_transcript`/`resolve_models` rather
than reimplementing it, so replacing the `DEFERRED: M6b` raise site reaches both commands
through one seam — and `dnd-audio mix` needs no adapter at all, so an adapter regression can
never cost a session its audio deliverable.

**Real DJI metadata has still not been validated.** Acquiring the H1 fixture is the oldest
outstanding item in the project, and five milestones have now been built on assumptions it
would settle. M2 added OQ-015 to what it must settle; M3 added **OQ-017**, M4 added
**OQ-018**, and M5 added **OQ-019** and **OQ-020** — none of which H1 can answer. A two-minute
metadata fixture cannot tune a bleed threshold, a text-similarity threshold, an automix
constant, or an encoder's real overshoot. OQ-017 and OQ-019 wait for H2 or a first real
session; OQ-018's first three parts are M6b's smoke test; OQ-020 is answered by encoding one
real session once, because every attempt's measurements are already retained in the report.

**Nothing in the pipeline is blocked on any of them.** Every default is conservative by
construction, two configuration validators refuse a combination the gain rule cannot deliver,
and the encode stage fails rather than claiming a compliance it did not measure.
