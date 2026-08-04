# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M8 — real-session readiness (**in progress**, branch
  `milestone/M8-real-session-readiness`). H1 is still the
  oldest outstanding item, but M8 is what makes recording it worth doing: the
  2026-08-03 `samples` jam-capture run found seven structural defects, one of which can cost a
  whole session and one of which corrupts both deliverables at once.
- **Branch:** `main`
- **Last closed milestone:** M6b — Qwen adapter
- **Gate status at HEAD:** passes, zero skips (8 checks, 2294 tests)
- **Blocked on:** **real recordings, and nothing else.** Every remaining open question that
  blocks anything needs audio rather than code. The MVP's code path is complete: `inspect`,
  `ingest`, `activity`, `mix`, `transcribe` and `process` all run end to end, and
  `transcribe` now produces a real transcript from real speech with no `--fake-models`.

  **What works, in one line each.** `dnd-audio models fetch` installs Silero (~2 MB);
  `./scripts/fetch-models.sh` installs the Qwen ASR model and forced aligner (~6 GB, one
  time, pinned by commit and verified file by file); `dnd-audio doctor` reports tools, disk,
  every model, the GPU, and what your configured device/dtype resolves to; `dnd-audio
  process <session>` runs both branches and produces `session.mp3`, `transcript.json`,
  `transcript.md` and an ingest report naming every model, revision and package version that
  produced them.

  **Two environments, and it is load-bearing.** `.venv` never contains Torch; `.venv-rocm`
  is where the `asr-qwen` group installs, from inside the FHS shell. The gate runs against
  `.venv`, which keeps INV-05's group-absent case continuously *proved* rather than proved
  once. **Run the default suite from `.venv-rocm` occasionally — no gate does.** M6a found a
  real INV-05 breach there, and M6b found six tests asserting a property of the machine
  instead of the code, plus an adapter that started HIP before checking for weights. It is
  the single highest-yield thing to do that no automation does for you.

  **H1 is the oldest outstanding item and is now the only thing gating progress.** The
  2026-08-02 sample probe answered **OQ-001, OQ-002, OQ-004 and OQ-005**; **OQ-004's
  assumption is false on both halves** — DJI's `bext.time_reference` is not samples since
  midnight and is frame-quantized to 33.3 ms. `rg 'OQ-004'` finds every site. **OQ-007**
  (`ingest` refuses real 24-bit `_orig` files, with a reason that is wrong for 24-bit
  specifically) is likewise recorded and not fixed.

  **The timing model is no longer the largest outstanding risk — as of 2026-08-03 it is a
  bounded relabelling.** A jam verification capture
  (`docs/fixtures/2026-08-03-jam-verification.md`) answered the two questions that decided
  it:

  - **OQ-023 — the jam reaches the files.** Two receivers started 5.28 s apart; their
    independently written `time_reference` values agree on that offset to **17–30 ms**,
    inside one 30 fps frame, verified against audio cross-correlation. Cross-receiver
    placement is free from metadata. The absolute value is still meaningless, and still
    does not matter — `session_position` is a subtraction.
  - **OQ-006 — the clocks are stable.** Relative sample-clock drift across all six
    transmitter pairs measures **≈1 ppm**, bounded ±3 ppm on a 30 s baseline. Projected over
    a 4-hour session that is 14–43 ms: **the same size as or smaller than the quantization
    already present at file start.** The cross-track error budget does not grow materially
    with session length, which is what made "each transmitter is an independent recorder
    placed by timecode" viable. H2 still owes the long baseline.

  Total cross-track budget: ~33 ms fixed + ~15–45 ms drift ≈ **80 ms worst case**, against
  syllables of 150–250 ms and lav-geometry spread of 16–48 ms that is already present. The
  one consumer it could hurt is a mix stage that *sums* correlated tracks; M5's automix ducks
  rather than sums, and that is worth not breaking.

  **Two corrections this capture forced.** **OQ-024**: the receiver's frame-rate setting does
  **not** reach an `orig` file — a receiver set to 60 fps wrote `TIMECODE_RATE 30/1` on
  1600-sample boundaries like the 30 fps unit beside it, so the 60 fps instruction has been
  retracted from H1 and 33.3 ms is the floor. And **`bext.origination_time` must never anchor
  a cross-receiver offset**: the two receivers' wall clocks were **48.7 s apart** while their
  timecode agreed to under a frame. Nothing in the code stops that use today; the guard
  belongs with OQ-004's other scoped M1/M2 work.

  **The highest-value code left is a verifier, not a fix.** A *failed* jam produces files that
  look perfectly normal — the 2026-08-02 probe is proof — and nothing at capture or ingest
  time flags it. `session.sync_qa` already correlates tracks; what it does not do is compare
  its measured offset against what `time_reference` predicts and fail loudly on disagreement.
  That turns an invisible operator ritual into a pipeline assertion. **OQ-025** asks whether a
  deliberate acoustic sync signal (a chirp, not a sine) should feed it, and concludes: keep
  the jam as the alignment mechanism, spend the effort on the verifier.

  What H1 still owes: OQ-003's counter across a power cycle, OQ-007's `orig`/`edit` pairing,
  **OQ-015** (receiver displays read against wall clock — unrecoverable afterwards), the
  **third receiver** (OQ-012 is answered for two), six transmitters, and real speech at a
  real table. Breadth and operational questions — no longer an existential one.

  **M6b added a defect to H1's list that only a real model could have found.** The model
  hears an utterance's first word and the transcript does not contain it: the aligner places
  it a few tens of milliseconds before the VAD candidate's ownership interval begins, and
  M4's rule correctly drops it. Five of eleven segments lost their opening word on the sample
  capture. `activity.vad.pad_ms` = 30 is M3's number chosen against synthetic audio,
  registered under **OQ-017**, and a real table is what should move it. **Nothing raises or
  warns** — the symptom is prose that reads fine and is missing a word.

  **OQ-008 answered** (M6a): `torch 2.9.1+rocm7.13.0` (HIP `7.13.99004-3309c6114a`) on
  `Radeon 8060S Graphics` / `gfx1151`, bfloat16 and float32 both exact.
  **OQ-009 answered** (M6b): the package chunks its timestamp path at 180 s, and that path
  is one this project never calls.
  **OQ-022 answered** (M6b): Qwen inference on this stack is reproducible in process and
  across cold processes, so **INV-02 stands unamended**.
  **OQ-018 items 1–3 answered** (M6b); item 4 and the low-energy-split half of item 3 need
  a real session and are H2's.
  **OQ-017**, **OQ-019**, **OQ-020** wait on H2 or a first real session. **OQ-021** asks
  which render node backs the compute device on a multi-GPU host; nothing is blocked on it.

  **Do not run a transcription alongside a heavy ComfyUI or large-LLM workload.** The host
  has unified memory, so a GPU allocation and system RAM come from one pool and
  `systemd-oomd` will kill a user process under sustained pressure — in the worst case an
  hours-long session four hours in. The pipeline bounds its own footprint and cannot bound
  anything else's.

## Milestone status

| ID  | Milestone                  | Status      | Closed at |
| --- | -------------------------- | ----------- | --------- |
| M0  | Foundation                 | closed      | `67b70ed` |
| M1  | Inspection                 | closed      | `fd16931` |
| M2  | Timeline                   | closed      | `f33ad6d` |
| M3  | Activity                   | closed      | `38bc989` |
| M4  | Fake transcript            | closed      | `8556f43` |
| M5  | Automix                    | closed      | `d282688` |
| M6a | ROCm environment           | closed      | `f5c6632` |
| M6b | Qwen adapter               | closed      | `07cebdb` |
| M8  | Real-session readiness     | in progress | —         |
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

`uv run dnd-audio doctor` — now including the GPU, and it **opens** the device nodes rather
than inferring access from group membership. On the target host from the ROCm environment:
`opened /dev/kfd`, `opened /dev/dri/renderD128`, `torch 2.9.1+rocm7.13.0 (HIP 7.13.99004-3309c6114a)`,
`Radeon 8060S Graphics, gfx1151 — verified bfloat16, float32`, and
`auto resolves to cuda:0 / bfloat16`. `--device` and `--dtype` answer whether *your*
configuration works here before a four-hour session finds out during it: an explicitly
requested combination this machine cannot deliver exits 1 with a diagnostic, never a quiet
downgrade to a different precision.

`dnd_audio.runtime` splits probing from resolution — probing imports torch, opens nodes and
runs kernels; resolution is a pure function of what it found — so the spec's whole
device/dtype matrix is tested on a machine with no GPU. The smoke test is per device **and**
per dtype, and compares exactly rather than within a tolerance.

The lock holds exactly five packages from AMD's index and everything else from PyPI, with
`accelerate 1.12.0` in it wanting `torch>=2.0.0` and not getting a CUDA build. It pins
**versions, not bytes**: AMD publishes no hashes (ADR-0025).


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

**Without `--fake-models`, `transcribe` now transcribes.** `_default_transcriber` verifies
both Qwen snapshots, resolves the runtime, loads the pair offline onto `cuda:0` in bfloat16
with SDPA attention, and returns a real `Transcriber` (M6b). On a machine that has not run
`./scripts/fetch-models.sh` it fails naming that command; on a machine without the
`asr-qwen` group it fails naming the `uv sync` that installs it — and under `process`,
either failure costs the transcript and **not** the mix (INV-09).

Weights before hardware, in that order, and it is load-bearing: a session whose model is not
installed cannot run on any device, so checking first avoids spinning up HIP to learn what a
`stat` already knew — and keeps the default suite from importing Torch (INV-05).

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

**H1 — the real DJI hardware fixture.** The MVP's code path is complete; every remaining
question that blocks anything needs audio. Read `docs/plan/milestones/H1-hardware-fixture.md`
— its recording recipe is written for the owner, and the items that cannot be recovered
afterwards (**the receiver displays read against wall clock**, for OQ-012 and OQ-015) are
called out as such. (Claude Code: `/ms-start H1`.)

Nothing in H1 is a code task until the recordings exist. What the pipeline will do with them
on arrival is already built: `inspect` names the strategy, the evidence and the assumption
*by OQ id* in every manifest, so answering several of these is reading one manifest rather
than writing an analysis. `tests/test_qwen_smoke.py` discovers `samples/*.wav` by glob, so
dropping better recordings in re-runs every OQ-018 and OQ-022 measurement M6b took, without
a code change.

**If code is wanted before the recordings arrive**, the two known defects are both recorded
and both deliberately unscheduled: **OQ-004's timing model** (the largest known piece of
outstanding work in the project — `rg 'OQ-004'` finds every site) and **OQ-007** (`ingest`
refuses real 24-bit `_orig` files with a reason that is wrong for 24-bit specifically).
Neither should be attempted without deciding first what the real fixture is expected to say.

**`pytest-xdist` parallelism is done** (2026-08-03), outside any milestone because it
touches every milestone's tests. **The suite went from 120 s to ~30 s and the whole gate
from ~2m20s to ~30 s.** No test or source file changed.

It landed in **`addopts` in `pyproject.toml`, not in `./scripts/gate.sh`** — which is a
different decision from the one queued here, and the more important half of the change.
In the gate script it would have been a rule held by convention: the gate parallel, and
every ad-hoc `uv run pytest` an agent types mid-implementation still serial. In
configuration it is a mechanism, and it reaches Codex, an editor's runner and a human at
a shell without any of them knowing it exists. The gate now passes no `-n` at all and
defers to that number, so only one place states it. `PYTEST_WORKERS=<n>` still retunes
the gate; `-n 0` on any command line forces a serial in-process run, which is what a
debugger or `-s` needs.

The predicted flake risk — a pair of tests passing only because of shared
`tmp_path_factory` ordering — **did not materialize**. Both session-scoped fixtures in
`tests/conftest.py` build into their own `tmp_path_factory` directory, so each worker
gets its own copy and there is nothing to share across a process boundary; a dozen full
parallel runs at worker counts from 4 to 32 were green, 2122 passed every time.

Three things measured that the estimate above got wrong, worth knowing before tuning
this again. The box has **32 cores, not 16**. The speedup ceiling is **not core count**:
the serial run already burned **8m44s of CPU inside 2m of wall clock**, because the
pipeline stages thread internally, so the curve is flat from roughly 8 workers up — 8,
12, 16 and 32 all land within noise at 30–37 s. And what is *not* flat is **worker
startup**, which every invocation pays and which only matters once parallelism is the
default everywhere: on one fast unit file `-n 8` costs 0.7 s where `-n auto` costs
1.8 s. Hence 8 rather than `auto` — the whole speedup at a quarter of the fixed cost,
against mildly oversubscribing a 4-core machine.

**Real DJI metadata has still not been validated.** Six milestones have now been built on
assumptions H1 would settle. M6a is the first that neither needed nor touched them.
