# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** M6b — Qwen adapter (not started)
- **Branch:** `main`
- **Last closed milestone:** M6a — ROCm environment
- **Gate status at HEAD:** passes, zero skips (8 checks, 2122 tests)
- **Blocked on:** nothing for M6b — its environment is built, locked, and proved on the
  real device. **OQ-008 is answered:** `torch 2.9.1+rocm7.13.0` (HIP `7.13.99004-3309c6114a`)
  on `Radeon 8060S Graphics` / `gfx1151`, bfloat16 and float32 both exact, and the
  `rocm[libraries]` sdist built first time in the FHS shell with no additions to the package
  list M0 guessed.

  **The one thing most likely to cost M6b an afternoon** is not the adapter: it is that
  `[tool.uv.sources]` only routes packages that are also **direct** members of a dependency
  list, and ignores a transitive-only requirement *silently* — no warning, no error, just
  the wrong registry in the lock. `qwen-asr` pulls Gradio, Flask, `nagisa`, `soynlp` and
  Python SoX; if any brings an AMD-only requirement, it must go in the group **and** the
  sources table. `transformers==4.57.6` and `accelerate==1.12.0` are already locked at
  `qwen-asr` 0.0.6's exact pins, so adding it should not relock — if it wants to, something
  moved and that is worth understanding before accepting it.

  **There are two environments now**, and it is load-bearing: `.venv` never contains torch,
  `.venv-rocm` is where the `asr-qwen` group installs from inside the FHS shell. The gate
  runs against `.venv`, which is what keeps INV-05's group-absent case continuously proved.
  **Run the default suite from `.venv-rocm` occasionally — no gate does**, and that is
  exactly where M6a found a real INV-05 breach that was invisible everywhere else.

  **H1 is still the oldest outstanding item.** M6a neither needed nor touched real DJI
  metadata. The 2026-08-02 sample probe answered **OQ-001, OQ-002, OQ-004 and OQ-005**;
  **OQ-004's assumption is false on both halves** — DJI's `bext.time_reference` is not
  samples since midnight and is frame-quantized to 33.3 ms — and **reworking the timing
  model remains the largest known piece of outstanding work in the project**, deliberately
  unscheduled. `rg 'OQ-004'` finds every site. **OQ-007** (`ingest` refuses real 24-bit
  `_orig` files, with a reason that is wrong for 24-bit specifically) is likewise recorded
  and not fixed.

  What H1 still owes: OQ-003's counter across a power cycle, OQ-007's `orig`/`edit` pairing,
  **OQ-012 and OQ-015**, and a second recording from one power-on cycle to confirm OQ-004's
  epoch reading.

  **OQ-017** (what separates real speech from lav bleed at a real table) waits on H2 or a
  first real session; the sample probe took the first real measurements and found level
  separates by an order of magnitude while **correlation does not discriminate at all**.
  **OQ-018** (what Qwen needs at a request boundary) is now **M6b's to answer** — its smoke
  test settles padding, timestamp stability across overlapping requests, and whether a
  low-energy split beats the midpoint. **OQ-019** and **OQ-020** wait on a real session.
  **OQ-021**, new in M6a, asks which render node backs the compute device on a multi-GPU
  host; nothing is blocked on it.

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

Begin **M6b — Qwen adapter**. Read its "What M6a already provides" section first. The seam
is finished and exercised: `transcript/runner.py::_default_transcriber` holds the only
`DEFERRED: M6b` raise, and replacing it reaches **both** `transcribe` and `process` through
one construction site (M5). `dnd-audio mix` needs no adapter at all, so an adapter
regression can never cost a session its audio deliverable. (Claude Code: `/ms-start 6b`.)

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
