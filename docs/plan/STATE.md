# Project state

The single source of truth for "where are we". Updated at the start and close of
every milestone. Keep it short — detail belongs in milestone closeouts and ADRs.

---

## Right now

- **Current milestone:** H1 — hardware fixture (**not started**). The minimal
  acoustic direction capture is complete enough to guide later event-first work but does not
  close or replace H1. M7b deliberately retains the post-session publication, retention, cache,
  and deletion decisions.

  **M10 is closed.** The intended-phone/six-DJI bench selected cand-b as marker v1; ADR-0042
  freezes its exact waveform, hash, detector constants, and margins. H1/H2 may use the offline
  phone player as an alternative to the three-clap pattern. It remains acoustic verification,
  never timecode replacement or automatic correction.
- **Branch:** `milestone/M10-acoustic-sync-marker`
- **Last closed milestone:** M10 — acoustic synchronization marker
- **Gate status at HEAD:** passes, zero skips (8 checks, 3,111 tests); M10's real-DJI
  false-positive host smoke also passes 8 tests. `main` was last verified
  the same way at 2 656 tests. The same default suite passes from `.venv-rocm` — **re-sync that
  environment after any dependency change**, which M7a's first attempt there proved by failing
  five ways on two missing packages.
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

  **M8 closed every structural defect the jam capture found.** Real 24-bit `orig` sources now
  ingest bit-exactly; `time_reference` is a frame-quantized recorder-domain counter rather
  than invented midnight; wall-clock tags are descriptive only; a source-quantum rounding
  overlap places cleanly; and `sync_qa` compares acoustic and metadata offsets while keeping
  constant offset, drift, weak evidence and no signal distinct. The activity reference uses
  attributed winners with an overlap-only fallback, the mix warning names its real inputs,
  and three-way duplicates resolve best-source-first under a separately versioned assembly
  semantic.

  **The diagnostics now make H1/H2 evidence rather than a listening anecdote.** Per-track
  activity counts and references are explicit. Every dropped `(request, word)` pair reports
  exact edge-distance geometry, side and word position. OQ-027's initial seconds-scale causal
  claim was corrected: production damage is bounded by the 500 ms request padding.

  **The 30/50/100 ms real-model A/B did not produce a new default.** At 30 ms there were
  30 dropped pairs and 10 rendered segments; at 50, 26 and 12; at 100, 21 and 12. The 100 ms
  run retained all four known direct-source openings, but also retained more wrong-track/short
  fragments and worsened two speech-reference clamps by up to 1.44 dB. `vad.pad_ms` stays at
  30. H1 records hard-onset phrases against intended track ids and compares 30 with 100; H2 or
  a real table decides.

  **M9 recovers transcript edges without moving activity.** A 20 ms leading ownership grace is
  applied after ASR and bounded by each submitted occurrence. Conservative contained-fragment
  collapse removes only a proper word-sequence fragment under full graph evidence and decisive
  source dominance. Granular records remain authoritative while JSON and Markdown share
  lineage-preserving presentation joins. On the four-file replay, all intended openings remain,
  long bleed fragments disappear, the final phrase renders coherently, and both unresolved
  one-word `Okay` copies remain. Activity, mix and ASR cache identity are unchanged.

  **The next architecture question is recorded, not decided.** The minimal two-person capture
  tests whether joint waveform evidence and session-local wearer information can represent one
  voice heard on several microphones as one latent event while retaining two people saying the
  same short word as two events. A successful result would justify a separately chartered
  event-first software milestone; weak evidence leaves M9's conservative canonical transcript
  in place and may justify only a traceable editorial/LLM view. No production default changes
  from the minimal corpus.

  **Archival is now split at the authority boundary.** M7a may explicitly upload byte-exact
  zstd archives to an owner-controlled private cold bucket, but only through an archive
  command that fully downloads and restores every hash before committing its manifest. It has
  no delete or publishing command. M7b waits for an accepted real session before deciding
  public delivery, retention, cache reclamation, or the INV-01 exception local deletion would
  require. A four-file zstd trial saved 30.4% and restored every original SHA-256 exactly.

  **M7a was closed once without a verify phase, and the phase found two P0s when it was
  run.** `archive verify --report` pointed at a recording overwrote it with JSON — the CLI's
  INV-01 guard was conditioned on having a session directory, which the two remote-only
  commands never have. And one `503 Slow Down` on a single PUT stored a zero-byte object at
  an immutable content-addressed key, which nothing may overwrite and no command may delete,
  making that session permanently unarchivable. Both are fixed, mutation-checked, and now
  covered by tests that enter through the CLI — **which nothing did before**: the whole of
  `_run_archive` was untested, including the block the first review's P0 fix lived in. The
  cost of skipping a phase was two ways to lose the recordings this milestone exists to
  protect. Full account in `docs/plan/reviews/M7a-code-20260804-2109.md`.

  **The host smoke has now been run against the real bucket on the final code** — the one
  gate criterion whose proof had predated the last round of fixes. 9 passed: upload, `list`,
  `verify`, delete the session directory, remote-only whole-session restore, track-scoped
  restore, forced multipart. It failed first, on a defect in the test rather than the
  archive: two pagination tests shared bucket state across xdist workers and had been
  passing on the previous run's leftover objects. Both now seed their own, and the fix was
  proved by emptying the prefix rather than by re-running on top of it.

  **The acoustic marker is bench-validated and frozen as v1.** `dnd-audio marker build` writes one
  canonical PCM WAV plus a standalone offline phone page embedding those exact bytes, and
  `dnd-audio marker analyze` finds the full chirp sequence at integer-sample positions on every
  track. Cand-b's 800 Hz–6 kHz, 250 ms sweeps won: all four fixed plays reached all six tracks,
  weakest score 404 permille, 0–1 sample opening repeatability, 29-sample maximum gap error,
  and no clipping, weak signal, ambiguity, or extra event at approximately 90% phone volume.
  The canonical WAV SHA-256 is
  `70355baad6bb72b38e0b606cddbbaa3428c11429bec74cd127aa6f8935ecdf6f`.
  It remains jam QA, never timeline authority. A normal session reports differential arrival;
  only fixed source **and** lav geometry, asserted in an event log, licenses a drift claim
  (ADR-0040). H1/H2 retain claps as the fallback.

  **A measurement worth retaining:** detection survives about 1000 ppm of
  playback speed error and fails by 2000 ppm, and what breaks is per-chirp correlation
  detuning rather than the inter-chirp gap tolerance the constant's name suggests. It scales
  with the chirp's time-bandwidth product, so the candidates trade reach against timing
  tolerance: `cand-b`'s long sweeps carry farthest, `cand-c`'s short ones tolerate the most
  browser resampling (**OQ-029**).

  **The jam/timing result remains strong.** Two receivers started 5.28 s apart and their
  independently written references agree on that offset to 17–30 ms, inside one 30 fps frame
  (**OQ-023**). Relative sample-clock drift measured ≈1 ppm, bounded ±3 ppm over 30 s
  (**OQ-006**); H2 still owes the long baseline. A receiver set to 60 fps still wrote 30 fps
  quantum boundaries (**OQ-024**), and two receivers' wall clocks were 48.7 s apart while the
  jammed counter agreed — the case M8 now guards against.

  What H1 still owes: OQ-003's counter across a power cycle, OQ-007's `orig`/`edit` pairing,
  **OQ-015** (receiver displays read against wall clock — unrecoverable afterwards), the
  **third receiver** (OQ-012 is answered for two), six transmitters, and real speech at a
  real table. Breadth and operational questions — no longer an existential one.

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
| M8  | Real-session readiness     | closed      | `8ad15e3` |
| M9  | Transcript assembly quality | closed      | `d3e2cbb` |
| M7a | Verified private raw archive | closed      | `5bc24a3` |
| H1  | Hardware fixture (2 min)   | not started | —         |
| H2  | Drift soak / first session | not started | —         |
| M7b | Publishing and reclamation | sketch      | —         |
| M10 | Acoustic sync marker       | closed      | `38a18c9` |

**Closed at** is the milestone's close commit, and it is recorded by a small follow-up
commit — a commit cannot contain its own hash (the same limit ADR-0003 names for the report).
M0–M3 each wrote theirs by amending instead, so those four SHAs are the pre-amend close commit
and do not resolve in a fresh clone. Left as they are rather than rewritten history; from M4
on the column is reachable.

M7a was closed once at `69e583c` without a verify phase, reopened when the phase was run, and
closed again at the SHA above. The column names the **final** close, so a bisect for the
boundary lands after the fixes rather than before them.

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

**Record H1 — six transmitters, three receivers, two human voices, and the exact runbook.**
Marker v1 is prepared for its start/end landmarks; the three-clap pattern remains the fallback.
The unrecoverable evidence is still the receiver-display/wall-clock observation, the real
receiver/channel map, spoken ground truth, power-cycle order, and any phone/lav movement.

**Archive the first real session as soon as it is inspected.** `dnd-audio archive upload
<session>`, then `dnd-audio archive verify --session-id <id>` — the second is what turns the
backup from a belief into a fact, and it is a full download by design.

Nothing in H1 is a code task until the recordings exist. What the pipeline will do with them
on arrival is already built: `inspect` names the strategy, the evidence and the assumption
*by OQ id* in every manifest, so answering several of these is reading one manifest rather
than writing an analysis. `tests/test_qwen_smoke.py` discovers `samples/*.wav` by glob, so
dropping better recordings in re-runs every OQ-018 and OQ-022 measurement M6b took, without
a code change. Keep the first pass at `activity.vad.pad_ms: 30`, compare transcript ownership
grace at 0, 20 and 100 ms against the logged hard-onset phrases, exercise the exact-short and
320/350 ms pause controls, and score granular records and public turns separately. The M8/M9
studies prove that neither dropped-word count nor rendered-line count is a loss function alone.

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

**Real DJI metadata has been validated — for two receivers, one take, no speech.** This
sentence used to read "has still not been validated", which was written at M6a's close and
was already false the next day, when the 2026-08-03 jam capture landed four real Mic 3
files in `samples/` and M8 was built on them. It survived two milestones because nothing
re-reads the bottom of this file. **Not existence any more; breadth.**

What real files settled: what the Mic 3 embeds (**OQ-001**), `TX01`/`TX02` uniqueness
(**OQ-002**), `time_reference` present, midnight-relative and at the file rate
(**OQ-004**), no DJI-private or iXML chunk carrying timing (**OQ-005**), the displayed
timecode reaching `bext.time_reference` (**OQ-023**), the receiver's frame-rate setting
**not** reaching the transmitter's file (**OQ-024**), and identical post-jam timecode
across **two** receivers (**OQ-012**).

What no file on this disk can settle, and H1 still owes: the sequence counter across a
power cycle (**OQ-003**), `orig`/`edit` pairing (**OQ-007**), an exact PCM sample count
from `ffprobe` (**OQ-011**), timecode zero against wall clock (**OQ-015**), the third
receiver, six transmitters, and real speech at a real table.
