# D&D Audio Pipeline — MVP Implementation Brief

Use this document as the implementation prompt for a coding agent. It is intended
to be sufficiently specific for the agent to begin work without revisiting the
capture-system design.

---

## Prompt for the implementation agent

You are implementing the first production version of a fully local audio-ingestion
and transcription pipeline for long tabletop-RPG sessions. Begin by inspecting the
existing repository and following its conventions. If the repository is empty,
scaffold the Python project described below.

Do not merely produce another architecture document: implement the smallest
end-to-end vertical slice, add tests, run them, and continue through the milestones
in order. Ask questions only when genuinely blocked by missing information that
would change the implementation materially. Record ordinary assumptions in the
README instead of stopping for approval.

### Product goal

Turn six internally recorded DJI Mic 3 transmitter tracks into:

1. `transcript.md` — a readable, timestamped, speaker-attributed transcript.
2. `transcript.json` — the same transcript in a stable machine-readable schema.
3. `session.mp3` — a listenable mono automix of the six synchronized lav tracks.
4. `ingest-report.json` — source-file inventory, validation results, alignment
   decisions, warnings, model/configuration versions, and output provenance.

The first two are collectively the “diarized transcript.” Speaker attribution does
not need to be inferred blindly: every person wears a known, physically labeled
transmitter, and the session configuration maps that transmitter to a person.

### Firm scope and decisions

- There are three DJI Mic 3 kits, each containing two transmitters and one receiver.
  Label the receivers `rx-a` through `rx-c` and the six physical transmitters
  `tx-a` through `tx-f`.
- Six transmitter recordings are the complete normal capture system. The kits remain
  three independent two-transmitter groups; DJI Mic 3 supports at most four
  transmitters in one linked group, so do not assume all six can share automatic
  group synchronization.
- Before each session, physically connect receiver A's LTC output to receiver B's LTC
  input, perform Sync, disconnect it, and then repeat from receiver A to receiver C.
  Verify that all three receivers show the same absolute timecode and frame rate after
  each sync. Keep all receivers powered throughout the session; normally keep the
  transmitters powered too. Embedded transmitter-file timecode is the primary
  synchronization source.
- The Zoom H5 is not a normal pipeline input. Do not design around it.
- Processing is local. Audio must not be sent to a cloud API. **One narrow exception,
  added in M7a:** an explicit `archive` command may send byte-exact compressed copies of
  a session's immutable source files to an owner-controlled private cold-storage bucket,
  as off-site backup against disk loss. It is opt-in, never on a processing path, never
  a publication, and never a transfer of audio to anything that *processes* it. Every
  other command remains network-denied, and no ASR, alignment, or detection ever leaves
  this machine. See "Archival extension" below.
- Do not read from or write to the campaign wiki.
- An optional local `glossary.txt` may bias ASR spelling, but absence of a glossary
  must not block a run.
- Input audio is the transmitter's original internal recording, normally 32-bit
  float. When DJI dual-file recording produces both `orig` and `edit` variants,
  consume `orig` and ignore `edit`. A transmitter set to a narrower width is a
  capture mistake rather than an unusable file: accept any sample format that
  converts to float32 with no loss, and refuse the rest with the reason that is true
  of it (ADR-0030). Two of four transmitters in the 2026-08-02 probe recorded 24-bit
  from a per-transmitter setting the operator had not matched (OQ-007).
- Preserve all raw input files byte-for-byte. Never rewrite, rename, delete, or
  normalize files under `raw/`.
- Accuracy and recoverability matter more than runtime. Overnight processing is
  acceptable.
- Character/NPC identification is out of scope. The speaker is the human wearing
  the transmitter, including when the GM performs character voices.
- Generic speaker diarization, summaries, LLM cleanup, wiki integration, neural
  source separation, and multichannel crosstalk cancellation are not MVP features.
- Do not naively sum all six lav tracks to make the MP3. That multiplies bleed and
  room noise.

### Default technology choices

Use these defaults unless the repository already has an equally suitable choice:

- Python 3.12.
- `uv` for environments and dependency locking.
- A typed `src/` package with `pytest`, `ruff`, and a strict type checker.
- `ffmpeg` and `ffprobe` as explicit system dependencies.
- `Typer` (or an existing repository CLI framework) for the command-line interface.
- Pydantic models for external configuration, manifests, and output schemas.
- NumPy/SciPy/SoundFile for CPU signal processing where FFmpeg is not sufficient.
- Silero VAD for speech candidates.
- `Qwen/Qwen3-ASR-1.7B` through the official `qwen-asr` package for local ASR.
- `Qwen/Qwen3-ForcedAligner-0.6B` for word timestamps.

Keep ASR behind a small interface so tests can use a deterministic fake and another
local model can be substituted later. Do not let GPU/ROCm setup block implementation
of ingestion, synchronization, VAD, mixing, schemas, or tests. Support a configurable
device (`auto`, `cpu`, or `cuda`; PyTorch uses the `cuda` API on ROCm too), and do not
pin NVIDIA-only wheels in the base project.

Pin the project interpreter to Python 3.12 with both `.python-version` and a
`requires-python` upper bound that excludes 3.13. Keep Qwen and its heavyweight
runtime dependencies isolated behind an `asr-qwen` dependency group or equivalent,
with lazy imports, so the fake-ASR test suite and all pre-ASR stages run without a GPU
environment.

### Target-host runtime

The production host is an AMD Ryzen AI MAX+ 395 / Radeon 8060S system with 128 GB of
unified memory. Its GPU target is `gfx1151`. Do not assume a normal PyPI or nixpkgs
PyTorch build will support it.

- Add a repository-local Nix development environment as a flake — `flake.nix` plus a
  committed `flake.lock`, with `nixpkgs` pinned to the same channel the host's NixOS
  configuration tracks — exposing two shells:
  - `devShells.default`: an ordinary `mkShell` providing Python 3.12, `uv`, FFmpeg,
    SoX, the native libraries that CPU wheels link against, and a build toolchain.
    This is the everyday shell, and everything through Milestone 5 must work inside
    it without the FHS shell.
  - `devShells.fhs`: a `buildFHSEnv` `.env` for the Milestone 6a ROCm work, where
    AMD's Torch wheels and the `rocm[libraries]` sdist expect a `/usr/lib` layout.
    Entered explicitly with `nix develop .#fhs`. Model it on the same approach the
    host's ComfyUI service uses — an FHS env supplying a compiler toolchain and the
    native libraries wheels link against, wrapping a pip-managed venv rather than
    building Torch through nixpkgs.

  Do not reuse or modify ComfyUI's venv; the project's environment is uv's `.venv`
  inside the repository, and it must be usable from both shells.
- Commit a `.envrc` containing `use flake` so the default shell activates on `cd`
  (the host already runs `direnv` with `nix-direnv`). Keep `flake.lock` committed and
  `.direnv/` ignored. The FHS shell is deliberately not direnv-activated — see
  ADR-0002.
- Provision PyTorch from AMD's stable `gfx1151` wheel index, or another explicitly
  tested `gfx1151` build. Lock the exact wheel/runtime versions. Make PyTorch an
  explicit project/runtime dependency and constrain the `torch` package itself to
  AMD's `gfx1151` index with uv's per-package source/index configuration. Do not let
  `accelerate`'s transitive `torch>=2.0.0` requirement resolve a CUDA build from
  ordinary PyPI.
  Keep the AMD index explicit rather than making it the default source for unrelated
  packages, and verify the lock contains the intended AMD Torch and ROCm artifacts.
- The AMD Torch wheel pulls a `rocm[libraries]` source distribution that builds at
  install time. Keep the FHS compiler/build tools available and do not constrain
  setuptools below 70.2; the proven AMD index currently supplies newer setuptools.
  The official `qwen-asr` package also pins Transformers and directly brings in
  Accelerate, Gradio, Flask, and Python SoX, which is why the entire Qwen stack belongs
  in the isolated `asr-qwen` group rather than the base/test environment.

The uv configuration must be equivalent to this per-package routing (with versions
pinned by the lock file), rather than a command-line index choice that a later sync
can forget:

```toml
[[tool.uv.index]]
name = "amd-gfx1151"
url = "https://repo.amd.com/rocm/whl/gfx1151/"
explicit = true

[tool.uv.sources]
torch = { index = "amd-gfx1151" }
```

- Use the Transformers backend, `torch.bfloat16`, and `cuda:0` for the initial
  production adapter. Do not make vLLM or FlashAttention a prerequisite. Use PyTorch
  SDPA as the baseline. On this host,
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is the knob that permits SDPA to use
  AOTriton instead of the slow math backend, while `HSA_ENABLE_SDMA=0` is primarily a
  gfx1151 stability measure for transfer-related ring timeouts/GPU resets. Promote
  both to host defaults after the short Qwen transcription/alignment smoke test
  passes. Treat `HSA_USE_SVM=0`, `MIOPEN_FIND_MODE`, and other ComfyUI settings as
  separate performance tuning rather than assuming they help ASR.
- Before enabling GPU inference, verify `torch.cuda.is_available()`,
  `torch.version.hip`, the detected device name, and a small BF16 GPU operation.
  Fail with an actionable diagnostic if `device: cuda` was requested but this test
  fails; `auto` may fall back to CPU with a prominent warning.
- Resolve `dtype: auto` together with the final device: use BF16 on validated ROCm
  and normally float32 after CPU fallback. Use BF16 on CPU only when a separate CPU
  BF16 smoke test succeeds. Reject an explicitly requested device/dtype combination
  that fails its smoke test rather than silently changing precision.
- The invoking user or service account must be able to open `/dev/kfd` and the ROCm
  render node (currently `/dev/dri/renderD128`); ROCm compute does not require access
  to the display card node. Do not infer access solely from Unix group membership.
  On this host those compute nodes are currently mode `0666`, so the invoking user
  can use them despite not belonging to `render`/`video`. `doctor` must test actual
  openability and then run the BF16 GPU operation. Adding that user to `render` and
  `video`, or pinning equivalent udev permissions declaratively, remains optional
  hardening against a
  future distro-default permission change rather than a current prerequisite.
- This is a UMA machine. Process long audio in bounded windows and do not hold six
  complete session waveforms in RAM. Document that Qwen processing should not run
  concurrently with a heavy ComfyUI or large-LLM workload. The host's systemd-oomd
  policy may kill a user process under sustained memory pressure.
- Record Python, `qwen-asr`, Transformers, Torch, HIP runtime, device, dtype,
  attention implementation, and resolved model revisions in the report.

Model downloads are allowed during an explicit setup/fetch step, but session audio
must always be passed as local paths or arrays and must never be sent to a URL or API
**by any processing or model path**. The `archive` commands added in M7a are the sole
exception and are not a processing path: they upload byte-exact compressed copies of
immutable source files to a private bucket the owner controls, and pass audio to no model
or service that reads, decodes, or derives anything from it. The storage provider is a
third party operating that bucket — saying otherwise would be comfortable and untrue — and
what the exception turns on is that it stores opaque compressed bytes and processes
none of them. After models are installed, production processing must support
Hugging Face offline mode. Keep model caches outside session directories and out of
version control.

### Repository and command shape

Prefer a single user-facing command:

```bash
uv run dnd-audio process /path/to/session
```

Also expose independently resumable stages for development and recovery:

```bash
uv run dnd-audio inspect /path/to/session
uv run dnd-audio ingest /path/to/session
uv run dnd-audio transcribe /path/to/session
uv run dnd-audio mix /path/to/session
uv run dnd-audio render /path/to/session
```

Command/stage boundaries are:

- `inspect`: discover and validate sources and write the Milestone 1 manifest.
- `ingest`: run `inspect` as needed, then construct/cache the Milestone 2 timeline
  maps, lossless working path, and 16 kHz derivatives.
- `transcribe`: run/cache Milestone 3 activity attribution and Milestone 4 ASR,
  alignment, duplicate collapse, and normalized transcript records.
- `mix`: run/cache Milestone 3 as needed, then perform the Milestone 5 automix and
  MP3 encoding. It must never require ASR or `transcribe` outputs.
- `render`: regenerate `transcript.json` and `transcript.md` from existing normalized
  transcript records; it must not invoke ASR or audio mixing. Fail clearly if the
  required transcript records do not exist.
- `process`: dependency-aware orchestration of all applicable stages. Run activity
  once, attempt both downstream branches independently, render the transcript branch
  when transcription succeeds, and always finalize the structured report. A failed
  transcription branch must not cancel or skip the mix branch.

The executable stage DAG is:

```text
inspect -> reconstruct -> activity
                            |-> transcribe -> transcript render
                            `-> mix --------> MP3

all attempted stages ----------------------> report finalization (always)
```

`reconstruct` is the internal Milestone 2 operation exposed by `ingest`; `activity`
is the shared cached Milestone 3 operation invoked by `transcribe`, `mix`, or
`process`. Report finalization is an always-run internal sink, not a separate command.

Provide non-processing host/setup checks as well:

```bash
uv run dnd-audio doctor
uv run dnd-audio models fetch
```

`doctor` performs system-dependency, writable-path, disk-space, model-availability,
and requested-device checks without processing session audio. `models fetch` is the
only command that requires network access for model installation; it resolves
and records immutable Hugging Face snapshot revisions for later offline use.

Provide off-site backup of the raw sources as a separate command group (M7a):

```bash
uv run dnd-audio archive upload  /path/to/session
uv run dnd-audio archive status  /path/to/session
uv run dnd-audio archive list
uv run dnd-audio archive verify  --session-id SESSION_ID [--track tx-a]
uv run dnd-audio archive restore --session-id SESSION_ID [--track tx-a] --to EMPTY_DIR
```

`archive` is the second and last command group permitted network access, and the only
one permitted to send session audio anywhere. See "Archival extension" below. No
processing command gains network authority from it, and `process` never invokes it.

Use content-addressed or content-hash-aware caching. A failed run should resume
without retranscoding or retranscribing unchanged inputs. Cache keys must include
the relevant source hash, configuration, implementation/schema version, and model
identifier.

Never commit session audio, model weights, secrets, Hugging Face tokens, generated
working audio, or output artifacts. Add appropriate ignore rules.

### Session input contract

A canonical session directory should look like this:

```text
session-2026-08-15/
  session.yaml
  glossary.txt                  # optional; local, no wiki dependency
  raw/
    tx-a/
      ...DJI WAV chunks...
    tx-b/
      ...DJI WAV chunks...
    tx-c/
      ...DJI WAV chunks...
    tx-d/
      ...DJI WAV chunks...
    tx-e/
      ...DJI WAV chunks...
    tx-f/
      ...DJI WAV chunks...
  work/                         # generated and disposable
  output/                       # generated deliverables
```

The physical transmitters will be labeled `tx-a` through `tx-f`. Directory identity
is authoritative. Do not treat DJI's `TX01`/`TX02` filename component as globally
unique. DJI documents it as a receiver-assigned pairing-order identifier that changes
after re-pairing, so independent kits can reuse the same value; confirm the observed
behavior in the real hardware fixture. Even if a future firmware changes the naming,
filename identity remains only a secondary validation hint.
`receiver_id` and `receiver_channel` document and validate the stable physical setup,
but they are not permitted to override the authoritative `track_id` directory.

Example `session.yaml`:

```yaml
schema_version: 1
session_id: "2026-08-15"
title: "Session 01"
language: "English"
active_tracks: "auto"
timecode:
  frame_rate: "30F"
  origin_date: "2026-08-15"
  origin_timecode: null
  rollover_policy: "infer_forward"
tracks:
  - track_id: "tx-a"
    receiver_id: "rx-a"
    receiver_channel: 1
    speaker_id: "alice"
    speaker_name: "Alice"
    input: "raw/tx-a"
  - track_id: "tx-b"
    receiver_id: "rx-a"
    receiver_channel: 2
    speaker_id: "bob"
    speaker_name: "Bob"
    input: "raw/tx-b"
  - track_id: "tx-c"
    receiver_id: "rx-b"
    receiver_channel: 1
    speaker_id: "carol"
    speaker_name: "Carol"
    input: "raw/tx-c"
  - track_id: "tx-d"
    receiver_id: "rx-b"
    receiver_channel: 2
    speaker_id: "dan"
    speaker_name: "Dan"
    input: "raw/tx-d"
  - track_id: "tx-e"
    receiver_id: "rx-c"
    receiver_channel: 1
    speaker_id: "erin"
    speaker_name: "Erin"
    input: "raw/tx-e"
  - track_id: "tx-f"
    receiver_id: "rx-c"
    receiver_channel: 2
    speaker_id: "frank"
    speaker_name: "Frank"
    input: "raw/tx-f"
asr:
  model: "Qwen/Qwen3-ASR-1.7B"
  aligner: "Qwen/Qwen3-ForcedAligner-0.6B"
  context_file: "glossary.txt"
  device: "auto"
  dtype: "auto"
  max_segment_s: 120
  max_new_tokens: 1024
activity:
  correlation_max_lag_ms: 30
mix:
  integrated_lufs: -16.0
  true_peak_dbtp: -1.5
  mp3_bitrate_kbps: 128
```

The configured `tracks` list is the authoritative known roster and permanent
track/receiver-to-person mapping. With the default `active_tracks: "auto"`, file
discovery derives the active participants from configured track directories that
contain at least one usable original recording. This supports absent players and test
recordings without editing the durable roster. The report must show known-roster,
observed-active, and per-track file counts, and list missing, empty, and extra track
directories.

File presence cannot distinguish an intentional absence from a capture failure. For
a session where attendance is known in advance, allow `active_tracks` to be an
explicit list of track IDs; then every listed track is required and a missing usable
original is fatal. Under `auto`, a configured roster track with no usable original is
reported as inactive with a warning, not silently omitted from diagnostics. An
unconfigured directory must never be assigned to a speaker merely from its filename
or presence.

The stable receiver/transmitter-to-person mapping may be copied from a durable local
template, but `session.yaml` must contain the resolved mapping so a session remains
self-describing after that template changes.

File presence cannot recover missing timing metadata. Support exceptional recovery
overrides keyed by source-relative path (and verify the configured SHA-256 when one is
provided), for example:

```yaml
recovery:
  allow_processed_audio: false
  source_time_overrides:
    "raw/tx-a/TX01_MIC002_20260815_190000_orig.wav":
      sha256: "<optional expected source hash>"
      recording_date: "2026-08-15"
      start_timecode: "19:00:00:00"
      reason: "BWF time reference was damaged; value copied from the field log"
```

Permit an optional ISO `recording_date` and either a replacement `start_timecode` or
a signed integer `start_offset_samples` at the canonical 48 kHz rate, measured
relative to session time zero, but not both timing values. An
override applies only to the named source file; every affected chunk needs its own
timing evidence. Record overrides prominently in
the manifest and report. Never infer a recovery time from filesystem modification
time. A track-level correction may exist only as a separately named post-metadata
calibration offset and must not substitute for missing per-chunk gap information.

### Milestone 1: inspection and immutable ingest manifest

Implement file discovery and an `inspect` command first.

For every candidate audio file, run `ffprobe` and retain:

- Relative path and SHA-256 hash.
- File size, displayed duration, `duration_ts`, time base, and an exact decoded or
  container-derived PCM sample count where available.
- Codec, sample format/bit depth, sample rate, and channel count.
- The complete raw JSON from `ffprobe -show_format -show_streams`, including every
  format and stream metadata tag FFprobe exposes, before project-specific parsing.
- A generic RIFF/RF64 chunk inventory containing chunk ID, offset, and size. Do not
  assume FFprobe exposes unknown DJI-private or iXML chunks as tags; retain bounded
  textual/custom metadata payloads when safe and record hashes for larger chunks.
- Detected `orig`/`edit` variant.
- DJI filename sequence information when available, but only as a secondary hint.
- Parsed timecode or Broadcast-WAV sample reference when available.
- All assumptions, fallbacks, and warnings used by the parser.

Do not invent a DJI metadata layout. Implement timecode extraction as a small,
testable strategy chain based on what `ffprobe` actually exposes. Prefer standard
BWF time-reference/sample metadata when present; otherwise parse a standard timecode
tag plus configured frame rate. Correctly support fractional and drop-frame rates
listed in the configuration model, even if the initial fixture uses 30 fps.

The configuration must accept DJI's exact rate labels and map them to rational frame
rates: `23.98F` = 24000/1001 non-drop, `24F` = 24/1, `25F` = 25/1,
`29.97F` = 30000/1001 non-drop, `29.97DF` = 30000/1001 drop-frame,
`30F` = 30/1, `50F` = 50/1, and `60F` = 60/1. Reject incompatible timecode
syntax, such as a drop-frame separator at a non-drop rate. Do not represent
fractional rates as binary floating-point during timestamp arithmetic.

A BWF `time_reference` is a sample count at the file's sample rate, counted from the
**recorder's own timecode origin**. EBU Tech 3285 defines it as samples since
midnight and this hardware does not honour that: the origin is device-local, is
shared between receivers only by a jam, and is frame-quantized rather than
sample-exact (OQ-004, OQ-023, ADR-0031). Placement is a subtraction, so a shared
origin is sufficient and its position in the day is not needed. Keep the value as an
integer and do not round it through a frame count; size the chunk-overlap tolerance
by the recorder's quantum rather than by one sample.

A date or time read *from a file* is descriptive only. It must never anchor a
cross-receiver offset or assign a 24-hour cycle: two receivers' real-time clocks were
measured 48.7 s apart while their timecode agreed to under one frame. Only an
operator assertion — `timecode.origin_date`, or a source-time override's
`recording_date` — may place a session on a calendar.

Define 24-hour wrap handling. `rollover_policy: infer_forward` may infer a single
forward rollover only when chunk sequence and session-span constraints make it
unambiguous, and must record that decision. `timecode.origin_date` is the ISO
calendar date of timecode day zero and must not be inferred from a date-looking
`session_id`.
`timecode.origin_timecode`, when non-null, is the absolute timecode corresponding to
session time zero on `origin_date`; when null, session zero remains the earliest
normalized valid source time. If forward rollover remains ambiguous, require a
non-null dated origin and, where necessary, a source-time override that supplies the
affected source's recording date rather than inventing an ad hoc interpretation.

If no reliable timecode can be extracted and no explicit override exists, fail with
an actionable diagnostic. Modification time is not a trustworthy synchronization
source and must not silently become one.

Selection rules:

- Select only original recordings by default.
- If both original and processed recordings exist, associate them in the manifest
  and ignore the processed one.
- If only a processed recording exists, report an error unless an explicit
  `allow_processed_audio` recovery option is enabled.
- Detect duplicate files by content hash.
- Warn about unexpected formats, files belonging to more than one apparent
  transmitter, and sequence discontinuities.

Write a deterministic `work/manifest.json` and store the raw FFprobe JSON beside it
under a content-hash-addressed path. Running inspection twice on unchanged inputs
must produce byte-identical canonical manifest JSON. Sort paths and keys explicitly,
use atomic temporary-file-plus-rename writes, and do not include wall-clock run times
or cache-hit telemetry in the manifest.

Inspection cache identities must include the source hash, exact FFmpeg and FFprobe
versions, FFprobe command/options, and the RIFF parser plus manifest-schema versions.
A tool or parser upgrade must re-run inspection even when the source bytes are
unchanged.

### Milestone 2: reconstruct six synchronized virtual tracks

DJI may split an internal recording into multiple chunks. Reconstruct each person's
timeline according to embedded timecode, not merely filename order.

Required behavior:

- Establish session time zero from the earliest valid source start time unless the
  config supplies an explicit origin.
- Sort chunks by parsed start time.
- Validate each chunk's expected end against the next chunk's start.
- Preserve real gaps as silence. A transmitter being switched off and later back on
  must not cause later audio to slide earlier.
- Detect overlaps. Resolve only tiny overlaps explainable by timestamp/frame
  quantization; otherwise retain a warning and require an explicit policy rather
  than silently discarding audio.
- Preserve a lossless 48 kHz floating-point working path for the mix.
- Derive cached 16 kHz mono working audio for VAD and ASR.
- Record the exact mapping between source samples, working samples, and the session
  timeline.
- The aligned output duration must be determined by the latest track end, not by the
  shortest track.

Use integer sample indices and rational conversions for internal timeline arithmetic.
Do not repeatedly add floating-point durations. Distinguish the exact 48 kHz source
mapping from rounded public millisecond timestamps, and account for resampler delay
and end rounding in the 48 kHz-to-16 kHz mapping.

“Virtual track” does not require loading or materializing a session-length NumPy
array. Prefer a segment map plus streamed/windowed reads and writes. If a contiguous
floating-point WAV intermediate is useful, use RF64 (`-rf64 auto`) or another format
that remains valid beyond RIFF's 4 GiB limit. Preflight estimated work-space usage and
available disk space before expanding long sessions.

Jammed timecode is timeline synchronization, not a shared word clock. Do not promise
phase-coherent multichannel cancellation. Sample-clock drift correction is a future
enhancement. Add a hook/interface for a future affine time warp, but do not make it
an MVP dependency.

If a distinctive start/end acoustic event is present — a clap, or the generated
synchronization marker produced by `dnd-audio marker build` and detected by
`dnd-audio marker analyze` — optional cross-correlation may be used as
synchronization QA. It should report disagreement with timecode, not override valid
timecode automatically. Measure each track's relative lag near both ends and warn
when the lag changes materially.

**A changing lag is not by itself evidence of sample-clock drift** (ADR-0040). What
is measured acoustically is the sum of two independent quantities: where the
recordings sit on the timeline, and how far the sound travelled to each capsule.
Six lavs at a table are 0.5–3 m from any one source, so moving the source **or any
compared lav** between the two measurements changes the acoustic term by
milliseconds — the same order as the drift being looked for, since measured drift is
≈1 ppm bounded at ±3 ppm, or 14–43 ms across four hours. A wearer leaning back is
enough.

So a start-to-end change is always reported as **differential acoustic arrival**. It
may be called recorder-drift evidence only when the source and every compared
transmitter/lav are asserted to have stayed fixed between the two occurrences — which
a fixed-transmitter soak can assert and an ordinary session with people wearing the
microphones cannot. Neither reading ever applies a correction.

The two-minute hardware fixture validates metadata and synchronization plumbing, not
multi-hour clock stability. Before relying on the no-drift-correction MVP assumption,
run a roughly four-hour soak fixture with synchronized transients near the beginning
and end, or use the first full session's start/end clap measurements as that evidence.
Record the measured differential lag and configured warning threshold. Automatic
affine drift correction remains post-MVP.

### Milestone 3: conservative speech activity and bleed rejection

Run VAD separately on each 16 kHz track. Merge nearby speech regions and pad segment
boundaries so words are not clipped. Keep all thresholds configurable and persist
the VAD probabilities/decisions needed to debug a bad result.

Keep VAD behind an `ActivityDetector` interface and provide a deterministic fake or
ground-truth-mask implementation for tests. Synthetic speech-shaped noise must not
be expected to trigger a particular learned Silero release. Pin the Silero model
artifact by upstream release and commit **and by its content hash**, pin the runtime
that executes it and the interface it is invoked through, load it locally rather than
through an unpinned runtime `torch.hub` fetch, and include that identity in cache keys
and the report. CPU or ONNX inference is the preferred baseline for this small 16 kHz
model: it keeps Torch and the ROCm stack out of the environment the default test suite
runs in, and the GPU's scarce resource during a session is compute for ASR.

_Amended twice (ADR-0013)._ _First:_ this paragraph originally said "pin the Silero
package and model artifact/revision". Installing the `silero-vad` distribution would
drag `torch` and `torchaudio` into the environment the default test suite runs in, for
a model this project drives through ONNX Runtime and never through that package's API —
and it would collide with the AMD-sourced Torch M6a installs. What reproducibility
actually requires is the artifact's bytes, the runtime, and the calling interface, which
is what the wording now names. _Second:_ it justified CPU inference as avoiding
contention "for unified GPU memory". The target host is a unified-memory machine, so a
CPU tensor and a GPU tensor draw on the same pool and the contention is not avoided by
choosing CPU. The preference is right for other reasons, which the sentence now gives.

Because each lav also hears the room, VAD alone will create duplicate candidates.
Implement a conservative two-stage attribution strategy:

1. **Pre-ASR bleed gate:** compute calibrated speech-band energy, VAD probability,
   and cross-channel similarity for overlapping candidates. Similarity must use
   normalized speech-band cross-correlation over a configurable bounded lag, default
   ±30 ms, rather than zero-lag correlation. Record both the peak correlation and its
   selected lag. Suppress a candidate as obvious bleed only when another track is
   convincingly stronger and the signals are strongly related. Default to keeping
   ambiguous candidates; losing real overlapped speech is worse than spending more
   ASR compute.
2. **Post-ASR duplicate collapse:** candidates with substantial temporal overlap,
   strongly similar normalized text, and supporting acoustic evidence may be
   duplicate captures of the same utterance. Keep the version with the best source
   score and record the rejected alternatives. Never collapse a short/common
   utterance such as “yes” or “no” from text similarity alone; require configurable
   minimum text length plus cross-channel correlation or compelling source-dominance
   evidence. If overlapping tracks contain materially different text, or the
   duplicate evidence is ambiguous, retain both and mark them as overlapping speech.
   As a separate conservative rule, a weaker segment may be collapsed when the
   acoustically preferred segment properly contains its normalized words as a contiguous
   sequence, the graph contains pairwise evidence, temporal overlap is substantial, and
   source dominance is compelling. Run ordinary whole-text similarity collapse first so the
   extra rule cannot change its decisions. Exact matching short utterances never qualify for
   containment merely because they are equal (ADR-0033).

Do not use a single global loudness comparison that always awards a time interval to
the loudest person; that would erase a quieter speaker during real overlap. Source
scores should combine track-relative speech level, VAD confidence, cross-track
dominance, and correlation evidence. Keep the scoring function isolated and make
its diagnostics visible in `ingest-report.json`.

For the initial baseline, it is acceptable to attribute every retained candidate to
the person mapped to that track. Generic pyannote speaker clustering is not required
for the MVP.

### Milestone 4: local ASR and word alignment

Define a model-neutral interface such as:

```python
class Transcriber(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...
```

Provide:

- A deterministic fake implementation for unit/integration tests.
- A Qwen3-ASR implementation using the official local package.

Transcribe retained VAD segments from their owner's lav rather than transcribing the
six full-length files blindly. Prefer segments around natural utterance boundaries;
merge very short adjacent regions but cap a request at a configurable duration well
below the effective limit of the installed adapter. `max_segment_s` applies to the
entire padded waveform submitted to Qwen, not just the unpadded ownership interval.
Default it to 120 seconds.
Although the aligner model is advertised for inputs up to five minutes, the official
`qwen-asr` 0.0.6 high-level timestamp path currently chunks at 180 seconds; do not
configure the MVP above 120 seconds or assume the advertised model limit is the
package limit.

Configure the Qwen Transformers adapter's generation ceiling explicitly; default
`max_new_tokens` to 1024 rather than inheriting the upstream wrapper's 512-token
default. Include it in the report and cache key. Detect likely truncation when public
backend metadata reports a length stop, or when the returned text retokenizes within
a small configured margin of the ceiling and appears incomplete. On likely
truncation, split the unpadded core at a natural low-energy boundary, retry both
halves with their own padding, and deterministically stitch them. Bound retries and
retain the original response plus a warning if truncation cannot be resolved. Do not
depend on a private Qwen finish-reason API that the installed public wrapper does not
provide.

Include enough leading/trailing padding for word recovery, then translate returned
timestamps back to session time. Keep an unpadded core/ownership interval for every
request. When padded requests touch or overlap, assign words to core intervals and
deterministically stitch boundaries so padding cannot duplicate words or utterances.
After ASR, a separately configured small leading ownership grace may recover an aligned
word placed just before an activity edge. Bound it by audio actually submitted, clip it
against preceding half-open ownership on the same track, and preserve both the activity and
effective per-piece intervals in normalized records. It changes assembly semantics, never
activity, request audio, or ASR cache identity (ADR-0033).
For a resolved truncation retry, "actually submitted" means the particular retained leaf
submission that returned that word: preserve each leaf's sliced ownership and padded bounds,
and never let a word from one retry child be owned through another child's interval.

Force English by default, but keep language configurable. If `glossary.txt` exists,
pass its text through Qwen's context parameter. Save the unmodified public
`ASRTranscription` result returned by the official package before applying pipeline
normalization. The high-level API already parses the model's raw token decode; do not
depend on private Qwen methods merely to capture a lower-level response. If a future
official API exposes raw generation safely, store it as an additional diagnostic.
“Save” means losslessly serialize all public fields, including language, text, and
timestamp items when present, into a versioned JSON artifact. Do not pickle the
Python object.

Use the forced aligner for word-level times. If alignment fails for one segment,
retain the segment-level transcript and emit a warning rather than failing the
entire session.

Cache every ASR result by the exact segment-audio hash, model/aligner identifiers,
context hash, language, and inference parameters. Model versions and package
versions must appear in the report.

Resolve mutable Hugging Face model names to exact snapshot commit revisions before
inference. Include those resolved revisions, backend, dtype, attention implementation,
and all output-affecting generation/alignment parameters in the cache key. Cache
writes must be atomic and incomplete entries must never count as hits.
Allow explicit model and aligner revisions in configuration. If they are omitted,
`models fetch` must create a local model lock containing the resolved commits, and
`process` must use that lock rather than re-resolving a moving branch online.

Do not add an LLM prose-cleanup pass in the MVP. Preserve what ASR produced, with
only deterministic whitespace/punctuation normalization necessary for rendering.

### Milestone 5: merged MP3 automix

Build `session.mp3` from the synchronized 48 kHz originals. Do not concatenate
speakers end-to-end, and do not sum six full-volume channels.

Implement a smooth speech-aware automixer:

- Estimate a conservative per-track voice-level correction from high-confidence
  speech attributed to that track; clamp correction to a safe range.
- Convert VAD/attribution decisions into continuously smoothed gain envelopes.
- During one-person speech, favor that person's lav and strongly attenuate the other
  five.
- During genuine overlap, keep each active person's own lav audible using
  equal-power or otherwise bounded gain sharing.
- During silence or uncertainty, blend low-level room tone without allowing six
  tracks of noise to add coherently. A Dugan-style normalized gain-share is a good
  baseline.
- Use short attack and longer release/crossfade times so words are not clipped and
  channel changes do not click or pump.
- Preserve the mix as mono; spatial reconstruction is out of scope.
- Apply final two-pass loudness normalization toward `-16 LUFS` integrated by
  default. The configured `-1.5 dBTP` true-peak ceiling applies to the decoded final
  MP3 deliverable, not merely the lossless pre-encode intermediate.
- **The ceiling outranks the target.** Loudness and peak can be mutually unreachable —
  high-crest-factor material cannot be lifted to `-16 LUFS` without clipping, and a
  session nobody spoke in should not be amplified toward it at all. Where they
  conflict, honour the ceiling, encode without loudness normalization, and record a
  warning naming which guard fired. A run that deliberately did not aim at the target
  is not then failed for missing it. See ADR-0023 (M5).
- Encode a 128 kbps mono MP3 with metadata containing the session ID/title, decode it,
  and measure integrated loudness and true peak. Because lossy encoding can introduce
  peak overshoot, reduce the pre-encode gain or true-peak target and re-encode from
  the lossless intermediate when necessary. Bound the retry count, retain all
  measurements in the report, and fail the mix stage rather than claim compliance if
  the decoded MP3 remains outside configured tolerances.

Define a canonical, model-independent pre-ASR activity/attribution graph from VAD and
the conservative bleed gate. The automixer consumes that graph. Transcript assembly
starts from the same graph but may add post-ASR duplicate and overlap decisions; those
text-dependent decisions must not change the mix. A transcription-model failure must
still permit an ingest report and merged MP3.

Keep a lossless mix intermediate in `work/` for debugging/cache reuse, not as a
required user-facing deliverable.

### Archival extension

Added in M7a, after the MVP path was complete and before the first irreplaceable
recording. It protects against the one failure later software cannot repair: loss of the
original transmitter files. It is the narrow exception to "processing is local" stated in
the firm-scope section above, and it is not a processing feature.

**What it does.** `archive upload` builds an independent, hardened inventory of every
regular non-symlink file beneath the configured source roots — selected audio, ignored
`edit` variants, duplicates, unassigned audio, unexpected file types and nested notes
alike — compresses each byte-for-byte with a frozen zstd recipe, uploads it as one
immutable object to an owner-controlled **private** cold-storage bucket, reads every
object back in full, and proves it decompresses to the original SHA-256. Only then does it
publish a single small manifest object, which is the commit marker. After loss of the
local session directory an operator can discover, verify, and restore a whole session or a
single track from the session id alone, without object keys and without the old
`session.yaml`.

**What it must not do.** It publishes nothing: no MP3, transcript, report, or wiki
artifact. It deletes nothing, locally or remotely — the application exposes no object
delete operation and calls none, `AbortMultipartUpload` for its own incomplete uploads
excepted. It never modifies, renames, or normalizes anything under a source root, so
INV-01 stands unamended. It is never invoked by `process`, and no processing command gains
network authority from its existence. Archive configuration lives outside `session.yaml`
entirely and must not enter any processing cache identity.

**Requirements on the implementation.**

- Enumeration uses `lstat`, refuses a symlink at every path component, and proves each
  resolved file stays inside a resolved configured source root. Track identity is optional
  and assigned only where a path belongs unambiguously to one configured track input;
  unassigned files stay unassigned rather than being attributed (INV-11).
- The source inventory is hashed before work and re-verified on every exit path, including
  every failure path.
- Compression, upload, verification and restore are bounded streams. Decompression carries
  an output-size ceiling and aborts the moment it would exceed the declared original size,
  rather than discovering it at the final hash. Worst-case disk is preflighted from the
  compression bound, never from an observed compression ratio.
- Object keys are content-addressed, canonical, and reversibly encoded over filesystem
  bytes rather than decoded text, because a filename need not be valid UTF-8.
- The archive format is versioned in the key prefix. A changed encoding recipe requires a
  new version, never different bytes at an existing key.
- Multipart upload is mandatory above the provider's single-PUT limit and respects its
  minimum part size and maximum part count. An S3 ETag is never treated as a content
  checksum. Retries are bounded and the client's own retry machinery is disabled so that
  one bound governs.
- Every operation writes a local structured report, separate from `ingest-report.json`,
  distinguishing an upload that committed, a previous verification recorded at commit
  time, and a verification performed *now*. Only a current full download and decompression
  may be called verified. Partial failure never exits zero.
- Reports, logs, and exceptions carry no endpoint credential, signed URL, or secret.

### Output schemas

Use versioned schemas. A transcript JSON baseline:

```json
{
  "schema_version": 1,
  "session_id": "2026-08-15",
  "title": "Session 01",
  "duration_s": 14432.417,
  "speakers": [
    {
      "speaker_id": "alice",
      "speaker_name": "Alice",
      "track_id": "tx-a"
    }
  ],
  "segments": [
    {
      "segment_id": "seg_000123",
      "start_s": 4821.44,
      "end_s": 4824.91,
      "speaker_id": "alice",
      "speaker_name": "Alice",
      "track_id": "tx-a",
      "text": "We should go back to Zephyrine.",
      "overlap": true,
      "words": [
        {
          "start_s": 4821.44,
          "end_s": 4821.68,
          "text": "We"
        }
      ],
      "provenance": {
        "asr_model": "Qwen/Qwen3-ASR-1.7B",
        "asr_model_revision": "<resolved Hugging Face commit>",
        "alignment_status": "aligned",
        "source_candidate_id": "candidate_000456",
        "source_candidate_ids": ["candidate_000456"],
        "source_segment_ids": ["seg_000123"]
      }
    }
  ]
}
```

Do not manufacture an ASR confidence value if the model does not expose a meaningful
one. Keep signal-quality/source-selection scores separate from model confidence.

Generate and check in JSON Schema artifacts for `session.yaml`, `manifest.json`,
`transcript.json`, and `ingest-report.json` from the authoritative Pydantic models.
Tests must validate real outputs against those artifacts, not merely round-trip them
through the same Pydantic class that created them.

Keep exact internal times as samples/rationals, but serialize public transcript times
deterministically to millisecond precision. Define stable sorting tie-breakers and
derive segment/candidate IDs deterministically from sorted source identity and time,
not from task completion order. `overlap` means that a segment overlaps another
retained, non-duplicate speaker segment by at least the configured overlap threshold.

Normalized records remain candidate-granular and retain their audit trail. Public JSON and
Markdown may coalesce adjacent compatible records into one presentation turn only under a
separate bounded exact-sample gap and shared request lineage; batching alone does not define
a conversational turn. Both public views must use the same grouping. Plural provenance names
every source record and candidate, and `overlap` is recomputed over the resulting public
intervals (ADR-0034).

Render Markdown like:

```markdown
# Session 01

**[01:20:21.440] Alice [overlap]:** We should go back to Zephyrine.

**[01:20:23.120] Bob [overlap]:** Absolutely not.
```

Sort by start time, preserve overlapping turns as separate entries, use millisecond
timestamps, and escape user/model text safely for Markdown.

### Error handling and observability

The pipeline must distinguish fatal errors from recoverable warnings.

Examples of fatal errors:

- A track explicitly listed in `active_tracks` has no usable original recording.
- Timecode cannot be established and no explicit recovery offset exists.
- A source file cannot be decoded.
- A selected source is not 48 kHz, or selected chunks within one track disagree on
  sample rate. Do not silently resample the lossless mix timeline; require an
  explicit future recovery policy for nonconforming hardware files.
- Output paths would overwrite raw inputs.

Examples of warnings:

- A transmitter has a real gap.
- A processed duplicate was ignored.
- One segment failed forced alignment.
- Sync QA disagrees modestly with embedded timecode.
- One participant's track is much noisier/quieter than the others.

Provide human-readable progress and a structured report. Include stage timings, cache
hits/misses, warnings, source hashes, configuration hash, dependency versions, and
the exact commands/parameters used for FFmpeg outputs. Do not include secrets.

The report must include `overall_status`, a status for every stage (`complete`,
`failed`, or `skipped`), structured errors, and the hashes of every deliverable that
was successfully produced, other than the report itself — a file cannot contain the
hash of its own final bytes (ADR-0003). Write/update it atomically even on partial
failure. If ASR
fails but mixing succeeds, retain the MP3 and report, mark the transcript stage
failed, and make the top-level `process` command exit nonzero so automation cannot
mistake partial output for full success.

Separate deterministic provenance from per-run telemetry within the report. Manifest,
transcript JSON, Markdown, schemas, and cached semantic results must be byte-stable on
an unchanged rerun. `ingest-report.json` is not required to be byte-identical because
stage timings and cache-hit information legitimately change; its provenance and
decision subsections must nevertheless be semantically stable.

### Tests and acceptance criteria

Build a small synthetic fixture generator rather than checking audio binaries into
the repository. It should create six virtual mono tracks with:

- Multiple chunks per transmitter.
- Different timecode start offsets.
- A real gap in one transmitter.
- A shared clap/transient.
- Solo speech-shaped/noise-shaped activity that bleeds quietly into other tracks.
- One interval representing two simultaneous speakers.
- Deterministic fake-VAD/ground-truth activity decisions.
- Deterministic fake-ASR results.

At minimum, automate these checks:

1. Original/processed file selection, duplicate detection, `active_tracks: "auto"`,
   and explicit-required-track behavior are correct; an unconfigured directory is
   never attributed to a person.
2. BWF sample references plus non-drop, fractional, drop-frame, midnight-rollover,
   and explicit source-override cases map to the expected integer sample positions.
3. Chunk order, gap preservation, and global alignment match the synthetic truth.
4. The lossless aligned output duration matches the latest source end within one
   48 kHz working sample.
5. Obvious bleed duplicates collapse to the intended track.
6. Distinct overlapping utterances and simultaneous matching short utterances both
   survive and are marked as overlap.
7. Transcript JSON validates against its checked-in schema and Markdown order is
   stable.
8. The MP3 exists, decodes successfully, is mono, is within one MP3 frame (or another
   documented codec-appropriate tolerance) of the expected duration, is within 1 LU
   of the configured integrated loudness target, and does not exceed the true-peak
   target beyond a documented measurement tolerance. The loudness clause applies to a
   run that aimed at the target; where the ceiling, the master-gain clamp, or the
   silence floor forbade aiming at it, the run instead carries the warning naming that
   guard, and the duration and true-peak clauses still apply unchanged (ADR-0023).
9. Re-running unchanged input uses caches and produces byte-stable manifest,
   transcript JSON, and Markdown. Do not require byte-stable per-run telemetry.
10. Raw source hashes are unchanged before and after a complete run.
11. A model failure does not prevent generation of the MP3 and diagnostic report,
    but makes `process` exit nonzero with the transcript stage marked failed.
12. The default test suite passes without Qwen, model weights, network access, or a
    GPU. A separately marked host smoke test verifies actual compute-device-node
    openability, the locked gfx1151 Torch build, a BF16 operation, and a short real
    Qwen transcription/alignment.
13. A selected 44.1 kHz or internally sample-rate-inconsistent track fails before
    timeline construction, and `render` can regenerate outputs from cached transcript
    records without loading Qwen or running the mixer.
14. The submitted padded waveform never exceeds `max_segment_s`; changing
    `max_new_tokens` invalidates the ASR cache; and a fake length-truncated response
    triggers bounded split/retry and deterministic stitching.
15. Correlated bleed delayed within the configured lag window is still detected, the
    peak lag is reported, and a synthetic change in start-versus-end acoustic lag is
    reported as a change in differential acoustic arrival, without applying an
    automatic correction. It emits a **drift** warning only where the source and every
    compared transmitter/lav are asserted to have stayed fixed between the two
    measurements; with geometry unasserted or known to have changed, the same
    measurement is reported and explicitly not attributed to the clocks (ADR-0040).

Test automixer behavior at the gain-envelope level using the deterministic activity
graph; decoded loudness alone is not evidence of correct channel selection. Assert
with explicit configurable tolerances that:

- After the attack interval, a solo speaker's channel gain dominates every inactive
  channel by at least the configured attenuation margin.
- During genuine two-person overlap, both active source channels retain nontrivial
  audible gain.
- The chosen normalized/equal-power gain invariant remains bounded at every sample or
  control frame, including silence and transitions.
- Obvious correlated bleed is not promoted on two channels simultaneously.
- Gain envelopes contain no discontinuities and do not exceed configured attack,
  release, or maximum-slew limits.

Before claiming real DJI support is complete, validate against a short hardware
fixture. If none is available, implement everything possible with synthetic data
and clearly mark this one integration step as pending rather than guessing DJI's
metadata tags.

Recommended real fixture recording:

- All six labeled transmitters and all three synchronized receivers.
- About two minutes total.
- Keep the kits as independent two-transmitter groups. Jam receiver B from receiver
  A's LTC output, disconnect it, then jam receiver C from A and disconnect it. Record
  the displayed timecode/rate on all three receivers after the procedure.
- Start the transmitters a few seconds apart.
- Each wearer states their transmitter label and speaks alone for several seconds.
- Include one two-person overlap.
- Turn one transmitter off, wait several seconds, turn it back on, and record again.
- Make a distinctive three-clap pattern near the start and end. Once the generated
  synchronization marker has passed its phone/hardware bench, it may be played from
  one fixed central position instead — same role, detected automatically rather than
  picked by hand. It supplements the LTC jam and never replaces it: timecode places a
  transmitter that was switched off and back on, and a sound at the top of the session
  cannot.
- Export both `orig` and `edit` files if dual-file mode is enabled.

Document the discovered DJI file naming and metadata in a fixture note, and store
only sanitized `ffprobe` JSON plus the generic RIFF chunk inventory in the repository
unless the user explicitly approves committing a tiny audio sample.

### Implementation order

Work in this order and keep the main branch runnable after each step:

1. Project scaffold, repo-local Nix flake + direnv environment, schemas, CLI,
   configuration loading, and `inspect`.
2. Deterministic manifest and synthetic fixture generation.
3. Timecode/chunk reconstruction and 16 kHz derivatives.
4. VAD, conservative bleed gating, and diagnostic features.
5. Fake ASR end-to-end transcript outputs.
6. Automix and MP3 rendering.
7. Qwen ASR/forced-aligner adapter and model-independent caching.
8. Real DJI fixture validation and threshold tuning.

At the end of each work session, report:

- What now works end to end.
- Tests and commands run, with results.
- Assumptions made.
- Remaining blockers, especially whether real DJI metadata has been validated.
- The next smallest implementation step.

### Explicit non-goals for this version

Do not expand scope into any of the following until the MVP has been measured on a
real session:

- H5 ingestion or fallback recovery.
- Wiki reads/writes or automated summaries.
- Speaker embeddings or voice-character recognition.
- Neural source separation or crosstalk subtraction.
- Phase-coherent multichannel processing.
- Video synchronization or editing.
- A web UI.
- Real-time processing.
- Cloud ASR.

Design extension points where cheap, but do not implement these features now.

### Primary references

- DJI Mic 3 product/recording behavior and integrated timecode:
  <https://www.dji.com/mic-3/faq>
- DJI Mic 3 supported frame rates, LTC connections, and clock accuracy:
  <https://www.dji.com/mic-3/specs>
- DJI Mic 3 user manual, including Master Run, Auto Jam, and group limits:
  <https://dl.djicdn.com/downloads/DJI%20Mic%203/202508282/DJI_Mic_3_User_Manual__EN.pdf>
- Official Qwen3-ASR implementation and usage:
  <https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR/forced-aligner model documentation:
  <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- AMD's stable PyTorch wheel index for this host's `gfx1151` GPU:
  <https://repo.amd.com/rocm/whl/gfx1151/>

---

## Owner notes before handing this to the agent

1. Put durable `A`–`C` labels on the receivers and `A`–`F` labels on the physical
   transmitters. The DJI `TX01` and `TX02` designations are not sufficient across
   three kits. Keep the stable receiver/transmitter/person roster in configuration
   and snapshot its resolved mapping into every `session.yaml`.
2. Keep the receiver frame-rate setting consistent across all three kits. Set
   receiver A to LTC output, physically connect it to receiver B's LTC input, perform
   Sync and disconnect, then repeat from A to C. Confirm matching displayed
   timecodes and rates before recording, and keep all receivers powered. Do not rely
   on one automatic group for all six transmitters because a group supports only
   four transmitters.
3. Before each session, confirm on all six transmitters that 32-bit-float internal
   recording is enabled, sufficient free storage remains for the planned duration,
   batteries/external power are adequate, and every expected transmitter shows an
   active internal-recording indicator after recording starts.
4. Disable transmitter loop recording for normal sessions unless overwriting the
   oldest internal recordings is deliberately desired and separately backed up.
5. Give the implementation agent the short real hardware fixture described above as
   soon as practical. That is the one thing an architecture prompt cannot safely
   substitute for.
6. For the first real session, keep raw files and pipeline outputs even if the
   transcript is imperfect; the diagnostics will be the basis for tuning bleed
   thresholds and the automixer.
