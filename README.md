# dnd-audio

A fully local audio-ingestion and transcription pipeline for long tabletop-RPG
sessions. Six DJI Mic 3 transmitter recordings in; a diarized transcript, an automix
MP3, and an ingest report out. No audio ever leaves the machine.

- [`dnd-audio-ingestion-agent-spec.md`](dnd-audio-ingestion-agent-spec.md) — the
  product spec.
- [`docs/plan/STATE.md`](docs/plan/STATE.md) — where the project actually is.
- [`AGENTS.md`](AGENTS.md) — how work on this repository is organized.

## Before running a session

```bash
dnd-audio models fetch      # the VAD model, ~2 MB
./scripts/fetch-models.sh   # the ASR model and aligner, ~6 GB, one time
dnd-audio doctor            # tools, disk, models, GPU, and your device/dtype
```

`models fetch` is the only command that reaches the network. Everything else reads
models from a local directory or refuses to run.

**Do not run a transcription alongside a heavy ComfyUI or large-LLM workload.** The
target host has unified memory, so a GPU allocation and the system's RAM come out of
the same pool, and `systemd-oomd` will kill a user process under sustained pressure —
including, in the worst case, an hours-long session four hours in. The pipeline
processes audio in bounded windows and holds one ASR request at a time, which bounds
*its* footprint; it cannot bound anything else's. `dnd-audio doctor` reports free disk
but says nothing about what else is using the GPU.

Transcription needs neither if you only want the audio: `dnd-audio mix` runs the whole
automix branch with no ASR model, no GPU, and no adapter.

Licensed under Apache-2.0.
