# ADR-0013 — Silero through ONNX Runtime, pinned by content hash, with no Torch

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M3

**Amends the spec** (Milestone 3, the VAD paragraph). This is the second amendment, after
ADR-0003.

## Context

The spec asked for the Silero *package* and model artifact/revision to be pinned, loaded
locally rather than through an unpinned runtime `torch.hub` fetch, with the identity in
cache keys and the report, and CPU or ONNX inference as the baseline. OQ-010 asked how that
is actually done offline.

Three facts, measured before planning rather than assumed:

1. `silero-vad` 6.2.1 declares `torch>=1.12.0` and `torchaudio>=0.12.0` as **hard**
   dependencies, not extras. Installing it puts Torch in the environment the default test
   suite runs in, which INV-05 exists to keep free of model machinery, and puts a
   PyPI-sourced Torch in the lock file that M6a intends to fill from AMD's `gfx1151` index
   with per-package sourcing.
2. The wheel ships the model files as package data. `silero_vad/data/silero_vad.onnx` is
   2 327 524 bytes, sha256
   `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` — **byte-identical** to
   the same path in the repository at tag `v6.2.1`, commit
   `7e30209a3e901f9842f81b225f3e93d8199902b1`.
3. Driving that file needs nothing from the package. Its ONNX signature is
   `input` (batch, 64 context samples + 512 chunk), `state` (2, batch, 128), and `sr`
   (int64 scalar), returning a probability and the next state. Verified by running it under
   ONNX Runtime 1.28.0 on CPU from a plain NumPy loop.

So the spec's instruction and the spec's mechanism point in opposite directions: pinning the
package is what makes the model *harder* to load reproducibly here.

## Decision

### The artifact is pinned; the package is not installed

`dnd_audio.models` holds one immutable descriptor: upstream repository, release tag,
**commit**, path within the repository, expected size, and expected sha256. `models fetch`
downloads that URL, verifies the hash before the file is moved into place, and records what
it resolved in a local lock. A hash mismatch is fatal and the file is discarded.

The model lives outside any session — `$XDG_CACHE_HOME/dnd-audio/models`, overridable — 
because it is shared across sessions and is not session data. Nothing under a session's
`raw/` is involved (INV-01), and nothing about the model is ever committed.

### The runtime and the interface are pinned too

`onnxruntime` becomes a runtime dependency, locked like every other. "Reproducible
inference" is the artifact **and** the thing executing it **and** the way it is called, so
`DetectorIdentity` records all three: the model's sha256, release, and commit; the ONNX
Runtime version and the execution provider; and the interface identity — frame size, context
size, state shape, input names, and sample rate. A future model with the same name and a
different frame protocol therefore produces a different cache key rather than different
answers under the same one (INV-08).

`onnxruntime` ships no `py.typed`, so it is declared `ignore_missing_imports` in mypy and
wrapped behind a typed adapter; nothing outside `activity/silero.py` sees an untyped object.

### CPU, and the spec's reason for it was wrong

CPU execution provider by default. The spec justified this as avoiding contention "for
unified GPU memory", which does not hold on the target host: it is a unified-memory machine,
so a CPU tensor and a GPU tensor come out of the same 128 GB pool and choosing CPU does not
avoid the contention. The preference is still right, for reasons the spec now states — it
keeps Torch and ROCm out of the default environment entirely, so M3 does not wait on M6a and
the offline suite stays model-free; the GPU's scarce resource during a session is *compute*
for ASR; and a 2.3 MB model over 16 kHz audio is fast enough on CPU that the question does
not arise.

### One detector instance per track, contiguous windows, and it says so

The model is recurrent: it carries a 128-wide state and 64 samples of context between
frames. `ActivityDetector.detect(window)` looks stateless, which is how independent review
found this: reusing one instance across tracks leaks one speaker's state into another, and
rebuilding it per window makes the answer depend on the window partitioning.

The protocol M0 froze is **not** changed. Instead the contract is made explicit and
enforced: a `SileroActivityDetector` instance belongs to exactly one track, windows must
arrive in order and contiguously, and a violation of either **raises** rather than silently
resetting. The runner builds one detector per track through a `DetectorFactory` callable.
A loud failure beats an inferred convention, and an interface two later milestones already
type against does not move for an implementation detail.

Frames are 512 samples at 16 kHz — 32 ms — and a track whose derivative is not a whole
number of frames has its last frame zero-padded, the same rule and the same direction as the
resampler's `ceil` length.

## Alternatives considered

- **Install `silero-vad` and use its `get_speech_timestamps`.** The spec's literal reading.
  Rejected: Torch and torchaudio in the default environment (INV-05), a PyPI Torch in the
  lock M6a must control, and roughly 800 MB of wheels to call a function whose thresholds
  this milestone is required to own anyway.
- **Install `silero-vad` without using it**, purely to satisfy "pin the package". Rejected
  as pinning that improves nothing: the bytes that determine the answer are the ONNX file's,
  and they are already pinned by hash.
- **Vendor the ONNX file into the repository.** It is 2.3 MB and would make the model always
  present. Forbidden outright: never commit model weights.
- **The JIT model through Torch**, or the `silero_vad_16k_op15.onnx` variant. The first needs
  Torch; the second is a different interface with no matching upstream documentation of its
  frame protocol. Neither buys anything here.
- **Fetch at first use, cached.** Rejected: it makes an ordinary `activity` run capable of
  touching the network, which INV-06 permits only `models fetch` to do.
- **Extend the `ActivityDetector` protocol with an explicit stream lifecycle**, as the
  reviewer proposed. A cleaner shape in isolation, and it changes an M0 interface that M4
  and M6b both type against, to express a constraint an assertion expresses just as
  precisely.

## Consequences

- **OQ-010 is answered.** Silero is pinned by commit and content hash, loaded from a local
  file, executed by a pinned runtime through a recorded interface, with no `torch.hub` path
  in the process at all.
- The default test suite never loads the model. Real inference is `host_smoke`; the offline
  tests prove the adapter *refuses* an absent or wrong-hashed artifact, and the production
  code path's statefulness is proved against a deterministic fake ONNX session, so partition
  invariance and cross-track isolation are covered without weights (INV-05, INV-10).
- Three packages enter the default environment: `onnxruntime`, `flatbuffers`, `protobuf`.
- `models fetch` is implemented here, four milestones early, because INV-06 makes it the only
  command allowed to reach the network. It fetches only the VAD model and its lock format is
  **provisional until M6b closes**, which is the treatment M0 gave the transcript schema.
- If a future Silero release changes the frame protocol, the interface identity in the cache
  key changes with it, so no cached detection survives the change silently.
