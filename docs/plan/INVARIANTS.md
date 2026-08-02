# Invariants

Cross-cutting rules that no milestone may break. These are the properties most
likely to decay silently across context resets, so each one gets an ID and, once
the relevant milestone lands, a test that references that ID in its name or
docstring.

When a milestone adds a mechanism that could violate an invariant, it also adds
the test that proves it does not.

---

**INV-01 — `raw/` is immutable.**
No pipeline stage writes, renames, deletes, or normalizes anything under a
session's `raw/`. Enforced by hashing every source before and after a complete
run. Output paths that would land inside `raw/` are a fatal error.
_Owner: M1. Test: full-run hash equality._
_Future exception (M7): the owner may delete raw files manually after verified
archival. That happens outside a pipeline run, by explicit human action. No
pipeline stage ever deletes from `raw/`. Amend this wording when M7 is planned._

**INV-02 — Deterministic artifacts are byte-stable.**
`manifest.json`, `transcript.json`, `transcript.md`, generated JSON Schema files,
and cached semantic results are byte-identical when rerun on unchanged inputs with
unchanged configuration. Sort keys and paths explicitly; write atomically via
temp-file-plus-rename; derive IDs from sorted source identity and time, never from
completion order. `ingest-report.json` is exempt as a whole, but its provenance and
decision subsections must be semantically stable.
_Owner: M0 helper, enforced per artifact from M1 onward._

**INV-03 — No wall-clock or telemetry in deterministic artifacts.**
Run times, cache hit/miss counts, and hostnames belong in the report's telemetry
section only. They must never reach the manifest or transcript outputs.
_Owner: M1._

**INV-04 — Timeline arithmetic is exact.**
Integer sample indices and rational conversions only. Never accumulate floating
point durations; never represent a fractional frame rate (24000/1001, 30000/1001)
as a binary float during timestamp arithmetic. A BWF `time_reference` stays an
integer sample count and is never rounded through a frame count. Floats appear
only when serializing public millisecond timestamps.
_Owner: M2._

**INV-05 — The default test suite is offline, CPU-only, and model-free.**
Everything except tests marked `host_smoke` must pass with no network, no GPU, and
no model weights. Socket access is blocked by an autouse fixture so a violation
fails loudly instead of quietly depending on the developer's machine.
_Owner: M0._

**INV-06 — Session audio never leaves the machine.**
Audio is passed to models as local paths or in-memory arrays. No cloud ASR, no
URL uploads, no telemetry containing audio. `models fetch` is the only command
permitted to touch the network at all.
_Owner: M0 (policy), M6b (enforcement in the adapter)._
_Future exception (M7): archival deliberately uploads audio to owner-controlled
object storage. That is opt-in, never on a processing path, and not the cloud-ASR
prohibition this invariant exists to prevent. Amend this wording when M7 is
planned rather than working around it._

**INV-07 — Memory stays bounded.**
Never hold six full-session waveforms in RAM. Long audio is processed in bounded
windows over a segment map with streamed reads and writes. Contiguous float
intermediates use RF64 so they stay valid past RIFF's 4 GiB limit. Work-space and
disk are preflighted before expanding a long session. This is a UMA host — memory
pressure kills processes.
_Owner: M2._

**INV-08 — Cache identity is complete.**
Every cache key includes the relevant source hash, the resolved configuration, the
implementation/schema version, external tool versions (FFmpeg/FFprobe for
inspection), and model/aligner identity plus resolved revision and all
output-affecting inference parameters for ASR. A tool or parser upgrade must
re-run the work even when source bytes are unchanged. Cache writes are atomic and
an incomplete entry is never a hit.
_Owner: M1 (inspection), M6b (ASR)._

**INV-09 — The mix never depends on ASR.**
The automixer consumes only the model-independent pre-ASR activity/attribution
graph. Post-ASR duplicate-collapse and overlap decisions are text-dependent and
must not change a single sample of the mix. A transcription failure still yields
`session.mp3` and `ingest-report.json`, with `process` exiting nonzero.
_Owner: M3 (graph freeze), M5 (enforcement)._

**INV-10 — Models and detectors sit behind interfaces with deterministic fakes.**
`Transcriber` and `ActivityDetector` are protocols. Every stage above them is
testable without a real model. Synthetic speech-shaped noise is never expected to
trigger a particular learned Silero release.
_Owner: M0 (protocols), M3, M4._

**INV-11 — Track identity comes from the configured directory.**
`track_id` is authoritative and derives from the configured input directory. DJI's
`TX01`/`TX02` filename component is a secondary validation hint only, never an
identity. `receiver_id`/`receiver_channel` document and validate the physical
setup but may not override the directory. An unconfigured directory is never
attributed to a speaker.
_Owner: M1._

**INV-12 — Timing is never invented.**
If no reliable timecode can be extracted and no explicit override exists, fail
with an actionable diagnostic. Filesystem modification time is never a
synchronization source. A track-level calibration offset may exist only under a
separate name and never substitutes for missing per-chunk gap information.
_Owner: M1, M2._

**INV-13 — Fatal and recoverable are distinguished explicitly.**
Every stage reports `complete`, `failed`, or `skipped`; the report carries
`overall_status`, structured errors, and hashes of every deliverable actually
produced **other than the report itself** — a file cannot contain the hash of its
own final bytes (ADR-0003). It is written atomically even on partial failure.
Partial success never exits zero.
_Owner: M0 (report skeleton), every milestone thereafter._
