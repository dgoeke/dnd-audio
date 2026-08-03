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
session's `raw/`. Enforced by hashing every file under the sources — not only the
selected ones — before and after a complete run. Output paths that would land inside
`raw/` are a fatal error, compared **after resolving symlinks**: a lexical comparison is
defeated by a single `output -> raw/tx-a` link (M1's verify phase found exactly that).
When the report's own location is the offending one, no report is written; this invariant
outranks INV-13 there, because a report is regenerable and a source directory written into
is not. The session's own generated directories are excluded from the snapshot **at the
session root only**: excluding any path component named `work` or `output` anywhere in the
tree left `raw/tx-a/work/notes.txt` unhashed and therefore freely mutable, which M2's
verify phase found in code inherited from M1.
**Failure cleanup runs after the carve-out, never before it** (ADR-0021). Every composed
runner deletes the artifacts a failed run may have left, so a stale file cannot sit beside a
report calling its stage failed. When an output path resolves inside a source directory, those
unlinks *are* the violation: `work -> raw/tx-a` makes `work/timeline.json` resolve to
`raw/tx-a/timeline.json`. `ingest`, `activity` and `transcribe` all cleaned up first and
checked second, so the run that correctly detected the violation committed it on the way out —
found in all three at once by M4's verify phase, five months of milestones after the check
itself was written and tested.
_Owner: M1, extracted to `raw_guard.py` in M2 so each stage declares its own protected
outputs. Test: full-run hash equality, plus a run that corrupts a source mid-flight to
prove the check can fail, plus a source under a directory named `work`, plus
`TestCleanupNeverWritesIntoRaw` over every composed command._
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
only when serializing public millisecond timestamps. There is exactly one quantizer
(`determinism.to_samples`, half away from zero) and one float-producing conversion
(`public_seconds`); a second of either is how this rule dies. Take a session-relative
difference *before* quantizing, never after — rounding two absolute positions and
subtracting them is a sample short. A 24-hour wrap is unwrapped in the evidence's **own**
units, because a timecode day at 30000/1001 fps is 86 486.4 real seconds (ADR-0008,
ADR-0009).
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
pressure kills processes. **Prove it over the composed path, not one component:** bounding
a reader says nothing about a caller that collects every window. M2's technique is to
instrument reads and writes into one ordered event log and assert that a write happens
before the last read — a property nothing accumulating a session-length array can satisfy
(`tests/test_memory.py`).
_Owner: M2._

**INV-08 — Cache identity is complete.**
Every cache key includes the relevant source hash, the resolved configuration, the
implementation/schema version, external tool versions (FFmpeg/FFprobe for
inspection), and model/aligner identity plus resolved revision and all
output-affecting inference parameters for ASR. A tool or parser upgrade must
re-run the work even when source bytes are unchanged. Cache writes are atomic and
an incomplete entry is never a hit — which requires a *size* check, not just the presence
of the file the entry names. **An entry is committed only after INV-01 has been
re-verified**, never at publish time: a run that correctly fails on a changed source must
not leave behind an entry keyed on the bytes it read, because restoring the original file
makes that key match again forever (M2's verify phase).

**This holds for every cache a run touches, and the test must be scoped that way.** A helper
that commits its own cache defeats the verification its caller performs, and it reads as
correct from the outside when the caller commits again afterwards — `_inspect` carried a
docstring promising it returned the cache uncommitted and published it three lines below,
which survived M2 and was caught only when M3 composed inspection into a longer run. The
regression test could not have seen it: it asserted over the one cache that milestone had
added. **Assert by glob rather than by naming the caches you know about** (M3's verify phase).

**Scoped to a commit point, not to a run** (ADR-0021). A composed run may commit more than
once — M4's `transcribe` commits the activity caches after the first verification and the ASR
cache after the second, so that an ASR failure, which reads no source audio, does not discard
six tracks of verified inference. A failure then leaves no sidecar for any cache **downstream
of the last successful commit point**, which is the region to glob. Caches committed after a
verification that did happen were built from bytes that run confirmed; the hazard this rule
exists to prevent is an entry keyed on bytes nobody checked, and an earlier commit point does
not create one. Say which region a test globs, because a name promising "anywhere" over a body
checking one directory is worse than no test (M4's verify phase).
_Owner: M1 (inspection), M2 (derivatives), M3 (detection and attribution), M6b (ASR)._

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
