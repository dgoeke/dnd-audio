# ADR-0026 — Probing is impure, resolution is pure

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M6a

## Context

The spec states five rules about turning a configured device and dtype into the ones a run
actually uses:

- verify `torch.cuda.is_available()`, `torch.version.hip`, the detected device name, and a
  small BF16 GPU operation before enabling GPU inference;
- fail with an actionable diagnostic if `device: cuda` was requested and that verification
  fails;
- `auto` may fall back to CPU with a prominent warning;
- resolve `dtype: auto` *together with the final device* — BF16 on validated ROCm, normally
  float32 after a CPU fallback, BF16 on CPU only when a separate CPU BF16 smoke test
  succeeds;
- reject an explicitly requested device/dtype combination that fails its smoke test rather
  than silently changing precision.

Every one of those is a rule about a machine. The obvious implementation asks the machine
the question at the moment it needs the answer — and an implementation shaped that way can
only be exercised on a host that has the hardware, along whichever single path that host
happens to take. The interesting cases are the ones no machine can be asked to produce on
demand: a driver that enumerates a device and then computes the wrong answer, a CUDA build
that arrived where a ROCm one was meant, a GPU whose BF16 fails while its float32 works.

This milestone's charter names that risk directly: *"This milestone is where 'it works on my
machine' is most tempting."*

There is a second constraint. INV-05 requires the default suite to pass with no GPU, no
model weights, and no Torch — and after ADR-0025 the Torch stack is in an opt-in group that
the project environment deliberately never installs. So the code holding these rules must
be importable, and testable, in an environment where `import torch` fails.

## Decision

`dnd_audio.runtime` is split along the purity line, and the seam is the data.

**Probing is impure and total.** `probe_torch()`, `probe_cpu_bf16()` and
`open_device_nodes()` import Torch lazily, open character devices, and run arithmetic on a
GPU. They never raise — a machine with no Torch, no GPU, or a broken ROCm stack is a fact
to record, not an exception for every call site to handle — and each returns a frozen
record of what it found.

**Resolution is pure.** `resolve_runtime(device=…, dtype=…, probe=…, cpu_bf16_ok=…)` is a
total function of those records. Every rule above is therefore exercised offline, over the
whole matrix, by constructing the probe rather than by owning the hardware.

**The smoke test is per device *and* per dtype.** The probe records which dtypes produced
exactly the right answer on which device, and resolution requires the pair it is about to
hand back. A single BF16 verdict standing in for every combination gets two cases wrong in
opposite directions: an explicit `cuda` + `float32` request is refused because BF16 failed,
though the requested combination works; and an explicit `float32` request is accepted
having never been smoke-tested at all. The charter says an explicitly requested combination
that fails *its* smoke test is rejected, and only a per-combination result can say that.

**The arithmetic is compared exactly, not within a tolerance.** The operands are chosen so
every input and every product is exact in bfloat16's 8-bit significand. A tolerance is how
a smoke test passes on a device that returned nearly the right numbers, which is the
signature of a miscompiled or wrongly targeted kernel — the exact failure this check exists
to catch.

**Device-node evidence lives in the same record, and does not gate resolution.** The nodes
are opened, recorded, and reported, so M6b inherits one document rather than two to keep in
step. But successful arithmetic *on the device* is strictly stronger evidence than an
`open()` on a character device, so the arithmetic decides. A rule that failed a working GPU
because a node glob came back empty would be inventing a failure.

**Nothing about `gfx1151` is in the resolver.** The gfx target is recorded and reported;
asserting it belongs in the `host_smoke` test and in `doctor`'s output, which are claims
about *this* host. A general resolver that hardcoded one architecture would be wrong on
every other machine.

**`doctor` gains `--device` and `--dtype`.** Without them the explicit-request rules would
have no production caller in M6a at all, and would be demonstrated only by module-level
tests — while the criterion is phrased about what an operator experiences. Unused
production code is how a rule rots between milestones.

## Alternatives considered

- **Resolve against the live machine at the point of use.** Rejected: it makes the rules
  testable on one host along one path, and the failure states that matter most are the ones
  that host is not in. This is the whole decision.
- **Let `probe_torch()` raise on a machine without Torch.** Rejected: the CPU-fallback path
  is a normal outcome, not an error, and making it an exception would put a `try` around
  every call and invite one of them to swallow a real failure.
- **One BF16 verdict for everything.** Rejected on the review finding above; it is wrong in
  both directions and green in both.
- **Require node openability before accepting a GPU.** Rejected as a precondition, accepted
  as evidence. See above.
- **Put the resolution rules in `AsrConfig` validators.** Rejected: a validator runs when
  configuration is loaded, and the answer depends on hardware that a configuration file
  knows nothing about. `session.yaml` would then be valid or invalid depending on which
  machine read it.
- **Have `transcribe`/`process` resolve a runtime before raising `DEFERRED: M6b`.**
  Rejected. It would give the resolver a production caller a milestone earlier, but at the
  cost of replacing ADR-0005's "this pipeline has not built that yet" with a device
  diagnostic on a host that simply has no adapter — which M5 deliberately established and
  is a worse message for the situation an operator is actually in.

## Consequences

- **The rules are provable without hardware**, which is what lets the completion gate be
  demonstrated by the offline suite rather than by a transcript from one machine.
- **A probe can go stale within a run.** It is measured once and reused; a GPU that dies
  mid-session is not detected by this layer. Acceptable — that failure surfaces as a kernel
  error where it happens, and re-probing per request would cost far more than it caught.
- **M6b inherits the seam rather than the rules.** The adapter calls `probe_torch()` once,
  resolves, records `RuntimeProvenance` in the report, and adds those fields to
  `TranscriberIdentity` so they reach the ASR cache key (INV-08). Nothing in the resolution
  logic should need to change for it.
- **Nothing in M6a resolves a runtime during a pipeline run.** `mix` and `activity` load no
  model; `transcribe` and `process` raise `DEFERRED: M6b` first. `doctor` is the only
  production caller, which is why the report's `runtime` subsection exists here and is
  filled in M6b — recorded as an amendment in the M6a charter rather than left as a gap.
- **Which render node backs the compute device is unanswered** on a host with more than
  one — see OQ-021. Until then, at least one openable render node is a pass and the others
  are reported.
