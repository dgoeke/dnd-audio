# M6a — ROCm environment and GPU doctor

**Status:** closed
**Depends on:** M0 (can run in parallel with M1–M5)
**Spec sections:** Target-host runtime

## Goal

A locked, reproducible PyTorch installation for the target host's gfx1151 GPU inside the
repo-local FHS shell, and a `doctor` that proves the device is genuinely usable —
without any Qwen code yet.

## Completion gate

- [x] `[[tool.uv.index]] amd-gfx1151` declared `explicit = true` with
      `[tool.uv.sources] torch = { index = "amd-gfx1151" }`, in configuration rather
      than as a command-line index choice a later sync can forget.
- [x] The AMD index is **not** the default source for unrelated packages.
- [x] `uv.lock` demonstrably contains the intended AMD Torch and ROCm artifacts and
      no CUDA build resolved from ordinary PyPI. Asserted by a test that reads the
      lock, so `accelerate`'s transitive `torch>=2.0.0` cannot quietly win.
      **`transformers` and `accelerate` are therefore in the `asr-qwen` group from
      this milestone**, beyond the original non-goals below: with torch alone in the
      lock nothing competes for it and the assertion passes vacuously. `qwen-asr`
      itself stays out. Amended during the start phase; see ADR-0025.
- [x] The `rocm[libraries]` sdist builds at install time in the FHS shell;
      setuptools is not constrained below 70.2.
- [x] Heavyweight runtime stays in the `asr-qwen` dependency group with lazy
      imports. A default `uv sync` installs no torch, and the default test suite
      still passes with the group absent (INV-05).
- [x] `doctor` device checks: **actually open** `/dev/kfd` and the ROCm render node
      (currently `/dev/dri/renderD128`) rather than inferring access from group
      membership; then check `torch.cuda.is_available()`, `torch.version.hip`, the
      detected device name, and a small BF16 GPU operation.
- [x] `device: cuda` requested but failing → actionable diagnostic and failure.
      `auto` → CPU fallback with a prominent warning.
- [x] `dtype: auto` resolves with the final device: BF16 on validated ROCm,
      float32 after CPU fallback, BF16 on CPU only if a separate CPU BF16 smoke
      test succeeds. An explicitly requested device/dtype combination that fails
      its smoke test is rejected, never silently downgraded.
- [x] A `host_smoke`-marked test runs the BF16 op on the real device and passes on
      the target host.
- [x] Python recorded in every run's report provenance, and the `runtime` envelope
      that carries Torch, HIP runtime, device and dtype exists and is filled by
      whatever resolves them. **Nothing in M6a resolves a runtime**: `mix` and
      `activity` never load torch, and `transcribe`/`process` raise `DEFERRED: M6b`
      before any model is constructed (ADR-0005), which M5 deliberately established
      and this milestone does not disturb. So M6a owes the resolver, the envelope,
      and the Python half; M6b's adapter fills the rest, and M6b's charter now says
      so. Amended during the start phase; see ADR-0026.
- [x] `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and `HSA_ENABLE_SDMA=0`
      documented, with their rationale, and applied by the dev shell. Promotion to
      host defaults waits for M6b's smoke test.

## Explicitly not in this milestone

- Qwen, `qwen-asr`, the aligner, or any transcription.
- vLLM or FlashAttention. SDPA is the baseline.
- ComfyUI's tuning knobs (`HSA_USE_SVM`, `MIOPEN_FIND_MODE`) — separate performance
  work, not assumed to help ASR.

## Known risks and open questions

- Depends on **OQ-008**. If the stable gfx1151 index does not yield a working
  build, the fallback is another explicitly tested gfx1151 build — record the
  search in an ADR rather than leaving it in shell history.
- **`nix develop .#fhs --command CMD` silently runs nothing and exits 0.**
  `buildFHSEnv`'s shellHook `exec`s into `bwrap` before your command is reached, so you
  get no output, no error, and a success exit code. This wasted time in M0. Use
  `nix run .#fhs -- -c 'CMD'` — `packages.fhs` exists for exactly this — or an
  interactive session. Anything scripted against the FHS shell must use the wrapper.
- The FHS shell's `targetPkgs` list is modelled on the host's ComfyUI service and was
  never validated against a real ROCm install. Treat it as a starting point to refine,
  not as a tested set.
- Do not reuse or modify ComfyUI's venv.
- This milestone is where "it works on my machine" is most tempting. The lock-file
  assertion is what makes it reproducible.

---

## Closeout

### What works end to end

**The GPU is real, and `doctor` proves it rather than claiming it.** From the ROCm
environment on the target host:

```
    ok  kfd node       opened /dev/kfd
    ok  render node    opened /dev/dri/renderD128
    ok  torch          2.9.1+rocm7.13.0 (HIP 7.13.99004-3309c6114a)
    ok  gpu            Radeon 8060S Graphics, gfx1151 — verified bfloat16, float32
    ok  device/dtype   device: auto, dtype: auto resolves to cuda:0 / bfloat16
    ok  rocm env       TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, HSA_ENABLE_SDMA=0
```

Every one of those lines is measured. The nodes are *opened* — never inferred from group
membership, which on this host would report no access on a machine where access plainly
works. The device line means the GPU multiplied two bfloat16 vectors and got exactly the
right answer, and separately did the same in float32.

`dnd-audio doctor --device cuda --dtype bfloat16` asks whether *your* configuration works
here before a four-hour session finds out during it. On this host all three usable
combinations exit 0. On a machine without the runtime, `--device cuda` exits 1 naming the
`asr-qwen` group, and `--dtype bfloat16` exits 1 refusing to compute in another precision
under the name of the one requested.

**There are two environments, and that is load-bearing.** `.venv` is the project
environment and never contains Torch; `.venv-rocm` is where the `asr-qwen` group installs,
from inside the FHS shell. The everyday gate runs `--no-sync` against `.venv`, so it keeps
*running* the group-absent case INV-05 describes rather than proving it once (ADR-0025).

**The lock is what makes it reproducible, in both directions.** Exactly five packages come
from AMD's index — `torch 2.9.1+rocm7.13.0`, `rocm 7.13.0`, `rocm-sdk-core 7.13.0`,
`rocm-sdk-libraries-gfx1151 7.13.0`, `triton 3.5.1+rocm7.13.0` — and every other package in
the lock comes from PyPI. `accelerate 1.12.0` sits in that lock wanting `torch>=2.0.0` and
does not get a CUDA build, which is the failure the spec names and the reason `accelerate`
is here a milestone early.

Underneath: `dnd_audio.runtime` splits probing from resolution, so every rule about devices
and dtypes is tested on a machine with no GPU; the report carries a `runtime` subsection
that M6b fills; and the Python version is recorded in every run's provenance.

Nothing else changed. `ingest`, `activity`, `mix`, `transcribe`, `render` and `process`
behave exactly as they did at M5's close.

### Tests and commands run, with results

`./scripts/gate.sh` — **8 checks, zero skips, 2122 tests** (2028 at M5's close):

```
== gate summary ==
  pass  system dependencies      pass  pytest (offline, cpu)
  pass  ruff check               pass  lock is current
  pass  ruff format              pass  placeholder scan
  pass  type check               pass  plan consistency

GATE PASSED
```

**The same suite from the ROCm environment: `2122 passed, 5 deselected`.** This run is not
in any gate and is the one that matters most here — see the notes below. `host_smoke` on
the real device: `3 passed`.

Every completion-gate criterion, proved by a named command executed in the verify phase:

| Criterion | Proof | Result |
| --- | --- | --- |
| Index `explicit`, `torch` routed in configuration | `TestAmdIndex` | 3 passed |
| AMD index not the default source | same class, plus the PyPI-side assertion | passed |
| Lock holds AMD Torch and ROCm artifacts, no CUDA | `TestTheLockIsWhatWeAskedFor` | 6 passed |
| `rocm[libraries]` builds; setuptools ≥ 70.2 | install transcript; `test_the_fhs_shell_still_carries_the_build_toolchain` | `Built rocm==7.13.0` |
| Group isolated; suite passes with it absent | `TestDependencyIsolation`; both environments | 4 passed; 2122 from each |
| Nodes **opened**, not inferred | `TestDeviceNodes`, `TestGpuChecks` | 15 passed |
| `cuda` failing is fatal; `auto` warns | `TestDeviceResolution`; the real CLI | 12 passed; exit 1 |
| `dtype: auto` follows the final device | `TestDtypeResolution` | 11 passed |
| BF16 on the real device | `test_bf16_runs_on_the_real_gfx1151_device` | passed, gfx1151 |
| Python in provenance; runtime envelope | `TestProvenanceTelemetrySplit`, `TestProvenance` | 12 passed |
| Env vars documented and applied by the shell | `TestRocmEnvironment`; `doctor` | both set |

**Fourteen findings in the verify phase**, from Codex, an independent reviewer agent, self-
review, and mutation. Nothing rejected; twelve fixed, one deferred with its limitation
named, one documented as unfixable. Distilled in
[`../reviews/M6a-code-20260803-0724.md`](../reviews/M6a-code-20260803-0724.md); the plan
review that preceded any code is in
[`../reviews/M6a-plan-20260803-0643.md`](../reviews/M6a-plan-20260803-0643.md).

**Every fix was then mutated and watched go red** — eight source mutations, each caught by
1–6 tests; plus five earlier mutations of the packaging and flake assertions. One of those
runs found that a shipped fix had **no regression test at all**: see the notes.

### Decisions made (→ ADRs)

- **[ADR-0025](../decisions/0025-provisioning-amd-gfx1151-torch.md)** — provisioning Torch
  from AMD's gfx1151 index, and the two environments. Records the version choice and why
  it is *not* the newest build, why every AMD-only package is a direct dependency, why the
  index stays `explicit`, why `transformers`/`accelerate` arrive a milestone early, and why
  `.venv` never receives the group.
- **[ADR-0026](../decisions/0026-probing-is-impure-resolution-is-pure.md)** — the runtime
  contract. Probing imports Torch and runs kernels; resolution is a pure function of what
  probing found, so the spec's whole device/dtype matrix is testable with no GPU. Also
  records why the smoke test is per device *and* per dtype, why the arithmetic is compared
  exactly rather than within a tolerance, and why `gfx1151` is asserted in the smoke test
  rather than inside the resolver.

### Assumptions made and open questions raised

**OQ-008 answered — yes, and the assumption was right about the build and wrong about the
routing.** `torch 2.9.1+rocm7.13.0` resolves, installs, and computes correctly; the
`rocm==7.13.0` sdist **built first time** in the FHS shell with the `targetPkgs` list M0
guessed from the host's ComfyUI module — no additions, no compiler errors, setuptools left
unconstrained. What was wrong is the half nobody asked about: the spec's configuration
sketch does not resolve, because `[tool.uv.sources]` only routes *direct* dependencies.

**OQ-021 raised** — which render node backs the compute device on a host with more than one
GPU. Today there is one, every `renderD*` is opened, and one that opens is a pass. The
failure direction worth closing is a host whose *compute* node is the restricted one; there
`torch.cuda.is_available()` is already false, so the `gpu` check catches it and only the
explanation is wrong. `/sys/class/kfd/kfd/topology/nodes/` carries the DRM render minor per
agent if this ever needs answering.

**Assumptions worth naming that did not become open questions**, because they are about
tooling rather than the world: that `torch.version.hip` distinguishes a ROCm build from a
CUDA one (defended by also reading `gcnArchName`, and the `host_smoke` test fails loudly if
either disappears), and that the AMD index keeps publishing these exact versions — which is
weaker than it sounds, see the note on hashes below.

No milestone here can touch OQ-001..OQ-007 or OQ-009..OQ-020; none of them is about a GPU.

### Notes for future implementors

**`[tool.uv.sources]` does nothing for a package that is not also a direct dependency.**
This cost the milestone its first attempt and it fails *silently*: no warning, no error,
just the wrong registry recorded in the lock. Torch needs `rocm[libraries]` and `triton`,
and `rocm[libraries]` needs `rocm-sdk-core` and `rocm-sdk-libraries-gfx1151`; all four are
listed in the `asr-qwen` group for no reason except that listing them is the only way the
routing reaches them. When M6b adds `qwen-asr` — which pulls Gradio, Flask, `nagisa`,
`soynlp` and Python SoX — the same trap is one AMD-only transitive requirement away. The
fix is always: add it to the group **and** the sources table.
`test_every_routed_package_is_also_a_direct_dependency` is the guard, and
`test_everything_else_comes_from_pypi` is what would catch it if the guard were bypassed.

**Run the default suite from `.venv-rocm` occasionally. No gate does, and that is where the
INV-05 breach was found.** `doctor` legitimately probes the GPU, so the four `test_cli.py`
invocations of it began importing Torch and launching kernels inside the suite the gate
calls CPU-only. On `.venv` nothing happens, because there is no Torch to import — the
breach exists only on the environment nobody runs the suite in, and it announced itself as
an unrelated test in `test_silero.py` failing on run order and blaming itself. There is now
an autouse fixture beside the socket block that fails the test which imported Torch rather
than the one that noticed.

**That fixture cannot see across a process boundary**, exactly as `conftest.py` already
documents for the socket block. Two subprocess tests were launching real HIP kernels and
nothing noticed. Both now shadow `torch` on the child's `PYTHONPATH`, which also turns them
from environment-dependent into deterministic. If you add a subprocess test that touches
`doctor` or `runtime`, shadow it — `test_runtime.py::shadow` and `run_shadowed` are there.

**`git checkout --` in a mutation harness restores to HEAD, not to your working tree.** The
first mutation run over the verify fixes silently *deleted* four of them and the last four
rows measured nothing. The numbers were implausible, which is the only reason it was
caught. **Commit the fixes first, then mutate.** This sits alongside M4's `.pyc` lesson:
`PYTHONDONTWRITEBYTECODE=1` and clearing `__pycache__` are still required for same-length
source edits.

**The re-run then found a fix that had shipped with no test at all.** Reverting the
`device_name`-on-CPU fix left all 204 tests green. That is the same omission as the finding
it fixed, one level up, and only mutation could have surfaced it. **After fixing a review
finding, revert the fix and watch a test fail.** If none does, the finding is not closed.

**The valuable review findings were all "the assertion is adjacent to the rule".** Third
milestone running: M5's review found a test asserting a normalized share instead of the
applied gain; here, `_check_gpu` decided severity on `cuda_available` instead of the build
identity its own docstring named, and `_check_resolution` reported `ok` over a resolver
warning documented as "not decoration". In every case the logic one level down was correct
and got the most design attention. **When reviewing, read the docstring as a claim and go
check the body against it** — two of this milestone's defects were found exactly that way.

**`probe_runtime` must catch `Exception`, not `ImportError`.** A ROCm build with a missing
or mismatched shared library raises `OSError` from the dynamic loader, which is the failure
this whole milestone exists to diagnose, and it escaped as a traceback where the actionable
diagnostic belongs. The old "never raises" test probed *this* machine, whose Torch is
healthy, so it exercised only the path that already worked. If you add a probe, construct
its failures rather than hoping the host provides them.

**The lock pins versions, not bytes.** All eight AMD artifacts carry a URL and no hash,
where every PyPI artifact carries a sha256, because the index publishes none. A re-upload
at the same version would be invisible to `uv lock --check`. No mitigation exists short of
vendoring three gigabytes of wheels. `test_the_amd_index_publishes_no_hashes_...` asserts
the asymmetry, so the day AMD starts publishing hashes it fails and ADR-0025's paragraph
gets rewritten.

**Things that look wrong but are deliberate:**

- **float32 on the CPU needs no smoke test; every dtype on the GPU does.** The GPU is the
  thing being validated, and float32 on a CPU is what every Python numeric stack does. The
  spec asks only for a *separate CPU BF16* test.
- **The smoke test compares for equality, not within a tolerance.** Every operand and every
  product is exact in bfloat16's 8-bit significand — verified arithmetically, twice, by two
  reviewers. A tolerance is precisely how a device that returned nearly-right numbers, the
  signature of a wrongly targeted kernel, would pass the one test meant to catch it.
- **`gfx1151` appears in the `host_smoke` test and in `doctor`'s output, never in the
  resolver.** A resolver that hardcoded one architecture would be wrong on every other
  machine; a claim about *this* host belongs where this host is the subject.
- **Device-node evidence is recorded but does not gate resolution.** Arithmetic that came
  out right *on the device* is strictly stronger than an `open()` on a character device.
  Failing a working GPU because a node glob came back empty would be inventing a failure.
- **The render node is discovered, not hardcoded.** The spec says `renderD128` and qualifies
  it with "currently"; that numbering shifts with DRM enumeration order (OQ-021).
- **`ComputeError`, not `RuntimeError`.** The builtin is taken, and shadowing it inside a
  module about compute runtimes would be genuinely confusing.

**Where the test suite's two minutes go**, since parallelism is the next thing anyone will
reach for: roughly 80% is ~200 end-to-end tests that each run a real pipeline stage over
the canonical fixture — `ffprobe` per source, the checked-in FIR over 10.5 s × 6 tracks,
real Silero inference, `ffmpeg` encode and decode. The remaining ~1900 unit tests take about
24 seconds between them. The canonical-fixture copy is 1.7 ms and is not the problem.

### Deviations from this charter, and why

1. **`transformers` and `accelerate` are in the `asr-qwen` group**, beyond the charter's
   "Explicitly not in this milestone". The charter's own lock criterion is about
   `accelerate`'s transitive `torch>=2.0.0`, and with torch alone in the lock nothing
   competes for it and the assertion passes vacuously. Pinned at `qwen-asr` 0.0.6's exact
   versions so M6b does not relock. Amended in the start phase; ADR-0025.
2. **The report criterion was split.** Nothing in M6a resolves a runtime during a run, so
   M6a owes the resolver, the `runtime` envelope, and the Python half; M6b's adapter fills
   torch, HIP, device and dtype. Amended in the start phase; ADR-0026.
3. **Two environments rather than one `.venv`.** The spec says the project's environment is
   uv's `.venv` in the repository — that sentence is about not reusing ComfyUI's venv, and
   both of these are the repository's. The second one is what keeps INV-05 continuously
   proved instead of proved once. ADR-0025 records it; the spec was not amended, because it
   is not contradicted.
4. **Four ROCm packages are listed as direct dependencies** although nothing here imports
   them. The spec's configuration sketch routes only `torch` and does not resolve. The spec
   prefaces it with "must be equivalent to", and this is equivalent with the closure filled
   in, so the spec was not amended — ADR-0025 is the record.
5. **`doctor` gained `--device` and `--dtype`**, beyond the charter. Without them the
   "`device: cuda` requested but failing" criterion would be demonstrated only by unit
   tests, never through the command an operator runs — and the resolver's explicit-request
   paths would have had no production caller at all. From the plan review.
6. **The `## Working plan` section was corrected during verify** rather than only replaced
   here: it named three functions the restructure had removed and repeated a "newest build"
   claim ADR-0025 had already corrected. Scratch or not, it was what a future implementor
   would read.

### Downstream charters updated

- **M6b** — a new "What M6a already provides" section: the working environment and its
  exact versions, the two-environment rule, the `[tool.uv.sources]` trap that `qwen-asr`
  will meet, the finished `probe_runtime`/`resolve_runtime` seam, the `runtime` provenance
  subsection it must fill and add to `TranscriberIdentity` for INV-08, the note that
  attention implementation has no home yet, and that promoting the two environment
  variables to host defaults is *its* smoke test's job.
- **`INVARIANTS.md`** — INV-05 gains the two-environment rule, the subprocess boundary, and
  the instruction to run the suite from the ROCm environment periodically.
- **`OPEN-QUESTIONS.md`** — OQ-008 answered with versions and evidence; OQ-021 raised.
- **`ROADMAP.md`** — M6a's entry notes the environment split and the routing finding. The
  dependency graph did not move.
- **`ADR-0025`** — amended in verify with the hashes consequence.

### Next smallest step

Begin **M6b — Qwen adapter**. Its dependencies are now closed and its environment is built,
locked, and proved on the real device. Read its "What M6a already provides" section first;
the short version is that `_default_transcriber`'s `DEFERRED: M6b` raise is the only seam
to replace, and it reaches both `transcribe` and `process`.

Before that, one queued piece of unrelated work the owner asked for: **`pytest-xdist`
parallelism in the gate**, as its own commit with its own gate run. It touches every
milestone's tests, so it does not belong inside a milestone.

**Real DJI metadata had not yet been fully validated at this closeout.** M6a neither needed nor
touched that evidence; the later captures and M8 closed it.
