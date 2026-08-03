# M6a — ROCm environment and GPU doctor

**Status:** not started
**Depends on:** M0 (can run in parallel with M1–M5)
**Spec sections:** Target-host runtime

## Goal

A locked, reproducible PyTorch installation for the target host's gfx1151 GPU inside the
repo-local FHS shell, and a `doctor` that proves the device is genuinely usable —
without any Qwen code yet.

## Completion gate

- [ ] `[[tool.uv.index]] amd-gfx1151` declared `explicit = true` with
      `[tool.uv.sources] torch = { index = "amd-gfx1151" }`, in configuration rather
      than as a command-line index choice a later sync can forget.
- [ ] The AMD index is **not** the default source for unrelated packages.
- [ ] `uv.lock` demonstrably contains the intended AMD Torch and ROCm artifacts and
      no CUDA build resolved from ordinary PyPI. Asserted by a test that reads the
      lock, so `accelerate`'s transitive `torch>=2.0.0` cannot quietly win.
      **`transformers` and `accelerate` are therefore in the `asr-qwen` group from
      this milestone**, beyond the original non-goals below: with torch alone in the
      lock nothing competes for it and the assertion passes vacuously. `qwen-asr`
      itself stays out. Amended during the start phase; see ADR-0025.
- [ ] The `rocm[libraries]` sdist builds at install time in the FHS shell;
      setuptools is not constrained below 70.2.
- [ ] Heavyweight runtime stays in the `asr-qwen` dependency group with lazy
      imports. A default `uv sync` installs no torch, and the default test suite
      still passes with the group absent (INV-05).
- [ ] `doctor` device checks: **actually open** `/dev/kfd` and the ROCm render node
      (currently `/dev/dri/renderD128`) rather than inferring access from group
      membership; then check `torch.cuda.is_available()`, `torch.version.hip`, the
      detected device name, and a small BF16 GPU operation.
- [ ] `device: cuda` requested but failing → actionable diagnostic and failure.
      `auto` → CPU fallback with a prominent warning.
- [ ] `dtype: auto` resolves with the final device: BF16 on validated ROCm,
      float32 after CPU fallback, BF16 on CPU only if a separate CPU BF16 smoke
      test succeeds. An explicitly requested device/dtype combination that fails
      its smoke test is rejected, never silently downgraded.
- [ ] A `host_smoke`-marked test runs the BF16 op on the real device and passes on
      the target host.
- [ ] Python recorded in every run's report provenance, and the `runtime` envelope
      that carries Torch, HIP runtime, device and dtype exists and is filled by
      whatever resolves them. **Nothing in M6a resolves a runtime**: `mix` and
      `activity` never load torch, and `transcribe`/`process` raise `DEFERRED: M6b`
      before any model is constructed (ADR-0005), which M5 deliberately established
      and this milestone does not disturb. So M6a owes the resolver, the envelope,
      and the Python half; M6b's adapter fills the rest, and M6b's charter now says
      so. Amended during the start phase; see ADR-0026.
- [ ] `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and `HSA_ENABLE_SDMA=0`
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

## Working plan

_Scratch section, written during the start phase and replaced by the Closeout. Names
and test identifiers were corrected in the verify phase after the probe was restructured —
until then this section named three functions that no longer existed, which is the kind of
repository memory a future implementor would have trusted._

### Order of work

1. **`pyproject.toml`** — the `amd-gfx1151` index (`explicit = true`), `[tool.uv.sources]`
   routing, the `asr-qwen` group, and a `torch` entry in `[[tool.mypy.overrides]]`. The
   gate's venv will never contain torch, so strict mode needs the same treatment
   `onnxruntime` got in ADR-0013.
2. **`uv lock`, then `uv sync --group asr-qwen` inside `nix run .#fhs -- -c`.** This is
   OQ-008's answer and everything after it depends on the versions it produces. The index
   publishes **no PEP 658 `.metadata` sidecars and no hashes** (checked during planning),
   so uv must download the whole multi-GB wheel to resolve.
3. **`flake.nix`** — refine `targetPkgs` against whatever step 2 actually needed, and apply
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and `HSA_ENABLE_SDMA=0` in both shells with
   their rationale. `flake.lock` does not move.
4. **`src/dnd_audio/runtime.py`** — probing and resolution, separated (below).
5. **`src/dnd_audio/doctor.py`** — the device checks, with an injectable environment.
6. **`src/dnd_audio/artifacts/report.py`** — `RuntimeProvenance`, `Provenance.runtime`,
   `ReportBuilder.record_runtime()`, Python on every run; regenerate the report schema.
7. **Tests**, written alongside each of 4–6 rather than after.
8. **ADR-0025, ADR-0026, OQ-008's answer, OQ-021**, and M6b's inherited section.

The target is `torch 2.9.1+rocm7.13.0` (cp312, linux_x86_64). **Not the newest build on
the index** — 2.10.0 and 2.11.0 are also published for cp312, and the first draft of this
plan wrongly said otherwise. ROCm 7.13 is the generation this host already runs a GPU
workload on, which is evidence about the *runtime* and none about a torch version; among
three candidates with no host evidence a patch release beats two `x.y.0` releases. Tested
beats newest (ADR-0025).

### The one design decision worth stating up front

**Probing is impure; resolution is pure.** `probe_runtime()` imports torch lazily, opens
the device nodes, runs the smoke operations, and returns one frozen `RuntimeProbe`.
`resolve_runtime(device, dtype, *, probe)` is then a total function of that probe, so every
rule the spec states — `cuda` requested and failing is fatal, `auto` falls back to CPU with
a warning, `dtype: auto` follows the final device, BF16 on CPU only behind its own smoke
test, an explicit combination that fails is rejected rather than downgraded — is unit-tested
offline on a machine with no GPU and no torch. A resolver that had to be exercised through a
real device could only ever be tested on one host, on the happy path, which is the shape of
"it works on my machine" this charter warns about.

The render node is **discovered** by globbing `/dev/dri/renderD*` rather than hardcoding
`renderD128`, and every candidate is opened. The spec names the current node with the word
"currently"; which node is the compute one on a host with more than one GPU is **OQ-021**.

### Completion gate → the proof for each criterion

| Criterion | Proof |
| --- | --- |
| Index declared `explicit`, `torch` routed to it | `test_packaging.py::TestAmdIndex` |
| AMD index is not the default source | same class: `explicit = true`, and `numpy`/`pydantic` still resolve from PyPI in the lock |
| Lock holds AMD Torch, no CUDA build from PyPI | `test_the_lock_resolves_torch_from_the_amd_index` (registry is the AMD index, version carries `+rocm`) and `test_no_cuda_wheels_are_in_the_lock` (no `nvidia-*`) — non-vacuous because `accelerate` is in the lock wanting `torch>=2.0.0` |
| `rocm[libraries]` builds; setuptools ≥ 70.2 | `test_setuptools_is_not_constrained_below_70_2`, plus the executed `uv sync --group asr-qwen` transcript from inside the FHS wrapper |
| Default sync installs no torch; suite passes without the group | `test_asr_qwen_is_not_a_default_group`; and the gate itself, which runs `--no-sync` against a venv that has none |
| `doctor` **opens** `/dev/kfd` and the render node | `test_runtime.py::TestDeviceNodes` — real `os.open` against temporary files: openable → ok, `chmod 000` → fail, absent → warn |
| `torch.cuda`, `torch.version.hip`, device name, BF16 op | `test_doctor.py::TestGpuChecks` with injected probes; the real path under `host_smoke` |
| `cuda` requested and failing is fatal; `auto` warns and falls back | `test_runtime.py::TestDeviceResolution` |
| `dtype: auto` follows the device; explicit failing combination rejected | `test_runtime.py::TestDtypeResolution`, over the whole matrix |
| BF16 op on the real device | `test_runtime.py::test_bf16_runs_on_the_real_gfx1151_device`, `host_smoke`, quoted in verify |
| Python in the report; the runtime envelope | `test_report.py`, `test_inspect_report.py`, `test_schema_drift.py` |
| Env vars documented and applied by the dev shell | `flake.nix` comments, `doctor`'s `rocm env` check, ADR-0025 |

### Invariants this could plausibly violate

- **INV-05 is the real exposure.** A module-scope `import torch`, or a doctor test that
  reaches a real device, makes the default suite depend on the group being installed. Stopped
  by a subprocess test proving `import dnd_audio.runtime` loads no torch — the same technique
  `test_silero.py` uses for `onnxruntime` — and by injectable probes in every doctor test, so
  no default test ever constructs a real one.
- **INV-08.** Torch, HIP, device and dtype all belong in the ASR cache key. M6a defines them
  in one place (`RuntimeProvenance`) so M6b adds them to `TranscriberIdentity` without a
  second vocabulary to disagree with.
- **INV-02/INV-03.** The new provenance subsection carries versions and a resolution only —
  no timings, no counters, nothing that differs between two identical runs.

### Known risk carried into implementation

`transformers` and `accelerate` widen a single universal resolution, so they can move base
pins (`numpy`, `typing-extensions`, `pyyaml`). The `uv.lock` diff gets read for changes
**outside** the new group and the full gate re-run; a moved base pin is a finding to report,
not something to absorb quietly.

### Deliberately not doing

Everything under "Explicitly not in this milestone" above, unchanged: no Qwen, no
`qwen-asr`, no aligner, no transcription, no vLLM or FlashAttention, and none of ComfyUI's
`HSA_USE_SVM`/`MIOPEN_FIND_MODE` tuning. The two env vars are applied by the dev shell and
**not** promoted to host defaults — that waits for M6b's smoke test, as the charter says.
