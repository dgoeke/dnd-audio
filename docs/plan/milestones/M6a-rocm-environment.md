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
- [ ] Python, Torch, HIP runtime, device, and dtype recorded in the report.
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
