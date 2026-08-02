# M0 — Foundation

**Status:** not started
**Depends on:** nothing
**Spec sections:** Default technology choices; Target-host runtime (environment
only); Repository and command shape; Output schemas; Error handling and
observability

## Goal

A real Python project that runs, tests, lints, and type-checks on the target host
with no GPU, no model weights, and no network — plus the enforcement rails that make
every later milestone's gate mean something. No audio processing yet.

## Completion gate

- [ ] `pyproject.toml` (Python 3.12, `requires-python` excluding 3.13,
      `license = "Apache-2.0"` to match the repository `LICENSE`),
      `.python-version`, and a committed `uv.lock`.
- [ ] Repo-local Nix flake: `flake.nix` + committed `flake.lock`, `nixpkgs` pinned to
      the host channel (`nixos-25.11`), with `devShells.default` (`mkShell`: Python
      3.12, `uv`, FFmpeg, SoX, native libs, build toolchain) and `devShells.fhs`
      (`buildFHSEnv` `.env`, reserved for M6a). Does not reuse or modify the host's
      ComfyUI venv. See ADR-0002, and `LOCAL.md` (uncommitted) for the host's config
      path and the ComfyUI module to model the FHS shell on.
- [ ] `.envrc` containing `use flake`, committed. `direnv allow` yields a shell where
      `python --version` is 3.12 and `uv`, `ffmpeg`, `ffprobe`, `sox` resolve into the
      Nix store. Demonstrated with executed output in the verify phase. This is not a
      formality: the host's own `python3` is 3.13 and it has no `sox` at all, so an
      unactivated shell fails differently than a broken one.
- [ ] Everything in this milestone works in `devShells.default`; the FHS shell is not
      required to run the gate. `nix develop .#fhs` is proven to open a shell, nothing
      more.
- [ ] Typer CLI with all commands registered and wired to stubs that fail with a
      clear "not implemented in M<N>" error: `process`, `inspect`, `ingest`,
      `transcribe`, `mix`, `render`, `doctor`, `models fetch`.
- [ ] `doctor` genuinely works for its non-GPU checks: system dependencies
      (ffmpeg/ffprobe presence and versions), writable paths, disk space.
      GPU checks land in M6a.
- [ ] Pydantic models for `session.yaml` (including the full `timecode`,
      `asr`, `activity`, `mix`, and `recovery` shapes) and skeletons for
      `manifest.json`, `transcript.json`, and `ingest-report.json`.
- [ ] DJI frame-rate labels map to exact rational rates
      (`23.98F`, `24F`, `25F`, `29.97F`, `29.97DF`, `30F`, `50F`, `60F`), with
      incompatible drop-frame syntax rejected. Unit-tested (INV-04).
- [ ] JSON Schema artifacts generate from the Pydantic models into a checked-in
      location, and a test fails when a model changes without regeneration.
- [ ] `Transcriber` and `ActivityDetector` protocols exist with deterministic fake
      implementations (INV-10).
- [ ] Report writer produces a valid `ingest-report.json` skeleton with
      `overall_status`, per-stage status, and separated provenance vs. telemetry
      sections; writes atomically (INV-13, INV-02).
- [ ] Autouse pytest fixture blocks socket access; a test proves an attempted
      connection fails (INV-05).
- [ ] `host_smoke` pytest marker registered and excluded by the gate.
- [ ] Atomic-write and canonical-JSON helpers exist with tests (INV-02).
- [ ] `.gitignore` covers session audio, weights, secrets, work, and output.
- [ ] `./scripts/gate.sh` passes end to end with its `TYPE_CHECK` command filled in,
      and its system-dependency step extended to fail when the flake environment is
      not active — `sox` present, and `python --version` reporting 3.12 from
      `/nix/store`. Invoking `nix` from the gate is not allowed (it would need the
      network on a cold store).

## Explicitly not in this milestone

- Any real audio I/O, ffprobe invocation, or file discovery. That is M1.
- PyTorch, ROCm, the AMD wheel index, or the `asr-qwen` dependency group beyond
  declaring the group's existence and keeping it out of the default sync.
- Silero.

## Known risks and open questions

- Choice of strict type checker is undecided — make it and record an ADR.
- Do not try to make `.envrc` load the FHS shell. `nix print-dev-env` ends with
  `eval "$shellHook"` and `buildFHSEnv`'s hook `exec`s into `bwrap`, so direnv would
  hang or die. This was already tested; see ADR-0002 before relitigating it.
- Keeping heavyweight deps out of the base environment is easy to get wrong now
  and expensive to untangle in M6a. Verify the default `uv sync` installs no
  torch.
- The report's provenance/telemetry split (INV-02, INV-03) is much cheaper to
  design now than to retrofit once five stages write to it.
