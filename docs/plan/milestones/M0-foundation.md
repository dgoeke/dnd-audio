# M0 — Foundation

**Status:** in progress
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

---

## Working plan

_Scratch section written at start time. Replaced by the Closeout at close time._

### Preconditions checked

Working tree clean at `82fb3b8`; M0 depends on nothing; `./scripts/gate.sh` passes at
HEAD with the single expected skip (`ruff / types / pytest: no pyproject.toml yet`),
which this milestone removes.

Facts verified before planning: the host interpreter is 3.13 and `sox` is absent, so the
flake shell is load-bearing rather than decorative. nixpkgs at the channel the host
tracks provides Python 3.12.13, and its `ffmpeg` carries `libmp3lame`, `loudnorm`, and
`ebur128` — what M5 will need, confirmed now rather than discovered later.

### Split into two phases

The milestone is implemented as two commits with a gate-green boundary between them, at
the invoker's request: the environment is built first, and the rest of the work is done
from inside it. Phase A lands no Python, so the tree stays green (with the pre-existing
skip) at the pause. `STATE.md` records M0 as in progress at that point so a cleared
context can resume from the repository alone.

**Phase A — flake and direnv machinery.** `flake.nix` (nixpkgs pinned to the channel the
host's configuration tracks, `x86_64-linux` only, since `buildFHSEnv` is Linux-only and
the target host is the only host), committed `flake.lock`, `.envrc` containing
`use flake`, and `.python-version`.

- `devShells.default` — `mkShell` with Python 3.12, `uv`, FFmpeg, SoX, the native
  libraries CPU wheels link against, and a build toolchain. Its `shellHook` points `uv`
  at the flake's interpreter and disables uv's own interpreter downloads, so uv can never
  quietly substitute a different Python.
- `devShells.fhs` — a `buildFHSEnv` `.env` whose `targetPkgs` is seeded from the host's
  ComfyUI module. Reserved for M6a; M0 proves only that it opens.

Verified by executing `python --version` and tool lookups inside `nix develop`, opening
the FHS shell, and re-running the gate.

**Phase B — the Python project**, in this order, each step verifiable before the next
depends on it:

1. `pyproject.toml` + `uv.lock`. Runtime: Typer, Pydantic, PyYAML, NumPy. Dev group:
   pytest, ruff, the chosen type checker, `jsonschema`. The `asr-qwen` group is declared
   **empty** with a comment naming M6a/M6b — declaring its existence is in scope,
   resolving the heavyweight stack into the lock is not.
2. `determinism.py` — canonical JSON, atomic write, SHA-256 helpers, and the single
   rational-to-millisecond boundary conversion INV-04 permits.
3. `timecode.py` — the eight DJI rate labels to exact rationals plus a drop-frame flag,
   and timecode-string validation that rejects drop-frame syntax at a non-drop rate.
   Deliberately stops short of timecode-to-sample conversion, which is M2's.
4. `config.py` — the full `session.yaml` model with unknown keys forbidden, including
   `timecode`, `asr`, `activity`, `mix`, and `recovery.source_time_overrides`. INV-11 is
   structural here: `track_id` is the key, and receiver fields validate without ever
   becoming identity.
5. `artifacts/` — `manifest.py` (a small skeleton explicitly marked as M1's extension
   point), `transcript.py` (matching the spec's baseline exactly), and `report.py` with a
   structural provenance/telemetry split so INV-03 cannot be violated by accident.
6. Schema export plus checked-in JSON Schema artifacts. One function returns the
   filename-to-canonical-JSON mapping; the generator script writes it and the drift test
   compares against it, so the two cannot disagree.
7. `interfaces.py` and `fakes.py` — `Transcriber` and `ActivityDetector` protocols with a
   content-hash-seeded deterministic fake transcriber and a scripted-mask detector
   (INV-10).
8. `report.py` — stage accumulation, `overall_status` rollup, atomic write on partial
   failure (INV-13).
9. `doctor.py` — tool presence and versions, interpreter identity, writable-path probe,
   free disk space, with machine-readable output.
10. `cli.py` — every command in the spec's list registered. Unimplemented stages raise
    `NotImplementedError` annotated `DEFERRED: M<n>`, deliberately visible to
    `scripts/scan_placeholders.py` rather than hidden behind a custom exception type, and
    the entry point turns that into a clean message and a dedicated exit code.
11. `tests/` — an autouse fixture blocking outbound connects (AF_UNIX still permitted,
    `host_smoke` exempt), then one module per source module.
12. `scripts/gate.sh` — `TYPE_CHECK` filled in; the system-dependency step extended to
    require SoX and a store-resolved Python 3.12 with a "run `direnv allow`" hint; every
    tool invocation switched to a no-sync form so the gate provably performs no network
    I/O, with a preflight that names `uv sync` when the environment is missing.
13. ADRs for the strict type checker, the schema-artifact generation and drift mechanism,
    and the status/enum vocabularies the spec left open.

### Gate criteria mapped to proof

| Criterion | Proof |
| --- | --- |
| `pyproject.toml`, `.python-version`, committed lock | the gate's `uv lock --check` step |
| Flake with both shells, committed lock | executed `nix develop` and `nix develop .#fhs` output, quoted in the closeout |
| `.envrc`; activated shell gives 3.12 and store-resolved tools | the gate's system-dependency step, which fails outside the shell, plus quoted output |
| Everything works in the default shell | the gate runs there and never invokes `nix` |
| All commands registered; stubs fail clearly | `tests/test_cli.py`: every command invocable, exits with the not-implemented code, names its milestone |
| `doctor` works for non-GPU checks | `tests/test_doctor.py`: real tool versions parsed, non-writable path detected, free space reported |
| `session.yaml` models, full shape | `tests/test_config.py`: a valid fixture plus a rejection table — unknown key, `active_tracks` naming an unrostered track, duplicate `track_id`, absolute or escaping input path, both timing values in one override, malformed hash, invalid bitrate |
| Manifest / transcript / report skeletons | `tests/test_artifacts.py`: constructed instances validated against the **checked-in** schema files, not round-tripped through the model that produced them |
| Rate labels to exact rationals; bad drop-frame syntax rejected (INV-04) | `tests/test_timecode.py`: all eight labels asserted against exact rationals; drop-frame separator at a non-drop rate, out-of-range frames, and legitimately skipped drop-frame numbers all raise |
| Schemas generate; drift test fails on model change | `tests/test_schema_drift.py`: checked-in bytes equal generated bytes, plus a test that mutates a model in-process and asserts the comparison then fails |
| Protocols and deterministic fakes (INV-10) | `tests/test_fakes.py`: runtime protocol conformance, and identical input twice yielding identical output |
| Report skeleton, statuses, provenance/telemetry split, atomicity | `tests/test_report.py`: a complete plus a failed stage does not roll up to complete; schema-validates; no time-typed field in provenance (INV-03); an interrupted write leaves the previous file intact and no temp file behind |
| Socket block proven (INV-05) | `tests/test_network_blocked.py`: an outbound connect raises the block error, and a `host_smoke` control test shows the exemption works |
| `host_smoke` registered and excluded | strict marker configuration plus the gate's `-m 'not host_smoke'` |
| Atomic-write and canonical-JSON helpers (INV-02) | `tests/test_determinism.py`: repeated writes byte-identical, key order independent of insertion order, non-finite floats rejected, no temp files left behind |
| `.gitignore` coverage | reviewed against the hard rules; extended only if something is missing |
| Gate passes end to end | `./scripts/gate.sh` with zero skips, quoted in the closeout |

### Invariants at risk, and what stops it

- **INV-02** — a helper that looks deterministic but is not. One canonical serializer
  used everywhere, sorted keys, non-finite floats rejected, and a write-twice byte
  comparison.
- **INV-03** — a timestamp landing in provenance because it was convenient. The split is
  structural, and a test asserts provenance carries no time-typed field.
- **INV-04** — a fractional rate as a binary float. Rates are exact rationals and the
  tests compare against rationals, never against a decimal literal.
- **INV-05** — over-blocking (breaking pytest internals) or under-blocking. Outbound
  connects are blocked rather than socket construction; AF_UNIX stays available; both the
  block and the `host_smoke` exemption are proven.
- **INV-10** — a fake too thin to prove anything. The fake transcriber derives its text
  and word times from the audio content hash, so M4 can assert real properties.
- **INV-13** — a stub CLI that exits zero. Distinct exit codes, asserted per command.

### Charter points that look wrong now that the spec has been reread

1. **`doctor` versus the "no ffprobe invocation" non-goal.** The gate requires FFmpeg and
   FFprobe *versions*, which means executing them. The non-goal is about probing session
   audio. The version probe is implemented; the clarification is recorded rather than
   silently resolved.
2. **"Gate passes end to end" has to mean zero skips.** It already passes *with* one. M0
   is not complete until that skip is gone.

Neither is large enough to need an ADR; both belong in the closeout's deviations.

### Deliberately not doing

Real audio I/O, session-file probing, discovery, and the synthetic fixture generator (M1).
PyTorch, ROCm, the AMD wheel index, and any real content in the `asr-qwen` group
(M6a/M6b). Silero (M3). Any manifest content beyond a skeleton M1 will replace.
