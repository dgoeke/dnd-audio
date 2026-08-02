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
- [ ] Typer CLI with every command registered: `process`, `inspect`, `ingest`,
      `transcribe`, `mix`, `render`, `doctor`, `models fetch`. All except `doctor`
      are wired to stubs that fail with a clear "not implemented in M<N>" error.
      (Amended during the start phase: the original wording listed `doctor` among
      the stubs and then required it to work. `doctor` is implemented.)
- [ ] `doctor` genuinely works for its non-GPU checks: system dependencies
      (ffmpeg/ffprobe presence and versions), writable paths, disk space.
      GPU checks land in M6a. Invoking `ffmpeg -version` / `ffprobe -version` is
      part of this and is not the "no ffprobe invocation" non-goal below, which is
      about probing session audio.
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
depends on it. Amended after the independent plan review; the record of what was
accepted and rejected is at the end of this section.

1. `pyproject.toml` + `uv.lock`. Runtime: Typer, Pydantic, PyYAML, NumPy. Dev group:
   pytest, ruff, the chosen type checker, `jsonschema`. The `asr-qwen` group is declared
   **empty** with a comment naming M6a/M6b — declaring its existence is in scope,
   resolving the heavyweight stack into the lock is not.
2. `determinism.py` — canonical JSON, atomic write, and hashing. `sha256_file` streams in
   a bounded chunk and never reads a whole file into memory (INV-07 applies to every
   helper a later milestone will point at a multi-gigabyte recording). The only
   float-producing conversion is built on an integer-millisecond quantizer with an
   explicit, documented tie rule, so no caller can reach a general float helper by
   accident (INV-04).
3. `timecode.py` — the eight DJI rate labels to exact rationals plus a drop-frame flag,
   and timecode-string validation that rejects drop-frame syntax at a non-drop rate.
   Deliberately stops short of timecode-to-sample conversion, which is M2's.
4. `config.py` — the full `session.yaml` model with unknown keys forbidden, including
   `timecode`, `asr` (with the optional explicit model and aligner revisions the spec
   requires), `activity`, `mix`, and `recovery.source_time_overrides`. INV-11 is
   structural here: `track_id` is the key, and receiver fields validate without ever
   becoming identity. `max_segment_s` is capped at 120 with `OQ-009` cited at the cap.
   This step also defines the **resolved-configuration projection**: the validated model
   dumped with defaults materialized and paths normalized, canonically serialized, and
   hashed. Every later cache key is built on it (INV-08), so a config that omits a
   default must hash identically to one that states it.
5. `artifacts/` — `manifest.py`, `transcript.py` (matching the spec's baseline exactly),
   and `report.py` with a structural provenance/telemetry split so INV-03 cannot be
   violated by accident. Stage records are held in a fixed stage order, never in
   completion order. The manifest and report schemas are explicitly **provisional** until
   the milestone that owns the artifact closes; after that, additive optional fields only,
   and anything else bumps `schema_version`.
6. Schema export plus checked-in JSON Schema artifacts. One function returns the
   filename-to-canonical-JSON mapping; the generator script writes it and the drift test
   compares against it, so the two cannot disagree.
7. `interfaces.py` and `fakes.py` — `Transcriber` and `ActivityDetector` protocols with a
   **scripted** fake transcriber and a scripted-mask detector (INV-10). Request types
   carry integer sample coordinates and a bounded audio window, never an unconstrained
   session-length array (INV-07).
8. `report.py` — stage accumulation, `overall_status` rollup, atomic write on partial
   failure (INV-13), and deliverable hashes for everything except the report itself
   (ADR-0003).
9. `doctor.py` — tool presence and versions, interpreter identity, writable-path probe,
   free disk space, with machine-readable output.
10. `cli.py` — every command registered. Everything except `doctor` raises
    `NotImplementedError` annotated `DEFERRED: M<n>`, deliberately visible to
    `scripts/scan_placeholders.py` rather than hidden behind a custom exception type, and
    the entry point turns that into a clean message and a dedicated exit code.
11. `tests/` — the network block (see below), then one module per source module.
12. `scripts/gate.sh` — `TYPE_CHECK` filled in; the system-dependency step extended to
    require SoX and a store-resolved Python 3.12 with a "run `direnv allow`" hint; every
    tool invocation switched to a no-sync form so the gate provably performs no network
    I/O, with a preflight that names `uv sync` when the environment is missing.
13. ADRs for the strict type checker, the schema-artifact generation/drift/versioning
    mechanism, and the status, exit-code, and enum vocabularies the spec left open.
    ADR-0003 is already written.

### The network block

Blocking outbound `connect` is not enough: unconnected UDP and name resolution both leave
the machine. The autouse fixture blocks socket **creation** for `AF_INET`/`AF_INET6` and
blocks name resolution (`getaddrinfo`, `gethostbyname`), while leaving `AF_UNIX`
available — a Unix socket cannot reach the network, and pytest internals use them.

There is no `host_smoke` exemption. INV-06 permits network access to exactly one command,
`models fetch`, so the opt-out is a separate explicit `allow_network` marker reserved for
that, not a side effect of needing a GPU. The test that proves the block works therefore
lives in the default suite, where the gate actually runs it.

The honest boundary: a subprocess the tests spawn has its own address space and is not
covered. Nothing in the default suite spawns a network-capable subprocess other than this
project's own CLI, and OS-level isolation is not worth its complexity here. This limit is
documented in `conftest.py` rather than papered over.

### Gate criteria mapped to proof

| Criterion | Proof |
| --- | --- |
| `pyproject.toml`, `.python-version`, committed lock | `tests/test_packaging.py` reads `pyproject.toml` with `tomllib` and asserts `requires-python`, the Apache-2.0 license, and the console-script entry point; `.python-version` content asserted; the gate's `uv lock --check`; `git ls-files` checked during verification |
| Base environment stays free of heavyweight deps | `tests/test_packaging.py` parses `uv.lock` and asserts no `torch` in the default resolution and that `asr-qwen` is declared but empty |
| Flake with both shells, committed lock | executed `nix develop` and `nix run .#fhs` output, quoted in the closeout |
| `.envrc`; activated shell gives 3.12 and store-resolved tools | a clean `direnv exec .` invocation — not a shell that was already active — plus the gate's system-dependency step, which fails outside the flake |
| Everything works in the default shell | the gate runs there and never invokes `nix` |
| Every command registered; stubs fail clearly | `tests/test_cli.py` via `CliRunner`, **plus** a subprocess test running the installed `dnd-audio` console script offline, so a broken build backend or `src/` discovery cannot pass |
| `doctor` works for non-GPU checks | `tests/test_doctor.py`: real tool versions parsed, non-writable path detected, free space reported, `--json` shape asserted |
| `session.yaml` models, full shape | `tests/test_config.py`: a valid fixture exercising every field and default, plus a rejection table — unknown key, `active_tracks` naming an unrostered track, duplicate `track_id`, absolute or escaping input path, both timing values in one override, an override carrying no information at all, malformed hash, `max_segment_s` above the cap, invalid bitrate, `receiver_channel` out of range |
| Resolved-config hash is a sound cache identity (INV-08) | `tests/test_config.py`: a config omitting defaults hashes identically to one stating them; changing any output-affecting field changes the hash |
| Manifest / transcript / report skeletons | `tests/test_artifacts.py`: constructed instances validated against the **checked-in** schema files, not round-tripped through the model that produced them |
| Rate labels to exact rationals; bad drop-frame syntax rejected (INV-04) | `tests/test_timecode.py`: all eight labels asserted against exact rationals; drop-frame separator at a non-drop rate, out-of-range frames, and legitimately skipped drop-frame numbers all raise |
| Schemas generate; drift test fails on model change | `tests/test_schema_drift.py`: checked-in bytes equal generated bytes, generated in a **subprocess under a different `PYTHONHASHSEED`** so iteration order cannot hide; plus a test that mutates a model in-process and asserts the comparison then fails |
| Protocols and deterministic fakes (INV-10) | `tests/test_fakes.py`: runtime protocol conformance; a scripted response is returned verbatim; identical input twice yields identical output |
| Report: every stage has a status, structured errors, deliverable hashes | `tests/test_report.py`: `complete`, `failed`, and `skipped` all represented; a structured error survives serialization; deliverable hashes present and `ingest-report.json` never among them (ADR-0003) |
| Report determinism and the provenance/telemetry split | `tests/test_report.py`: the same stage updates applied in two different orders produce byte-identical provenance and decisions (INV-02); no time-typed field exists in provenance (INV-03) |
| Report atomicity (INV-13) | `tests/test_report.py`: an interrupted write leaves the previous file intact and no temp file behind |
| Socket block proven (INV-05) | `tests/test_network_blocked.py`, in the **default** suite: an outbound TCP connect, an unconnected UDP `sendto`, and a DNS lookup each raise the block error; an `AF_UNIX` socket still works |
| `host_smoke` registered and excluded | strict marker configuration plus the gate's `-m 'not host_smoke'` |
| Atomic-write, canonical-JSON, and hashing helpers (INV-02, INV-07) | `tests/test_determinism.py`: repeated writes byte-identical, key order independent of insertion order, non-finite floats rejected, no temp files left behind, `sha256_file` matches a known digest while never reading more than its chunk size, millisecond quantization exact at tie boundaries |
| `.gitignore` coverage | `tests/test_packaging.py`: `git check-ignore` asserted against sentinel paths for session audio, weights, secrets, work, and output |
| Gate passes end to end | `./scripts/gate.sh` with zero skips, quoted in the closeout |

### Invariants at risk, and what stops it

- **INV-02** — a helper that looks deterministic but is not, or an artifact whose content
  depends on which stage finished first. One canonical serializer, sorted keys, fixed
  stage ordering, non-finite floats rejected, a write-twice byte comparison, an
  applied-in-two-orders comparison, and schema generation under a varied hash seed.
- **INV-03** — a timestamp landing in provenance because it was convenient. The split is
  structural, and a test asserts provenance carries no time-typed field.
- **INV-04** — a fractional rate as a binary float. Rates are exact rationals, the tests
  compare against rationals rather than decimal literals, and the only float-producing
  path is an integer-millisecond quantizer with a documented tie rule.
- **INV-05** — over-blocking (breaking pytest internals) or under-blocking (UDP, DNS,
  a blanket `host_smoke` exemption). Addressed in "The network block" above.
- **INV-07** — a hashing or interface helper that quietly reads a whole session into
  memory. `sha256_file` streams with a bounded chunk and is tested for it; request types
  carry sample coordinates and a bounded window rather than a session-length array.
- **INV-08** — a cache identity built on raw YAML bytes, so that omitting a default looks
  different from stating it. The resolved-configuration projection is defined in M0 and
  tested for exactly that equivalence.
- **INV-10** — a fake too clever to be useful. A scripted fake returns what the test told
  it to, which is what M4 needs for truncation, overlap, and alignment-failure cases.
- **INV-13** — a stub CLI that exits zero, or a report missing a stage. Distinct exit
  codes asserted per command; all three stage statuses represented in the report test.

### Charter points amended during this phase

1. **The `doctor` contradiction.** The gate listed `doctor` among the commands wired to
   not-implemented stubs and then required it to work. Amended above: everything except
   `doctor` is a stub.
2. **`doctor` versus the "no ffprobe invocation" non-goal.** Reading tool versions means
   executing them; the non-goal is about probing session audio. Clarified above.
3. **"Gate passes end to end" has to mean zero skips.** It already passes *with* one. M0
   is not complete until that skip is gone. Recorded here rather than in the gate list,
   which already says what it says.

### Independent review (Codex, `docs/plan/reviews/M0-plan-20260802-0912.md`)

**Accepted.** The network block redesign (UDP, DNS, no `host_smoke` exemption, proof test
inside the default suite). The report coverage gaps — `skipped` status, structured
errors, deliverable hashes, order-independent provenance. Ordering determinism generally,
including varying `PYTHONHASHSEED` for schema generation. Bounded-memory contracts on
`sha256_file` and on the protocol request types. A provisional-until-owned policy for the
skeleton schema versions. The missing `session.yaml` cases: explicit model and aligner
revisions, a `max_segment_s` cap citing OQ-009, an override supplying no information.
The console-script subprocess test, since a `CliRunner` test passes even when packaging is
broken. Executable proofs in place of assertion-free rows: `tomllib` for project metadata,
`git check-ignore` for the ignore rules, `uv.lock` inspection for the no-Torch rule, and a
clean `direnv exec .` for activation. A canonical resolved-configuration projection for
INV-08. A scripted fake transcriber instead of a content-hash-derived one. An integer
quantizer instead of a general float helper.

**Accepted with a change of remedy.** The report self-hash impossibility is real, and the
review is right that the wording had to change; ADR-0003 records the fix, amends INV-13
and the spec, and rejects the sidecar alternative it proposed for reasons written there.
On network-capable subprocesses, OS-level isolation is not worth its complexity in M0 —
the boundary is documented in `conftest.py` instead.

**Rejected.** *"Enumerate the configurable thresholds later required by activity and
automix."* Those fields belong to M3 and M5, which have not chosen them; inventing them
now would guess at values those milestones must derive from real signal behaviour, and
`extra="forbid"` guarantees that adding one later is a visible, deliberate change rather
than a silent drift. M0 models exactly the fields the spec's own example defines.

### Deliberately not doing

Real audio I/O, session-file probing, discovery, and the synthetic fixture generator (M1).
PyTorch, ROCm, the AMD wheel index, and any real content in the `asr-qwen` group
(M6a/M6b). Silero (M3). Any manifest content beyond a skeleton M1 will extend.
