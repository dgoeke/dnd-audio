# M0 — Foundation

**Status:** closed
**Depends on:** nothing
**Spec sections:** Default technology choices; Target-host runtime (environment
only); Repository and command shape; Output schemas; Error handling and
observability

## Goal

A real Python project that runs, tests, lints, and type-checks on the target host
with no GPU, no model weights, and no network — plus the enforcement rails that make
every later milestone's gate mean something. No audio processing yet.

## Completion gate

- [x] `pyproject.toml` (Python 3.12, `requires-python` excluding 3.13,
      `license = "Apache-2.0"` to match the repository `LICENSE`),
      `.python-version`, and a committed `uv.lock`.
- [x] Repo-local Nix flake: `flake.nix` + committed `flake.lock`, `nixpkgs` pinned to
      the host channel (`nixos-25.11`), with `devShells.default` (`mkShell`: Python
      3.12, `uv`, FFmpeg, SoX, native libs, build toolchain) and `devShells.fhs`
      (`buildFHSEnv` `.env`, reserved for M6a). Does not reuse or modify the host's
      ComfyUI venv. See ADR-0002, and `LOCAL.md` (uncommitted) for the host's config
      path and the ComfyUI module to model the FHS shell on.
- [x] `.envrc` containing `use flake`, committed. `direnv allow` yields a shell where
      `python --version` is 3.12 and `uv`, `ffmpeg`, `ffprobe`, `sox` resolve into the
      Nix store. Demonstrated with executed output in the verify phase. This is not a
      formality: the host's own `python3` is 3.13 and it has no `sox` at all, so an
      unactivated shell fails differently than a broken one.
- [x] Everything in this milestone works in `devShells.default`; the FHS shell is not
      required to run the gate. `nix develop .#fhs` is proven to open a shell, nothing
      more.
- [x] Typer CLI with every command registered: `process`, `inspect`, `ingest`,
      `transcribe`, `mix`, `render`, `doctor`, `models fetch`. All except `doctor`
      are wired to stubs that fail with a clear "not implemented in M<N>" error.
      (Amended during the start phase: the original wording listed `doctor` among
      the stubs and then required it to work. `doctor` is implemented.)
- [x] `doctor` genuinely works for its non-GPU checks: system dependencies
      (ffmpeg/ffprobe presence and versions), writable paths, disk space.
      GPU checks land in M6a. Invoking `ffmpeg -version` / `ffprobe -version` is
      part of this and is not the "no ffprobe invocation" non-goal below, which is
      about probing session audio.
- [x] Pydantic models for `session.yaml` (including the full `timecode`,
      `asr`, `activity`, `mix`, and `recovery` shapes) and skeletons for
      `manifest.json`, `transcript.json`, and `ingest-report.json`.
- [x] DJI frame-rate labels map to exact rational rates
      (`23.98F`, `24F`, `25F`, `29.97F`, `29.97DF`, `30F`, `50F`, `60F`), with
      incompatible drop-frame syntax rejected. Unit-tested (INV-04).
- [x] JSON Schema artifacts generate from the Pydantic models into a checked-in
      location, and a test fails when a model changes without regeneration.
- [x] `Transcriber` and `ActivityDetector` protocols exist with deterministic fake
      implementations (INV-10).
- [x] Report writer produces a valid `ingest-report.json` skeleton with
      `overall_status`, per-stage status, and separated provenance vs. telemetry
      sections; writes atomically (INV-13, INV-02).
- [x] Autouse pytest fixture blocks socket access; a test proves an attempted
      connection fails (INV-05).
- [x] `host_smoke` pytest marker registered and excluded by the gate.
- [x] Atomic-write and canonical-JSON helpers exist with tests (INV-02).
- [x] `.gitignore` covers session audio, weights, secrets, work, and output.
- [x] `./scripts/gate.sh` passes end to end with its `TYPE_CHECK` command filled in,
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

## Closeout

### What works end to end

Nothing processes audio yet — by design. What exists is a project that runs, and the
rails that make every later milestone's gate mean something.

`direnv allow`, then `cd` into the repository, gives a shell with Python 3.12.13, `uv`,
FFmpeg 8.0 (with `libmp3lame`, `loudnorm`, and `ebur128`, which M5 needs), and SoX, all
resolved out of `/nix/store`. `nix develop .#fhs` opens the FHS sandbox held for M6a,
and `nix run .#fhs -- -c '<command>'` runs something inside it non-interactively.

`uv run dnd-audio` exposes every command the spec names. `doctor` genuinely works:

```
    ok  python         3.12.13 at .venv/bin/python
    ok  ffmpeg         ffmpeg version 8.0 ...
    ok  ffprobe        ffprobe version 8.0 ...
    ok  sox            SoX v14.4.2 ...
    ok  writable path  .
    ok  free space     633.3 GiB free at .
```

Every other command is a stub that says which milestone it lands in and exits 3 —
distinct from Click's usage exit 2 and from a pipeline failure's 1:

```
$ uv run dnd-audio inspect /tmp/session
not implemented yet: `inspect` lands in M1 (/tmp/session)
$ echo $?
3
```

Underneath: a validated `session.yaml` model, skeleton schemas for `manifest.json`,
`transcript.json`, and `ingest-report.json` with checked-in JSON Schema artifacts, exact
rational frame rates, model seams with scripted fakes, and a report writer that
distinguishes a partial run from a successful one.

### Tests and commands run, with results

`./scripts/gate.sh` — **8 checks, zero skips, 311 tests**:

```
== gate summary ==
  pass  system dependencies      pass  pytest (offline, cpu)
  pass  ruff check               pass  lock is current
  pass  ruff format              pass  placeholder scan
  pass  type check               pass  plan consistency

GATE PASSED
```

Per-area: config 56, timecode 43, determinism 29, report 28, fakes 22, cli 20, doctor 17,
artifacts 15, packaging 32, schema drift 11, network block 12.

Each rail was also proven able to *fail*, because a rail that cannot fail is decoration:

| Rail | How it was falsified | Result |
| ---- | -------------------- | ------ |
| Placeholder scan | Planted `raise NotImplementedError` with no `DEFERRED:` | exit 1, named the line |
| Placeholder scan | Planted `pytest.skip("waiting on nothing")` | exit 1, named the line |
| Schema drift | Added a field to `Manifest` without regenerating | `test_checked_in_bytes_match_the_models[manifest]` failed |
| Gate's flake check | Ran under `env -i PATH=/run/current-system/sw/bin` | `sox MISSING`, `Python 3.13.12 (expected 3.12.x)`, `run: direnv allow`, exit 1 |
| Marker exclusion | `--collect-only` with and without the gate's selector | 311 vs 310 — it really deselects |

Environment proofs: `direnv exec .` (not a shell that was already active) resolves python
and sox from `/nix/store`; `nix run .#fhs -- -c` reports Python 3.12.13 with `/usr/lib`
present; `flake.lock` pins nixpkgs `b6018f87da91d19d0ab4cf979885689b469cdd41` — the same
revision the host's own configuration is on, so the repo shell and the host cannot drift.

### Decisions made (→ ADRs)

- **[ADR-0003](../decisions/0003-report-deliverable-hashes.md)** — the report hashes every
  deliverable except itself. The spec lists `ingest-report.json` as a deliverable *and*
  requires the report to carry the hash of every deliverable produced, which has no fixed
  point: writing the hash changes the bytes it describes. INV-13 and the spec's
  error-handling section each gained the same one-clause carve-out. **This is the first
  amendment to the spec.**
- **[ADR-0004](../decisions/0004-mypy-strict-as-the-type-checker.md)** — mypy `strict`
  with the Pydantic plugin. pyright and basedpyright both need a Node runtime the offline
  gate cannot have; `ty` is not stable enough to base a per-milestone gate on.
- **[ADR-0005](../decisions/0005-vocabularies-the-spec-left-open.md)** — the vocabularies
  the spec implied but did not name: `overall_status`, exit codes, `rollover_policy`,
  `alignment_status`, `asr.dtype`, and why an information-free recovery override is
  rejected.

### Assumptions made and open questions raised

**OQ-013 raised** — how much working disk a full session actually consumes. `doctor`
warns below 40 GiB, derived from arithmetic in `src/dnd_audio/doctor.py`: roughly 15 GiB
of 48 kHz float32 working audio, 5 GiB of 16 kHz derivatives, and 3 GiB of mix
intermediate for a four-hour six-transmitter session. The intermediate *count* is the
guess. M2's preflight settles it.

No existing open question was answered — every one of OQ-001..OQ-012 needs hardware or a
milestone M0 does not touch. **OQ-009** is cited in `config.py` at the `max_segment_s`
cap, which is the one place M0's code depends on an assumption about model behaviour.

Assumptions worth naming that did *not* become open questions, because they are about
tooling rather than the world: that the flake's nixpkgs pin tracks the host channel
(ADR-0002 already owns this), and that `uv run --no-sync` performs no network I/O
(asserted by a test that greps the gate script).

### Notes for future implementors

**The environment is not optional and the gate will tell you so.** `./scripts/gate.sh`
fails outside the flake shell with `sox MISSING` and a `direnv allow` hint. If you are
seeing weird failures, check `python --version` first — the host's own is 3.13, the
version `requires-python` excludes.

**`nix develop .#fhs --command CMD` silently runs nothing and exits 0.** `buildFHSEnv`'s
shellHook `exec`s into `bwrap` before your command is reached, so you get no output and
no error. This cost time to diagnose. Use `nix run .#fhs -- -c 'CMD'` — that is what
`packages.fhs` exists for — or an interactive session. **M6a will hit this immediately.**

**The gate runs everything under `uv run --no-sync`.** That is what makes "the gate is
offline" a fact rather than a habit, and it means a stale `.venv` fails the gate instead
of being silently repaired. Run `uv sync` after changing dependencies; the gate says so
if `.venv` is missing.

**`ReportBuilder.build()` refuses to assemble a report with any stage unaccounted for.**
If your milestone runs only `inspect`, you must call `stage_skipped()` with a reason for
the other five. This is deliberate: a stage that is simply absent is indistinguishable
from one nobody remembered, and before the verify phase a builder with no stages produced
`overall_status: complete` and exit 0 — the exact thing INV-13 exists to prevent.

**A track's `input` directory must be named for its `track_id`.** `track_id: tx-a` with
`input: raw/tx-f` used to validate, and would have attributed every word Frank said to
Alice with nothing downstream able to notice. Directory identity is now structural rather
than a docstring claim (INV-11). The directory may live anywhere in the session; only its
final component is constrained. If a session legitimately needs person-named directories,
that is a deliberate change to `TrackConfig`, not an oversight to work around.

**Build a test that proves your test can fail.** Two of M0's own tests — both named "the
atomic write survives a failure" — short-circuited before `write_atomic` was ever
entered. Both would have passed against `path.write_bytes()`. Independent review caught
it; self-review did not. When a test is the sole proof of an invariant, break the
implementation on purpose once and watch the test go red.

**Use `write_atomic` for artifacts, never for audio.** It holds the whole payload in
memory. INV-07 forbids materializing a session-length waveform, so M2's working-audio
writes need their own streamed path.

**`resolved_config()` is what "the configuration" means to a cache key.** It materializes
defaults and sorts the roster, so a session file that omits a default hashes identically
to one that states it, and reordering tracks does not invalidate caches. Build M1's
inspection cache identity on `config_hash()` plus tool versions rather than hashing raw
YAML — hashing the file directly would make an added blank line a cache miss.

**Times are `Fraction` everywhere internal.** `public_seconds()` is the only float-producing
conversion, built on an integer-millisecond quantizer with an explicit half-away-from-zero
tie rule. Python's `round()` is banker's rounding and would make 0.5 ms and 1.5 ms
disagree about which way a half goes. Do not add a second float path.

**`scripts/codex-review.sh code` had never worked.** This Codex version rejects a custom
prompt alongside `--base`; the review exited immediately with a usage error that the tee
captured and nobody read. It now drives `codex exec` the way `plan` does. If a review
returns suspiciously fast, read the raw file before believing it found nothing.

**Reviewer transcripts quote `LOCAL.md`.** They now land as gitignored `*.raw.md`; commit
the distilled findings beside them. This repository is public.

**Things that look wrong but are deliberate:** the CLI raises the builtin
`NotImplementedError` rather than a project exception, so `scan_placeholders.py` can see
it — a custom type would evade the check that exists to surface placeholder work. The
schema-drift test compares a committed file against the function that generated it, which
is circular by nature; correctness is covered separately by validating real payloads,
including the spec's own transcript example checked into `tests/data/`.

### Deviations from this charter, and why

1. **`doctor` is implemented, not stubbed.** The gate listed it among the not-implemented
   commands and then required it to work. Amended in the gate list during the start phase.
2. **Reading tool versions means executing `ffmpeg -version` and `ffprobe -version`.** The
   "no ffprobe invocation" non-goal is about probing session audio, not about reading a
   version string. Clarified in the gate list.
3. **"Gate passes end to end" was read as "with zero skips."** It already passed *with* a
   skip before this milestone started.
4. **`packages.fhs` was added beyond the charter's `devShells.fhs`.** Same sandbox, exposed
   as a runnable wrapper, because `devShells.fhs` cannot be driven from a script.
5. **The spec was amended** (one clause, ADR-0003). Flagged separately because the working
   agreement says the spec does not change casually.
6. **INV-13's wording was amended** with the same carve-out.
7. **Three enforcement rails were fixed mid-milestone**: `scan_placeholders.py` matched the
   bare string `NotImplementedError` (flagging an `except` handler and two docstrings) and
   was blind to runtime skips; `codex-review.sh` was broken for `code` mode and was teeing
   `LOCAL.md` into a public directory. All three were doing less than they claimed.

### Downstream charters updated

- **M1** — added the four M0 contracts it inherits: every stage needs a recorded outcome,
  the input directory is the track identity, the manifest schema version is provisional
  until M1 closes, and the inspection cache builds on `config_hash()`.
- **M2** — noted that `write_atomic` is for artifacts only, and that its preflight is what
  settles OQ-013.
- **M4** — noted that the transcript schema version is provisional until M4 closes.
- **M6a** — noted the `nix develop .#fhs --command` trap, with the working alternative.
- **`OPEN-QUESTIONS.md`** — OQ-013 added.
- **`AGENTS.md`** — records that reviewer transcripts are gitignored and distilled.
- **`ROADMAP.md`** — unchanged; the dependency graph did not move.

### Next smallest step

Begin M1: the synthetic fixture generator first, then discovery and `ffprobe` capture.
The generator is the thing everything after it is tested against, and the charter is
explicit that real DJI evidence should be acquired during M1 rather than wait.

Real DJI metadata has **not** been validated. Every layout assumption in M1 must sit
behind a named strategy tagged with its `OQ-` ID.
