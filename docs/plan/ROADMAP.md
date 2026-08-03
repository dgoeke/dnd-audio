# Roadmap

Milestones, dependencies, and completion gates. Charters live in
`milestones/`; live status lives in `STATE.md`.

This file is stable but not frozen. Early work will change what later milestones
should do — when it does, edit the affected charter (and this file) at close time
rather than discovering the drift six weeks later.

## Dependency graph

```text
M0 Foundation
 └─ M1 Inspection ──┬─ M2 Timeline ── M3 Activity ─┬─ M4 Fake transcript ─┐
                    │                              └─ M5 Automix ─────────┤
                    │                                                     │
                    └─ H1 Hardware fixture (parallel; unblocks itself)     │
                                                                          │
                              M6a ROCm env ── M6b Qwen adapter ────────────┴─ MVP
                                                            │                 │
                                                            │                 └─ M7 Archival
                                                            └─ H2 Drift soak      (sketch)
```

M5 depends only on M3, never on M4 or M6 — the mix must survive a transcription
failure. M6a can start any time after M0; it is sequenced late only because
nothing else blocks on it.

## Milestones

### M0 — Foundation

Real git repo, `pyproject.toml` + uv lock, repo-local Nix flake with a
direnv-activated default shell (plus an FHS shell held for M6a), CLI skeleton,
Pydantic config/output schemas, fake interfaces, `doctor`, and the enforcement
rails that make every later gate trustworthy.

**Gate:** `./scripts/gate.sh` passes with no GPU, no models, no network. Schema
artifacts generate and a drift test fails when a model changes without regenerating.
An attempted socket connection inside the default test suite fails the test.

### M1 — Inspection

Discovery, FFprobe capture, RIFF/RF64 chunk inventory, `orig`/`edit` selection,
`active_tracks: auto` inference, timecode strategy chain, deterministic manifest.
Synthetic fixture generator lands here.

**Gate:** Synthetic fixtures exercise selection, duplicates, roster/active-track
rules, and the timecode cases; inspecting unchanged input twice produces
byte-identical `manifest.json`; a tool-version bump invalidates the inspection cache.

### M2 — Timeline

Rational timecode arithmetic, chunk ordering, gap preservation, overlap detection,
midnight rollover, recovery overrides, streamed 48 kHz working path and 16 kHz
derivatives, sample-exact source↔working↔session mapping.

**Gate:** Exact sample-position tests for non-drop, fractional, drop-frame,
rollover, and override cases; aligned duration matches the latest source end within
one 48 kHz sample; a 44.1 kHz or internally inconsistent track fails before timeline
construction.

### M3 — Activity

Silero VAD behind an `ActivityDetector` interface plus a deterministic fake,
lag-tolerant normalized cross-correlation, conservative pre-ASR bleed gate, and the
versioned activity/attribution graph.

**Gate:** Solo, genuine-overlap, quiet-bleed, and delayed-correlation cases behave
as specified; peak correlation and its lag are recorded; ambiguous candidates are
kept. **The activity graph schema is checked in and frozen here** — M4 and M5 both
consume it, and text-dependent decisions must never flow back into it.

### M4 — Fake transcript

Fake ASR, segment request construction with padding and core intervals, normalized
transcript records, post-ASR duplicate collapse, alignment fallback,
`transcript.json` + `transcript.md` rendering.

**Gate:** End-to-end transcript from synthetic input with no Qwen; output validates
against the checked-in JSON Schema; Markdown order and IDs are byte-stable on
rerun; distinct overlapping utterances and matching short utterances both survive;
padded requests never exceed `max_segment_s`; a faked length-truncated response
triggers bounded split/retry and deterministic stitching.

### M5 — Automix

Per-track voice-level correction, smoothed gain envelopes from the activity graph,
Dugan-style bounded gain sharing, streamed mono mix, two-pass loudness, MP3 encode
+ decode verification with bounded re-encode retries.

**Gate:** Envelope-level assertions (solo dominance after attack, both channels
audible during overlap, bounded gain invariant, no slew violations) plus decoded
MP3 duration and true peak within configured tolerances, and integrated loudness
within tolerance **on a run that aimed at the target** — where the true-peak
ceiling, the master-gain clamp, or the silence floor forbade aiming at it, the run
carries the warning naming that guard instead (ADR-0023, and the spec's acceptance
criterion 8 as amended in M5). A simulated ASR failure still yields MP3 + report
with `process` exiting nonzero.

`process` lands here too, not in a milestone of its own: it is the dependency-aware
orchestration of both branches, and it cannot exist before both branches do.

### M6a — ROCm environment

AMD `gfx1151` Torch wheel index wired into uv with per-package sourcing, FHS build
toolchain for the `rocm[libraries]` sdist, locked versions, `doctor` device checks
(`/dev/kfd` and render-node openability tested by opening them, `torch.cuda`,
`torch.version.hip`, device name, BF16 op), and device/dtype resolution rules.

Two things the original entry did not anticipate, both recorded in ADR-0025: the AMD-only
packages torch depends on must be **direct** members of the dependency group, because
`[tool.uv.sources]` silently ignores a transitive-only requirement; and the group installs
into a **separate** `.venv-rocm`, so the everyday gate keeps running the group-absent case
INV-05 describes instead of proving it once.

**Gate:** `doctor` reports a healthy GPU on the target host; the marked host smoke
test runs a BF16 op on gfx1151; the lock file demonstrably contains AMD Torch and
not a CUDA build from PyPI.

### M6b — Qwen adapter

`models fetch` with snapshot-revision locking, offline execution, Qwen3-ASR
Transformers adapter, forced aligner, token-limit truncation handling, full cache
identity, and report provenance.

Three things the original entry did not anticipate, all recorded in ADR-0027, ADR-0028 and
M6b's closeout. The ~6 GB download is a **one-time setup step** driven by the `hf` CLI —
but behind `models fetch --qwen` rather than beside it, so `models fetch` remains the only
network authority and neither INV-06 nor the spec needed amending. Snapshot installation is
keyed by `(repository, resolved commit)` and the installed tree is verified as an **exact
allowlist in both directions**, because Transformers loads a directory and an unpinned file
in one is a file a model may read. And `asr.model`/`asr.aligner` accept exactly one value
each: this build carries snapshots for two repositories and no command can install a third,
so a configured third would have run Qwen and recorded something else as having produced
the transcript.

**Gate:** A short real transcription + alignment smoke test passes on the target
host; changing `max_new_tokens` invalidates the cache; the default (non-`host_smoke`)
suite still passes without any of it installed.

### M7 — Archival and local disk reclamation (sketch)

Compress `raw/` (WavPack or zstd) with self-describing sidecar metadata, upload to
DigitalOcean cold storage, publish processed outputs to Spaces for wiki use, verify
remote hashes, and then — manually, separately — reclaim local disk.

**Status: deliberately unplanned.** The charter exists so the work is not
forgotten and so its tension with INV-01 and INV-06 is recorded while it is cheap
to think about. Plan it properly when the MVP has run on a real session.

**Gate:** provisional; see the charter.

### H1 — Hardware fixture (parallel track)

A ~2-minute real six-transmitter, three-receiver recording per the spec's recipe,
used to validate DJI naming, metadata layout, and timecode assumptions. Blocked on
physical recording, not on code.

**Gate:** Every `OQ-` question tagged `needs: H1` is answered or explicitly
re-scoped; the fixture note is written; sanitized `ffprobe` JSON + RIFF inventory
are committed; assumptions the fixture disproves are fixed in M1/M2 with tests.

### H2 — Drift soak / first session

A ~4-hour soak fixture with synchronized transients near both ends, or the first
real session's start/end clap measurements, used as evidence for the
no-drift-correction assumption.

**Gate:** Measured differential lag and the configured warning threshold are
recorded; the drift warning fires on synthetic drift without applying correction.

## Working rules

- Each milestone is one branch merged to `main`; `main` stays runnable and green.
- Start acquiring the H1 fixture during M1. Do not postpone metadata discovery.
- H2 is a validation gate, not a blocker for the first vertical slice.
- No placeholder implementations and no unexplained skipped tests satisfy a gate.
- When real hardware disproves an assumption, update the fixture note, the affected
  `OQ-` entry, and the schemas in the same change.
- If a milestone's charter turns out to be wrong mid-flight, amend the charter and
  write an ADR before continuing. Do not silently drift.

## Changes from the original milestone list

Recorded so the reasoning is not lost:

1. **Hardware validation became a parallel track (H1/H2), not milestone 7.** It is
   gated on a physical recording session, so it should not sit behind five
   software milestones. Every DJI-metadata guess made in M1/M2 gets an `OQ-`
   marker; H1's arrival triggers a targeted revisit instead of a large late
   milestone.
2. **The old M6 split into M6a (environment) and M6b (adapter).** The AMD Torch
   index, the `rocm[libraries]` build, and device permissions fail in completely
   different ways than adapter code, and M6a can be verified on its own.
3. **M0 absorbed the enforcement rails** (gate script, network-blocked test
   suite, determinism helpers, schema-drift test, placeholder scan). These are
   painful to retrofit and every later gate's credibility depends on them.
4. **The activity graph contract is explicitly frozen at M3's gate**, because M4
   and M5 both consume it and the spec forbids text-dependent decisions from
   changing the mix.
5. **Report contributions are part of every milestone's gate**, not M4's job.
   Otherwise the report gets bolted on at the end and the provenance is guesswork.
6. **M7 archival was added as a sketch**, at the owner's request, to hold the idea
   without designing it now. Its one time-sensitive contribution is recording that
   archival conflicts with INV-01 and INV-06 by design — both invariants now point
   at it, so a future implementor neither violates them silently nor assumes the
   work is forbidden.
