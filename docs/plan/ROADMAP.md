# Roadmap

Milestones, dependencies, and completion gates. Charters live in
`milestones/`; live status lives in `STATE.md`.

This file is stable but not frozen. Early work will change what later milestones
should do — when it does, edit the affected charter (and this file) at close time
rather than discovering the drift six weeks later.

## Dependency graph

```text
M0 Foundation
 ├─ M1 Inspection ── M2 Timeline ── M3 Activity ─┬─ M4 Fake transcript ─┐
 │                                               └─ M5 Automix ─────────┤
 └─ M6a ROCm environment ── M6b Qwen adapter ───────────────────────────┴─ MVP
                                                                           └─ M8 Readiness ─┬─ M9 Transcript assembly ── M7a Verified raw archive ─┐
                                                                                             └─ M10 Acoustic marker ───────────────────────────────┤
                                                                                                                                                 └─ Live Session Zero
                                                                                                                                                    └─ M11 Validate/tune ── M7b Publish/reclaim
```

M8 sits between the MVP and live Session Zero. M7a was split out of the old M7 so the off-site
copy exists **before** that irreplaceable recording; it needs an inspected session, not an
accepted transcript. M10 supplies the bench-validated acoustic QA instrument. ADR-0043 retires
the dedicated short metadata capture and long clock-stability capture because the sample
probe, jam capture,
minimal acoustic capture, and six-transmitter/three-receiver marker bench already settled their
structural questions. M11 starts after Session Zero and owns only measured real-play tuning or a
genuine technical contingency. M7b then uses the accepted result for publication, retention,
cache sizing, and any deletion-policy decision.

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

### M8 — Real-session readiness

Structural fixes and diagnostics that make the first real session safe to record and
worth analyzing. M8 predates the M7 split: it is a prerequisite for safe capture, M7a is the
pre-session backup milestone, and M7b still waits on a processed real session.

Seven defects, all structural rather than threshold-shaped, so none waits on tuning data:
the bleed veto's speech reference is computed from bleed and gets worse with roster size;
`ingest` refuses 24-bit `orig` files with a reason that is false for 24-bit; the timing
model still encodes the midnight-relative semantic OQ-004 disproved; nothing prevents
`origination_time` being used as a cross-receiver anchor when it was measured 48.7 s wrong;
`sync_qa` warns below the hardware quantum and discards correct low-correlation
measurements; the mix's clamp warning names mounting when the cause is bleed; and M4's
deferred three-way collapse decision now has the real output it was waiting for. Plus two
diagnostics — per-track speech reference in the activity graph, and a count of words
silently dropped at the ownership boundary.

**Gate:** each defect has a test that would fail if it regressed; a 24-bit source converts
bit-exactly; a 11.31 ms cross-receiver offset raises no disagreement and a 120 ms one does;
dropped words are counted and named; a synthetic fixture calibrated from the measured
jam-capture acoustics (17.4 dB rejection) is checked in, with no session audio committed.

### M9 — Transcript assembly quality

**Status: closed.** The implementation and fixed-response four-file evaluation are recorded in
the M9 closeout; M11 retains the narrow empirical calibration questions for ordinary play.

Transcript-only recovery and presentation changes established by the four-file local
evaluation: 20 ms of bounded leading ownership grace after ASR, conservative collapse of a
proper contained fragment under compelling source dominance, and public joining of adjacent
same-track records without joining the authoritative records themselves. M9 depends on M8's
assembly-semantic/cache split and leaves activity and mix unchanged.

**Gate:** every ownership boundary and every unsafe collapse contrast has a deterministic
CPU/offline regression test; assembly-only changes reuse the ASR cache; records preserve both
activity and effective transcript ownership; JSON and Markdown expose the same coalesced turns
with full lineage; the ambiguous exact short `Okay` pair survives; the default gate and the
default suite from `.venv-rocm` both pass with zero skips.

### M7a — Verified private raw archive

Compress every raw source byte-exactly with zstd, upload it to private DigitalOcean Cold
Storage under a human-readable content-addressed key, read the complete object back, verify
both compressed and restored SHA-256 values, and support ergonomic whole-session/one-track
status, verification, and restore.

**Status: closed.** The explicit opt-in archive exception to INV-06, narrowed in wording from
"audio never leaves the machine" to "audio never reaches anything that processes it" across
all four places that stated it — `AGENTS.md` twice and the spec twice. It never deletes or
publishes, so INV-01 remains intact and unamended.

**Gate:** full local and remote round trips reconstruct every original hash; manifest-last
commit, retry, restore, redaction, bounded streaming, raw immutability, and offline default
tests all fail closed under corruption.

### M10 — Acoustic synchronization marker

Build one canonical 48 kHz PCM marker through the CLI, embed those exact bytes in a standalone
offline phone HTML player, and detect the complete marker automatically on every track. It
verifies the LTC jam and measures start/end differential acoustic arrival without changing the
timeline; only fixed phone-and-lav geometry isolates recorder drift.

**Status: closed 2026-08-05.** Cand-b is frozen as marker v1 after the intended-phone/six-DJI
bench; live Session Zero uses it, and claps remain the fallback.

**Gate:** CLI WAV and HTML-embedded WAV are byte-identical; offline page playback, matched-filter
detection, exact integer-sample lags, false-positive negatives, bounded streaming, unchanged
pipeline identities, and the physical phone/DJI bench all pass.

### M7b — Processed publishing and local reclamation (sketch)

After a real session is accepted, publish selected versioned outputs through a suitable
delivery bucket, choose privacy and retention policy, prune reproducible caches, and decide
whether a separately confirmed local-raw reclamation command is safe and worthwhile.

**Status: deliberately unplanned.** Raw archive format, upload, read-back integrity, and
restore belong to M7a and must not be redesigned here.

**Gate:** provisional; see the charter. Any raw deletion requires a fresh full M7a restore
verification, dry-run-first exact targeting, and an explicit INV-01 amendment.

### M11 — Session Zero validation and tuning

Process the campaign's live Session Zero at production defaults, verify and archive its
immutable sources, audit real-table activity/transcript/mix behavior, measure cache footprint,
and change only defaults supported by concrete evidence. The conservative production pipeline
is the expected outcome. Affine timing or event-first work remains a conditional response to a
demonstrated defect, not presumed scope.

**Status: blocked on the live recording.** This is the last implementation milestone in the
list and the only remaining tuning milestone.

**Gate:** the baseline completes and remains reproducible; OQ-013, OQ-017, OQ-018, OQ-019, and
OQ-027 are answered or narrowed from ordinary play; every changed default has evidence and a
regression; unsupported changes are explicitly declined; M7b receives measured sizing and the
accepted-session boundary; the full gate remains green.

## Working rules

- Each milestone is one branch merged to `main`; `main` stays runnable and green.
- Live Session Zero is external evidence, not a controlled milestone. Preserve and archive it
  before M11 tuning.
- No placeholder implementations and no unexplained skipped tests satisfy a gate.
- When real hardware disproves an assumption, update the fixture note, the affected
  `OQ-` entry, and the schemas in the same change.
- If a milestone's charter turns out to be wrong mid-flight, amend the charter and
  write an ADR before continuing. Do not silently drift.

## Changes from the original milestone list

Recorded so the reasoning is not lost:

1. **Hardware assumptions were registered separately from software.** This made every early
   DJI-metadata guess traceable while the implementation proceeded on synthetic evidence. The
   later sample probe, jam verification, minimal acoustic capture, and marker bench supplied
   the real evidence without requiring a monolithic late validation milestone.
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
   without designing it prematurely.
7. **M7 split into M7a backup and M7b publication/reclamation** once the first-session risk
   became concrete. Verified off-site raw backup is useful before Session Zero and needs only
   an inspected session; publication, cache sizing, retention, and deletion need a processed
   real session and materially different authority. M7a is therefore a narrow INV-06
   exception with no deletion, while INV-01 remains untouched until M7b can justify one.
8. **M10 gives OQ-025 an explicit software owner.** The standalone phone page embeds the CLI's
   exact WAV rather than maintaining a second JavaScript synthesizer; physical phone output was
   bench-validated before the live-session capture guide adopted it.
9. **ADR-0043 retires the two dedicated hardware captures and adds M11 last.** Existing real
   evidence already validates the hardware breadth and MVP clock decision. Live Session Zero
   now supplies ordinary-play tuning data, while only genuine technical contingencies survive
   into M11.
