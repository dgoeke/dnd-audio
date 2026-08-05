# M10 — Acoustic synchronization marker

**Status:** not started
**Depends on:** M2 and M8 (closed), plus a short operator bench recording on the intended
phone/browser and DJI hardware
**Spec sections:** Milestone 2 synchronization QA; Tests and acceptance criteria item 15;
recommended real fixture and owner notes — amend before implementation

## Goal

Replace hand-picked claps as the preferred **acoustic verification** signal with a distinctive,
deterministic marker that can be generated through the normal CLI, played from a single offline
HTML file on a phone, and detected automatically at integer-sample positions on every track.
Use it to expose a failed LTC jam and measure start-to-end differential acoustic arrival. That
change is recorder-drift evidence only when **both endpoints of every acoustic path**—the phone
and each transmitter/lav—remained fixed. It never overrides valid timecode, never places a file
that did not record the marker, and never becomes a hidden timeline correction.

The CLI WAV and the phone page must not be two approximately equivalent synthesizers. The CLI
builds one canonical PCM WAV and embeds those exact bytes in the standalone HTML. JavaScript
plays the embedded asset; it does not recreate the chirp with `Math.sin`, an oscillator, or a
second floating-point implementation. Extracting the page's embedded WAV must produce the same
byte length and SHA-256 as the CLI WAV.

That is the strongest equivalence software can promise. A browser may resample 48 kHz PCM to
the phone's hardware rate, and the phone speaker, media volume, dynamics processing, table, and
room alter the acoustic waveform. M10 must measure those effects on the intended device and
make the detector robust to them; it must never claim that equal source bytes imply equal sound
pressure at six lavs.

## Operator-facing contract

### Build once

```text
dnd-audio marker build OUTPUT_DIRECTORY
```

The command has no session input and writes deterministic, versioned artifacts:

```text
dnd-audio-sync-marker-v1.wav
dnd-audio-sync-marker-v1.html
marker-manifest.json
```

- The WAV is mono, 48 kHz, integer PCM with conservative headroom and a frozen marker semantic
  version. Rebuilding under the same code/version produces identical bytes.
- The single HTML file contains all CSS, JavaScript, instructions, and the exact WAV bytes. It
  performs no fetch, uses no CDN/font/analytics/service worker, works after transfer to a phone,
  and offers the embedded WAV as a download as well as a playback source.
- The manifest records the WAV and HTML filenames, sizes, SHA-256 values, PCM format, marker
  semantics, expected duration, chirp/gap sample intervals, and generator implementation
  version. Like `ingest-report.json`, it does not hash itself. The builder writes candidate
  files atomically, validates both, then publishes the manifest atomically **last** as the
  completeness marker. It contains no host, path, or wall-clock telemetry.
- A schema and drift test make checked-in semantics, generated artifacts, and documentation
  disagree loudly.

### Play deliberately

The HTML page provides one large `Play marker` button, a visual countdown, an occurrence
counter, visible marker ID/SHA-256, and a clear playing/finished state. Mobile autoplay is never
assumed; every playback begins from a user gesture. The page prevents overlapping playback and
keeps all instructions usable without network access. The operator keeps the screen awake;
Wake Lock is intentionally outside M10.

The operator keeps the phone out of Bluetooth, at the bench-tested media-volume step, in the
same orientation and fixed central table position for the primary start and end markers.
Different positions are allowed as separately logged diagnostic occurrences: moving either the
source **or a lav** changes acoustic propagation by milliseconds. A fixed phone with worn,
moving lavs yields useful `differential_arrival_change_samples`, but not an unqualified clock-
drift measurement. Only a fixed-transmitter soak, with source and receiver geometry unchanged,
may classify that change as drift.

### Analyze after ingest

```text
dnd-audio marker analyze SESSION_DIRECTORY \
  --reference-track tx-a \
  --start-window-s 120 \
  --end-window-s 120 \
  --event-log marker-events.json
```

`analyze` consumes the existing 48 kHz virtual tracks and timeline after `ingest`. It first
validates the current sources, manifest identity, timeline identity, and marker-analysis
identity without rewriting any existing pipeline artifact; a merely present or stale
`timeline.json` is not accepted. It streams bounded windows, finds the complete marker sequence
by matched filtering, and writes a
deterministic `work/sync-marker-analysis.json` plus a separate per-run
`output/marker-report.json`. It never rewrites `ingest-report.json`: marker QA has its own
versioned command/report boundary so existing report schemas and processing provenance do not
move. The analysis records:

- marker and detector semantic versions plus canonical reference SHA-256;
- every detected occurrence within the searched half-open intervals, not only the chosen
  start/end pair;
- per track: integer session anchor sample, mapped `(source_relative_path, source_sample)` when
  the anchor falls in a real source segment, normalized score, local peak ambiguity,
  clipping/weak-signal diagnostics, and detected chirp order/gaps;
- selected reference track and deterministic selection/tie-break behavior;
- reference-anchored occurrence groups, per-track matched/unmatched detections, and per-track
  relative lag for the independently labelled start/end occurrences;
- start-to-end differential-arrival change for matched pairs, plus a `clock_drift_evidence`
  classification only when the event log asserts one unchanged geometry ID for the phone and
  every compared transmitter/lav;
- comparison against the metadata-predicted alignment, with constant disagreement, measured
  drift, weak evidence, missing marker, and ambiguous duplicate occurrences kept distinct.

The optional event log is a separate versioned marker-analysis input, not part of
`session.yaml`. It supplies independently observed approximate half-open search intervals,
event roles (`start`, `end`, or `diagnostic`), playback order, and an acoustic-geometry ID. If
it is absent, the default start and end windows must each contain exactly one accepted
reference occurrence before they can be paired; repeated or moved-position events stay
enumerated but inconclusive. Overlapping configured intervals are canonicalized into a disjoint
searched set without detecting an occurrence twice.

`marker-report.json` follows INV-13 at its own command boundary: it has `overall_status`, one
`complete`/`failed`/`skipped` marker-analysis status, structured errors and warnings, and hashes
of every successful deliverable other than itself. Missing, weak, or ambiguous marker evidence
is a completed but inconclusive measurement. Invalid inputs, stale timeline identity, corrupt
sources, or unsafe output paths are failures; partial success exits nonzero. The report is
written atomically even on an ordinary analysis failure, except when its own resolved path
would violate INV-01.

No result mutates `ingest-report.json`, `timeline.json`, source mappings, activity, mix,
transcript, or any existing cache. Changing marker/detector semantics invalidates marker
analysis only.

## Canonical signal design

The provisional candidate is three band-limited linear chirps spanning approximately
500 Hz–8 kHz, with a smooth integer-sample amplitude envelope, conservative peak level,
leading/trailing silence, and asymmetric inter-chirp gaps. The asymmetry makes the whole
sequence far less likely than one speech/music sweep and allows the detector to reject reversed,
truncated, or partly obscured patterns.

The exact frequencies, chirp durations, directions, gaps, sample format, and peak level are
**not frozen by this planning document**. M10 first generates a small candidate matrix, then
chooses one v1 waveform from objective bench evidence:

- correlation peak sharpness and ambiguity on every DJI track;
- tolerance to phone/browser resampling, lav band limiting, room reverberation, gain changes,
  moderate clipping, and background speech/music;
- audibility and operator comfort at a level high enough for the farthest lav;
- absence of clipping at the nearest lav;
- reliable distinction from normal table audio;
- stable detection on the intended phone/browser when played repeatedly.

Candidate tooling remains private and uses candidate names/hashes; nothing is called public
`v1` before the physical bench selects it. The resulting ADR freezes the complete integer PCM
sample sequence by SHA-256, its marker anchor, thresholds, and human-readable recipe. Public
generation must be platform-stable—an exact integer/fixed-point construction, not output that
depends on a particular platform's `libm` result. Future marker changes receive a new semantic/
versioned filename; they do not silently replace v1.

## Detector semantics and limits

- Match against PCM returned by the same canonical marker function that builds the WAV; never
  write a second detector-side synthesis formula. The frozen anchor is an exact sample relative
  to WAV start, intervals are half-open, `relative_lag_samples = track_anchor -
  reference_anchor`, and equal scores choose the lower session sample and then lexical track/
  source identity. Search at 48 kHz and keep peak locations as integer samples. Sub-sample
  interpolation is unnecessary while acoustic geometry contributes roughly 1.5–9 ms across a
  table.
- Detect the three-chirp timing/code as a sequence. A strong isolated chirp is insufficient.
  Thresholds cover normalized peak score, runner-up separation, gap tolerance, and required
  chirp count; all are versioned and recorded.
- Search only bounded configured start/end windows by default. Use fixed-size overlap-save
  blocks with template-length carry, online normalization, and online candidate suppression;
  maximum working memory is independent of session and requested-search-range length. A
  four-hour whole-session scan is explicit rather than accidental.
- Form accepted occurrences first on the reference track using a versioned non-maximum-
  suppression radius. Associate other tracks one-to-one within a bounded lag interval around
  each reference anchor; never pair by list index. Missing detections remain unmatched, and
  ambiguity compares local alternatives only after other accepted reference occurrences are
  excluded. Event roles come from the independent event log or the one-event-per-default-window
  rule, never from peak strength.
- Multiple receivers hearing the same marker at different times measure acoustic arrival plus
  recording alignment. Report the geometry term; do not present a central phone as an
  electrical timecode source.
- A constant marker/timecode disagreement is jam QA, but its warning threshold retains M8's
  measured timecode-quantization floor; higher matched-filter precision does not turn a healthy
  within-one-quantum offset into a failed jam. A start-to-end change is always reported as
  differential acoustic arrival and becomes a drift warning only under fixed source **and lav**
  geometry. Neither automatically corrects placement.
- Missing, clipped, weak, or multiply plausible evidence stays inconclusive. The analyzer never
  fabricates a lag because one is required by the report.
- A restarted transmitter file that missed the marker remains placeable only through embedded
  timecode or an explicit recovery override. M10 cannot replace that property of the LTC jam.

## Standalone HTML equivalence and safety

The page is a transport/player for canonical bytes, not a second signal generator:

1. Python creates the canonical WAV once in memory/streamed output.
2. The same byte sequence is written as `.wav` and encoded into the HTML.
3. Tests extract and decode the HTML payload and require byte-for-byte equality and the same
   SHA-256 as the WAV and manifest.
4. JavaScript builds a `Blob`/media source from those bytes and plays it at unity page gain.
   It may display the browser audio-context rate, but it cannot promise or force the phone's
   hardware sample rate or media volume.
5. No Web Audio oscillator, browser `Math.sin`, lossy codec, remote asset, or base64 copy kept
   separately in source code may become a second truth.

The default gate statically proves that the generated page has no external URL or network API,
contains the canonical payload once, and exposes one non-overlapping playback state machine. An
opt-in browser smoke loads the page from a local file with network denied, activates playback by
user-equivalent gesture, observes exactly one ended event, and downloads/extracts the canonical
WAV. The physical bench remains necessary because browser automation cannot test a phone
speaker or room.

## Bench protocol

No second person is required. Place the six transmitters around the intended table geometry,
start all recordings, and use the intended phone/browser:

1. Play each named candidate three times from the fixed central position at the candidate
   media-volume step, leaving several seconds between occurrences.
2. Repeat near the end of a 10–15 minute take without moving or rotating the phone.
3. In a separate diagnostic block, deliberately move the phone to two logged positions and
   replay once at each; prove the analyzer enumerates them but does not call their differing
   lags drift.
4. ~~Include room tone, speech, and ordinary media without the marker to measure false
   positives.~~ **Amended 2026-08-05 (A4): measured off-bench, already answered.** The detector
   was run over every real DJI recording this project holds — 13.7 minutes across two captures,
   two voices overlapping deliberately, hand claps at both ends of one — and accepted zero
   sequences for all three candidates, strongest single chirp 186 permille against a 550
   threshold. More audio and more adversarial material than this step would have produced, at no
   cost in the room, and dropping it is what makes a one-person bench feasible. See
   `../../fixtures/2026-08-05-marker-false-positive-sweep.md`.
5. ~~Keep an independent event log naming candidate, role, approximate time, and geometry ID.~~
   **Amended 2026-08-05 (A5): the bench records no event log.** What it would supply arrives
   instead as fixed block order plus spoken slates on the take itself — see the working plan.
   Run `inspect`, `ingest`, and `marker analyze`; hash raw before and after.
6. Confirm every track detects the fixed-position events, nearest tracks do not clip, farthest
   tracks remain decisive, and repeated same-position lag is inside the documented tolerance.
   The false-positive half of this confirmation is discharged by step 4's replacement.

Only after this evidence selects one candidate does the ADR name and freeze marker v1; the
public builder/analyzer and their golden proofs follow. If any intended track fails, retain the
three-clap H1/H2 procedure. Do not improvise a new waveform during Session Zero.

## Explicitly not in this milestone

- Replacing the LTC jam, automatically correcting the timeline, affine resampling, or claiming
  phase-coherent synchronization.
- Using different playback positions as independent clock anchors or averaging away propagation
  delay without measured geometry.
- A hosted web application, service worker, installable PWA, network transfer, telemetry, or
  remote browser dependency. The deliverable is one offline HTML file.
- Recording through the phone microphone, synchronizing the phone clock, ultrasonic signaling,
  Bluetooth playback, or electrical injection into receivers.
- General-purpose acoustic measurement, room impulse response estimation, speaker calibration,
  or source localization.
- Wake Lock, a public candidate-management CLI, and automatic whole-session scanning by
  default.
- Feeding marker detections into activity, automix, ASR, or transcript semantics.
- Checking generated audio binaries into the repository. The CLI creates WAV/HTML artifacts;
  source, schemas, golden hashes, and tests are tracked.

## Working plan

_Scratch section, written during the start phase and replaced by the Closeout. It
supersedes the eight-step plan this charter carried before 2026-08-05; that version is in
the commit history._

### Charter amendments, and why

Three, all approved by the operator on 2026-08-05 before any code was written.

**A1 — the bench is scored through the shipped commands, not through private tooling.**
The old steps 3–6 built a *private* candidate generator, player and detector for the bench,
then a *separate* public `marker build`/`marker analyze` afterwards from the frozen v1. That
is two synthesizers and two detectors — one level up, it is exactly the failure the charter's
own "the CLI WAV and the phone page must not be two approximately equivalent synthesizers"
rule exists to prevent. It also means the public builder's first exercise would come *after*
v1 was already frozen, which is verbatim the defect M7a's closeout records: nine test files
and complete runner coverage did not compensate for the fact that nothing ran a command.

So there is one generator, one page builder and one detector, and the bench drives them
through `dnd-audio marker build` and `dnd-audio marker analyze`.

**This reverses an accepted finding, and that is said out loud rather than left to be
noticed.** Finding 5 of `../reviews/M10-plan-20260804-1735.md` was *"marker-v1 selection was
circular — the public v1 builder could not be proven before the bench that selects v1"*, and
the charter's private-tooling order was the accepted remedy. The circularity is real. What
A2 changes is that it is no longer the *builder* that is circular: the builder is
parameterized by a spec, not written against v1, so it is fully provable before the bench —
against candidates — and v1 enters later as **data**, with its golden hash frozen at that
point. The reviewer's actual requirement, that nothing be called public `v1` before the bench
selects it, is now enforced by an absent registry key rather than by a duplicate toolchain.
If that reasoning is wrong, this is the paragraph to attack.

**A2 — the marker is a named candidate until the bench names v1.** `MARKER_SPECS` is a frozen
registry of candidates; `marker build OUTDIR --marker cand-a` writes that candidate's WAV,
page and manifest under its own name. **There is no `v1` key**, so `marker build OUTDIR` with
no `--marker` exits nonzero naming the bench. The charter's "nothing is called public `v1`
before the physical bench selects it" is therefore enforced by the registry rather than by
convention. Phase B adds the `v1` entry — a copy of the winning candidate's spec — and the
ADR freezes its exact integer PCM by SHA-256.

`--marker` is a **hidden** option (`hidden=True`), documented in the bench protocol rather
than in `--help`. The plan review was right that this charter's non-goals exclude "a public
candidate-management CLI" and that A2 was advertising one; it was wrong that the remedy is a
private bench script, which would re-open the exact gap A1 closes — the CLI wiring *is* where
M7a's P0 lived. Hiding the option honours the non-goal on the surface that matters while the
bench still drives the real command through its real guards. After the freeze the
discoverable interface is `marker build OUTPUT_DIRECTORY` and nothing else.

**A3 — the opt-in local-browser smoke is deferred, not dropped.** Nothing in the flake
provides a browser, and adding one (a `pkgs.chromium` closure, or a third environment
carrying Playwright plus a network-fetched browser binary) is a larger change to the
environment than this milestone is otherwise making. What the default gate keeps is the
*static* half, strengthened so it is a proof rather than a grep: the page's playback state
machine is a declarative JSON transition table the page's own JavaScript reads, and the test
parses that same table and asserts no transition starts a second playback while one is
running. The dynamic half is subsumed by the physical bench, which loads the real file on the
real phone and is strictly stronger evidence than headless Chromium. Recorded here so it is a
decision rather than an omission.

This narrows the prior review's *"the browser test owns only offline byte transport/playback
state; the phone/DJI bench owns physical fidelity"* (`../reviews/M10-plan-20260804-1735.md`,
Scope restraint). Byte transport keeps a mechanical proof; **playback state** keeps a
declarative one; only *executed* playback moves to the bench.

**Completion-gate criteria 3 and 12 are amended in the gate itself**, not merely here. The
second plan review was right that a scratch section cannot override the charter's contract,
and right that a parsed transition table proves the table's graph rather than that the page's
JavaScript applies it, that `ended` resets the UI, or that the embedded asset reaches
playback. Those three claims now belong to the physical bench, and the gate says so rather
than implying a software proof that will not exist.

### Phase A — every part that does not need hardware

1. **Documents first.** ADR-0040 separates the four things this milestone must never
   conflate: LTC timecode *placement*, acoustic *verification* of the jam, *differential
   acoustic arrival*, and *fixed-endpoint recorder-drift evidence*.

   **The spec is currently wrong about the fourth, and naming the marker alongside the clap
   would not have fixed it.** Both the Milestone-2 synchronization-QA paragraph and
   acceptance criterion 15 state unconditionally that a changing acoustic lag *"is evidence
   of sample-clock drift"*. That is false when the phone or a lav moved — which is finding 3
   of `../reviews/M10-plan-20260804-1735.md`, accepted into this charter and never propagated
   into the spec. Both passages gain the fixed source-**and-lav** geometry condition before a
   drift warning may be emitted, amended in the same commit as ADR-0040, so spec,
   implementation and gate stop disagreeing.

   ADR-0041 records the marker artifact set, the candidate-before-v1 registry (A1/A2), and —
   **frozen before production code, because the second review found each of them undefined** —
   the exact anchor and lag signs; the integer **permille** score domain and its single
   rounding rule, in which every threshold, runner-up separation and equal-score tie is
   compared; occurrence grouping and the versioned occurrence ceiling; the event log's
   serialized units and half-open boundary behaviour; one-to-one event↔occurrence assignment
   with ambiguity resolved as inconclusive; the outcome vocabulary; commit points; and the
   full analysis identity. Update OQ-025 and raise **OQ-029** (below).

2. **Exact, platform-stable synthesis** — `src/dnd_audio/marker/{spec,sine,synth}.py`.
   The charter requires "an exact integer/fixed-point construction, not output that depends
   on a particular platform's `libm` result", and a chirp's phase has a closed form in
   integers. For a linear chirp from `f0` to `f1` over `N` samples at rate `R`, the phase in
   turns at sample `n` is

   ```
   phase(n) = [ n·f0·(N−1) + (f1−f0)·n(n−1)/2 ] / ( R·(N−1) )
   ```

   — every term an integer for integer `f0`/`f1`, and a closed form rather than an
   accumulation, so no rounding compounds. The table index and its interpolation remainder
   come from one integer division by `R·(N−1)`.

   Sine comes from a checked-in quarter-wave integer table,
   `src/dnd_audio/marker/data/sine_table.json`, regenerated by `scripts/design_sine_table.py`
   — deliberately the pattern M2 established for the decimation FIR (`data/fir_48k_16k.json`
   + `scripts/design_fir.py`): the coefficients are *data*, so a NumPy or libm upgrade cannot
   silently change what a frozen SHA-256 describes, and the **tests are the contract, not the
   array** (endpoints, quarter-wave monotonicity, the symmetry identities the evaluator uses,
   and a stated maximum error against `math.sin`). The amplitude envelope is a raised cosine
   from the same table over an integer number of samples. One rounding rule, stated once, at
   the single point where fixed point becomes an integer sample.

3. **The candidate matrix** — three specs, chosen to span the questions the bench exists to
   answer rather than to be three flavours of the same guess:

   | name | band | chirp | gaps | asks |
   | --- | --- | --- | --- | --- |
   | `cand-a` | 500 Hz → 8 kHz | 3 × 180 ms up | asymmetric | the charter's provisional candidate |
   | `cand-b` | 800 Hz → 6 kHz | 3 × 250 ms up | asymmetric, wider | more processing gain in the band a phone speaker actually radiates — for the farthest lav |
   | `cand-c` | 400 Hz → 10 kHz | 3 × 120 ms, up/down/up | asymmetric, tight | whether direction asymmetry buys rejection, and whether a short chirp survives the room |

   Mono, 48 kHz, `pcm_s16le`, conservative peak with headroom, leading and trailing silence.
   The frozen **anchor is the first sample of the first chirp** (`lead_silence_samples`);
   every interval is half-open; `relative_lag_samples = track_anchor − reference_anchor`.

4. **`marker build`** — `marker/{wav,page,manifest,builder}.py` plus the CLI subcommand.

   **It needs its own INV-01 guard, and the first draft of this plan did not give it one.**
   The command takes an arbitrary destination and has no session argument, so
   `dnd-audio marker build SESSION/raw/tx-a` — or an output symlink resolving there — would
   write three files beneath a source root. That is verbatim the P0 M7a's second code review
   found, where a guard conditioned on *having a session directory* was defeated by the one
   command that never has one. `cli.py::{_sessions_above,_reject_report_inside_raw}`
   generalize into one helper refusing any resolved path under any enclosing session's
   configured source roots; `marker build` calls it **before** creating a directory, writing
   a candidate, or unlinking anything. The regression drives the real command against a real
   source filename and asserts that file's bytes are unchanged.

   `marker/wav.py` assembles a minimal `fmt `+`data` RIFF itself rather than reusing
   `fixtures/wav.py`: that module is fixture support and says so, and having the marker's
   frozen byte layout stated where it is hashed is worth twenty lines. The container is then
   verified by `inspection.riff.read_inventory` and `timeline.pcm.open_pcm` — two independent
   parsers this project already owns, which is stronger evidence than sharing a helper with
   the writer would be.

   The page carries the payload as **one** base64 string; JavaScript builds a `Blob` from it
   for both `<audio>` playback and the download link, so there is no second copy. Isolation is
   an **allowlist plus a policy, not a denylist**: the page carries a
   `Content-Security-Policy` meta with `default-src 'none'`, and the test parses the document
   and asserts every URL-bearing attribute is either absent or a `blob:`/`data:` the page
   generated itself. Enumerating known network APIs was the first draft's approach and it
   already missed CSS `url(...)`, `navigator.sendBeacon`, form actions and media/iframe
   attributes.

   **Publication order: unlink the manifest first, then write and validate the WAV and the
   HTML, then publish the manifest.** "Manifest last" alone is not a completeness marker on a
   *rebuild* — a crash between replacing the WAV and the HTML leaves the previous manifest
   describing bytes that are no longer there. Removing it first makes every interrupted state
   manifest-less and therefore detectable. Interruption tests at each publish boundary. The
   manifest is a pydantic model with a checked-in schema and a drift test, and does not hash
   itself (ADR-0003).

5. **The detector** — `marker/detect.py`. **Per-chirp matched filters, then sequence
   assembly**, rather than one correlation against the whole template. Each filter's template
   is a slice produced by the same synthesis function that built the WAV — never a second
   formula. A sequence is accepted only when all chirps are present, in order, with
   inter-chirp gaps inside tolerance; a strong isolated chirp is not a detection. This choice
   is also the hedge against the one bench outcome that would otherwise cost real code: if
   the phone's dynamics processing warps playback, a whole-template correlation stops
   compressing while per-chirp filters plus a gap tolerance degrade gracefully.

   Fixed-size overlap-save blocks with template-length carry, online normalization over a
   bounded ring, and online non-maximum suppression, so the *correlation* working set is
   independent of session length and of requested search length.

   **That is not the whole bound, and the first draft claimed it was.** NMS bounds *nearby*
   candidates; it does nothing about the number of *separated* accepted occurrences, and the
   analyzer retains every one of them, plus groups and unmatched detections, in Python lists
   that are then serialized. Over a long or adversarial range those grow with the search
   range — an INV-07 breach on the composed path, and one the proposed proof would have
   passed straight through, because a tenfold longer *sparse* search keeps the arrays equal
   while the lists grow. So: a **versioned occurrence ceiling that fails explicitly**, names
   itself in the report, and never truncates — this project does not do silent caps — and a
   memory regression built on **dense** accepted candidates rather than sparse ones. Disk
   spooling was the reviewer's other option and is machinery for a command whose default is
   two 120-second windows.

   Every threshold here is empirical and cites its open question until the bench resolves it:
   the bounded cross-track association lag, the clipping and weak-signal thresholds, the
   score and ambiguity thresholds, the NMS radius against reverberation and repeats, the
   "material" differential-change threshold, and candidate bandwidth survival through the
   phone/lav/DJI path — **OQ-025** or **OQ-029** while provisional, ADR-0042 and the measured
   evidence once frozen. The first draft promised a citation for the gap tolerance alone.

6. **`marker analyze`** — `marker/{eventlog,analysis,report,runner}.py` plus the CLI
   subcommand. Order, and it is load-bearing:

   1. Resolve both prospective outputs and `reject_outputs_inside_raw` **before anything
      else**, including the report's own path (`cli.py::_reject_report_inside_raw` is the
      pattern; M7a's second review found the P0 that exists when this is conditioned on
      having a session directory).
   2. `snapshot` the raw roots once, around the whole run.
   3. **Validate the existing artifacts read-only; never rebuild, never rewrite, and never
      run inspection.** The first draft said "re-run inspection **in memory** … publishing
      nothing", and that is simply false: on a cold or missing sidecar,
      `inspection/runner.py::_inspect_one` writes `work/ffprobe/…`. "Warm from M1's content
      cache" is an assumption about the machine, not a contract — the same shape as the six
      tests M6b found asserting a property of the machine rather than of the code.

      The validator is therefore built from four things this command already has or can read
      without writing: the source snapshot taken at step 2, the contents of
      `work/manifest.json`, the currently resolved configuration, and the loaded timeline's
      own provenance. It compares **every** identity component, not just the manifest digest
      — `Timeline.config_hash` against the current `config_hash`, `manifest_sha256` against
      the hash of the manifest on disk, and each of `TimelineProvenance`'s
      `timeline_semantics_version`, `inspection_semantics_version`, `numpy_version` and
      `scipy_version` against the current constants, plus schema version and sample rate.
      Comparing the manifest digest alone would accept a timeline built by obsolete logic
      that is still internally consistent with the same manifest.

      Distinct codes per component so a diagnostic says which thing is stale, each with its
      own test. A further test asserts the entire pre-existing `work/`, `output/` and cache
      trees are byte-identical after a run. This is the one place M10 departs from ADR-0015's
      "rebuild, do not validate": rebuilding would rewrite `timeline.json` and
      `ingest-report.json`, which this charter forbids outright, and comparing every identity
      component is what keeps the departure from being a weakening.
   4. Canonicalize the searched intervals — from `--event-log` if given, otherwise the
      default start and end windows — into a disjoint half-open set, so overlapping
      configured windows cannot detect one occurrence twice. Event intervals are serialized
      as **integer milliseconds** and converted through `determinism.to_samples`, the one
      quantizer: taking approximate seconds and separately rounding each end would be a
      second quantization path, which INV-04 forbids. Each interval is clamped to the
      session, and carries a matching halo of the marker's own length so an occurrence whose
      anchor sits near an edge is still found rather than half-scanned.
   5. Stream `TrackReader` windows over that set at 48 kHz, per track, through the detector.
   6. Form accepted occurrences on the **reference track** first, under a versioned NMS
      radius; associate other tracks one-to-one inside a bounded lag interval around each
      reference anchor. Never by list index. Equal scores choose the lower session sample,
      then lexical track/source identity — compared in the integer permille domain, so no
      threshold or tie is ever decided by a float comparison. Assigning *roles* is a second
      one-to-one problem the first draft missed: a single reference occurrence can fall
      inside both a `start` and an `end` event interval once those intervals are unioned for
      scanning, and canonicalizing the searched set prevents detecting it twice but not
      labelling it arbitrarily. Assignment is one-to-one against logged events, tie-broken by
      playback order, and **ambiguity is reported as inconclusive rather than resolved into a
      role**. Boundary tests one sample either side of every edge, plus an overlapping
      start/end-event case.
   7. Map each anchor to `(source_relative_path, source_sample)` where it falls in a real
      audio segment, and to nothing where it falls in silence.
   8. Compare against the metadata-predicted alignment, **reusing
      `timeline.syncqa.offset_floor_samples`** so M8's measured quantization floor is the
      threshold here too — higher matched-filter precision must not turn a healthy
      within-one-quantum offset into a failed jam.
   9. Classify a start-to-end change as `clock_drift_evidence` **only** when the event log
      asserts one unchanged geometry ID for the phone and every compared transmitter;
      otherwise it is `differential_arrival_change_samples` and says so.
   10. `verify_unchanged` after the last read and before publication; write
       `work/sync-marker-analysis.json` (deterministic, INV-02) then `output/marker-report.json`
       (per-run, INV-13); cleanup on failure runs **after** the INV-01 carve-out (ADR-0021).

   The event log is a separate versioned model loaded from a path — **never a `SessionConfig`
   field**, for the reason `archive/config.py` records: a new section changes `config_hash`,
   every stage projection and therefore gigabytes of cached inference (ADR-0016).

   **No new cache.** The analysis records a complete identity document in the
   `derivative_identity_document` shape, separate from its hash so a test can assert *which*
   components are present. Adding a cache would be machinery for a command run twice a
   session.

   Naming only the marker and detector versions — the first draft's list — would let a change
   to event assignment, grouping, start/end selection, geometry classification or
   source-coordinate mapping move the analysis without moving its claimed identity. So the
   document carries a distinct **`marker_analysis_semantics_version`** alongside the marker
   and detector ones, the **event-log schema version and a canonical digest of the event log
   itself**, and the **schema version of every consumed artifact**, plus the canonical PCM
   SHA-256, config/manifest/timeline identities, the exact half-open searched intervals, the
   reference track, thresholds and tie-breaks in permille, event roles and geometry IDs, and
   NumPy/SciPy versions.

7. **Wiring and the invariant matrix.** `run_marker_analyze` joins
   `tests/test_raw_guard.py::COMPOSED`, which parametrizes all three INV-01 properties over
   every composed runner. `tests/test_memory.py` gains `TestTheMarkerAnalysisPathStreams` on
   M2's ordered-event-log technique. `schema_export.py` gains three schemas.

8. **Gate green, zero skips**, plus `./scripts/codex-review.sh plan M10` before implementing
   (findings recorded with accept/reject reasons) and the `.venv-rocm` default suite.

### The bench — the operator's step, mid-milestone

Phase A ends by producing the three candidates' WAV + HTML + manifest sets and a printable
protocol at `docs/M10-marker-bench-protocol.md` (the charter's `## Bench protocol`, made
executable). The operator records; `raw/` is hashed before and after; the independent event
log names candidate, role, approximate time and geometry ID. No assistant is required.

**A4 — the false-positive block is measured off-bench, and the bench shrinks accordingly.**
Approved by the operator on 2026-08-05, whose constraint was concrete: alone in the house, able
to place transmitters and press a button, unable to produce ten minutes of conversation and
media on demand.

The charter's bench step 4 asks for "room tone, speech, and ordinary media without the marker to
measure false positives". That measurement does not need the bench at all — this host already
holds 13.7 minutes of real DJI audio across two captures, including two people overlapping
deliberately and hand claps at both ends, none of which contains a marker. Running the detector
over it is strictly better evidence than the planned block: more audio, more adversarial, real
capture chain, and zero minutes in the room. Result: zero accepted sequences for all three
candidates, strongest single chirp 186 permille against a 550 threshold
(`../../fixtures/2026-08-05-marker-false-positive-sweep.md`).

So the bench keeps only what needs a room — the acoustic path from a phone speaker to six lavs —
and drops to **fourteen button presses**: nine in the opening block, three at the close, two
moved-phone diagnostics. The ten-minute gap between opening and close is unattended, and exists
only so the closing block is genuinely later. The closing block drops from three plays per
candidate to one, because repeatability is established by the opening block and the close only
has to anchor the other end.

**What this does not do.** Not producing a false positive on speech and being reliably found
across a table are opposite failure directions; a detector that accepted nothing would score
perfectly on the sweep. Question 1 of the bench — reach at the farthest seat — is untouched and
is still the question a candidate wins or loses on. The sweep bounds one side of the score
threshold; the bench sets the other, and ADR-0042 freezes the pair together.

**A5 — the bench records no event log.** Approved by the operator on 2026-08-05: they will hand
over the recording and nothing else.

`--event-log` stays a real, tested feature — it is what a *session* will use — but this bench
does not need it, and pretending otherwise would have cost the operator a written log for
nothing. Of the four things it supplies, two are recoverable and two are testimony:

| the log supplies | at this bench |
| --- | --- |
| which times to search | the whole take: `--start-window-s`/`--end-window-s` wide open |
| which waveform was played | the occurrence itself — three candidates in one take were checked and cross-detected zero times |
| which play is the start and which the end | **the fixed block order**, which the protocol pins |
| that the phone did not move between them | **a spoken slate**, in the operator's voice, on all six tracks |

The last is the only irreducible one, and ADR-0040 is explicit that nothing in the audio can
establish it. Moving it from a YAML field to a sentence on the recording does not weaken it —
it is timestamped by construction, captured six times over, and cannot be misremembered a day
later. A written log filled in from memory afterwards would have been the weaker artifact.

**The cost, stated exactly.** Without a log `_assign_roles` cannot name a start/end pair (its
fallback needs exactly one occurrence per default window; three plays per block means it will
not fire), so no `differential_arrival` or `clock_drift_evidence` classification is emitted. It
warns, enumerates everything, and exits zero. That comparison is arithmetic over two groups'
per-track lags and gets done by hand against the block structure — the classification is a label
on a subtraction, not a measurement that is lost. Everything the bench exists to measure —
per-track scores, clipping, weak-signal flags, exact integer lags, source coordinates, and the
timecode cross-check, which accepts an unlabelled first group — is untouched.

**Verified rather than assumed:** a bench-shaped session (three candidates, nine opening plays,
three closing, two moved) was analyzed with no log. Every detection mapped to a planted position
exactly, no candidate detected another's waveform, and the runs exited zero with a warning
naming precisely what was unlabelled and why.

### Phase B — freeze against the evidence, then close

1. Score every take with `dnd-audio marker analyze`, once per candidate:

   ```bash
   dnd-audio marker analyze <session> --marker cand-a --start-window-s 1200 --end-window-s 1200
   ```

   **There is no event log (A5), so the take's own structure is the record.** The operator
   played a fixed order and spoke a slate before each block; the four blocks are, in order:
   nine opening plays (three of each candidate), a ten-minute unattended gap, three closing
   plays (one of each), then two moved-phone plays of `cand-a` only. Sorting a candidate's
   occurrences by `anchor_sample` therefore identifies which block each belongs to. **Listen to
   a track to confirm the slates before trusting that mapping**, and if the counts disagree with
   the structure, that is a finding to record rather than a discrepancy to reconcile.

   Each run overwrites both artifacts, so copy each candidate's pair aside before the next.

   What to read out of them: per-track `score_permille` and `runner_up_permille` at the nearest
   and farthest seats, `clipped` on the nearest, `weak` on the farthest, `gap_errors_samples`
   (the direct OQ-029 measurement — how much the phone's playback stretched), spread of
   `relative_lag_samples` across the three same-position repeats, and the timecode comparison.

   `marker_roles_unassigned` will warn on every run. **That is expected**, not a failure: the
   start/end pair is named by hand from the block structure, and the start-to-end differential
   arrival is arithmetic over two groups' per-track lags. Compute it; do not reach for
   `--event-log` to make the warning go away. The moved-phone plays must be enumerated and
   **must not** be called drift.

   The false-positive half is already discharged (A4) — do not expect a speech/media block in
   the take, and do not treat its absence as missing evidence.
2. Select v1. **ADR-0042** freezes its exact integer PCM sequence by SHA-256, its anchor,
   the detector thresholds and tolerances, and the human-readable recipe. Add the `v1` entry
   to `MARKER_SPECS`, so `marker build OUTDIR` resolves; the candidate names stay as history.

   **The trap here is fitting the thresholds to the observations.** Every constant in
   `DetectorThresholds` currently cites OQ-025 or OQ-029, and the temptation on seeing one
   bench is to set each to whatever the take produced. Two guards against that: the
   false-positive sweep says the score threshold may fall as far as **100 permille** before
   real speech starts assembling sequences (`../../fixtures/2026-08-05-marker-false-positive-sweep.md`),
   so a weak farthest seat is a reason to *lower the threshold*, not to reject a candidate; and
   a threshold set exactly at the observed value has no margin, so state the margin and its
   reasoning in the ADR rather than the observation alone. If the evidence argues for changing
   the detector's *shape* rather than its constants, say so — that is one of the two outcomes
   this mid-milestone bench exists to surface before the schemas freeze.
3. Re-baseline whatever the evidence moves. The synthetic regression battery is
   **parametrized over every candidate spec** precisely so that a different winner is a
   parameter and not a re-baseline; only a change to the candidate *design space* or to the
   detector's shape costs real work, and those are the two outcomes the mid-milestone bench
   exists to surface before the schemas are frozen.
4. Update H1's and H2's charters, the spec, OQ-025 and OQ-029 with the measured result, and
   fold the marker into `docs/H1-two-person-recording-runbook.md` as an *alternative* to the
   three-clap pattern — never as a replacement for the LTC jam, and never for a restarted
   file that missed the marker.
5. `./scripts/codex-review.sh code M10 main`, mutation checks on every load-bearing proof,
   the full zero-skip gate, and the `.venv-rocm` default suite. Then Verify, then Close.

### Files

**New:** `src/dnd_audio/marker/{__init__,spec,sine,synth,wav,page,manifest,builder,detect,eventlog,analysis,report,runner}.py`; `src/dnd_audio/marker/data/sine_table.json`; `scripts/design_sine_table.py`;
`schemas/marker-{manifest,analysis,report}.schema.json`; `docs/M10-marker-bench-protocol.md`;
`docs/plan/decisions/00{40,41,42}-*.md`; `tests/test_marker_{synth,wav,page,build,detect,eventlog,analyze}.py`.

**Changed:** `cli.py` (the `marker` sub-app); `schema_export.py`; `tests/test_raw_guard.py`
(`COMPOSED`); `tests/test_memory.py`; `tests/test_schema_drift.py`;
`dnd-audio-ingestion-agent-spec.md`; `docs/plan/{OPEN-QUESTIONS,INVARIANTS,ROADMAP,STATE}.md`;
`docs/plan/milestones/{H1,H2}-*.md`; `docs/H1-two-person-recording-runbook.md`.

### Completion gate → the proof that demonstrates it

| # | Criterion | Proof | Phase |
| --- | --- | --- | --- |
| 1 | Docs agree on what the marker is and is not | ADR-0040/0042, the spec amendment, OQ-025/OQ-029, H1/H2, the runbook; `check_plan.py` in the gate | B |
| 2 | Byte-stable WAV/HTML/manifest; extracted WAV identical to the CLI WAV | `test_marker_build.py` — two builds byte-identical per candidate; the page's payload decoded and compared to the WAV bytes **and** to the manifest's SHA-256; the payload asserted to occur exactly once | A (per candidate), B (v1) |
| 3 | The page works offline on the intended phone, one playback at a time, no external resource, no second synthesis | `test_marker_page.py` — the document parsed and **every** URL-bearing attribute asserted absent or a `blob:`/`data:` the page generated, plus a `default-src 'none'` CSP; no oscillator or `Math.sin`; the declarative state-machine table parsed and asserted to admit no play-while-playing transition. That the JS *applies* the table, that `ended` resets the UI, and that the download works are **bench** claims — see the amended criterion 3 | A (static) + bench |
| 4 | The bench detects every fixed-position marker on all tracks, no clipping near, decisive far, no false sequence on ordinary material | Split, and the halves are answered in different places. **False positives: already done** — `tests/test_marker_false_positives.py` (`host_smoke`) over every real DJI recording on the host, written up in `../../fixtures/2026-08-05-marker-false-positive-sweep.md`. **Reach, clipping and decisiveness:** the bench takes, scored through `marker analyze`; **sanitized** measurements, commands, hashes and conclusions in `docs/fixtures/` — never the takes, and no audio committed | A (negatives) + bench (positives) |
| 5 | All occurrences in the canonical searched set; one-to-one groups and unmatched detections; exact lags and source coordinates; missing/weak/clipped/ambiguous kept distinct; repeats, a missed event, moved-position diagnostics and overlapping windows cannot silently change the chosen pair | `test_marker_analyze.py` — one class per outcome, each on a fixture built from **independent** ground truth (the generator places the marker at a declared sample; the assertion is that exact integer, never a value the detector produced) | A |
| 6 | The marker report satisfies INV-13 for complete/inconclusive/failed/skipped/partial; ordinary failures write it atomically and exit correctly; an unsafe resolved path writes nothing under `raw/` | `test_marker_analyze.py::TestTheReport` driven **through the CLI**, including `--report SESSION/raw/tx-a/<a real recording>` **and `marker build SESSION/raw/tx-a`**; in both the file's bytes are asserted unchanged and no directory is created | A |
| 7 | Synthetic regressions: exact delays, stream seams, gain/EQ, reverberation, noise, clipping, truncation, reversed/partial, duplicate peaks, speech/music negatives, sample-rate/time-scale perturbation; M8's within-quantum offset stays healthy; material fixed-geometry change warns | `test_marker_detect.py`, parametrized over **every** candidate spec | A |
| 8 | Fixed memory independent of session and search length, proven over the composed path | `test_memory.py::TestTheMarkerAnalysisPathStreams` — one ordered event log over reads, correlation blocks and the write; a block produced before the last read; and, because a sparse long search passes that while occurrence lists grow, a **dense**-accepted-candidate case asserting the versioned occurrence ceiling fails explicitly rather than truncating or accumulating | A |
| 9 | Outputs declared before cleanup; resolved paths under sources refused; every source snapshotted; verified after the last read and before publication; in the central INV-01 matrix | `tests/test_raw_guard.py::{TestCleanupNeverWritesIntoRaw,TestEveryComposedRunVerifiesItsSources}` with `marker-analyze` added to `COMPOSED`; plus a mid-run source mutation; plus `marker build`'s own destination guard, which is **not** covered by `COMPOSED` because it takes no session (see step A4) | A |
| 10 | Marker semantics invalidate only their own artifacts; no `SessionConfig` field; stale timelines refused | `test_marker_analyze.py::TestIdentity` — vary **each** identity component (marker, detector *and* analysis semantics versions, event-log digest, every consumed schema version) and assert the analysis identity moves, while `manifest.json`, `timeline.json`, `activity.json` bytes and every existing cache key do not (the `archive/config.py` regression's shape: compare **cache keys**, not output bytes); `test_config.py` asserts no marker field on `SessionConfig`; each staleness component refused with its own code and its own test; and the whole pre-existing `work/`/`output/`/cache tree asserted byte-identical after a run | A |
| 11 | Raw hashes match before and after; nothing generated is committed | The bench's own before/after hashes; `.gitignore` and a test that the repository contains no `.wav`/`.html` marker artifact | bench |
| 12 | Default suite offline/CPU/model-free with zero skips; benches separately recorded; both reviews dispositioned | `./scripts/gate.sh`; the `.venv-rocm` run; `docs/plan/reviews/M10-{plan,code}-*.md` | A + B |

### Invariants this could plausibly violate, and what stops it

- **INV-01** — a new command that writes two artifacts and reads every source. Stopped by
  joining `COMPOSED` rather than by a test of its own, which is the lesson three milestones
  in a row had to learn the hard way (ADR-0021).
- **INV-02** — `sync-marker-analysis.json` is a new deterministic artifact. No wall clock, no
  counters, no float comparisons; `canonical_json`; a walk over the serialized document
  asserting it contains no float, as `timeline.json` has.
- **INV-04** — every lag, anchor and interval is an integer sample; the only float is at the
  public millisecond boundary through `public_seconds`. No second quantizer.
- **INV-05** — no new heavyweight dependency, no browser, no network. The page is a string.
- **INV-06** — `marker build` and `marker analyze` join `_COMMANDS` in
  `tests/test_archive_isolation.py`, which runs every network-denied command in one child
  interpreter under a `sitecustomize` socket trap on its `PYTHONPATH`. That list is
  deliberately stated rather than derived, so a new command is one visible missing entry.
- **INV-07** — the overlap-save block, proved over the composed path rather than over the
  component.
- **INV-08** — no new cache, so nothing to key wrongly; the identity document is recorded
  instead, and the test asserts existing cache keys do not move.
- **INV-13** — a second report format at its own command boundary, following ADR-0039's
  precedent rather than growing `ingest-report.json` a seventh stage.

### OQ-029, raised here

**Does a phone browser's playback of embedded 48 kHz PCM preserve the marker's internal
timing well enough for sequence gap detection?** A browser may resample to the hardware rate,
and media-pipeline dynamics processing may stretch or compress. The inter-chirp gap tolerance
is a number chosen against synthetic audio; the bench is what sets it. Cited from the gap
tolerance constant, so `rg 'OQ-029'` finds it. Needs: M10's bench. Blocks: freezing v1.

### Deliberately not doing

Everything in `## Explicitly not in this milestone`, and specifically: no timeline correction
or affine resampling; no marker field on `SessionConfig`; no change to `ingest-report.json`,
its schema, or any existing cache identity; no marker detections reaching activity, mix, ASR
or transcript semantics; no whole-session scan by default; no Wake Lock; no hosted page,
service worker or PWA; no committed audio binaries; and no browser dependency (A3).

## Completion gate

- [ ] The spec, ADR, OQ-025, H1/H2 charters, and operator runbook agree that marker v1 verifies
      the jam, always reports differential acoustic arrival, and calls it recorder-drift
      evidence only with fixed phone **and lav** geometry; it never replaces timecode or
      corrects a track.
- [ ] `marker build` produces byte-stable WAV, standalone HTML, and manifest artifacts; the WAV
      extracted from HTML is byte-identical and has the same frozen SHA-256 as the CLI WAV.
- [ ] The HTML works offline on the intended phone/browser with one user-initiated playback at a
      time, no external resource/network API, clear marker identity/state, and no independent
      waveform synthesis.
      **Amended 2026-08-05** (A3, and the second plan review's finding 4, which was right that
      a scratch section may not override this contract). The split of proof is now stated
      rather than assumed. *Mechanically, in the default gate:* the document is parsed and
      every URL-bearing attribute is absent or a `blob:`/`data:` the page generated, under a
      `default-src 'none'` CSP; the payload occurs exactly once and decodes to the CLI WAV's
      bytes; there is no oscillator and no second synthesis; and the playback state machine is
      a declarative JSON table the page's own JavaScript reads, which the test parses and
      asserts admits no play-while-playing transition. *At the physical bench, and nowhere
      else:* that the JavaScript applies that table, that `ended` resets the UI, that the
      download yields the canonical WAV, and that the embedded asset reaches the speaker.
      The opt-in headless-browser smoke is **deferred** — nothing in the flake provides a
      browser, and the phone bench is strictly stronger evidence for every claim it would
      have made.
- [ ] The bench recording detects every fixed-position marker decisively on all intended DJI
      tracks without clipping the nearest or losing the farthest, and ordinary speech/media
      supplies no accepted false marker sequence.
      **Amended 2026-08-05** (A4): the second clause is discharged **without the bench**, over
      more and more adversarial real audio than an in-room block would have supplied —
      `tests/test_marker_false_positives.py` over every real DJI recording on the host, zero
      accepted sequences for all three candidates
      (`../../fixtures/2026-08-05-marker-false-positive-sweep.md`). The first clause is
      untouched and remains entirely the bench's: not inventing a marker from speech and
      reliably finding one across a table are opposite failure directions, and a detector that
      accepted nothing at all would satisfy the second clause perfectly.
- [ ] `marker analyze` reports all occurrences inside the canonical searched interval set,
      reference-anchored one-to-one groups and unmatched detections, exact integer-sample lags
      and source coordinates, and distinct missing/weak/clipped/ambiguous outcomes. Repeated
      events, one missed event, moved-position diagnostics, and overlapping windows cannot
      silently change the chosen start/end pair.
- [ ] The separate marker report satisfies INV-13 for complete, inconclusive, failed, skipped,
      and partial outcomes; ordinary failures write it atomically and exit correctly, while an
      unsafe resolved report path writes nothing under `raw/`.
- [ ] Synthetic CPU/offline regressions cover exact delays, stream seams, gain/EQ, reverberation,
      noise, moderate clipping, truncation, reversed/partial patterns, duplicate peaks,
      deterministic speech/music negatives, and sample-rate/time-scale perturbation with
      independent ground truth. M8's within-quantum offset stays healthy and material fixed-
      geometry lag change emits a drift warning.
- [ ] Fixed-size overlap-save processing has maximum memory independent of session/search length,
      proven over the composed analyzer path by observing processing/publication progress before
      the last read. **Amended 2026-08-05:** that proof alone is insufficient and passes while
      retained occurrence, group and unmatched-detection lists grow with the search range
      (second plan review, P0-2). The bound also requires a **versioned occurrence ceiling that
      fails explicitly rather than truncating**, demonstrated on a *dense*-accepted-candidate
      input rather than a longer sparse one.
- [ ] `marker analyze` declares every prospective output before cleanup, rejects resolved paths
      under sources, snapshots every configured source, verifies after the last read and before
      publication, and joins the central INV-01 composed-command regression matrix.
- [ ] **`marker build` refuses a destination resolving under any enclosing session's configured
      source roots**, before it creates a directory, writes a candidate, or unlinks a prior
      manifest — proved by driving the real command at a real recording's directory and
      asserting that recording's bytes are unchanged. **Added 2026-08-05** (second plan review,
      P0-1): the command takes an arbitrary destination and no session argument, so it is
      outside the composed-command matrix and INV-01 would otherwise be unguarded there. Same
      shape as the P0 M7a's second code review found.
- [ ] Marker build/analyze semantics invalidate only their own deterministic artifacts; existing
      session config hashes, inspection/timeline/activity/mix/transcript artifacts, and every ASR
      cache identity remain unchanged. Marker invocation/event-log data is not a
      `SessionConfig` field, and stale timelines are refused rather than trusted or silently
      rebuilt.
- [ ] Raw hashes match before and after the bench and analysis; no generated WAV/HTML, recording,
      browser artifact, or device-specific private path is committed.
- [ ] The default suite remains offline/CPU/model-free with zero skips, the hardware bench is
      separately recorded, and independent plan/code reviews are fixed or explicitly
      dispositioned before close. **Amended 2026-08-05:** the local-browser bench is deferred
      with its reason — see the amended criterion 3 — so this criterion no longer requires a
      recording that will not exist. Two plan reviews are on record
      (`../reviews/M10-plan-20260804-1735.md`, `../reviews/M10-plan-20260805-0606.md`); the
      second reverses part of the first, and both dispositions are written down.

## Known risks and open questions

- Mobile browsers and audio hardware may resample or process PCM differently. Exact embedded
  bytes prevent implementation drift; only the physical bench establishes usable acoustics.
- Phone speaker output below 500 Hz and near 8 kHz varies. The provisional band is a candidate,
  not a default justified by theory alone.
- Moving the phone **or any compared lav** between start and end aliases propagation change into
  apparent drift. Normal sessions with moving wearers therefore provide differential-arrival
  evidence; a fixed-transmitter soak provides the clean clock-drift measurement.
- The page can request playback at unity gain but cannot read or lock the physical media-volume
  step reliably. Operator setup and bench evidence remain part of the instrument.
- A marker obvious enough to detect is audible and briefly interrupts the room. H1/H2 should
  place it before play begins and after play ends, not inside conversational content.
