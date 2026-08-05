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
4. Include room tone, speech, and ordinary media without the marker to measure false positives.
5. Keep an independent event log naming candidate, role, approximate time, and geometry ID. Run
   `inspect`, `ingest`, and the private candidate analyzer; hash raw before and after.
6. Confirm every track detects the fixed-position events, nearest tracks do not clip, farthest
   tracks remain decisive, false-positive material produces no accepted sequence, and repeated
   same-position lag is inside the documented tolerance.

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

1. Amend the product spec and OQ-025, then write an ADR separating LTC placement, acoustic QA,
   differential arrival, and fixed-endpoint drift evidence. Freeze the marker-analysis and
   INV-13 report schemas, exact anchor/lag signs, occurrence grouping, outcomes, commit points,
   and identity before production code.
2. Keep marker options out of `SessionConfig`. Define a separate versioned invocation/event-log
   model whose complete identity includes detector/artifact semantics, canonical PCM SHA-256,
   current source/manifest/timeline identities, exact half-open searched intervals, reference
   track, thresholds/tie-breaks, event roles/geometry IDs, and output-affecting numeric-library
   versions. Define read-only stale-timeline validation.
3. Implement private deterministic candidate generation/player tooling plus a fixed-memory
   overlap-save detector. Exercise delayed, filtered, reverberant, noisy, gain-scaled, clipped,
   truncated, reversed, time-scaled/resampled, and deterministic speech/music-shaped negative
   fixtures.
4. Run the no-assistant candidate bench on the intended phone/browser and DJI hardware with
   fixed source **and transmitter** geometry, an independent event log, and raw hashes. Include
   a separately labelled moved-phone diagnostic. Select marker v1 from the evidence, then freeze
   its exact integer PCM, anchor, thresholds, SHA-256, and tolerance in the ADR.
5. Implement public `marker build` from that frozen canonical sequence, with deterministic
   manifest/schema, standalone inline HTML, page state machine, atomic files, manifest-last
   publication, and byte-extraction equivalence tests.
6. Implement `marker analyze`: complete-sequence detection, reference-anchored one-to-one
   occurrence association, explicit matched/unmatched events, fixed-geometry classification,
   deterministic analysis/schema, non-corrective M8-aware timecode comparison, stale-timeline
   refusal, INV-01 raw guarding, atomic publication, and an INV-13-compliant report.
7. Add `run_marker_analyze` to the central composed-command invariant matrix. Add CPU/offline
   tests for output symlinks, mid-run source mutation, cleanup ordering, fixed memory over long
   ranges, stream seams, ambiguity, exact sample/source mapping, marker identity, report failure
   semantics, page network absence, and unchanged existing artifacts/caches/config hashes.
8. Run the opt-in local-browser smoke, frozen-v1 phone/DJI confirmation, independent code
   review, full zero-skip gate, and the `.venv-rocm` default suite. Authorize v1 in H1/H2 only
   after every bench and invariant gate passes.

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
- [ ] The bench recording detects every fixed-position marker decisively on all intended DJI
      tracks without clipping the nearest or losing the farthest, and ordinary speech/media
      supplies no accepted false marker sequence.
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
      the last read.
- [ ] `marker analyze` declares every prospective output before cleanup, rejects resolved paths
      under sources, snapshots every configured source, verifies after the last read and before
      publication, and joins the central INV-01 composed-command regression matrix.
- [ ] Marker build/analyze semantics invalidate only their own deterministic artifacts; existing
      session config hashes, inspection/timeline/activity/mix/transcript artifacts, and every ASR
      cache identity remain unchanged. Marker invocation/event-log data is not a
      `SessionConfig` field, and stale timelines are refused rather than trusted or silently
      rebuilt.
- [ ] Raw hashes match before and after the bench and analysis; no generated WAV/HTML, recording,
      browser artifact, or device-specific private path is committed.
- [ ] The default suite remains offline/CPU/model-free with zero skips, the local-browser and
      hardware benches are separately marked/recorded, and independent plan/code reviews are
      fixed or explicitly dispositioned before close.

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
