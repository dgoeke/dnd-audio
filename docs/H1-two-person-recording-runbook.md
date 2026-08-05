# H1 two-person recording runbook

> **Need architecture direction, not H1 completion?** Use the much smaller
> [minimal two-person acoustic direction check](minimal-acoustic-direction-check.md): four
> continuously recording transmitters, one two-to-three-minute take, spoken slates instead of
> written logs, and one transfer afterward. The full procedure below exists only to close every
> H1 hardware/metadata question.

This is the executable capture protocol for
[H1 — Real DJI hardware fixture](plan/milestones/H1-hardware-fixture.md). It satisfies H1 with
**two people, six transmitters, and three receivers**. It is intentionally more explicit than
the charter: observations made at capture time cannot be reconstructed after the hardware is
packed away.

The fixture is expected to take 30--60 minutes to configure and verify, followed by roughly
4--6 minutes of useful audio. The original two-minute estimate predates the hard-onset,
exact-overlap, controlled-pause, power-cycle, and M9 controls. Keep the fixture short, but do not
rush those controls merely to hit two minutes.

This document uses `Person One` and `Person Two`; replace those names in the private recording
log if desired. Public evidence must not contain a hostname, username, absolute home-directory
path, or audio. Never modify anything after it has been copied under a session's `raw/`.

## What two people can and cannot prove

Two people are enough to produce genuine different-voice overlap and exact simultaneous
`Okay`, while each alternately wears three transmitters for direct-source controls:

| Human voice | Direct-source rounds | Receiver groups |
| --- | --- | --- |
| Person One | `tx-a`, `tx-c`, `tx-e` | `rx-a`, `rx-b`, `rx-c` |
| Person Two | `tx-b`, `tx-d`, `tx-f` | `rx-a`, `rx-b`, `rx-c` |

Treat each transmitter as a separate **test role** in `session.yaml` (`h1-tx-a` through
`h1-tx-f`). Keep a separate private `human_voice` column in the recording log. Assigning six
pseudo-speakers prevents the public transcript from pretending that one track permanently
represents Person One when that person moves among three transmitters.

This fixture does **not** reproduce a six-person table or provide six independent voice
profiles. It can validate all H1 metadata requirements, direct-versus-bleed geometry for two
real voices, and genuine two-speaker overlap. Threshold calibration over natural six-person
conversation remains H2 work.

## People and materials

- Person One: hardware operator, primary receiver/timecode operator, and speaker.
- Person Two: independent checker, phone-camera/timekeeper operator, and speaker.
- Three receivers labeled durably `rx-a`, `rx-b`, `rx-c`.
- Six transmitters labeled durably `tx-a` through `tx-f`.
- A written receiver/channel map. The examples assume:

  | transmitter | receiver | channel | human during direct round |
  | --- | --- | --- | --- |
  | `tx-a` | `rx-a` | 1 | Person One |
  | `tx-b` | `rx-a` | 2 | Person Two |
  | `tx-c` | `rx-b` | 1 | Person One |
  | `tx-d` | `rx-b` | 2 | Person Two |
  | `tx-e` | `rx-c` | 1 | Person One |
  | `tx-f` | `rx-c` | 2 | Person Two |

- The LTC cable used to jam receivers.
- A phone camera showing seconds and the local time zone, or a clock that can be filmed in the
  same shot as each receiver display.
- A stopwatch or visual interval timer for 250/330/500 ms pause targets. It must be silent or
  visible only to the speaker; do not play cue speech into the fixture.
- Six marked table positions, approximately where six players would sit. Unworn transmitters
  remain at those positions rather than in a pile.
- A computer with this repository's `direnv` environment and enough storage to copy the files.
- A fresh private recording log made from the templates below.

If the actual receiver/channel map differs, use the real map everywhere. Do not rename a DJI
file to make it resemble this example; the containing `raw/<track-id>/` directory supplies
identity and the original filename is evidence.

## Stop conditions

Do not begin the main take if any of these are true:

- A label, receiver/channel assignment, battery level, storage check, or recording format is
  unknown.
- The three receiver displays do not agree after the jam.
- The ten-second jam-check files do not confirm the shared clap at the timecode-predicted
  position across all three receiver groups.
- Any transmitter fails to show its recording indicator.
- A take was copied under `raw/` and someone proposes renaming, normalizing, trimming, or
  rewriting it.

If a jam check is inconclusive, re-jam and repeat the check. A failed or low-confidence check is
useful operator feedback, not permission to proceed and hope ingest repairs it.

## Private recording-log header

Complete this before recording:

```text
fixture id:
calendar date:
local time zone:
location/room description:
table dimensions or approximate mic spacing:
Person One private identifier:
Person Two private identifier:
receiver/transmitter firmware versions:
LTC cable or external generator used:
dual-file mode enabled on:
unexpected setup observations:
```

Record one row per receiver immediately after the jam and again after the main take:

| observation | wall clock with zone | receiver | displayed timecode | displayed rate | matches `rx-a`? | photo/video id |
| --- | --- | --- | --- | --- | --- | --- |
| after jam | | `rx-a` | | | reference | |
| after jam | | `rx-b` | | | | |
| after jam | | `rx-c` | | | | |
| after main take | | `rx-a` | | | reference | |
| after main take | | `rx-b` | | | | |
| after main take | | `rx-c` | | | | |

This table and the image/video are the unrecoverable evidence for OQ-012 and OQ-015. Record
what the displays actually show; do not infer or “correct” a value from the WAV metadata.

## Phase 1 — hardware configuration

- [ ] Photograph or write down the durable label and serial number of all nine devices.
- [ ] Confirm the receiver/channel map and that the kits remain in their intended independent
      groups. Six transmitters require at least two DJI groups; only the cross-group boundary
      is manually jammed.
- [ ] Set all receiver timecode displays to the same rate. Use 30 fps unless the operator has a
      reason to use another supported value. A 60 fps setting does not improve the `orig` file's
      measured 1600-sample/33.3 ms reference quantum.
- [ ] Confirm 32-bit-float internal recording on every transmitter. If one differs, record the
      mismatch rather than silently changing the log after capture.
- [ ] Enable dual-file `orig`/`edit` recording on every transmitter if practical. At minimum,
      enable it on `tx-a` and one transmitter in another receiver group. Record exactly which
      devices have it enabled; OQ-007 needs a real pair.
- [ ] Confirm battery, free storage, and recording format on all six transmitters.
- [ ] Confirm receiver battery and the timecode/LTC mode needed for the jam.
- [ ] Place the six unworn transmitters at the six marked table positions.

Do not format storage unless the desired recordings have already been backed up. Formatting is
outside this runbook.

## Phase 2 — jam all three receivers

1. Connect `rx-a` LTC out to `rx-b` LTC in.
2. Perform the receiver's Sync operation.
3. With both displays visible, confirm `rx-b` matches `rx-a`; only then disconnect.
4. Connect `rx-a` LTC out to `rx-c` LTC in.
5. Perform Sync, confirm `rx-c` matches `rx-a`, then disconnect.
6. Person Two films each receiver display beside the wall clock and reads aloud, off-fixture,
   the receiver label, displayed timecode, displayed rate, wall-clock time, and time zone.
7. Fill the three “after jam” rows in the display table.

Do not use `bext.origination_time` to decide whether the jam worked. Prior captures found two
receiver wall clocks 48.7 seconds apart while their jammed counters agreed.

## Phase 3 — mandatory ten-second jam check

This take proves that the visible jam reached files before the main fixture is recorded.

1. Keep every receiver powered for the remainder of H1.
2. Start internal recording on `tx-a` through `tx-f`, approximately two seconds apart, and log
   each wall-clock start time. Confirm every recording indicator.
3. After all six are recording, Person One says, “H1 jam check,” waits one second, and makes
   **one sharp hand clap** at the middle of the table. Do not clap beside a lav.
4. Wait at least five seconds, then stop all six transmitters without powering them off.
5. Copy the check take from every transmitter to a temporary verification session, preserving
   filenames exactly. At minimum the check must include one transmitter from each receiver;
   using all six also catches a within-group problem.
6. Inspect and ingest the temporary session with `sync_qa.enabled: true`, `window_s: 5`, and
   `max_lag_ms: 100`. The normal offset warning floor should be derived from the 30 fps/BWF
   quantum; do not configure a threshold below it.
7. Inspect `output/ingest-report.json`. Proceed only if the sources use the expected shared
   timecode origin, the clap produces cross-receiver `sync_qa_measured` evidence within one
   30 fps frame, and there is no `timecode_disagreement`. Repeat the take if correlation is low
   confidence or a receiver group is absent.

Use the main-session configuration shape later in this document, reduced to the tracks copied
for the check take and with the shorter sync-QA window stated above. Then run:

```bash
uv run dnd-audio inspect "$H1_JAM_CHECK_DIR"
uv run dnd-audio ingest "$H1_JAM_CHECK_DIR"
```

Example report query:

```bash
jq '.decisions[]
    | select(.code == "sync_qa_measured"
             or .code == "sync_qa_low_confidence"
             or .code == "timecode_disagreement")' \
  "$H1_JAM_CHECK_DIR/output/ingest-report.json"
```

The variable name is illustrative. Set `H1_JAM_CHECK_DIR` to the actual temporary session
directory; never use a session's `raw/` as a scratch target.

The check take is also the required second recording in the same receiver power-on cycle. Keep
its original directory listing and metadata even after the main take succeeds.

## Phase 4 — prepare the main take

Leave all receivers powered after the successful jam check.

- [ ] Reconfirm all three receiver displays still match `rx-a`.
- [ ] Start `tx-a` through `tx-f` about two seconds apart; log both order and wall-clock time.
- [ ] Confirm every recording indicator.
- [ ] Person One wears only the transmitter named for a direct-source round. Person Two does
      the same. A transmitter may be clipped at the normal chest position for its line and
      returned to its marked table position afterward.
- [ ] When two transmitters are worn for overlap, all other transmitters remain distributed at
      their table positions as bleed observers.
- [ ] Start a stopwatch at the slate. Log actual elapsed times; the schedule below is an order,
      not an assertion that humans hit exact timestamps.

## Phase 5 — spoken main-take script

Text in **bold quotation marks** is spoken into the fixture. Bracketed directions are silent
actions and are not spoken. Person Two logs actual elapsed time, intended direct transmitter,
human voice, pause target, and whether the delivery was clean, late, or repeated.

### A. Slate and start landmark

1. Person One, near the centre of the table:
   **“H1 main fixture. Two human voices, six transmitters, three receivers. Main take one.”**
2. [One second of silence.]
3. Make a distinctive **three-clap pattern**: clap, short pause, clap, longer pause, clap.
   If the OQ-025 generated marker and its matched-filter detector have both landed and passed
   their bench test before H1, play that same prepared three-chirp WAV from the center of the
   table instead. Do not substitute an ad-hoc tone, and do not skip the LTC jam.
4. [Two seconds of silence.]

### B. Six solo/direct-source rounds

Before every line, clip only the named transmitter at the speaker's normal chest position.
Everyone else stays silent. Leave at least one clean second before the hard-onset sentence.

| order | human | direct transmitter | spoken text |
| --- | --- | --- | --- |
| 1 | Person One | `tx-a` | **“Transmitter A, Person One.”** [one second] **“Testing amber lanterns beside the northern doorway.”** |
| 2 | Person Two | `tx-b` | **“Transmitter B, Person Two.”** [one second] **“Pick seven silver coins from the wooden table.”** |
| 3 | Person One | `tx-c` | **“Transmitter C, Person One.”** [one second] **“Take the crimson map toward the western tower.”** |
| 4 | Person Two | `tx-d` | **“Transmitter D, Person Two.”** [one second] **“Testing quiet dragons beneath the stone bridge.”** |
| 5 | Person One | `tx-e` | **“Transmitter E, Person One.”** [one second] **“Pick the brass key before opening the cellar.”** |
| 6 | Person Two | `tx-f` | **“Transmitter F, Person Two.”** [one second] **“Take four blue candles into the eastern chamber.”** |

Each hard-onset sentence must be logged against the intended `track_id`. Repeat a line only if
the log records both attempts; never erase an imperfect attempt from the ground truth.

### C. Power-cycle control

1. After the `tx-f` solo line, stop `tx-f`, turn it fully off, and log the wall-clock time.
2. Leave it off for at least ten seconds while section D begins.
3. Turn `tx-f` back on, confirm format/storage/battery, start a new internal recording, and log
   the wall-clock time and displayed filename/counter if available.
4. Person Two wears `tx-f` and says:
   **“Transmitter F after power cycle.”** [one second] **“Testing the final counter after restart.”**

Keep both pre- and post-cycle files. Their original filenames and directory listing are the
evidence for OQ-003.

### D. Controlled same-track pause controls

Person One wears `tx-a`. Person Two silently operates a visible interval timer. Record the
target and later measure the actual word gap; human timing is not ground truth merely because a
number was requested.

1. **Below threshold, target 250 ms, one intended turn:**
   **“The lantern is blue”** [250 ms] **“and hangs beside the door.”**
2. **Near threshold, target 330 ms, one intended turn:**
   **“We cross the bridge”** [330 ms] **“and enter the keep.”**
3. **Above threshold, target 500 ms, two intended statements:**
   **“The corridor is empty.”** [500 ms] **“The chest is locked.”**

These three controls test the 350 ms presentation threshold. They do not authorize changing it
inside H1.

### E. Quick-handoff hard onset

Person One wears `tx-a`; Person Two wears `tx-e`. Person Two begins as soon as Person One ends,
aiming for a 0--200 ms handoff without overlap:

- Person One: **“I finish beside the old gate.”**
- Person Two immediately: **“Pick the bright token before we leave.”**

Log `tx-e` as the intended direct track for `Pick`. This contrasts with the clean-pause hard
onsets in section B.

### F. Genuine two-voice overlap

Keep Person One on `tx-a` and Person Two on `tx-e`, which places the direct speakers in
different receiver groups. Use a silent finger countdown.

1. **Exact-short overlap:** on the same visual cue, both people say exactly **“Okay.”** Do not
   continue immediately; leave one second of silence so it remains an isolated short event.
2. **Different-words overlap:** on the same visual cue:
   - Person One: **“Red dragons guard the northern gate.”**
   - Person Two: **“Blue goblins cross the southern bridge.”**

Log whether either person started late, stopped, laughed, or repeated a line. This is genuine
two-speaker ground truth; do not repair it in the log.

### G. End landmark and tail

1. Confirm the post-power-cycle `tx-f` line has been recorded.
2. Person One says: **“H1 end landmark.”**
3. Make the same distinctive three-clap pattern used at the start, or replay the exact same
   bench-validated OQ-025 marker from the same central table position if it was used there.
4. Leave five seconds of room tone.
5. Stop all transmitters one at a time, logging order and wall-clock time.
6. Before powering down receivers, film and log all three displays in the “after main take”
   rows. Note anything surprising.

## Event log template

Fill one row for every scripted event and every repeat:

| event | elapsed start | elapsed end | human voice(s) | intended direct track(s) | target gap | actual delivery notes |
| --- | --- | --- | --- | --- | --- | --- |
| start three-clap | | | Person One | all observe | pattern | |
| `tx-a` solo | | | Person One | `tx-a` | clean 1 s | |
| `tx-b` solo | | | Person Two | `tx-b` | clean 1 s | |
| `tx-c` solo | | | Person One | `tx-c` | clean 1 s | |
| `tx-d` solo | | | Person Two | `tx-d` | clean 1 s | |
| `tx-e` solo | | | Person One | `tx-e` | clean 1 s | |
| `tx-f` solo | | | Person Two | `tx-f` | clean 1 s | |
| `tx-f` power off | | | — | `tx-f` | ≥10 s | |
| pause 250 | | | Person One | `tx-a` | 250 ms | |
| pause 330 | | | Person One | `tx-a` | 330 ms | |
| pause 500 | | | Person One | `tx-a` | 500 ms | |
| quick handoff | | | both | `tx-a` → `tx-e` | 0--200 ms | |
| simultaneous `Okay` | | | both | `tx-a` + `tx-e` | simultaneous | |
| different-word overlap | | | both | `tx-a` + `tx-e` | simultaneous | |
| `tx-f` after restart | | | Person Two | `tx-f` | clean 1 s | |
| end three-clap | | | Person One | all observe | pattern | |

## Phase 6 — transfer without mutating raw audio

1. Copy every `orig` and `edit` file, preserving exact DJI filenames and bytes, into private
   intake storage first.
2. Save a complete directory listing per transmitter, including the jam check, main take, and
   both `tx-f` power-cycle files.
3. Build the H1 session with one directory per track:

   ```text
   <session>/raw/tx-a/<unchanged DJI filenames>
   <session>/raw/tx-b/<unchanged DJI filenames>
   ...
   <session>/raw/tx-f/<unchanged DJI filenames>
   ```

4. Never rename, normalize, trim, retag, resample, or rewrite a file under `raw/`.
5. Hash every raw file before running the pipeline and store the hash list outside `raw/`:

   ```bash
   mkdir -p "$H1_SESSION_DIR/work"
   find "$H1_SESSION_DIR/raw" -type f -print0 \
     | sort -z \
     | xargs -0 sha256sum \
     > "$H1_SESSION_DIR/work/raw-sha256.before.txt"
   ```

6. After every analysis run, verify the same list:

   ```bash
   sha256sum --check "$H1_SESSION_DIR/work/raw-sha256.before.txt"
   ```

If the check fails, stop. Do not “restore” or replace a raw file until the exact cause is known.

## Minimal main-session configuration shape

Use the actual date, map, and private speaker labels. The six pseudo-speakers below describe
test roles, while the private event log records which of the two human voices performed each
role.

```yaml
schema_version: 1
session_id: "YYYY-MM-DD-h1"
title: "H1 two-person hardware fixture"
language: "English"
active_tracks: ["tx-a", "tx-b", "tx-c", "tx-d", "tx-e", "tx-f"]
timecode:
  frame_rate: "30F"
  origin_date: "YYYY-MM-DD"
  origin_timecode: null
  rollover_policy: "infer_forward"
tracks:
  - {track_id: "tx-a", receiver_id: "rx-a", receiver_channel: 1, speaker_id: "h1-tx-a", speaker_name: "H1 TX A", input: "raw/tx-a"}
  - {track_id: "tx-b", receiver_id: "rx-a", receiver_channel: 2, speaker_id: "h1-tx-b", speaker_name: "H1 TX B", input: "raw/tx-b"}
  - {track_id: "tx-c", receiver_id: "rx-b", receiver_channel: 1, speaker_id: "h1-tx-c", speaker_name: "H1 TX C", input: "raw/tx-c"}
  - {track_id: "tx-d", receiver_id: "rx-b", receiver_channel: 2, speaker_id: "h1-tx-d", speaker_name: "H1 TX D", input: "raw/tx-d"}
  - {track_id: "tx-e", receiver_id: "rx-c", receiver_channel: 1, speaker_id: "h1-tx-e", speaker_name: "H1 TX E", input: "raw/tx-e"}
  - {track_id: "tx-f", receiver_id: "rx-c", receiver_channel: 2, speaker_id: "h1-tx-f", speaker_name: "H1 TX F", input: "raw/tx-f"}
sync_qa:
  enabled: true
  window_s: 30
  max_lag_ms: 100
  drift_warn_ms: 5
  offset_warn_ms: null
  min_correlation: 0.5
```

Omitted sections use the checked-in defaults. Add the normal ASR context/glossary only if it
is part of the intended real-session configuration.

## Phase 7 — baseline processing

From the repository's `direnv` environment:

```bash
uv run dnd-audio doctor
uv run dnd-audio inspect "$H1_SESSION_DIR"
uv run dnd-audio ingest "$H1_SESSION_DIR"
uv run dnd-audio process "$H1_SESSION_DIR"
sha256sum --check "$H1_SESSION_DIR/work/raw-sha256.before.txt"
```

Do not process alongside a heavy GPU workload. Preserve the complete report, manifest,
activity graph, transcript records, public transcript, mix, ASR cache, and raw hash guard.

The baseline succeeds only if:

- all six configured tracks are present and usable;
- the manifest records `container.sample_count_agrees` for the real PCM files (OQ-011);
- the timeline contains the logged staggered starts, `tx-f` power-cycle gap, and both clap
  landmarks;
- cross-receiver sync QA is measured or explicitly low-confidence, never silently absent;
- both `orig` and `edit` candidates are inventoried and `orig` is selected according to the
  configured policy;
- the raw hash check passes.

## Phase 8 — transcript controls to score

Do not tune thresholds in the H1 session. Record evidence and open a separate software
milestone if a semantic change is justified.

### Leading ownership grace (OQ-027)

In isolated working copies, keep the baseline activity graph and cached ASR response documents
fixed while comparing:

```text
transcript.leading_ownership_grace_ms: 0
transcript.leading_ownership_grace_ms: 20
transcript.leading_ownership_grace_ms: 100
```

For every hard-onset phrase, record whether its first word appears on the logged direct track
and on weaker tracks. Confirm activity, request identities, ASR cache documents, and mix remain
unchanged. Inspect candidate/piece-specific original and effective ownership in
`work/transcript-records.json`; do not score only the aggregate dropped-pair count.

### Exact-short and different-word overlap (OQ-018)

- Both human voices' simultaneous `Okay` must remain represented unless acoustic evidence can
  independently prove one event. Record granular records separately from public turns.
- Both different-word utterances must survive and be marked as overlap.
- Do not use this fixture's earlier same-human mic rounds as evidence that two different voices
  should collapse.

### Presentation pauses (OQ-018)

Compare the measured 250, approximately 330, and 500 ms word gaps with:

- the granular records;
- `transcript.json` public turns;
- `transcript.md` public turns;
- source-record and source-candidate lineage.

The approximately 330 ms intended turn is the principal join control. The 500 ms distinct
statements are the negative control. Actual measured gaps, not stopwatch targets, determine the
comparison.

## H1 evidence map

| requirement/question | evidence produced by this runbook |
| --- | --- |
| OQ-001 metadata layout | all six untouched files, raw FFprobe sidecars, generic RIFF inventory |
| OQ-002 repeated `TX##` identity | complete filenames plus receiver/track directory map |
| OQ-003 counter across power cycle | `tx-f` filenames before and after the logged full power cycle |
| OQ-004 recorder-domain reference | multiple takes in one unchanged receiver power cycle |
| OQ-005 private/iXML chunks | six generic RIFF inventories and FFprobe sidecars |
| OQ-007 `orig`/`edit` pairing | dual-file outputs with unchanged paired filenames |
| OQ-011 exact sample count | manifest `container.sample_count_agrees` on every real file |
| OQ-012 third receiver jam/hold | filmed displays after jam/after take plus file/audio cross-check |
| OQ-015 display zero vs wall clock | receiver display table filmed beside zoned wall clock |
| OQ-017 real-table bleed | six direct-source rounds, distributed observers, and two real voices |
| OQ-018 duplicate/overlap/presentation | exact `Okay`, different words, and 250/330/500 ms controls |
| OQ-023/OQ-025 per-session jam check | ten-second preflight take with shared clap and `sync_qa` |
| OQ-027 leading-word recovery | six hard onsets, clean/quick handoff, and fixed-response 0/20/100 comparison |

## Evidence to retain and evidence safe to commit

Retain privately:

- every original audio file and raw hash list;
- phone video/photos of receiver displays and wall clock;
- the unsanitized recording log;
- complete pipeline working/output artifacts;
- receiver/transmitter serial numbers.

Safe to commit after review and sanitization:

- a fixture note under `docs/fixtures/` with measurements and deviations;
- sanitized FFprobe JSON;
- generic RIFF chunk inventories;
- filename grammar and counter observations without private paths or serial numbers;
- transcript/diagnostic aggregates that quote no private conversation;
- OQ/ADR/charter updates and regression fixtures derived from metadata, not audio.

Do not commit audio, host-specific paths, serial numbers, raw reviewer transcripts, model
weights, or tokens. A tiny audio excerpt still requires explicit owner approval.

## Final capture checklist

- [ ] Nine durable labels and the receiver/channel map were verified.
- [ ] All three receiver rates matched.
- [ ] `rx-a` jammed `rx-b` and `rx-c`; displays matched before disconnect.
- [ ] Receiver displays were filmed against zoned wall clock after jam and after the main take.
- [ ] Format, storage, battery, and recording indicator were checked on all six transmitters.
- [ ] Dual-file mode produced at least one real `orig`/`edit` pair.
- [ ] The ten-second jam check passed before the main take.
- [ ] Main transmitter starts were staggered and logged.
- [ ] All six labeled solo/direct-source lines were recorded.
- [ ] Clean-pause and quick-handoff hard onsets were recorded with intended tracks.
- [ ] 250/330/500 ms target-pause controls were recorded and logged.
- [ ] Two different human voices said exact `Okay` simultaneously.
- [ ] The same two voices spoke different sentences simultaneously.
- [ ] `tx-f` was fully power-cycled and recorded before and after.
- [ ] Distinctive three-clap patterns exist near the start and end.
- [ ] Complete filenames, surprising displays, repeats, mistakes, and deviations were logged.
- [ ] Every file was copied unchanged, directory listings retained, and raw hashes recorded.
- [ ] `inspect`, `ingest`, and baseline `process` completed without modifying raw files.
- [ ] The raw hash guard passed after processing.
