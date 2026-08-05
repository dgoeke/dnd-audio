# H1 — Real DJI hardware fixture (parallel track)

**Status:** not started
**Depends on:** a physical recording session, not on code. Start acquiring during M1.
**Spec sections:** Tests and acceptance criteria (recommended real fixture); Owner
notes 1, 2, 5

## Goal

A ~2-minute real recording from all six labeled transmitters and all three
synchronized receivers, used to settle every DJI-metadata assumption the synthetic
fixtures cannot. This validates metadata and synchronization plumbing — **not**
multi-hour clock stability, which is H2.

## Recording recipe (for the owner)

The printable, executable version of this checklist is
[`docs/H1-two-person-recording-runbook.md`](../../H1-two-person-recording-runbook.md). It
covers all six transmitters and all three receivers with one operator and one other person,
including the private event log, timed spoken script, immediate jam check, power cycle, transfer
guard, and post-capture scoring controls.

If the immediate goal is only to choose between the current candidate/deduplication architecture
and joint acoustic-event/speaker inference, use the deliberately non-H1-closing
[`docs/minimal-acoustic-direction-check.md`](../../minimal-acoustic-direction-check.md) first.
It needs one short continuous take and no capture-time paperwork or intermediate analysis.

- [ ] Durable labels on receivers `rx-a`–`rx-c` and transmitters `tx-a`–`tx-f`.
- [ ] **Timecode frame rate the same on all three receivers.** 30 fps is fine. An earlier
      version of this recipe asked for 60 fps to halve the quantum from 1600 samples to 800;
      **that was measured on 2026-08-03 and the setting does not reach the file** — a receiver
      set to 60 wrote `TIMECODE_RATE 30/1` on 1600-sample boundaries exactly like the 30 fps
      receiver beside it (**OQ-024**). 33.3 ms is the floor, and it is inside the error budget
      (**OQ-025**). Matching the rates is hygiene, not a dependency: the same capture had two
      receivers on different rates and the jam held regardless.
- [ ] Kits kept as independent groups. **A group holds up to four transmitters and eight
      receivers**, so six transmitters force at least two groups — but receivers *within* a
      group auto-sync timecode wirelessly, at intervals, with no user action. Only the
      cross-group boundary needs a jam.
- [ ] Jam: receiver A LTC out → receiver B LTC in, Sync, disconnect. Then A → C,
      Sync, disconnect. **Confirm on each display that the timecode matches A's before
      disconnecting.** DJI also documents jamming from an external generator (Deity TC-1,
      Tentacle Sync), which is worth preferring over a daisy chain if one is available.
- [ ] **Verify the jam reached the files before recording anything that matters.**
      **OQ-023 is answered** — on 2026-08-03 a jam propagated into `bext.time_reference` and
      placed two receivers to within one frame, so the strategy is sound. What is *not*
      established is that it works every time, and a failed jam is invisible at capture and at
      ingest. So this stays in the recipe as a per-session check.
      **Do not compare implied epochs using `bext.origination_time`** — the same capture
      showed two receivers' wall clocks 48.7 s apart while their timecode agreed to under a
      frame (**OQ-004**). The valid check is the audio: record ten seconds on every receiver
      simultaneously with one sharp shared transient, then confirm the offset
      `time_reference` predicts matches the offset the audio shows. That is the comparison
      `session.sync_qa` should grow (**OQ-023**, **OQ-025**).
- [ ] Record the displayed timecode and rate on all three receivers after the
      procedure, **against wall-clock time** — this is the only evidence for OQ-012 and
      OQ-015 and it cannot be recovered later. OQ-015 asks whether `00:00:00:00` is real
      midnight; at a fractional non-drop rate a timecode day is 86 486.4 s rather than
      86 400, so a session mixing BWF and timecode evidence rests on the answer (ADR-0009).
- [ ] Confirm 32-bit-float internal recording, storage, and battery on all six
      transmitters; confirm the recording indicator on each after starting.
- [ ] Start the transmitters a few seconds apart.
- [ ] Each wearer states their transmitter label, then speaks alone for several seconds.
- [ ] Each wearer then repeats a short, logged phrase beginning with a hard onset (`Testing`,
      `Pick`, `Take`, or equivalent), with the intended `track_id` written in the recording
      log. Include both a clean pause before the phrase and a quick handoff from the preceding
      speaker. This distinguishes a direct-source opening word from the same word in another
      lav's padding; a dropped-word count alone cannot do that (**OQ-017**, **OQ-027**).
- [ ] One deliberate two-person overlap. Include a logged exact short response such as `Yes`
      or `Okay` spoken by both people at nearly the same time, plus one different-words
      overlap. Preserve the granular records and public turns separately when evaluating the
      result: the exact match must not disappear merely because correlation exists
      (**OQ-018**, ADR-0033).
- [ ] One wearer delivers two distinct statements on the same transmitter with controlled
      pauses on both sides of 350 ms, and one intended turn is naturally split by a pause near
      320--350 ms. Log the words and pause targets. This tests M9's presentation-only join
      without treating a shared ASR batch as a conversational boundary (**OQ-018**, ADR-0034).
- [ ] Turn one transmitter off, wait several seconds, turn it back on, record again.
- [ ] A distinctive three-clap pattern near the start and near the end. If the separately
      chartered OQ-025 chirp generator **and matched-filter detector** have landed and passed
      their bench gate first, the same prepared three-chirp marker may replace each clap
      pattern. Play it from one fixed central table position and keep the LTC jam; a generated
      sound verifies the jam but does not place a restarted file that missed the marker.
- [ ] Export both `orig` and `edit` if dual-file mode is enabled.
- [ ] Note anything surprising the receivers displayed.

## Partial evidence received — 2026-08-02 sample probe

Not the H1 fixture, and it does not close this milestone. Four transmitters (not six), two
receivers (not three), ~47 s, no LTC jam, no power cycle, no clap pattern, no `edit` files,
and the operator held one mic at a time in front of their mouth with the others about two
feet away rather than wearing six around a table. Recorded here because **four of the six
questions this milestone gates turned out to be answerable from metadata alone**, and because
one of the answers invalidates an assumption M1 and M2 are built on.

Ledger effect — details and measurements live in each entry, not here:

| question | outcome |
| --- | --- |
| **OQ-001** metadata layout | **answered** — `bext` + `iXML` + `cue`(0 points) + `PAD`; no `INFO`/`ISMP` |
| **OQ-002** `TX##` uniqueness | **answered, assumption confirmed** — two receivers each produced `TX01`/`TX02`; INV-11 stands |
| **OQ-004** `time_reference` semantics | **answered, assumption false on both halves** — not midnight-relative, and frame-quantized to 33.3 ms |
| **OQ-005** private/iXML chunks | **answered** — none; the iXML duplicates `bext` and adds only the 30/1 timecode rate |
| **OQ-003** filename grammar | grammar confirmed; the `MIC###` counter's behaviour across a power cycle still open |
| **OQ-007** dual-file `orig`/`edit` | `orig` is **not** always 32-bit float; `orig`/`edit` pairing untested |
| **OQ-017** bleed thresholds | first real measurements, from a harder geometry than a table — and M6b added a second symptom, `vad.pad_ms` clipping utterance-opening words |
| **OQ-012**, **OQ-015** | untouched — both need receiver displays read against wall clock. M6b's capture is *reported* to have had a failed jam between receivers, which is consistent with what OQ-012 records but is not the measurement |

Two things the pipeline did on real files, worth knowing before the real fixture arrives:

- **`inspect` handled all four unchanged**, and the manifest named the strategy, the evidence,
  and the assumption *by OQ id* — which is what turned "acquire a fixture" into "read one
  manifest". That design paid for itself here.
- **`ingest` refused the two 24-bit files** (OQ-007), and **`activity` ran end to end on the
  float pair with the real Silero release** — the first real speech this project has seen.

### What the real fixture must still deliver

Everything in the recipe above, and these in particular, because the sample probe showed why
they matter rather than merely that they were on a list:

- **A second recording from the same power-on cycle**, to confirm OQ-004's epoch reading.
  One session cannot distinguish "since power-on" from another fixed epoch.
- **The receiver displays read against wall clock** — still the only evidence for OQ-012 and
  OQ-015, and still unrecoverable afterwards.
- **A power cycle**, for OQ-003's counter.
- **All six transmitters confirmed on the same recording format**, per the recipe — and note
  that the pipeline should tolerate a mismatch rather than fail, which OQ-007 now records.
- **Speech whose opening consonant a VAD can be late on** — added by M6b, which found the
  first real defect only a real model on real speech could have surfaced. With the adapter
  running, the model heard `'Testing a first transmitter…'` and the transcript recorded
  `'a first transmitter…'`: the aligner places "Testing" 50 ms before the VAD candidate's
  ownership interval begins, and M4's rule — a word belongs to the interval containing its
  **start** — correctly drops it. Five of eleven retained segments lost their opening word
  this way. `activity.vad.pad_ms` = 30 is M3's number, chosen against synthetic audio and
  registered under **OQ-017**; 47 seconds of one operator testing microphones is not evidence
  to retune a detector on, and a real table is. **The symptom to look for is a transcript
  quietly missing the first word of an utterance** — nothing raises, nothing warns, and the
  text reads as plausible prose.
- **The M9 transcript-only control.** Keep the activity graph and cached ASR responses fixed
  while comparing leading ownership grace at 0, 20 and 100 ms. Attribute each recovered word
  to the logged direct wearer rather than scoring drops alone, and confirm the activity graph,
  request identities and mix do not change. Inspect the piece-specific original/effective
  ownership lineage in the granular records (**OQ-027**, ADR-0033).

## Partial evidence received — 2026-08-03 jam verification capture

Not the H1 fixture either: four transmitters, two receivers, ~47 s, one operator. But it was
recorded to settle one specific question and it did, along with three others. Full evidence in
`docs/fixtures/2026-08-03-jam-verification.md`.

| question | outcome |
| --- | --- |
| **OQ-023** does a jam reach `bext.time_reference` | **answered — yes.** Two receivers started 5.28 s apart; their timecode agrees on the offset to **17–30 ms**, inside one 30 fps frame. Cross-receiver alignment is free from metadata |
| **OQ-024** does the frame-rate setting reach the file | **answered — no.** A receiver set to 60 fps wrote `30/1` on 1600-sample boundaries. The 60 fps instruction is retracted from this recipe |
| **OQ-006** sample-clock drift | **first measurement — ≈1 ppm**, bounded ±3 ppm on a 30 s baseline. Rules out the catastrophic case; H2 still owns the long baseline |
| **OQ-012** jam holds across receivers | **answered for two of three.** The third is still owed |
| **OQ-004** wall clock as a cross-receiver hint | **killed.** The two receivers' `origination_time` implied epochs **48.7 s apart** while their timecode agreed to under a frame |
| **OQ-003** `MIC###` counter | differs per *transmitter* within one receiver group (005 vs 003), so the counter is per transmitter |

**What this changes for this milestone.** The single highest-value unknown named at the bottom
of this charter is no longer unknown: DJI's metadata carries a usable shared origin, and M1/M2's
timecode plumbing rests on something real. What remains here is breadth — the third receiver,
six transmitters, a power cycle, `edit` files, a real table — not the existential question.

**Two things the recipe now inherits.** Never use `origination_time` as a cross-receiver
anchor. And the jam's *outcome* is still invisible without an audio check, so the ten-second
verification stays in the recipe permanently rather than being a one-off (**OQ-025**).

## Completion gate

- [ ] Every `OQ-` entry with `Needs: H1` is marked `answered` (with the evidence) or
      explicitly re-scoped: **OQ-001, OQ-002, OQ-003, OQ-004, OQ-005, OQ-007,
      OQ-011, OQ-012, OQ-015**.
      **OQ-023 and OQ-024 are already answered** by the 2026-08-03 jam verification capture
      (`docs/fixtures/2026-08-03-jam-verification.md`), which also gave OQ-006 its first
      measurement. **OQ-012 is answered for two of the three receivers** — this fixture owes
      the third.
- [ ] A fixture note in `docs/` documents the discovered filename grammar, metadata
      layout, and timecode behavior, including what differed from the assumptions.
- [ ] Sanitized `ffprobe` JSON and the generic RIFF chunk inventory are committed.
      **No audio committed** unless the owner explicitly approves a tiny sample.
- [ ] Every assumption the fixture disproves is fixed in M1/M2 **with a test built
      from the real metadata**, and the affected charters/ADRs are updated.
- [ ] `inspect` and `ingest` run cleanly on the real fixture, and the reconstructed
      timeline agrees with the recording log (start offsets, the power-cycle gap,
      the two clap positions).
- [ ] The logged hard onsets, simultaneous exact-short overlap and controlled pauses are
      compared against both granular transcript records and coalesced public turns. Any M9
      threshold change is deferred to H2 or a separate software milestone; H1 records the
      evidence without hiding transcript-semantic work inside this hardware fixture.
- [ ] The full gate still passes; synthetic tests are not weakened to accommodate
      real data.

## Explicitly not in this milestone

- Drift measurement over a long session. That is H2.
- Threshold tuning against real speech quality — that needs a real session, not a
  two-minute fixture.

## Known risks and open questions

- Was the single highest-value unknown in the project. **Substantially de-risked on
  2026-08-03**: the jam propagates into the files (OQ-023) and the clocks are stable
  (OQ-006), so M1's use of DJI metadata is no longer a guess. What is left is breadth and
  the operational questions — the third receiver, six transmitters, a power cycle, `edit`
  files, and real speech at a real table.
- If the metadata differs substantially from the BWF assumption, expect real work
  in M1's strategy chain rather than a one-line fix. Budget for it.
