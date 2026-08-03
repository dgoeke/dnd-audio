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

- [ ] Durable labels on receivers `rx-a`–`rx-c` and transmitters `tx-a`–`tx-f`.
- [ ] **Timecode frame rate set to 60 fps on all three receivers.** Same rate everywhere is
      the requirement; 60 is the choice, because DJI's written reference is quantized to one
      frame and 60 is the finest rate the Mic 3 offers. It halves the quantum from 1600
      samples to 800 — 33.3 ms to 16.7 ms. The sample probe was recorded at 30/1 (**OQ-004**).
- [ ] Kits kept as independent groups. **A group holds up to four transmitters and eight
      receivers**, so six transmitters force at least two groups — but receivers *within* a
      group auto-sync timecode wirelessly, at intervals, with no user action. Only the
      cross-group boundary needs a jam.
- [ ] Jam: receiver A LTC out → receiver B LTC in, Sync, disconnect. Then A → C,
      Sync, disconnect. **Confirm on each display that the timecode matches A's before
      disconnecting.** DJI also documents jamming from an external generator (Deity TC-1,
      Tentacle Sync), which is worth preferring over a daisy chain if one is available.
- [ ] **Verify the jam reached the files before recording anything that matters
      (OQ-023).** Immediately after jamming, record ten seconds on every receiver
      simultaneously, then run `dnd-audio inspect` on the result and compare the epoch each
      receiver's file implies — `time_reference ÷ 48000` subtracted from
      `bext.origination_time`. **They must agree.** This is the assumption the entire
      cross-receiver strategy rests on and it has never been checked; the display agreeing
      does not establish it, because the pipeline never sees the display. Five minutes here
      decides whether the session's timing is usable at all.
- [ ] Record the displayed timecode and rate on all three receivers after the
      procedure, **against wall-clock time** — this is the only evidence for OQ-012 and
      OQ-015 and it cannot be recovered later. OQ-015 asks whether `00:00:00:00` is real
      midnight; at a fractional non-drop rate a timecode day is 86 486.4 s rather than
      86 400, so a session mixing BWF and timecode evidence rests on the answer (ADR-0009).
- [ ] Confirm 32-bit-float internal recording, storage, and battery on all six
      transmitters; confirm the recording indicator on each after starting.
- [ ] Start the transmitters a few seconds apart.
- [ ] Each wearer states their transmitter label, then speaks alone for several seconds.
- [ ] One deliberate two-person overlap.
- [ ] Turn one transmitter off, wait several seconds, turn it back on, record again.
- [ ] A distinctive three-clap pattern near the start and near the end.
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

## Completion gate

- [ ] Every `OQ-` entry with `Needs: H1` is marked `answered` (with the evidence) or
      explicitly re-scoped: **OQ-001, OQ-002, OQ-003, OQ-004, OQ-005, OQ-007,
      OQ-011, OQ-012, OQ-015, OQ-023**.
- [ ] **OQ-023 answered first, and separately.** It is the only one here that can be settled
      in five minutes with no session at all, and it decides what OQ-004's rework has to
      achieve — so it should be answered *before* the fixture is recorded rather than
      alongside it. If a jammed display does not reach `bext.time_reference`, cross-receiver
      alignment has to come from the audio and the recipe needs a shared transient loud
      enough for `session.sync_qa` to correlate on.
- [ ] A fixture note in `docs/` documents the discovered filename grammar, metadata
      layout, and timecode behavior, including what differed from the assumptions.
- [ ] Sanitized `ffprobe` JSON and the generic RIFF chunk inventory are committed.
      **No audio committed** unless the owner explicitly approves a tiny sample.
- [ ] Every assumption the fixture disproves is fixed in M1/M2 **with a test built
      from the real metadata**, and the affected charters/ADRs are updated.
- [ ] `inspect` and `ingest` run cleanly on the real fixture, and the reconstructed
      timeline agrees with the recording log (start offsets, the power-cycle gap,
      the two clap positions).
- [ ] The full gate still passes; synthetic tests are not weakened to accommodate
      real data.

## Explicitly not in this milestone

- Drift measurement over a long session. That is H2.
- Threshold tuning against real speech quality — that needs a real session, not a
  two-minute fixture.

## Known risks and open questions

- The single highest-value unknown in the project. An architecture prompt cannot
  substitute for it; everything M1 does with DJI metadata is a guess until this lands.
- If the metadata differs substantially from the BWF assumption, expect real work
  in M1's strategy chain rather than a one-line fix. Budget for it.
