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
- [ ] Same frame rate configured on all three receivers.
- [ ] Kits kept as three independent two-transmitter groups (a group holds only four).
- [ ] Jam: receiver A LTC out → receiver B LTC in, Sync, disconnect. Then A → C,
      Sync, disconnect.
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

## Completion gate

- [ ] Every `OQ-` entry with `Needs: H1` is marked `answered` (with the evidence) or
      explicitly re-scoped: **OQ-001, OQ-002, OQ-003, OQ-004, OQ-005, OQ-007,
      OQ-011, OQ-012, OQ-015**.
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
