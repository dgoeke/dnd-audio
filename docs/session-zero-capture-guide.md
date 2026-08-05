# Live Session Zero capture guide

This is the normal-campaign capture procedure. Session Zero is real play with all players, not
a hardware experiment. The goal is to protect an irreplaceable recording, run the finished
pipeline at its current defaults, and leave enough evidence for M11 to tune only what ordinary
play shows needs tuning.

## Before people arrive

1. Confirm durable receiver labels `rx-a` through `rx-c` and transmitter labels `tx-a`
   through `tx-f`; prepare the wearer-to-track roster for `session.yaml`.
2. On every transmitter, confirm internal recording format, free storage, battery/external
   power, loop recording disabled, and the expected physical label.
3. Jam receiver A's LTC output to receiver B's input, then A to C. Confirm matching displayed
   timecode and rate after each sync, disconnect the cable, and keep receivers powered.
4. Run `dnd-audio marker build <prepared-assets-directory>` before the room is occupied and
   load the standalone offline player on the intended phone. The generated WAV and embedded
   player are byte-identical.
   Keep the three-clap pattern as the fallback.
5. Choose one fixed central phone position, orientation, and media-volume step for the opening
   and closing marker. Record those facts in the capture notes. The marker verifies the jam; it
   never replaces timecode or corrects the timeline.

## Start of the live recording

1. Start internal recording on all six transmitters and visually confirm every indicator.
2. Have each wearer state their name and transmitter label once as a natural roster slate.
3. Play marker v1 once from the logged fixed position, or make the fallback three-clap
   pattern. Note the approximate wall time and any capture irregularity.
4. Proceed with Session Zero normally. Do not stage power cycles, dual-file tests, controlled
   pauses, scripted overlap, or microphone-count experiments. If a transmitter must restart
   for an operational reason, log it and continue; timecode places the new file.

## End and transfer

1. If the phone and lav geometry can genuinely be restored to the opening arrangement, play
   marker v1 again with the same settings and log that fact. Otherwise still play it for jam
   QA, but record that geometry changed; ADR-0040 forbids calling the difference recorder
   drift.
2. Leave several seconds of room tone, stop every transmitter, and confirm all six recordings
   exist before packing the hardware.
3. Copy each transmitter's original WAV bytes into its authoritative `raw/tx-*` directory.
   Do not rename, normalize, trim, or otherwise modify anything under `raw/`.
4. Build `session.yaml` from the physical roster, run `dnd-audio inspect <session>`, then
   immediately run `dnd-audio archive upload <session>` followed by
   `dnd-audio archive verify --session-id <session-id>`. Verification is a full readback and
   is what establishes the backup.
5. Run `dnd-audio process <session>` with production defaults and preserve the baseline
   configuration, report, transcript records, transcript views, and MP3 before M11 evaluates
   tuning. Then run `dnd-audio marker analyze <session>` with
   `--start-window-s 120 --end-window-s 120` to record jam QA and differential arrival without
   changing the timeline.

The first pass is evidence, not a demo to optimize live. A plausible-looking transcript can
still lose an opening word, and a lower duplicate count can still mean deleted speech. Review
the transcript, activity graph, mix, provenance, and diagnostics together.
