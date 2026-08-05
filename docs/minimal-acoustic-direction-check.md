# Minimal two-person acoustic direction check

This is the protocol used for the completed 2026-08-04 minimal acoustic capture. It supplied
two-person speech, deliberate overlap, exact-short controls, handoffs, and a hard marker
false-positive corpus. The separate sample probe, jam verification, and six-transmitter marker
bench settled the hardware breadth; ADR-0043 records why no controlled follow-up is planned.

The burden on the second person is about five minutes: wear one transmitter, read a few lines,
and speak on silent visual cues. There is no paper log, receiver jam, wall-clock observation,
mid-session file transfer, or headless-server work.

## What this capture decides

The present pipeline compares per-track VAD candidates and later collapses duplicate text. The
alternative is to infer one or more latent acoustic events jointly across channels, identify
the active wearer(s), select source audio, and run ASR afterward.

The full hypothesis, candidate stages, alternatives, invariants, and decision outcomes are
recorded in the
[event-first transcript architecture spike](plan/EVENT-FIRST-ARCHITECTURE-SPIKE.md). That note
is exploratory, not an approved production design.

This capture supplies the missing contrasts:

- one voice saying `Okay`, heard directly and as bleed;
- two different voices saying the same `Okay` at nearly the same time;
- two different voices saying different words simultaneously;
- ordinary solo speech from each direct transmitter;
- a quick speaker handoff;
- `Okay` followed by a deliberate long rhetorical pause and a continuation.

If acoustic correlation, channel dominance, timing, and speaker features distinguish those
cases, the next software direction should be event/speaker inference before ASR. If they do
not, the conservative transcript plus an editorial cleanup layer remains the safer design.

## Minimum equipment and layout

Use four transmitters. Receivers need not be jammed for this experiment; the opening and
closing clap patterns provide acoustic alignment.

- Person One wears `tx-a` at the normal chest position.
- Person Two wears `tx-b` at the normal chest position.
- Put `tx-c` on the table approximately halfway between them.
- Put `tx-d` at another plausible player position, farther from both direct speakers.

All four transmitters record continuously. Take one phone photo of the layout if convenient;
the spoken slate below is sufficient if not. Do not move transmitters during the take.

If four devices are still too tedious, three are usable: omit `tx-d`. Two direct tracks plus
one observer are the minimum that preserve the central contrast.

## Preparation

1. Confirm the four labels.
2. Confirm battery, storage, and internal recording format.
3. Start all four transmitters and confirm their recording indicators.
4. Stand or sit in normal tabletop positions.
5. Person One operates a silent finger countdown. Do not count aloud during overlap events.

No exact timing is required. Leave approximately two seconds of silence after each spoken
event slate and between trials. If someone laughs, starts late, or misspeaks, continue; say
**“repeat”** and do that event once more. The audio itself is the log.

## Spoken script

Bold quoted text is spoken. Bracketed directions are silent actions.

### 1. Self-describing slate and alignment

Person One says:

**“Minimal acoustic direction check. Person One is wearing transmitter A. Person Two is
wearing transmitter B. Transmitter C is the middle observer. Transmitter D is the far
observer.”**

[Two seconds of silence.]

Make three hand claps: clap, short pause, clap, longer pause, clap. Do not clap beside a
transmitter.

[Two seconds of silence.]

### 2. Solo voice controls

Person One says:

**“Event one, Person One solo.”**

[Two seconds of silence.]

**“Okay. The red dragon waits beside the northern gate.”**

[Two seconds of silence.]

Person Two says:

**“Event two, Person Two solo.”**

[Two seconds of silence.]

**“Okay. The blue lantern hangs above the southern door.”**

[Two seconds of silence.]

These give each direct wearer a longer clean voice region and one isolated short `Okay` while
the observer tracks record bleed.

### 3. Repeated one-source short-word controls

Person One says:

**“Event three, Person One says Okay three times.”**

[Two seconds of silence, then Person One says `Okay` once on each of three silent finger cues,
with about two seconds between cues. Person Two remains silent.]

Person Two says:

**“Event four, Person Two says Okay three times.”**

[Repeat the same three-cue pattern with only Person Two speaking.]

### 4. Two-source exact-short overlap

Person One says:

**“Event five, both people say Okay together three times.”**

[Two seconds of silence. On each of three silent finger cues, both people say exactly
`Okay`. Leave about two seconds between trials.]

Three trials matter more than precise simultaneity: ordinary human start variation becomes
useful evidence rather than a failed take.

### 5. Two-source different-word overlap

Person One says:

**“Event six, both people speak different sentences together.”**

[Two seconds of silence, then a silent finger cue.]

- Person One: **“Red dragons guard the northern gate.”**
- Person Two: **“Blue goblins cross the southern bridge.”**

[Two seconds of silence. Repeat once on another silent cue.]

### 6. Quick handoff

Person One says:

**“Event seven, quick handoff from Person One to Person Two.”**

[Two seconds of silence.]

- Person One: **“I finish beside the old gate.”**
- Person Two begins immediately after Person One: **“Pick the bright token before we leave.”**

Do not force overlap; an ordinary fast conversational handoff is the target.

### 7. Reproduce the `Okay ... now` case

Person One says:

**“Event eight, one sentence with a long rhetorical pause.”**

[Two seconds of silence.]

Person One says **“Okay,”** waits about three seconds, then continues:

**“now I am talking into transmitter A.”**

This tests the difference between an acoustic turn boundary and an editorial sentence join.

### 8. End alignment and room tone

[Two seconds of silence.]

Person One says **“End of minimal acoustic direction check.”**

Make the same three-clap pattern used at the start, then leave five seconds of room tone.
Stop all four transmitters.

## One transfer after the other person is done

The second person is finished as soon as recording stops.

On any convenient computer, copy the one continuous original WAV from each transmitter into
four plainly named directories while preserving the DJI filenames and file bytes:

```text
minimal-direction-check/
  tx-a/<original filename>
  tx-b/<original filename>
  tx-c/<original filename>
  tx-d/<original filename>
```

Upload that directory to the headless machine once, using the operator's normal `rsync`, `scp`,
or file-transfer method. Do not trim, normalize, rename, or convert the WAV files. The outer
directory names carry the transmitter mapping.

There is no need to create `session.yaml`, run the pipeline, or analyze anything during the
capture. After the directory reaches the project machine, give the agent its path. The agent
can hash the originals, construct an isolated session, align the claps, run the current
pipeline, and perform waveform/event comparisons without further work from either speaker.

## Analysis questions

The follow-up analysis should answer only these questions before proposing more capture work:

1. Do the three one-speaker `Okay` trials form one coherent source observed across tracks?
2. Do the three two-speaker `Okay` trials show two direct-source winners or distinguishable
   speaker/acoustic components?
3. Does calibrated cross-track lag add information beyond level and correlation?
4. Can session-local speaker features distinguish Person One from Person Two on these short
   events when supported by their longer solo regions?
5. Would an event-first representation yield one event for solo bleed and two events for
   simultaneous speakers without consulting ASR text?
6. Does Qwen preserve both different-word overlap utterances after source selection?
7. Should the three-second `Okay ... now` gap remain acoustically separate and be joined only
   in an editorial view?

The result is architecture-direction evidence, not a new production threshold. Do not modify
activity defaults, transcript semantics, or the mix merely to make this tiny corpus look
perfect. M11 may revisit the event-first hypothesis only if live Session Zero exposes a
concrete limitation in the conservative baseline.
