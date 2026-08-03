# ADR-0022 — The gain envelope: a control grid, two floors, and an invariant over what reaches a sample

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M5

## Context

The spec asks for "continuously smoothed gain envelopes" from the VAD/attribution decisions,
with solo speech favouring one lav and strongly attenuating the other five, genuine overlap
keeping each active person audible "using equal-power or otherwise bounded gain sharing",
silence blending room tone "without allowing six tracks of noise to add coherently", and
"short attack and longer release/crossfade times so words are not clipped and channel changes
do not click or pump". It names "a Dugan-style normalized gain-share" as a good baseline.

Then it asks for something harder, and M5's charter repeats it as the milestone's central
risk: **test the automixer at the gain-envelope level, with explicit configurable tolerances**
— because decoded loudness is not evidence of correct channel selection, and a mix that picks
the wrong speaker passes every loudness test there is.

That turns the design question around. The envelope is not merely a mechanism that has to
sound right; it is a mechanism whose gate criteria have to be *provable*. A rule that
usually produces 20 dB of dominance on this fixture is not the same thing as a rule that
cannot produce less than 20 dB, and only the second survives a session nobody has recorded
yet (OQ-019).

M3 froze the input (ADR-0012): each track's retained candidates with a `score_permille`, and
`speech_reference_mbfs` as the per-track voice level. The graph deliberately contains no gain.

## Decision

### A 1 kHz control grid, with everything about it validated rather than assumed

Gains are computed per **control frame** and linearly interpolated to samples, so the applied
gain is continuous by construction rather than by a smoothing filter's good behaviour.
`mix.envelope.control_rate_hz` defaults to 1000 — 48 samples per frame. 100 Hz was the first
choice and is wrong: a 10 ms attack would be a single frame, which is no slew limit at all.

Three properties are checked when the configuration loads, because each is a way the grid
stops being exact:

1. `control_rate_hz` divides `CANONICAL_SAMPLE_RATE`, so samples-per-frame is an integer.
2. `attack_ms` and `release_ms` land on whole control frames (`ms · rate % 1000 == 0`).
   Divisibility of the *rate* does not give this: 800 Hz divides 48 000, and an 11 ms attack
   is 8.8 frames.
3. The configured dominance margin is achievable from the floors and the clamp — below.

A candidate `[start_sample, end_sample)` is active on control frames
`[start // spf, ceil(end / spf))`. The start floors and the end **ceils**, so the frame
interval always *covers* the sample interval — the same rule as
`resample.to_derivative_interval`, for the same reason M2's closeout gives: rounding both
ends alike shrinks a speech region, which is how a word loses its first phoneme. The
session's final control frame may cover fewer than `spf` samples; interpolation is clipped to
`duration_samples`, and every duration in this milestone is compared as integer samples.

### Weights with two floors, because one floor cannot prove the gate criteria

Per track, per frame:

```
active = 1 if a retained candidate covers this frame else 0
weight = active ? min_active_share + (1 - min_active_share) * score : 0
weight = max(weight, room_tone_share)
```

**Suppressed candidates sit at the room-tone share. Every retained candidate — `ambiguous`
included — is eligible.** That is the whole of M5's reading of the graph. `ambiguous` marks a
candidate the track-level veto kept *because a lav hearing its wearer at that wearer's normal
level is probably not hearing someone else* (ADR-0014); it is the least obvious bleed case
there is, not the most, and M3's and M4's closeouts both say so. The gate's "obvious
correlated bleed is not promoted on two channels simultaneously" is about the **suppressed**
ones. M5's charter said the opposite and has been corrected.

The two floors are what make the criteria bounds rather than observations. An active channel
never falls below `min_active_share` however badly it scored; an inactive one never falls
below `room_tone_share`. Worst-case solo dominance is therefore
`20·log10(min_active_share / room_tone_share)`, computable at configuration load. A single
floor makes dominance proportional to the winner's score, and a genuinely-speaking candidate
that happened to score 0.1 would be mixed 14 dB above the room rather than 40.

### Smoothing is a slew-limited linear ramp, not a one-pole

Rising by at most `1/attack_frames` per frame, falling by at most `1/release_frames`. An
exponential one-pole never reaches its target and its "attack time" is a time constant rather
than a bound, which makes the gate's "do not exceed configured attack, release, or
maximum-slew limits" unassertable — there is no frame at which it is true or false.

`attack_ms` defaults to 10, a third of `activity.vad.pad_ms`, so the ramp completes inside
the padding the candidate already carries and the channel is open before the word starts. A
test pins that relationship (OQ-019).

### Dugan-style normalized sharing

`g_t = w_t / Σ_j w_j`, so `Σ g = 1` at every control frame including silence and transitions.

Equal-power (`Σ g² = 1`) is the spec's other named option and loses on the silence clause:
with six equal shares of `1/√N`, six uncorrelated room-tone floors sum to the power of one
full-scale track. Under Dugan they sum to `1/√N` of one track — 8 dB down at six tracks,
which is what "without allowing six tracks of noise to add coherently" is asking for. Dugan
is also what the spec names as its baseline.

### The invariant is stated over the coefficient that reaches a sample

The per-track voice-level correction `c_t` — the spec's "conservative per-track voice-level
correction, clamped to a safe range" — is applied to the audio, **not** folded into the
weights. Folding it in would count track-relative level twice, because `score_permille`
already carries it (ADR-0012's four scoring terms include a track-relative one).

That means the coefficient multiplying a sample is `g_t · c_t`, and `Σ g = 1` bounds nothing
audible. **Two** checks run as frames are produced, not only in tests:

- `Σ g_t = 1` exactly — the share, and
- `c_min ≤ Σ (g_t · c_t) ≤ c_max` — the clamp's own bounds.

The correction also erodes the dominance margin, by up to `2 · max_level_correction_db` when
a quiet track is lifted while a loud one is cut. So the achievability validator is:

```
20*log10(min_active_share / room_tone_share) - 2*max_level_correction_db
    >= solo_attenuation_margin_db
```

Defaults `0.5 / 0.005` with a 6 dB clamp give `40 − 12 = 28 dB` against a 20 dB margin. The
first draft's `0.5 / 0.02` and 12 dB gave `3.96 dB` and would have been refused by this
validator — which is how the plan review stated the finding, and why the defaults moved.
Lowering `room_tone_share` costs nothing during silence: every weight is equal there and the
shares are `1/N` whatever the floor's absolute value is.

`c_t` is constant in time, so the applied slew rate is the share's rate times a
time-invariant constant. The slew bound is stated on the share; no discontinuity can arise
from the correction.

### The envelope is produced in bounded chunks and never written down

`1 kHz × 6 tracks × 4 hours` is 690 MB of gains. The envelope is an iterator over bounded
control-frame chunks carrying its own slew state across them, exactly as the 3:1 decimator
carries filter state across windows (ADR-0011), and `tests/test_memory.py`'s ordered event log
is extended to cover it: a write happens before the last envelope chunk is *produced*. A proof
over the audio path alone is passed by a renderer that materializes all 690 MB first and only
then interleaves reads and writes.

### There is no `work/mix.json`

An earlier draft of M5's working plan added one. It is dropped, on the argument ADR-0011 used
to reject a 16-bit derivative: a new schema, version and interface **with no named consumer**
is a choice made on behalf of milestones that have not stated a need. The per-track
corrections and every measurement go to the report's `decisions` subsection, which INV-02
already requires to be semantically stable even though the report as a whole is exempt; the
render identity goes to the cache sidecar, where identity belongs. Byte-stability is proved
on the intermediate itself, which is a stronger thing to prove it on than a document
describing it.

## Alternatives considered

- **Equal-power sharing (`Σ g² = 1`).** Named by the spec, and better for a two-person
  overlap (−3 dB each rather than −6). Rejected on the silence clause: six uncorrelated room
  floors then reach the level of one full-scale track, which is the failure the spec's own
  silence sentence exists to prevent. The overlap difference is 3 dB and the loudness pass
  makes it up; the silence difference is 8 dB in the wrong direction and nothing makes it up.
- **A one-pole attack/release, the classic compressor shape.** Smoother corners. Rejected
  because the gate asks for a *bound*, and a time constant is not one. Also: at 48 samples per
  control frame a linear ramp's corners are already inaudible.
- **Gain from `score_permille` alone, with a single floor.** Simpler, one number to tune.
  Rejected: dominance then scales with the winner's score, so the gate criterion becomes a
  property of the fixture rather than of the rule, and a low-scoring genuine speaker is mixed
  down for having been recorded badly — which is what the level correction exists to fix.
- **Folding the correction into the weights**, so the normalized share bounds the applied
  gain directly. Tempting, and it would make one invariant do both jobs. Rejected: the
  correction would then stop correcting — normalizing by `Σ(w·c)` removes exactly the
  per-wearer level equalization it was applied for — and level would be counted twice.
- **A per-sample envelope with no control grid.** No interpolation, no grid arithmetic, and
  six session-length gain arrays or a great deal of per-sample branching. Rejected on INV-07
  and on the slew limit having no natural unit.
- **Binary presence with a crossfade**, ignoring the score entirely. Rejected: the score is
  the graph's own confidence and is the only thing distinguishing a marginal candidate from a
  confident one during overlap.

## Consequences

- Every default here is a number chosen against 10.5 seconds of shaped noise. **OQ-019** is
  the record of that, and each one cites it, so `rg 'OQ-019'` finds them together.
- The achievability validator means some configurations are *refused* rather than silently
  producing a mix that fails its own gate. That is deliberate, and it is the reason the
  margins are configuration rather than test constants.
- The dominance bound is worst-case, not typical. A confident solo candidate on a session
  with matched levels gets far more than 28 dB. The bound exists so the criterion holds on a
  session nobody has recorded.
- A future milestone wanting per-sample gain automation, ducking against music, or a
  different sharing law changes `envelope.py` and nothing else: the graph carries no gain
  (ADR-0012 kept it out) and the renderer consumes an iterator of coefficients.
- If OQ-019 comes back saying the release is too slow or the room-tone floor too high, those
  are configuration changes that invalidate the render cache and nothing upstream.
