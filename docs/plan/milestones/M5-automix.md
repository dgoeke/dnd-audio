# M5 — Merged MP3 automix

**Status:** in progress
**Depends on:** M3 (only — never M4 or M6)
**Spec sections:** Milestone 5; Tests and acceptance criteria 8, 11; automixer
gain-envelope assertions

## Goal

`dnd-audio mix` turns the synchronized 48 kHz originals plus the activity graph
into a listenable mono `session.mp3`: smoothed speech-aware gain envelopes, a
streamed mix, two-pass loudness normalization, and MP3 encode/decode verification.

`dnd-audio process` also lands here. The completion gate below already requires it —
"`process` exiting nonzero" — and the spec's stage-boundary section defines it as the
dependency-aware orchestration that runs activity once and attempts both downstream
branches independently. It cannot exist before both branches do, and both branches exist
at the end of this milestone. See ADR-0024.

## Completion gate

- [ ] Conservative per-track voice-level correction estimated from high-confidence
      speech attributed to that track, clamped to a safe range.
- [ ] Activity decisions become continuously smoothed gain envelopes: solo speech
      favors that lav and strongly attenuates the other five; genuine overlap keeps
      each active person's own lav audible via equal-power or otherwise bounded
      gain sharing; silence blends low-level room tone without six noise floors
      adding coherently (Dugan-style normalized gain-share is the baseline).
- [ ] Short attack, longer release/crossfade — no clipped words, clicks, or pumping.
- [ ] **Envelope-level tests** against the deterministic activity graph, with
      explicit configurable tolerances:
  - [ ] after the attack interval a solo speaker's gain dominates every inactive
        channel by at least the configured margin;
  - [ ] during genuine two-person overlap both active channels retain nontrivial gain;
  - [ ] the normalized/equal-power invariant stays bounded at every sample or
        control frame, including silence and transitions;
  - [ ] obvious correlated bleed is not promoted on two channels simultaneously;
  - [ ] envelopes have no discontinuities and respect attack, release, and max-slew limits.
- [ ] Mono output; streamed/windowed mixing, never six full waveforms in RAM (INV-07).
- [ ] Two-pass loudness toward `-16 LUFS` integrated by default, **or** one of ADR-0023's
      three guards fires and the run says which: the ceiling forbade the gain, the gain
      exceeded the master clamp, or the mix measured below the silence floor. The spec is
      amended in this milestone to say the same thing (acceptance criterion 8).
- [ ] 128 kbps mono MP3 with session ID/title metadata, then **decoded and measured**:
      integrated loudness within 1 LU of target on a run that aimed at it, true peak within
      the `-1.5 dBTP` ceiling plus a documented measurement tolerance, duration within one MP3
      frame of expected. Pre-encode gain reduction and re-encode from the lossless
      intermediate when the decoded file overshoots; retries bounded; all
      measurements retained in the report **on the failing run as well as the passing one**;
      a measurement nobody took is never a pass; the stage **fails** rather than claiming
      compliance, and the report never claims a tolerance it did not check.
- [ ] Lossless mix intermediate kept in `work/` for debugging and cache reuse, not
      as a user-facing deliverable.
- [ ] A simulated transcription failure still produces the MP3 and report, with the
      transcript stage marked `failed` and `process` exiting nonzero (INV-09, INV-13).
- [ ] The mixer imports nothing from the ASR/transcript layer. Verified structurally,
      not just by convention.

## Explicitly not in this milestone

- Stereo, spatial reconstruction, or any phase-coherent processing.
- Naive summing of six channels — explicitly forbidden by the spec.
- Neural source separation or crosstalk subtraction.

## What M2 already provides (read before starting)

- **The 48 kHz working path is a segment map, not files.** `work/timeline.json` is the
  authoritative document; `timeline.reader.TrackReader.read(start, n)` returns a bounded
  window of one reconstructed track, silence included. `ingest --materialize-48k` will
  write contiguous RF64 files, but they are **disposable content-addressed cache
  artifacts** and nothing in the mix may depend on their existence (ADR-0011). Mixing means
  stepping six `TrackReader`s over the same window range.
- **Every track answers to the session's aligned `duration_samples`**, returning silence
  past its own end, so the mix does not need to pad or special-case a short track.
- **`timeline.wavwrite` is the streamed float32 writer** for the lossless mix intermediate:
  temp-then-rename, and it chooses RF64 from the *declared* length rather than discovering
  the 4 GiB limit partway through. Use it. `determinism.write_atomic` is for JSON and holds
  its whole payload in memory — reaching for it here is a direct INV-07 violation.
- **`timeline.preflight` sizes a run from the timeline's actual duration and the artifacts
  requested.** M5 must add its own term: the mix intermediate is the third term of
  `doctor`'s original 40 GiB estimate and it does not exist yet (OQ-013).
- **The technique for proving INV-07 over a composed path is in `tests/test_memory.py`.**
  Instrument reads and writes into one ordered event log and assert a write happens before
  the last read — nothing that accumulates a session-length array can satisfy that.
  Bounding one component proves nothing about a caller that collects every window.
- **`TrackReader` holds one file descriptor per audio segment.** Six tracks with a handful
  of chunks each is fine; if M5 ever opens many sessions' worth at once, that is where the
  limit is.

## What M3 already provides (read before starting)

- **`work/activity.json` is the only thing the mix may consult about who was speaking**
  (INV-09, whose *enforcement* this milestone owns). It is frozen at schema version 1
  (ADR-0012) and carries no floats.
- **One track's active intervals, with a confidence, are a direct read.** Filter
  `candidates` to `decision == "retained"` and that `track_id`; each carries
  `probability_permille`, `peak_probability_permille`, `band_level_mbfs`, and
  `relative_level_mb`. `test_activity_artifact.py::TestTheConsumerReads::test_m5_takes_one_tracks_active_intervals_with_a_confidence`
  is that access pattern, written before this milestone existed.
- **`ActivityTrack.speech_reference_mbfs` is the per-track voice level the first gate
  criterion asks you to estimate** — the 75th percentile of that track's own candidate
  levels, band-limited to 300–3400 Hz. It is `None` where the track had too little speech to
  establish one; treat that as "unknown", never as zero. Clamping is still M5's job, and the
  number itself is unsettled until OQ-017 has a real session.
- **`ambiguous` is where obvious correlated bleed nearly won.** It marks a candidate the
  numbers condemned — margin *and* correlation both satisfied — that the track-level veto
  kept (ADR-0014). `evidence` names the competitor, the peak correlation, and the lag it
  occurred at.

  _Corrected by M5's plan review, which was right that this sentence originally read "the
  'obvious correlated bleed is not promoted on two channels simultaneously' criterion is
  about exactly these" — and that is backwards._ Under ADR-0014 an `ambiguous` candidate is
  the **least** obvious case there is: a lav hearing its wearer at that wearer's normal level
  is probably not hearing someone else, which is why the veto kept it and why M3's and M4's
  closeouts both say `ambiguous` does not mean "uncertain". The criterion is about
  **suppressed** candidates. The M5 rule, stated once: suppressed candidates sit at the
  room-tone share; every retained candidate, `ambiguous` included, is eligible.
- **The graph does not contain gain.** It says who was speaking and how sure the pipeline is;
  envelopes are entirely this milestone's (ADR-0012 deliberately kept them out).

## What M4 already provides (read before starting)

M5 depends on M3, never on M4 — the graph is unchanged by anything the transcript branch
decides, asserted by a re-hash inside the composed run and by a structural import test. What
M4 leaves M5 is not data, it is three runner patterns and one trap.

- **Failure cleanup runs *after* the `output_inside_raw` carve-out, never before it**
  (ADR-0021). `run_mix` will delete the stale artifacts a failed run left; when an output path
  resolves inside a source directory those unlinks *are* the INV-01 violation. All three
  existing composed runners had this backwards, from M2 until M4's verify phase.
- **`tests/test_raw_guard.py::TestCleanupNeverWritesIntoRaw` needs a `mix` parameter the moment
  `run_mix` exists.** It is parametrized over every composed command precisely because M2, M3
  and M4 each wrote a regression test naming only the runner that milestone had added, and all
  three carried the same bug. Adding the parameter is the whole obligation; forgetting it is
  what the parametrization makes visible.
- **A stage that completed keeps its artifacts on a partial failure; a stage that did not
  keeps nothing.** `ReportBuilder.completed` is the predicate — distinct from `recorded`, which
  answers INV-13's no-gaps question. This is directly useful here: after a failed `transcribe`
  the graph is still on disk, which is what lets `mix` run against it and `process` attempt
  both branches independently.
- **Two commit points are legitimate and INV-08 is scoped to them** (ADR-0021). If `mix`
  commits its own caches at a point of its own, verify before publishing at *each* point, and
  say in the test's name which region it globs.
- **Nothing text-derived is in the graph, and it is checked rather than trusted.** M4 also
  closed the hazard M3's review deferred: a test asserts no ASR-derived text reaches
  `ActivityDecision.detail` or `ActivityNote.message`. The prohibition on M5 *reading* those
  two fields (below) still stands on its own.

## Known risks and open questions

- Decoded loudness alone is *not* evidence of correct channel selection. If the
  only tests are loudness tests, a mix that picks the wrong speaker will pass.
  The envelope assertions are the real gate.
- **The INV-09 field allowlist freezes property names, not prose.**
  `ActivityDecision.detail` and `ActivityNote.message` are unrestricted strings. **The mixer
  must not read either** — they are human-facing audit text, and deriving a single sample
  from them would make the mix depend on content no test constrains. The structural
  "imports nothing from the ASR layer" check does not catch this. Raised by independent
  review in M3's verify phase; see `../reviews/M3-code-20260802-1708.md`.
- **`activity.bleed.compare_pairs` is quadratic in the session's candidate count.** It
  enumerates every pair via `itertools.combinations` and only then rejects non-overlaps.
  Candidates are already sorted by start sample, so a sweep would be `O(n log n)` plus the
  pairs that genuinely overlap. Deferred in M3 with no measured evidence either way — the
  per-pair work rejected is one integer comparison, and a real session's candidate count is
  unknown until H2. If M5 walks the candidate set at scale and it hurts, that is the fix.
- The true-peak ceiling applies to the decoded MP3, not the pre-encode
  intermediate. Lossy encoding introduces overshoot.


---

## Working plan

_Scratch. Written during the start phase, revised against
`../reviews/M5-plan-20260802-2109.md`, replaced by the Closeout at the end. Where the review
changed something, it says so — the reasoning is in the review, not repeated here._

### Three decisions, recorded as ADRs before the code

#### ADR-0022 — the gain envelope

**The control grid.** Gains live on a **1 kHz control grid** and are linearly interpolated
to samples, so the applied gain is continuous by construction. 1 kHz rather than 100 Hz
because a 10 ms attack has to be *many* frames before a max-slew limit means anything.
Three things are validated at configuration load, not assumed:

- `control_rate_hz` divides `CANONICAL_SAMPLE_RATE`, so samples-per-frame is integral;
- `attack_ms` and `release_ms` land on whole control frames (`ms · rate % 1000 == 0`);
- the configured solo margin is achievable — see the weights below.

Endpoints are explicit: a candidate `[start, end)` is active on control frames
`[start // spf, ceil(end / spf))`, the same **covering** rule as
`resample.to_derivative_interval` and for the same reason. The session's last control frame
may cover fewer than `spf` samples, and per-sample interpolation is clipped to
`duration_samples`. Durations are compared as integer samples, never as floats.

**Weights, with two floors.** Per track per frame: `active` is 1 when a **retained**
candidate covers the frame; `weight = active ? min_active_share + (1 − min_active_share)·score
: 0`, then `weight = max(weight, room_tone_share)`. Two floors rather than one is what makes
the gate criteria provable instead of incidental — a low-scoring genuine speaker still gets
at least `min_active_share`, an inactive channel never falls below `room_tone_share`.

**Suppressed candidates sit at the room-tone share; every retained candidate, `ambiguous`
included, is eligible.** That is the whole of M5's reading of the graph, and it is the
correction the plan review forced into the risk note above.

**Smoothing is a slew-limited linear ramp**, not a one-pole: rising ≤ `1/attack_frames` per
frame, falling ≤ `1/release_frames`. An exponential never reaches its target and its "attack
time" is a time constant rather than a bound, which would make the gate's "respect attack,
release, and maximum-slew limits" unassertable.

**Sharing is Dugan-style normalized:** `g_t = w_t / Σ_j w_j`. Chosen over equal-power
(`Σ g² = 1`) because equal-power lets six room-tone floors reach the level of one full-scale
track during silence, the exact failure the spec's silence clause names; the spec names Dugan
as the baseline and permits "otherwise bounded gain sharing".

**The bounded-gain invariant is stated over what reaches a sample.** The per-track
voice-level correction `c_t` stays outside the share — folding it into the weights would
count track-relative level twice, since `score_permille` already carries it — so the applied
coefficient is `g_t · c_t` and **two** things are checked at runtime as frames are produced,
not only in tests:

- `Σ g_t = 1` exactly (the share), and
- `c_min ≤ Σ (g_t · c_t) ≤ c_max`, the clamp's own bounds.

_Raised by the plan review, which was right: the first statement alone bounds nothing
audible._ The same correction erodes the dominance margin by up to `2 · max_level_correction_db`,
so the achievability validator is
`20·log10(min_active_share / room_tone_share) − 2·max_level_correction_db ≥
solo_attenuation_margin_db`. With defaults `0.5 / 0.005` and a 6 dB clamp that is
`40 − 12 = 28 dB` against a 20 dB margin. The plan's original numbers (`0.5 / 0.02`, 12 dB)
gave `3.96 dB` and would have been refused by this validator — which is the finding, stated
arithmetically. `c_t` is constant in time, so the applied slew rate is the share's times a
time-invariant constant and no discontinuity can arise.

**There is no `work/mix.json`.** _Dropped on the plan review's argument, which is ADR-0011's:
a new schema, version and interface with no named consumer is a choice made on behalf of
milestones that have not stated a need._ Per-track corrections and every measurement go to
the report's `decisions` subsection, which INV-02 already requires to be semantically stable;
the render identity goes to the cache sidecar. Byte-stability is proved on the intermediate
itself, which is a stronger artifact to prove it on than a document describing it.

#### ADR-0023 — loudness measurement is FFmpeg's; the intermediate is unity gain

The lossless intermediate is written at unity master gain and cached; the master gain is an
**encode parameter** (`-af volume=…dB`), so a true-peak retry costs one encode rather than
one re-mix of six four-hour tracks, and the intermediate survives a change of loudness
target. **The render cache identity therefore carries `mix.envelope` and nothing else from
the mix section** — the loudness target, the bitrate, the tolerances and the retry budget sit
after that boundary and reach only the MP3, which is regenerated every run and never cached.
_The plan review caught the contradiction: keying the intermediate on the whole `mix`
projection would re-mix six tracks to change a bitrate._ Same split, same reason, as ADR-0016
between `activity.vad` and `activity.bleed`.

**One decode serves every measurement of the MP3.**
`ffmpeg -i session.mp3 -af ebur128=peak=true -f f32le -` puts the R128 summary on stderr and
the decoded samples on stdout; the samples are counted in bounded chunks and discarded, a
clean exit is required, and the exact integer count is what the duration tolerance is applied
to. _The plan first took duration from `ffprobe`; the review was right that a container
duration can stay plausible while decoding yields fewer samples, and the gate says "decoded"._
The summary parse is verified against ffmpeg 8.0 (`I: -21.1 LUFS`, `Peak: -18.1 dBFS`; 0.1
resolution against a 1.0 LU tolerance). Every command is recorded in the report verbatim and
the FFmpeg version enters provenance.

This is deliberately not ADR-0011's rejection of FFmpeg: that one is about the canonical
16 kHz derivative, a *cached artifact* whose identity must not move when a tool is upgraded
for unrelated reasons. A measurement is not that. Guards: the master gain is clamped to
`max_master_gain_db`, and a mix measuring below `silence_floor_lufs` is left un-normalized
with a warning rather than amplified by 50 dB — the canonical fixture through real Silero
yields zero candidates and is exactly that case.

#### ADR-0024 — `process`

One snapshot of `raw/`, activity performed once, then **the mix branch first** and the
transcript branch second, **each in its own handler so a failure in one collects an error
rather than short-circuiting the other**. _The review was right that a sequential mix-first
implementation can still abort on a mix exception and never transcribe; independence has to
be a property of the control flow, not of the ordering._ Mix-first still earns its place: it
makes "the mix cannot have consumed anything the transcript branch produced" true by
construction as well as by test, and it is the branch the spec says must survive.

**Three commit points** (activity caches, mix cache, ASR cache), each preceded by
`verify_unchanged` — ADR-0021 already scopes INV-08 to a commit point rather than a run, and
the mix's own point is load-bearing because the mix is the one stage that reads *source*
audio after inspection. **Plus one unconditional final `verify_unchanged` before report
finalization, on the success path and the failure path alike**: with three commit points, a
transcript failure before the ASR commit otherwise leaves the sources unverified after the
mix read them. A stage that completed keeps its artifacts on a partial failure; cleanup runs
**after** the `output_inside_raw` carve-out, never before it.

### Files

New package `src/dnd_audio/mix/`: `__init__.py` (constants), `levels.py` (per-track
correction), `envelope.py` (weights → slew-limited presence → normalized share, produced as
bounded chunks with carried state, plus the runtime checks), `render.py` (six `TrackReader`s
over one window range into `WavWriter`), `loudness.py` (the single-decode measurement and its
parser), `encode.py` (libmp3lame + measure + bounded gain-reduction retry), `cache.py`,
`runner.py`. Plus `src/dnd_audio/orchestrate.py` and the three ADRs.

Changed: `config.py` (`MixConfig` splits into `mix.envelope` and the post-boundary encode
settings, with the three validators above; `StageScope` gains `"mix"`; `_FIELD_SCOPES` maps
`mix.envelope` → `{"mix"}` and the encode settings → nothing, because they reach only an
artifact that is never cached), `cli.py` (real `mix` and `process`), `timeline/preflight.py`
(the mix-intermediate term M5 owes, OQ-013), `transcript/runner.py` (expose the transcript
half so `process` composes it rather than reimplementing it — ADR-0015's argument, two
milestones later).

**No new fixture, and no new schema.** The canonical session already carries solo speech
bleeding into four tracks (tx-a at 249600), two simultaneous speakers (tx-d and tx-e at
326400), post-gap speech and leading silence; `mutual_bleed_session` carries the `ambiguous`
case; `delayed_bleed_session` carries bleed outside the zero-lag window.

### Every gate criterion, and the test that proves it

| Criterion | Proof |
| --- | --- |
| Per-track voice-level correction, clamped | `test_mix_levels.py::TestVoiceLevelCorrection` — target from the tracks' own references, a loud track attenuated and a quiet one lifted, both clamped at `max_level_correction_db`; a `None` reference corrected by **zero and warned**, never treated as 0 dBFS; and a session where every reference is absent |
| Solo favours that lav, attenuates the other five | `test_mix_envelope.py::TestSolo` — after `attack_ms`, tx-a's **applied coefficient** exceeds every other track's by ≥ `solo_attenuation_margin_db`, tested at both correction clamp extremes; `test_the_configured_margin_is_achievable` proves the validator's bound rather than trusting the fixture's numbers |
| Genuine overlap keeps both audible | `TestGenuineOverlap` — canonical graph at 326400, an `ambiguous` candidate proved to mix identically to a plain one, and **the worst pair the rule admits**: score 1000 against score 0 with the quieter speaker cut by the full clamp. That last one is the criterion's real gate and it *failed* at the shipped `-15 dB`; `SessionConfig._check_overlap_gain_is_achievable` now refuses a promise `EnvelopeConfig.guaranteed_overlap_gain_db` cannot keep, the same treatment the solo margin already had. **Not `mutual_bleed_session`** — this row claimed it and no test used it; the cross-product above is the stronger proof and building a second session's graph would only re-test the fixture. Found by M5's code review |
| Silence blends room tone; six floors do not add coherently | `TestSilence` — every share exactly `1/N`; and on constructed inputs, **independent** equal-power noise sums to `1/√N` of one track's RMS while **perfectly correlated** noise cannot exceed one track's. _The plan first asserted `1/N`, which is arithmetically wrong and would have pushed an extra attenuation into the mixer to make it pass._ |
| Short attack, longer release, no clicks or pumping | `TestSlew` — per-frame change ≤ `1/attack_frames` rising and `1/release_frames` falling over **every** frame of every track; continuity is structural; `test_config.py::TestTheEnvelopeGridIsExact::test_the_default_attack_finishes_inside_the_default_vad_pad` pins the relationship the design rests on (OQ-019) |
| Bounded gain invariant at every frame | `TestTheBoundedGainInvariant` — `Σ g = 1` to 1e-9 **and** `c_min ≤ Σ(g·c) ≤ c_max` at every frame of every fixture, including opposite clamp extremes; a test that feeds the runtime checker an unnormalized matrix and proves it **fails**, because a check that cannot fire is decoration; and four fractional clamps, because the clamp used to be spelled two ways — millibels in `levels`, raw dB here — so any value whose hundredths rounded up made a track's own permitted correction breach the bound and failed the stage |
| Obvious correlated bleed not promoted on two channels | `TestBleedIsNotPromoted` — during tx-a's utterance the four bleed-receiving tracks sit at the room-tone share; **contrast test**: the same graph with those candidates flipped to `retained` does promote them, so the assertion is about the decision rather than about the numbers happening to work out |
| Mono; streamed; never six waveforms in RAM | `test_mix_render.py` (mono, exact length, sample-exact against a hand-computed short case) and `test_memory.py::TestTheMixPathStreams` — M2's ordered event log extended to the **envelope** *and* kept over the audio: a write happens before the last envelope chunk is produced **and** before the last `TrackReader.read`, and neither a chunk nor a read exceeds one window. _The plan review found the first draft proved only the audio path; the code review found the second had stopped proving it, and a renderer collecting six waveforms before streaming would have passed. Both are in one log now._ |
| Two-pass loudness toward −16 LUFS | `test_mix_encode.py::TestTheRealEncode` — through real FFmpeg: `test_the_decoded_file_meets_every_configured_target` puts the decoded MP3 within `loudness_tolerance_lu`, and `test_a_different_loudness_target_lands_somewhere_different` moves the target to −23 LUFS and measures it land there, so the target is read rather than baked. Plus `TestTheMasterGain`, which covers each of ADR-0023's three guards and proves a *normalized* run is still failed for missing the target |
| 128 kbps mono MP3, decoded and measured, bounded retries, **fails** rather than claims compliance | `test_mix_encode.py` against a scripted measurer — compliant first time; one overshoot then compliant; an always-overshooting one that exhausts `max_retries` and fails the stage; a summary with no true-peak line and a decode that reports `-inf` where the target was aimed at, both **failing** rather than passing on an absent number; and `test_mix_run.py::TestFailures::test_an_uncompliant_encode_still_records_every_measurement_it_took`, which drives the real retry loop to exhaustion and asserts four attempts and four encode commands reach **the report** — the earlier proof read the exception text instead. `test_mix_run.py::TestTheCanonicalSession` through real ffmpeg — exists, decodes cleanly, mono, 128 kbps, **decoded sample count** within `duration_tolerance_frames` of `duration_samples`, session id and title in the tags |
| Lossless intermediate in `work/`, not a deliverable | `test_mix_cache.py` — under `work/cache/mix/`, a hit on the second run, **absent** from `provenance.deliverables`, every identity component proven to invalidate it (the `derivative_identity_document` pattern: document separate from hash, so a test asserts *which* components are present), and INV-08's incomplete-entry half: a truncated intermediate, an orphaned sidecar, a self-inconsistent one, and a declared-size mismatch are each **not** a hit |
| Transcription failure still yields MP3 + report, `process` nonzero | `test_process_run.py` — six tests, because "independent" is a property of the control flow: activity executes exactly once (spied); a transcript failure leaves the MP3 present and hashed with `mix` complete; a **mix** failure does not cancel transcribe or render; either branch failing accounts for every stage and exits 4; tampering *after both branches verified*, where only the final unconditional check can see it (the earlier test was caught by the transcript branch's own verification and passed with the final check deleted); and a branch keeping its own diagnosis rather than the outer error, which ADR-0024 requires and `_failed` was overwriting |
| The mixer imports nothing from the ASR/transcript layer, structurally | `test_mix_run.py::TestInv09` — the **transitive** import closure of `dnd_audio.mix` inspected in a subprocess (the technique `test_silero.py` uses for Torch), not a grep of one directory; the intermediate byte-identical before and after `transcribe`; and one that rewrites every `ActivityDecision.detail` and `ActivityNote.message` and re-renders **with the cache disabled** so the mix genuinely re-executes. _The prose is rewritten on the `ActivityGraph` the renderer is **handed**, not on `work/activity.json`: `run_mix` rebuilds that document from the attribution cache before mixing, so the file-level version asserted nothing and a mixer reading `detail` passed it. Found by the code review; mutation-checked against a mixer that folds `len(detail)` into a weight._ |

### Invariants at risk, and what stops the violation

- **INV-01** — two new composed commands. `TestCleanupNeverWritesIntoRaw` gains a `mix` and a
  `process` parameter (the obligation M4's closeout hands over), and so do the full-run hash
  equality and mid-flight corruption tests — INV-01 names all three, and parametrizing only
  the first is how M2, M3 and M4 each covered one runner out of three. `process` additionally
  verifies unconditionally before report finalization. `mix_outputs()` declares the MP3 and
  the cache directory as data.
- **INV-07** — the largest read in the project. Bounded audio windows, `WavWriter` for the
  intermediate, `write_atomic` unreachable from the audio path, and the envelope produced in
  bounded chunks with carried slew state — never materialized, since 1 kHz × 6 tracks ×
  4 hours is 690 MB. Both paths are in the event log.
- **INV-08** — a new cache and a third commit point. Identity carries the timeline hash, the
  graph's `attribution_cache_key`, the `mix.envelope` projection, both versions and NumPy's;
  commit only after `verify_unchanged`; an incomplete entry is never a hit, proved four ways;
  and the failed-commit test globs the whole downstream region under `work/cache` and says so
  in its name.
- **INV-09** — this milestone owns enforcement; three tests above, each strengthened.
- **INV-13** — `mix` and `process` account for every stage; partial never exits zero.
- **INV-02** — the intermediate is byte-identical on an unchanged rerun.
- **INV-04** — not in play: the mix introduces no fractional frame rate and no accumulated
  float duration. What the control grid needs instead is stated above and validated rather
  than assumed — rate divisibility, whole-frame attack and release, covering endpoints, and
  integer sample comparison for every duration.

### Charter amendments

1. **`process` lands here**, added to the Goal above. Its own gate criterion already named it.
2. **The `ambiguous` note in "What M3 already provides" was backwards** and is corrected in
   place, with the M5 rule stated. Found by the plan review.
3. **`activity.bleed.compare_pairs` stays quadratic.** The risk note says "if M5 walks the
   candidate set at scale and it hurts, that is the fix". M5 walks the retained candidates
   once, linearly, to build the presence signal; it never enumerates pairs. Still no measured
   evidence and the real candidate count is unknown until H2 — recorded as not done, with the
   reason, rather than fixed speculatively.
4. **No `work/mix.json`**, per the plan review. The charter never asked for one; the first
   draft of this plan added it and the review argued it out again.

### Open questions raised

**OQ-019** (the six automix constants) and **OQ-020** (real encode overshoot and decoded
duration) are registered in `OPEN-QUESTIONS.md` **before** the first configuration default
lands, which is where the review found the first plan out of order. Every `MixConfig` default
cites one of them.

### Order of work

Adopted from the review's own "what I would change first":

1. ADR-0022, `MixConfig` + the three validators, `levels.py`, `envelope.py`, their tests.
   **The envelope assertions are the real gate** — a mix that picks the wrong speaker passes
   every loudness test there is — so they come before any audio is written, and they are
   written over the *applied* coefficient at both clamp extremes.
2. `render.py`, `cache.py`, the extended memory test, the preflight term.
3. ADR-0023, `loudness.py`, `encode.py`.
4. `runner.py`, the CLI `mix`, the three INV-01 parameters.
5. ADR-0024, `orchestrate.py`, the CLI `process`, the same parameters again.
