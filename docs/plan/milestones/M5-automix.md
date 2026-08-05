# M5 — Merged MP3 automix

**Status:** closed
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

- [x] Conservative per-track voice-level correction estimated from high-confidence
      speech attributed to that track, clamped to a safe range.
- [x] Activity decisions become continuously smoothed gain envelopes: solo speech
      favors that lav and strongly attenuates the other five; genuine overlap keeps
      each active person's own lav audible via equal-power or otherwise bounded
      gain sharing; silence blends low-level room tone without six noise floors
      adding coherently (Dugan-style normalized gain-share is the baseline).
- [x] Short attack, longer release/crossfade — no clipped words, clicks, or pumping.
- [x] **Envelope-level tests** against the deterministic activity graph, with
      explicit configurable tolerances:
  - [x] after the attack interval a solo speaker's gain dominates every inactive
        channel by at least the configured margin;
  - [x] during genuine two-person overlap both active channels retain nontrivial gain;
  - [x] the normalized/equal-power invariant stays bounded at every sample or
        control frame, including silence and transitions;
  - [x] obvious correlated bleed is not promoted on two channels simultaneously;
  - [x] envelopes have no discontinuities and respect attack, release, and max-slew limits.
- [x] Mono output; streamed/windowed mixing, never six full waveforms in RAM (INV-07).
- [x] Two-pass loudness toward `-16 LUFS` integrated by default, **or** one of ADR-0023's
      three guards fires and the run says which: the ceiling forbade the gain, the gain
      exceeded the master clamp, or the mix measured below the silence floor. The spec is
      amended in this milestone to say the same thing (acceptance criterion 8).
- [x] 128 kbps mono MP3 with session ID/title metadata, then **decoded and measured**:
      integrated loudness within 1 LU of target on a run that aimed at it, true peak within
      the `-1.5 dBTP` ceiling plus a documented measurement tolerance, duration within one MP3
      frame of expected. Pre-encode gain reduction and re-encode from the lossless
      intermediate when the decoded file overshoots; retries bounded; all
      measurements retained in the report **on the failing run as well as the passing one**;
      a measurement nobody took is never a pass; the stage **fails** rather than claiming
      compliance, and the report never claims a tolerance it did not check.
- [x] Lossless mix intermediate kept in `work/` for debugging and cache reuse, not
      as a user-facing deliverable.
- [x] A simulated transcription failure still produces the MP3 and report, with the
      transcript stage marked `failed` and `process` exiting nonzero (INV-09, INV-13).
- [x] The mixer imports nothing from the ASR/transcript layer. Verified structurally,
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
  unknown until a full live run. If M5 walks the candidate set at scale and it hurts, that is the fix.
- The true-peak ceiling applies to the decoded MP3, not the pre-encode
  intermediate. Lossy encoding introduces overshoot.


---

## Closeout

### What works end to end

`uv run dnd-audio mix /path/to/session` — the whole right branch of the spec's stage DAG,
and the branch that must survive a transcription failure.

It does everything `activity` does, then estimates a per-track voice-level correction from
each track's own `speech_reference_mbfs` (median target, clamped, a missing reference
corrected by **zero** and warned about), turns the graph's **retained** candidates into a gain
per track per 1 kHz control frame — two weight floors, a slew-limited linear ramp, a
Dugan-style normalized share — linearly interpolates that onto samples, steps six
`TrackReader`s and the envelope over the same window range into one streamed mono float32
intermediate under `work/cache/mix/`, verifies INV-01 a second time, commits the mix cache,
measures the intermediate with `ebur128`, encodes a 128 kbps mono MP3 at a master gain that
already aims at the true-peak ceiling, decodes it, measures it again, walks the gain down
under a bounded retry budget, and writes one report covering six stages.

On the canonical fixture through the **real** Silero release:

```
$ uv run dnd-audio mix sess
  mixed 10.500s to 1-channel 128 kbps MP3: -39.7 LUFS, -3.0 dBTP, 1 encode attempt(s)
  warn  mix_loudness_target_unreachable: reaching -16.0 LUFS needs +24.5 dB, but the
        -1.5 dBTP ceiling allows only +1.6 dB above this mix's true peak of -3.1 dBTP.
        The ceiling wins: it is a limit on clipping, and the loudness figure is a target.
        The MP3 will be about 22.9 LU quieter than asked for (OQ-020).
  mp3        sess/output/session.mp3
  report     sess/output/ingest-report.json
$ echo $?
0
```

That warning is the milestone's most important line of output and it is **correct**. Silero
finds zero candidates on synthetic noise (INV-10, M3's closeout), so every track sits at the
room-tone share, the mix is a quiet six-way blend, and reaching −16 LUFS would need +24.5 dB
that the ceiling forbids. The stage honours the ceiling, says so, and exits zero.

`uv run dnd-audio process /path/to/session` — dependency-aware orchestration of both
branches. One snapshot of `raw/`, activity performed **once**, then the mix branch and the
transcript branch each in its own handler so a failure in either collects an error rather than
short-circuiting the other, then one unconditional `verify_unchanged` before the report is
finalized, then both branches' stages recorded.

```
$ uv run dnd-audio process --fake-models sess
  mix        sess/output/session.mp3
  transcript 4 segment(s)
  report     sess/output/ingest-report.json

stages: inspect complete · reconstruct complete · activity complete
        transcribe complete · render complete · mix complete
cache:  18 hits / 12 misses
deliverables: output/session.mp3, output/transcript.json, output/transcript.md,
              work/activity.json, work/manifest.json, work/timeline.json,
              work/transcript-records.json
```

Without `--fake-models`, `process` raises the same `DEFERRED: M6b` `NotImplementedError`
`transcribe` does, **before any work** — an operator who wants the audio branch on such a host
runs `mix`, which needs no ASR adapter at all. `mix` and `process` were the last two stubs;
no command in the CLI is a stub any more.

### Tests and commands run, with results

```
$ ./scripts/gate.sh
  pass  system dependencies      pass  ruff check        pass  ruff format
  pass  type check               pass  lock is current   pass  placeholder scan
  pass  plan consistency         pass  pytest (offline, cpu)
  2028 passed, 3 deselected in 117.79s
GATE PASSED
```

Zero skips. The three deselected are `host_smoke`. M4 closed at 1768 tests; M5 adds 260.

Every gate criterion, and the test that proves it:

| Criterion | Proof |
| --- | --- |
| Per-track voice-level correction, clamped | `test_mix_levels.py::TestVoiceLevelCorrection` — target from the tracks' own median, a loud track cut and a quiet one lifted, both clamped; a `None` reference corrected by **zero and warned**, never read as 0 dBFS; a session where every reference is absent; and the clamp bounding every gain by construction |
| Solo favours that lav, attenuates the other five | `test_mix_envelope.py::TestSolo` — after the attack, tx-a's **applied coefficient** exceeds every other track's by ≥ `solo_attenuation_margin_db`, at scores 1000/800/300/**0** and at both extremes of the correction clamp; `test_the_configured_margin_is_achievable_from_the_floors_alone` proves the validator's bound using no fixture at all |
| Genuine overlap keeps both audible | `TestGenuineOverlap` — the canonical graph at 326400, an `ambiguous` candidate proved to mix **identically** to a plain one, and the worst pair the rule admits: score 1000 against score 0 with the quieter speaker cut by the full clamp. That last one is the criterion's real gate; see "Deviations" |
| Silence blends room tone; six floors do not add coherently | `TestSilence` — every share exactly `1/N`; on constructed inputs, **independent** equal-power noise sums to `1/√N` of one track's RMS while **perfectly correlated** noise cannot exceed one track's |
| Short attack, longer release, no clicks or pumping | `TestSlew` — per-frame change ≤ `1/attack_frames` rising and `1/release_frames` falling over **every** frame of every track; the attack reaches full weight in exactly the configured frames; continuity is structural and the expansion is asserted to land on each frame's value at its last sample; `test_config.py::…::test_the_default_attack_finishes_inside_the_default_vad_pad` pins the relationship the design rests on (OQ-019) |
| Bounded gain invariant at every frame | `TestTheBoundedGainInvariant` — `Σ g = 1` to 1e-12 **and** `c_min ≤ Σ(g·c) ≤ c_max` at every frame of silence, solo and everyone-at-once; a test that feeds the runtime checker an unnormalized matrix and proves it **fails**; and four fractional clamps, which the shipped code got wrong |
| Obvious correlated bleed not promoted on two channels | `TestBleedIsNotPromoted` — during tx-a's utterance the bleed-receiving track sits at the room-tone share; **contrast test**: the same graph with that candidate flipped to `retained` does promote it, so the assertion is about M3's decision rather than about the numbers working out |
| Mono; streamed; never six waveforms in RAM | `test_mix_render.py` (mono, exact length, silence mixing to the plain mean of six independently-read tracks) and `test_memory.py::TestTheMixPathStreams` — one ordered event log over reads, envelope chunks and writes: a write happens before the last read **and** before the last chunk is produced, and none of the three exceeds one window |
| Two-pass loudness toward −16 LUFS | `test_mix_encode.py::TestTheRealEncode` through real FFmpeg — the decoded MP3 within `loudness_tolerance_lu`, and a target moved to −23 LUFS measured landing there, so the target is read rather than baked; `TestTheMasterGain` covers each of ADR-0023's three guards and proves a *normalized* run is still failed for missing the target |
| 128 kbps mono MP3, decoded and measured, bounded retries, **fails** rather than claims compliance | `test_mix_encode.py` against a scripted measurer — compliant first time; one overshoot then compliant; an always-overshooting one exhausting `max_retries`; a reduction equal to the measured overshoot; an attempt exactly on the ceiling still moving; a failure a gain cannot fix not retried; and both "measurement nobody took" cases failing. `test_mix_run.py::TestTheCanonicalSession` through real ffmpeg — exists, decodes, mono, 128 kbps, **decoded sample count** within tolerance, session id and title in the tags |
| Lossless intermediate in `work/`, not a deliverable | `test_mix_cache.py` — under `work/cache/mix/`, a hit on the second run, **absent** from `provenance.deliverables`, every identity component asserted **by name** (the `derivative_identity_document` pattern), and INV-08's incomplete-entry half four ways: truncated audio, an orphaned sidecar, a self-inconsistent one, and a length the caller did not expect |
| Transcription failure still yields MP3 + report, `process` nonzero | `test_process_run.py` — activity executes exactly once (spied); a transcript failure leaves the MP3 present and hashed with `mix` complete and exit 4; a **mix** failure does not cancel transcribe or render; tampering after *both* branches verified, where only the final unconditional check can see it; and a branch keeping its own diagnosis |
| The mixer imports nothing from the ASR/transcript layer | `test_mix_run.py::TestInv09` — the **transitive** import closure of `dnd_audio.mix.runner` inspected in a subprocess; the intermediate byte-identical before and after `transcribe`; and every `ActivityDecision.detail` and `ActivityNote.message` rewritten on the graph the renderer is **handed** |

INV-01 is now parametrized over all five composed runners — `ingest`, `activity`,
`transcribe`, `mix`, `process` — for all three of its mechanisms:
`TestCleanupNeverWritesIntoRaw` (15 cases across the three properties), plus
`TestEveryComposedRunVerifiesItsSources`, which is new. That closes the obligation M4's
closeout handed over, and closes it wider than it was written: M2, M3 and M4 each tested only
the runner that milestone added, and only the first mechanism was parametrized at all.

Four fixes were **mutation-checked** — the behaviour was reverted and the proof was confirmed
to fail:

```
mixer reads ActivityDecision.detail    → test_rewriting_the_graphs_prose…  FAILED
overlap floor back to -15.0            → 139 failed (every session fixture refused)
_targets breaks at the first long span → test_overlapping_spans_on_one_track…  FAILED
clamp converted from raw dB            → test_a_fractional_clamp…[0.015]  FAILED (EnvelopeError)
```

Reviews: `../reviews/M5-plan-20260802-2109.md` before implementing,
`../reviews/M5-code-20260802-2358.md` before closing. The code review is where most of this
milestone's late work came from; it opened with "I would not close M5 in its current state"
and it was right.

### Decisions made (→ ADRs)

- **[ADR-0022](../decisions/0022-the-gain-envelope.md)** — the gain envelope. A 1 kHz control
  grid validated to be exact, two weight floors rather than one, a slew-limited linear ramp
  rather than a one-pole, Dugan-style normalized sharing rather than equal-power, the level
  correction kept *outside* the share, and the bounded-gain invariant stated over the
  coefficient that reaches a sample. No `work/mix.json`.
- **[ADR-0023](../decisions/0023-loudness-encoding-and-the-unity-gain-intermediate.md)** —
  loudness measurement is FFmpeg's, the intermediate is unity master gain, the master gain is
  an encode parameter, one decode serves every measurement, and **three guards** decide when
  not to normalize at all. Amended during verify: the third guard, and the amendment to the
  spec that goes with it.
- **[ADR-0024](../decisions/0024-process-orchestration.md)** — `process`. One snapshot,
  activity once, mix branch first, each branch in its own handler, three commit points plus
  one unconditional final verification.

### Assumptions made and open questions raised

- **[OQ-019](../OPEN-QUESTIONS.md)** — the automix constants, raised here. Six numbers chosen
  against 10.5 seconds of shaped noise: `attack_ms`, `release_ms`, `room_tone_share`,
  `min_active_share`, `max_level_correction_db`, and the two gate thresholds. None can make
  the mix *wrong* — the bounded-gain invariant holds for any admissible values and two
  validators refuse an unachievable combination — but they decide whether the result is
  pleasant, which no test can assert. Amended during verify: `overlap_min_gain_db` is now
  **derived** rather than estimated, and the entry records why the original −15 was 0.66 dB
  optimistic. Needs M11's live Session Zero; blocks nothing.
- **[OQ-020](../OPEN-QUESTIONS.md)** — what a real 128 kbps mono MP3 encode does to peak and
  duration, raised here. Three assumptions now: that the retry budget resolves real overshoot,
  that decoded duration lands within one MP3 frame, and (added during verify) that an
  overshoot large enough to need a retry is smaller than `loudness_tolerance_lu` — a retry
  reduces the gain by exactly the overshoot, so on a run aiming at the target a larger
  overshoot lands *outside* the loudness tolerance and fails rather than resolving. The
  measurements are already retained per attempt, so one real session answers all three.
- **[OQ-013](../OPEN-QUESTIONS.md)** — still open, further answered. The preflight's third
  term now exists: one mono float32 file at the session's own duration, 2.8 GiB for four
  hours against 5 GiB of derivatives, requested only when a run will actually mix. The ASR
  cache is still not sized.
- **OQ-004, OQ-007, OQ-017, OQ-018** — untouched by this milestone. M5 reads the graph and
  the timeline as given.

### Notes for future implementors

**The envelope assertions are the gate, and "assert the share" is the trap.** The share sums
to one by construction, so a test over it passes for any level correction whatever — six
tracks each lifted 6 dB sum to 2.0 while the share still reports 1.0. Every criterion here is
asserted over `EnvelopeChunk.applied`, the share times the correction, which is the number
that multiplies a sample. The plan review caught this before any code existed; it is the
single most valuable thing in either review.

**A validator beats a test for anything that is a promise.** `solo_attenuation_margin_db` had
an achievability validator from the start and never broke. `overlap_min_gain_db` had a worked
example in a docstring and was wrong by 0.66 dB — provably wrong, on an input the rule admits,
undetected by five tests that each varied one dimension. If a configured threshold is a claim
about what the rule guarantees, compute the guarantee and refuse a configuration that
overpromises. Both validators now exist; the overlap one lives on `SessionConfig` because its
bound depends on the track count.

**Two things can accumulate on the mix path, not one.** The audio is obvious. The gains are
not: 1 kHz × 6 tracks × 4 hours is 690 MB, so `EnvelopeStream` is an iterator carrying slew
state across chunk boundaries. `tests/test_memory.py` instruments reads, envelope chunks
**and** writes into one ordered log — and it got there by failing twice, once in each
direction. The plan's first draft watched only the audio; the shipped code watched only the
envelope and the writer. Either half alone is passed by an implementation that materializes
the other half. M7a's compression/upload path is separate from `process`, but its own
bounded-read test should use the same read/write event-log technique.

**`run_mix` rebuilds `work/activity.json` before it mixes.** Not from the file — from the
attribution cache, through `perform_activity`. Anything you write into that file between runs
is overwritten and never reaches the mixer. A test that edits the graph on disk and then calls
a composed runner proves nothing; edit the `ActivityGraph` object the renderer is handed. This
cost the INV-09 prose proof its entire value for one milestone and the second of two
independent reviewers still read it as sound.

**Switching detectors doubles the mix cache and nothing prunes it.** A real-Silero run and a
`--fake-models` run produce different graphs, so different `attribution_cache_key`s, so
different mix identities and two 2 MB intermediates side by side. Content-addressing working
as designed, and 2.8 GiB each at four hours. Nothing sweeps `work/cache/mix/`; if that becomes
a problem it is M7b's, and the sidecar-plus-audio layout makes an LRU sweep straightforward.

**Express one quantity once.** The correction clamp was `round(db × 100)` millibels in
`levels.py` and `10 ** (db / 20)` in `envelope.py`. Identical for whole decibels, which is why
`test_the_clamp_bounds_every_gain_by_construction` asserted the relationship at 6.0 dB and
passed. For `max_level_correction_db: 0.015` the millibel version is the larger, a track's own
permitted correction breached the bound the runtime checker enforces, and the mix stage failed
on an ordinary session with an invariant violation. Millibels are the project's unit because
the graph carries no floats; convert once, at the boundary.

**A cursor over sorted spans is not a position.** `_targets` advanced its per-track cursor past
any span it finished with and stopped scanning at the first span running past the chunk end.
With two overlapping candidates on one track the second was skipped in the chunk the first
straddled and applied in the next, so the envelope depended on the caller's window size — and
the cache identity carries no window size. M3's merge makes it unreachable, which is exactly
why it survived: the code defended against overlap with `np.maximum`, and the defence was
incomplete in a way only a partition test over an overlapping graph could see. The cursor is
now a lower bound and the scan covers every span touching the chunk.

**FFmpeg reports −70.0 LUFS for digital silence, not −inf.** Measured, not assumed —
`test_mix_loudness.py` pins it. A silence guard written as `is None` would never fire and
would lift a session nobody spoke in by the full master-gain clamp. But `Peak: -inf dBFS` *is*
what it prints for a silent file's true peak, so `None` there is a real measurement infinitely
below any ceiling, while an absent `Peak:` line means `peak=true` did not take effect and
nothing measured anything. `Measurement.true_peak_reported` is that distinction and it is
load-bearing: without it, a decode with no measurements at all was "compliant".

**`framelog=quiet` is not cosmetic.** Without it `ebur128` prints one line per 100 ms — 144 000
lines for a four-hour session — and a subprocess whose stderr pipe fills while its caller reads
stdout deadlocks. stderr also goes to a temp file, belt and braces.

**Independence in `process` is control flow, not ordering.** A sequential mix-first
implementation that lets a mix exception propagate satisfies every sentence the spec writes
about *transcription* failing, and violates the requirement anyway. Each branch has its own
handler; four of the six `process` tests exist to say so from both directions.

**`_failed` must not stamp its exception over a stage that already diagnosed itself.**
ADR-0024 said so and the code did it anyway, so a mix that fully succeeded was reported
`MIX: failed` with a source-tampering message. M7a deliberately must not add archive as a
third `process` branch; if some later milestone does, add it to the `owned` map in the same
breath as its `_State` field.

**The report must never claim a tolerance it did not check.** `mix_encoded` used to say
"within every configured tolerance" on runs where the loudness comparison had been waived. The
spec's whole point in that clause is that a compliance claim nobody can audit is worth less
than a failure that names the numbers — which applies to prose in the report as much as to the
exit code.

**Where the audit trail lives.** M5 publishes no deterministic document of its own (ADR-0022),
so the report's `decisions` subsection *is* it: `mix_level_correction` per track,
`mix_intermediate`, one `mix_encode_attempt` per attempt including the failing ones, and
`mix_encoded`. INV-02 requires that subsection to be semantically stable, so nothing per-run
belongs in it — `from_cache` was in there and had to come out.

### Deviations from this charter, and why

1. **The spec was amended.** Acceptance criterion 8 and the milestone-5 body now say the
   true-peak ceiling outranks the loudness target. Loudness and peak can be mutually
   unreachable — high-crest-factor material cannot reach −16 LUFS without clipping, and the
   canonical fixture through real Silero is exactly that case — so honouring the ceiling,
   encoding without normalization, and warning is the only reading that does not throw away a
   good mix. The code already behaved this way; the spec did not say so. ADR-0023 and the gate
   above are amended in the same commit. Raised by the code review, decided by the operator.
2. **`overlap_min_gain_db` moved from −15.0 to −16.0**, and gained a validator. The old value
   was an estimate ("two channels share roughly −6 dB each, and the clamp can take another
   6") that ignored the score asymmetry and the four room-tone floors; the rule's actual bound
   for six tracks is −15.66 dB. Derived now, from `guaranteed_overlap_gain_db`.
3. **`TestGenuineOverlap` does not use `mutual_bleed_session`**, which the working plan
   promised. The worst-admissible-pair test is a stronger proof of the same criterion, and
   building a second session's graph would have re-tested the fixture rather than the rule.
   The row says so.
4. **`activity.bleed.compare_pairs` stays quadratic.** The charter's risk note said "if M5
   walks the candidate set at scale and it hurts, that is the fix". M5 walks the retained
   candidates once, linearly, to build the presence signal; it never enumerates pairs. Still no
   measured evidence, and the real candidate count is unknown until a full live run.
5. **No `work/mix.json` and no new schema**, per the plan review — a new artifact with no named
   consumer is a choice made on behalf of milestones that have not asked. Byte-stability is
   proved on the intermediate itself, which is a stronger thing to prove it on.
6. **Five proof-table rows named tests that did not exist under those names.** Every proof
   existed; the names had drifted during implementation. Reconciled above. One row was worse
   than a rename — see deviation 3.

### Downstream charters updated

- **M6b — Qwen adapter.** `process` now composes the transcript branch through
  `perform_transcript`/`resolve_models` rather than reimplementing it, so M6b's real adapter
  reaches both `transcribe` and `process` by replacing one seam. The `DEFERRED: M6b` path is
  raised *before any work* in both commands.
- **M7b — Publishing and reclamation.** Two things land in its lap: nothing sweeps
  `work/cache/mix/`, and a session mixed under two different detectors keeps two
  intermediates. Noted in its charter; M7a raw backup does not need cache authority.
- **ROADMAP.md** — M5's gate line now names the ceiling-over-target rule, so the roadmap and
  the amended spec agree.

### Next smallest step

**M6a — ROCm environment.** It is the only milestone whose dependencies are all closed, and it
is pure environment work: the `gfx1151` Torch wheel index wired into uv with per-package
sourcing, the FHS shell for the `rocm[libraries]` sdist (ADR-0002), locked versions, and
`doctor` device checks that open `/dev/kfd` and the render node rather than inferring from
their existence. Nothing in M0–M5 depends on it, and M6b cannot start without it.

Start with `doctor`, not with the wheel index: the device checks are what tell you whether the
wheel index worked, and writing them second means debugging two unknowns at once.
(Claude Code: `/ms-start 6a`.)
