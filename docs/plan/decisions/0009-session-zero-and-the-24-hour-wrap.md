# ADR-0009 — Session zero, and the 24-hour wrap in each evidence domain

**Status:** accepted
**Date:** 2026-08-02
**Milestone:** M2

## Context

The spec says two things about where a session begins and one thing it does not finish
saying.

It says session time zero comes "from the earliest valid source start time unless the
config supplies an explicit origin", and that `rollover_policy: infer_forward` "may infer
a single forward midnight rollover only when chunk sequence and session-span constraints
make it unambiguous, and must record that decision". It also says, in one sentence with no
elaboration: **"Define 24-hour wrap handling."**

That sentence is the whole difficulty. The plan's first draft read it as "add a day", and
independent review (`docs/plan/reviews/M2-plan-20260802-1241.md`) showed that a day is not
one quantity. Verified:

| Domain | One 24-hour cycle | In real seconds |
| --- | --- | --- |
| BWF `time_reference` | `86400 × file_rate` samples | 86 400 exactly |
| Timecode at 24F / 25F / 30F / 50F / 60F | 2 592 000 … frames | 86 400 exactly |
| Timecode at **29.97F or 23.98F non-drop** | 2 592 000 frames | **86 486.4** |
| Timecode at 29.97DF | 2 589 408 frames | 86 399.9136 |

Non-drop fractional timecode exists precisely because it *does not* track wall time; drop
frame exists to pull it back. So a wrapped chunk at 29.97F, unwrapped by adding 86 400
seconds, lands 86.4 seconds before the frame that preceded it — producing a large false
overlap out of correct evidence, and threatening both INV-04 and INV-12.

Two further questions arrived with the same review. The spec explicitly permits a
**signed** `start_offset_samples`, so a blanket "audio before session zero is fatal" would
quietly delete half of that field's range. And a session mixing evidence domains needs a
stated reconciliation rather than whichever one the loop happened to see first.

## Decision

### The wrap is unwrapped in the evidence's own domain

Rollover adds **whole cycles in the units the evidence is counted in**, before any
conversion to seconds:

- a BWF sample reference gains `86400 × sample_rate` samples;
- a timecode gains one code cycle **in frames** — `frame_index` of `24:00:00:00` at that
  rate, which is 2 592 000 non-drop and 2 589 408 at 29.97DF;
- a session offset gains nothing, because it is already relative to session zero and has
  no midnight in it.

Conversion to exact seconds (ADR-0008) happens after unwrapping, never before.

### Session zero

1. **`timecode.origin_timecode` with `origin_date`** — session zero is that instant.
   `origin_date` is never inferred from a date-shaped `session_id`; the spec forbids it and
   the two can legitimately differ.
2. **Otherwise, the earliest valid source start.** Placements are computed in an
   intermediate coordinate system and the whole timeline is shifted so the earliest lands
   at zero.

### Audio before session zero

- **With an explicit origin**, a source placed before zero is **fatal**. The operator
  asserted where zero is; silently truncating audio, or silently moving their origin, are
  both worse than an error naming the source and how far before zero it starts.
- **Without an explicit origin**, it cannot happen: zero *is* the earliest start. This is
  what keeps a negative `start_offset_samples` meaningful — the offsets form a relative
  coordinate system whose distances are exact, and only its origin is unknown.

### Mixed evidence

Absolute evidence (BWF references and timecodes) determines session zero; session-relative
offsets are then placed at `zero + offset`. When a session contains no absolute evidence
at all, rule 2 applies to the relative set. A relative offset that contradicts an origin
fixed by absolute evidence — by placing audio before it — is the fatal case above.

At a rate where a timecode day is not 86 400 seconds, a session that mixes BWF and
timecode evidence rests on an assumption about where the recorder's timecode was jammed
relative to real midnight. That is recorded as a warning and as **OQ-015**, not silently
assumed.

### When a rollover may be inferred

Under `infer_forward`, exactly when both hold:

- the same-day reading would place the chunk **before session zero**, and
- the resulting session span stays **within one cycle** — beyond that the mapping is
  genuinely not unique, which is the mathematical definition of the ambiguity the spec
  refuses to resolve by guessing.

**With no configured origin there is no such anchor, and the rule is weaker than this
document originally claimed.** The first version of this ADR said the widest-gap analysis
made the ordering "unambiguous". It does not. Starts at 23:00 and 01:00 admit two readings
— a two-hour session across midnight, or a twenty-two-hour session within one day — and
the algorithm picks the first because it is the **shortest arc** containing every start,
not because the evidence excludes the second. Independent review caught the overstatement
(`docs/plan/reviews/M2-code-20260802-1508.md`).

That is still the right default: sessions are short, and the spec's own wording points at
"session-span constraints". But it is a *heuristic about how long a session is*, so it is
registered as **OQ-016**, cited from the code that applies it, and every session that
relies on it gets a `midnight_rollover_inferred` warning saying the day was inferred rather
than read. An operator who records `origin_date` and `origin_timecode` never meets it.

Every inference is recorded as a decision naming the source, the cycle added, and the
resulting position. `reject` never infers and fails with the same diagnostic. A span that
is unambiguous but implausibly long warns rather than failing (**OQ-014**).

**Rollover inference reads time evidence only.** It never consults DJI's `MIC###` filename
counter. The charter's phrase "chunk sequence" means the chunks' order *in time*; reading
it as the filename sequence would make timing depend on a filename, which INV-12 forbids
and which `inspection.starttime.SourceContext` is deliberately shaped to prevent. M2's
rollover therefore does **not** depend on OQ-003.

## Alternatives considered

- **Normalize every domain to real seconds first, then unwrap by 86 400.** The rejected
  first draft. Wrong by 86.4 s at 29.97 non-drop, and wrong in a way that looks like a
  recording problem rather than an arithmetic one.
- **Convert timecode to a wall-clock datetime and let a date library carry the day.** Same
  bug wearing a library's clothes: `timedelta(days=1)` is 86 400 seconds no matter what
  the frame rate is.
- **Forbid mixed evidence domains in one session.** Tempting, and it would fail the
  canonical fixture, which mixes a BWF reference on five tracks with an `INFO`/`ISMP`
  timecode on `tx-f` — a shape the hardware can genuinely produce.
- **Allow negative session positions** and render output bounds from the minimum. Rejected:
  every downstream consumer, `interfaces.AudioWindow` included, treats a sample index as
  non-negative, and a negative timeline is a large change to serve a case the shift rule
  already handles exactly.
- **Infer more than one rollover.** Rejected: past one cycle the unwrap is not unique, and
  the spec says a *single* forward rollover.

## Consequences

- Rollover is tested crossed with rates rather than once: 23.98F, 29.97F, and 29.97DF each
  wrap by a different number of real seconds, and only a domain-correct implementation
  passes all three.
- A signed `start_offset_samples` keeps its full documented range without a spec
  amendment.
- Sessions are still capped at one 24-hour cycle. A recording that genuinely spans longer
  needs a dated origin, which is the actionable diagnostic the failure prints.
- The shortest-arc rule is a heuristic and is named as one (**OQ-016**). A session that
  genuinely spans more than half a day with no configured origin will be read as a short
  session across midnight, which is wrong. Nothing in the evidence distinguishes the two;
  a dated origin does, and the diagnostic says so.
- If H1 shows DJI stamps an origination *date* on every chunk (OQ-001), most of the
  inference disappears: a dated chunk needs no rollover reasoning at all, and this ADR's
  inference rules become the fallback rather than the common path.
