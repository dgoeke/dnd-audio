# M9 — Transcript assembly quality

**Status:** in progress
**Depends on:** M8, and the four-file local transcript-quality evaluation
**Spec sections:** Milestone 3 (post-ASR duplicate collapse); Milestone 4 (ownership and
stitching); Output schemas; Tests and acceptance criteria

## Goal

Recover utterance-opening words that the aligner places just before a VAD ownership edge,
remove only acoustically compelling contained bleed fragments, and present adjacent pieces of
one retained turn coherently. These are transcript semantics: the activity graph, ASR requests,
cached model responses, and automix remain unchanged.

The four-file evaluation establishes the bounded default, not a general permission to optimize
this one corpus. The exact short `Okay` ambiguity remains in the transcript until genuine
multi-speaker evidence can distinguish repetition from bleed.

## Evidence entering the milestone

- Production activity settings (`vad.pad_ms: 30`, `merge_gap_ms: 200`) yielded 30 dropped
  request/word pairs, 10 rendered lines, and clipped all four announced microphone openings.
- Raising activity padding to 80 ms recovered the openings but produced 12 lines and moved
  speech references by as much as 1.08 dB. Raising the activity merge gap to 300 ms joined one
  phrase but moved one reference from about -40.77 to -59.86 dBFS and altered the mix clamp.
  Neither activity-side change is admissible here.
- With the 30 ms activity graph and cached ASR fixed, 20 ms of leading transcript-only grace
  recovered all four openings and reduced dropped pairs from 30 to 18. Values through 80 ms
  changed no intended content; 100 ms claimed three additional weak-track words.
- A conservative prototype removed long/suffix fragments only when the acoustically preferred
  segment properly contained the weaker segment's normalized words, substantially overlapped
  it, had graph evidence, and led by at least 300/1000 in source score.
- The two exact one-word `Okay` segments have 648/1000 correlation but only 39/1000 source-score
  separation. This corpus has no genuine multi-speaker overlap, so collapsing one would be an
  unsupported deletion.

## Completion gate

- [ ] `transcript.leading_ownership_grace_ms` defaults to 20 ms, is applied only after ASR,
      cannot extend beyond audio actually submitted as request padding, and does not alter the
      activity graph, request audio, request identity, ASR cache key, or mix.
- [ ] Effective ownership remains deterministic and half-open. A returned word start can be
      owned at most once across merged requests, adjacent candidates, long-candidate divisions,
      and truncation/retry seams; session start and clipped request padding are covered.
- [ ] Candidate/activity ownership and effective transcript ownership are both retained as
      candidate/piece-specific lineage in `work/transcript-records.json`, including the
      submitted padded bounds that constrain each occurrence. Aggregate ranges must not hide
      gaps, predecessor clipping, or retry seams; each aligned word start resolves to exactly
      one effective interval.
- [ ] The existing three-condition similarity collapse produces the same complete first-pass
      survivor verdicts and decision representation as before. Only its remaining survivors
      enter a separately auditable `contained_fragment` pass, which requires substantial
      overlap, graph evidence for the full Cartesian product of contributing candidate pairs,
      at least 300/1000 source-score dominance, and proper contiguous normalized-word
      containment by the acoustically preferred survivor. If the second pass
      removes a first-pass survivor, records preserve the original edge in an acyclic audit
      chain that terminates at the retained contained-fragment winner.
- [ ] Negative collapse tests retain genuine overlap, unrelated text, pairs with absent graph
      evidence, weak dominance, a shorter segment with the better source score, and exact short
      `Yes`/`Okay` matches.
- [ ] Adjacent retained records from the same track may coalesce into one public presentation
      turn only when they share request lineage **and** their exact-sample gap is within an
      independently named presentation threshold (provisionally 350 ms, owned by OQ-018;
      the measured target split has a 320 ms word gap).
      `transcript-records.json` stays granular; `transcript.json` and `transcript.md` use the
      same deterministic grouping and retain source-record and source-candidate lineage.
- [ ] Presentation joining never crosses a retained intervening speaker, an overlap-marked
      record, a track change, an alignment-status change, or records with no shared request
      lineage. Public overlap is recomputed from the coalesced exact-sample intervals, including
      a different-speaker segment spanning the joined pieces. ADR-0017 remains true: requests
      merge; ownership does not, and request batching alone never defines a turn.
- [ ] Transcript assembly semantics are bumped. Records/public lineage uses only optional
      additive schema fields with explicit old-record fallbacks, unless an ADR justifies an
      artifact schema-version bump; required fields are not added silently to schema version 1.
      ASR request semantics stay unchanged. A direct identity test excludes assembly semantics
      from the ASR key, and a nonempty warm run proves all submissions hit while raw ASR files
      and sidecars remain byte-identical.
- [ ] Deterministic CPU/offline tests cover the new behavior and would fail on its removal;
      no default test reads sample audio, needs model weights, imports Torch, or reaches the
      network.
- [ ] The four-file cached evaluation is rerun where useful with source hashes verified before
      and after. It retains both short `Okay` records, contains every intended phrase, removes
      the long contained bleed fragments, and renders the last same-track phrase coherently.
- [ ] `./scripts/codex-review.sh plan M9` runs before implementation; findings are distilled
      and dispositioned. `./scripts/codex-review.sh code M9 main` runs before close and every
      accepted finding has a regression proof.
- [ ] `./scripts/gate.sh` passes with zero skips, and the same default suite passes from
      `.venv-rocm`.

## Explicitly not in this milestone

- Changing `activity.vad.pad_ms`, `activity.vad.merge_gap_ms`, any speech-reference logic, the
  activity graph, or the mix.
- Collapsing exact short utterances, lowering the existing similarity length floors globally,
  or treating this one-operator corpus as evidence about genuine simultaneous speakers.
- Joining candidate records in `work/transcript-records.json`, changing their segment ids, or
  erasing their request/candidate lineage to make the public transcript tidier.
- Re-running Qwen merely because assembly or presentation semantics changed.

## Known risks and open questions

- **OQ-018** owns the model-boundary assumptions and duplicate-text safety. This milestone
  narrows item 4 with contained-fragment evidence but does not answer exact-short overlap. It
  also owns the provisional presentation-gap default: shared batching lineage is only one
  prerequisite and is not conversational-boundary evidence by itself.
- **OQ-027** gains a bounded transcript-only remedy for the measured leading-word failure.
  Multi-wearer evidence is still required before changing any activity default.
- **INV-02**, **INV-04**, **INV-08**, and **INV-09** are the primary risks: presentation must
  stay byte-stable and time-exact, assembly changes must not poison/recompute ASR, and no text
  decision may reach the mix.

## Working plan

1. Write ADRs before source changes: one for post-ASR leading ownership and contained-fragment
   collapse, and one for granular records versus coalesced public presentation. Amend the spec
   wherever its baseline semantics need the distinction.
2. Add the grace setting and a canonical assembly-owned interval partition in derivative
   samples. Preserve candidate/piece-specific activity, effective, request, and padded bounds;
   determine predecessors globally per track so two outcomes cannot claim the same half-open
   sample. Apply the transform after `transcribe_plans`, before draft construction and
   dropped-word diagnostics, without overwriting `RequestOutcome.plan.ownership`. A resolved
   retry carries its retained leaf submissions separately so one child's words can be compared
   only with that child's sliced ownership and padded bounds.
3. Freeze representative legacy collapse verdicts and decision serialization. Run the complete
   existing similarity algorithm as the unchanged first global pass; only then run a second
   pass over its survivors for proper word-sequence containment, preferred-survivor direction,
   overlap, full Cartesian candidate-pair evidence, and the separately configured 300/1000
   dominance floor. Persist the rule name only for the new `contained_fragment` decision path,
   and preserve any completed
   first-pass edge as a terminating audit chain if its survivor loses in the second pass.
4. Build one presentation-turn iterator over retained records. Join only adjacent compatible
   records with shared request lineage and the separate exact-sample presentation-gap bound,
   keep the first record id as the public turn id, add optional additive lineage lists to public
   provenance, then recompute overlap across the final public intervals. Drive both JSON and
   Markdown from that iterator, with a spanning-other-speaker regression.
5. Prove cache scope directly and end to end. Assert the assembly version is absent from
   `asr_identity_document`; on a nonempty warm fake session require hits equal submissions,
   zero misses, and byte-identical raw ASR documents and sidecars. Also assert the cached mix
   intermediate's content hash, activity graph, and submitted request audio are unchanged.
   Exercise every boundary and negative-collapse case with deterministic fakes.
6. Re-run the cached four-file assembly with raw hashes before and after, compare the exact
   retained records and public lines with the stated target, then perform verify/review/close.

---

## Closeout

_Filled in during the close phase. Leave the headings; they are the checklist._

### What works end to end

### Tests and commands run, with results

### Decisions made (→ ADRs)

### Assumptions made and open questions raised

### Notes for future implementors

### Deviations from this charter, and why

### Downstream charters updated

### Next smallest step
