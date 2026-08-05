# M9 — Transcript assembly quality

**Status:** closed
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

- [x] `transcript.leading_ownership_grace_ms` defaults to 20 ms, is applied only after ASR,
      cannot extend beyond audio actually submitted as request padding, and does not alter the
      activity graph, request audio, request identity, ASR cache key, or mix.
- [x] Effective ownership remains deterministic and half-open. A returned word start can be
      owned at most once across merged requests, adjacent candidates, long-candidate divisions,
      and truncation/retry seams; session start and clipped request padding are covered.
- [x] Candidate/activity ownership and effective transcript ownership are both retained as
      candidate/piece-specific lineage in `work/transcript-records.json`, including the
      submitted padded bounds that constrain each occurrence. Aggregate ranges must not hide
      gaps, predecessor clipping, or retry seams; each aligned word start resolves to exactly
      one effective interval.
- [x] The existing three-condition similarity collapse produces the same complete first-pass
      survivor verdicts and decision representation as before. Only its remaining survivors
      enter a separately auditable `contained_fragment` pass, which requires substantial
      overlap, graph evidence for the full Cartesian product of contributing candidate pairs,
      at least 300/1000 source-score dominance, and proper contiguous normalized-word
      containment by the acoustically preferred survivor. If the second pass
      removes a first-pass survivor, records preserve the original edge in an acyclic audit
      chain that terminates at the retained contained-fragment winner.
- [x] Negative collapse tests retain genuine overlap, unrelated text, pairs with absent graph
      evidence, weak dominance, a shorter segment with the better source score, and exact short
      `Yes`/`Okay` matches.
- [x] Adjacent retained records from the same track may coalesce into one public presentation
      turn only when they share request lineage **and** their exact-sample gap is within an
      independently named presentation threshold (provisionally 350 ms, owned by OQ-018;
      the measured target split has a 320 ms word gap).
      `transcript-records.json` stays granular; `transcript.json` and `transcript.md` use the
      same deterministic grouping and retain source-record and source-candidate lineage.
- [x] Presentation joining never crosses a retained intervening speaker, an overlap-marked
      record, a track change, an alignment-status change, or records with no shared request
      lineage. Public overlap is recomputed from the coalesced exact-sample intervals, including
      a different-speaker segment spanning the joined pieces. ADR-0017 remains true: requests
      merge; ownership does not, and request batching alone never defines a turn.
- [x] Transcript assembly semantics are bumped. Records/public lineage uses only optional
      additive schema fields with explicit old-record fallbacks, unless an ADR justifies an
      artifact schema-version bump; required fields are not added silently to schema version 1.
      ASR request semantics stay unchanged. A direct identity test excludes assembly semantics
      from the ASR key, and a nonempty warm run proves all submissions hit while raw ASR files
      and sidecars remain byte-identical.
- [x] Deterministic CPU/offline tests cover the new behavior and would fail on its removal;
      no default test reads sample audio, needs model weights, imports Torch, or reaches the
      network.
- [x] The four-file cached evaluation is rerun where useful with source hashes verified before
      and after. It retains both short `Okay` records, contains every intended phrase, removes
      the long contained bleed fragments, and renders the last same-track phrase coherently.
- [x] `./scripts/codex-review.sh plan M9` runs before implementation; findings are distilled
      and dispositioned. `./scripts/codex-review.sh code M9 main` runs before close and every
      accepted finding has a regression proof.
- [x] `./scripts/gate.sh` passes with zero skips, and the same default suite passes from
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

Transcript assembly now gives each post-ASR word occurrence a deterministic, half-open
effective ownership interval. The default 20 ms leading grace is clipped by the exact audio
submitted for that occurrence and by the preceding interval on the track. Original candidate
ownership and request identity remain unchanged. Resolved truncation retries retain their leaf
submission plans, so a child's words can be claimed only through that child's sliced ownership
and padded bounds; atomic fallback retains the parent occurrence.

Duplicate collapse still completes the pre-M9 three-condition pass first. A separately audited
`contained_fragment` pass may then remove a proper contiguous word-sequence fragment only when
the longer segment is the acoustic winner, all contributing candidate pairs have graph
evidence, overlap is substantial, and source dominance reaches 300/1000. Exact matching text,
including short `Yes` and `Okay`, cannot enter this path. A pre-M9 golden produced by executing
`main` at `9421d03` freezes every legacy verdict and decision field for representative cases.

The authoritative transcript records remain granular. JSON and Markdown share one deterministic
presentation iterator that may join adjacent compatible same-track records within 350 ms when
they share request lineage, carries every source record/candidate id forward, and recomputes
public overlap from the joined exact-sample intervals. Assembly semantics are version 3; ASR
request semantics and cache identity remain version 1, and the activity graph and mix are not
inputs to these presentation decisions.

On the four-file fixed-response evaluation, 16 plans yielded 17 drafts, 18 dropped pairs,
seven retained granular records, and six public turns. All four announced openings survive,
the long/suffix bleed fragments do not, and the final two same-track records render as one
phrase. Both acoustically unresolved one-word `Okay` records remain visible.

### Tests and commands run, with results

- Baseline `./scripts/gate.sh` at `9421d03` — **8 checks, 2 360 passed, zero skips**.
- `./scripts/codex-review.sh plan M9` before implementation — five P1 and two P2 findings;
  all accepted, incorporated into the charter, and recorded in the distilled plan review.
- `./scripts/codex-review.sh code M9 main` before close — two P1 findings and one P2 finding;
  all accepted and fixed. The reviewer ran its focused transcript/process selection with
  **263 passed**; its read-only sandbox declined the full gate.
- Post-review focused transcript, cache, retry, rendering, config and process suites —
  **510 passed**; Ruff, mypy and the plan ledger checker passed.
- Final `./scripts/gate.sh` — **8 checks, 2 397 passed, zero skips; gate passed**.
- Restricted gate attempts first lacked the Nix C++ runtime outside `direnv`, then passed all
  repository checks except the five socket-guard tests because the execution sandbox refused
  socket construction. The identical `direnv` gate above passed in the normal project
  environment where those tests can exercise the application's own network block.
- `nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm
  UV_CACHE_DIR=/tmp/dnd-audio-uv-cache uv run --no-sync pytest -m
  "not host_smoke and not allow_network" -q'` — **2 397 passed** from `.venv-rocm`.
- Isolated replay of the four cached ASR response sets — exact target above; source `raw/`
  hashes matched before and after, and the activity and mix content hashes remained
  `16403d16...a8d` and `621121cb...132` respectively. Full hashes and method are in the
  committed evaluation note.

### Decisions made (→ ADRs)

- **ADR-0033** — transcript-only leading ownership is an occurrence-specific post-ASR
  partition; retry leaf submissions remain distinct; conservative contained-fragment collapse
  is a second pass with full Cartesian graph evidence and no exact-text path.
- **ADR-0034** — granular records are authoritative while JSON and Markdown may expose the
  same lineage-preserving presentation turns under a separately named gap threshold.
- **ADR-0032 amended** — the ASR/assembly cache boundary now explicitly covers the new
  assembly semantics. Assembly changed from version 2 to 3; ASR semantics did not change.

### Assumptions made and open questions raised

- **OQ-027 remains open.** Twenty milliseconds is the smallest useful leading grace in this
  one-operator capture, not yet a multi-wearer calibration. It is bounded by submitted padding
  and has no activity-side effect.
- **OQ-018 remains open.** The 300/1000 containment margin and 350 ms presentation gap are
  conservative defaults supported by this capture. Exact-short duplicate safety and natural
  conversational pauses still need genuine multi-speaker evidence.
- No new open question was needed: the empirical claims belong to OQ-018/OQ-027, and both now
  name the M9 evidence and remaining M11 live-session work.

### Notes for future implementors

Do not flatten request lineage. `request_ids` includes attempted submissions for audit, but
effective ownership must be built from `contributing_submissions`: the leaf plan that actually
returned each word, or the parent only after atomic fallback. Projecting stitched child words
through the parent recreates the cross-seam ownership bug found in code review.

For a segment with multiple contributing candidates, "every candidate appears somewhere" is
not enough acoustic evidence. Containment deletion requires the complete Cartesian set of
candidate pairs. Keep the pre-M9 similarity pass global and complete before considering any
new containment edge; otherwise an early containment decision can change a legacy survivor.

Public turns are a view, not new authoritative segments. Any future renderer must use the
shared presentation iterator or reproduce its exact grouping, lineage and post-group overlap
semantics. Do not use the 1.5 s request batching gap as a conversational boundary.

The reusable fixed-response evaluation and listening material remain under the operator's
isolated scratch directory and `/tmp`; neither belongs in the repository. The committed note
contains hashes and aggregate results without host-specific paths or audio.

### Deviations from this charter, and why

- Code review found that the first implementation retained only a resolved retry's parent plan.
  M9 therefore added explicit leaf contribution lineage and occurrence-specific assembly before
  closing; this strengthens the charter's retry-seam requirement without widening scope.
- Code review also found that candidate coverage was weaker than the charter's intended full
  Cartesian evidence. The predicate and a 2-by-2 negative/control regression were corrected.
- The representative legacy freeze became an independently generated complete golden rather
  than assertions assembled under the branch implementation. This is the stronger proof the
  plan intended.
- No artifact schema-version bump was needed. All record/public lineage fields are optional
  additive schema-version-1 fields with explicit old-record fallbacks, as ADR-0034 permits.

### Downstream charters updated

- **M11** measures natural multi-wearer hard onsets at 0/20/100 ms grace, exact simultaneous
  `Yes`/`Okay`, contained fragments, and controlled 320/350 ms same-speaker pauses while scoring
  granular records and public turns separately.
- The same live-session evaluation covers natural speech and truncation/retry seams, checks
  contained-fragment audit chains, and keeps exact-short matches until evidence supports more.
- **OPEN-QUESTIONS.md**, the product spec, ADR-0032 and both new ADRs carry the semantic and
  empirical boundaries. The activity defaults remain 30 ms padding and 200 ms merge gap.

### Next smallest step

Record and process live Session Zero. Use the production activity defaults, keep both unresolved short
copies in the baseline, and compare the transcript-only controls against known wearer/phrase
ground truth. M9 has no remaining software or documentation work.
