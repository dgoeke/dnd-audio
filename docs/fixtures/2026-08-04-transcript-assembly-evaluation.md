# Transcript assembly evaluation — 2026-08-04

This is a fixed-response evaluation of M9 against the four-file jam-verification capture. It
is a limited capture: one operator announced microphones while holding one at a time, so the
capture contains no ground-truthed genuine multi-speaker speech. It can prove recovery and
bleed behavior for these utterances; it cannot authorize aggressive exact-short collapse.

## Method

The evaluation used the production 30 ms VAD-padding and 200 ms activity-merge graph from an
isolated prior run. It replayed the already cached ASR response documents through the M9
assembly, collapse and rendering code. No model ran, no activity candidate changed, and no
session `raw/` file was written.

The four source hashes were computed before and after the evaluation and matched exactly:

```text
589951badc07c531be38f95070b1325a3b83bbb6d94816b41c0c2dec3b0579c7
705ae841e612aef4d125a3f3cf73ab3f78477478538f2fd6c6c5486f36a64bcb
bb85cdeed2f0b84e3a1e5843f098b4cf5a39d84624bbe51a193c4b3d69c9e827
27a1cc9b86440d208f1a84ba17d53a38b92710b5fb1d8cfec4a9b91f239d8f87
```

The fixed activity graph hash remained
`16403d16a2df3cf64448a816c922ff3667e9de1c3ef2006fabfa592466a7fa8d`; the fixed mix
intermediate hash remained
`621121cbcd9461f13de5de438e44cb7a349f65ba02c16834448d00bf481e4132`.

## Result

Sixteen planned requests and their cached responses produced 17 drafts and 18 dropped
request/word pairs with the default 20 ms leading transcript-only ownership grace. Conservative
collapse retained seven granular records. Presentation-only joining coalesced the final two
same-track records, yielding six public turns:

1. `Hello Now I'm talking into the first microphone`
2. `Hello This is me talking into the second microphone`
3. `Okay` (receiver 2)
4. `Okay` (receiver 1)
5. `Now I'm talking into the third microphone`
6. `Finally here's the fourth microphone`

All four direct utterance openings are present. The long and suffix bleed fragments are absent.
The final phrase retains two granular segment ids and candidate/request audit lineage while the
JSON and Markdown views render it as one turn. Both exact one-word `Okay` records remain: their
648/1000 correlation and 39/1000 source-score difference do not satisfy the contained-fragment
rule, and exact text is excluded from that rule in any case.

The result meets the conservative target for this capture: intended words are complete,
activity and mix are unchanged, and unresolved evidence remains visible. M11 still audits
genuine overlap, multi-wearer hard onsets and natural conversational pauses before the
20 ms grace, 300/1000 containment margin or 350 ms presentation gap can be treated as broadly
calibrated defaults.
