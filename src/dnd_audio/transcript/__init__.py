"""The transcript branch — M4's stage.

M3 decided *who was speaking* without reading a word, and froze that graph (INV-09). This
package is everything downstream of it that reads words, and nothing here may flow back:
the mix must produce identical samples whether or not ASR ever ran.

The modules follow the data:

* :mod:`~dnd_audio.transcript.requests` turns retained activity candidates into transcription
  requests — merging adjacent regions so the model hears sentences rather than fragments,
  padding them so it does not clip the first and last word, and capping the padded waveform.
  **Merging joins the audio; it does not join ownership** (ADR-0017).
* :mod:`~dnd_audio.transcript.asr` submits them, handles a truncated response by splitting and
  retrying under a bounded budget, and assigns each returned word to the ownership interval
  that contains its start (ADR-0020).
* :mod:`~dnd_audio.transcript.cache` is the per-request ASR cache and the versioned raw
  artifact the spec requires before any normalization.
* :mod:`~dnd_audio.transcript.normalize` is the *only* text processing in this project:
  deterministic whitespace and punctuation, never an LLM cleanup pass.
* :mod:`~dnd_audio.transcript.collapse` decides which segments are the same utterance heard
  on two lavs, and marks the rest as overlapping.
* :mod:`~dnd_audio.transcript.render` turns records into `transcript.json` and `transcript.md`.
* :mod:`~dnd_audio.transcript.runner` orchestrates, and owns `transcribe` and `render`.

Two rules hold everywhere in here. **Losing real speech is the worst outcome** — it is why
collapse needs three independent kinds of evidence and why an unresolved truncation keeps the
original response rather than a tidier partial one. And **the audio is the 16 kHz derivative**
(ADR-0017): one grid, already cached and byte-stable, converted to canonical 48 kHz session
samples through M2's own helper at the boundary and nowhere else.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ASR_DIRNAME",
    "FAKE_MODELS_FILENAME",
    "RECORDS_RELATIVE_PATH",
    "TRANSCRIPT_CACHE_DIRNAME",
    "TRANSCRIPT_JSON_RELATIVE_PATH",
    "TRANSCRIPT_MARKDOWN_RELATIVE_PATH",
    "TRANSCRIPT_SEMANTICS_VERSION",
]

#: Session-relative. The normalized records, and the only thing `render` reads (ADR-0019).
RECORDS_RELATIVE_PATH: Final = "work/transcript-records.json"

#: The two deliverables. `output/` is where the spec puts generated deliverables.
TRANSCRIPT_JSON_RELATIVE_PATH: Final = "output/transcript.json"
TRANSCRIPT_MARKDOWN_RELATIVE_PATH: Final = "output/transcript.md"

#: Everything under here is regenerable cache; deleting it costs a re-run and nothing else.
TRANSCRIPT_CACHE_DIRNAME: Final = "work/cache/transcript"

#: Per-request ASR results, keyed by everything that could change one (INV-08).
ASR_DIRNAME: Final = f"{TRANSCRIPT_CACHE_DIRNAME}/asr"

#: A session's declared fake model outputs, beside `session.yaml` and never under `raw/`.
#: Written by the fixture generator, read only under an explicit flag (ADR-0018).
FAKE_MODELS_FILENAME: Final = "fake-models.json"

#: Bumped when **any** module in this package changes what it would produce from unchanged
#: inputs — how requests are cut, how words are assigned, how text is normalized, or what
#: collapses. One version for the package, for the reason M1, M2 and M3 all give: a cache
#: identity that varied one module's version but not another's keeps serving the answer a
#: fixed bug produced.
TRANSCRIPT_SEMANTICS_VERSION: Final = 1
