# ADR-0044 — Outline is the delivery surface, and publishing lives outside this repository

**Status:** accepted
**Date:** 2026-08-05

## Context

M7b was written as a sketch on the assumption that this repository would eventually grow a
publishing adapter: a provider boundary separate from M7a's archive, holding its own bucket
credentials, uploading the accepted processed artifacts to a Standard Space or a comparable
delivery surface. Nothing about that was decided — the charter listed it as deferred, precisely
because publication ergonomics cannot be designed honestly from four short synthetic files.

The owner has since written an independent implementation brief for the wiki host's NixOS
repository, describing the reader-facing product end to end. One Outline document per session:
`session.mp3` uploaded as a **private document attachment**, the transcript rendered as
**ordinary Outline document text** so the wiki's own search indexes it, one compact audio player
embedded near the top through a same-origin static route, and a clickable timestamp at the head
of every diarized turn that seeks the player. Attachment bytes live in a private S3-compatible
Space that Outline itself manages; retrieval is authorized by Outline's existing document
permissions.

That brief settles, from the consumer's side, most of what M7b had left open. Before adopting
it, four properties of this repository's output were checked against what it assumes:

- **The transcript interface matches exactly.** `schemas/transcript.schema.json` is
  `schema_version` const `1`, and every field the publisher reads — `start_s`, `end_s`,
  `speaker_name`, `text`, and `overlap` — is present, with `overlap` defaulting to `false`.
  M9's presentation joins mean `output/transcript.json` already carries coalesced turns rather
  than granular records (ADR-0034), which is the per-turn paragraph the publisher wants.
- **The MP3 is constant bitrate.** `mix/encode.py` passes `libmp3lame -b:a <n>k`, not a VBR
  quality target, so byte-offset seeking stays accurate across a multi-hour file. A VBR encode
  without a Xing header would have made a seek near the end of a four-hour session wrong by
  minutes while remaining perfectly correct in every short test.
- **The transcript and the mix share a time origin.** Transcript times are
  `_seconds(segment.start_sample, ...)` in the session-sample domain, and the mix renders from
  session sample 0 to the aligned duration. A timestamp link therefore addresses the same
  instant in the MP3. This was true by construction rather than by assertion, which INV-14 now
  fixes.
- **A 4–5 hour mono 128 kbps encode is roughly 275 MiB**, inside the brief's configured upload
  limit with about a factor of two to spare.

The spec's firm scope says "Do not read from or write to the campaign wiki." The brief places
the publisher in the **wiki host's** repository, not this one, so that line remains true.

## Decision

**The delivery surface is the owner's private Outline wiki.** Not a Standard Space written
directly by this project. Attachment bytes reach a private S3-compatible Space through Outline's
own storage configuration, under credentials this repository never holds.

**Publishing is implemented outside this repository.** This project gains no Outline API client,
no wiki credential, and no publish command. The spec's prohibition on touching the campaign wiki
stands unamended, and M7b's provisional plan steps for a publication provider adapter are
withdrawn. What this project owes the publisher is a set of accepted artifacts and a record
identifying them — nothing that speaks a delivery protocol.

**The published set is `session.mp3` and the transcript as native document text.**
`output/transcript.md` remains a local deliverable: the publisher renders its own Markdown from
`transcript.json`, with its own timestamp-link format, so shipping ours would publish a second
divergent rendering of the same content. `output/ingest-report.json` and `work/activity.json`
stay local as well. Attaching the supporting JSON to the document was considered and declined
for now — it is retrievable from the session directory and from the M7a archive, and every
published copy is one more artifact to re-upload whenever a render changes.

**Privacy is document-scoped, not URL-scoped.** The Space stays private and the document's
Outline permissions authorize retrieval. This closes the charter's public-read-versus-signed-URL
question in a third way: neither, because the reader is already authenticated to the wiki. No
published object is public and no expiry or rotation scheme is needed.

**Document and attachment identity are stable across re-publication.** A corrected render
updates the existing document and replaces the attachment only after its replacement is
confirmed, rather than minting a new URL per processing version. Provenance is carried by the
manifest below, not by URL proliferation.

**`transcript.json` schema version 1 is the contract between the two repositories.** This side
already has a schema-drift test. The publisher pins the version and fails closed on anything
else, so a future bump surfaces as a refusal to publish rather than as a mangled page.

**This project's remaining publication-side obligation is the output manifest.** One record,
emitted here, identifying an accepted run: session id, processing and schema versions, and for
each artifact its byte length, SHA-256, and media type. The publisher consumes it and echoes its
identity into its own publish report. Without it the two repositories hold complementary halves
of one record — this side knows what was produced, that side knows what was uploaded, and
neither can tell a newer render of the same session from a different recording, which is the
property M7b's charter demands.

## Consequences

- **M7b loses its largest deliverable and keeps the rest.** Publication implementation,
  provider adapter, and URL/privacy policy are settled or relocated. The output manifest, cache
  reclamation, remote retention, local raw reclamation, and the INV-01 question remain, and all
  of them still need M11's measurements.
- **Two cross-repository properties are now load-bearing and must not drift.** The shared time
  origin becomes **INV-14** with a test, because every timestamp link in the wiki is wrong by a
  constant if the mix ever gains a lead-in or a trim, and nothing would have caught it here.
  Constant-bitrate encoding is the other; `mix.mp3_bitrate_kbps` already refuses anything that
  is not an MPEG-1 Layer III bitrate, and a move to VBR would need its own decision.
- **The final shape of the pipeline is fixed.** Raw transmitter files go to M7a's private cold
  bucket, byte-exact and verified by full readback, and are never published. Processed output
  goes to the wiki, private and document-scoped. The two paths share no credential, no bucket,
  and no command — which is the separation M7b's safety properties already required, now
  enforced by the code living in different repositories.
- **The event-first transcript idea, if M11 ever justifies it, is a schema-version event.** A
  transcript beyond version 1 stops the publisher rather than silently changing what the wiki
  shows.
- Retention periods, remote lifecycle, deletion authority, cache budgets, and whether local raw
  reclamation should exist at all remain open and belong to M7b after a real processed session.

## The scope line survives this decision, but probably not the next one

The spec's "do not read from or write to the campaign wiki" is preserved here on a technicality
worth stating plainly: publishing a session **is** writing to the wiki, and the only reason this
project's scope line holds is that a different repository does the writing. The owner expects
that line to need relaxing eventually, in both directions, and it is better to name the shape of
that now than to have it argued from scratch later.

**Writing** is the smaller change. If a publish command ever moves into this repository, it
inherits everything ADR-0044 already requires — private attachments, document-scoped
authorization, stable identity, manifest-driven provenance — and adds only credential handling
and a network authority that no processing command may reach. It would be a convenience
relocation, not a new capability.

**Reading is the substantive one, and the spec already has the seam for it.** The intended use is
harvesting campaign proper nouns — names, places, factions — to bias ASR spelling. That input
already exists as `glossary.txt`, which the spec defines as "optional; local, no wiki dependency"
and whose absence must not block a run. So the future feature is not "ASR reads the wiki"; it is
a **separate, explicitly invoked command that writes a local glossary file**, after which nothing
about the transcription path changes. Constraints any such relaxation must respect, all of them
consequences of rules already in force rather than new ones:

- It is a third permitted kind of network traffic, and `AGENTS.md` says a third is almost
  certainly wrong. It would need the same treatment M7a's exception got: narrow wording amended
  in every place the prohibition is stated, and an ADR saying why the seam is safe.
- It runs off the processing path. `process`, `transcribe`, and every other command stay
  network-denied; the glossary is a file on disk by the time ASR sees it.
- The glossary is an ASR input, so it enters cache identity (INV-08). A wiki edit that changes
  the glossary must invalidate transcripts, which means the fetched content is hashed and
  recorded, not read live.
- It reads text and never sends audio, so INV-06's actual prohibition is untouched.

None of this is decided. It is recorded so that when it is, the decision starts from the seam the
spec already provides instead of from a new dependency in the transcription path.
