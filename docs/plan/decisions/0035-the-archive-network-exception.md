# ADR-0035 — The archive network exception, and the permission boundary under it

**Status:** accepted
**Date:** 2026-08-04
**Milestone:** M7a

## Context

This project has said "no audio leaves this machine" in four places since M0: the
working agreement's summary line and its hard rules, the spec's firm-scope section, and
the spec's target-host section. INV-06 restated it and named `models fetch` as the only
command permitted to open a socket at all.

M7a needs to upload audio. The risk it addresses is the one class of failure no later
software can repair — the original transmitter recordings are gone, the session is
gone, and there is nothing to re-run. Six milestones of pipeline work are worth nothing
against a failed disk. The first real session is imminent, so the backup has to exist
*before* it, not after.

The prohibition and the requirement are both correct, which means the wording was
wrong rather than the policy. "No audio leaves the machine" was always a proxy for the
thing actually being prevented: **audio reaching something that processes it**, and in
particular a cloud ASR service that would see every word spoken at a private table. An
encrypted-at-rest byte-exact copy of a WAV sitting in the owner's own private bucket is
not that, and the two were only ever conflated because until now there was no reason to
tell them apart.

The plan review for this milestone caught that the first draft amended two of the four
locations. Leaving the working agreement's hard rule in place while implementing upload
would have been a direct hard-rule violation with a green gate — exactly the shape this
repository's closeouts keep recording.

There is a second decision tangled with the first, and it is why the two are one ADR
rather than two. DigitalOcean bundles the ability to abort an incomplete multipart
upload with broad object Read/Write/**Delete** permission. The credential this project
uploads with therefore *can* delete a committed object, whatever the charter says. So
"we have no delete authority" is not a claim the provider's permission model supports,
and the honest boundary has to be drawn somewhere the project actually controls.

## Decision

**INV-06 is reworded from "session audio never leaves the machine" to "session audio
never reaches anything that processes it".** All four prohibition sites are amended
together, in the same commit, with the same narrow wording.

The exception is exactly this: **an explicit `dnd-audio archive` subcommand may send
byte-exact compressed copies of a session's immutable source files to the configured
owner-controlled private cold-storage bucket, and may read them back.** Nothing else.
Not outputs, not transcripts, not reports — publication is M7b's and does not exist.
Never invoked by `process`. Never on a processing path.

**"Owner-controlled" means the bucket, not the infrastructure.** DigitalOcean operates the
storage; it is a third party, and an earlier wording of INV-06 that contrasted this bucket
with "a third-party service" was flattering and false. What the exception actually turns on
is narrower and defensible: the destination stores opaque compressed bytes, decodes none of
them, derives nothing from them, and returns them unchanged — which is why a full readback
can prove the archive at all. The prohibition INV-06 exists to enforce is third-party
*processing* of session audio, and that is untouched. Named here because a rule stated more
broadly than it is true gets quietly ignored the first time it is inconvenient. Found by
M7a's third code review.

Every other command — `inspect`, `ingest`, `activity`, `mix`, `transcribe`, `render`,
`process`, `doctor` — stays network-denied, **and that is proved behaviourally rather
than asserted.** Each is invoked as a subprocess carrying a trap on its `PYTHONPATH`
that fails on socket construction *and* on storage-client construction. The weaker test
this replaced — asking whether `boto3` appears in `sys.modules` — would pass on an
implementation that opened a raw socket, and covered only half the commands. A
subprocess has its own address space and escapes the autouse socket fixture, which is
the honest boundary INV-05 already records.

**On deletion: the boundary is the application's operation surface, not the
credential.** The `ArchiveStorage` protocol has no delete member. The concrete adapter
calls no delete operation. `AbortMultipartUpload` is the single destructive operation
permitted, and only against this project's own incomplete uploads.

That is enforced by an **operation allowlist on a recording client**, not by grepping
for a name. The first draft proposed scanning for `DeleteObject`; boto3 spells the call
`client.delete_object(...)`, so the scan would have passed with the forbidden call
present in the adapter. The allowlist rejects `delete_object` and `delete_objects` by
name and admits `abort_multipart_upload` alone among destructive operations.

Read-only commands — `list`, `status`, `verify`, `restore` — accept a **separate
read-only credential**, so the broad key need only exist in the environment that
uploads.

## Alternatives considered

**Leave INV-06 alone and put the archive outside this repository** — a shell script the
operator runs. Rejected: the byte-exactness guarantee is the entire value, and it
depends on the same hashing, the same source-set rules, and the same INV-01 verification
the pipeline already implements. A script would reimplement all of it, worse, and
nothing would test it.

**Encrypt client-side so the uploaded bytes are not audio in any meaningful sense**,
sidestepping the invariant. Rejected as scope, and it is the wrong trade for this threat
model: the failure being defended against is disk loss, and adding a key that must
itself survive the disaster makes recovery *more* fragile. It is a reasonable future
addition and an explicit non-goal here.

**Publish outputs in the same milestone**, since the upload machinery would exist.
Rejected — that is M7b, it needs a delivery surface Cold Storage deliberately lacks, and
retention and privacy policy need a real processed session to reason about.

**Claim the upload credential has no delete capability.** Rejected as false. The
provider bundles it. Writing down a guarantee the platform does not give would be worse
than writing down the real one, because the next person would rely on it.

**A dedicated `archive` dependency group**, so a processing environment has no S3 client
installed at all. Genuinely stronger, and rejected only on cost: it adds a second
group-absent case to maintain beside `asr-qwen`, and the subprocess boundary test proves
the property directly rather than by absence. Revisit if the boundary test ever proves
hard to keep honest.

## Consequences

The project can now lose its disk without losing a session, which is the point.

What this makes harder: every future milestone that adds a command has to keep the
boundary test's command list current, and a command added without being listed is a
command nobody proved network-denied. That list is deliberately explicit rather than
derived from the Typer app, because deriving it would make an unregistered command pass
by construction — but it means the list is a thing to remember. `tests/test_cli.py`
asserting the full command surface is the cross-check.

The credential asymmetry is real residual risk and is not engineered away: an attacker
with the upload key can delete committed objects. What reduces it is keeping that key
only in the environment that uploads and using the read-only key everywhere else. What
would remove it is provider-side object lock or versioning, which is defense in depth
this milestone deliberately does not depend on — correctness rests on downloaded bytes.

What would make us revisit: a provider offering conditional create (which would give a
real compare-and-swap for the manifest — see the single-writer discussion in ADR-0038),
or evidence that the exception has been widened by a later milestone, which is what the
subprocess boundary test exists to make loud.
