# ADR-0025 — Provisioning Torch from AMD's gfx1151 index, and the two environments

**Status:** accepted
**Date:** 2026-08-03
**Milestone:** M6a

## Context

The spec requires PyTorch to come from AMD's stable `gfx1151` wheel index rather than from
ordinary PyPI or nixpkgs, with the exact wheel and runtime versions locked, `torch`
constrained to that index through uv's per-package sourcing, and the AMD index kept
explicit rather than becoming the default source for unrelated packages. It gives a
configuration sketch:

```toml
[[tool.uv.index]]
name = "amd-gfx1151"
url = "https://repo.amd.com/rocm/whl/gfx1151/"
explicit = true

[tool.uv.sources]
torch = { index = "amd-gfx1151" }
```

and names the failure it exists to prevent: `accelerate`'s transitive `torch>=2.0.0`
resolving a CUDA build from PyPI.

**That sketch does not resolve.** Torch 2.9.1+rocm7.13.0 requires `rocm[libraries]==7.13.0`
and `triton==3.5.1+rocm7.13.0`, neither of which is on PyPI in the form it needs — `rocm`
on PyPI is an unrelated placeholder at 0.1.0. Routing them in `[tool.uv.sources]` does not
help, because **uv applies that table only to packages that are also direct members of a
dependency list.** A requirement discovered inside another package's metadata is looked up
on the default index no matter what the sources table says, and it is looked up silently:
no warning, no error, just the wrong registry recorded in the lock.

This was established twice, independently, within minutes: once by reading uv's index
semantics, and once empirically, by adding `rocm = { index = "amd-gfx1151" }`, watching
`uv lock` continue to query `pypi.org/simple/rocm/`, and reproducing the same behaviour in
an isolated project where a transitive-only `urllib3` routed to a second explicit index
still locked from PyPI.

Three further constraints were in play. `rocm[libraries]` is a source distribution that
builds at install time and wants a `/usr/lib` layout, which is what ADR-0002's FHS shell
exists for. The AMD index publishes no PEP 658 metadata sidecars and no hashes, so
resolution downloads whole multi-gigabyte wheels to read their metadata. And INV-05
requires the default test suite to pass with none of this installed.

## Decision

**Every AMD-only package torch needs is a direct member of the `asr-qwen` group and has a
`[tool.uv.sources]` entry.** Listing them is not a statement that this project imports
them; it is the only mechanism by which the routing reaches them. The group is pinned
exactly, never with floors.

**The index stays `explicit = true`.** It carries its own `numpy`, `setuptools`, `jinja2`
and `typing-extensions`, and an inexplicit index would make those candidates for every
resolution in the project. What the spec asks for — the AMD index not being the default
source for unrelated packages — is exactly this flag, and the cost of it is the direct
membership above.

**Torch is `2.9.1+rocm7.13.0`, and not because it is newest.** AMD also publishes 2.10.0
and 2.11.0 for cp312 Linux. ROCm 7.13 is the generation this host already runs a GPU
workload on, which is evidence about the *runtime* and none whatever about a torch version;
among three candidates with no host evidence, a patch release is preferred to two `x.y.0`
releases. Tested beats newest, and the reason is recorded here so that nobody later
"corrects" it to the highest number on the page.

**`transformers` and `accelerate` land in M6a, at `qwen-asr` 0.0.6's exact pins**
(`4.57.6` and `1.12.0`). They are M6b's dependencies, but the lock assertion this milestone
owes is about them: with torch alone in the lock nothing competes for it and "no CUDA build
won" is vacuously true. Pinning them to what M6b will actually consume also means M6b does
not relock and reinstall a multi-gigabyte environment that this milestone just settled.
`qwen-asr` itself stays in M6b.

**There are two environments, not one.** `.venv` is the project environment and never
receives the group; `.venv-rocm`, selected with `UV_PROJECT_ENVIRONMENT`, is where the
group is installed, from inside the FHS shell:

```
nix run .#fhs -- -c 'UV_PROJECT_ENVIRONMENT=.venv-rocm uv sync --group asr-qwen'
```

This is what makes INV-05 continuously true rather than true once. The gate runs
`uv run --no-sync` against `.venv`, so if the group were installed there the everyday
suite would silently stop being the group-absent case it claims to prove.

**The lock is asserted in both directions.** `tests/test_packaging.py` reads `uv.lock` and
requires that every package resolved from the AMD index is one of the expected ones at its
expected version, *and* that every package outside that set resolves from PyPI. The second
direction is the one that catches a silently-ignored routing entry, which is the failure
mode this ADR is mostly about.

## Alternatives considered

- **Follow the spec's sketch literally and route only `torch`.** Rejected on evidence: it
  does not resolve. The spec's example is correct about mechanism and incomplete about
  AMD's current package topology. Amending the spec was considered and rejected as
  disproportionate — the sketch is prefaced "must be equivalent to", and this is equivalent
  with the closure filled in. This ADR is the record.
- **Drop `explicit = true` so the AMD index serves the transitive requirements.** Rejected:
  it is the one thing the spec's Target-host runtime section forbids in as many words, and
  it would silently move unrelated packages onto AMD's mirror. The failure it invites is
  the same class as the one it fixes, with a wider blast radius.
- **Install the group into `.venv` and prove the group-absent case once, in verify.**
  Rejected. A property proven once decays; the whole reason the gate is mechanical is that
  nobody re-derives it every milestone. The second environment costs one variable.
- **Pin `transformers`/`accelerate` with floors and let M6b tighten them.** Rejected after
  checking `qwen-asr` 0.0.6's metadata: floors guarantee a relock and a multi-gigabyte
  redownload in the next milestone, where the cost would be least visible and most annoying.
- **Take torch 2.11.0 as the newest gfx1151 build.** Rejected. Newer torch pulls newer
  triton and widens the surface with nothing to show for it; the only host evidence
  available points at the ROCm generation, not at a torch release.
- **nixpkgs' torch, or `nix-ld` instead of FHS.** Already rejected in ADR-0002 and nothing
  here changes that: nixpkgs' torch is broken on gfx1151, and a source build at install
  time is what FHS is for.

## Consequences

- **A re-lock needs the network and real time.** No PEP 658 metadata means uv downloads
  full wheels to resolve. This is a deliberate, occasional operation, never part of the
  gate, which stays offline (ADR-0002).
- **Adding a torch-adjacent dependency may fail in a confusing way.** If a new package
  brings an AMD-only transitive requirement, resolution will fail naming a package nobody
  wrote down. The fix is always the same: add it to the group *and* the sources table. This
  ADR exists so that is a two-minute lookup rather than an afternoon.
- **Two environments is two things to keep working**, the same tension ADR-0002 records for
  two shells. The mitigation is the same: `.venv` is what everything except GPU work uses,
  and the ROCm one is entered deliberately.
- **`host_smoke` tests only pass from `.venv-rocm`.** They are excluded from the gate
  anyway (INV-05), but a run from the wrong environment reports "torch is not installed"
  rather than a GPU failure, and the assertion messages say so.
- **The pins will need deliberate bumping.** Exact versions mean a ROCm or torch upgrade is
  an explicit edit with its own gate run — which is the intent, since the alternative is a
  GPU stack moving underneath a session nobody was watching.
- **Whether this yields a working build was OQ-008**, and it is answered in
  `OPEN-QUESTIONS.md` with the versions and the smoke results rather than here.
