"""The local model store, and the only code in this project that opens a socket.

INV-06 permits exactly one command to touch the network: `models fetch`. Everything
else — inspection, ingest, activity, ASR — reads models from a directory that already
exists, or refuses to run. That is why fetching is a separate verb rather than a
lazy download on first use: a stage that could fetch is a stage that could send, and
"it only downloads" is not a property anyone can check by reading a stack trace.

**What a pin is here.** A model is identified by its bytes, not by a name that resolves
to bytes. :class:`ModelDescriptor` carries the upstream repository, the release tag, the
**commit**, the path inside the repository, the expected size, and the expected SHA-256.
The URL is derived from the commit rather than the tag, because a tag is a mutable
pointer and a commit is not: `v6.2.1` can be moved and the file it names replaced,
and nothing about the download would look different (ADR-0013).

**What "present" means.** A file at the right path is not a model. :func:`find_model`
answers only for a file that exists, has the pinned size, *and* hashes to the pinned
digest. Anything else — truncated, half-written, substituted — is treated exactly as
absence, because the alternative is loading it and finding out from the transcript.

**Where it lives.** Outside any session (`$XDG_CACHE_HOME/dnd-audio/models`, overridable),
because the model is shared across sessions and is not session data. Nothing under a
session's `raw/` is involved, and no model byte is ever committed.

**Two kinds of model live here, and the difference is real** (M6b, ADR-0027). A
:class:`ModelDescriptor` is one file fetched over HTTPS by this module — Silero, 2.3 MB,
verified in memory before anything is written. A :class:`SnapshotDescriptor` is a
Hugging Face repository at one commit: eight or ten files, gigabytes, downloaded by the
``hf`` CLI that ``models fetch`` shells out to. The pin is the same idea in both cases —
a commit and a set of digests, never a name that resolves to bytes — but the second
cannot be held in memory and is not fetched by the code in this file.

**A snapshot is verified in both directions.** Every pinned file present at its pinned
size and digest, *and no unpinned file anywhere in the tree*. The second half is not
fussiness: Transformers loads a directory, not a manifest, so a stray ``config.json`` or
custom-code module left behind by a download tool is something a model would read. The
first half alone would call that tree valid.

**A snapshot directory is keyed by commit**, not by model name:
``<models_dir>/<key>/<revision>``. ``asr.model_revision`` may name a different commit,
and two commits sharing a directory would mean the second silently ran on the first's
weights.

**The lock format stopped being provisional here.** M3 wrote it for one small artifact
and said outright that M6b owns the multi-model semantics. It now carries two sections —
``models`` for single files, ``snapshots`` for repositories at a commit — hence
:data:`LOCK_VERSION` 2. A version-1 lock reads as no lock at all, which costs one
re-verification and no download.

**For a snapshot the lock is authoritative; for a single model it is not.** That
asymmetry is deliberate and is the one thing in this module most likely to look like an
inconsistency. :func:`find_model` consults the bytes against a descriptor checked into
this file, so the lock is only an audit trail. A snapshot at a *configured* revision has
no checked-in descriptor to consult — nothing else in the system knows what that commit
should contain — so its lock entry is the manifest. See ADR-0027.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dnd_audio import __version__
from dnd_audio.determinism import sha256_bytes, sha256_file, write_atomic, write_json_atomic
from dnd_audio.errors import DndAudioError

__all__ = [
    "LOCK_VERSION",
    "MODEL_HASH_MISMATCH",
    "MODEL_LOCK_FILENAME",
    "MODEL_SIZE_MISMATCH",
    "MODEL_UNAVAILABLE",
    "QWEN3_ALIGNER",
    "QWEN3_ASR",
    "QWEN_SNAPSHOTS",
    "REVISION_PATTERN",
    "SILERO_VAD",
    "SNAPSHOT_FETCH_COMMAND",
    "Downloader",
    "ModelDescriptor",
    "ModelError",
    "SnapshotDescriptor",
    "SnapshotDownloader",
    "SnapshotFile",
    "default_download",
    "fetch",
    "find_model",
    "find_snapshot",
    "hf_download",
    "install_snapshot",
    "lock_path",
    "lock_record",
    "measure_snapshot",
    "model_path",
    "models_dir",
    "read_lock",
    "read_snapshots",
    "record_snapshot_in_lock",
    "require_model",
    "require_snapshot",
    "snapshot_dir",
    "snapshot_lock_record",
    "snapshot_manifest",
    "snapshot_present",
    "verify_snapshot",
    "verify_tree",
    "write_lock",
]

#: Stable machine-readable codes. Reworded prose is fine; these are not (see
#: :mod:`dnd_audio.errors`). Three distinct conditions, because a caller wants to tell
#: "you never fetched it" from "what you fetched is not what was pinned".
MODEL_UNAVAILABLE: Final = "model_unavailable"
MODEL_HASH_MISMATCH: Final = "model_hash_mismatch"
MODEL_SIZE_MISMATCH: Final = "model_size_mismatch"

#: A snapshot's tree holds a file the manifest does not pin. Distinct from a hash
#: mismatch because the fix is different: this one is a stale or hand-edited directory,
#: and deleting it is the repair.
MODEL_UNPINNED_FILE: Final = "model_unpinned_file"

#: A revision was configured that nothing has installed. Distinct from
#: :data:`MODEL_UNAVAILABLE` because the model store may be perfectly healthy at another
#: commit, and "run the fetch command" is only actionable once you know which revision.
MODEL_REVISION_NOT_INSTALLED: Final = "model_revision_not_installed"

MODEL_LOCK_FILENAME: Final = "models.lock.json"

#: Bumped when the lock's *shape* changes. A lock this version does not recognize is
#: read as no lock at all rather than half-understood — see :func:`read_lock`.
#:
#: 2 (M6b): a second top-level ``snapshots`` section beside ``models``. A version-1 lock
#: therefore reads as empty, which costs one re-verification of a file already on disk
#: and no download — the same repair path :func:`fetch` already takes for a deleted lock.
LOCK_VERSION: Final = 2

#: The command that fixes an absent model. Named in every message about one, because a
#: diagnostic that does not say what to run is a diagnostic someone has to search for.
_FETCH_COMMAND: Final = "dnd-audio models fetch"

#: The command that fixes an absent *snapshot*. Separate from :data:`_FETCH_COMMAND`
#: because it needs an environment the everyday one does not: `hf` ships with the
#: `huggingface_hub` that lives in the ROCm environment, so this runs inside the FHS
#: shell and `scripts/fetch-models.sh` is the wrapper that does it (ADR-0027).
SNAPSHOT_FETCH_COMMAND: Final = "./scripts/fetch-models.sh"

#: A Hugging Face commit, and the only revision syntax this project accepts. A branch or
#: tag is refused at configuration load, which is what makes "`process` uses the lock
#: rather than re-resolving a moving branch" true by construction: there is no mutable
#: name left in the system for anything to re-resolve (ADR-0027).
REVISION_PATTERN: Final = r"^[0-9a-f]{40}$"

_ENV_MODELS_DIR: Final = "DND_AUDIO_MODELS_DIR"
_ENV_XDG_CACHE_HOME: Final = "XDG_CACHE_HOME"

#: Raw-content host. The `<repo>/<commit>/<path>` form serves the bytes at one immutable
#: revision; the `.../releases/download/<tag>/...` form does not, because a release's
#: assets can be replaced under a tag that never moves.
_RAW_CONTENT_BASE: Final = "https://raw.githubusercontent.com"

_DOWNLOAD_TIMEOUT_S: Final = 120.0

#: Refuse to buffer more than this from one URL. The pinned VAD model is 2.3 MB, and
#: :data:`Downloader` returns the whole body in memory, so an unbounded read would let a
#: wrong or hostile URL decide this process's memory use (INV-07). M6b's models are
#: gigabytes and will need a streaming seam instead — one more reason the lock and this
#: interface are provisional.
_DOWNLOAD_LIMIT_BYTES: Final = 64 * 1024 * 1024


class ModelError(DndAudioError):
    """A pinned model is absent, or what is on disk is not the artifact that was pinned.

    Fatal in every case (INV-13). There is no fallback to an unverified file: the whole
    point of pinning by content hash is that a model whose bytes we cannot vouch for
    produces answers we cannot vouch for, and the failure would surface as a slightly
    wrong transcript rather than as an error.
    """

    default_code = MODEL_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """One immutable model artifact: where it comes from, and what it must be.

    Args:
        key: Stable identifier used in the lock and in cache identities (INV-08).
        filename: Name the artifact takes inside the models directory. Not derived from
            the URL: a rename upstream must not silently relocate the local file.
        repository: Upstream `owner/name`.
        release: The tag the commit belongs to. Recorded for humans and for the
            detector identity; never used to build the URL.
        commit: The immutable revision the bytes come from.
        path_in_repository: Path within the repository at that commit.
        url: Where to fetch it. Built from the commit — see :func:`_raw_url`.
        size_bytes: Expected size. Cheap enough to check that it runs before hashing.
        sha256: Expected digest, lowercase hex. The identity that actually matters.
    """

    key: str
    filename: str
    repository: str
    release: str
    commit: str
    path_in_repository: str
    url: str
    size_bytes: int
    sha256: str


def _raw_url(repository: str, commit: str, path_in_repository: str) -> str:
    """The raw-content URL for one path at one commit.

    A function rather than a literal so the URL cannot drift from the commit beside it:
    editing the pin to a new revision without editing the URL is the exact mistake this
    removes.
    """
    return f"{_RAW_CONTENT_BASE}/{repository}/{commit}/{path_in_repository}"


_SILERO_REPOSITORY: Final = "snakers4/silero-vad"
_SILERO_COMMIT: Final = "7e30209a3e901f9842f81b225f3e93d8199902b1"
_SILERO_PATH_IN_REPOSITORY: Final = "src/silero_vad/data/silero_vad.onnx"

#: The voice-activity model, pinned by commit and content hash (ADR-0013).
#:
#: The size and digest below were verified twice, against two independent sources of the
#: same file: the repository at commit ``7e30209a`` and the published ``silero-vad``
#: 6.2.1 wheel's copy at ``silero_vad/data/silero_vad.onnx``. They are byte-identical.
#: That is the evidence the pin rests on — it is what makes "we did not install the
#: package" a packaging decision rather than a change of artifact.
SILERO_VAD: Final[ModelDescriptor] = ModelDescriptor(
    key="silero-vad",
    filename="silero_vad.onnx",
    repository=_SILERO_REPOSITORY,
    release="v6.2.1",
    commit=_SILERO_COMMIT,
    path_in_repository=_SILERO_PATH_IN_REPOSITORY,
    url=_raw_url(_SILERO_REPOSITORY, _SILERO_COMMIT, _SILERO_PATH_IN_REPOSITORY),
    size_bytes=2327524,
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)


def models_dir() -> Path:
    """Where models live on this machine.

    ``$DND_AUDIO_MODELS_DIR`` wins, then ``$XDG_CACHE_HOME/dnd-audio/models``, then
    ``~/.cache/dnd-audio/models``. An empty variable counts as unset — an exported
    ``DND_AUDIO_MODELS_DIR=`` otherwise resolves to the process's working directory,
    which is how a model ends up downloaded into somebody's session folder.
    """
    override = os.environ.get(_ENV_MODELS_DIR)
    if override:
        return Path(override).expanduser()

    cache_home = os.environ.get(_ENV_XDG_CACHE_HOME)
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "dnd-audio" / "models"


def _resolve_directory(directory: Path | None) -> Path:
    return models_dir() if directory is None else directory


def model_path(descriptor: ModelDescriptor, *, directory: Path | None = None) -> Path:
    """Where this model's file would live. Says nothing about whether it is there."""
    return _resolve_directory(directory) / descriptor.filename


def lock_path(*, directory: Path | None = None) -> Path:
    """Where the lock lives: beside the models it describes, never in the repository."""
    return _resolve_directory(directory) / MODEL_LOCK_FILENAME


def find_model(descriptor: ModelDescriptor, *, directory: Path | None = None) -> Path | None:
    """The model's path if it is present **and complete**, else ``None``.

    Complete means the file exists, is exactly ``size_bytes`` long, and hashes to
    ``sha256``. A wrong-sized or wrong-hashed file is not a usable model and is not
    reported as one: an interrupted download leaves a plausible-looking file at the
    right path, and treating that as "present" is how a truncated ONNX graph reaches
    a session.

    The size is checked first and short-circuits. It is not a substitute for the
    digest — it is the cheap way to avoid reading megabytes to reject a file that a
    single ``stat`` already disproved.
    """
    path = model_path(descriptor, directory=directory)
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size != descriptor.size_bytes:
        return None
    if sha256_file(path) != descriptor.sha256:
        return None
    return path


def require_model(descriptor: ModelDescriptor, *, directory: Path | None = None) -> Path:
    """:func:`find_model`, but a missing or unverifiable model is fatal.

    What every consumer of a model calls — an adapter must never decide for itself that
    running without weights is acceptable. The message names :data:`_FETCH_COMMAND`
    because "model not found" with no path forward is an hour of someone's evening.
    """
    path = find_model(descriptor, directory=directory)
    if path is None:
        expected = model_path(descriptor, directory=directory)
        message = (
            f"{descriptor.key} ({descriptor.release}) is not available at {expected}: "
            f"the file is missing, or its contents do not match the pinned sha256. "
            f"Run `{_FETCH_COMMAND}`."
        )
        raise ModelError(message)
    return path


#: ``(url) -> bytes``. Injectable so that every test but one runs offline; the seam is
#: what makes "the default suite does not reach the network" a property of the code
#: rather than a promise about it (INV-05).
Downloader = Callable[[str], bytes]


def default_download(url: str) -> bytes:
    """Fetch ``url`` over HTTPS. The only outbound request this project ever makes.

    Referenced by name inside :func:`fetch` rather than bound as a default argument, so
    a test can replace this one module attribute and be sure nothing else reaches out.
    """
    if not url.startswith("https://"):
        message = f"refusing to fetch a model over a non-HTTPS URL: {url}"
        raise ModelError(message, code=MODEL_UNAVAILABLE)

    # The scheme is checked above rather than trusted, so this is never a `file:` or
    # `data:` read wearing a URL's clothes.
    request = urllib.request.Request(url, headers={"User-Agent": f"dnd-audio/{__version__}"})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        payload: bytes = response.read(_DOWNLOAD_LIMIT_BYTES + 1)

    if len(payload) > _DOWNLOAD_LIMIT_BYTES:
        message = f"{url} returned more than {_DOWNLOAD_LIMIT_BYTES} bytes; refusing to buffer it"
        raise ModelError(message, code=MODEL_SIZE_MISMATCH)
    return payload


def fetch(
    descriptor: ModelDescriptor,
    *,
    download: Downloader | None = None,
    directory: Path | None = None,
) -> Path:
    """Ensure the pinned model is present locally, downloading it only if it is not.

    The order is the whole design. The payload is verified **in memory**, against both
    the pinned size and the pinned digest, *before* anything is written where a later
    run would find it. A file that fails verification is never created, so there is
    nothing to clean up and nothing that could be mistaken for a model afterwards —
    a half-verified file at the right path is worse than no file, because absence is
    diagnosable and corruption is not. The write itself is
    :func:`~dnd_audio.determinism.write_atomic` — temp file in the models directory,
    then rename — so an interrupted write cannot leave a partial model either.

    An already-present, verifying model is not downloaded again; the lock is still
    rewritten, which repairs a deleted lock and is byte-stable when it does not (INV-02).

    Args:
        descriptor: The pin.
        download: Injected downloader. Defaults to :func:`default_download`, resolved at
            call time.
        directory: Models directory. Defaults to :func:`models_dir`.

    Returns:
        The path to the verified model.

    Raises:
        ModelError: if the fetched bytes do not match the pinned size or digest.
    """
    target_directory = _resolve_directory(directory)
    present = find_model(descriptor, directory=target_directory)
    if present is not None:
        _record_in_lock(descriptor, directory=target_directory)
        return present

    fetch_bytes = default_download if download is None else download
    payload = fetch_bytes(descriptor.url)
    _verify(payload, descriptor)

    target = model_path(descriptor, directory=target_directory)
    write_atomic(target, payload)
    _record_in_lock(descriptor, directory=target_directory)
    return target


def _verify(payload: bytes, descriptor: ModelDescriptor) -> None:
    """Reject anything that is not exactly the pinned artifact.

    Size first, and separately from the digest: the two failures mean different things
    to whoever has to fix them. A short body is usually a proxy, a captive portal, or a
    404 page; a right-sized body with the wrong digest is a substituted artifact.
    """
    if len(payload) != descriptor.size_bytes:
        message = (
            f"{descriptor.key}: {descriptor.url} returned {len(payload)} bytes, "
            f"expected {descriptor.size_bytes}. Nothing was written."
        )
        raise ModelError(message, code=MODEL_SIZE_MISMATCH)

    digest = sha256_bytes(payload)
    if digest != descriptor.sha256:
        message = (
            f"{descriptor.key}: {descriptor.url} returned sha256 {digest}, "
            f"expected {descriptor.sha256}. Nothing was written."
        )
        raise ModelError(message, code=MODEL_HASH_MISMATCH)


def lock_record(descriptor: ModelDescriptor) -> dict[str, Any]:
    """The lock entry for one fetched model.

    Deliberately a superset of what is strictly needed to find the file again: release
    and commit are what a human reads when asking "which Silero is this", and the URL is
    what makes the record reproducible by hand.
    """
    return {
        "commit": descriptor.commit,
        "filename": descriptor.filename,
        "key": descriptor.key,
        "path_in_repository": descriptor.path_in_repository,
        "release": descriptor.release,
        "repository": descriptor.repository,
        "sha256": descriptor.sha256,
        "size_bytes": descriptor.size_bytes,
        "url": descriptor.url,
    }


_REQUIRED_LOCK_FIELDS: Final = (
    "commit",
    "filename",
    "key",
    "release",
    "repository",
    "sha256",
    "size_bytes",
    "url",
)


def _lock_document(*, directory: Path | None = None) -> dict[str, Any]:
    """The lock as a mapping, or ``{}`` for anything that is not a lock of this version.

    Shared by :func:`read_lock` and :func:`read_snapshots` so the two sections cannot
    disagree about what counts as a readable lock — an absent file, invalid JSON, a
    non-object, and an unrecognized ``lock_version`` are all "no lock", and a lock that
    was half-readable would be worse than none.
    """
    try:
        raw = lock_path(directory=directory).read_bytes()
    except OSError:
        return {}
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(document, dict) or document.get("lock_version") != LOCK_VERSION:
        return {}
    return document


def read_lock(*, directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every model the lock claims was fetched, keyed by :attr:`ModelDescriptor.key`.

    An absent, unparseable, or unrecognized-version lock reads as ``{}``, and an entry
    missing any required field is dropped rather than returned half-populated — the
    same rule the inspection cache applies to a stored record. The lock is a convenience
    and an audit trail; :func:`find_model` is the authority, and it consults the bytes.

    That last sentence is **not** true of :func:`read_snapshots`, and the module docstring
    says why: a snapshot at a configured revision has no checked-in descriptor to consult.
    """
    document = _lock_document(directory=directory)
    models = document.get("models")
    if not isinstance(models, dict):
        return {}

    records: dict[str, dict[str, Any]] = {}
    for key, record in models.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            continue
        if any(field not in record for field in _REQUIRED_LOCK_FIELDS):
            continue
        records[key] = record
    return records


def write_lock(
    records: Mapping[str, Mapping[str, Any]],
    *,
    snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    directory: Path | None = None,
) -> Path:
    """Write the lock atomically and canonically (INV-02). Returns its path.

    Writes exactly what it is given: omitting ``snapshots`` writes no snapshot section,
    it does not preserve one. Preserving is :func:`_record_in_lock`'s job and
    :func:`record_snapshot_in_lock`'s, which is where "merge" belongs — a low-level
    writer that quietly read the file it was about to overwrite would be the surprising
    one.
    """
    path = lock_path(directory=directory)
    document: dict[str, Any] = {
        "lock_version": LOCK_VERSION,
        "models": {key: dict(r) for key, r in records.items()},
    }
    if snapshots is not None:
        document["snapshots"] = {key: dict(r) for key, r in snapshots.items()}
    write_json_atomic(path, document)
    return path


def _record_in_lock(descriptor: ModelDescriptor, *, directory: Path) -> None:
    """Merge one model into the lock, leaving every other entry as it was.

    Merging rather than replacing is what lets M6b's fetch of the ASR models coexist
    with this one: a lock rewritten from a single descriptor would silently forget
    everything the previous command recorded.
    """
    records = read_lock(directory=directory)
    records[descriptor.key] = lock_record(descriptor)
    write_lock(records, snapshots=read_snapshots(directory=directory), directory=directory)


# --- snapshots: a Hugging Face repository at one commit (M6b, ADR-0027) -------------


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """One file inside a snapshot, and what it must be.

    Args:
        path: Relative to the snapshot directory, POSIX-separated. Never absolute and
            never containing ``..`` — :func:`verify_snapshot` refuses both, because a
            manifest is data and this one decides which paths get read.
        size_bytes: Expected size. Checked first because a ``stat`` can reject a file
            without reading gigabytes of it.
        sha256: Expected digest, lowercase hex. The identity that actually matters.
    """

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SnapshotDescriptor:
    """One Hugging Face repository, pinned at one commit, with its whole file manifest.

    ``files`` is the *complete* set of what may be in the tree, not a subset that must be
    present. Files a model does not load — ``README.md``, ``.gitattributes`` — are
    deliberately absent from it and are not downloaded, so their presence in a snapshot
    directory is a verification failure like any other unpinned file.

    Args:
        key: Stable identifier, used in the lock, in the directory layout, and in cache
            identities (INV-08).
        repository: Upstream ``owner/name``, as ``hf download`` takes it.
        revision: The default commit. A configured revision overrides it, and then the
            lock rather than this manifest is what verification consults (ADR-0027).
        files: Every file, sorted by path so the manifest is canonical.
    """

    key: str
    repository: str
    revision: str
    files: tuple[SnapshotFile, ...]


#: `Qwen/Qwen3-ASR-1.7B` at the commit that was current on 2026-08-03.
#:
#: Every digest below was resolved from Hugging Face's own metadata before this
#: descriptor was written, which is the order M3 used for Silero: establish the pin, then
#: write the code that verifies it. For the two `safetensors` shards the digest is the
#: LFS object id, which *is* the sha256 of the file's contents; for the small JSON and
#: vocabulary files, which are stored as ordinary git blobs and therefore carry a sha1
#: git oid rather than a content sha256, each was downloaded and hashed.
QWEN3_ASR: Final[SnapshotDescriptor] = SnapshotDescriptor(
    key="qwen3-asr",
    repository="Qwen/Qwen3-ASR-1.7B",
    revision="7278e1e70fe206f11671096ffdd38061171dd6e5",
    files=(
        SnapshotFile(
            "chat_template.json",
            1161,
            "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
        ),
        SnapshotFile(
            "config.json",
            6194,
            "2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f",
        ),
        SnapshotFile(
            "generation_config.json",
            142,
            "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c",
        ),
        SnapshotFile(
            "merges.txt",
            1671853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        SnapshotFile(
            "model-00001-of-00002.safetensors",
            4220320824,
            "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
        ),
        SnapshotFile(
            "model-00002-of-00002.safetensors",
            478200688,
            "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
        ),
        SnapshotFile(
            "model.safetensors.index.json",
            64821,
            "f994739fe38e5210b9e3e8ce6c6307315e2ceac3cb630e7b7414d69dce520f60",
        ),
        SnapshotFile(
            "preprocessor_config.json",
            330,
            "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d",
        ),
        SnapshotFile(
            "tokenizer_config.json",
            12487,
            "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
        ),
        SnapshotFile(
            "vocab.json",
            2776833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
)

#: `Qwen/Qwen3-ForcedAligner-0.6B`, pinned the same way and on the same date.
#:
#: Four of its files are byte-identical to the ASR model's — `chat_template.json`,
#: `merges.txt`, `preprocessor_config.json`, `vocab.json` carry the same digests. That is
#: a fact about the two repositories, not a copy-paste error, and it is worth noticing
#: before anyone "fixes" it.
QWEN3_ALIGNER: Final[SnapshotDescriptor] = SnapshotDescriptor(
    key="qwen3-forced-aligner",
    repository="Qwen/Qwen3-ForcedAligner-0.6B",
    revision="c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    files=(
        SnapshotFile(
            "chat_template.json",
            1161,
            "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
        ),
        SnapshotFile(
            "config.json",
            5982,
            "d616c65d46c4b90bdc651b0a0963ea932732241140f337f9bb6b0335a9c8ef09",
        ),
        SnapshotFile(
            "generation_config.json",
            115,
            "948d089b23bca1d214e768d59c4438365665f52ec6d33678f4062206b3fbbb8c",
        ),
        SnapshotFile(
            "merges.txt",
            1671853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        SnapshotFile(
            "model.safetensors",
            1835544544,
            "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7",
        ),
        SnapshotFile(
            "preprocessor_config.json",
            330,
            "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d",
        ),
        SnapshotFile(
            "tokenizer_config.json",
            12666,
            "3ab80063f8511deb9566e6ad438d17b7a6277fcffd52d92854112f19d36bd81c",
        ),
        SnapshotFile(
            "vocab.json",
            2776833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
)

#: Both, in the order `models fetch` installs them and `doctor` reports them.
QWEN_SNAPSHOTS: Final[tuple[SnapshotDescriptor, ...]] = (QWEN3_ASR, QWEN3_ALIGNER)


def snapshot_dir(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Where this repository at this commit lives: ``<models_dir>/<key>/<revision>``.

    Keyed by commit, so two revisions of one model cannot share a directory. Without that
    a configured revision would be verified against one tree and loaded from another's
    leftovers, and the run would report the revision it asked for.
    """
    resolved = descriptor.revision if revision is None else revision
    return _resolve_directory(directory) / descriptor.key / resolved


def snapshot_manifest(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
) -> tuple[SnapshotFile, ...] | None:
    """What the tree at ``revision`` must contain, or ``None`` if nothing knows.

    The default revision's manifest is checked into this file, where it was reviewed. Any
    other revision's is whatever the lock recorded when `models fetch` installed it —
    there is no third source, and inventing one would mean trusting a directory to
    describe itself (ADR-0027).
    """
    resolved = descriptor.revision if revision is None else revision
    if resolved == descriptor.revision:
        return descriptor.files

    record = read_snapshots(directory=directory).get(descriptor.key)
    if record is None or record.get("revision") != resolved:
        return None
    return _manifest_from_record(record)


def _manifest_from_record(record: Mapping[str, Any]) -> tuple[SnapshotFile, ...] | None:
    """Parse a lock entry's file list, refusing anything half-formed.

    A lock is a file on disk that a person can edit, so every field is checked rather than
    trusted. An unparseable entry reads as no entry, which sends the caller to the fetch
    command instead of to a confusing exception.
    """
    rows = record.get("files")
    if not isinstance(rows, list) or not rows:
        return None
    try:
        files = tuple(
            SnapshotFile(
                path=str(row["path"]),
                size_bytes=int(row["size_bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return None
    return files


def verify_snapshot(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
) -> str | None:
    """``None`` when the snapshot is exactly what was pinned, else why it is not.

    A reason string rather than an exception, so :func:`find_snapshot` and
    :func:`require_snapshot` are one verification with two presentations. Two
    verification paths that agreed today would be two that disagreed later.

    Checked, in order:

    1. Something knows what this revision should contain.
    2. Every manifest path is relative and free of ``..``. The manifest decides which
       files get opened, so it is validated before it is used, not after.
    3. Every pinned file exists, is exactly its pinned size, and hashes to its pinned
       digest. Size first: a ``stat`` rejects a truncated 4 GB shard without reading it.
    4. **No unpinned file is anywhere in the tree.** Transformers loads a directory, not
       a manifest.
    """
    resolved = descriptor.revision if revision is None else revision
    manifest = snapshot_manifest(descriptor, revision=resolved, directory=directory)
    if manifest is None:
        return (
            f"no manifest for {descriptor.repository} at {resolved}: it is not the "
            f"revision pinned in this build and the model lock has no record of it"
        )

    root = snapshot_dir(descriptor, revision=resolved, directory=directory)
    return verify_tree(root, manifest, key=descriptor.key)


def verify_tree(root: Path, manifest: Sequence[SnapshotFile], *, key: str) -> str | None:
    """``None`` when ``root`` is exactly ``manifest``, else why it is not.

    Split out from :func:`verify_snapshot` because :func:`install_snapshot` has to check a
    freshly-moved tree against a manifest it measured, *before* that manifest is written to
    the lock. Going through :func:`verify_snapshot` there would mean recording a lock entry
    in order to check whether it deserved to be recorded.
    """
    if not root.is_dir():
        return f"{root} does not exist"

    expected: set[Path] = set()
    for entry in manifest:
        candidate = Path(entry.path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return (
                f"the manifest for {key} names {entry.path!r}, which is not a "
                f"relative path inside the snapshot"
            )
        path = root / candidate
        expected.add(path)
        try:
            size = path.stat().st_size
        except OSError:
            return f"{entry.path} is missing from {root}"
        if size != entry.size_bytes:
            return (
                f"{entry.path} is {size} bytes, expected {entry.size_bytes} — an "
                f"interrupted download leaves a plausible-looking file at the right path"
            )
        if sha256_file(path) != entry.sha256:
            return f"{entry.path} does not hash to the pinned {entry.sha256}"

    for found in sorted(root.rglob("*")):
        if found.is_dir() or found in expected:
            continue
        return (
            f"{found.relative_to(root)} is in {root} but is not pinned. Transformers "
            f"loads a directory rather than a manifest, so an unpinned file is one a "
            f"model may read; delete the directory and re-run `{SNAPSHOT_FETCH_COMMAND}`"
        )
    return None


def find_snapshot(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
) -> Path | None:
    """The snapshot's directory if it is present **and exactly what was pinned**, else
    ``None``.

    :func:`find_model`'s rule, for :func:`find_model`'s reason: anything that is not the
    pinned artifact is treated as absence, because a half-downloaded 4 GB shard at the
    right path is worse than no file at all — absence is diagnosable and corruption
    surfaces as a slightly wrong transcript.
    """
    if verify_snapshot(descriptor, revision=revision, directory=directory) is not None:
        return None
    return snapshot_dir(descriptor, revision=revision, directory=directory)


def snapshot_present(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
) -> bool:
    """Do the pinned files exist at the pinned sizes? **Digests are not checked.**

    For `doctor`, which answers "is this machine ready" and must not spend a minute
    hashing six gigabytes to do it. Deliberately a separate function rather than a
    ``verify=False`` argument on :func:`find_snapshot`: a boolean that switches off a
    check is a boolean somebody passes by accident, and every caller that is about to
    *load* a model must go through the verifying one.
    """
    resolved = descriptor.revision if revision is None else revision
    manifest = snapshot_manifest(descriptor, revision=resolved, directory=directory)
    if manifest is None:
        return False
    root = snapshot_dir(descriptor, revision=resolved, directory=directory)
    for entry in manifest:
        candidate = Path(entry.path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return False
        try:
            if (root / candidate).stat().st_size != entry.size_bytes:
                return False
        except OSError:
            return False
    return True


def require_snapshot(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
) -> Path:
    """:func:`find_snapshot`, but an absent or unverifiable snapshot is fatal.

    What every consumer of a snapshot calls. An adapter must never decide for itself that
    running against unverified weights is acceptable, and the message names both the
    reason and the command that fixes it — "model not found" with no path forward is an
    hour of somebody's evening.
    """
    reason = verify_snapshot(descriptor, revision=revision, directory=directory)
    if reason is None:
        return snapshot_dir(descriptor, revision=revision, directory=directory)

    resolved = descriptor.revision if revision is None else revision
    code = MODEL_REVISION_NOT_INSTALLED if "no manifest for" in reason else MODEL_UNAVAILABLE
    if "is not pinned" in reason:
        code = MODEL_UNPINNED_FILE
    elif "does not hash" in reason:
        code = MODEL_HASH_MISMATCH
    elif "expected" in reason and "bytes," in reason:
        code = MODEL_SIZE_MISMATCH
    message = (
        f"{descriptor.key} ({descriptor.repository} at {resolved}) is not usable: "
        f"{reason}. Run `{SNAPSHOT_FETCH_COMMAND}`."
    )
    raise ModelError(message, code=code)


def snapshot_lock_record(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    files: Sequence[SnapshotFile] | None = None,
) -> dict[str, Any]:
    """The lock entry for one installed snapshot.

    For the default revision the manifest recorded is the checked-in one, so the lock and
    this file agree by construction. For a configured revision ``files`` is what was
    measured on disk after installation, and the record becomes the only manifest that
    revision has (ADR-0027).
    """
    resolved = descriptor.revision if revision is None else revision
    manifest = descriptor.files if files is None else tuple(files)
    return {
        "files": [
            {"path": entry.path, "sha256": entry.sha256, "size_bytes": entry.size_bytes}
            for entry in sorted(manifest, key=lambda entry: entry.path)
        ],
        "key": descriptor.key,
        "repository": descriptor.repository,
        "revision": resolved,
    }


_REQUIRED_SNAPSHOT_FIELDS: Final = ("files", "key", "repository", "revision")


def read_snapshots(*, directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every snapshot the lock claims was installed, keyed by
    :attr:`SnapshotDescriptor.key`.

    Same rules as :func:`read_lock`: an absent, unparseable, or unrecognized-version lock
    reads as ``{}``, and an entry missing a required field is dropped rather than returned
    half-populated.
    """
    document = _lock_document(directory=directory)
    snapshots = document.get("snapshots")
    if not isinstance(snapshots, dict):
        return {}

    records: dict[str, dict[str, Any]] = {}
    for key, record in snapshots.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            continue
        if any(field not in record for field in _REQUIRED_SNAPSHOT_FIELDS):
            continue
        records[key] = record
    return records


def record_snapshot_in_lock(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    files: Sequence[SnapshotFile] | None = None,
    directory: Path | None = None,
) -> Path:
    """Merge one snapshot into the lock, leaving every other entry as it was.

    Merging for the same reason :func:`_record_in_lock` does: the ASR model, the aligner
    and Silero all live in one lock, and a rewrite from a single descriptor would silently
    forget the other two.
    """
    snapshots = read_snapshots(directory=directory)
    snapshots[descriptor.key] = snapshot_lock_record(descriptor, revision=revision, files=files)
    return write_lock(read_lock(directory=directory), snapshots=snapshots, directory=directory)


#: What ``hf download`` fetches that no model loads, and that therefore must not end up in
#: a verified snapshot. Repository furniture, not artifacts: excluding them is why the
#: manifests in this file have ten and eight entries rather than twelve and ten.
_SNAPSHOT_EXCLUDED_NAMES: Final = frozenset({"README.md", ".gitattributes"})

#: ``hf`` writes its own bookkeeping under this name inside ``--local-dir``. Never moved
#: into place, which is most of why the installer stages and moves rather than downloading
#: straight into the snapshot directory.
_HF_METADATA_DIRNAME: Final = ".cache"

#: How long one repository may take. Six gigabytes over a slow link is minutes, not hours;
#: an hour is the point at which something has gone wrong and a hung command is worse than
#: a failed one.
_SNAPSHOT_DOWNLOAD_TIMEOUT_S: Final = 3600.0

#: ``(repository, revision, target) -> None``. Injectable for the same reason
#: :data:`Downloader` is: it is what makes "the default suite reaches no network" a
#: property of the code rather than a promise about it (INV-05). The production
#: implementation is :func:`hf_download`.
SnapshotDownloader = Callable[[str, str, Path], None]


def hf_download(repository: str, revision: str, target: Path) -> None:
    """Run ``hf download`` into ``target``. The one subprocess that reaches the network.

    Output is deliberately **not** captured. This downloads several gigabytes once, and a
    command that prints nothing for ten minutes looks hung; `hf`'s own progress bars are
    the right thing for an operator to be watching. The cost is that a failure's detail is
    on the terminal rather than in the exception, so the message says to look there.
    """
    command = [
        "hf",
        "download",
        repository,
        "--revision",
        revision,
        "--local-dir",
        str(target),
    ]
    try:
        # A fixed argv and no shell. `repository` and `revision` come from a descriptor in
        # this file or from a configuration value already validated against
        # `REVISION_PATTERN`, and neither is ever interpreted by a shell.
        completed = subprocess.run(command, check=False, timeout=_SNAPSHOT_DOWNLOAD_TIMEOUT_S)
    except FileNotFoundError as exc:
        message = (
            f"`hf` is not on PATH, so {repository} cannot be downloaded. It ships with "
            f"`huggingface_hub`, which lives in the ROCm environment rather than the "
            f"project one — run `{SNAPSHOT_FETCH_COMMAND}`, which enters that environment "
            f"for you."
        )
        raise ModelError(message, code=MODEL_UNAVAILABLE) from exc
    except subprocess.TimeoutExpired as exc:
        message = (
            f"`hf download {repository}` did not finish within "
            f"{_SNAPSHOT_DOWNLOAD_TIMEOUT_S:.0f}s. Nothing was moved into place; re-run to "
            f"resume."
        )
        raise ModelError(message, code=MODEL_UNAVAILABLE) from exc

    if completed.returncode != 0:
        message = (
            f"`hf download {repository}` exited {completed.returncode}. Its output is "
            f"above. Nothing was moved into place."
        )
        raise ModelError(message, code=MODEL_UNAVAILABLE)


def measure_snapshot(root: Path) -> tuple[SnapshotFile, ...]:
    """Every model file under ``root``, with its measured size and digest.

    What an *overridden* revision's manifest is made of: nothing in this build knows what
    that commit should contain, so the manifest is what arrived, recorded in the lock so a
    later run can tell whether it is still what arrived (ADR-0027).
    """
    found: list[SnapshotFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == _HF_METADATA_DIRNAME or relative.name in _SNAPSHOT_EXCLUDED_NAMES:
            continue
        found.append(
            SnapshotFile(
                path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return tuple(found)


def install_snapshot(
    descriptor: SnapshotDescriptor,
    *,
    revision: str | None = None,
    directory: Path | None = None,
    download: SnapshotDownloader | None = None,
) -> tuple[Path, bool]:
    """Ensure the pinned snapshot is present and verified. Returns ``(path, downloaded)``.

    The order is the design, and it is :func:`fetch`'s order scaled up to a tree:

    1. **An already-verifying snapshot is not downloaded again.** Re-running this is cheap
       and safe, which is what makes it usable as the "am I set up?" command.
    2. The download lands in a **staging** directory beside the target, never in it. So a
       failed or interrupted download cannot leave a half-tree where a later run would
       find one, and `hf`'s own `.cache` bookkeeping never reaches a verified snapshot.
    3. **Only manifest files move into place.** For the pinned revision that is the
       manifest checked into this file, so a repository that grew a file upstream cannot
       smuggle it in; for an overridden one it is what arrived, minus repository furniture.
    4. The moved tree is verified **before** anything is written to the lock. Recording
       first would mean the lock vouching for bytes nobody had checked — which is the
       whole failure INV-08 exists to prevent, one level up from a cache entry.
    5. Only then is the lock written.

    Raises:
        ModelError: if the download fails, or if what it produced is not what was pinned.
    """
    resolved = descriptor.revision if revision is None else revision
    target = snapshot_dir(descriptor, revision=resolved, directory=directory)
    if verify_snapshot(descriptor, revision=resolved, directory=directory) is None:
        return target, False

    staging = target.parent / f".staging-{resolved}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    (hf_download if download is None else download)(descriptor.repository, resolved, staging)

    manifest = descriptor.files if resolved == descriptor.revision else measure_snapshot(staging)
    missing = [entry.path for entry in manifest if not (staging / entry.path).is_file()]
    if missing:
        message = (
            f"`hf download {descriptor.repository}` did not produce {', '.join(missing)}. "
            f"The pin in this build expects them at {resolved}; if upstream removed them, "
            f"this build's descriptor is stale and needs updating rather than working "
            f"around."
        )
        raise ModelError(message, code=MODEL_UNAVAILABLE)

    shutil.rmtree(target, ignore_errors=True)
    for entry in manifest:
        destination = target / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        (staging / entry.path).replace(destination)
    shutil.rmtree(staging, ignore_errors=True)

    reason = verify_tree(target, manifest, key=descriptor.key)
    if reason is not None:
        message = (
            f"{descriptor.key} downloaded from {descriptor.repository} at {resolved}, but "
            f"what arrived is not what was pinned: {reason}. Nothing was recorded."
        )
        raise ModelError(message, code=MODEL_HASH_MISMATCH)

    record_snapshot_in_lock(descriptor, revision=resolved, files=manifest, directory=directory)
    return target, True
