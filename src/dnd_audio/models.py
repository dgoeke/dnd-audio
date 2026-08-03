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

**The lock format is provisional until M6b closes.** M6b adds the ASR and alignment
models and owns the multi-model semantics — several artifacts per logical model, sizes
in gigabytes rather than megabytes, and a downloader that must stream rather than
return :class:`bytes`. M3 needs a lock to record what it resolved; it deliberately does
not get to freeze the shape M6b will have to live in. This is the treatment M0 gave the
transcript schema.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Mapping
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
    "SILERO_VAD",
    "Downloader",
    "ModelDescriptor",
    "ModelError",
    "default_download",
    "fetch",
    "find_model",
    "lock_path",
    "lock_record",
    "model_path",
    "models_dir",
    "read_lock",
    "require_model",
    "write_lock",
]

#: Stable machine-readable codes. Reworded prose is fine; these are not (see
#: :mod:`dnd_audio.errors`). Three distinct conditions, because a caller wants to tell
#: "you never fetched it" from "what you fetched is not what was pinned".
MODEL_UNAVAILABLE: Final = "model_unavailable"
MODEL_HASH_MISMATCH: Final = "model_hash_mismatch"
MODEL_SIZE_MISMATCH: Final = "model_size_mismatch"

MODEL_LOCK_FILENAME: Final = "models.lock.json"

#: Bumped when the lock's *shape* changes. A lock this version does not recognize is
#: read as no lock at all rather than half-understood — see :func:`read_lock`.
LOCK_VERSION: Final = 1

#: The command that fixes an absent model. Named in every message about one, because a
#: diagnostic that does not say what to run is a diagnostic someone has to search for.
_FETCH_COMMAND: Final = "dnd-audio models fetch"

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


def read_lock(*, directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every model the lock claims was fetched, keyed by :attr:`ModelDescriptor.key`.

    An absent, unparseable, or unrecognized-version lock reads as ``{}``, and an entry
    missing any required field is dropped rather than returned half-populated — the
    same rule the inspection cache applies to a stored record. The lock is a convenience
    and an audit trail; :func:`find_model` is the authority, and it consults the bytes.
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


def write_lock(records: Mapping[str, Mapping[str, Any]], *, directory: Path | None = None) -> Path:
    """Write the lock atomically and canonically (INV-02). Returns its path."""
    path = lock_path(directory=directory)
    write_json_atomic(
        path,
        {"lock_version": LOCK_VERSION, "models": {key: dict(r) for key, r in records.items()}},
    )
    return path


def _record_in_lock(descriptor: ModelDescriptor, *, directory: Path) -> None:
    """Merge one model into the lock, leaving every other entry as it was.

    Merging rather than replacing is what lets M6b's fetch of the ASR models coexist
    with this one: a lock rewritten from a single descriptor would silently forget
    everything the previous command recorded.
    """
    records = read_lock(directory=directory)
    records[descriptor.key] = lock_record(descriptor)
    write_lock(records, directory=directory)
