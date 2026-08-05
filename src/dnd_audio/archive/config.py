"""Archive settings, deliberately nowhere near `session.yaml`.

The first plan review's opening finding was that an `archive:` section in
:class:`~dnd_audio.config.SessionConfig` would change the global config hash, the
processing schema version, every stage projection, and transcript-record compatibility —
so setting a bucket name would invalidate gigabytes of cached inference. That is not a
theoretical coupling; `config_hash` is what the inspection cache is keyed on, and
`stage_config_hash` is what four more are keyed on.

So this is a separate model, loaded from the environment or an operator profile, read
**only** by archive commands, and it touches no processing identity. The regression that
proves it compares real cache keys — not output bytes, which would be identical after a
cache miss and would therefore prove nothing (INV-08).

**Credentials are `SecretStr`.** Pydantic then prints `**********` from every repr, which
matters because a validation error on one field renders the whole model, and a
`ValidationError` traceback is the single most likely place for a secret access key to
end up in a log (ADR-0039).

**Two credentials, not one.** DigitalOcean bundles multipart-abort with broad
Read/Write/Delete object permission, so the key that uploads is more capable than this
application ever is. Keeping it out of the environment where `list`, `verify` and
`restore` run reduces — and does not eliminate — what a compromised profile can do
(ADR-0035).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from dnd_audio.errors import DndAudioError

__all__ = [
    "ENV_PREFIX",
    "ENV_PROFILE",
    "ArchiveConfigError",
    "ArchiveRuntimeConfig",
    "default_report_dir",
    "load_archive_config",
    "state_dir",
]

#: Every setting is `DND_AUDIO_ARCHIVE_<FIELD>`, uppercased. One prefix, mechanically
#: derived from the field name, so adding a field cannot forget to add its variable.
ENV_PREFIX: Final = "DND_AUDIO_ARCHIVE_"

#: Points at a gitignored YAML profile holding the same keys in lowercase. Environment
#: variables win over it, so a profile is a convenience rather than a second authority.
ENV_PROFILE: Final = "DND_AUDIO_ARCHIVE_PROFILE"

_ENV_STATE_HOME: Final = "XDG_STATE_HOME"

#: DigitalOcean documents 5 GB as the single-PUT ceiling. This is the *hard* limit, not
#: the threshold: above it multipart is mandatory rather than preferred (ADR-0038).
SINGLE_PUT_HARD_LIMIT_BYTES: Final = 5_000_000_000

#: The provider's documented minimum for every part but the last.
MIN_MULTIPART_PART_BYTES: Final = 5 * 1024 * 1024

#: The provider's documented maximum. Part size is raised to `ceil(size / this)` when a
#: large object would otherwise need more parts than exist.
MAX_MULTIPART_PARTS: Final = 10_000


class ArchiveConfigError(DndAudioError):
    """Archive settings are absent, incomplete, or unusable.

    Separate from :class:`~dnd_audio.errors.ConfigError` because it is a different file,
    a different lifecycle, and a different audience: `session.yaml` describes a recording,
    and this describes one machine's access to one bucket.
    """

    default_code = "invalid_archive_configuration"


class ArchiveRuntimeConfig(BaseModel):
    """Where the archive lives and how to reach it. Never part of a session.

    Deliberately **not** frozen-and-hashed the way `SessionConfig` is: nothing here may
    ever enter a cache key, so there is no `resolved_config` counterpart and no
    `archive_config_hash`. If one is ever added, that is the moment this milestone's
    central guarantee quietly stops holding.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Regional S3 endpoint, e.g. `https://nyc3.digitaloceanspaces.com`. HTTPS is required
    #: rather than defaulted: a plaintext endpoint would send session audio in the clear,
    #: and the failure would be invisible.
    endpoint_url: str
    region: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    access_key_id: SecretStr
    secret_access_key: SecretStr

    #: Objects at or above this go multipart. Distinct from the provider's hard limit:
    #: this is tunable down so the multipart path is exercised on modest files, which is
    #: how the host smoke forces it without a 5 GiB fixture.
    multipart_threshold_bytes: int = Field(
        default=256 * 1024 * 1024, ge=MIN_MULTIPART_PART_BYTES, le=SINGLE_PUT_HARD_LIMIT_BYTES
    )
    multipart_part_bytes: int = Field(
        default=64 * 1024 * 1024, ge=MIN_MULTIPART_PART_BYTES, le=SINGLE_PUT_HARD_LIMIT_BYTES
    )

    #: Additional attempts against `503 Slow Down`. The SDK's own retries are disabled
    #: where the client is built, so this is the only bound that exists (ADR-0038).
    max_retries: int = Field(default=5, ge=0, le=32)
    #: First backoff step, doubling. Injectable in tests so a retry test costs no seconds.
    retry_base_seconds: float = Field(default=0.5, gt=0.0, le=60.0)

    @field_validator("endpoint_url")
    @classmethod
    def _require_https(cls, value: str) -> str:
        """Refuse a non-HTTPS endpoint rather than defaulting one.

        The same rule `models.default_download` applies to a model URL, for a stronger
        reason: this direction carries session audio.
        """
        if not value.startswith("https://"):
            # The offending value is deliberately not quoted. The charter lists the
            # endpoint alongside the keys among values that must never reach an
            # exception, and a validation error renders into reports and scrollback.
            message = (
                "must be an https:// URL. This connection carries session audio, so a "
                "plaintext endpoint is refused rather than used. The value is not echoed "
                "here; check the variable you set."
            )
            raise ValueError(message)
        return value.rstrip("/")


def state_dir() -> Path:
    """Where archive state that belongs to no session lives.

    Locks and resumable multipart upload ids, specifically. Outside every session
    directory because a lock inside a source root would violate INV-01, and outside
    `work/` because an upload's lock must survive someone deleting a session's caches.
    """
    override = os.environ.get(_ENV_STATE_HOME)
    base = Path(override).expanduser() if override else Path.home() / ".local" / "state"
    return base / "dnd-audio" / "archive"


def default_report_dir() -> Path:
    """Where `list`, `verify` and `restore` write their reports absent `--report`.

    Those three exist for the case where the session directory is gone, so they cannot
    default to `work/` the way `upload` and `status` do (ADR-0039).
    """
    return state_dir() / "reports"


def load_archive_config(environ: Mapping[str, str] | None = None) -> ArchiveRuntimeConfig:
    """Assemble the configuration from the environment and an optional profile.

    Environment variables win over the profile, so an operator can override one setting
    for one command without editing a file they will forget they edited.

    Raises:
        ArchiveConfigError: if a required setting is missing or a value is unusable. The
            message names the environment variables to set, because "invalid archive
            configuration" with no field list is an error nobody can act on — and because
            the pydantic report cannot be shown verbatim here the way `load_session_config`
            shows it: it renders the model, and the model holds credentials.
    """
    env = os.environ if environ is None else environ
    values: dict[str, Any] = {}

    profile_path = env.get(ENV_PROFILE)
    if profile_path:
        values.update(_read_profile(Path(profile_path).expanduser()))

    for field in ArchiveRuntimeConfig.model_fields:
        found = env.get(f"{ENV_PREFIX}{field.upper()}")
        if found is not None and found != "":
            values[field] = found

    try:
        return ArchiveRuntimeConfig.model_validate(values)
    except ValidationError as exc:
        raise ArchiveConfigError(_describe(exc)) from exc


def _read_profile(path: Path) -> dict[str, Any]:
    """Read the operator profile, which is YAML holding the same keys in lowercase."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        message = (
            f"{ENV_PROFILE} points at {path}, which cannot be read: {exc}. Unset it to "
            f"configure the archive from the environment alone."
        )
        raise ArchiveConfigError(message) from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        message = f"the archive profile at {path} is not valid YAML: {exc}"
        raise ArchiveConfigError(message) from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        message = (
            f"the archive profile at {path} must contain a YAML mapping, got {type(raw).__name__}"
        )
        raise ArchiveConfigError(message)
    return {str(key): value for key, value in raw.items()}


def _describe(exc: ValidationError) -> str:
    """Turn a pydantic report into a message that names fields but quotes no values.

    `load_session_config` includes pydantic's report verbatim, and that is right there:
    it names the exact field, which is what an operator needs, and a session file holds
    no secrets. Here it would render `secret_access_key`'s input into an exception
    message that ends up in a report or a terminal scrollback, so the fields are named
    and the values are not (ADR-0039).
    """
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(model)"
        variable = f"{ENV_PREFIX}{location.upper()}"
        problems.append(f"  {variable}: {error['msg']}")
    listed = "\n".join(sorted(problems))
    return (
        f"the archive is not configured on this machine. Set these in the environment, "
        f"or point {ENV_PROFILE} at a gitignored profile holding the lowercase keys:\n"
        f"{listed}\n"
        f"Values are deliberately not echoed here — this message reaches reports and "
        f"terminal history."
    )
