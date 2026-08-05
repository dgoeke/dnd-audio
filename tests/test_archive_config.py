"""`ArchiveRuntimeConfig`: loaded from outside a session, and never leaking a key."""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_audio.archive.config import (
    ENV_PREFIX,
    ENV_PROFILE,
    ArchiveConfigError,
    default_report_dir,
    load_archive_config,
    state_dir,
)

SECRET = "wJalrXUtnFEMI-K7MDENG-bPxRfiCYEXAMPLEKEY"
KEY_ID = "DO00EXAMPLEACCESSKEY"


def environment(**overrides: str) -> dict[str, str]:
    """A complete archive environment, with any field replaced or removed."""
    values = {
        "ENDPOINT_URL": "https://nyc3.digitaloceanspaces.com",
        "REGION": "nyc3",
        "BUCKET": "example-cold",
        "ACCESS_KEY_ID": KEY_ID,
        "SECRET_ACCESS_KEY": SECRET,
    }
    values.update(overrides)
    return {f"{ENV_PREFIX}{name}": value for name, value in values.items() if value}


class TestLoading:
    def test_it_reads_a_complete_environment(self) -> None:
        config = load_archive_config(environment())
        assert config.bucket == "example-cold"
        assert config.access_key_id.get_secret_value() == KEY_ID
        assert config.secret_access_key.get_secret_value() == SECRET

    def test_an_empty_variable_counts_as_unset(self) -> None:
        """The `models_dir()` lesson: an exported-but-empty variable is not a value.

        `DND_AUDIO_ARCHIVE_BUCKET=` otherwise validates as a bucket named "", and the
        first thing that would notice is a signed request to a nonexistent bucket.
        """
        with pytest.raises(ArchiveConfigError) as caught:
            load_archive_config(environment(BUCKET=""))
        assert "BUCKET" in str(caught.value)

    def test_it_names_every_missing_variable_at_once(self) -> None:
        """One message listing all four, not four runs discovering one each."""
        with pytest.raises(ArchiveConfigError) as caught:
            load_archive_config({})
        message = str(caught.value)
        for field in ("ENDPOINT_URL", "REGION", "BUCKET", "ACCESS_KEY_ID"):
            assert f"{ENV_PREFIX}{field}" in message

    def test_a_plaintext_endpoint_is_refused(self) -> None:
        with pytest.raises(ArchiveConfigError) as caught:
            load_archive_config(environment(ENDPOINT_URL="http://nyc3.digitaloceanspaces.com"))
        assert "https" in str(caught.value)

    def test_a_trailing_slash_is_normalized_away(self) -> None:
        config = load_archive_config(
            environment(ENDPOINT_URL="https://nyc3.digitaloceanspaces.com/")
        )
        assert config.endpoint_url == "https://nyc3.digitaloceanspaces.com"

    def test_an_unknown_setting_in_a_profile_is_an_error_not_a_shrug(self, tmp_path: Path) -> None:
        """`extra="forbid"`, for `SessionConfig`'s reason: a typo would be ignored.

        Driven through a profile rather than through `model_validate` directly, because
        the profile is the surface where a typo is actually reachable — an operator
        hand-writes it, and an ignored `buckett:` would silently archive to whatever the
        environment happened to say.
        """
        profile = tmp_path / "archive.yaml"
        profile.write_text("buckett: typo\n", encoding="utf-8")
        with pytest.raises(ArchiveConfigError) as caught:
            load_archive_config({ENV_PROFILE: str(profile), **environment()})
        assert "buckett" in str(caught.value).lower()


class TestProfile:
    def test_a_profile_supplies_the_same_keys_in_lowercase(self, tmp_path: Path) -> None:
        profile = tmp_path / "archive.yaml"
        profile.write_text(
            "endpoint_url: https://nyc3.digitaloceanspaces.com\n"
            "region: nyc3\n"
            "bucket: from-profile\n"
            f"access_key_id: {KEY_ID}\n"
            f"secret_access_key: {SECRET}\n",
            encoding="utf-8",
        )
        config = load_archive_config({ENV_PROFILE: str(profile)})
        assert config.bucket == "from-profile"

    def test_the_environment_wins_over_the_profile(self, tmp_path: Path) -> None:
        """So one command can be overridden without editing a file you then forget."""
        profile = tmp_path / "archive.yaml"
        profile.write_text(
            "endpoint_url: https://nyc3.digitaloceanspaces.com\n"
            "region: nyc3\n"
            "bucket: from-profile\n"
            f"access_key_id: {KEY_ID}\n"
            f"secret_access_key: {SECRET}\n",
            encoding="utf-8",
        )
        config = load_archive_config(
            {ENV_PROFILE: str(profile), f"{ENV_PREFIX}BUCKET": "from-environment"}
        )
        assert config.bucket == "from-environment"

    def test_an_unreadable_profile_is_fatal_rather_than_ignored(self, tmp_path: Path) -> None:
        """ADR-0007's rule: a pointer at nothing is a mistake, not a silent default."""
        with pytest.raises(ArchiveConfigError) as caught:
            load_archive_config({ENV_PROFILE: str(tmp_path / "absent.yaml")})
        assert ENV_PROFILE in str(caught.value)


class TestSecretsNeverEscape:
    """The single most likely place for an access key to leak is an exception."""

    def test_the_repr_of_the_model_holds_no_secret(self) -> None:
        config = load_archive_config(environment())
        rendered = repr(config) + str(config)
        assert SECRET not in rendered
        assert KEY_ID not in rendered

    def test_a_validation_error_quotes_no_value(self) -> None:
        """Pydantic renders offending input by default, and one of these is a key.

        `load_session_config` includes pydantic's report verbatim and is right to; a
        session file holds nothing secret. Here the same habit would print an access key
        into a terminal scrollback and, through the report writer, into a file.
        """
        with pytest.raises(ArchiveConfigError) as caught:
            load_archive_config(environment(ENDPOINT_URL="http://insecure.example"))
        message = str(caught.value)
        assert SECRET not in message
        assert KEY_ID not in message
        assert "insecure.example" not in message

    def test_serialization_does_not_expose_the_key(self) -> None:
        dumped = str(load_archive_config(environment()).model_dump())
        assert SECRET not in dumped
        assert KEY_ID not in dumped


class TestStatePaths:
    def test_state_lives_outside_every_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lock inside a source root would itself violate INV-01."""
        monkeypatch.setenv("XDG_STATE_HOME", "/tmp/example-state")
        assert state_dir() == Path("/tmp/example-state/dnd-audio/archive")
        assert default_report_dir() == Path("/tmp/example-state/dnd-audio/archive/reports")

    def test_it_falls_back_to_the_conventional_location(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        assert state_dir().parts[-3:] == (".local", "state", "dnd-audio") or state_dir().match(
            "*/.local/state/dnd-audio/archive"
        )
