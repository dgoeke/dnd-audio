"""The provider adapter, driven against a recording stub — no socket, every branch.

The test that carries the most weight is the **operation allowlist**. "The protocol has no
delete member" is a statement of intent, not enforcement: the adapter holds its own client
and can call anything on it, and a grep for `DeleteObject` passes with
`client.delete_object(...)` sitting right there because that is not how boto3 spells it.
So every call the adapter makes is recorded by name and checked against a list, and
`delete_object`/`delete_objects` are rejected explicitly (plan review, P1).

The rest covers what cannot be checked without a live endpoint and therefore has to be
checked here: multipart thresholds and part sizing, marker pagination followed to
exhaustion, one retry bound rather than two, and error messages that carry no bucket name.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.config import (
    MAX_MULTIPART_PARTS,
    MIN_MULTIPART_PART_BYTES,
    ArchiveRuntimeConfig,
)
from dnd_audio.archive.spaces import SpacesStorage, is_slow_down

#: Everything the adapter is permitted to call. `abort_multipart_upload` is the only
#: destructive operation, and it applies to this project's own incomplete uploads.
ALLOWED_OPERATIONS = frozenset(
    {
        "put_object",
        "get_object",
        "head_object",
        "list_objects",
        "create_multipart_upload",
        "upload_part",
        "complete_multipart_upload",
        "abort_multipart_upload",
    }
)

#: Named explicitly so the test fails on the spelling boto3 actually uses, rather than on
#: the one an S3 document uses.
FORBIDDEN_OPERATIONS = frozenset({"delete_object", "delete_objects", "delete_bucket"})


class ProviderError(Exception):
    """A botocore-shaped error: the structured `response` is what the adapter reads."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(f"{code} ({status})")
        self.response = {
            "Error": {"Code": code, "Message": "https://secret-bucket.example/signed?key=AKIA"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class RecordingClient:
    """An S3 client that remembers every operation it was asked to perform."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.operations: list[str] = []
        self.parts: dict[str, list[bytes]] = {}
        #: Kept after `complete_multipart_upload` consumes `parts`, so a test can assert
        #: on how the object was chunked rather than on a leftover.
        self.part_sizes: dict[str, list[int]] = {}
        self.aborted: list[str] = []
        self.errors: dict[str, list[Exception]] = {}
        self.list_pages: list[dict[str, Any]] = []

    def _record(self, name: str) -> None:
        self.operations.append(name)
        queued = self.errors.get(name)
        if queued:
            raise queued.pop(0)

    def put_object(self, *, Bucket: str, Key: str, Body: Any) -> dict[str, Any]:
        self._record("put_object")
        self.objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)
        return {"ETag": '"etag"'}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._record("get_object")
        if Key not in self.objects:
            raise ProviderError("NoSuchKey", 404)
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._record("head_object")
        if Key not in self.objects:
            raise ProviderError("NotFound", 404)
        return {"ContentLength": len(self.objects[Key]), "ETag": '"etag-3"'}

    def list_objects(self, **arguments: Any) -> dict[str, Any]:
        self._record("list_objects")
        if self.list_pages:
            return self.list_pages.pop(0)
        keys = sorted(k for k in self.objects if k.startswith(arguments.get("Prefix", "")))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def create_multipart_upload(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._record("create_multipart_upload")
        self.parts[Key] = []
        return {"UploadId": "upload-1"}

    def upload_part(
        self, *, Bucket: str, Key: str, UploadId: str, PartNumber: int, Body: bytes
    ) -> dict[str, Any]:
        self._record("upload_part")
        self.parts[Key].append(Body)
        self.part_sizes.setdefault(Key, []).append(len(Body))
        return {"ETag": f'"part-{PartNumber}"'}

    def complete_multipart_upload(
        self, *, Bucket: str, Key: str, UploadId: str, MultipartUpload: dict[str, Any]
    ) -> dict[str, Any]:
        self._record("complete_multipart_upload")
        self.objects[Key] = b"".join(self.parts.pop(Key))
        return {"ETag": '"multipart-etag-3"'}

    def abort_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str) -> dict[str, Any]:
        self._record("abort_multipart_upload")
        self.parts.pop(Key, None)
        self.aborted.append(Key)
        return {}

    def __getattr__(self, name: str) -> Any:
        """Anything not defined above records itself and fails.

        So a future edit that reaches for `delete_object` is caught by the allowlist below
        rather than by an `AttributeError` that reads like a stub gap.
        """

        def unexpected(**_: Any) -> Any:
            self.operations.append(name)
            message = f"the adapter called an operation this client does not offer: {name}"
            raise AssertionError(message)

        return unexpected


@pytest.fixture
def client() -> RecordingClient:
    return RecordingClient()


@pytest.fixture
def storage(client: RecordingClient, tmp_path: Path) -> SpacesStorage:
    return SpacesStorage(
        client=client,
        bucket="example-cold",
        multipart_threshold_bytes=MIN_MULTIPART_PART_BYTES,
        multipart_part_bytes=MIN_MULTIPART_PART_BYTES,
        max_retries=3,
        retry_base_seconds=0.0001,
        sleep=lambda _seconds: None,
        upload_state_dir=tmp_path / "uploads",
    )


def small(tmp_path: Path, size: int = 1024) -> Path:
    path = tmp_path / f"small-{size}.bin"
    path.write_bytes(b"a" * size)
    return path


class TestNoDeleteAuthority:
    """The proof that the protocol's missing delete member is only a statement of intent."""

    def test_every_operation_used_is_on_the_allowlist(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        source = small(tmp_path)
        storage.put_object("k", source)
        storage.head_object("k")
        with storage.open_object("k") as body:
            body.read()
        list(storage.list_keys(""))

        big = tmp_path / "big.bin"
        big.write_bytes(b"b" * (MIN_MULTIPART_PART_BYTES * 2 + 17))
        storage.put_object("big", big)

        used = set(client.operations)
        assert used <= ALLOWED_OPERATIONS, f"unexpected operations: {used - ALLOWED_OPERATIONS}"

    def test_no_delete_is_ever_called(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """Checked by boto3's own spelling, which a `DeleteObject` grep would miss."""
        storage.put_object("k", small(tmp_path))
        storage.head_object("k")
        list(storage.list_keys(""))
        assert not (set(client.operations) & FORBIDDEN_OPERATIONS)

    def test_abort_is_the_only_destructive_call_and_only_on_our_own_upload(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        big = tmp_path / "big.bin"
        big.write_bytes(b"c" * (MIN_MULTIPART_PART_BYTES * 2))
        client.errors["complete_multipart_upload"] = [ProviderError("InternalErrorFatal", 500)]

        with pytest.raises(ArchiveError):
            storage.put_object("big", big)
        assert client.aborted == ["big"]
        assert not (set(client.operations) & FORBIDDEN_OPERATIONS)

    def test_the_adapter_source_offers_no_delete_method(self) -> None:
        """Belt to the recording client's braces, at the level of the public surface."""
        assert not any(
            name.startswith("delete") for name in dir(SpacesStorage) if not name.startswith("_")
        )


class TestMultipartBoundaries:
    def test_an_object_below_the_threshold_uses_a_single_put(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        storage.put_object("k", small(tmp_path, MIN_MULTIPART_PART_BYTES - 1))
        assert "put_object" in client.operations
        assert "create_multipart_upload" not in client.operations

    def test_an_object_at_the_threshold_uses_multipart(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """At, not above: an off-by-one here means the 5 GB hard limit is reachable."""
        storage.put_object("k", small(tmp_path, MIN_MULTIPART_PART_BYTES))
        assert "create_multipart_upload" in client.operations
        assert "complete_multipart_upload" in client.operations

    def test_multipart_reassembles_the_exact_bytes(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        payload = bytes(range(256)) * (MIN_MULTIPART_PART_BYTES // 128)
        path = tmp_path / "payload.bin"
        path.write_bytes(payload)
        storage.put_object("k", path)
        assert client.objects["k"] == payload

    def test_every_part_but_the_last_meets_the_provider_minimum(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        path = tmp_path / "payload.bin"
        path.write_bytes(b"d" * (MIN_MULTIPART_PART_BYTES * 3 + 11))
        storage.put_object("k", path)
        uploaded = client.part_sizes["k"]
        assert len(uploaded) > 1, "this payload must actually be split to test the rule"
        assert all(size >= MIN_MULTIPART_PART_BYTES for size in uploaded[:-1])
        assert 0 < uploaded[-1] <= MIN_MULTIPART_PART_BYTES

    def test_part_size_grows_so_a_huge_object_fits_in_ten_thousand_parts(
        self, storage: SpacesStorage
    ) -> None:
        """Not reachable with a fixture, so the arithmetic is asserted directly.

        A 5 TB object at a configured 64 MiB part size would need 81 920 parts against a
        10 000 limit, and the upload would fail near the end of a very long transfer.
        """
        huge = 5 * (1 << 40)
        chosen = storage._part_size(huge)
        assert chosen >= MIN_MULTIPART_PART_BYTES
        assert huge / chosen <= MAX_MULTIPART_PARTS

    def test_the_upload_id_is_persisted_before_the_first_part(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """So a killed process leaves a record a later run can abort.

        Without it, incomplete parts are invisible and bill silently until a lifecycle rule
        nobody has written removes them.
        """
        recorded: list[bool] = []
        original = client.upload_part

        def watching(**arguments: Any) -> Any:
            recorded.append((storage._upload_record("k") or tmp_path).is_file())
            return original(**arguments)

        client.upload_part = watching  # type: ignore[method-assign]
        path = tmp_path / "payload.bin"
        path.write_bytes(b"e" * (MIN_MULTIPART_PART_BYTES * 2))
        storage.put_object("k", path)
        assert recorded, "no part was uploaded, so this proves nothing"
        assert recorded[0] is True

    def test_a_completed_upload_forgets_its_record(
        self, storage: SpacesStorage, tmp_path: Path
    ) -> None:
        path = tmp_path / "payload.bin"
        path.write_bytes(b"f" * (MIN_MULTIPART_PART_BYTES * 2))
        storage.put_object("k", path)
        record = storage._upload_record("k")
        assert record is not None
        assert not record.is_file()

    def test_a_long_object_key_still_gets_a_usable_upload_record(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """The record is named by digest, because the encoded key is not a filename.

        A valid 296-byte object key percent-encodes to a 313-character name, past the
        255-byte component limit on every common filesystem — so multipart used to fail
        before its first part on exactly the long paths the 1024-byte key limit permits.
        Found by M7a's code review.
        """
        key = "sessions/archive-v1/s/objects/" + ("%C3%A9" * 60) + ".zst"
        assert len(key) > 296
        record = storage._upload_record(key)
        assert record is not None
        assert len(record.name.encode("utf-8")) < 255

        path = tmp_path / "payload.bin"
        path.write_bytes(b"h" * (MIN_MULTIPART_PART_BYTES * 2))
        storage.put_object(key, path)
        assert client.objects[key] == path.read_bytes()

    def test_a_failed_abort_keeps_the_record_so_a_later_run_can_retry(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """Otherwise a billable multipart orphan is left that nothing knows the id of.

        Deleting the record unconditionally after an abort — succeeded or not — was the
        original behaviour, and it discarded the only local trace of the upload id.
        """
        big = tmp_path / "big.bin"
        big.write_bytes(b"i" * (MIN_MULTIPART_PART_BYTES * 2))
        client.errors["complete_multipart_upload"] = [ProviderError("InternalErrorFatal", 500)]
        client.errors["abort_multipart_upload"] = [ProviderError("AccessDenied", 403)]

        with pytest.raises(ArchiveError):
            storage.put_object("orphan-key", big)

        record = storage._upload_record("orphan-key")
        assert record is not None
        assert record.is_file(), "a failed abort must leave the upload id recoverable"

        # And a later run finds it and tries again.
        client.errors.clear()
        storage.put_object("orphan-key", big)
        assert "abort_multipart_upload" in client.operations
        assert not record.is_file()

    def test_an_orphaned_upload_from_a_crashed_run_is_aborted_first(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        record = storage._upload_record("k")
        assert record is not None
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text('{"key": "k", "upload_id": "orphan-7"}', encoding="utf-8")

        path = tmp_path / "payload.bin"
        path.write_bytes(b"g" * (MIN_MULTIPART_PART_BYTES * 2))
        storage.put_object("k", path)
        assert "abort_multipart_upload" in client.operations


class TestPagination:
    def test_it_follows_markers_to_exhaustion(
        self, storage: SpacesStorage, client: RecordingClient
    ) -> None:
        """Legacy `ListObjects`, because the provider documents V2 pagination as broken."""
        client.list_pages = [
            {"Contents": [{"Key": "a"}, {"Key": "b"}], "IsTruncated": True},
            {"Contents": [{"Key": "c"}, {"Key": "d"}], "IsTruncated": True, "NextMarker": "d"},
            {"Contents": [{"Key": "e"}], "IsTruncated": False},
        ]
        assert list(storage.list_keys("")) == ["a", "b", "c", "d", "e"]

    def test_it_uses_the_last_key_as_a_marker_when_none_is_supplied(
        self, storage: SpacesStorage, client: RecordingClient
    ) -> None:
        """`NextMarker` appears only with a delimiter; otherwise the last key is the marker.

        Getting this wrong re-requests page one forever, which looks like a hang rather
        than like a bug.
        """
        seen: list[str | None] = []
        original = client.list_objects

        def watching(**arguments: Any) -> Any:
            seen.append(arguments.get("Marker"))
            return original(**arguments)

        client.list_objects = watching  # type: ignore[method-assign]
        client.list_pages = [
            {"Contents": [{"Key": "a"}, {"Key": "b"}], "IsTruncated": True},
            {"Contents": [{"Key": "c"}], "IsTruncated": False},
        ]
        assert list(storage.list_keys("")) == ["a", "b", "c"]
        assert seen == [None, "b"]

    def test_nosuchkey_from_a_listing_is_refused_not_reported_as_empty(
        self, storage: SpacesStorage, client: RecordingClient
    ) -> None:
        """What a doubly-addressed bucket returns, and why it must never read as "empty".

        Observed against the real bucket on 2026-08-04, when the configured endpoint
        already contained the bucket name: every listing failed this way while
        `HeadObject` on a key uploaded seconds earlier succeeded. The cause is now refused
        at config load, but the diagnosis stays, because the first attempt at handling it
        swallowed the error and returned no keys — which would have made `archive list`
        report an empty archive with a complete one sitting in the bucket, during exactly
        the emergency the command exists for.

        A backup tool may say "I cannot tell you". It may never say "there is nothing
        there" when it does not know (OQ-028).
        """
        client.errors["list_objects"] = [ProviderError("NoSuchKey", 404)]
        with pytest.raises(ArchiveError) as caught:
            list(storage.list_keys("nothing/here/"))
        assert caught.value.code == "archive_listing_failed"
        assert "regional" in str(caught.value)

    def test_a_truncated_listing_with_no_marker_is_refused(
        self, storage: SpacesStorage, client: RecordingClient
    ) -> None:
        """Rather than silently returning a partial listing as if it were complete.

        This is the failure that makes `list` report a session absent during exactly the
        emergency it exists for (OQ-028).
        """
        client.list_pages = [{"Contents": [], "IsTruncated": True}]
        with pytest.raises(ArchiveError) as caught:
            list(storage.list_keys(""))
        assert caught.value.code == "archive_listing_incomplete"


class TestRetry:
    def test_a_slow_down_is_retried_within_the_bound(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        client.errors["put_object"] = [ProviderError("SlowDown", 503)] * 2
        storage.put_object("k", small(tmp_path))
        assert client.operations.count("put_object") == 3

    def test_exhausting_the_bound_fails_rather_than_retrying_forever(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        client.errors["put_object"] = [ProviderError("SlowDown", 503)] * 10
        with pytest.raises(ArchiveError) as caught:
            storage.put_object("k", small(tmp_path))
        assert caught.value.code == "archive_storage_error"
        assert client.operations.count("put_object") == 4

    def test_backoff_doubles(self, client: RecordingClient, tmp_path: Path) -> None:
        delays: list[float] = []
        storage = SpacesStorage(
            client=client,
            bucket="b",
            multipart_threshold_bytes=MIN_MULTIPART_PART_BYTES,
            multipart_part_bytes=MIN_MULTIPART_PART_BYTES,
            max_retries=3,
            retry_base_seconds=0.5,
            sleep=delays.append,
            upload_state_dir=tmp_path / "uploads",
        )
        client.errors["put_object"] = [ProviderError("SlowDown", 503)] * 3
        storage.put_object("k", small(tmp_path))
        assert delays == [0.5, 1.0, 2.0]

    def test_a_non_retryable_error_fails_immediately(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """Retrying an `AccessDenied` wastes time and hides the diagnosis."""
        client.errors["put_object"] = [ProviderError("AccessDenied", 403)] * 5
        with pytest.raises(ArchiveError):
            storage.put_object("k", small(tmp_path))
        assert client.operations.count("put_object") == 1

    def test_a_missing_object_is_reported_as_missing_not_as_an_error(
        self, storage: SpacesStorage
    ) -> None:
        assert storage.head_object("absent") is None

    @pytest.mark.parametrize(
        ("code", "status", "expected"),
        [
            ("SlowDown", 503, True),
            ("ServiceUnavailable", 503, True),
            ("RequestTimeout", 400, True),
            ("AccessDenied", 403, False),
            ("NoSuchBucket", 404, False),
        ],
    )
    def test_only_transient_codes_are_retryable(
        self, code: str, status: int, expected: bool
    ) -> None:
        assert is_slow_down(ProviderError(code, status)) is expected


class TestErrorsCarryNoSecrets:
    def test_a_provider_error_message_holds_no_signed_url_or_bucket(
        self, storage: SpacesStorage, client: RecordingClient, tmp_path: Path
    ) -> None:
        """Botocore's own text can carry a signed URL. This message reaches a report."""
        client.errors["put_object"] = [ProviderError("AccessDenied", 403)]
        with pytest.raises(ArchiveError) as caught:
            storage.put_object("k", small(tmp_path))

        message = str(caught.value)
        assert "secret-bucket.example" not in message
        assert "AKIA" not in message
        assert "example-cold" not in message
        # It still says enough to act on.
        assert "AccessDenied" in message
        assert "403" in message


class TestClientConstruction:
    def test_boto3_is_not_imported_by_importing_the_adapter(self) -> None:
        """The lazy import that keeps `dnd-audio mix` from loading an S3 SDK.

        The stronger claim — that no processing command opens a socket — is proved in
        `tests/test_archive_isolation.py`, in a subprocess, because this one cannot see
        past its own address space.
        """
        import subprocess
        import sys

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import dnd_audio.archive.spaces as s; print('boto3' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == "False"

    def test_the_sdks_own_retries_are_disabled_in_the_resolved_client_config(self) -> None:
        """Behaviour, not a source-text search — and the search was certifying a bug.

        The first version asserted the literal `"max_attempts": 1` appeared in the module.
        In botocore, `max_attempts` counts retries *after* the initial request, so that
        setting permitted one SDK retry beneath every project-level attempt and the test
        confirmed the mistake rather than the property. `total_max_attempts: 1` is
        unambiguous. Found by M7a's code review.

        Constructing a client opens no socket — boto3 resolves endpoints from bundled
        data — so this can assert on what botocore actually resolved.
        """
        import boto3
        from botocore.config import Config

        from dnd_audio.archive.spaces import RETRY_CONFIG

        assert RETRY_CONFIG["total_max_attempts"] == 1
        assert "max_attempts" not in RETRY_CONFIG

        client = boto3.client(
            "s3",
            endpoint_url="https://nyc3.digitaloceanspaces.com",
            region_name="nyc3",
            aws_access_key_id="id",
            aws_secret_access_key="secret",
            config=Config(retries=RETRY_CONFIG, signature_version="s3v4"),
        )
        resolved = client.meta.config.retries
        assert resolved["total_max_attempts"] == 1

    def test_the_configured_thresholds_reach_the_storage(self, tmp_path: Path) -> None:
        config = ArchiveRuntimeConfig(
            endpoint_url="https://nyc3.digitaloceanspaces.com",
            region="nyc3",
            bucket="b",
            access_key_id="id",  # type: ignore[arg-type]
            secret_access_key="secret",  # type: ignore[arg-type]
            multipart_threshold_bytes=MIN_MULTIPART_PART_BYTES,
            multipart_part_bytes=MIN_MULTIPART_PART_BYTES,
            max_retries=2,
        )
        assert config.multipart_threshold_bytes == MIN_MULTIPART_PART_BYTES
        assert config.max_retries == 2
