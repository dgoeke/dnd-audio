"""The DigitalOcean Spaces adapter: the only module that opens a socket for the archive.

`boto3` is imported **inside** :func:`build_storage` and nowhere else. That is what keeps
INV-06's boundary a property of the code rather than a habit — and it is only half the
guarantee, because "did not import a library" is weaker than "did not open a socket". The
other half is `tests/test_archive_isolation.py`, which runs every non-archive command as a
subprocess under a trap that fails on socket *and* client construction.

Four provider facts decide most of what is here, all re-read on 2026-08-04:

* **Single PUT tops out at 5 GB.** Above the configured threshold, multipart is mandatory
  rather than preferred.
* **A part must be at least 5 MiB** (except the last) and there may be at most **10 000**,
  so the part size is raised to `ceil(size / 10000)` for a large object rather than left at
  a configured value that would need more parts than exist.
* **Listing pages with `ListObjectsV2` do not work.** DigitalOcean's compatibility page says
  V2 is supported; its limits page's known-issues section says its *pagination* is not. This
  adapter uses legacy `ListObjects` marker pagination outright rather than V2 with a
  fallback — a path that only runs when a provider bug is present is a path nobody
  exercises, and it would first be exercised during a recovery (**OQ-028**).
* **`503 Slow Down` is ordinary under load**, and Cold Storage's write limit is 450/s.

**Botocore's own retries are turned off.** It retries by default, so a project-level retry
loop layered on top bounds nothing: the real attempt count would be the product of the two
and the real delay their sum. One bound, stated in `ArchiveRuntimeConfig`, and this is where
the other is disabled.

**No delete operation exists here.** DigitalOcean bundles multipart-abort with broad
object Read/Write/Delete permission, so the credential can delete even though this code
never does (ADR-0035). `AbortMultipartUpload` against this project's own incomplete uploads
is the only destructive call, and `tests/test_archive_spaces.py` enforces that through an
operation allowlist on a recording client rather than by grepping for a method name boto3
does not use.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dnd_audio.archive import ArchiveError
from dnd_audio.archive.config import (
    MAX_MULTIPART_PARTS,
    MIN_MULTIPART_PART_BYTES,
    SINGLE_PUT_HARD_LIMIT_BYTES,
    ArchiveRuntimeConfig,
    state_dir,
)
from dnd_audio.archive.paths import encode_component
from dnd_audio.archive.storage import ObjectHead
from dnd_audio.determinism import BinaryReader, write_json_atomic

__all__ = ["SpacesStorage", "build_storage", "is_slow_down"]

#: Provider error codes worth another attempt. Everything else is fatal on the first try:
#: retrying a `NoSuchBucket` or an `AccessDenied` wastes time and hides the diagnosis.
_RETRYABLE: Final = frozenset({"SlowDown", "RequestTimeout", "InternalError", "ServiceUnavailable"})

#: Where an in-flight multipart upload's id is recorded, so a later run can abort what a
#: crashed one left behind. Local, never an object: Cold Storage bills anything under
#: 128 KiB as 128 KiB, and this is bookkeeping rather than archive content.
_UPLOADS_DIRNAME: Final = "uploads"


def is_slow_down(exc: BaseException) -> bool:
    """Whether a provider error is worth retrying.

    Reads the structured error code rather than matching on the message, because the
    message is prose a provider reserves the right to reword.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = response.get("Error", {}).get("Code")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _RETRYABLE or status == 503


@dataclass
class SpacesStorage:
    """`ArchiveStorage` against one S3-compatible bucket.

    Constructed by :func:`build_storage`, which is where `boto3` is imported. The client is
    injected rather than built here so the default suite can drive every branch — multipart
    boundaries, pagination, retry — against a recording stub without a socket.
    """

    client: Any
    bucket: str
    multipart_threshold_bytes: int
    multipart_part_bytes: int
    max_retries: int
    retry_base_seconds: float
    #: Injected so a retry test costs no wall-clock seconds. Production passes `time.sleep`.
    sleep: Callable[[float], None] = time.sleep
    upload_state_dir: Path | None = None

    # --- the protocol ----------------------------------------------------------------

    def put_object(self, key: str, source: Path) -> None:
        size = source.stat().st_size
        if size >= SINGLE_PUT_HARD_LIMIT_BYTES or size >= self.multipart_threshold_bytes:
            self._put_multipart(key, source, size)
            return
        with source.open("rb") as body:
            self._call(lambda: self.client.put_object(Bucket=self.bucket, Key=key, Body=body))

    def head_object(self, key: str) -> ObjectHead | None:
        try:
            response = self._call(lambda: self.client.head_object(Bucket=self.bucket, Key=key))
        except ArchiveError as exc:
            if exc.code == "archive_object_missing":
                return None
            raise
        return ObjectHead(
            key=key,
            size_bytes=int(response["ContentLength"]),
            # Recorded, never compared against a digest. For a multipart object this is a
            # hash of part hashes and identifies nothing about the content (ADR-0038).
            etag=response.get("ETag"),
        )

    @contextmanager
    def open_object(self, key: str) -> Iterator[BinaryReader]:
        response = self._call(lambda: self.client.get_object(Bucket=self.bucket, Key=key))
        body = response["Body"]
        try:
            yield body
        finally:
            body.close()

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Legacy `ListObjects` marker pagination, followed to exhaustion (OQ-028).

        Not `ListObjectsV2`, and not a paginator: the provider documents V2 pagination as
        not working, and boto3's paginator would silently use whichever operation it was
        given. Following `NextMarker`/`IsTruncated` by hand is a dozen lines and is the
        thing that has to be right during a recovery.
        """
        marker: str | None = None
        while True:
            arguments: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if marker is not None:
                arguments["Marker"] = marker
            response = self._call(lambda: self.client.list_objects(**arguments))  # noqa: B023

            contents = response.get("Contents", [])
            for item in contents:
                yield str(item["Key"])

            if not response.get("IsTruncated"):
                return
            # `NextMarker` is only present when a delimiter was used; otherwise the marker
            # is the last key returned. Getting this wrong loops forever on the first page,
            # which is why the empty-page guard below exists rather than being trusted away.
            marker = response.get("NextMarker") or (str(contents[-1]["Key"]) if contents else None)
            if marker is None:
                message = (
                    f"the provider reported a truncated listing of {prefix!r} but supplied "
                    f"no marker to continue from, so the listing cannot be completed. "
                    f"Refusing to treat a partial listing as complete (OQ-028)."
                )
                raise ArchiveError(message, code="archive_listing_incomplete")

    # --- multipart -------------------------------------------------------------------

    def _part_size(self, size: int) -> int:
        """At least the provider minimum, and small enough to fit in the part limit."""
        required = math.ceil(size / MAX_MULTIPART_PARTS)
        return max(self.multipart_part_bytes, MIN_MULTIPART_PART_BYTES, required)

    def _put_multipart(self, key: str, source: Path, size: int) -> None:
        self._abort_orphaned(key)

        created = self._call(
            lambda: self.client.create_multipart_upload(Bucket=self.bucket, Key=key)
        )
        upload_id = str(created["UploadId"])
        # Persisted **before the first part**, so a process killed mid-upload leaves a
        # record a later run can abort. Without it the incomplete parts are invisible and
        # bill silently until a lifecycle rule nobody has written removes them.
        self._record_upload(key, upload_id)

        part_size = self._part_size(size)
        parts: list[dict[str, Any]] = []
        try:
            with source.open("rb") as handle:
                number = 1
                while chunk := handle.read(part_size):
                    tag = self._call(self._part_sender(key, upload_id, number, chunk))
                    parts.append({"ETag": tag["ETag"], "PartNumber": number})
                    number += 1

            self._call(
                lambda: self.client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            )
        except BaseException:
            # Abort rather than resume. A resumed multipart has to reproduce the same part
            # boundaries and carry forward every ETag, and getting that subtly wrong
            # produces an object that completes and is not the file — which the readback
            # would catch, at the cost of the whole transfer anyway. Re-uploading one
            # object is cheaper than being clever here.
            self._abort(key, upload_id)
            raise
        finally:
            self._forget_upload(key)

    def _part_sender(
        self, key: str, upload_id: str, number: int, chunk: bytes
    ) -> Callable[[], Any]:
        """Bind one part's arguments into a thunk the retry loop can safely re-run.

        A bare closure over the loop variables would capture them by reference, so a retry
        after a `503` would resend whatever part the loop had reached by then — uploading
        the wrong bytes under a part number that had already succeeded, and producing an
        object that completes and is not the file. The readback would catch it, at the cost
        of the entire transfer.
        """

        def send() -> Any:
            return self.client.upload_part(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=number,
                Body=chunk,
            )

        return send

    def _abort(self, key: str, upload_id: str) -> None:
        """The only destructive provider call this application makes (ADR-0035)."""
        try:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
        except Exception:
            return

    def _abort_orphaned(self, key: str) -> None:
        """Clean up after a previous run that was killed mid-upload."""
        path = self._upload_record(key)
        if path is None or not path.is_file():
            return
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
            upload_id = str(recorded["upload_id"])
        except (OSError, json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return
        self._abort(key, upload_id)
        path.unlink(missing_ok=True)

    def _upload_record(self, key: str) -> Path | None:
        base = self.upload_state_dir or (state_dir() / _UPLOADS_DIRNAME)
        return base / f"{encode_component(key)}.json"

    def _record_upload(self, key: str, upload_id: str) -> None:
        path = self._upload_record(key)
        if path is None:
            return
        write_json_atomic(path, {"key": key, "upload_id": upload_id})

    def _forget_upload(self, key: str) -> None:
        path = self._upload_record(key)
        if path is not None:
            path.unlink(missing_ok=True)

    # --- retry -----------------------------------------------------------------------

    def _call(self, operation: Callable[[], Any]) -> Any:
        """Run one provider call, retrying only what is worth retrying.

        The single retry bound in the system: botocore's own is disabled in
        :func:`build_storage`, because two bounds multiply into an attempt count nobody
        stated and a delay nobody chose.
        """
        delay = self.retry_base_seconds
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except ArchiveError:
                raise
            except Exception as exc:
                if _is_missing(exc):
                    message = "the object does not exist"
                    raise ArchiveError(message, code="archive_object_missing") from exc
                if not is_slow_down(exc) or attempt == self.max_retries:
                    message = _sanitize(exc)
                    raise ArchiveError(message, code="archive_storage_error") from exc
                self.sleep(delay)
                delay *= 2
        # Unreachable: the loop either returns or raises on its final attempt. Stated so a
        # future edit to the bounds cannot fall out of the bottom silently.
        message = "the retry loop completed without a result"
        raise ArchiveError(message, code="archive_storage_error")


def _is_missing(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = response.get("Error", {}).get("Code")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def _sanitize(exc: BaseException) -> str:
    """A provider error, with nothing that identifies the bucket or the credential.

    Botocore's exception text can carry a signed URL, a bucket name, or a host. This
    message reaches a report and a terminal, so it carries the structured code and the
    HTTP status and nothing else (ADR-0039).
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "unknown")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", "unknown")
        return (
            f"the storage provider refused the request: {code} (HTTP {status}). The "
            f"endpoint, bucket and credentials are deliberately not included here."
        )
    return (
        f"the storage request failed: {type(exc).__name__}. Details are omitted because "
        f"this message reaches reports and terminal history."
    )


def build_storage(
    config: ArchiveRuntimeConfig, *, sleep: Callable[[float], None] = time.sleep
) -> SpacesStorage:
    """Construct the client. **The only place `boto3` is imported** (INV-06, ADR-0035).

    Imported inside the function rather than at module scope so that importing
    `dnd_audio.archive` — which `dnd_audio.cli` does, to register the subcommands — does not
    pull in an S3 SDK on a machine that will only ever run `mix`.
    """
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        region_name=config.region,
        aws_access_key_id=config.access_key_id.get_secret_value(),
        aws_secret_access_key=config.secret_access_key.get_secret_value(),
        config=Config(
            # One bound, and it is `ArchiveRuntimeConfig.max_retries`. Leaving botocore's
            # default on would multiply the two into an attempt count nobody stated.
            retries={"max_attempts": 1, "mode": "standard"},
            signature_version="s3v4",
        ),
    )
    return SpacesStorage(
        client=client,
        bucket=config.bucket,
        multipart_threshold_bytes=config.multipart_threshold_bytes,
        multipart_part_bytes=config.multipart_part_bytes,
        max_retries=config.max_retries,
        retry_base_seconds=config.retry_base_seconds,
        sleep=sleep,
    )
