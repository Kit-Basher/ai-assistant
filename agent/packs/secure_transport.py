from __future__ import annotations

"""Core-owned HTTPS transport for pack acquisition and scoped public data.

The transport intentionally avoids urllib proxy/cookie/auth behavior.  DNS is
resolved and classified before connecting, the TLS socket is opened directly
to one validated address, and the connected peer is checked again.
"""

from dataclasses import asdict, dataclass
import hashlib
import http.client
import ipaddress
import os
from pathlib import Path
import socket
import ssl
import tempfile
import time
import urllib.parse
from typing import Any, Callable, Iterable

from agent.packs.wp5_contracts import WP5ContractError, _normalized_https_url


MAX_REDIRECTS = 3
MAX_HEADER_BYTES = 32 * 1024
MAX_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 10.0
DEFAULT_TOTAL_TIMEOUT = 30.0
ALLOWED_ARCHIVE_TYPES = {
    "application/zip", "application/x-zip-compressed", "application/x-tar",
    "application/gzip", "application/x-gzip", "application/octet-stream",
}
ALLOWED_PUBLIC_DATA_TYPES = {
    "application/json", "text/plain", "text/csv", "text/html",
    "application/xml", "text/xml", "application/octet-stream",
}


class SecureTransportError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@dataclass(frozen=True)
class SecureFetchResult:
    requested_target: str
    final_target: str
    status: int
    media_type: str
    bytes_received: int
    sha256: str
    redirect_count: int
    elapsed_ms: int
    resolved_address_class: str = "public"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sanitized_target(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = str(parsed.hostname or "").lower()
    netloc = f"[{host}]" if ":" in host else host
    return urllib.parse.urlunsplit(("https", netloc, parsed.path or "/", "", ""))


def _address_allowed(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_private
    )


class SafeHttpsTransport:
    def __init__(
        self,
        *,
        resolver: Callable[[str, int], Iterable[str]] | None = None,
        socket_factory: Callable[[str, int, float], socket.socket] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        allow_query: bool = False,
    ) -> None:
        self._resolver = resolver or self._resolve
        self._socket_factory = socket_factory or self._connect
        self._ssl_context = ssl_context or ssl.create_default_context()
        self._allow_query = bool(allow_query)

    @staticmethod
    def _resolve(host: str, port: int) -> Iterable[str]:
        rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        return tuple(dict.fromkeys(str(row[4][0]) for row in rows))

    @staticmethod
    def _connect(address: str, port: int, timeout: float) -> socket.socket:
        return socket.create_connection((address, port), timeout=timeout)

    def fetch_bytes(
        self,
        url: str,
        *,
        method: str = "GET",
        max_bytes: int = MAX_BODY_BYTES,
        allowed_content_types: set[str] | None = None,
        cancellation: Callable[[], bool] | None = None,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
    ) -> tuple[bytes, SecureFetchResult]:
        chunks: list[bytes] = []
        result = self._request(
            url,
            method=method,
            max_bytes=max_bytes,
            allowed_content_types=allowed_content_types,
            cancellation=cancellation,
            total_timeout=total_timeout,
            sink=chunks.append,
        )
        return b"".join(chunks), result

    def fetch_to_temp(
        self,
        url: str,
        *,
        parent: Path,
        max_bytes: int = MAX_BODY_BYTES,
        allowed_content_types: set[str] | None = None,
        cancellation: Callable[[], bool] | None = None,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
    ) -> tuple[Path, SecureFetchResult]:
        parent = parent.resolve()
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="pack-fetch-", suffix=".partial", dir=str(parent))
        os.chmod(name, 0o600)
        path = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                result = self._request(
                    url,
                    method="GET",
                    max_bytes=max_bytes,
                    allowed_content_types=allowed_content_types,
                    cancellation=cancellation,
                    total_timeout=total_timeout,
                    sink=handle.write,
                )
                handle.flush()
                os.fsync(handle.fileno())
            return path, result
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _request(
        self,
        url: str,
        *,
        method: str,
        max_bytes: int,
        allowed_content_types: set[str] | None,
        cancellation: Callable[[], bool] | None,
        total_timeout: float,
        sink: Callable[[bytes], Any],
    ) -> SecureFetchResult:
        method = str(method or "GET").upper()
        if method not in {"GET", "HEAD"}:
            raise SecureTransportError("http_method_denied", "Only bounded HTTPS GET and HEAD requests are supported.")
        if max_bytes < 1 or max_bytes > MAX_BODY_BYTES:
            raise SecureTransportError("response_limit_invalid", "The requested response limit is outside the broker ceiling.")
        started = time.monotonic()
        try:
            current = _normalized_https_url(url, allow_query=self._allow_query)
        except WP5ContractError as exc:
            raise SecureTransportError(str(exc), "The HTTPS target failed source validation.") from exc
        requested = current
        redirects = 0
        while True:
            if time.monotonic() - started > total_timeout:
                raise SecureTransportError("total_timeout", "The HTTPS request exceeded its total time limit.")
            parsed = urllib.parse.urlsplit(current)
            host = str(parsed.hostname or "").lower()
            port = int(parsed.port or 443)
            addresses = tuple(dict.fromkeys(str(item) for item in self._resolver(host, port)))
            if not addresses:
                raise SecureTransportError("dns_no_addresses", "The HTTPS target did not resolve to an address.")
            if any(not _address_allowed(address) for address in addresses):
                raise SecureTransportError("private_address_denied", "The HTTPS target resolved to a non-public address and was blocked.")
            address = addresses[0]
            remaining = max(0.1, total_timeout - (time.monotonic() - started))
            raw: socket.socket | None = None
            tls: ssl.SSLSocket | None = None
            try:
                raw = self._socket_factory(address, port, min(DEFAULT_CONNECT_TIMEOUT, remaining))
                tls = self._ssl_context.wrap_socket(raw, server_hostname=host)
                raw = None
                tls.settimeout(min(DEFAULT_READ_TIMEOUT, remaining))
                peer = str(tls.getpeername()[0])
                if peer.split("%", 1)[0] != address.split("%", 1)[0] or not _address_allowed(peer):
                    raise SecureTransportError("peer_address_mismatch", "The connected HTTPS peer did not match the validated public target.")
                target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                host_header = host if port == 443 else f"{host}:{port}"
                request = f"{method} {target} HTTP/1.1\r\nHost: {host_header}\r\nUser-Agent: Personal-Agent/pack-broker-v1\r\nAccept: */*\r\nConnection: close\r\n\r\n"
                tls.sendall(request.encode("ascii"))
                response = http.client.HTTPResponse(tls, method=method)
                response.begin()
                header_bytes = sum(len(str(key).encode()) + len(str(value).encode()) + 4 for key, value in response.getheaders())
                if header_bytes > MAX_HEADER_BYTES:
                    raise SecureTransportError("headers_too_large", "The HTTPS response headers exceeded the limit.")
                location = response.getheader("Location")
                if response.status in {301, 302, 303, 307, 308}:
                    if not location:
                        raise SecureTransportError("redirect_location_missing", "The HTTPS redirect did not include a destination.")
                    redirects += 1
                    if redirects > MAX_REDIRECTS:
                        raise SecureTransportError("redirect_limit_exceeded", "The HTTPS request exceeded the redirect limit.")
                    redirected = urllib.parse.urljoin(current, location)
                    try:
                        current = _normalized_https_url(redirected, allow_query=self._allow_query)
                    except WP5ContractError as exc:
                        raise SecureTransportError("redirect_target_denied", "The HTTPS redirect target failed validation.") from exc
                    continue
                if response.status < 200 or response.status >= 300:
                    raise SecureTransportError("http_status_error", f"The HTTPS source returned status {response.status}.")
                media_type = str(response.getheader("Content-Type") or "application/octet-stream").split(";", 1)[0].strip().lower()
                if allowed_content_types is not None and media_type not in allowed_content_types:
                    raise SecureTransportError("content_type_denied", "The HTTPS response content type is not allowed for this operation.")
                content_length = str(response.getheader("Content-Length") or "").strip()
                if content_length:
                    try:
                        declared = int(content_length)
                    except ValueError as exc:
                        raise SecureTransportError("content_length_invalid", "The HTTPS response content length was invalid.") from exc
                    if declared < 0 or declared > max_bytes:
                        raise SecureTransportError("response_too_large", "The HTTPS response exceeded the byte limit.")
                hasher = hashlib.sha256()
                received = 0
                if method != "HEAD":
                    while True:
                        if cancellation and cancellation():
                            raise SecureTransportError("request_cancelled", "The HTTPS request was cancelled.")
                        if time.monotonic() - started > total_timeout:
                            raise SecureTransportError("total_timeout", "The HTTPS request exceeded its total time limit.")
                        chunk = response.read(min(64 * 1024, max_bytes - received + 1))
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > max_bytes:
                            raise SecureTransportError("response_too_large", "The HTTPS response exceeded the byte limit.")
                        hasher.update(chunk)
                        sink(chunk)
                    if content_length and received != int(content_length):
                        raise SecureTransportError("truncated_response", "The HTTPS response ended before its declared body length.")
                return SecureFetchResult(
                    requested_target=_sanitized_target(requested),
                    final_target=_sanitized_target(current),
                    status=int(response.status),
                    media_type=media_type,
                    bytes_received=received,
                    sha256=hasher.hexdigest(),
                    redirect_count=redirects,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
            except SecureTransportError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException, socket.timeout) as exc:
                code = "tls_or_transport_failure" if isinstance(exc, ssl.SSLError) else "transport_failure"
                raise SecureTransportError(code, "The HTTPS connection failed safely.") from exc
            finally:
                if tls is not None:
                    try:
                        tls.close()
                    except OSError:
                        pass
                if raw is not None:
                    try:
                        raw.close()
                    except OSError:
                        pass


__all__ = [
    "ALLOWED_ARCHIVE_TYPES", "ALLOWED_PUBLIC_DATA_TYPES", "MAX_BODY_BYTES",
    "SafeHttpsTransport", "SecureFetchResult", "SecureTransportError", "_address_allowed",
]
