from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, Protocol, Self

import httpx

from litellm.rust_bridge.loader import get_native_bridge

_TRANSPORT_HEADERS: Final = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class RustHttpResponse(Protocol):
    status_code: int

    @classmethod
    async def request(
        cls,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        read_timeout_seconds: float | None,
    ) -> Self: ...

    def headers(self) -> list[tuple[str, bytes]]: ...

    async def next_chunk(self) -> bytes | None: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class _HttpTransportState:
    connection: type[RustHttpResponse] | None = None


_STATE: Final = _HttpTransportState()


def set_http_response_connection(*, connection: type[RustHttpResponse] | None) -> None:
    _STATE.connection = connection


def load_http_response_connection() -> type[RustHttpResponse] | None:
    if _STATE.connection is not None:
        return _STATE.connection
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    connection: Final = getattr(native_bridge, "HttpResponseConnection", None)
    return connection if isinstance(connection, type) else None


def rust_http_transport_available() -> bool:
    return load_http_response_connection() is not None


def _read_timeout(timeout: float | httpx.Timeout | None) -> float | None:
    if isinstance(timeout, httpx.Timeout):
        return timeout.read
    if timeout is None:
        return None
    return float(timeout)


def _upstream_headers(headers: httpx.Headers) -> dict[str, str]:
    excluded = set(_TRANSPORT_HEADERS)
    for value in headers.get_list("connection"):
        excluded.update(token.strip().lower() for token in value.split(","))
    return {name: value for name, value in headers.multi_items() if name.lower() not in excluded}


class _RustAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, response: RustHttpResponse) -> None:
        self._response: Final = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while (chunk := await self._response.next_chunk()) is not None:
            yield chunk

    async def aclose(self) -> None:
        await self._response.close()


class RustAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, timeout: float | httpx.Timeout | None = None) -> None:
        self._read_timeout = _read_timeout(timeout)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        connection_type: Final = load_http_response_connection()
        if connection_type is None:
            raise httpx.TransportError("LiteLLM Rust HTTP transport is unavailable", request=request)
        try:
            response: Final = await connection_type.request(
                method=request.method,
                url=str(request.url),
                headers=_upstream_headers(request.headers),
                body=await request.aread(),
                read_timeout_seconds=self._read_timeout,
            )
        except Exception as error:
            raise httpx.TransportError(str(error), request=request) from error
        headers: Final = httpx.Headers([(name.encode("ascii"), value) for name, value in response.headers()])
        headers["x-litellm-rust"] = "true"
        return httpx.Response(
            status_code=response.status_code,
            headers=headers,
            stream=_RustAsyncByteStream(response),
        )


def rust_async_client(timeout: float | httpx.Timeout | None = None) -> httpx.AsyncClient | None:
    if not rust_http_transport_available():
        return None
    return httpx.AsyncClient(transport=RustAsyncTransport(timeout), timeout=timeout)


def rust_async_client_if_enabled(
    timeout: float | httpx.Timeout | None, *, request_override: bool | None
) -> httpx.AsyncClient | None:
    from litellm.rust_bridge.configuration import rust_enabled

    return rust_async_client(timeout) if rust_enabled(request_override=request_override) else None
