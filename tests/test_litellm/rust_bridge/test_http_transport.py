from __future__ import annotations

import json
from collections.abc import Generator
from typing import Self

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.rust_bridge import http_transport


class _FakeResponse:
    status_code = 200

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)

    def headers(self) -> list[tuple[str, bytes]]:
        return [("content-type", b"application/json")]

    async def next_chunk(self) -> bytes | None:
        return next(self._chunks, None)

    async def close(self) -> None:
        pass


class _FakeConnection(_FakeResponse):
    calls: list[dict[str, object]] = []
    chunks: list[bytes] = []

    @classmethod
    async def request(
        cls,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        read_timeout_seconds: float | None,
    ) -> Self:
        cls.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": body,
                "read_timeout_seconds": read_timeout_seconds,
            }
        )
        return cls(cls.chunks)


@pytest.fixture(autouse=True)
def reset_transport() -> Generator[None, None, None]:
    _FakeConnection.calls = []
    _FakeConnection.chunks = []
    http_transport.set_http_response_connection(connection=_FakeConnection)
    yield
    http_transport.set_http_response_connection(connection=None)


@pytest.mark.asyncio
async def test_non_streaming_request_preserves_tools_and_json_schema() -> None:
    _FakeConnection.chunks = [json.dumps({"choices": [{"message": {"tool_calls": []}}]}).encode()]
    body = {
        "messages": [{"role": "user", "content": "classify"}],
        "tools": [{"type": "function", "function": {"name": "result", "parameters": {"type": "object"}}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}}},
    }
    client = http_transport.rust_async_client_if_enabled(httpx.Timeout(30.0), request_override=True)
    assert client is not None
    async with client:
        response = await client.post("https://example.test/v1/chat/completions", json=body)

    assert response.headers["x-litellm-rust"] == "true"
    assert response.json()["choices"][0]["message"]["tool_calls"] == []
    request_body = _FakeConnection.calls[0]["body"]
    assert isinstance(request_body, bytes)
    assert json.loads(request_body) == body


@pytest.mark.asyncio
async def test_streaming_request_yields_chunks_without_buffering() -> None:
    _FakeConnection.chunks = [b"data: one\n\n", b"data: two\n\n"]
    client = http_transport.rust_async_client(30.0)
    assert client is not None
    async with client:
        async with client.stream("POST", "https://example.test/v1/responses", json={"stream": True}) as response:
            chunks = [chunk async for chunk in response.aiter_bytes()]

    assert chunks == [b"data: one\n\n", b"data: two\n\n"]
    assert _FakeConnection.calls[0]["read_timeout_seconds"] == 30.0


def test_client_selection_handles_disabled_and_missing_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    assert http_transport.rust_async_client_if_enabled(30.0, request_override=False) is None

    http_transport.set_http_response_connection(connection=None)
    monkeypatch.setattr(http_transport, "get_native_bridge", lambda: None)
    assert http_transport.rust_async_client_if_enabled(30.0, request_override=True) is None


@pytest.mark.asyncio
async def test_chat_output_matches_python_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "id": "chatcmpl-parity",
        "object": "chat.completion",
        "created": 1741476542,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "same"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    async def python_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    python_client = httpx.AsyncClient(transport=httpx.MockTransport(python_response))
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_async_http_client",
        staticmethod(lambda shared_session=None: python_client),
    )
    python = await litellm.acompletion(
        model="openai/test-model",
        api_base="https://python.example/v1",
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
        rust=False,
    )

    _FakeConnection.chunks = [json.dumps(body).encode()]
    rust = await litellm.acompletion(
        model="openai/test-model",
        api_base="https://rust.example/v1",
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
        rust=True,
    )

    assert rust.model_dump() == python.model_dump()
    assert _FakeConnection.calls[-1]["url"] == "https://rust.example/v1/chat/completions"


@pytest.mark.asyncio
async def test_responses_output_matches_python_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "id": "resp-parity",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "id": "msg-parity",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "same", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    async def python_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    python_client = httpx.AsyncClient(transport=httpx.MockTransport(python_response))
    monkeypatch.setattr(
        llm_http_handler,
        "get_async_httpx_client",
        lambda **kwargs: AsyncHTTPHandler(client=python_client),
    )
    python = await litellm.aresponses(
        model="openai/test-model",
        api_base="https://python.example/v1",
        api_key="test",
        input="hi",
        rust=False,
    )

    _FakeConnection.chunks = [json.dumps(body).encode()]
    rust = await litellm.aresponses(
        model="openai/test-model",
        api_base="https://rust.example/v1",
        api_key="test",
        input="hi",
        rust=True,
    )

    assert rust.model_dump() == python.model_dump()
    assert _FakeConnection.calls[-1]["url"] == "https://rust.example/v1/responses"
