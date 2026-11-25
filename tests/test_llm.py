from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.genai import errors, types
import httpx
import pytest

from storeassist.llm import (
    FakeLLMClient,
    GeminiClient,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
)
from storeassist.store_client import StoreClient
from storeassist.tools import Tool, build_tools


def _sample_tools() -> list[Tool]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "owner"})
        return httpx.Response(200, json=[])

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        return build_tools(client)


@dataclass
class _StubFunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class _StubPart:
    function_call: _StubFunctionCall | None = None
    thought_signature: bytes | None = None
    text: str | None = None


@dataclass
class _StubContent:
    parts: list[_StubPart]


@dataclass
class _StubCandidate:
    content: _StubContent


@dataclass
class _StubResponse:
    function_calls: list[_StubFunctionCall] | None = None
    candidates: list[_StubCandidate] | None = None
    text: str | None = None


class _StubModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        model: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _StubSDKClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.models = _StubModels(outcomes)


def test_llm_response_requires_exactly_one_payload() -> None:
    with pytest.raises(ValueError):
        LLMResponse()
    with pytest.raises(ValueError):
        LLMResponse(text="hi", tool_call=ToolCall(name="get_low_stock", arguments={}))


def test_fake_llm_client_records_calls_and_scripts_turns() -> None:
    tools = _sample_tools()
    fake = FakeLLMClient(
        [
            LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})),
            LLMResponse(text="Nothing is low."),
        ]
    )
    messages = [Message(role="user", content="What is low?")]

    first = fake.generate(messages, "System prompt", tools)
    messages.append(Message(role="model", tool_call=first.tool_call))
    messages.append(Message(role="tool", tool_name="get_low_stock", tool_result={"items": []}))
    second = fake.generate(messages, "System prompt", tools)

    assert first.tool_call == ToolCall(name="get_low_stock", arguments={})
    assert second.text == "Nothing is low."
    assert len(fake.calls) == 2
    assert len(fake.calls[0].messages) == 1
    assert len(fake.calls[1].messages) == 3
    assert fake.calls[0].system_prompt == "System prompt"
    assert "get_daily_sales" in fake.calls[0].tool_names

    with pytest.raises(LLMError):
        fake.generate(messages, "System prompt", tools)


def test_gemini_client_builds_manual_tool_config_and_parses_tool_call_and_text() -> None:
    tools = _sample_tools()
    sdk_client = _StubSDKClient(
        [
            _StubResponse(function_calls=[_StubFunctionCall(name="get_sales", args={"date": "2026-11-25"})]),
            _StubResponse(text="Total sales today are 500 rupees."),
        ]
    )
    gemini = GeminiClient(api_key="dummy-key", model="gemini-3.5-flash-lite", sdk_client=sdk_client)

    messages = [
        Message(role="user", content="Sales today?"),
        Message(role="model", tool_call=ToolCall(name="get_sales", arguments={"date": "2026-11-25"})),
        Message(role="tool", tool_name="get_sales", tool_result={"items": [], "total_count": 0}),
    ]

    first = gemini.generate([messages[0]], "Read-only assistant", tools)
    second = gemini.generate(messages, "Read-only assistant", tools)

    assert first == LLMResponse(tool_call=ToolCall(name="get_sales", arguments={"date": "2026-11-25"}))
    assert second == LLMResponse(text="Total sales today are 500 rupees.")

    recorded_call = sdk_client.models.calls[1]
    assert recorded_call["model"] == "gemini-3.5-flash-lite"
    config = recorded_call["config"]
    assert config.temperature == 0.0
    assert config.system_instruction == "Read-only assistant"
    assert config.automatic_function_calling == types.AutomaticFunctionCallingConfig(disable=True)
    assert config.tools is not None
    declarations = config.tools[0].function_declarations
    assert declarations is not None
    decl_names = [decl.name for decl in declarations]
    assert decl_names == [tool.name for tool in tools]
    get_sales_decl = next(decl for decl in declarations if decl.name == "get_sales")
    assert get_sales_decl.parameters_json_schema is not None
    assert "date" in get_sales_decl.parameters_json_schema["properties"]

    contents = recorded_call["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "get_sales"
    assert contents[2].parts[0].function_response.name == "get_sales"


def test_gemini_client_wraps_sdk_and_empty_response_errors_in_llm_error() -> None:
    tools = _sample_tools()
    sdk_client = _StubSDKClient([RuntimeError("invalid sdk payload"), _StubResponse(text="   ")])
    gemini = GeminiClient(api_key="dummy-key", model="gemini-3.5-flash-lite", sdk_client=sdk_client)

    with pytest.raises(LLMError, match="invalid sdk payload"):
        gemini.generate([Message(role="user", content="Hi")], "", tools)

    with pytest.raises(LLMError, match="empty response"):
        gemini.generate([Message(role="user", content="Hi")], "", tools)


def test_gemini_client_preserves_thought_signature_across_turns() -> None:
    tools = _sample_tools()
    signature = b"opaque-thought-sig-v1"
    turn1_response = _StubResponse(
        candidates=[
            _StubCandidate(
                content=_StubContent(
                    parts=[
                        _StubPart(text="thinking...", thought_signature=None),
                        _StubPart(
                            function_call=_StubFunctionCall(name="get_low_stock", args={}),
                            thought_signature=signature,
                        ),
                    ]
                )
            )
        ]
    )
    turn2_response = _StubResponse(text="Sugar 1kg is running low.")
    sdk_client = _StubSDKClient([turn1_response, turn2_response])
    gemini = GeminiClient(api_key="dummy-key", model="gemini-3.5-flash-lite", sdk_client=sdk_client)

    first = gemini.generate([Message(role="user", content="What is low?")], "System", tools)
    assert first.tool_call == ToolCall(
        name="get_low_stock",
        arguments={},
        thought_signature=signature,
    )

    messages = [
        Message(role="user", content="What is low?"),
        Message(role="model", tool_call=first.tool_call),
        Message(role="tool", tool_name="get_low_stock", tool_result={"items": []}),
    ]
    second = gemini.generate(messages, "System", tools)

    assert second == LLMResponse(text="Sugar 1kg is running low.")
    model_content = sdk_client.models.calls[1]["contents"][1]
    assert model_content.role == "model"
    assert model_content.parts[0].function_call.name == "get_low_stock"
    assert model_content.parts[0].thought_signature == signature


def test_gemini_client_retries_429_and_503_with_backoff_and_succeeds() -> None:
    tools = _sample_tools()
    rate_limit = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
    )
    unavailable = errors.ServerError(
        503,
        {"error": {"message": "Service unavailable", "status": "UNAVAILABLE"}},
    )
    sdk_client = _StubSDKClient([rate_limit, unavailable, _StubResponse(text="Recovered answer.")])
    sleeps: list[float] = []
    gemini = GeminiClient(
        api_key="dummy-key",
        model="gemini-3.5-flash-lite",
        sdk_client=sdk_client,
        sleep_fn=sleeps.append,
    )

    result = gemini.generate([Message(role="user", content="Hi")], "System", tools)

    assert result == LLMResponse(text="Recovered answer.")
    assert sleeps == [4.0, 8.0]
    assert len(sdk_client.models.calls) == 3


def test_gemini_client_fails_after_exhausting_transient_retries() -> None:
    tools = _sample_tools()
    unavailable = errors.ServerError(
        503,
        {"error": {"message": "Service unavailable", "status": "UNAVAILABLE"}},
    )
    sdk_client = _StubSDKClient([unavailable, unavailable, unavailable, unavailable])
    sleeps: list[float] = []
    gemini = GeminiClient(
        api_key="dummy-key",
        model="gemini-3.5-flash-lite",
        sdk_client=sdk_client,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(LLMError, match="503"):
        gemini.generate([Message(role="user", content="Hi")], "System", tools)

    assert sleeps == [4.0, 8.0, 16.0]
    assert len(sdk_client.models.calls) == 4


def test_gemini_client_does_not_retry_400_client_error() -> None:
    tools = _sample_tools()
    bad_request = errors.ClientError(
        400,
        {"error": {"message": "Invalid argument", "status": "INVALID_ARGUMENT"}},
    )
    sdk_client = _StubSDKClient([bad_request])
    sleeps: list[float] = []
    gemini = GeminiClient(
        api_key="dummy-key",
        model="gemini-3.5-flash-lite",
        sdk_client=sdk_client,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(LLMError, match="400"):
        gemini.generate([Message(role="user", content="Hi")], "System", tools)

    assert sleeps == []
    assert len(sdk_client.models.calls) == 1
