from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Callable, Literal, Sequence

from google import genai
from google.genai import types

from storeassist.tools import Tool

DEFAULT_RETRY_DELAYS: tuple[float, ...] = (4.0, 8.0, 16.0)
_TRANSIENT_STATUS_CODES = {429, 503}
_TRANSIENT_MARKERS = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    thought_signature: bytes | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str | None = None
    tool_call: ToolCall | None = None

    def __post_init__(self) -> None:
        has_text = self.text is not None
        has_tool = self.tool_call is not None
        if has_text == has_tool:
            raise ValueError("LLMResponse must contain exactly one of text or tool_call.")


@dataclass(frozen=True)
class Message:
    role: Literal["user", "model", "tool"]
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None


@dataclass(frozen=True)
class RecordedLLMCall:
    messages: list[Message]
    system_prompt: str
    tool_names: list[str]


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[Tool],
    ) -> LLMResponse:
        raise NotImplementedError


def _to_gemini_declaration(tool: Tool) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=tool.args_model.model_json_schema(),
    )


def _to_gemini_content(message: Message) -> types.Content:
    if message.role == "user":
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=message.content or "")],
        )
    if message.role == "model":
        if message.tool_call is not None:
            return types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name=message.tool_call.name,
                            args=message.tool_call.arguments,
                        ),
                        thought_signature=message.tool_call.thought_signature,
                    )
                ],
            )
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message.content or "")],
        )
    return types.Content(
        role="user",
        parts=[
            types.Part.from_function_response(
                name=message.tool_name or "",
                response=message.tool_result or {},
            )
        ],
    )


def _build_tool_call(call: Any, thought_signature: Any) -> ToolCall:
    call_name = getattr(call, "name", None)
    if not isinstance(call_name, str) or not call_name:
        raise LLMError("Gemini returned a tool call without a function name.")
    raw_args = getattr(call, "args", None)
    arguments = dict(raw_args) if isinstance(raw_args, dict) else {}
    sig = thought_signature if isinstance(thought_signature, bytes) else None
    return ToolCall(name=call_name, arguments=arguments, thought_signature=sig)


def _extract_candidate_parts(response: Any) -> list[Any]:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None)
    return list(parts) if parts else []


def _extract_gemini_response(response: Any) -> LLMResponse:
    for part in _extract_candidate_parts(response):
        call = getattr(part, "function_call", None)
        if call is not None:
            thought_signature = getattr(part, "thought_signature", None)
            return LLMResponse(tool_call=_build_tool_call(call, thought_signature))

    function_calls = getattr(response, "function_calls", None)
    if function_calls:
        first_call = function_calls[0]
        return LLMResponse(tool_call=_build_tool_call(first_call, None))

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return LLMResponse(text=text)

    raise LLMError("Gemini returned an empty response.")


def _is_transient_gemini_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in _TRANSIENT_STATUS_CODES
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in _TRANSIENT_STATUS_CODES
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.upper() in {"RESOURCE_EXHAUSTED", "UNAVAILABLE"}:
        return True
    upper_text = str(exc).upper()
    return any(marker in upper_text for marker in _TRANSIENT_MARKERS)


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        sdk_client: Any = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
    ) -> None:
        self._model = model
        self._client = sdk_client if sdk_client is not None else genai.Client(api_key=api_key)
        self._sleep_fn = sleep_fn
        self._retry_delays = tuple(retry_delays)

    def _call_with_retries(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> Any:
        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                return self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
            except LLMError:
                raise
            except Exception as exc:
                if attempt < len(self._retry_delays) and _is_transient_gemini_error(exc):
                    self._sleep_fn(self._retry_delays[attempt])
                    continue
                raise LLMError(f"Gemini request failed: {exc}") from exc
        raise LLMError("Gemini request failed after retries.")

    def generate(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[Tool],
    ) -> LLMResponse:
        contents = [_to_gemini_content(message) for message in messages]
        gemini_tools = (
            [types.Tool(function_declarations=[_to_gemini_declaration(tool) for tool in tools])]
            if tools
            else None
        )
        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system_prompt if system_prompt else None,
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._call_with_retries(contents, config)
        return _extract_gemini_response(response)


class FakeLLMClient(LLMClient):
    def __init__(self, responses: Sequence[LLMResponse | Exception]) -> None:
        self._responses: list[LLMResponse | Exception] = list(responses)
        self._index = 0
        self.calls: list[RecordedLLMCall] = []

    def generate(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[Tool],
    ) -> LLMResponse:
        self.calls.append(
            RecordedLLMCall(
                messages=list(messages),
                system_prompt=system_prompt,
                tool_names=[tool.name for tool in tools],
            )
        )
        if self._index >= len(self._responses):
            raise LLMError("FakeLLMClient ran out of scripted responses.")
        item = self._responses[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item
