from __future__ import annotations

from typing import Any

import httpx

from storeassist.agent import MAX_TOOL_STEPS, run_agent
from storeassist.llm import FakeLLMClient, LLMError, LLMResponse, ToolCall
from storeassist.store_client import StoreClient
from storeassist.tools import Tool, build_tools


def _make_tools(routes: dict[tuple[str, str], tuple[int, Any]]) -> list[Tool]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        match = routes.get((request.method, request.url.path))
        if match is not None:
            status_code, payload = match
            return httpx.Response(status_code, json=payload)
        return httpx.Response(404, json={"detail": "Not found"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        return build_tools(client)


def test_agent_multi_step_customer_balance_flow() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "owner"})
        if request.url.path == "/customers":
            return httpx.Response(200, json=[{"id": 7, "name": "Sharma Ji", "credit_limit_paise": 500000}])
        if request.url.path == "/customers/7/balance":
            return httpx.Response(200, json={"customer_id": 7, "balance_paise": 145000})
        return httpx.Response(404, json={"detail": "Not found"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        tools = build_tools(client)
        fake_llm = FakeLLMClient(
            [
                LLMResponse(tool_call=ToolCall(name="search_customers", arguments={"query": "Sharma"})),
                LLMResponse(tool_call=ToolCall(name="get_customer_balance", arguments={"customer_id": 7})),
                LLMResponse(text="Sharma Ji owes Rs 1450.0."),
            ]
        )

        result = run_agent(
            "How much does Sharma ji owe me?",
            fake_llm,
            tools,
            system_prompt="Custom prompt",
        )

    assert result.outcome == "answered"
    assert result.steps == 2
    assert result.answer == "Sharma Ji owes Rs 1450.0."
    assert len(result.trace) == 2
    assert result.trace[0].tool_name == "search_customers"
    assert result.trace[0].args == {"query": "Sharma"}
    assert result.trace[0].result["items"][0]["id"] == 7
    assert result.trace[1].tool_name == "get_customer_balance"
    assert result.trace[1].args == {"customer_id": 7}
    assert result.trace[1].result == {"customer_id": 7, "balance_paise": 145000, "balance_rupees": 1450.0}
    assert fake_llm.calls[0].system_prompt == "Custom prompt"


def test_agent_unknown_tool_returns_error_to_model_and_recovers() -> None:
    tools = _make_tools({("GET", "/users/me"): (200, {"id": 1, "role": "staff"})})
    fake_llm = FakeLLMClient(
        [
            LLMResponse(tool_call=ToolCall(name="delete_everything", arguments={})),
            LLMResponse(text="I cannot run that tool."),
        ]
    )

    result = run_agent("Delete all products", fake_llm, tools)

    assert result.outcome == "answered"
    assert result.steps == 1
    assert result.trace[0].tool_name == "delete_everything"
    assert result.trace[0].result == {"error": "Unknown tool: delete_everything"}
    assert fake_llm.calls[1].messages[-1].tool_result == {"error": "Unknown tool: delete_everything"}


def test_agent_recovers_when_tool_returns_structured_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "owner"})
        if request.url.path == "/customers/99/balance":
            return httpx.Response(404, json={"detail": "Customer not found"})
        return httpx.Response(404, json={"detail": "Not found"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        tools = build_tools(client)
        fake_llm = FakeLLMClient(
            [
                LLMResponse(tool_call=ToolCall(name="get_customer_balance", arguments={"customer_id": 99})),
                LLMResponse(text="I could not find customer 99."),
            ]
        )

        result = run_agent("Balance for customer 99?", fake_llm, tools)

    assert result.outcome == "answered"
    assert result.steps == 1
    assert "error" in result.trace[0].result
    assert result.answer == "I could not find customer 99."


def test_agent_enforces_hard_5_step_cap_and_never_runs_step_6() -> None:
    api_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_calls.append(request.url.path)
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "staff"})
        if request.url.path == "/stock/low":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    with StoreClient(
        "http://127.0.0.1:8000",
        "staff",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        tools = build_tools(client)
        fake_llm = FakeLLMClient(
            [LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})) for _ in range(7)]
        )

        result = run_agent("Keep checking low stock", fake_llm, tools)

    assert MAX_TOOL_STEPS == 5
    assert result.outcome == "step_limit"
    assert result.steps == 5
    assert len(result.trace) == 5
    assert api_calls.count("/stock/low") == 5
    assert len(fake_llm.calls) == 6
    assert "5-step" in result.answer


def test_agent_allows_final_answer_after_5_tool_steps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "staff"})
        if request.url.path == "/stock/low":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    with StoreClient(
        "http://127.0.0.1:8000",
        "staff",
        "secret",
        transport=httpx.MockTransport(handler),
    ) as client:
        tools = build_tools(client)
        fake_llm = FakeLLMClient(
            [
                *[LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})) for _ in range(5)],
                LLMResponse(text="Checked 5 times: nothing is low."),
            ]
        )

        result = run_agent("Check low stock", fake_llm, tools)

    assert result.outcome == "answered"
    assert result.steps == 5
    assert result.answer == "Checked 5 times: nothing is low."


def test_agent_handles_llm_error_gracefully() -> None:
    tools = _make_tools({("GET", "/users/me"): (200, {"id": 1, "role": "staff"})})
    fake_llm = FakeLLMClient([LLMError("Rate limit reached")])

    result = run_agent("What is low?", fake_llm, tools)

    assert result.outcome == "error"
    assert result.steps == 0
    assert result.trace == []
    assert "Rate limit reached" in result.answer
