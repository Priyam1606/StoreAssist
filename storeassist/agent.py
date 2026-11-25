from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from storeassist.guardrails import (
    TraceRecord,
    check_tool_ambiguity,
    is_out_of_scope_request,
    is_refusal_response,
    is_store_data_question,
    is_write_request,
    verify_number_grounding,
)
from storeassist.llm import LLMClient, LLMError, Message, ToolCall
from storeassist.prompts import build_system_prompt
from storeassist.tools import Tool

MAX_TOOL_STEPS = 5

Outcome = Literal["answered", "clarification", "refused", "step_limit", "error", "ungrounded"]


@dataclass(frozen=True)
class TraceStep:
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    trace: list[TraceStep]
    steps: int
    outcome: Outcome


def _execute_tool(tool_map: dict[str, Tool], tool_call: ToolCall) -> dict[str, Any]:
    tool = tool_map.get(tool_call.name)
    if tool is None:
        return {"error": f"Unknown tool: {tool_call.name}"}
    return tool.execute(tool_call.arguments)


def _finalize_answer(
    question: str,
    answer: str,
    trace: list[TraceStep],
) -> AgentResult:
    steps = len(trace)
    if is_write_request(question):
        return AgentResult(
            answer="I am a read-only assistant and cannot create, update, or delete store records.",
            trace=trace,
            steps=steps,
            outcome="refused",
        )

    if steps == 0:
        if is_out_of_scope_request(question):
            return AgentResult(
                answer="I can only help with StoreStock store questions and cannot answer out-of-scope requests.",
                trace=trace,
                steps=0,
                outcome="refused",
            )
        if is_store_data_question(question):
            return AgentResult(
                answer="Unable to answer store data questions without verifying against StoreStock tools.",
                trace=trace,
                steps=0,
                outcome="ungrounded",
            )
        if is_refusal_response(answer):
            return AgentResult(
                answer=answer,
                trace=trace,
                steps=0,
                outcome="refused",
            )

    records = [TraceRecord(tool_name=step.tool_name, args=step.args, result=step.result) for step in trace]
    is_grounded, _ = verify_number_grounding(answer, question, records)
    if not is_grounded:
        return AgentResult(
            answer="Could not verify all numbers in the response against StoreStock tool outputs.",
            trace=trace,
            steps=steps,
            outcome="ungrounded",
        )

    return AgentResult(
        answer=answer,
        trace=trace,
        steps=steps,
        outcome="answered",
    )


def run_agent(
    question: str,
    llm_client: LLMClient,
    tools: list[Tool],
    system_prompt: str | None = None,
    max_steps: int = MAX_TOOL_STEPS,
) -> AgentResult:
    resolved_prompt = build_system_prompt() if system_prompt is None else system_prompt
    tool_map = {tool.name: tool for tool in tools}
    messages: list[Message] = [Message(role="user", content=question)]
    trace: list[TraceStep] = []

    for _turn in range(max_steps + 1):
        try:
            response = llm_client.generate(messages, resolved_prompt, tools)
        except LLMError as exc:
            return AgentResult(
                answer=f"Unable to answer because the language model failed: {exc}",
                trace=trace,
                steps=len(trace),
                outcome="error",
            )

        if response.text is not None:
            return _finalize_answer(question, response.text, trace)

        tool_call = response.tool_call
        if tool_call is None:
            return AgentResult(
                answer="Unable to answer because the model returned an empty response.",
                trace=trace,
                steps=len(trace),
                outcome="error",
            )

        if len(trace) >= max_steps:
            return AgentResult(
                answer=f"Could not complete the answer within the {max_steps}-step tool limit.",
                trace=trace,
                steps=len(trace),
                outcome="step_limit",
            )

        result = _execute_tool(tool_map, tool_call)
        trace.append(TraceStep(tool_name=tool_call.name, args=tool_call.arguments, result=result))
        clarification = check_tool_ambiguity(question, tool_call.name, tool_call.arguments, result)
        if clarification is not None:
            return AgentResult(
                answer=clarification,
                trace=trace,
                steps=len(trace),
                outcome="clarification",
            )

        messages.append(Message(role="model", tool_call=tool_call))
        messages.append(Message(role="tool", tool_name=tool_call.name, tool_result=result))

    return AgentResult(
        answer=f"Could not complete the answer within the {max_steps}-step tool limit.",
        trace=trace,
        steps=len(trace),
        outcome="step_limit",
    )
