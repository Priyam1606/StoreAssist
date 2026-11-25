from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from storeassist.llm import LLMClient, LLMError, Message, ToolCall
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


def run_agent(
    question: str,
    llm_client: LLMClient,
    tools: list[Tool],
    system_prompt: str = "",
    max_steps: int = MAX_TOOL_STEPS,
) -> AgentResult:
    tool_map = {tool.name: tool for tool in tools}
    messages: list[Message] = [Message(role="user", content=question)]
    trace: list[TraceStep] = []

    for _turn in range(max_steps + 1):
        try:
            response = llm_client.generate(messages, system_prompt, tools)
        except LLMError as exc:
            return AgentResult(
                answer=f"Unable to answer because the language model failed: {exc}",
                trace=trace,
                steps=len(trace),
                outcome="error",
            )

        if response.text is not None:
            return AgentResult(
                answer=response.text,
                trace=trace,
                steps=len(trace),
                outcome="answered",
            )

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
        messages.append(Message(role="model", tool_call=tool_call))
        messages.append(Message(role="tool", tool_name=tool_call.name, tool_result=result))

    return AgentResult(
        answer=f"Could not complete the answer within the {max_steps}-step tool limit.",
        trace=trace,
        steps=len(trace),
        outcome="step_limit",
    )
