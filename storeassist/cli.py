from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, TextIO

from storeassist.agent import AgentResult, run_agent
from storeassist.config import Config, ConfigError, load_config
from storeassist.llm import GeminiClient, LLMClient
from storeassist.store_client import StoreClient, StoreClientError
from storeassist.tools import Tool, build_tools


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="storeassist")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--trace", action="store_true")

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--trace", action="store_true")

    return parser


def _format_trace(result: AgentResult) -> str:
    lines: list[str] = [f"[outcome={result.outcome}, steps={result.steps}]"]
    for index, step in enumerate(result.trace, start=1):
        args_json = json.dumps(step.args, ensure_ascii=False)
        result_json = json.dumps(step.result, ensure_ascii=False)
        lines.append(f"  step {index}: {step.tool_name}({args_json}) -> {result_json}")
    return "\n".join(lines)


def _print_result(result: AgentResult, show_trace: bool, stdout: TextIO) -> None:
    if show_trace:
        print(_format_trace(result), file=stdout)
    print(result.answer, file=stdout)


def _run_chat_loop(
    llm_client: LLMClient,
    tools: list[Tool],
    show_trace: bool,
    stdin: TextIO,
    stdout: TextIO,
) -> int:
    print("StoreAssist chat ready. Type 'exit' or 'quit' to leave.", file=stdout)
    while True:
        print("storeassist> ", end="", flush=True, file=stdout)
        try:
            raw_line = stdin.readline()
        except KeyboardInterrupt:
            print("", file=stdout)
            return 0
        if raw_line == "":
            print("", file=stdout)
            return 0
        question = raw_line.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0
        result = run_agent(question, llm_client, tools)
        _print_result(result, show_trace, stdout)


def _default_store_client_factory(config: Config) -> StoreClient:
    return StoreClient(
        base_url=config.storestock_url,
        username=config.storestock_username,
        password=config.storestock_password,
    )


def _default_llm_factory(config: Config) -> LLMClient:
    return GeminiClient(api_key=config.llm_api_key, model=config.llm_model)


def main(
    argv: list[str] | None = None,
    config_loader: Callable[[], Config] = load_config,
    store_client_factory: Callable[[Config], StoreClient] = _default_store_client_factory,
    llm_factory: Callable[[Config], LLMClient] = _default_llm_factory,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    err_stream = sys.stderr if stderr is None else stderr

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_loader()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=err_stream)
        return 1

    store_client: StoreClient | None = None
    try:
        store_client = store_client_factory(config)
        tools = build_tools(store_client)
        llm_client = llm_factory(config)
        if args.command == "ask":
            result = run_agent(args.question, llm_client, tools)
            _print_result(result, args.trace, out_stream)
            return 0
        return _run_chat_loop(llm_client, tools, args.trace, in_stream, out_stream)
    except StoreClientError as exc:
        print(f"StoreStock error: {exc}", file=err_stream)
        return 1
    finally:
        if store_client is not None:
            store_client.close()
