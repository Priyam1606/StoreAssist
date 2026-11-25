from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, TextIO

from storeassist.agent import AgentResult, run_agent
from storeassist.baseline import run_baseline
from storeassist.config import ConfigError, load_config
from storeassist.guardrails import TraceRecord, verify_number_grounding
from storeassist.llm import GeminiClient, LLMClient
from storeassist.prompts import today_ist_iso
from storeassist.store_client import StoreClient, StoreClientError
from storeassist.tools import Tool, build_tools


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    text: str
    expected_first_tool: str | None
    expected_args: dict[str, Any]
    expected_outcome: str


@dataclass(frozen=True)
class RunScores:
    first_tool_correct: bool
    args_correct: bool
    outcome_correct: bool
    grounded: bool
    failure_reasons: list[str]


def _default_questions_path() -> Path:
    return Path(__file__).resolve().parent / "questions.json"


def _default_results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalQuestion(
            id=str(item["id"]),
            category=str(item["category"]),
            text=str(item["text"]),
            expected_first_tool=item.get("expected_first_tool"),
            expected_args=dict(item.get("expected_args") or {}),
            expected_outcome=str(item["expected_outcome"]),
        )
        for item in raw
    ]


def _normalize_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _args_match(expected_args: dict[str, Any], actual_args: dict[str, Any]) -> bool:
    for key, expected_value in expected_args.items():
        if key not in actual_args:
            return False
        if _normalize_arg_value(actual_args[key]) != _normalize_arg_value(expected_value):
            return False
    return True


def score_result(question: EvalQuestion, result: AgentResult) -> RunScores:
    actual_first_tool = result.trace[0].tool_name if result.trace else None
    actual_args = result.trace[0].args if result.trace else {}

    first_tool_correct = actual_first_tool == question.expected_first_tool
    if question.expected_first_tool is None:
        args_correct = actual_first_tool is None
    else:
        args_correct = first_tool_correct and _args_match(question.expected_args, actual_args)

    outcome_correct = result.outcome == question.expected_outcome

    records = [
        TraceRecord(tool_name=step.tool_name, args=step.args, result=step.result)
        for step in result.trace
    ]
    if result.outcome == "ungrounded":
        grounded = False
    elif result.outcome == "answered":
        grounded, _ = verify_number_grounding(result.answer, question.text, records)
    else:
        grounded = True

    reasons: list[str] = []
    if not first_tool_correct:
        reasons.append(
            f"first_tool expected {question.expected_first_tool!r}, got {actual_first_tool!r}"
        )
    if not args_correct:
        reasons.append(
            f"args expected {question.expected_args!r}, got {actual_args!r}"
        )
    if not outcome_correct:
        reasons.append(
            f"outcome expected {question.expected_outcome!r}, got {result.outcome!r}"
        )
    if not grounded:
        reasons.append("ungrounded answer")

    return RunScores(
        first_tool_correct=first_tool_correct,
        args_correct=args_correct,
        outcome_correct=outcome_correct,
        grounded=grounded,
        failure_reasons=reasons,
    )


def _pct(passed: int, total: int) -> str:
    if total == 0:
        return "0/0 (0.0%)"
    percent = (passed / total) * 100.0
    return f"{passed}/{total} ({percent:.1f}%)"


def _summarize_runs(entries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(entries)
    first_tool_ok = sum(1 for item in entries if item[key]["scores"]["first_tool_correct"])
    args_ok = sum(1 for item in entries if item[key]["scores"]["args_correct"])
    outcome_ok = sum(1 for item in entries if item[key]["scores"]["outcome_correct"])
    grounded_ok = sum(1 for item in entries if item[key]["scores"]["grounded"])
    total_steps = sum(int(item[key]["steps"]) for item in entries)
    avg_steps = round(total_steps / total, 2) if total else 0.0

    categories: dict[str, dict[str, int]] = {}
    for item in entries:
        cat = item["category"]
        bucket = categories.setdefault(cat, {"passed": 0, "total": 0})
        bucket["total"] += 1
        if item[key]["scores"]["outcome_correct"]:
            bucket["passed"] += 1

    return {
        "total": total,
        "first_tool_correct": first_tool_ok,
        "args_correct": args_ok,
        "outcome_correct": outcome_ok,
        "grounded_correct": grounded_ok,
        "average_steps": avg_steps,
        "categories": categories,
    }


def evaluate_dataset(
    questions: list[EvalQuestion],
    llm_client: LLMClient,
    tools: list[Tool],
    model_name: str,
    delay_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        if index > 0 and delay_seconds > 0:
            sleep_fn(delay_seconds)
        agent_res = run_agent(question.text, llm_client, tools)
        agent_scores = score_result(question, agent_res)

        baseline_res = run_baseline(question.text, tools)
        baseline_scores = score_result(question, baseline_res)

        entries.append(
            {
                "id": question.id,
                "category": question.category,
                "text": question.text,
                "expected": {
                    "first_tool": question.expected_first_tool,
                    "args": question.expected_args,
                    "outcome": question.expected_outcome,
                },
                "agent": {
                    "answer": agent_res.answer,
                    "trace": [asdict(step) for step in agent_res.trace],
                    "steps": agent_res.steps,
                    "outcome": agent_res.outcome,
                    "scores": asdict(agent_scores),
                },
                "baseline": {
                    "answer": baseline_res.answer,
                    "trace": [asdict(step) for step in baseline_res.trace],
                    "steps": baseline_res.steps,
                    "outcome": baseline_res.outcome,
                    "scores": asdict(baseline_scores),
                },
            }
        )

    return {
        "model": model_name,
        "date": today_ist_iso(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "temperature": 0.0,
        "questions_count": len(questions),
        "summary": {
            "agent": _summarize_runs(entries, "agent"),
            "baseline": _summarize_runs(entries, "baseline"),
        },
        "results": entries,
    }


def format_report(report: dict[str, Any]) -> str:
    agent_sum = report["summary"]["agent"]
    base_sum = report["summary"]["baseline"]
    total = int(report["questions_count"])

    lines: list[str] = [
        f"Model: {report['model']} | Date: {report['date']} | Temperature: {report['temperature']}",
        "",
        f"{'Metric':<26} {'Agent':<20} {'Baseline':<20}",
        "-" * 68,
        f"{'First-tool accuracy':<26} {_pct(agent_sum['first_tool_correct'], total):<20} {_pct(base_sum['first_tool_correct'], total):<20}",
        f"{'Argument accuracy':<26} {_pct(agent_sum['args_correct'], total):<20} {_pct(base_sum['args_correct'], total):<20}",
        f"{'Outcome accuracy':<26} {_pct(agent_sum['outcome_correct'], total):<20} {_pct(base_sum['outcome_correct'], total):<20}",
        f"{'Grounded-answer rate':<26} {_pct(agent_sum['grounded_correct'], total):<20} {_pct(base_sum['grounded_correct'], total):<20}",
        f"{'Average steps':<26} {agent_sum['average_steps']:<20} {base_sum['average_steps']:<20}",
        "",
        "Category Breakdown (Outcome Accuracy):",
    ]

    for category, agent_cat in agent_sum["categories"].items():
        base_cat = base_sum["categories"].get(category, {"passed": 0, "total": 0})
        lines.append(
            f"  {category:<24} Agent: {_pct(agent_cat['passed'], agent_cat['total']):<16} Baseline: {_pct(base_cat['passed'], base_cat['total'])}"
        )

    failures: list[str] = []
    for item in report["results"]:
        for system_name in ("agent", "baseline"):
            reasons = item[system_name]["scores"]["failure_reasons"]
            if reasons:
                failures.append(
                    f"  [{system_name}] {item['id']} ({item['category']}): {item['text']} -> {'; '.join(reasons)}"
                )

    lines.append("")
    if failures:
        lines.append("Failures:")
        lines.extend(failures)
    else:
        lines.append("Failures: None")

    return "\n".join(lines)


def save_report(report: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_eval")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--questions", type=Path, default=_default_questions_path())
    parser.add_argument("--results-dir", type=Path, default=_default_results_dir())
    return parser


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out_stream = sys.stdout if stdout is None else stdout
    err_stream = sys.stderr if stderr is None else stderr
    args = _build_parser().parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=err_stream)
        return 1

    questions = load_questions(args.questions)
    store_client: StoreClient | None = None
    try:
        store_client = StoreClient(
            base_url=config.storestock_url,
            username=config.storestock_username,
            password=config.storestock_password,
        )
        tools = build_tools(store_client)
        llm_client = GeminiClient(api_key=config.llm_api_key, model=config.llm_model)
        report = evaluate_dataset(
            questions=questions,
            llm_client=llm_client,
            tools=tools,
            model_name=config.llm_model,
            delay_seconds=args.delay,
        )
        saved_path = save_report(report, args.results_dir)
        print(format_report(report), file=out_stream)
        print(f"\nSaved evaluation report to {saved_path}", file=out_stream)
        return 0
    except StoreClientError as exc:
        print(f"StoreStock error: {exc}", file=err_stream)
        return 1
    finally:
        if store_client is not None:
            store_client.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
