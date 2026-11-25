from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import httpx

from eval.run_eval import evaluate_dataset, format_report, load_questions, save_report, score_result
from storeassist.agent import AgentResult, TraceStep
from storeassist.llm import FakeLLMClient, LLMResponse, ToolCall
from storeassist.store_client import StoreClient
from storeassist.tools import build_tools


def _questions_path() -> Path:
    return Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def _make_fake_client() -> StoreClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "owner"})
        if request.url.path == "/stock/low":
            return httpx.Response(
                200,
                json=[{"product_id": 4, "name": "Pillsbury Atta 1kg", "current_stock": 6, "reorder_level": 20}],
            )
        if request.url.path == "/products":
            return httpx.Response(
                200,
                json=[{"id": 38, "sku": "MAGG-NOOD-70G", "name": "Maggi Noodles", "price_paise": 1400}],
            )
        return httpx.Response(404)

    return StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(handler),
    )


def test_questions_json_schema_and_minimum_category_counts() -> None:
    questions = load_questions(_questions_path())

    assert len(questions) >= 40
    ids = [question.id for question in questions]
    assert len(ids) == len(set(ids))

    counts = Counter(question.category for question in questions)
    assert counts["simple_english"] >= 10
    assert counts["hinglish"] >= 10
    assert counts["ambiguous"] >= 6
    assert counts["out_of_scope"] >= 6
    assert counts["multi_step"] >= 6

    for question in questions:
        assert question.text.strip()
        assert isinstance(question.expected_args, dict)
        assert question.expected_outcome in {"answered", "clarification", "refused"}


def test_evaluate_dataset_scores_agent_and_baseline_formats_report_and_saves_json(
    tmp_path: Path,
) -> None:
    sample_questions = load_questions(_questions_path())
    subset = [sample_questions[0], sample_questions[2], sample_questions[27]]

    fake_llm = FakeLLMClient(
        [
            LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})),
            LLMResponse(text="Pillsbury Atta 1kg is low (6 left)."),
            LLMResponse(tool_call=ToolCall(name="search_products", arguments={"query": "Maggi"})),
            LLMResponse(text="Maggi Noodles costs Rs 99.00."),
            LLMResponse(text="I can only help with StoreStock questions, not weather."),
        ]
    )
    delays_recorded: list[float] = []

    with _make_fake_client() as client:
        tools = build_tools(client)
        report = evaluate_dataset(
            questions=subset,
            llm_client=fake_llm,
            tools=tools,
            model_name="gemini-3.5-flash-lite",
            delay_seconds=1.5,
            sleep_fn=delays_recorded.append,
        )

    assert delays_recorded == [1.5, 1.5]
    assert report["model"] == "gemini-3.5-flash-lite"
    assert report["temperature"] == 0.0
    assert report["questions_count"] == 3

    agent_sum = report["summary"]["agent"]
    assert agent_sum["first_tool_correct"] == 3
    assert agent_sum["args_correct"] == 3
    assert agent_sum["outcome_correct"] == 2
    assert agent_sum["grounded_correct"] == 2

    base_sum = report["summary"]["baseline"]
    assert base_sum["outcome_correct"] == 3
    assert base_sum["grounded_correct"] == 3

    formatted = format_report(report)
    assert "First-tool accuracy" in formatted
    assert "3/3 (100.0%)" in formatted
    assert "2/3 (66.7%)" in formatted
    assert "Category Breakdown (Outcome Accuracy):" in formatted
    assert "Failures:" in formatted
    assert sample_questions[2].id in formatted

    saved_path = save_report(report, tmp_path)
    assert saved_path.exists()
    loaded: dict[str, Any] = json.loads(saved_path.read_text(encoding="utf-8"))
    assert loaded["questions_count"] == 3
    assert len(loaded["results"]) == 3


def test_score_result_uses_subset_matching_for_expected_args() -> None:
    questions = load_questions(_questions_path())
    daily_question = next(question for question in questions if question.id == "se_02")
    result = AgentResult(
        answer="Total sales today are Rs 4079.00.",
        trace=[
            TraceStep(
                tool_name="get_daily_sales",
                args={"date": "2026-11-25"},
                result={"date": "2026-11-25", "total_sales_rupees": "4079.00"},
            )
        ],
        steps=1,
        outcome="answered",
    )

    scores = score_result(daily_question, result)

    assert scores.first_tool_correct is True
    assert scores.args_correct is True
    assert scores.outcome_correct is True
    assert scores.grounded is True
    assert scores.failure_reasons == []
