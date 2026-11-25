from __future__ import annotations

from storeassist.guardrails import (
    TraceRecord,
    check_tool_ambiguity,
    is_out_of_scope_request,
    is_refusal_response,
    is_store_data_question,
    is_write_request,
    verify_number_grounding,
)
from storeassist.prompts import SYSTEM_PROMPT, build_system_prompt


def test_system_prompt_contains_core_safety_rules() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "read-only" in lowered
    assert "untrusted data" in lowered
    assert "calculate" in lowered
    assert "*_rupees" in lowered
    assert "hinglish" in lowered
    assert "clarif" in lowered

    dated_prompt = build_system_prompt("2026-11-25")
    assert "Asia/Kolkata" in dated_prompt
    assert "2026-11-25" in dated_prompt
    assert "yesterday" in dated_prompt.lower()
    assert "kal" in dated_prompt.lower()


def test_grounding_allows_rupees_commas_decimals_question_dates_and_list_numbers() -> None:
    trace = [
        TraceRecord(
            tool_name="get_daily_sales",
            args={"date": "2026-11-25"},
            result={"date": "2026-11-25", "total_sales_paise": 125050, "total_sales_rupees": "1250.50"},
        ),
        TraceRecord(
            tool_name="search_products",
            args={"query": "Salt"},
            result={
                "items": [{"id": 3, "name": "Tata Salt", "price_paise": 2800, "price_rupees": "28.00"}],
                "truncated": False,
                "total_count": 1,
                "returned_count": 1,
            },
        ),
    ]
    answer = (
        "1. Sales on 2026-11-25 were Rs 1,250.50.\n"
        "2. Tata Salt costs Rs 28.00."
    )

    is_ok, ungrounded = verify_number_grounding(answer, "Sales on 2026-11-25 and Tata Salt price?", trace)

    assert is_ok is True
    assert ungrounded == []


def test_grounding_blocks_ungrounded_numbers_and_bool_zero_one() -> None:
    trace = [
        TraceRecord(
            tool_name="search_products",
            args={"query": "Salt"},
            result={
                "items": [{"id": 3, "name": "Tata Salt", "price_paise": 2800, "price_rupees": "28.00"}],
                "truncated": False,
                "total_count": 1,
                "returned_count": 1,
            },
        )
    ]

    is_ok, ungrounded = verify_number_grounding("Tata Salt costs Rs 0.00 today.", "Tata Salt price?", trace)
    assert is_ok is False
    assert ungrounded == ["0.00"]

    is_ok_invented, invented = verify_number_grounding("Tata Salt costs Rs 45.", "Tata Salt price?", trace)
    assert is_ok_invented is False
    assert invented == ["45"]


def test_grounding_blocks_ungrounded_iso_date_not_in_question_or_trace() -> None:
    trace = [TraceRecord(tool_name="get_low_stock", args={}, result={"items": [], "total_count": 0})]
    is_ok, ungrounded = verify_number_grounding(
        "Checked on 2026-12-31: 0 items low.",
        "What is low?",
        trace,
    )
    assert is_ok is False
    assert ungrounded == ["2026", "12", "31"]


def test_ambiguity_check_for_customers_and_products() -> None:
    multiple_customers = {
        "items": [
            {"id": 1, "name": "Ravi Sharma", "phone": "9810011111"},
            {"id": 2, "name": "Amit Sharma", "phone": "9810022222"},
        ]
    }
    clarify_cust = check_tool_ambiguity(
        "How much does Sharma owe me?",
        "search_customers",
        {"query": "Sharma"},
        multiple_customers,
    )
    assert clarify_cust is not None
    assert "Ravi Sharma" in clarify_cust
    assert "Amit Sharma" in clarify_cust

    exact_customer = {
        "items": [
            {"id": 1, "name": "Sharma"},
            {"id": 2, "name": "Sharma Ji"},
        ]
    }
    assert (
        check_tool_ambiguity(
            "How much does Sharma owe me?",
            "search_customers",
            {"query": "Sharma"},
            exact_customer,
        )
        is None
    )

    multiple_products = {
        "items": [
            {"id": 10, "sku": "TATA-SALT", "name": "Tata Salt 1kg"},
            {"id": 11, "sku": "TATA-TEA", "name": "Tata Tea Gold"},
        ]
    }
    clarify_prod = check_tool_ambiguity(
        "What is the price of Tata?",
        "search_products",
        {"query": "Tata"},
        multiple_products,
    )
    assert clarify_prod is not None
    assert "Tata Salt 1kg" in clarify_prod
    assert "Tata Tea Gold" in clarify_prod

    no_clarify_list = check_tool_ambiguity(
        "List all Tata products",
        "search_products",
        {"query": "Tata"},
        multiple_products,
    )
    assert no_clarify_list is None


def test_request_classifiers() -> None:
    assert is_out_of_scope_request("what will the weather be tomorrow") is True
    assert is_write_request("delete this product") is True
    assert is_write_request("change the price of Maggi") is True
    assert is_write_request("what is the price of Maggi") is False
    assert is_store_data_question("kitna maal bika aaj?") is True
    assert is_store_data_question("what is running low?") is True
    assert is_store_data_question("hello") is False
    assert is_refusal_response("I can only help with StoreStock store questions.") is True
    assert is_refusal_response("Sorry, I am unable to answer that request.") is True
    assert is_refusal_response("Hello! How can I help with your store today?") is False
