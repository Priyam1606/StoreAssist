from __future__ import annotations

from typing import Any

import httpx

from storeassist.baseline import run_baseline
from storeassist.store_client import StoreClient
from storeassist.tools import Tool, build_tools


def _make_client(
    role: str = "owner",
    extra_routes: dict[tuple[str, str], tuple[int, Any]] | None = None,
) -> StoreClient:
    routes: dict[tuple[str, str], tuple[int, Any]] = {
        ("GET", "/users/me"): (200, {"id": 1, "role": role}),
        ("GET", "/stock/low"): (
            200,
            [{"product_id": 4, "name": "Pillsbury Atta 1kg", "current_stock": 6, "reorder_level": 20}],
        ),
        ("GET", "/reports/daily-summary"): (
            200,
            {"date": "2026-11-25", "total_sales_paise": 407900, "sales_count": 5},
        ),
        ("GET", "/purchase-orders/suggest"): (
            200,
            [{"product_id": 4, "name": "Pillsbury Atta 1kg", "suggested_qty": 34}],
        ),
        ("GET", "/stock"): (
            200,
            [{"product_id": 38, "name": "Maggi 2-Minute Masala Noodles 70g", "current_stock": 18}],
        ),
    }
    if extra_routes:
        routes.update(extra_routes)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/products":
            query = request.url.params.get("q", "").lower()
            if "amul" in query:
                return httpx.Response(
                    200,
                    json=[
                        {"id": 55, "sku": "DAIR-AMUL-500G", "name": "Amul Butter 500g", "price_paise": 27500},
                        {"id": 56, "sku": "DAIR-AMUL-100G", "name": "Amul Butter 100g", "price_paise": 5800},
                    ],
                )
            return httpx.Response(
                200,
                json=[{"id": 38, "sku": "MAGG-NOOD-70G", "name": "Maggi 2-Minute Masala Noodles 70g", "price_paise": 1400}],
            )
        if request.url.path == "/customers":
            query = request.url.params.get("q", "").lower()
            if query == "ra":
                return httpx.Response(
                    200,
                    json=[
                        {"id": 1, "name": "Ramesh Kumar", "phone": "9811100001"},
                        {"id": 6, "name": "Rajesh Gupta", "phone": "9811100006"},
                    ],
                )
            return httpx.Response(
                200,
                json=[{"id": 5, "name": "Priya Nair", "phone": "9811100005"}],
            )
        if request.url.path == "/customers/5/balance":
            return httpx.Response(200, json={"customer_id": 5, "balance_paise": 101500})
        if request.url.path == "/customers/5/ledger":
            return httpx.Response(200, json=[{"id": 1, "amount_paise": 50000}])

        match = routes.get((request.method, request.url.path))
        if match is not None:
            status_code, payload = match
            return httpx.Response(status_code, json=payload)
        return httpx.Response(404)

    return StoreClient(
        "http://127.0.0.1:8000",
        role,
        "secret",
        transport=httpx.MockTransport(handler),
    )


def test_baseline_routes_low_stock_daily_sales_and_reorder() -> None:
    with _make_client("owner") as client:
        tools = build_tools(client)
        low_res = run_baseline("Kaun sa maal kam hai dukaan mein?", tools)
        sales_res = run_baseline("Aaj kitna maal bika?", tools)
        reorder_res = run_baseline("Which items should I reorder?", tools)

    assert low_res.outcome == "answered"
    assert low_res.trace[0].tool_name == "get_low_stock"
    assert "Pillsbury Atta 1kg" in low_res.answer

    assert sales_res.outcome == "answered"
    assert sales_res.trace[0].tool_name == "get_daily_sales"
    assert "4079.00" in sales_res.answer

    assert reorder_res.outcome == "answered"
    assert reorder_res.trace[0].tool_name == "suggest_reorder"
    assert "34" in reorder_res.answer


def test_baseline_customer_balance_ledger_and_ambiguity() -> None:
    with _make_client("owner") as client:
        tools = build_tools(client)
        bal_res = run_baseline("How much does Priya Nair owe me?", tools)
        ledger_res = run_baseline("Check Priya Nair's recent credit ledger entries.", tools)
        amb_res = run_baseline("How much does customer Ra owe me?", tools)

    assert bal_res.outcome == "answered"
    assert [step.tool_name for step in bal_res.trace] == ["search_customers", "get_customer_balance"]
    assert "1015.00" in bal_res.answer

    assert ledger_res.outcome == "answered"
    assert [step.tool_name for step in ledger_res.trace] == ["search_customers", "get_customer_ledger"]

    assert amb_res.outcome == "clarification"
    assert amb_res.steps == 1
    assert "Ramesh Kumar" in amb_res.answer
    assert "Rajesh Gupta" in amb_res.answer


def test_baseline_product_lookup_stock_level_and_ambiguity() -> None:
    with _make_client("owner") as client:
        tools = build_tools(client)
        price_res = run_baseline("Maggi kitne ka hai?", tools)
        stock_res = run_baseline("How much stock of Maggi do we currently have?", tools)
        amb_prod = run_baseline("What is the price of Amul Butter?", tools)

    assert price_res.outcome == "answered"
    assert price_res.trace[0].tool_name == "search_products"
    assert "14.00" in price_res.answer

    assert stock_res.outcome == "answered"
    assert [step.tool_name for step in stock_res.trace] == ["search_products", "get_stock_level"]
    assert "18" in stock_res.answer

    assert amb_prod.outcome == "clarification"
    assert "Amul Butter 500g" in amb_prod.answer
    assert "Amul Butter 100g" in amb_prod.answer


def test_baseline_refuses_writes_and_weather_and_staff_owner_tools() -> None:
    with _make_client("owner") as owner_client:
        owner_tools = build_tools(owner_client)
        for question in (
            "What will the weather be tomorrow?",
            "Delete the Maggi product from the store.",
            "Change the price of Dettol soap to 50 rupees.",
        ):
            res = run_baseline(question, owner_tools)
            assert res.outcome == "refused"
            assert res.steps == 0

    with _make_client("staff") as staff_client:
        staff_tools = build_tools(staff_client)
        staff_sales = run_baseline("Aaj kitna maal bika?", staff_tools)
        assert staff_sales.outcome == "refused"
