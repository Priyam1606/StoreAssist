from __future__ import annotations

import json
from typing import Any

import httpx

from storeassist.store_client import StoreClient
from storeassist.tools import Tool, build_tools


def _make_client(
    routes: dict[tuple[str, str], tuple[int, Any]],
    recorded: list[tuple[str, str, dict[str, str]]] | None = None,
) -> StoreClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if recorded is not None:
            recorded.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok-test"})
        match = routes.get((request.method, request.url.path))
        if match is not None:
            status_code, payload = match
            return httpx.Response(status_code, json=payload)
        return httpx.Response(404, json={"detail": "Not found"})

    return StoreClient(
        "http://127.0.0.1:8000",
        "user",
        "pass",
        transport=httpx.MockTransport(handler),
    )


def _tool_map(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def test_role_based_registration_owner_vs_staff() -> None:
    recorded_owner: list[tuple[str, str, dict[str, str]]] = []
    with _make_client(
        {("GET", "/users/me"): (200, {"id": 1, "username": "owner", "role": "owner"})},
        recorded=recorded_owner,
    ) as owner_client:
        owner_tools = _tool_map(build_tools(owner_client))

    assert set(owner_tools) == {
        "search_products",
        "get_low_stock",
        "get_stock_level",
        "search_customers",
        "get_customer_balance",
        "get_customer_ledger",
        "get_sales",
        "get_daily_sales",
        "suggest_reorder",
        "calculate",
    }
    assert recorded_owner.count(("GET", "/users/me", {})) == 1

    recorded_staff: list[tuple[str, str, dict[str, str]]] = []
    with _make_client(
        {("GET", "/users/me"): (200, {"id": 2, "username": "staff1", "role": "staff"})},
        recorded=recorded_staff,
    ) as staff_client:
        staff_tools = _tool_map(build_tools(staff_client))

    assert set(staff_tools) == {
        "search_products",
        "get_low_stock",
        "get_stock_level",
        "search_customers",
        "get_customer_balance",
        "get_customer_ledger",
        "get_sales",
        "calculate",
    }
    assert "get_daily_sales" not in staff_tools
    assert "suggest_reorder" not in staff_tools
    assert recorded_staff.count(("GET", "/users/me", {})) == 1


def test_tools_call_exact_endpoints_and_params() -> None:
    recorded: list[tuple[str, str, dict[str, str]]] = []
    routes: dict[tuple[str, str], tuple[int, Any]] = {
        ("GET", "/users/me"): (200, {"id": 1, "role": "owner"}),
        ("GET", "/products"): (200, [{"id": 5, "name": "Tata Salt", "price_paise": 2800}]),
        ("GET", "/stock/low"): (200, [{"product_id": 5, "current_stock": 2, "reorder_level": 10}]),
        ("GET", "/stock"): (
            200,
            [
                {"product_id": 4, "name": "Milk", "current_stock": 12, "reorder_level": 5},
                {"product_id": 5, "name": "Tata Salt", "current_stock": 2, "reorder_level": 10},
            ],
        ),
        ("GET", "/customers"): (200, [{"id": 9, "name": "Sharma Ji", "credit_limit_paise": 500000}]),
        ("GET", "/customers/9/balance"): (200, {"customer_id": 9, "balance_paise": 125000}),
        ("GET", "/customers/9/ledger"): (200, [{"id": 1, "amount_paise": 125000}]),
        ("GET", "/sales"): (200, [{"id": 22, "total_paise": 5600}]),
        ("GET", "/reports/daily-summary"): (200, {"date": "2026-11-25", "total_sales_paise": 90000}),
        ("GET", "/purchase-orders/suggest"): (200, [{"product_id": 5, "suggested_qty": 20, "unit_cost_paise": 2000}]),
    }

    with _make_client(routes, recorded=recorded) as client:
        tools = _tool_map(build_tools(client))

        products_res = tools["search_products"].execute({"query": "salt"})
        low_res = tools["get_low_stock"].execute({})
        stock_res = tools["get_stock_level"].execute({"product_id": 5})
        customers_res = tools["search_customers"].execute({"query": "Sharma"})
        balance_res = tools["get_customer_balance"].execute({"customer_id": 9})
        ledger_res = tools["get_customer_ledger"].execute({"customer_id": 9})
        sales_res = tools["get_sales"].execute({"date": "2026-11-25", "customer_id": 9})
        daily_res = tools["get_daily_sales"].execute({"date": "2026-11-25"})
        reorder_res = tools["suggest_reorder"].execute({})

    assert products_res["items"][0]["price_rupees"] == "28.00"
    assert low_res["returned_count"] == 1
    assert stock_res == {"product_id": 5, "name": "Tata Salt", "current_stock": 2, "reorder_level": 10}
    assert customers_res["items"][0]["credit_limit_rupees"] == "5000.00"
    assert balance_res == {"customer_id": 9, "balance_paise": 125000, "balance_rupees": "1250.00"}
    assert ledger_res["items"][0]["amount_rupees"] == "1250.00"
    assert sales_res["items"][0]["total_rupees"] == "56.00"
    assert daily_res == {
        "date": "2026-11-25",
        "total_sales_paise": 90000,
        "total_sales_rupees": "900.00",
    }
    assert reorder_res["items"][0]["unit_cost_rupees"] == "20.00"

    assert ("GET", "/customers/9/ledger", {"limit": "20"}) in recorded
    assert ("GET", "/sales", {"date": "2026-11-25", "customer_id": "9"}) in recorded
    assert ("GET", "/reports/daily-summary", {"date": "2026-11-25"}) in recorded
    assert ("GET", "/purchase-orders/suggest", {}) in recorded


def test_list_truncation_caps_at_20_with_metadata() -> None:
    many_products = [{"id": idx, "name": f"Item {idx}"} for idx in range(1, 26)]
    with _make_client(
        {
            ("GET", "/users/me"): (200, {"id": 1, "role": "staff"}),
            ("GET", "/products"): (200, many_products),
        }
    ) as client:
        tools = _tool_map(build_tools(client))
        result = tools["search_products"].execute({"query": "Item"})

    assert result["truncated"] is True
    assert result["total_count"] == 25
    assert result["returned_count"] == 20
    assert len(result["items"]) == 20
    assert result["items"][0]["id"] == 1
    assert result["items"][-1]["id"] == 20


def test_calculate_valid_and_invalid_inputs() -> None:
    with _make_client({("GET", "/users/me"): (200, {"id": 1, "role": "staff"})}) as client:
        calc = _tool_map(build_tools(client))["calculate"]

        assert calc.execute({"operation": "add", "numbers": [10, 25, 5]}) == {
            "operation": "add",
            "numbers": [10, 25, 5],
            "result": 40,
        }
        assert calc.execute({"operation": "multiply", "numbers": [3, 4, 5]}) == {
            "operation": "multiply",
            "numbers": [3, 4, 5],
            "result": 60,
        }
        assert calc.execute({"operation": "subtract", "numbers": [50, 12, 8]}) == {
            "operation": "subtract",
            "numbers": [50, 12, 8],
            "result": 30,
        }

        for bad_args in (
            {"operation": "divide", "numbers": [10, 2]},
            {"operation": "add", "numbers": []},
            {"operation": "add", "numbers": [1, True]},
            {"operation": "add", "numbers": [1, 2.5]},
            {"operation": "add", "numbers": [1, "2"]},
        ):
            response = calc.execute(bad_args)
            assert "error" in response and isinstance(response["error"], str)


def test_structured_errors_never_raise_on_validation_http_or_connection_errors() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            return httpx.Response(200, json={"id": 1, "role": "owner"})
        if request.url.path == "/customers/999/balance":
            return httpx.Response(404, json={"detail": "Customer not found"})
        if request.url.path == "/stock":
            return httpx.Response(200, json=[{"product_id": 1, "current_stock": 5}])
        if request.url.path == "/sales":
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(500, json={"detail": "Server error"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(failing_handler),
    ) as client:
        tools = _tool_map(build_tools(client))

        missing_stock = tools["get_stock_level"].execute({"product_id": 99})
        not_found = tools["get_customer_balance"].execute({"customer_id": 999})
        conn_err = tools["get_sales"].execute({})
        bad_date = tools["get_daily_sales"].execute({"date": "25-11-2026"})
        bad_id = tools["get_customer_balance"].execute({"customer_id": True})
        extra_arg = tools["get_low_stock"].execute({"unexpected": 1})

    for err_result in (missing_stock, not_found, conn_err, bad_date, bad_id, extra_arg):
        assert set(err_result.keys()) == {"error"}
        assert err_result["error"]
        json.dumps(err_result)
