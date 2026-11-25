from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from storeassist.store_client import (
    StoreAPIError,
    StoreAuthError,
    StoreClient,
    StoreConnectionError,
    StoreNotFoundError,
    convert_paise_fields,
    paise_to_rupees,
)


def test_paise_to_rupees_guards_bool_and_none() -> None:
    assert paise_to_rupees(12550) == 125.5
    assert paise_to_rupees(0) == 0.0
    assert paise_to_rupees(-500) == -5.0
    assert paise_to_rupees(True) is None
    assert paise_to_rupees(False) is None
    assert paise_to_rupees(None) is None
    assert paise_to_rupees("1000") is None
    assert paise_to_rupees(12.5) is None


def test_convert_paise_fields_recursive_with_bool_and_none() -> None:
    payload: dict[str, Any] = {
        "id": 1,
        "total_sales_paise": 25075,
        "flag_paise": True,
        "missing_paise": None,
        "items": [
            {"unit_price_paise": 1500, "line_total_paise": 3000},
            {"unit_price_paise": False, "line_total_paise": None},
        ],
    }

    converted = convert_paise_fields(payload)

    assert converted == {
        "id": 1,
        "total_sales_paise": 25075,
        "total_sales_rupees": 250.75,
        "flag_paise": True,
        "missing_paise": None,
        "items": [
            {
                "unit_price_paise": 1500,
                "unit_price_rupees": 15.0,
                "line_total_paise": 3000,
                "line_total_rupees": 30.0,
            },
            {"unit_price_paise": False, "line_total_paise": None},
        ],
    }


def test_store_client_exposes_only_get_and_close_with_private_httpx_client() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    with StoreClient("http://127.0.0.1:8000", "owner", "secret", transport=transport) as client:
        assert not hasattr(client, "client")
        for attr in ("login", "post", "put", "patch", "delete", "request"):
            assert not hasattr(client, attr)
        public_methods = {
            name
            for name in dir(client)
            if not name.startswith("_") and callable(getattr(client, name))
        }
        assert public_methods == {"get", "close"}


def test_store_client_lazy_login_and_get_daily_summary() -> None:
    recorded: list[tuple[str, str, str | None, dict[str, str], dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        recorded.append(
            (
                request.method,
                request.url.path,
                request.headers.get("authorization"),
                dict(request.url.params),
                body,
            )
        )
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok-1", "token_type": "bearer"})
        if request.method == "GET" and request.url.path == "/reports/daily-summary":
            return httpx.Response(
                200,
                json={
                    "date": "2026-11-25",
                    "total_sales_paise": 50000,
                    "net_sales_paise": 47500,
                    "gross_profit_paise": 15500,
                    "sales_count": 5,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with StoreClient("http://127.0.0.1:8000/", "owner", "secret", transport=transport) as client:
        assert recorded == []
        result = client.get(
            "/reports/daily-summary",
            params={"date": "2026-11-25", "unused": None},
        )

    assert result == {
        "date": "2026-11-25",
        "total_sales_paise": 50000,
        "total_sales_rupees": 500.0,
        "net_sales_paise": 47500,
        "net_sales_rupees": 475.0,
        "gross_profit_paise": 15500,
        "gross_profit_rupees": 155.0,
        "sales_count": 5,
    }
    assert recorded == [
        ("POST", "/auth/login", None, {}, {"username": "owner", "password": "secret"}),
        ("GET", "/reports/daily-summary", "Bearer tok-1", {"date": "2026-11-25"}, None),
    ]


def test_store_client_relogins_once_on_401() -> None:
    calls: list[tuple[str, str, str | None]] = []
    login_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        calls.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.method == "POST" and request.url.path == "/auth/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"tok-{login_count}"})
        if request.method == "GET" and request.url.path == "/stock/low":
            if request.headers.get("authorization") == "Bearer tok-1":
                return httpx.Response(401, json={"detail": "Expired"})
            return httpx.Response(200, json=[{"product_id": 1, "current_stock": 2}])
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with StoreClient("http://127.0.0.1:8000", "owner", "secret", transport=transport) as client:
        data = client.get("/stock/low")

    assert data == [{"product_id": 1, "current_stock": 2}]
    assert calls == [
        ("POST", "/auth/login", None),
        ("GET", "/stock/low", "Bearer tok-1"),
        ("POST", "/auth/login", None),
        ("GET", "/stock/low", "Bearer tok-2"),
    ]


def test_store_client_does_not_retry_repeated_401_or_other_4xx() -> None:
    repeated_401_calls: list[tuple[str, str]] = []

    def repeated_401_handler(request: httpx.Request) -> httpx.Response:
        repeated_401_calls.append((request.method, request.url.path))
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(401, json={"detail": "Unauthorized"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(repeated_401_handler),
    ) as client:
        with pytest.raises(StoreAuthError):
            client.get("/stock")

    assert repeated_401_calls == [
        ("POST", "/auth/login"),
        ("GET", "/stock"),
        ("POST", "/auth/login"),
        ("GET", "/stock"),
    ]

    forbidden_calls: list[tuple[str, str]] = []

    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        forbidden_calls.append((request.method, request.url.path))
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(403, json={"detail": "Forbidden"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(forbidden_handler),
    ) as client:
        with pytest.raises(StoreAuthError):
            client.get("/reports/inventory-valuation")

    assert forbidden_calls == [
        ("POST", "/auth/login"),
        ("GET", "/reports/inventory-valuation"),
    ]

    not_found_calls: list[tuple[str, str]] = []

    def not_found_handler(request: httpx.Request) -> httpx.Response:
        not_found_calls.append((request.method, request.url.path))
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(404, json={"detail": "Not found"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(not_found_handler),
    ) as client:
        with pytest.raises(StoreNotFoundError):
            client.get("/products/999")

    assert not_found_calls == [
        ("POST", "/auth/login"),
        ("GET", "/products/999"),
    ]


def test_store_client_timeout_and_connection_errors() -> None:
    seen_timeouts: list[Any] = []

    def timeout_capture_handler(request: httpx.Request) -> httpx.Response:
        seen_timeouts.append(request.extensions.get("timeout"))
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(200, json={"ok": True})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        timeout=3.5,
        transport=httpx.MockTransport(timeout_capture_handler),
    ) as client:
        client.get("/users/me")

    assert len(seen_timeouts) == 2
    for timeout_ext in seen_timeouts:
        assert timeout_ext == {"connect": 3.5, "read": 3.5, "write": 3.5, "pool": 3.5}

    def timeout_error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out", request=request)

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(timeout_error_handler),
    ) as client:
        with pytest.raises(StoreConnectionError):
            client.get("/stock")


def test_store_client_login_failures_and_server_errors() -> None:
    def bad_creds_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Bad credentials"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "wrong",
        transport=httpx.MockTransport(bad_creds_handler),
    ) as client:
        with pytest.raises(StoreAuthError):
            client.get("/users/me")

    def server_error_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(500, json={"detail": "Boom"})

    with StoreClient(
        "http://127.0.0.1:8000",
        "owner",
        "secret",
        transport=httpx.MockTransport(server_error_handler),
    ) as client:
        with pytest.raises(StoreAPIError) as exc_info:
            client.get("/stock")
        assert exc_info.value.status_code == 500
