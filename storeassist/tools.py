from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
import math
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from storeassist.store_client import StoreClient, StoreClientError

MAX_LIST_ITEMS = 20


def _reject_bool_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Value must be an integer, not bool or non-integer.")
    return value


def _validate_iso_date(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    date_type.fromisoformat(stripped)
    return stripped


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchQueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="")


class ProductIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(gt=0)

    @field_validator("product_id", mode="before")
    @classmethod
    def _check_product_id(cls, value: Any) -> int:
        return _reject_bool_int(value)


class CustomerIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(gt=0)

    @field_validator("customer_id", mode="before")
    @classmethod
    def _check_customer_id(cls, value: Any) -> int:
        return _reject_bool_int(value)


class CustomerLedgerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(gt=0)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("customer_id", "limit", mode="before")
    @classmethod
    def _check_ints(cls, value: Any) -> int:
        return _reject_bool_int(value)


class GetSalesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str | None = None
    customer_id: int | None = Field(default=None, gt=0)

    @field_validator("date", mode="before")
    @classmethod
    def _check_date(cls, value: Any) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError("date must be an ISO YYYY-MM-DD string.")
        return _validate_iso_date(value)

    @field_validator("customer_id", mode="before")
    @classmethod
    def _check_customer_id(cls, value: Any) -> int | None:
        if value is None:
            return None
        return _reject_bool_int(value)


class GetDailySalesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _check_date(cls, value: Any) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError("date must be an ISO YYYY-MM-DD string.")
        return _validate_iso_date(value)


class CalculateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["add", "multiply", "subtract"]
    numbers: list[int] = Field(min_length=1)

    @field_validator("numbers", mode="before")
    @classmethod
    def _check_numbers(cls, value: Any) -> list[int]:
        if not isinstance(value, list) or not value:
            raise ValueError("numbers must be a non-empty list of integers.")
        checked: list[int] = []
        for item in value:
            checked.append(_reject_bool_int(item))
        return checked


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]

    def execute(self, raw_args: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {} if raw_args is None else raw_args
        if not isinstance(payload, dict):
            return {"error": f"Arguments for {self.name} must be an object."}
        try:
            validated = self.args_model.model_validate(payload)
            return self.handler(validated)
        except (StoreClientError, ValueError) as exc:
            return {"error": str(exc)}


def _truncate_list(items: Any, max_items: int = MAX_LIST_ITEMS) -> dict[str, Any]:
    if not isinstance(items, list):
        return {"error": "Expected a list response from StoreStock."}
    total_count = len(items)
    sliced = items[:max_items]
    return {
        "items": sliced,
        "truncated": total_count > max_items,
        "total_count": total_count,
        "returned_count": len(sliced),
    }


def _calculate(args: CalculateArgs) -> dict[str, Any]:
    if args.operation == "add":
        result = sum(args.numbers)
    elif args.operation == "multiply":
        result = math.prod(args.numbers)
    else:
        result = args.numbers[0] - sum(args.numbers[1:])
    return {
        "operation": args.operation,
        "numbers": args.numbers,
        "result": result,
    }


def _search_products(client: StoreClient, args: SearchQueryArgs) -> dict[str, Any]:
    data = client.get("/products", params={"q": args.query})
    return _truncate_list(data)


def _get_low_stock(client: StoreClient, args: EmptyArgs) -> dict[str, Any]:
    data = client.get("/stock/low")
    return _truncate_list(data)


def _get_stock_level(client: StoreClient, args: ProductIdArgs) -> dict[str, Any]:
    items = client.get("/stock")
    if not isinstance(items, list):
        return {"error": "Expected a list response from StoreStock."}
    for item in items:
        if isinstance(item, dict) and item.get("product_id") == args.product_id:
            return item
    return {"error": f"Stock level not found for product_id {args.product_id}."}


def _search_customers(client: StoreClient, args: SearchQueryArgs) -> dict[str, Any]:
    data = client.get("/customers", params={"q": args.query})
    return _truncate_list(data)


def _get_customer_balance(client: StoreClient, args: CustomerIdArgs) -> dict[str, Any]:
    data = client.get(f"/customers/{args.customer_id}/balance")
    if not isinstance(data, dict):
        return {"error": "Expected an object response from StoreStock."}
    return data


def _get_customer_ledger(client: StoreClient, args: CustomerLedgerArgs) -> dict[str, Any]:
    data = client.get(
        f"/customers/{args.customer_id}/ledger",
        params={"limit": args.limit},
    )
    return _truncate_list(data)


def _get_sales(client: StoreClient, args: GetSalesArgs) -> dict[str, Any]:
    data = client.get(
        "/sales",
        params={"date": args.date, "customer_id": args.customer_id},
    )
    return _truncate_list(data)


def _get_daily_sales(client: StoreClient, args: GetDailySalesArgs) -> dict[str, Any]:
    data = client.get("/reports/daily-summary", params={"date": args.date})
    if not isinstance(data, dict):
        return {"error": "Expected an object response from StoreStock."}
    return data


def _suggest_reorder(client: StoreClient, args: EmptyArgs) -> dict[str, Any]:
    data = client.get("/purchase-orders/suggest")
    return _truncate_list(data)


def build_tools(client: StoreClient) -> list[Tool]:
    user = client.get("/users/me")
    role = user.get("role") if isinstance(user, dict) else None

    tools: list[Tool] = [
        Tool(
            name="search_products",
            description="Search store products by name or SKU.",
            args_model=SearchQueryArgs,
            handler=lambda args: _search_products(client, args),
        ),
        Tool(
            name="get_low_stock",
            description="List products currently at or below reorder level.",
            args_model=EmptyArgs,
            handler=lambda args: _get_low_stock(client, args),
        ),
        Tool(
            name="get_stock_level",
            description="Get current stock and reorder level for one product_id.",
            args_model=ProductIdArgs,
            handler=lambda args: _get_stock_level(client, args),
        ),
        Tool(
            name="search_customers",
            description="Search customers by name or phone.",
            args_model=SearchQueryArgs,
            handler=lambda args: _search_customers(client, args),
        ),
        Tool(
            name="get_customer_balance",
            description="Get outstanding credit balance for one customer_id.",
            args_model=CustomerIdArgs,
            handler=lambda args: _get_customer_balance(client, args),
        ),
        Tool(
            name="get_customer_ledger",
            description="Get recent credit ledger entries for one customer_id.",
            args_model=CustomerLedgerArgs,
            handler=lambda args: _get_customer_ledger(client, args),
        ),
        Tool(
            name="get_sales",
            description="List sales filtered by optional date (YYYY-MM-DD) or customer_id.",
            args_model=GetSalesArgs,
            handler=lambda args: _get_sales(client, args),
        ),
        Tool(
            name="calculate",
            description="Perform exact integer arithmetic: add, multiply, or subtract.",
            args_model=CalculateArgs,
            handler=_calculate,
        ),
    ]

    if role == "owner":
        tools.extend(
            [
                Tool(
                    name="get_daily_sales",
                    description="Get daily sales and profit summary for an optional YYYY-MM-DD date.",
                    args_model=GetDailySalesArgs,
                    handler=lambda args: _get_daily_sales(client, args),
                ),
                Tool(
                    name="suggest_reorder",
                    description="Get suggested purchase order reorder quantities for low-stock items.",
                    args_model=EmptyArgs,
                    handler=lambda args: _suggest_reorder(client, args),
                ),
            ]
        )

    return tools
