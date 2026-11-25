from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

from storeassist.agent import AgentResult, TraceStep
from storeassist.guardrails import (
    check_tool_ambiguity,
    is_out_of_scope_request,
    is_write_request,
)
from storeassist.prompts import today_ist_iso
from storeassist.tools import Tool

_LOW_STOCK_RE = re.compile(r"\b(?:low|khatam|kam|running low|shortage)\b")
_REORDER_RE = re.compile(r"\b(?:reorder|purchase order|mangwana|restock)\b")
_DAILY_SALES_RE = re.compile(r"\b(?:sales|sale|bika|bikri|kamai|revenue|profit|summary)\b")
_TODAY_RE = re.compile(r"\b(?:today|aaj|daily)\b")
_YESTERDAY_RE = re.compile(r"\b(?:yesterday|kal)\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_CUSTOMER_BALANCE_RE = re.compile(r"\b(?:owe|owes|udhaar|balance|baaki|ledger|khata|customer)\b")
_LEDGER_RE = re.compile(r"\b(?:ledger|khata|entries|history)\b")
_STOCK_LEVEL_RE = re.compile(r"\b(?:stock|left|available|bacha|bache|units|quantity)\b")

_CUSTOMER_STOPWORDS = {
    "how",
    "much",
    "does",
    "do",
    "owe",
    "owes",
    "me",
    "what",
    "is",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "check",
    "show",
    "find",
    "get",
    "recent",
    "credit",
    "balance",
    "udhaar",
    "baaki",
    "ledger",
    "khata",
    "entries",
    "customer",
    "sales",
    "made",
    "ka",
    "ki",
    "ke",
    "pe",
    "kitna",
    "hai",
    "karo",
    "batao",
    "and",
    "his",
    "her",
}

_PRODUCT_STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "of",
    "for",
    "in",
    "on",
    "a",
    "an",
    "how",
    "much",
    "many",
    "price",
    "rate",
    "cost",
    "mrp",
    "daam",
    "bhav",
    "stock",
    "level",
    "left",
    "available",
    "units",
    "unit",
    "do",
    "we",
    "currently",
    "have",
    "show",
    "all",
    "list",
    "find",
    "check",
    "tell",
    "me",
    "product",
    "products",
    "item",
    "items",
    "kitne",
    "kitna",
    "ka",
    "ki",
    "ke",
    "kya",
    "hai",
    "rupaye",
    "rupees",
    "batao",
    "bacha",
    "bache",
}


def _run_tool(
    tool_map: dict[str, Tool],
    tool_name: str,
    args: dict[str, Any],
    trace: list[TraceStep],
) -> dict[str, Any]:
    tool = tool_map.get(tool_name)
    if tool is None:
        result: dict[str, Any] = {"error": f"Tool {tool_name} is not available for your role."}
    else:
        result = tool.execute(args)
    trace.append(TraceStep(tool_name=tool_name, args=args, result=result))
    return result


def _extract_query(question: str, stopwords: set[str]) -> str:
    cleaned = re.sub(r"[?!.,'\"]+", " ", question)
    tokens = [token for token in cleaned.split() if token.lower() not in stopwords]
    return " ".join(tokens).strip()


def _extract_customer_query(question: str) -> str:
    stripped = re.sub(r"'s\b", "", question)
    cleaned = re.sub(r"[?!.,\"]+", " ", stripped)
    tokens = [token for token in cleaned.split() if token.lower() not in _CUSTOMER_STOPWORDS]
    return " ".join(tokens).strip()


def _pick_customer(items: list[dict[str, Any]], query: str) -> dict[str, Any]:
    normalized = query.strip().lower()
    for item in items:
        name = str(item.get("name", "")).strip().lower()
        if name == normalized:
            return item
    return items[0]


def _pick_product(items: list[dict[str, Any]], query: str) -> dict[str, Any]:
    normalized = query.strip().lower()
    for item in items:
        name = str(item.get("name", "")).strip().lower()
        sku = str(item.get("sku", "")).strip().lower()
        if normalized in {name, sku}:
            return item
    return items[0]


def _handle_low_stock(tool_map: dict[str, Tool], trace: list[TraceStep]) -> AgentResult:
    result = _run_tool(tool_map, "get_low_stock", {}, trace)
    if "error" in result:
        return AgentResult(answer=result["error"], trace=trace, steps=len(trace), outcome="error")
    items = result.get("items", [])
    if not items:
        return AgentResult(
            answer="No products are currently low in stock.",
            trace=trace,
            steps=len(trace),
            outcome="answered",
        )
    summary = ", ".join(f"{item['name']} ({item['current_stock']})" for item in items)
    return AgentResult(
        answer=f"Low stock items: {summary}.",
        trace=trace,
        steps=len(trace),
        outcome="answered",
    )


def _handle_reorder(tool_map: dict[str, Tool], trace: list[TraceStep]) -> AgentResult:
    if "suggest_reorder" not in tool_map:
        return AgentResult(
            answer="Reorder suggestions are only available to owner accounts.",
            trace=trace,
            steps=0,
            outcome="refused",
        )
    result = _run_tool(tool_map, "suggest_reorder", {}, trace)
    if "error" in result:
        return AgentResult(answer=result["error"], trace=trace, steps=len(trace), outcome="error")
    items = result.get("items", [])
    if not items:
        return AgentResult(
            answer="No reorder suggestions right now.",
            trace=trace,
            steps=len(trace),
            outcome="answered",
        )
    summary = ", ".join(f"{item['name']}: {item['suggested_qty']}" for item in items)
    return AgentResult(
        answer=f"Suggested reorders: {summary}.",
        trace=trace,
        steps=len(trace),
        outcome="answered",
    )


def _handle_daily_sales(
    question: str,
    tool_map: dict[str, Tool],
    trace: list[TraceStep],
) -> AgentResult:
    if "get_daily_sales" not in tool_map:
        return AgentResult(
            answer="Daily sales summary is only available to owner accounts.",
            trace=trace,
            steps=0,
            outcome="refused",
        )
    lowered = question.lower()
    args: dict[str, Any] = {}
    date_match = _ISO_DATE_RE.search(question)
    if date_match:
        args["date"] = date_match.group(0)
    elif _YESTERDAY_RE.search(lowered):
        today_str = today_ist_iso()
        year, month, day = (int(part) for part in today_str.split("-"))
        args["date"] = (date(year, month, day) - timedelta(days=1)).isoformat()

    result = _run_tool(tool_map, "get_daily_sales", args, trace)
    if "error" in result:
        return AgentResult(answer=result["error"], trace=trace, steps=len(trace), outcome="error")
    total_rupees = result.get("total_sales_rupees", "0.00")
    sales_count = result.get("sales_count", 0)
    return AgentResult(
        answer=f"Total sales: Rs {total_rupees} across {sales_count} sales.",
        trace=trace,
        steps=len(trace),
        outcome="answered",
    )


def _handle_customer(
    question: str,
    tool_map: dict[str, Tool],
    trace: list[TraceStep],
) -> AgentResult:
    query = _extract_customer_query(question)
    search_res = _run_tool(tool_map, "search_customers", {"query": query}, trace)
    if "error" in search_res:
        return AgentResult(answer=search_res["error"], trace=trace, steps=len(trace), outcome="error")

    clarification = check_tool_ambiguity(question, "search_customers", {"query": query}, search_res)
    if clarification is not None:
        return AgentResult(answer=clarification, trace=trace, steps=len(trace), outcome="clarification")

    items = search_res.get("items", [])
    if not items:
        return AgentResult(
            answer=f"No customer found matching '{query}'.",
            trace=trace,
            steps=len(trace),
            outcome="answered",
        )

    customer = _pick_customer(items, query)
    customer_id = customer["id"]
    customer_name = customer["name"]
    if _LEDGER_RE.search(question.lower()):
        ledger_res = _run_tool(tool_map, "get_customer_ledger", {"customer_id": customer_id}, trace)
        if "error" in ledger_res:
            return AgentResult(answer=ledger_res["error"], trace=trace, steps=len(trace), outcome="error")
        count = ledger_res.get("returned_count", 0)
        return AgentResult(
            answer=f"Found {count} ledger entries for {customer_name}.",
            trace=trace,
            steps=len(trace),
            outcome="answered",
        )

    bal_res = _run_tool(tool_map, "get_customer_balance", {"customer_id": customer_id}, trace)
    if "error" in bal_res:
        return AgentResult(answer=bal_res["error"], trace=trace, steps=len(trace), outcome="error")
    return AgentResult(
        answer=f"{customer_name} has a balance of Rs {bal_res.get('balance_rupees', '0.00')}.",
        trace=trace,
        steps=len(trace),
        outcome="answered",
    )


def _handle_product(
    question: str,
    tool_map: dict[str, Tool],
    trace: list[TraceStep],
) -> AgentResult:
    query = _extract_query(question, _PRODUCT_STOPWORDS)
    if not query:
        return AgentResult(
            answer="Sorry, I could not understand that question.",
            trace=trace,
            steps=0,
            outcome="answered",
        )

    search_res = _run_tool(tool_map, "search_products", {"query": query}, trace)
    if "error" in search_res:
        return AgentResult(answer=search_res["error"], trace=trace, steps=len(trace), outcome="error")

    clarification = check_tool_ambiguity(question, "search_products", {"query": query}, search_res)
    if clarification is not None:
        return AgentResult(answer=clarification, trace=trace, steps=len(trace), outcome="clarification")

    items = search_res.get("items", [])
    if not items:
        return AgentResult(
            answer=f"No product found matching '{query}'.",
            trace=trace,
            steps=len(trace),
            outcome="answered",
        )

    if _STOCK_LEVEL_RE.search(question.lower()) and len(items) == 1:
        product = _pick_product(items, query)
        stock_res = _run_tool(tool_map, "get_stock_level", {"product_id": product["id"]}, trace)
        if "error" in stock_res:
            return AgentResult(answer=stock_res["error"], trace=trace, steps=len(trace), outcome="error")
        return AgentResult(
            answer=f"{stock_res['name']} current stock is {stock_res['current_stock']}.",
            trace=trace,
            steps=len(trace),
            outcome="answered",
        )

    summary = ", ".join(f"{item['name']} (Rs {item['price_rupees']})" for item in items)
    return AgentResult(
        answer=f"Matching products: {summary}.",
        trace=trace,
        steps=len(trace),
        outcome="answered",
    )


def run_baseline(question: str, tools: list[Tool]) -> AgentResult:
    if is_write_request(question):
        return AgentResult(
            answer="I am a read-only assistant and cannot create, update, or delete store records.",
            trace=[],
            steps=0,
            outcome="refused",
        )
    if is_out_of_scope_request(question):
        return AgentResult(
            answer="I can only help with StoreStock store questions and cannot answer out-of-scope requests.",
            trace=[],
            steps=0,
            outcome="refused",
        )

    tool_map = {tool.name: tool for tool in tools}
    trace: list[TraceStep] = []
    lowered = question.lower()

    if _LOW_STOCK_RE.search(lowered):
        return _handle_low_stock(tool_map, trace)
    if _REORDER_RE.search(lowered):
        return _handle_reorder(tool_map, trace)
    if _DAILY_SALES_RE.search(lowered) and (
        _TODAY_RE.search(lowered) or _YESTERDAY_RE.search(lowered) or _ISO_DATE_RE.search(lowered)
    ):
        return _handle_daily_sales(question, tool_map, trace)
    if _CUSTOMER_BALANCE_RE.search(lowered):
        return _handle_customer(question, tool_map, trace)
    return _handle_product(question, tool_map, trace)
