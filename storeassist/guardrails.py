from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any

_LINE_LIST_PREFIX_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])-?\d[\d,]*(?:\.\d+)?")

_WRITE_VERB_RE = re.compile(
    r"\b(?:delete|remove|drop|update|change|modify|edit|set|create|add|insert|"
    r"adjust|cancel|record|receive|import|pay|refund|hatao|badlo|mita)\b"
)
_STORE_WRITE_TARGET_RE = re.compile(
    r"\b(?:product|products|item|items|price|rate|cost|mrp|daam|bhav|stock|"
    r"inventory|sale|sales|customer|customers|supplier|suppliers|order|orders|"
    r"po|balance|payment|credit|udhaar|user|users|ledger)\b"
)
_OUT_OF_SCOPE_RE = re.compile(
    r"\b(?:weather|forecast|temperature|rain|mausam|barish|news|cricket|"
    r"match|score|movie|song|joke|poem|story|recipe|capital|president|"
    r"prime minister|bitcoin|crypto|horoscope|stock market)\b"
)
_REFUSAL_RESPONSE_RE = re.compile(
    r"(?:"
    r"\b(?:cannot|can't|unable to|won't|not able to)\b.{0,40}"
    r"\b(?:help|answer|assist|perform|modify|delete|update|create|change|provide|do that|fulfill)\b"
    r"|\b(?:only help|only answer|only assist|read-only|out of scope|outside my scope|"
    r"outside the scope|store questions only|store-related questions)\b"
    r")"
)
_STORE_DATA_RE = re.compile(
    r"\b(?:stock|inventory|product|products|item|items|price|rate|cost|mrp|"
    r"daam|bhav|sale|sales|sold|bika|bikri|profit|revenue|kamai|customer|"
    r"customers|udhaar|credit|balance|owe|owes|baaki|ledger|khata|reorder|"
    r"low|khatam|maal|dukaan|store|supplier|sku)\b"
)
_PRODUCT_LIST_INTENT_RE = re.compile(
    r"\b(?:list|all|every|catalog|catalogue|saare|sabhi|sab|dikhao|"
    r"what products|which products|kaun)\b"
)
_SPECIFIC_PRODUCT_INTENT_RE = re.compile(
    r"\b(?:price|rate|cost|mrp|daam|bhav|stock|left|available|bacha|bache|"
    r"kitne|kitna|how much|how many|sku|reorder)\b"
)
_CUSTOMER_LIST_INTENT_RE = re.compile(
    r"\b(?:list|all|every|saare|sabhi|sab|dikhao|what customers|which customers)\b"
)


@dataclass(frozen=True)
class TraceRecord:
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]


def _to_decimal(raw_token: str) -> Decimal | None:
    cleaned = raw_token.replace(",", "")
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return None
    return parsed


def _collect_iso_dates(text: str, dates: set[str]) -> None:
    for match in _ISO_DATE_RE.findall(text):
        dates.add(match)


def _collect_numbers_from_text(text: str, target: set[Decimal]) -> None:
    for token in _NUMBER_TOKEN_RE.findall(text):
        parsed = _to_decimal(token)
        if parsed is not None:
            target.add(parsed)


def _collect_from_value(value: Any, numbers: set[Decimal], dates: set[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        parsed = _to_decimal(str(value))
        if parsed is not None:
            numbers.add(parsed)
        return
    if isinstance(value, str):
        _collect_iso_dates(value, dates)
        _collect_numbers_from_text(value, numbers)
        return
    if isinstance(value, list):
        for item in value:
            _collect_from_value(item, numbers, dates)
        return
    if isinstance(value, dict):
        for item_value in value.values():
            _collect_from_value(item_value, numbers, dates)


def _strip_known_iso_dates(text: str, known_dates: set[str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        date_text = match.group(0)
        return " " if date_text in known_dates else date_text

    return _ISO_DATE_RE.sub(_replace, text)


def _extract_answer_numbers(answer: str, known_dates: set[str]) -> list[str]:
    without_list_nums = _LINE_LIST_PREFIX_RE.sub("", answer)
    without_dates = _strip_known_iso_dates(without_list_nums, known_dates)
    return _NUMBER_TOKEN_RE.findall(without_dates)


def verify_number_grounding(
    answer: str,
    question: str,
    trace: list[TraceRecord],
) -> tuple[bool, list[str]]:
    allowed_numbers: set[Decimal] = set()
    known_dates: set[str] = set()

    _collect_from_value(question, allowed_numbers, known_dates)
    for step in trace:
        _collect_from_value(step.args, allowed_numbers, known_dates)
        _collect_from_value(step.result, allowed_numbers, known_dates)

    ungrounded: list[str] = []
    for token in _extract_answer_numbers(answer, known_dates):
        parsed = _to_decimal(token)
        if parsed is None or parsed not in allowed_numbers:
            ungrounded.append(token)

    return len(ungrounded) == 0, ungrounded


def is_write_request(question: str) -> bool:
    lowered = question.lower()
    return bool(_WRITE_VERB_RE.search(lowered) and _STORE_WRITE_TARGET_RE.search(lowered))


def is_out_of_scope_request(question: str) -> bool:
    return bool(_OUT_OF_SCOPE_RE.search(question.lower()))


def is_refusal_response(answer: str) -> bool:
    return bool(_REFUSAL_RESPONSE_RE.search(answer.lower()))


def is_store_data_question(question: str) -> bool:
    return bool(_STORE_DATA_RE.search(question.lower()))


def _has_single_exact_match(
    items: list[dict[str, Any]],
    query: str,
    keys: tuple[str, ...],
) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return False
    exact_matches = 0
    for item in items:
        for key in keys:
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip().lower() == normalized:
                exact_matches += 1
                break
    return exact_matches == 1


def _should_clarify_products(question: str) -> bool:
    lowered = question.lower()
    if _SPECIFIC_PRODUCT_INTENT_RE.search(lowered):
        return True
    return not _PRODUCT_LIST_INTENT_RE.search(lowered)


def _should_clarify_customers(question: str) -> bool:
    return not _CUSTOMER_LIST_INTENT_RE.search(question.lower())


def _format_customer_candidates(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        name = str(item.get("name", "Unknown"))
        customer_id = item.get("id")
        phone = item.get("phone")
        details = f"ID {customer_id}" if customer_id is not None else ""
        if phone:
            details = f"{details}, {phone}" if details else str(phone)
        suffix = f" ({details})" if details else ""
        lines.append(f"- {name}{suffix}")
    return "\n".join(lines)


def _format_product_candidates(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        name = str(item.get("name", "Unknown"))
        sku = item.get("sku")
        product_id = item.get("id")
        parts: list[str] = []
        if sku:
            parts.append(f"SKU {sku}")
        if product_id is not None:
            parts.append(f"ID {product_id}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- {name}{suffix}")
    return "\n".join(lines)


def check_tool_ambiguity(
    question: str,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    items = result.get("items")
    if not isinstance(items, list) or len(items) <= 1:
        return None
    dict_items = [item for item in items if isinstance(item, dict)]
    if len(dict_items) <= 1:
        return None

    query = str(args.get("query", ""))
    if tool_name == "search_customers":
        if not _should_clarify_customers(question):
            return None
        if _has_single_exact_match(dict_items, query, ("name",)):
            return None
        candidates = _format_customer_candidates(dict_items)
        return f"Multiple customers matched your query. Please clarify which one you mean:\n{candidates}"

    if tool_name == "search_products":
        if not _should_clarify_products(question):
            return None
        if _has_single_exact_match(dict_items, query, ("name", "sku")):
            return None
        candidates = _format_product_candidates(dict_items)
        return f"Multiple products matched your query. Please clarify which one you mean:\n{candidates}"

    return None
