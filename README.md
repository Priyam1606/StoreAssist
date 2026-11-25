# StoreAssist

StoreAssist is a read-only CLI assistant for general store owners backed by the **StoreStock** REST API. An owner or staff member can ask store questions in plain English or Roman-script Hinglish (for example, *"what is running low?"*, *"kitna maal bika aaj?"*, or *"how much does Priya Nair owe me?"*) and receive answers grounded in live StoreStock HTTP responses.

## Architecture

StoreAssist uses a hand-written manual tool-calling loop (`storeassist/agent.py`) behind an `LLMClient` abstraction (`storeassist/llm.py`) with code-enforced guardrails (`storeassist/guardrails.py`):

```text
  User Question (English / Roman Hinglish)
        │
        ▼
  ┌───────────────────┐
  │     LLMClient     │◄───────────────────────────────────────────┐
  │ (Gemini / Fake)   │                                            │
  └────────┬──────────┘                                            │
           │                                                       │
     ┌─────┴──────────────────────────┐                            │
     ▼ (tool_call, step <= 5)         ▼ (final text)               │
┌────────────────────────┐   ┌─────────────────────────┐           │
│ Tool Execution         │   │ Final Answer Guardrails │           │
│ - Pydantic v2 args     │   │ - Write / OOS refusal   │           │
│ - StoreClient GET-only │   │ - No-tool store block   │           │
│ - *_paise -> *_rupees  │   │ - Decimal number check  │           │
│ - 20-item truncation   │   └───────────┬─────────────┘           │
└──────────┬─────────────┘               │                         │
           │                             ▼                         │
           ▼                     Final Answer / Refusal /          │
┌────────────────────────┐       Ungrounded Fallback               │
│ Ambiguity Guard        │                                         │
│ (>1 match & no exact   │──► Clarification (lists candidates)     │
│  single match -> stop) │                                         │
└──────────┬─────────────┘                                         │
           │ (clear result)                                        │
           └───────────────────────────────────────────────────────┘
```

- **Hard loop cap**: `run_agent` allows at most 5 tool-calling steps (`MAX_TOOL_STEPS = 5`); if the model asks for a 6th tool call, execution stops immediately with `outcome="step_limit"`.
- **SDK isolation**: `GeminiClient` disables automatic function calling (`AutomaticFunctionCallingConfig(disable=True)`), sets `temperature=0.0`, preserves Gemini 3+ `thought_signature` bytes across turns, and retries transient `429`/`503` errors with `4s / 8s / 16s` exponential backoff.

## Tools

At startup, `build_tools(client)` calls `GET /users/me` once to check the user's role (`owner` or `staff`). Owner-only tools are omitted from registration for `staff` accounts.

| Tool | StoreStock Endpoint / Implementation | Parameters | Role Access |
|---|---|---|---|
| `search_products` | `GET /products` (`q`) | `query: str = ""` | `owner`, `staff` |
| `get_low_stock` | `GET /stock/low` | *(none)* | `owner`, `staff` |
| `get_stock_level` | `GET /stock` (filtered by `product_id`) | `product_id: int` | `owner`, `staff` |
| `search_customers` | `GET /customers` (`q`) | `query: str = ""` | `owner`, `staff` |
| `get_customer_balance` | `GET /customers/{customer_id}/balance` | `customer_id: int` | `owner`, `staff` |
| `get_customer_ledger` | `GET /customers/{customer_id}/ledger` | `customer_id: int`, `limit: int = 20` | `owner`, `staff` |
| `get_sales` | `GET /sales` | `date: str \| None`, `customer_id: int \| None` | `owner`, `staff` |
| `calculate` | Pure Python integer math (`add`, `multiply`, `subtract`) | `operation`, `numbers: list[int]` | `owner`, `staff` |
| `get_daily_sales` | `GET /reports/daily-summary` | `date: str \| None` | `owner` only |
| `suggest_reorder` | `GET /purchase-orders/suggest` | *(none)* | `owner` only |

All list-returning tools cap output at 20 items and return `items`, `truncated`, `total_count`, and `returned_count`.

## Safety & Guardrails Design

1. **Read-only HTTP boundary (`storeassist/store_client.py`)**: `StoreClient` keeps its `httpx.Client` private (`self._client`) and exposes only `get(path, params=None)` and `close()`. Authentication uses lazy `POST /auth/login` with a single re-login retry on `401`, explicit request timeouts, `StoreConnectionError` on network failures, and no retries on other `4xx` responses.
2. **Role-based tool gating (`storeassist/tools.py`)**: Calls `GET /users/me` once at startup and registers `get_daily_sales` and `suggest_reorder` only when `role == "owner"`.
3. **Exact paise-to-rupees conversion (`storeassist/store_client.py`)**: Recursively adds `*_rupees` two-decimal strings via integer `divmod` (e.g. `2800 -> "28.00"`, `-50 -> "-0.50"`) beside every original integer `*_paise` field while keeping `*_paise` intact and ignoring `bool` and `None`.
4. **Code-enforced ambiguity clarification (`storeassist/guardrails.py`)**: When `search_customers` or `search_products` (for non-list queries) returns multiple matches with no single exact match, the loop stops immediately and returns `outcome="clarification"` listing the candidates so the model never guesses an ID.
5. **Decimal number grounding (`storeassist/guardrails.py`)**: Every number in the final answer is checked using `Decimal` equivalence (ignoring line-start `1.`/`2.` list markers, `bool` flags, and known ISO dates) against numbers in the user's question and tool outputs—including UTC and `Asia/Kolkata` (`UTC+05:30`) date/time components extracted from ISO-8601 timestamps and list item counts. Unverified numbers or zero-tool store answers are blocked with `outcome="ungrounded"`.
6. **Prompt-injection & out-of-scope defense (`storeassist/prompts.py`, `storeassist/guardrails.py`)**: `SYSTEM_PROMPT` marks tool outputs strictly as untrusted data, injects today's `Asia/Kolkata` date, and instructs quoting `*_rupees`. Write/modify requests and out-of-scope questions are classified in code and stopped with `outcome="refused"`.

## Windows PowerShell Setup & Usage

### 1. Create Virtual Environment & Install Pinned Dependencies

```powershell
cd D:\StoreAssist
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Never commit secrets to files. Set environment variables in your PowerShell session (use `gemini-3.5-flash-lite` or `gemini-3.8-flash` on the Gemini free tier):

```powershell
$env:STORESTOCK_URL = "http://127.0.0.1:8000"   # Optional; defaults to http://127.0.0.1:8000
$env:STORESTOCK_USERNAME = "<your-storestock-username>"
$env:STORESTOCK_PASSWORD = "<your-storestock-password>"
$env:LLM_API_KEY = "<your-gemini-api-key>"
$env:LLM_MODEL = "gemini-3.5-flash-lite"
```

### 3. Run the CLI, Tests, and Evaluation

```powershell
# Ask a single question (add --trace to print tool calls, args, results, steps, and outcome)
python -m storeassist ask "What products are running low on stock?"
python -m storeassist ask "kitna maal bika aaj?" --trace

# Start stateless interactive chat (type 'exit' or 'quit' to leave)
python -m storeassist chat --trace

# Run offline unit tests
pytest -v

# Run live evaluation against StoreStock + Gemini (with 5s delay for free-tier rate limits)
python -m eval.run_eval --delay 5.0
```

## Evaluation Results & Failure Analysis

Benchmark: 42 frozen questions in `eval/questions.json` (`10` `simple_english`, `10` `hinglish`, `7` `ambiguous`, `8` `out_of_scope`, `7` `multi_step`) run against live StoreStock with `gemini-3.5-flash-lite`, `temperature=0.0`, on `2026-11-26` (saved in `eval/results/20261125_183856.json`).

### Run 1 to Run 2 Iteration

Run 1 (`eval/results/20261125_181208.json`) achieved `33/42` agent outcome accuracy because live runs surfaced two real defects: 6 transient `429 RESOURCE_EXHAUSTED` / `503 UNAVAILABLE` failures and 3 false-positive `ungrounded` blocks on ISO-8601 timestamps (`ms_03`, `ms_04`, `ms_05`). Adding bounded exponential backoff (`4s / 8s / 16s`) in `GeminiClient` and UTC + `Asia/Kolkata` ISO-8601 timestamp extraction in `guardrails.py`—without altering `eval/questions.json`—raised agent outcome accuracy to **41/42 (97.6%)** in Run 2.

### Run 2 Side-by-Side Results (Agent vs Keyword Baseline)

| Metric | Agent (`gemini-3.5-flash-lite`) | Keyword Baseline |
|---|---|---|
| First-tool accuracy | 41/42 (97.6%) | 41/42 (97.6%) |
| Argument accuracy | 36/42 (85.7%) | 33/42 (78.6%) |
| Outcome accuracy | 41/42 (97.6%) | 37/42 (88.1%) |
| Grounded-answer rate | 41/42 (97.6%) | 42/42 (100.0%) |
| Average steps | 1.24 | 1.02 |

| Category (Outcome Accuracy) | Agent | Baseline |
|---|---|---|
| `simple_english` | 10/10 (100.0%) | 10/10 (100.0%) |
| `hinglish` | 10/10 (100.0%) | 10/10 (100.0%) |
| `ambiguous` | 7/7 (100.0%) | 2/7 (28.6%) |
| `out_of_scope` | 8/8 (100.0%) | 8/8 (100.0%) |
| `multi_step` | 6/7 (85.7%) | 7/7 (100.0%) |

### Concrete Failure Analysis from Run 2 Traces

1. **`ms_05` (`outcome="ungrounded"` — the sole agent outcome failure)**:
   - On *"Check Rajesh Gupta's credit balance and his recent ledger entries."*, the agent executed all three right tools in sequence: `search_customers({"query": "Rajesh Gupta"})` -> `get_customer_balance({"customer_id": 6})` (`"555.00"`) -> `get_customer_ledger({"customer_id": 6, "limit": 20})` (returning 3 entries with `amount_rupees`: `"290.00"`, `"-250.00"`, `"515.00"`).
   - `_finalize_answer` then blocked the model's final reply with `"Could not verify all numbers in the response against StoreStock tool outputs."`. Because `run_agent` replaces the rejected draft text rather than logging it to `eval/results/20261125_183856.json`, the exact offending token cannot be read directly from the trace; likely triggers in `ms_05` include rendering the negative payment `"-250.00"` as an unsigned `"250.00"` (`Decimal("250.00") != Decimal("-250.00")`) or inline numbering outside line-start `1.`/`2.` prefixes.
2. **First-turn multi-word & Hinglish substring queries (`hi_04`, `am_04`, `am_05`, `am_07`, `ms_02`)**:
   - StoreStock's `/products?q=...` performs contiguous SQL substring matching (`LIKE`). On Turn 1, the agent passed `"chini"` (`hi_04`), `"Aashirvaad atta"` (`am_04`), `"India Gate rice"` (`am_05`), `"Haldiram bhujia"` (`am_07`), and `"Dettol soap"` (`ms_02`), each returning `0` items and costing `arg_accuracy` (5 of the 6 agent argument misses).
   - Unlike the single-shot baseline, the agent recovered on Turn 2 (`"madhur"`, `"atta"`, `"India Gate"`, `"bhujia"`, `"Dettol"`), achieving the correct final outcome in all five cases (raising `average_steps` to `1.24`).
3. **`se_07` (`Which items should I reorder?`)**:
   - The agent called `get_low_stock` instead of `suggest_reorder`, returning the 10 low-stock items and offering to check reorder quantities (`first_tool` and `args` miss, while `outcome` remained `"answered"`).
4. **Why Baseline scored `7/7 (100.0%)` on `multi_step` yet collapsed to `2/7 (28.6%)` on `ambiguous`**:
   - In `eval/run_eval.py`, `outcome_correct` checks `result.outcome == expected_outcome`. On `ms_04` (*"Show the sales made to Manoj Singh."*), Baseline routed to the wrong endpoint (`search_products("sales made to Manoj Singh")`), received `0` results, and returned `"No product found..."` with `outcome="answered"`—which still matched `expected_outcome="answered"` despite failing `first_tool` and `args` (and on `ms_06`/`ms_07`, it returned raw lists without calling `calculate`).
   - Conversely, on `ambiguous`, Baseline's stopword stripper passed noisy strings (`"starting with Ra"`, `"starting with An"`, `"Aashirvaad atta"`, `"India Gate rice"`, `"Haldiram bhujia"`) that returned `0` items, bypassing `check_tool_ambiguity` and returning `"answered"` instead of `"clarification"` on 5 of 7 queries.

## Honest Limitations

- **Stateless `chat` mode**: Every prompt runs a fresh `run_agent` call with no multi-turn conversation memory across questions, so follow-up replies to clarification prompts must restate the full question.
- **Contiguous SQL substring search**: StoreStock matches `q` via literal substring matching rather than semantic or synonym search, so descriptive phrases (`"Dettol soap"`) or Hindi nouns (`"chini"`) miss on the first lookup unless narrowed to a store-name substring.
- **English and Roman-script Hinglish only**: Prompts and keyword rules are designed for English and Romanized Hindi; Devanagari script is not supported.
- **Free-tier Gemini latency and rate limits**: Multi-step queries and `429`/`503` backoff retries (`4s`, `8s`, `16s`) add latency under free-tier quotas.
- **Read-only by design**: StoreAssist intentionally refuses all create, update, stock-adjustment, payment, and delete operations.
