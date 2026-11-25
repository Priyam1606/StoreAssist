from __future__ import annotations

import io
import subprocess
import sys

import httpx
import pytest

from storeassist.cli import main
from storeassist.config import Config
from storeassist.llm import FakeLLMClient, LLMResponse, ToolCall
from storeassist.store_client import StoreClient


class _TrackingStoreClient(StoreClient):
    def __init__(self, handler: httpx.MockTransport) -> None:
        super().__init__("http://127.0.0.1:8000", "owner", "secret", transport=handler)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


def _test_config() -> Config:
    return Config(
        storestock_url="http://127.0.0.1:8000",
        storestock_username="owner",
        storestock_password="secret",
        llm_api_key="test-key",
        llm_model="gemini-3.5-flash-lite",
    )


def _mock_transport(fail_users_me: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/users/me":
            if fail_users_me:
                raise httpx.ConnectError("Connection refused", request=request)
            return httpx.Response(200, json={"id": 1, "role": "owner"})
        if request.url.path == "/stock/low":
            return httpx.Response(
                200,
                json=[{"product_id": 2, "name": "Sugar 1kg", "current_stock": 1, "reorder_level": 5}],
            )
        return httpx.Response(404, json={"detail": "Not found"})

    return httpx.MockTransport(handler)


def test_cli_ask_prints_answer_and_closes_store_client() -> None:
    client_box: list[_TrackingStoreClient] = []
    stdout = io.StringIO()
    stderr = io.StringIO()
    fake_llm = FakeLLMClient(
        [
            LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})),
            LLMResponse(text="Sugar 1kg is running low (1 left)."),
        ]
    )

    def make_client(config: Config) -> StoreClient:
        client = _TrackingStoreClient(_mock_transport())
        client_box.append(client)
        return client

    exit_code = main(
        ["ask", "What is running low?"],
        config_loader=_test_config,
        store_client_factory=make_client,
        llm_factory=lambda config: fake_llm,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert stdout.getvalue().strip() == "Sugar 1kg is running low (1 left)."
    assert len(client_box) == 1
    assert client_box[0].closed is True


def test_cli_ask_with_trace_prints_tool_steps_and_outcome() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    fake_llm = FakeLLMClient(
        [
            LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})),
            LLMResponse(text="Sugar 1kg is low (1 left)."),
        ]
    )

    exit_code = main(
        ["ask", "What is low?", "--trace"],
        config_loader=_test_config,
        store_client_factory=lambda config: _TrackingStoreClient(_mock_transport()),
        llm_factory=lambda config: fake_llm,
        stdout=stdout,
        stderr=stderr,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "[outcome=answered, steps=1]" in output
    assert "step 1: get_low_stock({})" in output
    assert "Sugar 1kg" in output
    assert "Sugar 1kg is low (1 left)." in output


def test_cli_chat_is_stateless_supports_trace_and_exits_cleanly() -> None:
    client_box: list[_TrackingStoreClient] = []
    stdin = io.StringIO("What is low?\n\nWho won the cricket match?\nexit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    fake_llm = FakeLLMClient(
        [
            LLMResponse(tool_call=ToolCall(name="get_low_stock", arguments={})),
            LLMResponse(text="Sugar 1kg is low (1 left)."),
            LLMResponse(text="I only help with StoreStock store questions."),
        ]
    )

    def make_client(config: Config) -> StoreClient:
        client = _TrackingStoreClient(_mock_transport())
        client_box.append(client)
        return client

    exit_code = main(
        ["chat", "--trace"],
        config_loader=_test_config,
        store_client_factory=make_client,
        llm_factory=lambda config: fake_llm,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "Sugar 1kg is low (1 left)." in output
    assert "[outcome=refused, steps=0]" in output
    assert len(fake_llm.calls) == 3
    assert len(fake_llm.calls[2].messages) == 1
    assert fake_llm.calls[2].messages[0].content == "Who won the cricket match?"
    assert client_box[0].closed is True


def test_cli_chat_handles_eof_and_keyboard_interrupt() -> None:
    eof_out = io.StringIO()
    eof_code = main(
        ["chat"],
        config_loader=_test_config,
        store_client_factory=lambda config: _TrackingStoreClient(_mock_transport()),
        llm_factory=lambda config: FakeLLMClient([]),
        stdin=io.StringIO(""),
        stdout=eof_out,
        stderr=io.StringIO(),
    )
    assert eof_code == 0

    class _InterruptStdin(io.StringIO):
        def readline(self, size: int | None = -1) -> str:
            raise KeyboardInterrupt

    int_client_box: list[_TrackingStoreClient] = []

    def make_client(config: Config) -> StoreClient:
        client = _TrackingStoreClient(_mock_transport())
        int_client_box.append(client)
        return client

    int_code = main(
        ["chat"],
        config_loader=_test_config,
        store_client_factory=make_client,
        llm_factory=lambda config: FakeLLMClient([]),
        stdin=_InterruptStdin(),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert int_code == 0
    assert int_client_box[0].closed is True


def test_cli_exits_nonzero_on_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("STORESTOCK_USERNAME", "STORESTOCK_PASSWORD", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(["ask", "What is low?"], stdout=stdout, stderr=stderr)

    assert exit_code == 1
    assert stdout.getvalue() == ""
    err_text = stderr.getvalue()
    assert "Configuration error:" in err_text
    assert "Traceback" not in err_text


def test_cli_exits_nonzero_on_storestock_startup_error_and_closes_client() -> None:
    client_box: list[_TrackingStoreClient] = []
    stdout = io.StringIO()
    stderr = io.StringIO()

    def make_client(config: Config) -> StoreClient:
        client = _TrackingStoreClient(_mock_transport(fail_users_me=True))
        client_box.append(client)
        return client

    exit_code = main(
        ["ask", "What is low?"],
        config_loader=_test_config,
        store_client_factory=make_client,
        llm_factory=lambda config: FakeLLMClient([]),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    err_text = stderr.getvalue()
    assert "StoreStock error:" in err_text
    assert "Traceback" not in err_text
    assert client_box[0].closed is True


def test_python_module_entrypoint_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "storeassist", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "ask" in completed.stdout
    assert "chat" in completed.stdout
