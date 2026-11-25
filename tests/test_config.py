import pytest

from storeassist.config import ConfigError, load_config


def test_load_config_reads_required_vars_and_defaults_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STORESTOCK_URL", raising=False)
    monkeypatch.setenv("STORESTOCK_USERNAME", "owner")
    monkeypatch.setenv("STORESTOCK_PASSWORD", "secret")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")

    config = load_config()

    assert config.storestock_url == "http://127.0.0.1:8000"
    assert config.storestock_username == "owner"
    assert config.storestock_password == "secret"
    assert config.llm_api_key == "test-key"
    assert config.llm_model == "gemini-3.5-flash-lite"


def test_load_config_strips_trailing_slash_from_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STORESTOCK_URL", "http://127.0.0.1:8000/")
    monkeypatch.setenv("STORESTOCK_USERNAME", "owner")
    monkeypatch.setenv("STORESTOCK_PASSWORD", "secret")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")

    config = load_config()

    assert config.storestock_url == "http://127.0.0.1:8000"
    assert config.llm_model == "gemini-3.8-flash"


@pytest.mark.parametrize(
    "missing_var",
    ["STORESTOCK_USERNAME", "STORESTOCK_PASSWORD", "LLM_API_KEY", "LLM_MODEL"],
)
def test_load_config_raises_on_missing_or_blank_required_var(
    monkeypatch: pytest.MonkeyPatch,
    missing_var: str,
) -> None:
    monkeypatch.setenv("STORESTOCK_USERNAME", "owner")
    monkeypatch.setenv("STORESTOCK_PASSWORD", "secret")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")

    monkeypatch.delenv(missing_var, raising=False)
    with pytest.raises(ConfigError, match=missing_var):
        load_config()

    monkeypatch.setenv(missing_var, "   ")
    with pytest.raises(ConfigError, match=missing_var):
        load_config()
