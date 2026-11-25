from dataclasses import dataclass
import os


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    storestock_url: str
    storestock_username: str
    storestock_password: str
    llm_api_key: str
    llm_model: str


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Missing or blank required environment variable: {name}")
    return value


def _optional_url_env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return (value or default).rstrip("/")


def load_config() -> Config:
    return Config(
        storestock_url=_optional_url_env("STORESTOCK_URL", "http://127.0.0.1:8000"),
        storestock_username=_require_env("STORESTOCK_USERNAME"),
        storestock_password=_require_env("STORESTOCK_PASSWORD"),
        llm_api_key=_require_env("LLM_API_KEY"),
        llm_model=_require_env("LLM_MODEL"),
    )
