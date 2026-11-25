from __future__ import annotations

from typing import Any

import httpx


class StoreClientError(Exception):
    pass


class StoreConnectionError(StoreClientError):
    pass


class StoreAuthError(StoreClientError):
    pass


class StoreNotFoundError(StoreClientError):
    pass


class StoreAPIError(StoreClientError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def paise_to_rupees(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        sign = "-" if value < 0 else ""
        rupees, paise = divmod(abs(value), 100)
        return f"{sign}{rupees}.{paise:02d}"
    return None


def convert_paise_fields(data: Any) -> Any:
    if isinstance(data, list):
        return [convert_paise_fields(item) for item in data]
    if isinstance(data, dict):
        return _convert_dict_paise_fields(data)
    return data


def _convert_dict_paise_fields(data: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if key.endswith("_paise"):
            converted[key] = value
            rupees = paise_to_rupees(value)
            if rupees is not None:
                converted[f"{key[:-6]}_rupees"] = rupees
        else:
            converted[key] = convert_paise_fields(value)
    return converted


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    return {key: value for key, value in params.items() if value is not None}


def _parse_json(response: httpx.Response, context: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise StoreAPIError(f"Invalid JSON returned from {context}.") from exc


class StoreClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._timeout = timeout
        self._access_token: str | None = None
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=transport,
        )

    def __enter__(self) -> StoreClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _login(self) -> str:
        if not self._username or not self._password:
            raise StoreAuthError("StoreStock username and password must be configured.")

        try:
            response = self._client.post(
                "/auth/login",
                json={"username": self._username, "password": self._password},
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StoreConnectionError(f"StoreStock login connection failed: {exc}") from exc

        if response.status_code in (401, 403, 422):
            raise StoreAuthError(f"StoreStock login failed with HTTP {response.status_code}.")
        if response.status_code >= 400:
            raise StoreAPIError(
                f"StoreStock login failed with HTTP {response.status_code}.",
                status_code=response.status_code,
            )

        payload = _parse_json(response, "POST /auth/login")
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise StoreAPIError("StoreStock login response missing access_token.")

        self._access_token = token
        return token

    def _auth_headers(self) -> dict[str, str]:
        if not self._access_token:
            self._login()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _send_get(self, path: str, params: dict[str, Any] | None) -> httpx.Response:
        headers = self._auth_headers()
        try:
            return self._client.get(
                path,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StoreConnectionError(f"StoreStock GET {path} failed: {exc}") from exc

    def _raise_for_error(self, response: httpx.Response, path: str) -> None:
        if response.status_code in (401, 403):
            raise StoreAuthError(
                f"Authorization failed for GET {path}: HTTP {response.status_code}."
            )
        if response.status_code == 404:
            raise StoreNotFoundError(f"Resource not found at GET {path}.")
        if response.status_code >= 400:
            raise StoreAPIError(
                f"StoreStock GET {path} failed with HTTP {response.status_code}.",
                status_code=response.status_code,
            )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        cleaned = _clean_params(params)
        response = self._send_get(path, cleaned)
        if response.status_code == 401:
            self._access_token = None
            self._login()
            response = self._send_get(path, cleaned)

        self._raise_for_error(response, path)
        payload = _parse_json(response, f"GET {path}")
        return convert_paise_fields(payload)
