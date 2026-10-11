"""Thin HTTP client for the FastAPI backend. The dashboard never touches the database directly."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ApiError(RuntimeError):
    """Raised with a readable message when the API returns an error or is unreachable."""


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = httpx.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Cannot reach the API at {self.base_url} ({exc.__class__.__name__}). Is it running?"
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if isinstance(detail, list):  # Pydantic validation errors
                detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
            raise ApiError(f"API error {response.status_code}: {detail}")
        return response.json()

    def get(self, path: str, **params):
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, payload: dict):
        return self._request("POST", path, json=payload)

    def put(self, path: str, payload: dict):
        return self._request("PUT", path, json=payload)
