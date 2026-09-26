"""Async HTTP foundation with bounded metadata retries."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from typing import Any

import httpx

from .errors import MetadataError

DEFAULT_TIMEOUT = 20.0
USER_AGENT = "orjev/0.1 (+https://github.com/sktime303/orjev)"


def _headers(api_key: str | None, referer: str | None, title: str | None) -> dict[str, str]:
    result = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    if referer:
        result["HTTP-Referer"] = referer
    if title:
        result["X-Title"] = title
    return result


class AsyncHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
        referer: str | None = None,
        title: str | None = None,
        metadata_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._headers = _headers(api_key, referer, title)
        self._retries = max(0, metadata_retries)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self._request_json("GET", path, params=params, retry=True)

    async def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        return await self._request_json("POST", path, json=payload, retry=False)

    async def _request_json(self, method: str, path: str, *, retry: bool, **kwargs: Any) -> Any:
        attempts = self._retries + 1 if retry else 1
        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method, f"{self.base_url}/{path.lstrip('/')}", headers=self._headers, **kwargs
                )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                if attempt + 1 >= attempts:
                    raise MetadataError(f"{method} {path} failed: {exc}") from exc
                await asyncio.sleep(_delay(attempt))
        raise AssertionError("unreachable")


def _delay(attempt: int) -> float:
    return min(1.5, 0.15 * (2**attempt)) + random.uniform(0, 0.05)
