"""OpenRouter model catalog discovery and normalization."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from .errors import MetadataError
from .http import AsyncHttpClient
from .types import CandidateModel, JsonValue


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class CatalogCache:
    """Process-local TTL cache; disk persistence is intentionally opt-in later."""

    def __init__(self, ttl_seconds: float = 900.0) -> None:
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._items: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None or item.expires_at < time.monotonic():
            return None
        return item.value

    def put(self, key: str, value: Any) -> None:
        self._items[key] = _CacheEntry(value, time.monotonic() + self.ttl_seconds)


class CatalogClient:
    def __init__(
        self,
        http: AsyncHttpClient,
        *,
        cache: CatalogCache | None = None,
    ) -> None:
        self.http = http
        self.cache = cache or CatalogCache()

    async def alist_models(self) -> list[CandidateModel]:
        cached = self.cache.get("models")
        if cached is not None:
            return cached
        payload = await self.http.get_json("/api/v1/models")
        models = normalize_models(_items(payload))
        self.cache.put("models", models)
        return models

    async def afetch_model(self, model_id: str) -> dict[str, Any] | None:
        key = f"model:{model_id}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        row = await _afetch_model_row(self.http, model_id)
        if row is not None:
            self.cache.put(key, row)
        return row


def normalize_models(rows: Iterable[dict[str, Any]]) -> list[CandidateModel]:
    result: list[CandidateModel] = []
    for row in rows:
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        architecture = _dict(row.get("architecture"))
        modalities = _modalities(row, architecture)
        reasoning = _reasoning(row)
        pricing = _dict(row.get("pricing"))
        benchmarks = _dict(row.get("benchmarks"))
        result.append(
            CandidateModel(
                id=model_id,
                name=str(row.get("name") or model_id),
                description=_text(row.get("description")),
                context_length=_int(row.get("context_length")),
                pricing_summary=_pricing_summary(pricing),
                supported_parameters=tuple(str(x) for x in row.get("supported_parameters", []) or []),
                reasoning=reasoning,
                modalities=modalities,
                architecture=_json_dict(architecture),
                benchmarks=_json_value(row.get("benchmarks")),
                intelligence_index=_intelligence_index(row, benchmarks),
            )
        )
    return sorted(result, key=lambda item: item.id)


def _intelligence_index(row: dict[str, Any], benchmarks: dict[str, Any]) -> float | None:
    value = row.get("intelligence_index")
    if value is None:
        artificial_analysis = benchmarks.get("artificial_analysis") or benchmarks.get(
            "artificial-analysis"
        )
        if isinstance(artificial_analysis, dict):
            value = artificial_analysis.get("intelligence_index")
    return _float(value)


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("models", []))
    else:
        data = payload
    if not isinstance(data, list):
        raise MetadataError("OpenRouter /models response did not contain a list")
    return [row for row in data if isinstance(row, dict)]


def _reasoning(row: dict[str, Any]) -> dict[str, JsonValue] | None:
    value = row.get("reasoning")
    if isinstance(value, dict):
        return _json_dict(value)
    if value is True:
        return {"supported": True}
    return None


def _modalities(row: dict[str, Any], architecture: dict[str, Any]) -> tuple[str, ...]:
    candidates = row.get("modalities") or architecture.get("input_modalities") or []
    if isinstance(candidates, dict):
        candidates = candidates.get("input", [])
    if isinstance(candidates, str):
        candidates = [candidates]
    return tuple(str(item) for item in candidates or [])


def _pricing_summary(pricing: dict[str, Any]) -> str:
    prompt = _float(pricing.get("prompt"))
    completion = _float(pricing.get("completion"))
    if prompt is None and completion is None:
        return ""
    return f"${prompt or 0:g} / M in · ${completion or 0:g} / M out"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_dict(value: dict[str, Any]) -> dict[str, JsonValue]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return _json_dict(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(value or "")


def _model_lookup_path(model_id: str) -> str:
    path = "/".join(quote(part, safe="") for part in model_id.split("/"))
    return f"/api/v1/model/{path}"


async def _afetch_model_row(
    http: AsyncHttpClient, model_id: str
) -> dict[str, Any] | None:
    try:
        payload = await http.get_json(_model_lookup_path(model_id))
    except MetadataError:
        return None
    return _single_model_row(payload)


def _single_model_row(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data", payload)
    return data if isinstance(data, dict) else None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
