"""OpenRouter endpoint discovery and capability normalization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import quote

from .catalog import CatalogCache
from .errors import MetadataError
from .http import AsyncHttpClient
from .types import CandidateProvider


class EndpointClient:
    def __init__(
        self,
        http: AsyncHttpClient,
        *,
        cache: CatalogCache | None = None,
    ) -> None:
        self.http = http
        self.cache = cache or CatalogCache()

    async def alist_for_model(self, model_id: str) -> list[CandidateProvider]:
        key = f"endpoints:{model_id}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        path = "/".join(quote(part, safe="") for part in model_id.split("/"))
        payload = await self.http.get_json(f"/api/v1/models/{path}/endpoints")
        providers = normalize_endpoints(_items(payload))
        self.cache.put(key, providers)
        return providers


def normalize_endpoints(rows: Iterable[dict[str, Any]]) -> list[CandidateProvider]:
    providers_by_tag: dict[str, CandidateProvider] = {}
    for row in rows:
        tag = str(row.get("tag") or row.get("provider_name") or "").strip()
        if not tag:
            continue
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
        supports = row.get("supported_parameters") or row.get("supports") or []
        if isinstance(supports, dict):
            supports = [key for key, value in supports.items() if value]
        supports_set = {str(item).lower() for item in supports or []}
        provider = CandidateProvider(
                tag=tag,
                provider_name=str(row.get("provider_name") or row.get("name") or tag),
                latency_p50_s=_number(performance.get("latency_p50_s", row.get("latency_p50_s"))),
                throughput_p50=_number(
                    performance.get("throughput_p50", row.get("throughput_p50"))
                ),
                price_prompt_per_million=_per_million(
                    pricing.get("prompt", row.get("price_prompt"))
                ),
                price_completion_per_million=_per_million(
                    pricing.get("completion", row.get("price_completion"))
                ),
                uptime_5m=_number(performance.get("uptime_5m", row.get("uptime_5m"))),
                supports_tools=(
                    bool(row["supports_tools"])
                    if "supports_tools" in row
                    else ("tools" in supports_set or "function_calling" in supports_set)
                ),
                supports_reasoning=(
                    bool(row["supports_reasoning"])
                    if "supports_reasoning" in row
                    else ("reasoning" in supports_set)
                ),
                quantization=str(row["quantization"]) if row.get("quantization") else None,
                context_length=_integer(row.get("context_length")),
            )
        # The endpoint API can expose multiple variants with the same routing
        # tag (for example quantization variants). OpenRouter's `only` accepts
        # tags, so compile a stable unique tag set and retain the first stats
        # record returned for that tag.
        providers_by_tag.setdefault(tag, provider)
    return sorted(providers_by_tag.values(), key=lambda item: item.tag)


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data", payload.get("endpoints", []))
        if isinstance(data, dict):
            data = data.get("endpoints", [])
    else:
        data = payload
    if not isinstance(data, list):
        raise MetadataError("OpenRouter endpoints response did not contain a list")
    return [row for row in data if isinstance(row, dict)]


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _per_million(value: Any) -> float | None:
    number = _number(value)
    return number * 1_000_000 if number is not None else None
