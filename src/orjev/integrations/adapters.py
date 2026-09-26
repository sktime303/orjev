"""Framework-neutral request adapters.

Adapters only merge fields. They never choose policy, retry chat calls, or call
the model provider.
"""

from __future__ import annotations

from typing import Any

from ..types import JsonValue, RoutingPlan


def apply_to_openai_kwargs(
    kwargs: dict[str, Any], plan: RoutingPlan, *, preserve_model: bool = False
) -> dict[str, Any]:
    result = dict(kwargs)
    if plan.model and not preserve_model:
        result["model"] = plan.model
    extra = dict(result.get("extra_body") or {})
    extra.update(plan.to_openrouter())
    if preserve_model:
        extra.pop("model", None)
    result["extra_body"] = extra
    return result


def apply_to_openrouter_kwargs(
    kwargs: dict[str, Any], plan: RoutingPlan
) -> dict[str, Any]:
    result = dict(kwargs)
    result.update(plan.to_openrouter())
    return result


def apply_to_httpx_payload(
    payload: dict[str, JsonValue], plan: RoutingPlan
) -> dict[str, JsonValue]:
    result = dict(payload)
    result.update(plan.to_openrouter())
    return result


def apply_to_langchain_kwargs(
    kwargs: dict[str, Any], plan: RoutingPlan
) -> dict[str, Any]:
    result = dict(kwargs)
    result["model"] = plan.model or result.get("model")
    model_kwargs = dict(result.get("model_kwargs") or {})
    if plan.reasoning is not None:
        model_kwargs["reasoning"] = plan.reasoning
    if plan.provider is not None:
        model_kwargs["provider"] = plan.provider
    result["model_kwargs"] = model_kwargs
    return result
