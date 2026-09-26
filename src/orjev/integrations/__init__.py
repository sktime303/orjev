"""Adapters that merge a RoutingPlan into caller-owned inference requests."""

from .adapters import (
    apply_to_httpx_payload,
    apply_to_langchain_kwargs,
    apply_to_openai_kwargs,
    apply_to_openrouter_kwargs,
)

__all__ = [
    "apply_to_httpx_payload",
    "apply_to_langchain_kwargs",
    "apply_to_openai_kwargs",
    "apply_to_openrouter_kwargs",
]
