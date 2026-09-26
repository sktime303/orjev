"""Deterministic candidate and provider option construction."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .capabilities import model_satisfies, reasoning_options
from .types import (
    CandidateModel,
    CandidateProvider,
    ModelConfigOption,
    ModelPoolPolicy,
    ReasoningConfig,
    RoutingConstraints,
)


def filter_models(
    models: Iterable[CandidateModel],
    *,
    constraints: RoutingConstraints,
    max_models: int = 8,
) -> tuple[list[CandidateModel], list[dict[str, str]]]:
    patterns = constraints.candidate_models
    selected: list[CandidateModel] = []
    rejected: list[dict[str, str]] = []
    for model in sorted(models, key=lambda item: item.id):
        if patterns and not model_matches_patterns(model.id, patterns):
            continue
        if model.id.endswith(":batch"):
            rejected.append({"id": model.id, "reason": "batch_variant"})
            continue
        if not model_satisfies(model, constraints):
            rejected.append({"id": model.id, "reason": "model_capability"})
            continue
        selected.append(model)
        if len(selected) >= max_models:
            break
    return selected, rejected


def model_matches_patterns(model_id: str, patterns: Iterable[str]) -> bool:
    """Match a model id against any caller-supplied full-id regex."""
    for pattern in patterns:
        try:
            if re.fullmatch(pattern, model_id) is not None:
                return True
        except re.error:
            if pattern == model_id:
                return True
    return False


def enumerate_model_options(
    models: Iterable[CandidateModel],
    *,
    route: frozenset[str],
    current_model: str | None,
    current_reasoning: ReasoningConfig | None,
    policy: ModelPoolPolicy,
) -> list[ModelConfigOption]:
    result: list[ModelConfigOption] = []
    for model in sorted(models, key=lambda item: item.id):
        if "model" not in route and current_model and model.id != current_model:
            continue
        values = reasoning_options(model) if "reasoning" in route else (current_reasoning,)
        if "reasoning" not in route and current_reasoning is None:
            values = (None,)
        for reasoning in values:
            if reasoning is None:
                suffix = "off"
            else:
                suffix = str(reasoning.get("effort") or "on")
            short_id = model.id.rsplit("/", 1)[-1]
            result.append(
                ModelConfigOption(
                    key=f"{short_id}__{suffix}",
                    model_id=model.id,
                    reasoning=reasoning,
                )
            )
            if len(result) >= policy.max_options:
                return result
    return result


JEV_MAX_CHOICE_OPTIONS = 255


def provider_choice_set(
    providers: Iterable[CandidateProvider], *, limit: int = JEV_MAX_CHOICE_OPTIONS
) -> list[CandidateProvider]:
    ordered = sorted(providers, key=lambda item: item.tag)
    return ordered[: max(1, min(limit, JEV_MAX_CHOICE_OPTIONS))]


def compile_provider(
    eligible: Iterable[CandidateProvider],
    probabilities: dict[str, float],
    *,
    choice: str | None = None,
    probability_floor: float = 0.01,
    max_order: int = 5,
    constraints: RoutingConstraints | None = None,
) -> dict[str, object]:
    providers = sorted(eligible, key=lambda item: item.tag)
    only = [provider.tag for provider in providers]
    ranked = sorted(
        providers,
        key=lambda item: (-float(probabilities.get(item.tag, 0.0)), item.tag),
    )
    order = [
        provider.tag
        for provider in ranked
        if float(probabilities.get(provider.tag, 0.0)) >= probability_floor
    ][:max_order]
    if choice and choice in only:
        order = [choice] + [tag for tag in order if tag != choice]
    if not order and choice and choice in only:
        order = [choice]
    elif not order and ranked:
        order = [ranked[0].tag]
    result: dict[str, object] = {
        "only": only,
        "order": order,
        "allow_fallbacks": True,
    }
    if constraints and constraints.require_parameters is not None:
        result["require_parameters"] = constraints.require_parameters
    elif constraints and constraints.require_tools:
        result["require_parameters"] = True
    return result
