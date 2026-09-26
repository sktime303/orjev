"""Hard capability filters and reasoning metadata helpers."""

from __future__ import annotations

from .types import CandidateModel, CandidateProvider, ReasoningConfig, RoutingConstraints


def reasoning_efforts(model: CandidateModel) -> tuple[str, ...]:
    """Return catalog-declared efforts only; never infer them from the name."""
    value = model.reasoning
    if not value:
        return ()
    efforts = value.get("supported_efforts") or value.get("efforts") or value.get("supported")
    if isinstance(efforts, dict):
        efforts = [key for key, enabled in efforts.items() if enabled]
    if isinstance(efforts, str):
        efforts = [efforts]
    if not isinstance(efforts, (list, tuple, set)):
        efforts = []
    result = tuple(dict.fromkeys(str(item).strip().lower() for item in efforts or () if str(item).strip()))
    return result


def supports_reasoning(model: CandidateModel) -> bool:
    return bool(model.reasoning is not None or "reasoning" in model.supported_parameters)


def reasoning_options(model: CandidateModel) -> tuple[ReasoningConfig | None, ...]:
    """Include the explicit off option and all catalog-supported efforts."""
    efforts = reasoning_efforts(model)
    if not efforts and not supports_reasoning(model):
        return (None,)
    return (None,) + tuple({"effort": effort} for effort in efforts)


def model_satisfies(model: CandidateModel, constraints: RoutingConstraints) -> bool:
    required_context = constraints.min_context_tokens or 0
    if model.context_length is not None and model.context_length < required_context:
        return False
    if required_context and model.context_length is None:
        return False
    if constraints.intelligence_index_range is not None:
        lower, upper = constraints.intelligence_index_range
        if model.intelligence_index is None:
            return False
        if lower is not None and model.intelligence_index < lower:
            return False
        if upper is not None and model.intelligence_index > upper:
            return False
    if constraints.requires_vision and not any(
        "image" in modality for modality in model.modalities
    ):
        return False
    if constraints.require_tools:
        if "tools" not in model.supported_parameters and "tool_choice" not in model.supported_parameters:
            return False
    if constraints.require_reasoning and not supports_reasoning(model):
        return False
    return True


def provider_satisfies(
    provider: CandidateProvider,
    *,
    model: CandidateModel,
    reasoning: ReasoningConfig | None,
    constraints: RoutingConstraints,
) -> tuple[bool, str | None]:
    if constraints.allowed_provider_tags and provider.tag not in constraints.allowed_provider_tags:
        return False, "provider_not_allowed"
    if provider.tag in constraints.denied_provider_tags:
        return False, "provider_denied"
    if constraints.max_latency_s is not None and (
        provider.latency_p50_s is None or provider.latency_p50_s > constraints.max_latency_s
    ):
        return False, "latency_limit"
    if constraints.min_uptime_5m is not None and (
        provider.uptime_5m is None or provider.uptime_5m < constraints.min_uptime_5m
    ):
        return False, "uptime_limit"
    if constraints.max_prompt_usd_per_million is not None and (
        provider.price_prompt_per_million is None
        or provider.price_prompt_per_million > constraints.max_prompt_usd_per_million
    ):
        return False, "prompt_price_limit"
    if constraints.max_completion_usd_per_million is not None and (
        provider.price_completion_per_million is None
        or provider.price_completion_per_million > constraints.max_completion_usd_per_million
    ):
        return False, "completion_price_limit"
    if constraints.require_tools:
        if provider.supports_tools is not True:
            return False, "tools_unsupported"
    if reasoning is not None:
        if provider.supports_reasoning is not True:
            return False, "reasoning_unsupported"
    if model.context_length and provider.context_length:
        required = constraints.min_context_tokens or 0
        if provider.context_length < required:
            return False, "provider_context_limit"
    return True, None


def eligible_providers(
    model: CandidateModel,
    reasoning: ReasoningConfig | None,
    constraints: RoutingConstraints,
) -> tuple[list[CandidateProvider], list[dict[str, str]]]:
    eligible: list[CandidateProvider] = []
    rejected: list[dict[str, str]] = []
    for provider in model.providers:
        okay, reason = provider_satisfies(
            provider, model=model, reasoning=reasoning, constraints=constraints
        )
        if okay:
            eligible.append(provider)
        else:
            rejected.append({"tag": provider.tag, "reason": reason or "ineligible"})
    return eligible, rejected
