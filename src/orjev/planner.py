"""Sequential model/reasoning then provider Jev orchestration."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from typing import Any

from .candidates import (
    compile_provider,
    enumerate_model_options,
    filter_models,
    provider_choice_set,
)
from .capabilities import eligible_providers, reasoning_options
from .catalog import CatalogCache, CatalogClient
from .endpoints import EndpointClient
from .env import load_dotenv
from .errors import OrjevError
from .http import AsyncHttpClient
from .jev import JevClient, JevConfig
from .privacy import Privacy
from .telemetry import TelemetryHook, TraceBuilder, emit
from .types import (
    CandidateModel,
    CandidateProvider,
    JevAnswer,
    JsonValue,
    ModelConfigOption,
    RouteRequest,
    RoutingConstraints,
    RoutingPlan,
)

OR_BASE_URL = "https://openrouter.ai"


class Router:
    """Plan OpenRouter request fields without performing chat inference."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = OR_BASE_URL,
        http: AsyncHttpClient | None = None,
        cache: CatalogCache | None = None,
        jev_config: JevConfig | None = None,
        privacy: Privacy | None = None,
        telemetry_hook: TelemetryHook | None = None,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        self.http = http or AsyncHttpClient(
            base_url=base_url,
            api_key=api_key,
            referer=referer,
            title=title,
        )
        cache = cache or CatalogCache()
        self.catalog = CatalogClient(self.http, cache=cache)
        self.endpoints = EndpointClient(self.http, cache=cache)
        self.jev = JevClient(self.http, config=jev_config)
        self.privacy = privacy or Privacy()
        self.telemetry_hook = telemetry_hook

    @classmethod
    def from_env(cls, **overrides: Any) -> Router:
        load_dotenv()
        settings: dict[str, Any] = {
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "base_url": os.environ.get("OPENROUTER_BASE_URL", OR_BASE_URL),
            "referer": os.environ.get("ORJEV_REFERER"),
            "title": os.environ.get("ORJEV_TITLE", "Orjev"),
            "jev_config": JevConfig(
                model=os.environ.get("JEV_MODEL", "typesafe/jev-1.13"),
                allow_moving_alias=os.environ.get("ORJEV_ALLOW_MOVING_JEV", "").lower()
                in {"1", "true", "yes", "on"},
            ),
        }
        settings.update(overrides)
        return cls(**settings)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def plan(self, request: RouteRequest, *, dry_run: bool = False) -> RoutingPlan:
        trace = TraceBuilder()
        route = frozenset(request.route)
        stages: list[JevAnswer] = []
        try:
            selected_models, rejected = await self.candidate_models(request)
            trace.candidate_rejections.extend(rejected)
            if not selected_models:
                return self._fallback(
                    request,
                    trace,
                    "no eligible catalog models",
                    dry_run=dry_run,
                )
            model_map = {model.id: model for model in selected_models}
            options = enumerate_model_options(
                selected_models,
                route=route,
                current_model=request.current_model,
                current_reasoning=request.current_reasoning,
                policy=request.pool,
            )
            selected_option: ModelConfigOption | None = None
            dry_payloads: list[dict[str, JsonValue]] = []
            joint_provider: CandidateProvider | None = None
            joint_eligible: list[CandidateProvider] = []
            joint_probabilities: dict[str, float] = {}
            joint_route = (
                "model" not in route
                and "reasoning" in route
                and "providers" in route
            )

            if joint_route:
                selected_option = self._pinned_option(request, selected_models)
                if selected_option is None:
                    return self._fallback(
                        request,
                        trace,
                        "current model is required for joint routing",
                        dry_run=dry_run,
                    )
                selected_model = model_map.get(selected_option.model_id)
                if selected_model is None:
                    return self._fallback(
                        request, trace, "current model is unavailable", dry_run=dry_run
                    )
                joint_options = self._joint_options(selected_model, request, trace)
                if not joint_options:
                    return self._fallback(
                        request,
                        trace,
                        "no eligible reasoning/provider combinations",
                        dry_run=dry_run,
                    )
                state = self._base_state(request, selected_models)
                provider_options: dict[str, dict[str, JsonValue]] = {}
                for _key, option, providers in joint_options:
                    effort = "off" if option.reasoning is None else str(
                        option.reasoning.get("effort") or "on"
                    )
                    provider_options.setdefault(
                        effort,
                        {
                            "reasoning": option.reasoning,
                            "providers": [item.to_dict() for item in providers],
                        },
                    )
                state["provider_options"] = [
                    {"reasoning_key": key, **value}
                    for key, value in provider_options.items()
                ]
                question = self._joint_question(joint_options)
                state = self.privacy.bounded_state(state)
                dry_payloads.append({"state": state, "questions": question})
                if len(joint_options) == 1:
                    key, selected_option, joint_eligible = joint_options[0]
                    joint_provider = next(
                        item for item in joint_eligible if key.endswith(f"__{item.tag}")
                    )
                    joint_probabilities = {joint_provider.tag: 1.0}
                elif dry_run:
                    return self._dry_plan(request, trace, dry_payloads)
                else:
                    started = time.perf_counter()
                    answer = await self.jev.adecide(state=state, questions=question)
                    trace.mark("jev_reasoning_provider", started)
                    stages.append(answer)
                    selected_joint = next(
                        (item for item in joint_options if item[0] == answer.choice), None
                    )
                    if selected_joint is None:
                        return self._fallback(
                            request,
                            trace,
                            "invalid joint reasoning/provider choice",
                            stages=stages,
                            selected_model=selected_model.id,
                        )
                    key, selected_option, joint_eligible = selected_joint
                    joint_provider = next(
                        item for item in joint_eligible if key.endswith(f"__{item.tag}")
                    )
                    joint_probabilities = {
                        provider.tag: answer.probabilities.get(
                            f"{key.split('__', 1)[0]}__{provider.tag}", 0.0
                        )
                        for provider in joint_eligible
                    }
            elif "model" in route or "reasoning" in route:
                if not options:
                    return self._fallback(
                        request, trace, "no eligible model configuration options", dry_run=dry_run
                    )
                state = self._base_state(request, selected_models)
                question = self._model_question(options)
                state = self.privacy.bounded_state(state)
                dry_payloads.append({"state": state, "questions": question})
                if len(options) == 1:
                    selected_option = options[0]
                elif dry_run:
                    return self._dry_plan(request, trace, dry_payloads)
                else:
                    started = time.perf_counter()
                    answer = await self.jev.adecide(state=state, questions=question)
                    trace.mark("jev_model_config", started)
                    stages.append(answer)
                    selected_option = self._validated_option(answer, options)
                    if selected_option is None:
                        return self._fallback(
                            request,
                            trace,
                            "invalid model configuration choice",
                            stages=stages,
                            dry_run=dry_run,
                        )
            else:
                selected_option = self._pinned_option(request, selected_models)
                if selected_option is None:
                    return self._fallback(
                        request, trace, "current model is unavailable", dry_run=dry_run
                    )

            selected_model = model_map.get(selected_option.model_id)
            if selected_model is None:
                return self._fallback(request, trace, "selected model disappeared", stages=stages)
            reasoning = selected_option.reasoning
            provider_body: dict[str, JsonValue] | None = None
            selected_provider: str | None = None
            if joint_route:
                selected_provider = joint_provider.tag if joint_provider else None
                provider_body = compile_provider(
                    joint_eligible,
                    joint_probabilities,
                    choice=joint_provider.tag if joint_provider else None,
                    probability_floor=request.pool.provider_probability_floor,
                    constraints=request.hard_constraints,
                )
            elif "providers" in route:
                eligible, rejected_providers = eligible_providers(
                    selected_model, reasoning, request.hard_constraints
                )
                trace.candidate_rejections.extend(
                    {"id": item["tag"], "reason": item["reason"]} for item in rejected_providers
                )
                if not eligible:
                    return self._fallback(
                        request,
                        trace,
                        "no eligible providers for selected model",
                        stages=stages,
                        selected_model=selected_model.id,
                        selected_reasoning=reasoning,
                    )
                choice_providers = provider_choice_set(eligible)
                state = self._provider_state(
                    request,
                    selected_model.id,
                    reasoning,
                    eligible,
                )
                criteria = _choice_criteria(item.tag for item in choice_providers)
                question = {
                    "provider": {
                        "type": "choice",
                        "instructions": (
                            "Choose the best primary provider tag for selection.model_id "
                            "and selection.reasoning using state.providers. Return "
                            "probabilities for the listed tags."
                        ),
                        "criteria": criteria,
                    }
                }
                state = self.privacy.bounded_state(state)
                dry_payloads.append({"state": state, "questions": question})
                if dry_run:
                    return self._dry_plan(request, trace, dry_payloads)
                if len(criteria) == 1:
                    selected_provider = choice_providers[0].tag
                    provider_body = compile_provider(
                        eligible,
                        {selected_provider: 1.0},
                        choice=selected_provider,
                        probability_floor=request.pool.provider_probability_floor,
                        constraints=request.hard_constraints,
                    )
                else:
                    started = time.perf_counter()
                    answer = await self.jev.adecide(state=state, questions=question)
                    trace.mark("jev_provider", started)
                    stages.append(answer)
                    if answer.choice not in {item.tag for item in eligible}:
                        return self._fallback(
                            request,
                            trace,
                            "invalid provider choice",
                            stages=stages,
                            selected_model=selected_model.id,
                            selected_reasoning=reasoning,
                        )
                    selected_provider = answer.choice
                    provider_body = compile_provider(
                        eligible,
                        answer.probabilities,
                        choice=answer.choice,
                        probability_floor=request.pool.provider_probability_floor,
                        constraints=request.hard_constraints,
                    )
            trace_obj = trace.trace(
                stages=tuple(stages),
                selected_model=selected_model.id,
                selected_reasoning=reasoning,
                selected_provider=selected_provider,
            )
            emit(trace_obj, self.telemetry_hook)
            return RoutingPlan(
                model=selected_model.id,
                reasoning=reasoning,
                provider=provider_body,
                trace=trace_obj,
            )
        except Exception as exc:
            reason = (
                str(exc)
                if isinstance(exc, OrjevError)
                else f"{type(exc).__name__}: {exc}"
            )
            return self._fallback(request, trace, reason, stages=stages, dry_run=dry_run)

    def plan_sync(self, request: RouteRequest, *, dry_run: bool = False) -> RoutingPlan:
        return asyncio.run(self.plan(request, dry_run=dry_run))

    async def candidate_models(
        self, request: RouteRequest
    ) -> tuple[list[CandidateModel], list[dict[str, str]]]:
        """Return the hard-filtered model shortlist used before Jev decides.

        The returned models are the same catalog candidates that ``plan`` sends
        to Jev.  Provider constraints are checked against endpoint metadata
        before the shortlist is capped, so a model is returned only when at
        least one usable endpoint remains.
        """
        models = await self.catalog.alist_models()
        constraints = request.hard_constraints
        route = frozenset(request.route)
        capability_constrained = constraints.capability_filters_active()
        scan_limit = (
            len(models)
            if capability_constrained and "providers" in route
            else request.pool.max_models
        )
        selected, rejected = filter_models(
            models,
            constraints=constraints,
            max_models=scan_limit,
        )
        if "providers" not in route:
            return selected[: request.pool.max_models], rejected

        enriched = await self._attach_endpoints(selected)
        eligible_models: list[CandidateModel] = []
        for model in enriched:
            eligible, provider_rejections = eligible_providers(
                model,
                None,
                constraints,
            )
            if not eligible:
                rejected.append(
                    {
                        "id": model.id,
                        "reason": (
                            provider_rejections[0]["reason"]
                            if provider_rejections
                            else "provider_constraint"
                        ),
                    }
                )
                continue
            eligible_models.append(model)
            if len(eligible_models) >= request.pool.max_models:
                break
        return eligible_models, rejected

    async def _attach_endpoints(self, models: list[CandidateModel]) -> list[CandidateModel]:
        pool = models
        results = await asyncio.gather(
            *(self.endpoints.alist_for_model(model.id) for model in pool),
            return_exceptions=True,
        )
        description_rows = await asyncio.gather(
            *(self.catalog.afetch_model(model.id) for model in pool),
            return_exceptions=True,
        )
        enriched: list[CandidateModel] = []
        for model, value, row in zip(pool, results, description_rows, strict=True):
            providers = value if isinstance(value, list) else []
            description = model.description
            if isinstance(row, dict):
                full = row.get("description")
                if isinstance(full, str) and full.strip():
                    description = full.strip()
            enriched.append(
                CandidateModel(
                    **{
                        **replace(model, description=description).__dict__,
                        "providers": tuple(providers),
                        "endpoint_summary": _endpoint_summary(providers),
                    }
                )
            )
        return enriched

    def _base_state(
        self, request: RouteRequest, models: list[CandidateModel]
    ) -> dict[str, JsonValue]:
        return {
            "request": self.privacy.request_state(request),
            "hints": self.privacy.hints(request.hints),
            "models": [model.to_dict() for model in models],
        }

    def _provider_state(
        self,
        request: RouteRequest,
        model_id: str,
        reasoning: dict[str, JsonValue] | None,
        providers: list[CandidateProvider],
    ) -> dict[str, JsonValue]:
        return {
            "request": self.privacy.request_state(request),
            "hints": self.privacy.hints(request.hints),
            "selection": {
                "model_id": model_id,
                "reasoning": reasoning,
            },
            "providers": [item.to_dict() for item in providers],
        }

    def _model_question(
        self, options: list[ModelConfigOption]
    ) -> dict[str, JsonValue]:
        return {
            "model_config": {
                "type": "choice",
                "instructions": (
                    "Pick the best model and reasoning effort for this request "
                    "using state.models and the user message. Prefer higher "
                    "reasoning only for complex or high-stakes work."
                ),
                "criteria": _choice_criteria(item.key for item in options),
            }
        }

    def _joint_options(
        self,
        model: CandidateModel,
        request: RouteRequest,
        trace: TraceBuilder,
    ) -> list[tuple[str, ModelConfigOption, list[CandidateProvider]]]:
        result: list[tuple[str, ModelConfigOption, list[CandidateProvider]]] = []
        for reasoning in reasoning_options(model):
            eligible, rejected = eligible_providers(
                model, reasoning, request.hard_constraints
            )
            trace.candidate_rejections.extend(
                {"id": item["tag"], "reason": item["reason"]} for item in rejected
            )
            if not eligible:
                continue
            for provider in provider_choice_set(eligible):
                effort = "off" if reasoning is None else str(reasoning.get("effort") or "on")
                key = f"{effort}__{provider.tag}"
                option = ModelConfigOption(
                    key=key,
                    model_id=model.id,
                    reasoning=reasoning,
                )
                result.append((key, option, eligible))
                if len(result) >= request.pool.max_options:
                    return result
        return result

    def _joint_question(
        self, options: list[tuple[str, ModelConfigOption, list[CandidateProvider]]]
    ) -> dict[str, JsonValue]:
        criteria = _choice_criteria(key for key, _, _ in options)
        return {
            "route_config": {
                "type": "choice",
                "instructions": (
                    "Choose one reasoning effort and provider combination for "
                    "the pinned model. Return probabilities for the listed "
                    "combination keys."
                ),
                "criteria": criteria,
            }
        }

    def _validated_option(
        self, answer: JevAnswer, options: list[ModelConfigOption]
    ) -> ModelConfigOption | None:
        by_key = {item.key: item for item in options}
        return by_key.get(answer.choice or "")

    def _pinned_option(
        self, request: RouteRequest, models: list[CandidateModel]
    ) -> ModelConfigOption | None:
        model_id = request.current_model or (models[0].id if models else None)
        model = next((item for item in models if item.id == model_id), None)
        if model is None:
            return None
        return ModelConfigOption(
            key="pinned",
            model_id=model.id,
            reasoning=request.current_reasoning,
        )

    def _fallback(
        self,
        request: RouteRequest,
        trace: TraceBuilder,
        reason: str,
        *,
        stages: list[JevAnswer] | None = None,
        selected_model: str | None = None,
        selected_reasoning: dict[str, JsonValue] | None = None,
        dry_run: bool = False,
    ) -> RoutingPlan:
        model = selected_model or request.current_model
        trace_obj = trace.trace(
            stages=tuple(stages or ()),
            selected_model=model,
            selected_reasoning=selected_reasoning or request.current_reasoning,
            fallback=True,
            fallback_reason=reason,
        )
        emit(trace_obj, self.telemetry_hook)
        provider: dict[str, JsonValue] | None = None
        if request.current_provider_only or request.current_provider_order:
            provider = {
                "only": list(request.current_provider_only),
                "order": list(request.current_provider_order),
                "allow_fallbacks": True,
            }
        return RoutingPlan(
            model=model,
            reasoning=selected_reasoning or request.current_reasoning,
            provider=provider,
            trace=trace_obj,
            dry_run={} if dry_run else None,
        )

    def _dry_plan(
        self, request: RouteRequest, trace: TraceBuilder, payloads: list[dict[str, JsonValue]]
    ) -> RoutingPlan:
        trace_obj = trace.trace(fallback=True, fallback_reason="dry_run")
        return RoutingPlan(
            model=request.current_model,
            reasoning=request.current_reasoning,
            trace=trace_obj,
            dry_run={"stages": payloads},
        )


def _endpoint_summary(providers: list[CandidateProvider]) -> dict[str, JsonValue]:
    latencies = [item for item in providers if item.latency_p50_s is not None]
    prices = [item for item in providers if item.price_prompt_per_million is not None]
    return {
        "count": len(providers),
        "cheapest_tag": min(prices, key=lambda item: item.price_prompt_per_million or 0).tag
        if prices
        else None,
        "fastest_tag": min(latencies, key=lambda item: item.latency_p50_s or 999999).tag
        if latencies
        else None,
        "min_price_prompt_per_million": min(
            (item.price_prompt_per_million for item in prices), default=None
        ),
        "median_price_prompt_per_million": _median(
            [item.price_prompt_per_million for item in prices]
        ),
    }


def _median(values: list[float | None]) -> float | None:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2


def _choice_criteria(keys: Any) -> dict[str, JsonValue]:
    return {str(key): None for key in keys}
