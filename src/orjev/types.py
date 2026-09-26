"""Public data contract for Orjev.

The package deliberately uses immutable-ish dataclasses instead of binding the
caller to an inference SDK.  ``RoutingPlan.to_openrouter()`` is the only
OpenRouter-specific compilation step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
ReasoningConfig: TypeAlias = dict[str, JsonValue]
RouteDimension: TypeAlias = Literal["model", "reasoning", "provider", "providers"]


def _json(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ModelPoolPolicy:
    max_models: int = 8
    max_options: int = 24
    provider_probability_floor: float = 0.01


@dataclass(frozen=True)
class RoutingConstraints:
    max_prompt_usd_per_million: float | None = None
    max_completion_usd_per_million: float | None = None
    min_context_tokens: int | None = None
    intelligence_index_range: tuple[float | None, float | None] | None = None
    max_latency_s: float | None = None
    min_uptime_5m: float | None = None
    require_parameters: bool | None = None
    require_tools: bool = False
    require_reasoning: bool = False
    requires_vision: bool = False
    allowed_provider_tags: tuple[str, ...] = ()
    denied_provider_tags: tuple[str, ...] = ()
    candidate_models: tuple[str, ...] = ()

    @classmethod
    def from_value(
        cls, value: RoutingConstraints | dict[str, Any] | None
    ) -> RoutingConstraints:
        if isinstance(value, cls):
            return value
        data = value or {}
        return cls(
            max_prompt_usd_per_million=_float_or_none(
                data.get("max_prompt_usd_per_million")
            ),
            max_completion_usd_per_million=_float_or_none(
                data.get("max_completion_usd_per_million")
            ),
            min_context_tokens=_int_or_none(data.get("min_context_tokens")),
            intelligence_index_range=_float_range(
                data.get("intelligence_index_range", data.get("intelligence_index"))
            ),
            max_latency_s=_float_or_none(data.get("max_latency_s")),
            min_uptime_5m=_float_or_none(data.get("min_uptime_5m")),
            require_parameters=(
                bool(data["require_parameters"]) if "require_parameters" in data else None
            ),
            require_tools=bool(data.get("require_tools", False)),
            require_reasoning=bool(data.get("require_reasoning", False)),
            requires_vision=bool(data.get("requires_vision", False)),
            allowed_provider_tags=_strings(data.get("allowed_provider_tags")),
            denied_provider_tags=_strings(data.get("denied_provider_tags")),
            candidate_models=_strings(data.get("candidate_models")),
        )

    def capability_filters_active(self) -> bool:
        """True when any hard filter besides model-id patterns is set."""
        return replace(self, candidate_models=()) != RoutingConstraints()

@dataclass(frozen=True)
class RouteRequest:
    """Inputs for a route plan.

    ``candidate_models`` contains regular expressions matched against complete
    catalog model ids. An empty list leaves the catalog unrestricted unless
    ``constraints.candidate_models`` is set. Hard filters are ANDed.
    """

    messages: list[dict[str, JsonValue]]
    current_model: str | None = None
    current_reasoning: ReasoningConfig | None = None
    current_provider_only: tuple[str, ...] = ()
    current_provider_order: tuple[str, ...] = ()
    candidate_models: list[str] = field(default_factory=list)
    route: frozenset[RouteDimension] = frozenset({"model", "reasoning", "providers"})
    constraints: RoutingConstraints | dict[str, Any] | None = None
    hints: dict[str, JsonValue] | None = None
    pool: ModelPoolPolicy = field(default_factory=ModelPoolPolicy)

    def __post_init__(self) -> None:
        normalized = frozenset(
            "providers" if dimension == "provider" else dimension for dimension in self.route
        )
        object.__setattr__(self, "route", normalized)
        invalid = set(normalized) - {"model", "reasoning", "providers"}
        if invalid:
            raise ValueError(f"Unsupported route dimensions: {sorted(invalid)}")

    @property
    def hard_constraints(self) -> RoutingConstraints:
        base = RoutingConstraints.from_value(self.constraints)
        top_level = _strings(self.candidate_models)
        if not top_level:
            return base
        merged: list[str] = []
        seen: set[str] = set()
        for pattern in (*base.candidate_models, *top_level):
            if pattern not in seen:
                seen.add(pattern)
                merged.append(pattern)
        return replace(base, candidate_models=tuple(merged))

    @property
    def latest_message(self) -> str:
        for item in reversed(self.messages):
            if item.get("role") == "user":
                content = item.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""


@dataclass(frozen=True)
class ModelConfigOption:
    key: str
    model_id: str
    reasoning: ReasoningConfig | None


@dataclass(frozen=True)
class CandidateProvider:
    tag: str
    provider_name: str | None = None
    latency_p50_s: float | None = None
    throughput_p50: float | None = None
    price_prompt_per_million: float | None = None
    price_completion_per_million: float | None = None
    uptime_5m: float | None = None
    supports_tools: bool | None = None
    supports_reasoning: bool | None = None
    quantization: str | None = None
    context_length: int | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {key: _json(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class CandidateModel:
    id: str
    name: str
    description: str = ""
    context_length: int | None = None
    pricing_summary: str = ""
    supported_parameters: tuple[str, ...] = ()
    reasoning: dict[str, JsonValue] | None = None
    modalities: tuple[str, ...] = ()
    architecture: dict[str, JsonValue] = field(default_factory=dict)
    benchmarks: JsonValue | None = None
    providers: tuple[CandidateProvider, ...] = ()
    endpoint_summary: dict[str, JsonValue] = field(default_factory=dict)
    intelligence_index: float | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "context_length": self.context_length,
            "pricing_summary": self.pricing_summary,
            "supported_parameters": list(self.supported_parameters),
            "reasoning": _json(self.reasoning),
            "modalities": list(self.modalities),
            "architecture": _json(self.architecture),
            "benchmarks": _json(self.benchmarks),
            "intelligence_index": self.intelligence_index,
            "endpoint_summary": _json(self.endpoint_summary),
        }
        return result


@dataclass(frozen=True)
class JevAnswer:
    stage: str
    choice: str | None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    raw: dict[str, JsonValue] = field(default_factory=dict)
    request: dict[str, JsonValue] | None = None

    def to_dict(self, *, include_io: bool = False) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "stage": self.stage,
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }
        if include_io:
            result["request"] = _json(self.request)
            result["response"] = _json(self.raw)
        return result


@dataclass(frozen=True)
class DecisionTrace:
    trace_id: str
    stages: tuple[JevAnswer, ...] = ()
    selected_model: str | None = None
    selected_reasoning: ReasoningConfig | None = None
    selected_provider: str | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    candidate_rejections: tuple[dict[str, JsonValue], ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self, *, include_io: bool = False) -> dict[str, JsonValue]:
        return {
            "trace_id": self.trace_id,
            "stages": [stage.to_dict(include_io=include_io) for stage in self.stages],
            "selected_model": self.selected_model,
            "selected_reasoning": _json(self.selected_reasoning),
            "selected_provider": self.selected_provider,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "candidate_rejections": _json(list(self.candidate_rejections)),
            "timings_ms": _json(self.timings_ms),
        }


@dataclass(frozen=True)
class RoutingPlan:
    model: str | None
    reasoning: ReasoningConfig | None = None
    provider: dict[str, JsonValue] | None = None
    trace: DecisionTrace | None = None
    dry_run: dict[str, JsonValue] | None = None

    def to_openrouter(self) -> dict[str, JsonValue]:
        request: dict[str, JsonValue] = {}
        if self.model:
            request["model"] = self.model
        if self.reasoning is not None:
            request["reasoning"] = _json(self.reasoning)
        if self.provider is not None:
            request["provider"] = _json(self.provider)
        return request

    def to_dict(self, *, include_jev_io: bool = False) -> dict[str, JsonValue]:
        return {
            "model": self.model,
            "reasoning": _json(self.reasoning),
            "provider": _json(self.provider),
            "trace": self.trace.to_dict(include_io=include_jev_io) if self.trace else None,
            "dry_run": _json(self.dry_run),
        }


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_range(value: Any) -> tuple[float | None, float | None] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        lower = value.get("min", value.get("minimum"))
        upper = value.get("max", value.get("maximum"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        raise ValueError(
            "intelligence_index_range must be a two-item sequence or a "
            "mapping with min/max"
        )

    lower_value = _float_or_none(lower)
    upper_value = _float_or_none(upper)
    if lower is not None and lower_value is None:
        raise ValueError("intelligence_index_range minimum must be numeric")
    if upper is not None and upper_value is None:
        raise ValueError("intelligence_index_range maximum must be numeric")
    if lower_value is not None and upper_value is not None and lower_value > upper_value:
        raise ValueError("intelligence_index_range minimum cannot exceed maximum")
    if lower_value is None and upper_value is None:
        return None
    return lower_value, upper_value
