"""OpenRouter Jev Decisions client.

Jev calls are billable and are therefore never retried automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import JevError
from .http import AsyncHttpClient
from .types import JevAnswer, JsonValue

JEV_ENDPOINT = "/api/alpha/decisions"
PINNED_JEV_MODEL = "typesafe/jev-1.13"


@dataclass(frozen=True)
class JevConfig:
    model: str = PINNED_JEV_MODEL
    allow_moving_alias: bool = False

    @property
    def effective_model(self) -> str:
        if self.allow_moving_alias and self.model == PINNED_JEV_MODEL:
            return "typesafe/jev"
        return self.model


class JevClient:
    def __init__(
        self,
        http: AsyncHttpClient,
        *,
        config: JevConfig | None = None,
    ) -> None:
        self.http = http
        self.config = config or JevConfig()

    async def adecide(
        self,
        *,
        state: dict[str, JsonValue],
        questions: dict[str, JsonValue],
    ) -> JevAnswer:
        payload = self._payload(state, questions)
        response = await self.http.post_json(JEV_ENDPOINT, payload)
        return parse_answer(
            response,
            stage=next(iter(questions), "unknown"),
            request=payload,
        )

    def _payload(
        self, state: dict[str, JsonValue], questions: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if "route" in state:
            raise JevError("route must remain an Orjev orchestration concern")
        return {
            "model": self.config.effective_model,
            "state": state,
            "questions": questions,
        }


def parse_answer(
    payload: Any,
    *,
    stage: str,
    request: dict[str, JsonValue] | None = None,
) -> JevAnswer:
    if isinstance(payload, dict):
        data = payload.get("answers", payload.get("data", payload))
    else:
        data = payload
    if not isinstance(data, dict):
        raise JevError("Jev response was not an object")
    answer = data.get(stage)
    if not isinstance(answer, dict):
        # Some versions return the sole question directly.
        answer = data if "choice" in data else None
    if answer is None:
        raise JevError(f"Jev response did not contain answer {stage!r}")
    choice = answer.get("choice")
    probabilities = answer.get("probabilities", {})
    if not isinstance(choice, str) or not isinstance(probabilities, dict):
        raise JevError(f"Jev answer {stage!r} has invalid choice/probabilities")
    normalized = {
        str(key): max(0.0, float(value))
        for key, value in probabilities.items()
        if _number(value) is not None
    }
    total = sum(normalized.values())
    if total > 0:
        normalized = {key: value / total for key, value in normalized.items()}
    confidence = _number(answer.get("confidence"))
    return JevAnswer(
        stage=stage,
        choice=choice,
        confidence=confidence,
        probabilities=normalized,
        raw=_json_dict(payload) if isinstance(payload, dict) else {"value": str(payload)},
        request=request,
    )


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
