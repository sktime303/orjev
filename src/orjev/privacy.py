"""Evidence shaping and byte limits before state is sent to Jev."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import PrivacyError
from .types import JsonValue, RouteRequest

Summarizer = Callable[[str], str]


@dataclass(frozen=True)
class PrivacyConfig:
    max_state_bytes: int = 48_000
    summarizer: Summarizer | None = None


class Privacy:
    def __init__(self, config: PrivacyConfig | None = None) -> None:
        self.config = config or PrivacyConfig()

    def request_state(self, request: RouteRequest) -> dict[str, JsonValue]:
        message = request.latest_message
        if self.config.summarizer:
            message = self.config.summarizer(message)
        result: dict[str, JsonValue] = {"latest_user_message": message}
        if request.hard_constraints.require_tools:
            result["requires_tools"] = True
        history = self._history(request)
        if history:
            result["conversation_history"] = history
        return result

    def _history(self, request: RouteRequest) -> list[dict[str, JsonValue]]:
        latest_index = next(
            (
                index
                for index in range(len(request.messages) - 1, -1, -1)
                if request.messages[index].get("role") == "user"
            ),
            len(request.messages) - 1,
        )
        previous = request.messages[:latest_index]
        result: list[dict[str, JsonValue]] = []
        for item in previous:
            role = str(item.get("role") or "unknown")
            content = str(item.get("content") or "")
            result.append({"role": role, "content": content})
        return result

    def hints(self, hints: dict[str, JsonValue] | None) -> dict[str, JsonValue]:
        if not hints:
            return {}
        return {str(key): self._safe_json(value) for key, value in hints.items()}

    def bounded_state(self, state: dict[str, JsonValue]) -> dict[str, JsonValue]:
        encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= self.config.max_state_bytes:
            return state
        trimmed = dict(state)
        while len(json.dumps(trimmed, ensure_ascii=False).encode("utf-8")) > self.config.max_state_bytes:
            if "models" in trimmed and isinstance(trimmed["models"], list) and trimmed["models"]:
                trimmed["models"] = trimmed["models"][:-1]
            elif "providers" in trimmed and isinstance(trimmed["providers"], list) and trimmed["providers"]:
                trimmed["providers"] = trimmed["providers"][:-1]
            else:
                raise PrivacyError("Jev state exceeds configured byte limit")
        return trimmed

    def _safe_json(self, value: Any) -> JsonValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._safe_json(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._safe_json(item) for key, item in value.items()}
        return str(value)
