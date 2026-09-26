"""Structured, prompt-free decision telemetry."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import DecisionTrace, JsonValue

logger = logging.getLogger("orjev")


@dataclass
class TraceBuilder:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidate_rejections: list[dict[str, JsonValue]] = field(default_factory=list)

    def mark(self, name: str, started_at: float) -> None:
        self.timings_ms[name] = round((time.perf_counter() - started_at) * 1000, 2)

    def trace(self, **kwargs: Any) -> DecisionTrace:
        return DecisionTrace(
            trace_id=self.trace_id,
            timings_ms=dict(self.timings_ms),
            candidate_rejections=tuple(self.candidate_rejections),
            **kwargs,
        )


TelemetryHook = Callable[[DecisionTrace], None]


def emit(trace: DecisionTrace, hook: TelemetryHook | None = None) -> None:
    if hook:
        hook(trace)
    logger.info(
        "orjev decision trace_id=%s fallback=%s model=%s provider=%s",
        trace.trace_id,
        trace.fallback,
        trace.selected_model,
        trace.selected_provider,
    )
