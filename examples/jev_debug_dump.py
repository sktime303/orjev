"""Write complete Jev request/response pairs to JSON files (no truncation).

Usage:
  cp .env.example .env   # then set OPENROUTER_API_KEY in .env
  PYTHONPATH=src python examples/jev_debug_dump.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from orjev.planner import Router
from orjev.types import RouteRequest

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".orjev-cache" / "jev-io"


def _jev_model_payload(client_model: str, state: dict, questions: dict) -> dict:
    return {"model": client_model, "state": state, "questions": questions}


async def _dump(label: str, request: RouteRequest, *, dry_run: bool) -> None:
    router = Router.from_env()
    try:
        plan = await router.plan(request, dry_run=dry_run)
        jev_model = router.jev.config.effective_model
        records: list[dict] = []

        if dry_run and plan.dry_run:
            for index, stage in enumerate(plan.dry_run.get("stages") or [], start=1):
                records.append(
                    {
                        "label": label,
                        "stage_index": index,
                        "billable": False,
                        "request": _jev_model_payload(
                            jev_model, stage["state"], stage["questions"]
                        ),
                        "response": None,
                        "note": "dry_run; Jev was not called",
                    }
                )
        trace = plan.trace
        if trace:
            for index, stage in enumerate(trace.stages, start=1):
                records.append(
                    {
                        "label": label,
                        "stage_index": index,
                        "stage": stage.stage,
                        "billable": True,
                        "request": stage.request,
                        "response": stage.raw,
                    }
                )

        payload = {
            "label": label,
            "dry_run": dry_run,
            "plan_summary": {
                "model": plan.model,
                "reasoning": plan.reasoning,
                "provider": plan.provider,
                "jev_calls": len(records),
                "fallback": trace.fallback if trace else None,
                "fallback_reason": trace.fallback_reason if trace else None,
            },
            "records": records,
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        safe = label.replace("/", "_").replace(" ", "-")
        path = OUT_DIR / f"{safe}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(records)} Jev record(s), {path.stat().st_size} bytes)")
    finally:
        await router.aclose()


async def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set; live dumps will fail (dry-run dumps still work).")

    await _dump(
        "deepseek-v3.2-pinned-default-route",
        RouteRequest(
            messages=[{"role": "user", "content": "Explain recursion briefly"}],
            current_model="deepseek/deepseek-v3.2",
            candidate_models=["deepseek/deepseek-v3.2"],
        ),
        dry_run=False,
    )
    await _dump(
        "deepseek-v4.1-flash-pinned-default-route",
        RouteRequest(
            messages=[{"role": "user", "content": "Write a Python quicksort"}],
            current_model="deepseek/deepseek-v4.1-flash",
            candidate_models=["deepseek/deepseek-v4.1-flash"],
        ),
        dry_run=False,
    )
    await _dump(
        "deepseek-v4.1-flash-default-route-dry-run",
        RouteRequest(
            messages=[{"role": "user", "content": "Write a Python quicksort"}],
            current_model="deepseek/deepseek-v4.1-flash",
            candidate_models=["deepseek/deepseek-v4.1-flash"],
        ),
        dry_run=True,
    )
    await _dump(
        "deepseek-v4.1-flash-provider-only-dry-run",
        RouteRequest(
            messages=[{"role": "user", "content": "Write a Python quicksort"}],
            current_model="deepseek/deepseek-v4.1-flash",
            candidate_models=["deepseek/deepseek-v4.1-flash"],
            route=frozenset({"providers"}),
        ),
        dry_run=True,
    )
    flash = "deepseek/deepseek-v4.1-flash"
    await _dump(
        "deepseek-v4.1-flash-build-game-from-scratch",
        RouteRequest(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Build a complete 2D platformer game from scratch in Python "
                        "using pygame: player movement, enemies, multiple levels, "
                        "scoring, save/load, and basic sound effects. Give me the "
                        "full project structure and implementation plan."
                    ),
                }
            ],
            current_model=flash,
            candidate_models=[flash],
        ),
        dry_run=False,
    )
    await _dump(
        "deepseek-v4.1-flash-latency-sensitive-quick",
        RouteRequest(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Tell me this quickly — no long explanation: is binary search "
                        "O(log n) for a sorted array?"
                    ),
                }
            ],
            current_model=flash,
            candidate_models=[flash],
            hints={"prefer_low_latency": True},
        ),
        dry_run=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
