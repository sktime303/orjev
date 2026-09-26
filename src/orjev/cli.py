"""The ``orjev`` command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

from .planner import Router
from .types import RouteRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orjev", description="Plan OpenRouter routes with Jev")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("models")
    endpoint = sub.add_parser("endpoints")
    endpoint.add_argument("model")
    plan = sub.add_parser("plan")
    plan.add_argument("--message", default="")
    plan.add_argument("--model")
    plan.add_argument(
        "--candidates",
        nargs="*",
        default=[],
        help="full-model-id regex patterns to consider",
    )
    plan.add_argument("--route", nargs="*", default=["model", "reasoning", "providers"])
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument("--debug", action="store_true", help="include complete Jev request/response IO")
    explain = sub.add_parser("explain")
    explain.add_argument("--message", default="")
    explain.add_argument("--model")
    explain.add_argument("--debug", action="store_true", help="include complete Jev request/response IO")
    sub.add_parser("benchmark", help="show metadata timing guidance")
    sub.add_parser("cache", help="show process cache configuration")
    return parser


async def _run(args: argparse.Namespace) -> Any:
    router = Router.from_env()
    try:
        if args.command == "doctor":
            return {
                "openrouter_api_key": bool(os.environ.get("OPENROUTER_API_KEY")),
                "jev_model": router.jev.config.effective_model,
                "status": "ready",
            }
        if args.command == "models":
            models = await router.catalog.alist_models()
            return [model.to_dict() for model in models]
        if args.command == "endpoints":
            endpoints = await router.endpoints.alist_for_model(args.model)
            return [endpoint.to_dict() for endpoint in endpoints]
        if args.command in {"plan", "explain"}:
            request = RouteRequest(
                messages=[{"role": "user", "content": args.message}],
                current_model=args.model,
                candidate_models=(
                    args.candidates
                    or ([re.escape(args.model)] if args.model else [])
                ),
                route=frozenset(args.route if args.command == "plan" else {"model", "reasoning", "providers"}),
            )
            plan = await router.plan(request, dry_run=args.dry_run or args.command == "explain")
            return plan.to_dict(include_jev_io=args.debug)
        if args.command == "benchmark":
            return {
                "metadata": "GET /models plus one GET /models/{id}/endpoints per candidate",
                "decisions": (
                    "up to two Jev calls (model/reasoning then provider), or one joint "
                    "reasoning×provider call when the model is pinned"
                ),
                "billable_retries": False,
            }
        if args.command == "cache":
            return {"scope": "process", "ttl_seconds": router.catalog.cache.ttl_seconds}
        raise ValueError(args.command)
    finally:
        await router.aclose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"orjev: {exc}", file=sys.stderr)
        return 1
    if args.json or not isinstance(result, list):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for item in result:
            print(json.dumps(item, ensure_ascii=False))
    return 0
