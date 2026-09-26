"""Local stdio MCP surface for route planning only."""

from __future__ import annotations

import re
from typing import Any

from .planner import Router
from .types import RouteRequest


def create_server() -> Any:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("orjev")
    router = Router.from_env()

    @server.tool()
    async def list_models() -> list[dict[str, Any]]:
        """List OpenRouter catalog models."""
        return [item.to_dict() for item in await router.catalog.alist_models()]

    @server.tool()
    async def get_model_endpoints(model_id: str) -> list[dict[str, Any]]:
        """List normalized endpoints for one OpenRouter model."""
        return [
            item.to_dict() for item in await router.endpoints.alist_for_model(model_id)
        ]

    @server.tool()
    async def plan_openrouter_route(
        message: str,
        model: str | None = None,
        candidate_models: list[str] | None = None,
        route: list[str] | None = None,
        constraints: dict[str, Any] | None = None,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Plan an OpenRouter request; candidate_models are full-id regexes."""
        request = RouteRequest(
            messages=[{"role": "user", "content": message}],
            current_model=model,
            candidate_models=candidate_models or ([re.escape(model)] if model else []),
            route=frozenset(route or ["model", "reasoning", "providers"]),
            constraints=constraints,
            hints=hints,
        )
        return (await router.plan(request)).to_dict()

    @server.tool()
    async def explain_route(
        message: str, model: str | None = None
    ) -> dict[str, Any]:
        """Return dry-run Jev state/questions without making a Jev call."""
        request = RouteRequest(
            messages=[{"role": "user", "content": message}],
            current_model=model,
            candidate_models=[re.escape(model)] if model else [],
        )
        return (await router.plan(request, dry_run=True)).to_dict()

    return server


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
