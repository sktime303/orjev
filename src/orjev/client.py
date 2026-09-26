"""Convenience sync and async Orjev clients."""

from __future__ import annotations

import asyncio
from typing import Any

from .planner import Router
from .types import CandidateModel, RouteRequest, RoutingPlan


class AsyncOrjev:
    def __init__(self, router: Router | None = None, **kwargs: Any) -> None:
        self.router = router or Router.from_env(**kwargs)

    async def plan(self, request: RouteRequest, *, dry_run: bool = False) -> RoutingPlan:
        return await self.router.plan(request, dry_run=dry_run)

    async def candidate_models(
        self, request: RouteRequest
    ) -> tuple[list[CandidateModel], list[dict[str, str]]]:
        return await self.router.candidate_models(request)

    async def aclose(self) -> None:
        await self.router.aclose()

    async def __aenter__(self) -> AsyncOrjev:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class Orjev:
    def __init__(self, router: Router | None = None, **kwargs: Any) -> None:
        self.router = router or Router.from_env(**kwargs)

    def plan(self, request: RouteRequest, *, dry_run: bool = False) -> RoutingPlan:
        return self.router.plan_sync(request, dry_run=dry_run)

    def candidate_models(
        self, request: RouteRequest
    ) -> tuple[list[CandidateModel], list[dict[str, str]]]:
        return asyncio.run(self.router.candidate_models(request))

    def close(self) -> None:
        asyncio.run(self.router.aclose())
