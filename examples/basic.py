import asyncio

from orjev import AsyncOrjev, RouteRequest


async def main() -> None:
    async with AsyncOrjev() as router:
        plan = await router.plan(
            RouteRequest(
                messages=[{"role": "user", "content": "Explain this traceback"}],
                current_model="deepseek/deepseek-v4.1-flash",
                candidate_models=[r"deepseek/deepseek-v4\.1-flash"],
                route=frozenset({"reasoning", "providers"}),
            )
        )
        print(plan.to_openrouter())


if __name__ == "__main__":
    asyncio.run(main())
