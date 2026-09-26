<h1 align="center">Orjev [OpenRouter Jev]</h1>

Orjev is a Python library and CLI for planning OpenRouter routes with Jev
Decisions. It discovers catalog and endpoint metadata, filters hard
requirements locally, then asks Jev Decisions two conditioned questions:

1. choose a model × catalog-declared reasoning effort;
2. choose a provider for that selected model and effort.

The result is a plain `RoutingPlan`. Orjev does not send chat inference and
does not append OpenRouter routing suffixes. With a caller-pinned model,
`route={"reasoning", "providers"}` is one joint Jev choice over reasoning ×
provider; `route={"providers"}` is one provider choice. Adding `"model"` uses
the conditioned two-stage model/reasoning → provider flow.

## Installation

Orjev requires Python 3.11 or newer:

```bash
python -m pip install orjev
```

Install optional integrations only when you need them:

```bash
python -m pip install "orjev[mcp]"       # MCP server
python -m pip install "orjev[adapters]"  # OpenAI and LangChain helpers
```

Copy `.env.example` to `.env` and add your OpenRouter key:

```bash
cp .env.example .env
```

The CLI, `Router.from_env()`, `AsyncOrjev()`, and `Orjev()` load `.env`
automatically. Explicit environment variables take precedence. They recognize
`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `ORJEV_REFERER`, `ORJEV_TITLE`,
`JEV_MODEL`, and `ORJEV_ALLOW_MOVING_JEV`. The default Jev model is pinned to
`typesafe/jev-1.13`.

## Python usage

```python
import asyncio
from orjev import AsyncOrjev, RouteRequest, Router


async def main() -> None:
    async with AsyncOrjev(Router.from_env()) as router:
        plan = await router.plan(
            RouteRequest(
                messages=[
                    {"role": "user", "content": "Refactor this module and add tests"}
                ],
                current_model="deepseek/deepseek-v4.1-flash",
                candidate_models=[
                    r"deepseek/deepseek-v4\.1-flash",
                    r"anthropic/claude-sonnet-4",
                ],
                route=frozenset({"model", "reasoning", "providers"}),
                constraints={
                    "require_tools": True,
                    "min_context_tokens": 64_000,
                    "intelligence_index_range": {"min": 40, "max": 80},
                },
                hints={"prefer_low_latency": True},
            )
        )

        # Merge these fields into the OpenRouter chat request made by your app.
        request_fields = plan.to_openrouter()
        print(request_fields)


asyncio.run(main())
```

To inspect the hard-filtered shortlist before asking Jev for a decision, use
the same request with `candidate_models`:

```python
async with AsyncOrjev(Router.from_env()) as router:
    models, rejected = await router.candidate_models(
        RouteRequest(
            messages=[{"role": "user", "content": "Refactor this module"}],
            route=frozenset({"model", "reasoning", "providers"}),
            constraints={"require_tools": True, "min_context_tokens": 64_000},
        )
    )
```

`RouteRequest.candidate_models` is a list of regular expressions matched against
the full model id; an empty list considers the whole catalog. The same patterns
may also appear under `constraints.candidate_models`; top-level and constraint
entries are merged. Every hard filter is combined with logical AND: a model must
match at least one candidate pattern (when any are set) and satisfy every other
constraint. The shortlist scans the full catalog when capability or provider
constraints are present so enough eligible models can be found. The
`intelligence_index_range` constraint is inclusive. Models without an
Artificial Analysis intelligence index are excluded when that constraint is
set. A two-item sequence such as `[40, 80]` is also accepted, and
`intelligence_index` is accepted as a shorthand constraint key.

`request_fields` contains the catalog model id, a unified `reasoning` object,
and, when providers are routed, `provider.only` plus `provider.order`.
`provider.only` is the full hard-filtered eligible set; `provider.order` is the
probability-ranked preference list. Jev sees the already-filtered eligible set.
There is no `routing_mode`, `provider.sort`, or `:nitro`/`:floor`/`:exacto`
generation in v1.

`RoutingPlan.to_openrouter()` only returns request fields. Your application is
responsible for sending the eventual chat request and for handling errors,
logging, retries, and provider-specific behavior.

## CLI and MCP

```bash
orjev doctor
orjev plan --model deepseek/deepseek-v4.1-flash --dry-run --message "debug this"
python -m orjev.mcp_server
```

Use `orjev --json ...` for machine-readable output. The MCP server exposes
model discovery, endpoint discovery, route planning, and dry-run explanations
only. It never performs chat inference.

## Authoritative references

- [Jev Decisions tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial)
- [OpenRouter models API](https://openrouter.ai/docs/guides/overview/models)
- [OpenRouter endpoints API](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model)
- [Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Provider selection](https://openrouter.ai/docs/guides/routing/provider-selection)

See [docs/architecture.md](docs/architecture.md) for the call graph and
workspace-policy limitations.
