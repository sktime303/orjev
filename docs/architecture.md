# Architecture

Orjev is a planner, not a chat SDK.

```text
caller
  -> RouteRequest
  -> GET /models (cached)
  -> GET /models/{id}/endpoints (cached, parallel)
  -> local hard filters and model × reasoning enumeration
  -> Jev: model_config and/or provider and/or joint route_config
  -> RoutingPlan.to_openrouter()
```

When the model is caller-pinned, reasoning plus providers is represented as one
joint choice over reasoning × provider, so it uses one Jev request. The
provider-only case also uses one request. If model selection is enabled, Orjev
uses two conditioned requests: model × reasoning first, then provider. This
keeps provider selection conditioned on the chosen model. Questions inside one
Jev request are parallel rather than conditioned, so the joint option is
explicit rather than two independent questions.

The provider state includes `selection` (model id and reasoning) plus all
eligible endpoint statistics in `providers`; it does not repeat the model
catalog blob. The provider question lists every eligible tag (up to Jev’s
255-option limit) as choice keys with null criterion labels; Jev reads the
stats from state. `provider.order` follows Jev’s `choice` first, then remaining
tags by Jev probability. The compiled `provider.only` retains every eligible tag.

Orjev can enforce facts visible in OpenRouter metadata: context capacity,
modalities, tool/reasoning support, prices, provider tags, latency, uptime, and
the Artificial Analysis `intelligence_index`. Candidate model ids are limited
with full-model-id regular expressions on `RouteRequest.candidate_models` and/or
`constraints.candidate_models`; an empty list means all catalog models. All hard
filters, including patterns, are ANDed. Orjev applies merged constraints locally before constructing
Jev state, so Jev receives only the already-filtered model/provider candidates.
An `intelligence_index_range` is inclusive and excludes models with no score.
Hosts provide workspace policy as hard constraints and product-specific
preferences as hints.

## Fallbacks

Metadata errors, Jev transport errors, unknown choices, and post-selection
constraint violations all produce a `DecisionTrace` with `fallback=true`.
The planner pins the caller's current model/reasoning first, then uses a
caller-supplied provider allow/order set if one exists. It never retries a
billable Jev request.

## Cost, context, and privacy

The default fully-routed plan makes up to two Jev requests. Orjev skips a
billable Jev POST when a stage has only one eligible choice (for example a
pinned model with a single model×reasoning option, or a single eligible
provider tag). Metadata is cached in
process and is safe to refresh with bounded retries. Normal logs contain trace
ids and route outcomes, not prompt text. Jev receives the complete
`RouteRequest.messages` history with the latest user message identified
separately. Use `PrivacyConfig.summarizer` if you want to reshape the latest
message before Jev sees it.
Complete Jev request/response IO is available only through the explicit debug
view. If the state exceeds the byte limit, Orjev falls back rather than
shortening user messages.

## Integrations

`orjev.integrations` contains field-merging helpers for raw HTTP, OpenAI
`extra_body`, OpenRouter SDK-style kwargs, and LangChain-style `model_kwargs`.
The helpers do not own a model pool or routing policy.
