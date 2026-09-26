# Contributing

Create a Python 3.11+ virtual environment, install `.[dev]`, and run:

```bash
ruff check .
mypy
python -m build
```

Keep Orjev inference-agnostic. Routing policy belongs in `RouteRequest` or a
host adapter; integrations should only merge a validated `RoutingPlan`.
