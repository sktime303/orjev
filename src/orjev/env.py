"""Small, dependency-free ``.env`` loader for local configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv() -> None:
    """Load the nearest ``.env`` without overriding process environment values."""
    path = _find_dotenv()
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, value = _parse_line(line)
        if name is not None and value is not None and name not in os.environ:
            os.environ[name] = value


def _find_dotenv() -> Path | None:
    directory = Path.cwd().resolve()
    for candidate_directory in (directory, *directory.parents):
        candidate = candidate_directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_line(line: str) -> tuple[str | None, str | None]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, None
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    name, separator, value = stripped.partition("=")
    name = name.strip()
    if not separator or not _NAME.fullmatch(name):
        return None, None
    return name, _parse_value(value.strip())


def _parse_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value
