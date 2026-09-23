"""Pull a JSON object out of whatever a model actually returned.

Models wrap JSON in prose, in ``` fences, or emit it with a trailing comma.
Treating the payload as untrusted text and scanning for the first balanced
object is far more forgiving than json.loads on the raw string, and costs
nothing when the provider already honoured a JSON mode.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None

    for candidate in _candidates(text):
        parsed = _try_parse(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def _candidates(text: str):
    stripped = text.strip()
    yield stripped

    for fenced in _FENCE.findall(text):
        yield fenced.strip()

    balanced = _first_balanced_object(stripped)
    if balanced is not None:
        yield balanced


def _try_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", text))
    except (json.JSONDecodeError, ValueError):
        return None


def _first_balanced_object(text: str) -> str | None:
    """Scan for the first {...} that balances, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
