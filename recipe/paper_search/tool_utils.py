"""Shared tool-call parsing helpers for paper search agent flows."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from verl.experimental.agent_loop.tool_parser import FunctionCall

logger = logging.getLogger(__name__)

PAPER_SEARCH_TOOL_NAMES = frozenset({"search", "expand"})
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def json_or_python_dict(raw: str) -> Any:
    """Parse JSON; on failure try Python literal dict (common small-model mistake).

    Args:
        raw: Raw tool-call payload string.

    Returns:
        Parsed object, or None when parsing fails.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    if len(text) > 8192 or "__" in text:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _coerce_bare_string_arguments(name: str, value: str) -> dict[str, Any] | None:
    """Map a bare string argument to the expected parameter dict."""
    stripped = value.strip()
    if not stripped:
        return None
    if name == "search":
        return {"query": stripped}
    if name == "expand":
        return {"paper_id": stripped}
    return None


def normalize_tool_call_dict(obj: Any) -> dict[str, Any] | None:
    """Return a normalized tool call dict with str name and dict arguments.

    Args:
        obj: Parsed tool-call object.

    Returns:
        Normalized dict with ``name`` and ``arguments`` keys, or None if invalid.
    """
    if not isinstance(obj, dict) or "name" not in obj:
        return None

    name = str(obj["name"])
    args = obj.get("arguments")
    if args is None:
        return None

    if isinstance(args, str):
        inner = json_or_python_dict(args)
        if isinstance(inner, dict):
            args = inner
        elif isinstance(inner, str):
            args = _coerce_bare_string_arguments(name, inner)
            if args is None:
                return None
        else:
            args = _coerce_bare_string_arguments(name, args)
            if args is None:
                return None

    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def decode_tool_arguments(name: str, arguments: str) -> dict[str, Any] | None:
    """Decode serialized tool arguments into a parameter dict.

    Args:
        name: Tool name (``search`` or ``expand``).
        arguments: Serialized arguments from ``FunctionCall.arguments``.

    Returns:
        Parameter dict when valid, otherwise None.
    """
    obj = json_or_python_dict(arguments)
    if obj is None:
        return None

    if isinstance(obj, str):
        inner = json_or_python_dict(obj)
        if isinstance(inner, dict):
            obj = inner
        else:
            obj = _coerce_bare_string_arguments(name, obj)
            if obj is None:
                return None

    if not isinstance(obj, dict):
        return None
    return obj


def recover_tool_calls_from_text(text: str) -> list[FunctionCall]:
    """Recover tool calls from raw response text when the parser fails.

    Args:
        text: Decoded model response.

    Returns:
        Recovered ``FunctionCall`` objects with normalized JSON arguments.
    """
    recovered: list[FunctionCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        obj = json_or_python_dict(raw.strip())
        if obj is None:
            continue
        norm = normalize_tool_call_dict(obj)
        if norm is None or norm["name"] not in PAPER_SEARCH_TOOL_NAMES:
            continue
        try:
            recovered.append(
                FunctionCall(
                    name=norm["name"],
                    arguments=json.dumps(norm["arguments"], ensure_ascii=False),
                )
            )
        except Exception as exc:
            logger.warning("Failed to serialize recovered tool call: %s", exc)
    return recovered


def extract_search_query(tool_args: dict[str, Any]) -> str | None:
    """Extract a non-empty search query from decoded tool arguments."""
    query = tool_args.get("query")
    if query is None:
        return None
    text = str(query).strip()
    return text or None


def extract_expand_paper_id(tool_args: dict[str, Any]) -> str | None:
    """Extract a non-empty paper id from decoded tool arguments."""
    paper_id = tool_args.get("paper_id")
    if paper_id is None:
        return None
    text = str(paper_id).strip()
    return text or None
