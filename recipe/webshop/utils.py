from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from recipe.webshop.prompts import WEBSHOP_SYSTEM_PROMPT, WEBSHOP_USER_PROMPT


def _short(text: str, limit: int = 1800) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_recent_history(history: list[dict[str, str]], *, limit: int = 2) -> str:
    if not history:
        return "None"
    recent = history[-limit:]
    start = len(history) - len(recent) + 1
    lines = []
    for offset, record in enumerate(recent):
        step_num = start + offset
        observation = _short(record.get("observation", ""))
        action = str(record.get("action", "")).strip()
        lines.append(f"[Observation {step_num}]\n{observation}\n[Action {step_num}]\n{action}")
    return "\n\n".join(lines)


def format_available_actions(actions: list[str] | None) -> str:
    if not isinstance(actions, list) or not actions:
        return "None"
    return "\n".join(f"- {action}" for action in actions)


def build_webshop_messages(
    *,
    instruction: str,
    observation: str,
    recent_history: list[dict[str, str]],
    available_actions: list[str] | None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": WEBSHOP_USER_PROMPT.format(
                instruction=instruction,
                observation=observation,
                recent_history=format_recent_history(recent_history),
                available_actions=format_available_actions(available_actions),
            ),
        },
    ]


def build_invalid_tool_call_observation(previous_observation: str, reason: str) -> str:
    return (
        "Invalid tool call. You must call the `env_step` tool exactly once with JSON arguments "
        'like {"command": "search[wireless headphones]"} or {"command": "click[Buy Now]"}. '
        f"Reason: {reason}\n\n"
        "The environment state did not change. Current Observation:\n"
        f"{previous_observation}"
    )


@dataclass
class WebShopEnvClient:
    base_url: str | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("WEBSHOP_ENV_BASE_URL") or "http://127.0.0.1:4100").rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def reset(self, goal_index: int) -> dict[str, Any]:
        resp = await self.client.post("/reset", json={"goal_index": int(goal_index)})
        resp.raise_for_status()
        return resp.json()

    async def step(self, goal_index: int, env_state: dict[str, Any], action: str) -> dict[str, Any]:
        resp = await self.client.post(
            "/step",
            json={"goal_index": int(goal_index), "env_state": env_state, "action": action},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self.client.aclose()
