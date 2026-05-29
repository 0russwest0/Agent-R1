from __future__ import annotations

from recipes.webshop.prompts import WEBSHOP_SYSTEM_PROMPT, WEBSHOP_USER_PROMPT


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
