from __future__ import annotations

from typing import Any

from agent_r1.env.base import AgentEnv, Observation
from agent_r1.env.envs.tool import ToolEnv

# Importing the module registers the recipe-local BaseTool implementation.
from recipes.gsm8k import tools as _tools  # noqa: F401
from recipes.gsm8k.prompts import build_agent_messages


@AgentEnv.register("gsm8k_agent")
class GSM8KToolEnv(ToolEnv):
    """Tool environment that builds GSM8K prompts dynamically from sample fields."""

    def __init__(
        self,
        tools: list[str] | None = None,
        tool_format: str = "hermes",
        tools_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tools=tools or ["calc_gsm8k_reward"],
            tool_format=tool_format,
            tools_kwargs=tools_kwargs,
            **kwargs,
        )

    def reset(self, **kwargs) -> Observation:
        question = _extract_question(kwargs)
        ground_truth = _extract_ground_truth(kwargs)
        if ground_truth is not None:
            self.tools_kwargs = {**self.tools_kwargs, "ground_truth": str(ground_truth)}
        self._messages = build_agent_messages(question)
        return Observation(messages=list(self._messages))


def _extract_question(kwargs: dict[str, Any]) -> str:
    question = kwargs.get("question")
    if question:
        return str(question).strip()
    extra_info = kwargs.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"]).strip()
    raw_prompt = kwargs.get("raw_prompt") or []
    for message in reversed(list(raw_prompt)):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _extract_ground_truth(kwargs: dict[str, Any]) -> Any:
    if kwargs.get("ground_truth") is not None:
        return kwargs["ground_truth"]
    reward_model = kwargs.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return reward_model["ground_truth"]
    extra_info = kwargs.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("ground_truth") is not None:
        return extra_info["ground_truth"]
    return None
