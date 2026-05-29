from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recipes.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv

INVALID_TOOL_CALL_ACTION = "<invalid_tool_call>"


@dataclass
class AlfworldToolExecutor:
    max_episode_steps: int = 50
    _env: AlfworldTextworldEnv = field(init=False)
    _history_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._env = AlfworldTextworldEnv(max_episode_steps=self.max_episode_steps)

    def reset(self, game_relative_path: str, task_id: str | None = None) -> str:
        self._history_actions.clear()
        return self._env.reset(game_relative_path=game_relative_path, task_id=task_id)

    def reset_with_info(self, game_relative_path: str, task_id: str | None = None) -> tuple[str, dict[str, Any]]:
        self._history_actions.clear()
        return self._env.reset_with_info(game_relative_path=game_relative_path, task_id=task_id)

    def step(self, command: str) -> dict[str, Any]:
        self._history_actions.append(command)
        observation, reward, done, info = self._env.step(command)
        return {
            "observation": str(observation),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "history_actions": list(self._history_actions),
        }
