from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EnvState(BaseModel):
    page_type: Literal["home", "search_results", "item", "subpage", "done"] = "home"
    query: str = ""
    page_num: int = 0
    asin: str | None = None
    subpage: str | None = None
    selected_options: dict[str, str] = Field(default_factory=dict)
    last_action: str | None = None


class ResetRequest(BaseModel):
    goal_index: int


class ResetResponse(BaseModel):
    observation: str
    env_state: EnvState
    info: dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    goal_index: int
    env_state: EnvState
    action: str


class StepResponse(BaseModel):
    observation: str
    env_state: EnvState
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

