"""GSM8K recipe."""

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

# Importing tool registers the recipe-local BaseTool implementation.
from . import tool as _tool  # noqa: F401

__all__ = ["AgentEnvLoop"]
