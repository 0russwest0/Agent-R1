from __future__ import annotations

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import register

# Import registers the recipe-local AgentEnv implementation. The actual loop
# logic is inherited from AgentEnvLoop; this class only provides a recipe entry.
from recipes.gsm8k.env import gsm8k_tool_env as _gsm8k_tool_env  # noqa: F401


@register("gsm8k_agent")
class GSM8KToolLoop(AgentEnvLoop):
    """Recipe-local registration shim for the GSM8K + Tool example."""
