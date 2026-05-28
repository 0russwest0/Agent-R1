ALFWORLD_SYSTEM_PROMPT = (
    "You are acting in ALFWorld TextWorld. "
    "Choose exactly one command from the provided admissible commands each turn. "
    "You may reason briefly inside <think></think> before acting. "
    "Call the env_step tool with that exact command. "
    "Do not output a final natural-language answer."
)


ALFWORLD_USER_PROMPT = """### Task
{task_text}

### Current Observation
{observation}

### History Actions
{history_actions}

### Admissible Commands
{admissible_commands}

### Instructions
- Think briefly about the current state inside `<think>...</think>` tags.
- Use exactly one command through the `env_step` tool.
- The command must exactly match one item from `Admissible Commands`.
- Follow ALFWorld TextWorld command style such as `go to dresser 1`, `take mug 1 from cabinet 3`, `use desklamp 1`.
- Use the official observation text as the source of truth.
- Do not output explanations or a final natural-language answer.

### Output Format
<think>
[Your brief reasoning about the current state and next command.]
</think>
<tool_call>
{{"name": "env_step", "arguments": {{"command": "[one admissible command]"}}}}
</tool_call>
"""


EXEC_ACTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_step",
        "description": (
            "Execute one ALFWorld TextWorld command and return the next official observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "A single ALFWorld TextWorld command such as "
                        "`go to dresser 1`, `open cabinet 3`, `take mug 1 from cabinet 3`, `use desklamp 1`. "
                        "It must exactly match one currently admissible command."
                    ),
                }
            },
            "required": ["command"],
        },
    },
}


ALFWORLD_TOOL_SCHEMAS = [EXEC_ACTION_TOOL_SCHEMA]
