WEBSHOP_SYSTEM_PROMPT = (
    "You are acting in the WebShop text environment. "
    "Your goal is to find and buy the product that best satisfies the shopping instruction. "
    "Use exactly one executable WebShop action each turn through the env_step tool. "
    "Do not explain."
)


WEBSHOP_USER_PROMPT = """### Shopping Instruction
{instruction}

### Current Observation
{observation}

### Recent History
{recent_history}

### Available Actions
{available_actions}

### Instructions
- Use exactly one action through the `env_step` tool.
- The `command` must be one available action exactly, except replace `<your query>` in `search[<your query>]` with concise product keywords.
- Click product ASINs, option values, `Description`, `Features`, `Reviews`, `Back to Search`, `Back to Item`, or `Buy Now` only when listed above.
- Buy only when the selected product and options satisfy the shopping instruction.
"""


EXEC_ACTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_step",
        "description": "Execute one WebShop action and return the next text observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A single action such as `search[wireless headphones]` or `click[Buy Now]`.",
                }
            },
            "required": ["command"],
        },
    },
}


WEBSHOP_TOOL_SCHEMAS = [EXEC_ACTION_TOOL_SCHEMA]
