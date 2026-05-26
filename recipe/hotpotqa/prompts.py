"""
HotpotQA prompts — same layout as `recipe/paper_search/prompts.py` (system + user
sections, Instructions, Output Format with `<analysis>` / `<tool_call>` placeholders).
HotpotQA-only: `### Retrieved Passages`, `### Recent tool / format issues`, and
`<answer>` when finishing from current evidence.
"""

HOTPOTQA_SYSTEM_PROMPT = "You are a research agent. Your goal is to answer the User Query using Wikipedia search evidence."

HOTPOTQA_USER_PROMPT = """### User Query 
{user_query}

### History Actions
{history_actions}

### Retrieved Passages
{passage_list}

### Recent tool / format issues
{tool_feedback}

### Instructions
Analyze the **Retrieved Passages** and **History Actions** to determine the next set of actions. Enclose your analysis of the state and decision logic within `<analysis>...</analysis>` tags.
**You support parallel tool calling.** You should output multiple tool calls in a single step if several independent actions are valuable at the current state.
**Attend to the history actions and avoid repeating the same search queries.**
When you can answer the question from the current passages, put the short final answer inside `<answer></answer>` tags (no explanation) instead of further tool calls.

### Output Format
<analysis>
[Your analysis of the current state and decision logic...]
</analysis>
<tool_call>
[Tool call 1]
</tool_call>
<tool_call>
[Tool call 2]
</tool_call>
...
"""

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search Wikipedia for passages relevant to the user question. "
            "Use natural-language or keyword queries; must differ from prior history queries when possible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A single search query (natural language or keywords). "
                        "Must differ from all history queries when seeking new evidence."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

HOTPOTQA_TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA]
