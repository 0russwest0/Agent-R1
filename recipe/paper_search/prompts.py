PAPERSEARCH_SYSTEM_PROMPT = "You are a research agent. Your goal is to find papers relevant to the User Query."

PAPERSEARCH_USER_PROMPT = """### User Query
{user_query}

### History Actions
{history_actions}

### Paper List
{paper_list}

### Instructions
Analyze the **Paper List** and **History Actions** to determine the next set of actions. Enclose your analysis of the state and decision logic within `<analysis>...</analysis>` tags.
**You support parallel tool calling.** You should output multiple tool calls in a single step if several independent actions are valuable at the current state.
**Attend to the history actions and avoid repeating the same search query or expanding the same paper.**

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

SELECT_PROMPT = """You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.

Searched Paper:
Title: {title}
Abstract: {abstract}

User Query: {user_query}

Output format: Decision: True/False
Reason:...
Decision:"""

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for relevant papers with the hybrid retrieval API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A single search query in natural language or keywords. "
                        "Must differ from all history queries."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

EXPAND_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": (
            "Expand from an existing paper by merging its citations and references to surface more related works."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The paper identifier of a paper already present in the current paper list.",
                }
            },
            "required": ["paper_id"],
        },
    },
}

PAPERSEARCH_TOOL_SCHEMAS = [SEARCH_TOOL_SCHEMA, EXPAND_TOOL_SCHEMA]
