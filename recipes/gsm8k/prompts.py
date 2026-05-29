"""Prompt templates for the GSM8K recipe."""

GSM8K_PLAIN_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

GSM8K_AGENT_SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step. "
    "Before giving the final answer, call the `calc_gsm8k_reward` tool at least once with your answer. "
    "Use the tool feedback to refine the answer if needed. "
    "Put the final answer in the format `#### <answer>`."
)

GSM8K_AGENT_USER_PROMPT = "{question}\n\nThink step by step and use the reward tool before the final answer."


def build_plain_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{question} {GSM8K_PLAIN_INSTRUCTION}"}]


def build_agent_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GSM8K_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": GSM8K_AGENT_USER_PROMPT.format(question=question)},
    ]
