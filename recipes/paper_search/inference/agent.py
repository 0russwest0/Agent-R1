"""Offline paper search inference agent."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from recipes.paper_search.env.paper_client import SelectorClient
from recipes.paper_search.prompts import PAPERSEARCH_TOOL_SCHEMAS
from recipes.paper_search.runtime import PaperSearchRuntime, PaperSearchRuntimeConfig
from recipes.paper_search.utils import recover_tool_calls_from_text


class PaperSearchInferenceAgent:
    def __init__(
        self,
        logger: logging.Logger,
        *,
        tokenizer: Any,
        llm: Any,
        tool_parser: Any,
        paper_client: Any,
        runtime_config: PaperSearchRuntimeConfig,
        selector_base_url: Optional[str] = None,
        selector_model_name: Optional[str] = None,
        response_length: int = 8192,
        temperature: float = 0.1,
        apply_chat_template_kwargs: Optional[dict[str, Any]] = None,
        thought_log_path: Optional[Path] = None,
    ) -> None:
        from vllm import SamplingParams

        self.logger = logger
        self.tokenizer = tokenizer
        self.llm = llm
        self.tool_parser = tool_parser
        self.response_length = response_length
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.thought_log_path = thought_log_path
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=response_length)
        self.runtime = PaperSearchRuntime(
            config=runtime_config,
            paper_client=paper_client,
            selector_client=SelectorClient(base_url=selector_base_url, model_name=selector_model_name, timeout=30.0),
            logger_=logger,
        )
        self.steps: list[dict[str, Any]] = []

    def _build_prompt_ids(self) -> list[int]:
        return self.tokenizer.apply_chat_template(
            self.runtime.make_messages(),
            tools=PAPERSEARCH_TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    def _llm_generate_token_ids(self, prompt_token_ids: list[int]) -> list[int]:
        from vllm.inputs import TokensPrompt

        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling_params=self.sampling_params,
        )
        token_ids = list(outputs[0].outputs[0].token_ids)
        return token_ids[: self.response_length]

    def _write_thought_log(self, step_idx: int, thought_text: str, tool_call_summaries: list[dict[str, Any]]) -> None:
        if not self.thought_log_path:
            return
        lines = [
            f"==================== Step {step_idx + 1} ====================",
            "[Assistant Reply]",
            thought_text,
            "[Parsed tool calls]",
            json.dumps(tool_call_summaries, ensure_ascii=False, indent=2),
            "",
        ]
        with self.thought_log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    async def run(self, user_query: str, save_path: Path) -> list[str]:
        self.runtime.reset(user_query)
        self.steps = []

        for step_idx in range(self.runtime.config.max_steps):
            paper_list_before = self.runtime.paper_pool.paper_list
            prompt_ids = await asyncio.to_thread(self._build_prompt_ids)

            try:
                response_ids = await asyncio.to_thread(self._llm_generate_token_ids, prompt_ids)
            except Exception as exc:
                self.logger.info("vLLM generate failed: %s", exc)
                break

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            thought_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if not tool_calls:
                tool_calls = recover_tool_calls_from_text(thought_text)

            tool_calls = tool_calls[: self.runtime.config.max_parallel_calls]
            thought_tool_calls = self.runtime.summarize_tool_calls(tool_calls)
            self._write_thought_log(step_idx, thought_text, thought_tool_calls)

            if not tool_calls:
                self.steps.append(
                    {
                        "step_idx": step_idx,
                        "tool_calls": [],
                        "paper_list_before": paper_list_before,
                        "paper_list_after": self.runtime.paper_pool.paper_list,
                        "reward_score": 0.0,
                    }
                )
                break

            self.logger.info("Step %d: %d tool call(s)", step_idx + 1, len(tool_calls))
            tool_reward_score, tool_call_summaries, _ = await self.runtime.execute_tool_calls(tool_calls)
            self.steps.append(
                {
                    "step_idx": step_idx,
                    "tool_calls": tool_call_summaries,
                    "paper_list_before": paper_list_before,
                    "paper_list_after": self.runtime.paper_pool.paper_list,
                    "reward_score": tool_reward_score,
                }
            )
            self.logger.info("Step %d done: papers in pool=%d", step_idx + 1, len(self.runtime.paper_pool.papers))

        save_items = self.runtime.build_save_items()
        save_items["steps"] = self.steps
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(save_items, f, ensure_ascii=False, indent=4)
        return save_items["ordered_ids"]

    async def close(self) -> None:
        await self.runtime.selector_client.close()
        await self.runtime.client.close()
