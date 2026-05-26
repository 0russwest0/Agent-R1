import asyncio
import logging
import os
import threading
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from recipe.paper_search.prompts import (
    PAPERSEARCH_SYSTEM_PROMPT,
    PAPERSEARCH_TOOL_SCHEMAS,
    PAPERSEARCH_USER_PROMPT,
    SELECT_PROMPT,
)
from recipe.paper_search.tool_utils import (
    PAPER_SEARCH_TOOL_NAMES,
    decode_tool_arguments,
    extract_expand_paper_id,
    extract_search_query,
    recover_tool_calls_from_text,
)
from recipe.paper_search.utils import Paper, PaperPool, PaperSearchClient, SelectorClient
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("paper_search_agent")
class PaperSearchAgentFlow(AgentFlowBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, reward_loop_worker, tokenizer, processor, **kwargs)
        self.max_steps = kwargs.get("max_steps", 5)
        self.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        self.reward_top_k = kwargs.get("reward_top_k", 3)
        self.reward_threshold = kwargs.get("score_threshold", 0.4)
        self.search_cost = kwargs.get("search_cost", 0)
        self.expand_cost = kwargs.get("expand_cost", 0)
        self.use_discrete_reward = kwargs.get("use_discrete_reward", False)
        self.search_top_k = kwargs.get("search_top_k", 10)
        self.citations_limit = kwargs.get("citations_limit", 30)
        self.references_limit = kwargs.get("references_limit", -1)

        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.tool_schemas = PAPERSEARCH_TOOL_SCHEMAS
        self.client = PaperSearchClient(timeout=30.0)
        self.selector_client = SelectorClient(timeout=30.0)

        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.history_search_queries: dict[str, int] = {}
        self.user_query = ""
        self.steps: list[AgentFlowStep] = []
        self.history_actions: list[tuple[str, str]] = []

    def _format_history_actions(self) -> str:
        if not self.history_actions:
            return "None"

        lines: list[str] = []
        for action, value in self.history_actions:
            if action == "search":
                lines.append(f"[Search] {value}")
            elif action == "expand":
                lines.append(f"[Expand] {value}")
            else:
                raise ValueError(f"Invalid action: {action}")
        return "\n".join(lines) if lines else "None"

    def _make_anchor_obs(self) -> str:
        return PAPERSEARCH_USER_PROMPT.format(
            user_query=self.user_query,
            paper_list=self.paper_pool.paper_list,
            history_actions=self._format_history_actions(),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        self.paper_pool = PaperPool()
        self.history_search_queries = {}
        self.user_query = raw_prompt[0]["content"]
        self.steps = []
        self.history_actions = []

        metrics: dict[str, Any] = {}
        total_search_action_count = 0
        total_expand_action_count = 0
        total_reward_score = 0.0
        num_steps = 0

        while num_steps < self.max_steps:
            num_steps += 1
            anchor_obs = self._make_anchor_obs()

            messages = [
                {"role": "system", "content": PAPERSEARCH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PAPERSEARCH_USER_PROMPT.format(
                        user_query=self.user_query,
                        paper_list=self.paper_pool.paper_list,
                        history_actions=self._format_history_actions(),
                    ),
                },
            ]

            prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

            response_ids = output.token_ids[: self.response_length]
            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            if not tool_calls:
                tool_calls = recover_tool_calls_from_text(response_text)

            if not tool_calls:
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=total_reward_score,
                    extra_fields={
                        "anchor_obs": anchor_obs,
                        "reward_extra_info": {
                            "search_actions_total": total_search_action_count,
                            "expand_actions_total": total_expand_action_count,
                        },
                    },
                )
                step = await self._postprocess(step, **kwargs)
                self.steps.append(step)
                break

            tool_calls = tool_calls[: self.max_parallel_calls]

            tasks = []
            for tool_call in tool_calls:
                if tool_call.name not in PAPER_SEARCH_TOOL_NAMES:
                    logger.warning("Unknown tool call: %s", tool_call.name)
                    continue

                tool_args = decode_tool_arguments(tool_call.name, tool_call.arguments)
                if not tool_args:
                    logger.warning(
                        "Invalid tool arguments for %s: %r",
                        tool_call.name,
                        tool_call.arguments,
                    )
                    continue

                if tool_call.name == "search":
                    query = extract_search_query(tool_args)
                    if query:
                        tasks.append(self.search(query, **kwargs))
                        self.history_actions.append(("search", query))
                        total_search_action_count += 1
                elif tool_call.name == "expand":
                    paper_id = extract_expand_paper_id(tool_args)
                    if paper_id:
                        tasks.append(self.expand(paper_id, **kwargs))
                        self.history_actions.append(("expand", paper_id))
                        total_expand_action_count += 1

            with simple_timer("tool_calls", metrics):
                reward_scores = await asyncio.gather(*tasks) if tasks else []

            step_reward_score = sum(reward_scores)
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=step_reward_score,
                extra_fields={
                    "anchor_obs": anchor_obs,
                    "reward_extra_info": {
                        "search_actions_total": total_search_action_count,
                        "expand_actions_total": total_expand_action_count,
                    },
                },
            )
            step = await self._postprocess(step, **kwargs)
            self.steps.append(step)

        return AgentFlowOutput(steps=self.steps, metrics=metrics)

    async def search(self, query: str, **kwargs) -> float:
        if query in self.history_search_queries:
            return -0.5

        try:
            papers = await self.client.search(query=query, limit=self.search_top_k)
        except Exception as exc:
            logger.warning("Error in search %s: %r", query, exc)
            self.history_search_queries[query] = 0
            return 0.0

        new_papers: list[Paper] = []
        tasks = []
        seen_paper_ids: set[str] = set()

        for paper in papers:
            if not paper.paper_id or paper.paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper.paper_id)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper, **kwargs))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_papers: list[Paper] = []
        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores):
            if score < 0.01:
                continue

            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

                kept_papers.append(paper)
                kept_scores.append(score)
                self.paper_pool.add_paper(paper, "search", query, score)

        self.history_search_queries[query] = len(kept_papers)

        if not kept_papers:
            return 0.0

        top_k_scores = sorted(kept_scores, reverse=True)[: self.reward_top_k]
        if self.use_discrete_reward:
            top_k_scores = [1.0 if score >= self.reward_threshold else 0.0 for score in top_k_scores]
        return sum(top_k_scores) - self.search_cost

    async def expand(self, paper_id: str, **kwargs) -> float:
        with self._paper_pool_lock:
            paper_pool_entry = self.paper_pool.get_paper(paper_id)
            if not paper_pool_entry:
                return -0.5
            if paper_pool_entry.expand:
                return -0.5
            paper_pool_entry.expand = True

        try:
            citations, references = await asyncio.gather(
                self.client.get_citations(paper_id, limit=self.citations_limit),
                self.client.get_references(paper_id, limit=self.references_limit),
            )
        except Exception as exc:
            logger.warning("Error in expand %s: %r", paper_id, exc)
            return 0.0

        merged_candidates: list[Paper] = []
        seen_paper_ids: set[str] = set()
        for paper in citations + references:
            if not paper.paper_id or paper.paper_id == paper_id or paper.paper_id in seen_paper_ids:
                continue
            if not paper.abstract:
                continue
            seen_paper_ids.add(paper.paper_id)
            merged_candidates.append(paper)

        new_papers: list[Paper] = []
        tasks = []
        for paper in merged_candidates:
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper, **kwargs))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_papers: list[Paper] = []
        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores):
            if score < 0.01:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

                kept_papers.append(paper)
                kept_scores.append(score)
                self.paper_pool.add_paper(paper, "expand", paper_pool_entry.paper.title, score)

        if not kept_papers:
            return 0.0

        top_k_scores = sorted(kept_scores, reverse=True)[: self.reward_top_k]
        if self.use_discrete_reward:
            top_k_scores = [1.0 if score >= self.reward_threshold else 0.0 for score in top_k_scores]
        return sum(top_k_scores) - self.expand_cost

    async def get_relevance_score(self, query: str, paper: Paper, **kwargs) -> float:
        prompt = SELECT_PROMPT.format(title=paper.title, abstract=paper.abstract, user_query=query)
        try:
            score = await self.selector_client.classify(prompt)
        except Exception as exc:
            logger.warning("Selector service failed for paper_id=%s: %r", paper.paper_id, exc)
            return 0.0
        return float(1.0 - score)
