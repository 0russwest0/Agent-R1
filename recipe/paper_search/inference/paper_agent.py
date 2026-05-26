"""Paper search agent for offline batch inference.

The multi-turn loop matches ``recipe.paper_search.paper_search_agent_flow.PaperSearchAgentFlow``
(search / expand / selector scoring). Policy tokens come from a local vLLM ``LLM`` engine
(``tokenizer.apply_chat_template(..., tools=...)`` + ``generate``), not an OpenAI HTTP server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

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
)
from recipe.paper_search.utils import Paper, PaperPool, SelectorClient
from recipe.paper_search.inference.inference_date_utils import parse_year_month_str
from recipe.paper_search.inference.inference_paper_client import InferencePaperClient
from verl.experimental.agent_loop.tool_parser import ToolParser


def _optional_year_month_cli(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return None
    parse_year_month_str(s)
    return s


class PaperSearchInferenceAgent:
    """Runs the same agent logic as training ``PaperSearchAgentFlow`` with in-process vLLM."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        tokenizer: AutoTokenizer,
        llm: LLM,
        tool_parser: ToolParser,
        **kwargs: Any,
    ) -> None:
        """Initialize the agent.

        Args:
            logger: Logger for this run.
            tokenizer: HF tokenizer (same model as ``llm``).
            llm: Initialized vLLM ``LLM`` engine.
            tool_parser: Parser for assistant generations (typically Hermes / recipe rollout format).
            **kwargs: Overrides for limits and costs (same keys as ``PaperSearchAgentFlow`` where applicable).
        """
        self.logger = logger
        self.thought_log_path: Optional[str] = kwargs.get("thought_log_path")
        self.tokenizer = tokenizer
        self.llm = llm
        self.tool_parser = tool_parser

        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.ordered_paper_ids: list[str] = []
        self.history_search_queries: dict[str, int] = {}
        self.history_actions: list[tuple[str, str]] = []
        self.user_query = ""
        self.steps: list[dict[str, Any]] = []

        self.max_steps: int = int(kwargs.get("max_steps", os.getenv("PAPER_SEARCH_INFERENCE_MAX_STEPS", "5")))
        self.max_parallel_calls: int = int(kwargs.get("max_parallel_calls", 5))
        self.reward_top_k: int = int(kwargs.get("reward_top_k", 3))
        self.search_cost: float = float(kwargs.get("search_cost", 0.0))
        self.expand_cost: float = float(kwargs.get("expand_cost", 0.0))
        self.search_top_k: int = int(kwargs.get("search_top_k", 10))
        self.citations_limit: int = int(kwargs.get("citations_limit", 30))
        self.references_limit: int = int(kwargs.get("references_limit", -1))
        self.search_year: str = str(kwargs.get("search_year", os.getenv("PAPER_SEARCH_SEARCH_YEAR", "-2024")))
        self.max_arxiv_yymm: int = int(kwargs.get("max_arxiv_yymm", os.getenv("PAPER_SEARCH_MAX_ARXIV_YYMM", "2410")))

        tpl_kw = kwargs.get("apply_chat_template_kwargs")
        if tpl_kw is None:
            raw = os.getenv("PAPER_SEARCH_APPLY_CHAT_TEMPLATE_KWARGS", "").strip()
            tpl_kw = json.loads(raw) if raw else {}
        self.apply_chat_template_kwargs: dict[str, Any] = tpl_kw

        resp_len = kwargs.get("response_length")
        if resp_len is None:
            resp_len = int(os.getenv("PAPER_SEARCH_INFERENCE_MAX_NEW_TOKENS", "8192"))
        self.response_length: int = int(resp_len)

        temp = float(kwargs.get("temperature", os.getenv("PAPER_SEARCH_INFERENCE_TEMPERATURE", "0.1")))
        self._sampling_params = SamplingParams(temperature=temp, max_tokens=self.response_length)

        _ps_base = kwargs.get("paper_search_base_url", os.getenv("PAPER_SEARCH_BASE_URL"))
        _search_src = (
            kwargs.get("search_source", os.getenv("PAPER_AGENT_V2_SEARCH_SOURCE", "local_db")).strip().lower()
        )
        if _search_src not in {"local_db", "google"}:
            raise ValueError("search_source / PAPER_AGENT_V2_SEARCH_SOURCE must be 'local_db' or 'google'")
        _from_m = _optional_year_month_cli(
            kwargs.get("paper_from_month", os.getenv("PAPER_AGENT_V2_PAPER_FROM"))
        )
        _to_m = _optional_year_month_cli(kwargs.get("paper_to_month", os.getenv("PAPER_AGENT_V2_PAPER_TO")))
        self.client = InferencePaperClient(
            base_url=_ps_base,
            search_source=_search_src,
            paper_from_month=_from_m,
            paper_to_month=_to_m,
            timeout=30.0,
        )
        _sel_base = kwargs.get("selector_base_url", os.getenv("PAPERSEARCH_SELECTOR_BASE_URL"))
        _sel_model = kwargs.get("selector_model_name", os.getenv("PAPERSEARCH_SELECTOR_MODEL_NAME"))
        self.selector_client = SelectorClient(base_url=_sel_base, model_name=_sel_model or None, timeout=30.0)

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

    def _build_prompt_ids(self, user_query: str) -> list[int]:
        messages = [
            {"role": "system", "content": PAPERSEARCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PAPERSEARCH_USER_PROMPT.format(
                    user_query=user_query,
                    paper_list=self.paper_pool.paper_list,
                    history_actions=self._format_history_actions(),
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tools=PAPERSEARCH_TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

    def _llm_generate_token_ids(self, prompt_token_ids: list[int]) -> list[int]:
        """Run sync vLLM generation; returns **new** token ids only."""
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling_params=self._sampling_params,
        )
        out = outputs[0]
        gen = out.outputs[0]
        token_ids = list(gen.token_ids)
        return token_ids[: self.response_length]

    def _summarize_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for tc in tool_calls:
            raw_args = getattr(tc, "arguments", "")
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except Exception:
                arguments = {"raw_arguments": raw_args}
            summaries.append({"name": tc.name, "arguments": arguments})
        return summaries

    def _write_thought_log(
        self,
        step_idx: int,
        thought_text: str,
        tool_call_summaries: list[dict[str, Any]],
    ) -> None:
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
        with open(self.thought_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _build_save_items(self) -> dict[str, Any]:
        ranked_entries = list(reversed(self.paper_pool.ranked_papers))
        save_items: dict[str, Any] = {
            "ordered_ids": list(self.ordered_paper_ids),
            "sorted_ids": [e.paper.paper_id for e in ranked_entries],
            "details": {},
        }
        for entry in ranked_entries:
            paper = entry.paper
            save_items["details"][paper.paper_id] = {
                "paper_id": paper.paper_id,
                "raw_paper_id": paper.raw_paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "year": paper.year,
                "score": entry.score,
                "source": entry.source,
                "origin": entry.origin,
                "expand": entry.expand,
            }
        return save_items

    def _is_before_arxiv_cutoff(self, paper: Paper) -> bool:
        arxiv_id = paper.arxiv_id or paper.paper_id
        prefix = arxiv_id.split("v", 1)[0].split(".", 1)[0]
        if len(prefix) == 4 and prefix.isdigit():
            return int(prefix) <= self.max_arxiv_yymm
        if paper.year is not None and int(paper.year) > 2024:
            return False
        return True

    async def get_relevance_score(self, query: str, paper: Paper, **_kwargs: Any) -> float:
        """Selector score in [0, 1], same transform as ``PaperSearchAgentFlow``."""
        prompt = SELECT_PROMPT.format(title=paper.title, abstract=paper.abstract, user_query=query)
        try:
            score = await self.selector_client.classify(prompt)
        except Exception as exc:
            self.logger.info("Selector failed for paper_id=%s: %r", paper.paper_id, exc)
            return 0.0
        return float(1.0 - score)

    async def search(self, query: str, **_kwargs: Any) -> float:
        if query in self.history_search_queries:
            return -0.5

        try:
            papers = await self.client.search(query=query, limit=self.search_top_k, year=self.search_year)
        except Exception as exc:
            self.logger.info("Error in search %s: %r", query, exc)
            self.history_search_queries[query] = 0
            return 0.0

        new_papers: list[Paper] = []
        tasks = []
        seen_paper_ids: set[str] = set()

        for paper in papers:
            if not paper.paper_id or paper.paper_id in seen_paper_ids:
                continue
            if not self._is_before_arxiv_cutoff(paper):
                continue
            seen_paper_ids.add(paper.paper_id)
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

            new_papers.append(paper)
            tasks.append(self.get_relevance_score(self.user_query, paper))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores):
            if score < 0.01:
                continue

            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

                self.paper_pool.add_paper(paper, "search", query, score)
                self.ordered_paper_ids.append(paper.paper_id)
            kept_scores.append(score)
            self.logger.info("[%.3f] %s", score, paper.title)

        self.history_search_queries[query] = len(kept_scores)
        if not kept_scores:
            return 0.0

        return sum(sorted(kept_scores, reverse=True)[: self.reward_top_k]) - self.search_cost

    async def expand(self, paper_id: str, **_kwargs: Any) -> float:
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
            self.logger.info("Error in expand %s: %r", paper_id, exc)
            return 0.0

        merged_candidates: list[Paper] = []
        seen_paper_ids: set[str] = set()
        for paper in citations + references:
            if not paper.paper_id or paper.paper_id == paper_id or paper.paper_id in seen_paper_ids:
                continue
            if not self._is_before_arxiv_cutoff(paper):
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
            tasks.append(self.get_relevance_score(self.user_query, paper))

        relevance_scores = await asyncio.gather(*tasks) if tasks else []

        kept_scores: list[float] = []
        for paper, score in zip(new_papers, relevance_scores):
            if score < 0.01:
                continue
            with self._paper_pool_lock:
                if self.paper_pool.has_paper(paper.paper_id):
                    continue

                self.paper_pool.add_paper(paper, "expand", paper_pool_entry.paper.title, score)
                self.ordered_paper_ids.append(paper.paper_id)
            kept_scores.append(score)
            self.logger.info("[%.3f] %s", score, paper.title)

        if not kept_scores:
            return 0.0

        return sum(sorted(kept_scores, reverse=True)[: self.reward_top_k]) - self.expand_cost

    async def _execute_tool_calls(self, tool_calls: list[Any]) -> tuple[float, list[dict[str, Any]]]:
        parsed: list[tuple[str, dict[str, Any]]] = []
        for tc in tool_calls:
            if tc.name not in PAPER_SEARCH_TOOL_NAMES:
                self.logger.info("Unknown tool call: %s", tc.name)
                continue

            tool_args = decode_tool_arguments(tc.name, tc.arguments)
            if not tool_args:
                self.logger.info("Bad tool arguments for %s: %r", tc.name, tc.arguments)
                continue
            parsed.append((tc.name, tool_args))

        tasks = []
        summaries: list[dict[str, Any]] = []
        for name, tool_args in parsed:
            summaries.append({"name": name, "arguments": tool_args})
            if name == "search":
                query = extract_search_query(tool_args)
                if query:
                    self.history_actions.append(("search", query))
                    tasks.append(self.search(query))
            elif name == "expand":
                paper_id = extract_expand_paper_id(tool_args)
                if paper_id:
                    self.history_actions.append(("expand", paper_id))
                    tasks.append(self.expand(paper_id))

        reward_total = sum(await asyncio.gather(*tasks)) if tasks else 0.0
        return reward_total, summaries

    async def run(self, user_query: str, save_path: str) -> list[str]:
        self.paper_pool = PaperPool()
        self._paper_pool_lock = threading.RLock()
        self.ordered_paper_ids = []
        self.history_search_queries = {}
        self.history_actions = []
        self.user_query = user_query
        self.steps = []

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            paper_list_before = self.paper_pool.paper_list
            prompt_ids = await asyncio.to_thread(self._build_prompt_ids, user_query)

            try:
                response_ids = await asyncio.to_thread(self._llm_generate_token_ids, prompt_ids)
            except Exception as exc:
                self.logger.info("vLLM generate failed: %s", exc)
                break

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            thought_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            tool_calls = tool_calls[: self.max_parallel_calls]
            thought_tool_calls = self._summarize_tool_calls(tool_calls)
            self._write_thought_log(num_steps - 1, thought_text, thought_tool_calls)

            if not tool_calls:
                self.steps.append(
                    {
                        "step_idx": num_steps - 1,
                        "tool_calls": [],
                        "paper_list_before": paper_list_before,
                        "paper_list_after": self.paper_pool.paper_list,
                        "reward_score": 0.0,
                    }
                )
                break

            self.logger.info("Step %d: %d tool call(s)", num_steps, len(tool_calls))
            tool_reward_score, tool_call_summaries = await self._execute_tool_calls(tool_calls)
            paper_list_after = self.paper_pool.paper_list

            self.steps.append(
                {
                    "step_idx": num_steps - 1,
                    "tool_calls": tool_call_summaries,
                    "paper_list_before": paper_list_before,
                    "paper_list_after": paper_list_after,
                    "reward_score": tool_reward_score,
                }
            )

            prev_count = len(self.paper_pool.papers)
            self.logger.info("Step %d done: papers in pool=%d", num_steps, prev_count)

        save_items = self._build_save_items()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_items, f, ensure_ascii=False, indent=4)

        return save_items["ordered_ids"]

    async def close(self) -> None:
        await self.selector_client.close()
        await self.client.close()


# Backward-compatible name for ``run_paper_agent``.
PaperSearchAgent = PaperSearchInferenceAgent
PaperSearchV2Agent = PaperSearchInferenceAgent
