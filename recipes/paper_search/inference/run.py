"""Hydra entry point for paper search batch inference and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf

from recipes.paper_search.inference.agent import PaperSearchInferenceAgent
from recipes.paper_search.inference.evaluation import evaluate_all_thresholds
from recipes.paper_search.inference.retrieval_client import InferencePaperClient
from recipes.paper_search.runtime import PaperSearchRuntimeConfig

try:
    from tqdm.auto import tqdm
except ImportError:
    T = TypeVar("T")

    def tqdm(iterable: Iterable[T], **_kwargs: object) -> Iterable[T]:  # type: ignore[no-redef]
        return iterable


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = Path(path).expanduser()
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _count_text_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def _load_existing_ids(details_dir: Path, sample_prefix: str) -> set[int]:
    existing_ids: set[int] = set()
    if not details_dir.exists():
        return existing_ids
    for path in details_dir.glob(f"{sample_prefix}_*.json"):
        try:
            existing_ids.add(int(path.stem.rsplit("_", 1)[1]))
        except Exception:
            continue
    return existing_ids


def _get_logger(save_dir: Path, idx: int, sample_prefix: str) -> logging.Logger:
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sample_prefix}_{idx}.log"
    if log_path.exists():
        log_path.unlink()

    logger = logging.getLogger(f"paper_search_inference_{idx}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger


def _get_thought_log_path(save_dir: Path, idx: int, sample_prefix: str) -> Path:
    thought_log_dir = save_dir / "th_logs"
    thought_log_dir.mkdir(parents=True, exist_ok=True)
    thought_log_path = thought_log_dir / f"{sample_prefix}_{idx}.log"
    if thought_log_path.exists():
        thought_log_path.unlink()
    return thought_log_path


def _load_engine(cfg: DictConfig) -> tuple[object, object, object]:
    from transformers import AutoTokenizer
    from vllm import LLM

    from verl.experimental.agent_loop.tool_parser import ToolParser

    model_path = str(cfg.model.path).strip()
    if not model_path:
        raise RuntimeError("model.path must point to a HuggingFace model name or local checkpoint directory.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(cfg.model.trust_remote_code))
    tool_parser = ToolParser.get_tool_parser(str(cfg.model.tool_parser), tokenizer)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=int(cfg.model.tensor_parallel_size),
        gpu_memory_utilization=float(cfg.model.gpu_memory_utilization),
        max_model_len=int(cfg.model.max_model_len),
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    return tokenizer, llm, tool_parser


async def _run_single_query(
    *,
    logger: logging.Logger,
    tokenizer: object,
    llm: object,
    tool_parser: object,
    cfg: DictConfig,
    query: str,
    save_path: Path,
    thought_log_path: Path,
) -> None:
    runtime_config = PaperSearchRuntimeConfig(
        max_steps=int(cfg.agent.max_steps),
        max_parallel_calls=int(cfg.agent.max_parallel_calls),
        reward_top_k=int(cfg.agent.reward_top_k),
        score_threshold=float(cfg.agent.score_threshold),
        search_cost=float(cfg.agent.search_cost),
        expand_cost=float(cfg.agent.expand_cost),
        use_discrete_reward=bool(cfg.agent.use_discrete_reward),
        search_top_k=int(cfg.search.top_k),
        citations_limit=int(cfg.search.citations_limit),
        references_limit=int(cfg.search.references_limit),
        search_year=None if cfg.search.year is None else str(cfg.search.year),
        max_arxiv_yymm=None if cfg.search.max_arxiv_yymm is None else int(cfg.search.max_arxiv_yymm),
    )
    apply_chat_template_kwargs = OmegaConf.to_container(cfg.model.apply_chat_template_kwargs, resolve=True)
    agent = PaperSearchInferenceAgent(
        logger,
        tokenizer=tokenizer,
        llm=llm,
        tool_parser=tool_parser,
        paper_client=InferencePaperClient(
            base_url=str(cfg.services.paper_search_base_url),
            search_source=str(cfg.search.source),
            paper_from_month=None if cfg.search.paper_from_month is None else str(cfg.search.paper_from_month),
            paper_to_month=None if cfg.search.paper_to_month is None else str(cfg.search.paper_to_month),
            timeout=float(cfg.services.timeout),
            serper_search_url=str(cfg.search.serper_search_url),
        ),
        runtime_config=runtime_config,
        selector_base_url=str(cfg.services.selector_base_url),
        selector_model_name=str(cfg.services.selector_model_name) if cfg.services.selector_model_name else None,
        response_length=int(cfg.model.max_new_tokens),
        temperature=float(cfg.model.temperature),
        apply_chat_template_kwargs=dict(apply_chat_template_kwargs or {}),
        thought_log_path=thought_log_path,
    )
    try:
        await agent.run(query, save_path)
    finally:
        await agent.close()


def run_inference(cfg: DictConfig) -> None:
    dataset_path = Path(str(cfg.dataset.path)).expanduser()
    save_dir = Path(str(cfg.output.dir)).expanduser()
    details_dir = save_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "logs").mkdir(parents=True, exist_ok=True)
    (save_dir / "th_logs").mkdir(parents=True, exist_ok=True)

    tokenizer, llm, tool_parser = _load_engine(cfg)
    line_total = _count_text_lines(dataset_path)
    sample_prefix = str(cfg.output.sample_prefix)

    for retry_idx in range(int(cfg.inference.retry_rounds)):
        existing_ids = _load_existing_ids(details_dir, sample_prefix)
        with dataset_path.open("r", encoding="utf-8") as f:
            bar = tqdm(
                enumerate(f),
                total=line_total,
                desc=f"Paper search inference (round {retry_idx + 1}/{int(cfg.inference.retry_rounds)})",
                unit="line",
                dynamic_ncols=True,
            )
            for idx, line in bar:
                if idx in existing_ids:
                    continue
                line = line.strip()
                if not line:
                    continue

                query = json.loads(line)["question"]
                save_path = details_dir / f"{sample_prefix}_{idx}.json"
                if save_path.exists():
                    continue

                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(idx=idx, refresh=False)
                asyncio.run(
                    _run_single_query(
                        logger=_get_logger(save_dir, idx, sample_prefix),
                        tokenizer=tokenizer,
                        llm=llm,
                        tool_parser=tool_parser,
                        cfg=cfg,
                        query=query,
                        save_path=save_path,
                        thought_log_path=_get_thought_log_path(save_dir, idx, sample_prefix),
                    )
                )


def run_evaluation(cfg: DictConfig) -> dict:
    save_dir = Path(str(cfg.output.dir)).expanduser()
    output_path = save_dir / str(cfg.evaluation.output_file)
    return evaluate_all_thresholds(
        dataset_path=Path(str(cfg.dataset.path)).expanduser(),
        details_dir=save_dir / "details",
        output_path=output_path,
        thresholds=[float(value) for value in cfg.evaluation.thresholds],
        top_k_values=[int(value) for value in cfg.evaluation.top_k_values],
        sample_prefix=str(cfg.output.sample_prefix),
    )


@hydra.main(config_path=".", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    _load_env_file(None if cfg.env_file is None else str(cfg.env_file))
    output_dir = Path(str(cfg.output.dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    if bool(cfg.inference.enabled):
        run_inference(cfg)
    if bool(cfg.evaluation.enabled):
        run_evaluation(cfg)


if __name__ == "__main__":
    main()
