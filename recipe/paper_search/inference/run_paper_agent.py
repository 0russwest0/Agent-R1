"""Batch driver: load queries from JSONL, run ``PaperSearchInferenceAgent``, write result JSON per line index."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, TypeVar

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from tqdm.auto import tqdm
except ImportError:
    T = TypeVar("T")

    def tqdm(iterable: Iterable[T], **_kwargs: object) -> Iterable[T]:  # type: ignore[no-redef]
        """Fallback when tqdm is not installed."""
        return iterable


from transformers import AutoTokenizer
from vllm import LLM

import env_config  # noqa: F401
from paper_agent import PaperSearchInferenceAgent
from verl.experimental.agent_loop.tool_parser import ToolParser


def get_logger(save_dir: str, idx: int) -> logging.Logger:
    log_save_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_save_dir, exist_ok=True)

    log_path = os.path.join(log_save_dir, f"Falcon_{idx}.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger(f"Falcon_v2_{idx}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_thought_log_path(save_dir: str, idx: int) -> str:
    thought_log_save_dir = os.path.join(save_dir, "th_logs")
    os.makedirs(thought_log_save_dir, exist_ok=True)

    thought_log_path = os.path.join(thought_log_save_dir, f"Falcon_{idx}.log")
    if os.path.exists(thought_log_path):
        os.remove(thought_log_path)
    return thought_log_path


def _count_text_lines(path: str) -> int:
    """Count newline characters in a text file using a buffered binary read.

    Args:
        path: UTF-8 text file path (e.g. JSONL).

    Returns:
        Number of ``\\n`` bytes seen.
    """
    count = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def load_existing_ids(details_dir: str) -> set[int]:
    existing_ids: set[int] = set()
    if not os.path.exists(details_dir):
        return existing_ids

    for fname in os.listdir(details_dir):
        if not (fname.startswith("Falcon_") and fname.endswith(".json")):
            continue
        try:
            existing_ids.add(int(fname[:-5].split("_")[1]))
        except Exception:
            continue
    return existing_ids


def _load_engine_and_agent_factory() -> tuple[AutoTokenizer, LLM, ToolParser]:
    model_path = os.getenv("PAPER_SEARCH_INFERENCE_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError(
            "Set PAPER_SEARCH_INFERENCE_MODEL_PATH to a HuggingFace or local model directory "
            "(same checkpoint you train with)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tool_format = os.getenv("PAPER_SEARCH_TOOL_PARSER", "hermes").strip()
    tool_parser = ToolParser.get_tool_parser(tool_format, tokenizer)

    tp = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    max_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "10240"))
    gpu_mem = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        trust_remote_code=True,
    )
    return tokenizer, llm, tool_parser


async def run_single_query(
    logger: logging.Logger,
    tokenizer: AutoTokenizer,
    llm: LLM,
    tool_parser: ToolParser,
    query: str,
    save_path: str,
    *,
    thought_log_path: str | None = None,
) -> None:
    agent = PaperSearchInferenceAgent(
        logger,
        tokenizer=tokenizer,
        llm=llm,
        tool_parser=tool_parser,
        thought_log_path=thought_log_path,
    )
    try:
        await agent.run(query, save_path)
    finally:
        await agent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run paper search agent over a JSONL query file (vLLM in-process, recipe-aligned loop)."
    )
    parser.parse_args()

    save_dir = os.getenv("PAPER_AGENT_V2_SAVE_DIR", "")
    test_file_path = os.getenv("PAPER_AGENT_V2_DATASET", "")
    retry_rounds = int(os.getenv("PAPER_AGENT_V2_RETRY_ROUNDS", "10"))

    os.makedirs(os.path.join(save_dir, "details"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "th_logs"), exist_ok=True)

    line_total = _count_text_lines(test_file_path)
    tokenizer, llm, tool_parser = _load_engine_and_agent_factory()

    for retry_idx in range(retry_rounds):
        details_dir = os.path.join(save_dir, "details")
        existing_ids = load_existing_ids(details_dir)

        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                bar = tqdm(
                    enumerate(f),
                    total=line_total,
                    desc=f"Paper agent (round {retry_idx + 1}/{retry_rounds})",
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
                    save_path = os.path.join(details_dir, f"Falcon_{idx}.json")
                    if os.path.exists(save_path):
                        continue

                    if hasattr(bar, "set_postfix"):
                        bar.set_postfix(idx=idx, refresh=False)
                    logger = get_logger(save_dir, idx)
                    thought_log_path = get_thought_log_path(save_dir, idx)
                    asyncio.run(
                        run_single_query(
                            logger,
                            tokenizer,
                            llm,
                            tool_parser,
                            query,
                            save_path,
                            thought_log_path=thought_log_path,
                        )
                    )
        except Exception as exc:
            print(exc)
            continue
