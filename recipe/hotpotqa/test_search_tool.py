#!/usr/bin/env python3
"""
Smoke test for HotpotQASearchToolLegacy (local FAISS + BGE).

Run from repo root (same Python env as training):

  python recipe/hotpotqa/test_search_tool.py
  python recipe/hotpotqa/test_search_tool.py "Who founded Apple?"

Optional env vars (same as recipe/hotpotqa/utils.py):
  HOTPOTQA_CORPUS_DATA_ROOT defaults to <repo>/data/corpus/hotpotqa_corpus
  HOTPOTQA_EMBEDDING_DEVICE defaults to cpu; for BGE encoding e.g. cuda:0 (when not using HOTPOTQA_EMBEDDING_PER_WORKER_GPU)
  HOTPOTQA_EMBEDDING_PER_WORKER_GPU  if 1, agent worker i uses cuda:i (colocate with training GPUs)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from recipe.hotpotqa.utils import (  # noqa: E402
    DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
    HOTPOTQA_CORPUS_JSONL,
    HOTPOTQA_CORPUS_DATA_ROOT,
    HOTPOTQA_DATA_ROOT,
    HOTPOTQA_INDEX_BIN,
    HotpotQASearchToolLegacy,
    default_hotpotqa_embedding_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test HotpotQA FAISS search tool.")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the capital of France?",
        help="Search query (natural language).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
        help="Must match the model used when building index.bin.",
    )
    args = parser.parse_args()

    print(f"HOTPOTQA_CORPUS_DATA_ROOT = {HOTPOTQA_CORPUS_DATA_ROOT}")
    print(f"HOTPOTQA_DATA_ROOT alias  = {HOTPOTQA_DATA_ROOT}")
    print(f"index.bin exists     = {HOTPOTQA_INDEX_BIN.exists()}  ({HOTPOTQA_INDEX_BIN})")
    print(f"hpqa_corpus.jsonl    = {HOTPOTQA_CORPUS_JSONL.exists()}  ({HOTPOTQA_CORPUS_JSONL})")
    print(f"query                = {args.query!r}")
    print(f"embedding_model      = {args.embedding_model}")
    print(f"HOTPOTQA_EMBEDDING_DEVICE (env) = {default_hotpotqa_embedding_device()}")
    print("---")

    tool = HotpotQASearchToolLegacy(embedding_model_name=args.embedding_model)
    print(f"HotpotQASearchToolLegacy.embedding_devices (after normalize) = {tool.embedding_devices}")
    out = tool.execute({"query": args.query})
    print(f"success = {out.get('success')}")
    content = out.get("content", "")
    if out.get("success"):
        try:
            payload = json.loads(content)
            results = payload.get("results", [])
            print(f"n_results = {len(results)}")
            for i, text in enumerate(results, start=1):
                snippet = (text[:200] + "…") if len(str(text)) > 200 else text
                print(f"  [{i}] {snippet}")
        except json.JSONDecodeError:
            print("content (raw):", content)
    else:
        print("content (error):", content)


if __name__ == "__main__":
    main()
