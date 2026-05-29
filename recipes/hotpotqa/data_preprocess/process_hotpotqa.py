#!/usr/bin/env python3
# Copyright 2025 Agent-R1 Teams
#
# Download HotpotQA (distractor setting) and export:
# 1) train.parquet / validation.parquet for verl/Agent-R1 RLHFDataset (prompt + reward_model + data_source)
# 2) hpqa_corpus.jsonl — deduplicated wiki paragraphs from all contexts (for FAISS indexing, see env/build_index.py)
#
# Usage:
#   pip install datasets pyarrow pandas
#   python recipes/hotpotqa/data_preprocess/process_hotpotqa.py \
#       --output_dir data/corpus/hotpotqa \
#       --corpus_output_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl
#
# Cross-eval only (2WikiMultiHopQA + MuSiQue validation, no corpus rebuild):
#   python recipes/hotpotqa/data_preprocess/process_hotpotqa.py \
#       --skip_hotpotqa \
#       --include_cross_eval \
#       --skip_corpus

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

import pandas as pd

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit("Please install dependencies: pip install datasets pyarrow pandas") from e


def _row_to_agent_r1(
    ex: dict[str, Any],
    split: str,
    row_index: int,
) -> dict[str, Any]:
    """Single HotpotQA example -> verl RLHFDataset row."""
    qid = ex.get("_id", f"{split}_{row_index}")
    question = ex["question"].strip()
    answer = ex["answer"]
    if not isinstance(answer, str):
        answer = str(answer)

    # Match paper_search-style rollout: first user message is the task text (agent reads raw_prompt[0]["content"]).
    prompt = [{"role": "user", "content": question}]

    reward_model = {
        "ground_truth": answer,
        "style": "rule",
    }

    extra_info: dict[str, Any] = {
        "index": row_index,
        "question_id": qid,
        "split": split,
        "type": ex.get("type"),
        "level": ex.get("level"),
    }

    return {
        "data_source": "hotpotqa_distractor",
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _normalize_answers(answers: Any) -> list[str]:
    """Normalize answer payloads from MultiHopQA-style datasets.

    Args:
        answers: Raw answer value from the dataset. It can be a string, a
            sequence of strings, or another scalar value.

    Returns:
        A list of non-empty answer strings.
    """
    if isinstance(answers, str):
        answer = answers.strip()
        return [answer] if answer else []
    if isinstance(answers, Sequence):
        return [str(answer).strip() for answer in answers if str(answer).strip()]
    if answers is None:
        return []
    answer = str(answers).strip()
    return [answer] if answer else []


def _cross_eval_row_to_agent_r1(
    ex: dict[str, Any],
    data_source: str,
    split: str,
    row_index: int,
) -> dict[str, Any]:
    """Convert one cross-eval QA example to the HotpotQA AGENT_R1 row schema.

    Args:
        ex: Raw example from a MultiHopQA-style dataset.
        data_source: Metric/reward grouping name for this validation set.
        split: Source split name.
        row_index: Row index in the source split.

    Returns:
        A dict containing ``data_source``, ``prompt``, ``reward_model``, and
        ``extra_info`` columns for ``RLHFDataset``.

    Raises:
        ValueError: If the example has no question/query field.
    """
    question = (ex.get("query") or ex.get("question") or "").strip()
    if not question:
        raise ValueError(f"Missing question/query in {data_source} row {row_index}")

    qid = ex.get("query_id") or ex.get("id") or f"{data_source}_{split}_{row_index}"
    answers = _normalize_answers(ex.get("answers", ex.get("golden_answers", ex.get("answer"))))
    primary_answer = answers[0] if answers else ""
    prompt = [{"role": "user", "content": question}]
    reward_model = {
        "ground_truth": primary_answer,
        "style": "rule",
    }
    extra_info: dict[str, Any] = {
        "index": row_index,
        "question_id": qid,
        "split": split,
        "source_dataset": data_source,
        "answers": answers,
    }

    return {
        "data_source": data_source,
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _write_cross_eval_parquets(
    output_dir: str,
    hf_name: str,
    configs: list[str],
    split: str,
    max_samples: int,
) -> None:
    """Write cross-eval validation parquets in HotpotQA AGENT_R1 schema.

    Args:
        output_dir: Directory where converted parquet files are written.
        hf_name: HuggingFace dataset name.
        configs: Dataset configs to load from ``hf_name``.
        split: Split to load for each config.
        max_samples: If positive, keep at most this many rows per config.
    """
    for config_name in configs:
        print(f"Loading {hf_name} / {config_name} / {split} from HuggingFace...")
        dataset = load_dataset(hf_name, config_name, split=split)
        n_rows = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
        rows = [_cross_eval_row_to_agent_r1(dataset[i], config_name, split, i) for i in range(n_rows)]
        out_path = os.path.join(output_dir, f"{config_name}_{split}.parquet")
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"Wrote {n_rows} rows -> {out_path}")


def _iter_context_paragraphs(ex: dict[str, Any]):
    """Yield (title, sentences_list) for one HotpotQA example.

    HuggingFace `hotpot_qa` uses context: {title: [...], sentences: [[...], ...]}.
    Official JSON dumps use context: [[title, [sent, ...]], ...].
    """
    ctx = ex.get("context") or []
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        titles = ctx["title"]
        sents_block = ctx["sentences"]
        for title, sents in zip(titles, sents_block, strict=False):
            yield title, sents
    else:
        for item in ctx:
            title, sents = item[0], item[1]
            yield title, sents


def _contexts_to_corpus_entries(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flatten HotpotQA context paragraphs into {title, text} records."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for ex in examples:
        for title, sents in _iter_context_paragraphs(ex):
            text = " ".join(sents).strip()
            title = str(title).strip()
            key = (title, text)
            if not text or key in seen:
                continue
            seen.add(key)
            out.append({"title": title, "text": text})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA for AGENT_R1 / verl RL training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/corpus/hotpotqa",
        help="Directory for train/validation parquet (created if missing).",
    )
    parser.add_argument(
        "--hf_name",
        type=str,
        default="hotpot_qa",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="distractor",
        help="HuggingFace config (distractor = 10 paragraphs per question).",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=-1,
        help="If >0, only keep first N training examples (debug).",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=-1,
        help="If >0, only keep first N validation examples (debug).",
    )
    parser.add_argument(
        "--skip_corpus",
        action="store_true",
        help="Do not write hpqa_corpus.jsonl.",
    )
    parser.add_argument(
        "--corpus_output_path",
        type=str,
        default="data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl",
        help="Output path for hpqa_corpus.jsonl used by search tool/index builder.",
    )
    parser.add_argument(
        "--skip_hotpotqa",
        action="store_true",
        help="Skip HotpotQA train/validation parquet generation.",
    )
    parser.add_argument(
        "--include_cross_eval",
        action="store_true",
        help="Also generate 2WikiMultiHopQA/MuSiQue validation parquets for training-time validation.",
    )
    parser.add_argument(
        "--cross_eval_hf_name",
        type=str,
        default="corag/multihopqa",
        help="HuggingFace dataset containing cross-eval QA validation splits.",
    )
    parser.add_argument(
        "--cross_eval_configs",
        type=str,
        default="2wikimultihopqa,musique",
        help="Comma-separated dataset configs to convert for cross-eval validation.",
    )
    parser.add_argument(
        "--cross_eval_split",
        type=str,
        default="validation",
        help="Split name to convert for cross-eval datasets.",
    )
    parser.add_argument(
        "--max_cross_eval",
        type=int,
        default=-1,
        help="If >0, only keep first N cross-eval examples per dataset (debug).",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    ds = None
    if not args.skip_hotpotqa:
        print(f"Loading {args.hf_name} / {args.hf_config} from HuggingFace...")
        ds = load_dataset(args.hf_name, args.hf_config)

        for split_name, parquet_name in [("train", "train.parquet"), ("validation", "validation.parquet")]:
            if split_name not in ds:
                print(f"Skip split {split_name} (not in dataset).")
                continue
            split = ds[split_name]
            rows = []
            max_n = args.max_train if split_name == "train" else args.max_val
            n = len(split) if max_n <= 0 else min(len(split), max_n)
            for i in range(n):
                rows.append(_row_to_agent_r1(split[i], split_name, i))
            df = pd.DataFrame(rows)
            path = os.path.join(out_dir, parquet_name)
            df.to_parquet(path, index=False)
            print(f"Wrote {n} rows -> {path}")

    if args.include_cross_eval:
        configs = [name.strip() for name in args.cross_eval_configs.split(",") if name.strip()]
        _write_cross_eval_parquets(
            output_dir=out_dir,
            hf_name=args.cross_eval_hf_name,
            configs=configs,
            split=args.cross_eval_split,
            max_samples=args.max_cross_eval,
        )

    if not args.skip_corpus and ds is not None:
        corpus_path = os.path.abspath(os.path.expanduser(args.corpus_output_path))
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        all_examples: list[dict[str, Any]] = []
        for split_name, max_n in (
            ("train", args.max_train),
            ("validation", args.max_val),
        ):
            if split_name not in ds:
                continue
            sp = ds[split_name]
            n = len(sp) if max_n <= 0 else min(len(sp), max_n)
            for i in range(n):
                all_examples.append(sp[i])

        entries = _contexts_to_corpus_entries(all_examples)
        with open(corpus_path, "w", encoding="utf-8") as f:
            for rec in entries:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(entries)} deduplicated paragraphs -> {corpus_path}")
    elif not args.skip_corpus:
        print("Skip corpus generation because --skip_hotpotqa was set.")


if __name__ == "__main__":
    main()
