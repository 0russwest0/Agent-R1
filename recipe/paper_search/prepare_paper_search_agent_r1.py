#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _row(example: dict[str, Any], split: str, row_index: int) -> dict[str, Any]:
    question = str(example["question"]).strip()
    answers = example.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    arxiv_ids = example.get("answer_arxiv_id") or []
    if isinstance(arxiv_ids, str):
        arxiv_ids = [arxiv_ids]

    return {
        "data_source": f"paper_search_{split}",
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answers": [str(answer) for answer in answers],
                "answer_arxiv_id": [str(arxiv_id) for arxiv_id in arxiv_ids],
            },
        },
        "extra_info": {
            "index": row_index,
            "split": split,
            "qid": example.get("qid", f"{split}_{row_index}"),
            "question": question,
            "source_meta": example.get("source_meta", {}),
        },
    }


def _convert(input_path: Path, output_path: Path, split: str, max_samples: int) -> None:
    examples = _read_jsonl(input_path, max_samples=max_samples)
    rows = [_row(example, split, i) for i, example in enumerate(examples)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Wrote {len(rows)} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Paper Search parquet data for Agent-R1.")
    parser.add_argument(
        "--input_dir",
        default="recipe/paper_search/inference/datasets/AutoScholarQuery",
        help="Directory containing train/test JSONL files.",
    )
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--test_file", default="test.jsonl")
    parser.add_argument("--output_dir", default="data/pasa")
    parser.add_argument("--max_train", type=int, default=-1)
    parser.add_argument("--max_test", type=int, default=-1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _convert(input_dir / args.train_file, output_dir / "train.parquet", "train", args.max_train)
    _convert(input_dir / args.test_file, output_dir / "test.parquet", "test", args.max_test)


if __name__ == "__main__":
    main()
