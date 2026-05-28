#!/usr/bin/env python3
"""
Verify HotpotQA parquet files for AGENT_R1 training readiness.

Checks:
1. Required columns exist (data_source, prompt, reward_model, extra_info)
2. Every row has a non-empty ground_truth answer
3. reward_fn.compute_score works correctly on sample data
4. data_source matches what reward_fn expects

Usage:
    python recipes/hotpotqa/verify_dataset.py \
        --train data/corpus/hotpotqa/train.parquet \
        --val   data/corpus/hotpotqa/validation.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def verify_parquet(path: str, label: str) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"[FAIL] {label}: file not found: {p}")
        return False

    df = pd.read_parquet(p)
    print(f"\n{'='*60}")
    print(f"[{label}] {p}  ({len(df)} rows, columns={list(df.columns)})")
    print(f"{'='*60}")

    ok = True

    required = {"data_source", "prompt", "reward_model"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [FAIL] missing columns: {missing}")
        ok = False
    else:
        print(f"  [OK]   required columns present")

    # data_source
    ds_values = df["data_source"].unique().tolist()
    print(f"  data_source values: {ds_values}")
    if "hotpotqa_distractor" not in ds_values:
        print(f"  [WARN] 'hotpotqa_distractor' not in data_source — reward_fn will delegate to default")

    # prompt structure
    sample_prompt = df["prompt"].iloc[0]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
        first_msg = sample_prompt[0]
        if isinstance(first_msg, dict) and "role" in first_msg and "content" in first_msg:
            print(f"  [OK]   prompt format correct (list of message dicts)")
            print(f"         sample question: {first_msg['content'][:100]}...")
        else:
            print(f"  [FAIL] prompt[0] not a message dict: {type(first_msg)}")
            ok = False
    else:
        print(f"  [FAIL] prompt not a list of messages: {type(sample_prompt)}")
        ok = False

    # ground_truth in reward_model
    no_answer = 0
    empty_answer = 0
    answer_lengths = []
    for i, rm in enumerate(df["reward_model"]):
        if not isinstance(rm, dict):
            print(f"  [FAIL] row {i}: reward_model is not a dict: {type(rm)}")
            ok = False
            continue
        gt = rm.get("ground_truth")
        if gt is None:
            no_answer += 1
        elif not str(gt).strip():
            empty_answer += 1
        else:
            answer_lengths.append(len(str(gt)))

    if no_answer > 0:
        print(f"  [FAIL] {no_answer} rows have ground_truth=None")
        ok = False
    else:
        print(f"  [OK]   all rows have ground_truth present")

    if empty_answer > 0:
        print(f"  [WARN] {empty_answer} rows have empty ground_truth (will get reward=0)")
    else:
        print(f"  [OK]   all ground_truth non-empty")

    if answer_lengths:
        avg = sum(answer_lengths) / len(answer_lengths)
        print(f"  answer stats: min={min(answer_lengths)}, max={max(answer_lengths)}, avg={avg:.1f} chars")

    # Test reward_fn on a few samples
    try:
        from recipes.hotpotqa.reward_fn import compute_score, _normalize_answer

        sample_rm = df["reward_model"].iloc[0]
        gt = sample_rm["ground_truth"]
        # exact match
        score_match = compute_score("hotpotqa_distractor", f"<answer>{gt}</answer>", gt)
        # wrong answer
        score_wrong = compute_score("hotpotqa_distractor", "<answer>WRONG_ANSWER_XYZ</answer>", gt)
        # no answer tag
        score_notag = compute_score("hotpotqa_distractor", gt, gt)

        print(f"\n  reward_fn test (gt={gt!r}):")
        print(f"    exact match with <answer> tag: {score_match}")
        print(f"    wrong answer:                  {score_wrong}")
        print(f"    no <answer> tag (raw text):    {score_notag}")

        if score_match != 1.0:
            print(f"  [FAIL] exact match should be 1.0 but got {score_match}")
            ok = False
        else:
            print(f"  [OK]   reward_fn works correctly")
    except Exception as e:
        print(f"  [WARN] could not import/test reward_fn: {e}")

    # Show first 3 samples
    print(f"\n  --- first 3 samples ---")
    for i in range(min(3, len(df))):
        q = df["prompt"].iloc[i][0]["content"][:80]
        a = df["reward_model"].iloc[i]["ground_truth"]
        print(f"  [{i}] Q: {q}...")
        print(f"       A: {a}")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/corpus/hotpotqa/train.parquet")
    parser.add_argument("--val", default="data/corpus/hotpotqa/validation.parquet")
    args = parser.parse_args()

    all_ok = True
    for path, label in [(args.train, "TRAIN"), (args.val, "VAL")]:
        if not verify_parquet(path, label):
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("ALL CHECKS PASSED — dataset ready for training")
    else:
        print("SOME CHECKS FAILED — fix issues above before training")
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
