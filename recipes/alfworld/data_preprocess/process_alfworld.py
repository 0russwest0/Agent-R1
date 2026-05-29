#!/usr/bin/env python3
"""
Prepare ALFWorld TextWorld data for AGENT_R1 / verl training.

This script:
1. Reads raw ALFWorld data from a source directory.
2. Keeps only TextWorld-usable trials from train / valid_seen / valid_unseen.
3. Copies runtime assets into data/alfworld/games/.
4. Writes verl-compatible parquet files into data/alfworld/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

TASK_FAMILY_MAP = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
}

SPLIT_TO_DATASOURCE = {
    "train": "alfworld_train",
    "valid_seen": "alfworld_valid_seen",
    "valid_unseen": "alfworld_valid_unseen",
}

SPLIT_TO_OUTPUT = {
    "train": "train.parquet",
    "valid_seen": "valid_seen.parquet",
    "valid_unseen": "valid_unseen.parquet",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_goal_text(game_data: dict[str, Any], traj_data: dict[str, Any]) -> str:
    grammar_text = game_data.get("grammar")
    if isinstance(grammar_text, str):
        try:
            grammar_obj = json.loads(grammar_text)
            task_entries = grammar_obj.get("task") or []
            if task_entries and isinstance(task_entries[0], dict):
                rhs = str(task_entries[0].get("rhs", "")).strip()
                if rhs:
                    return rhs
        except json.JSONDecodeError:
            pass

    anns = (traj_data.get("turk_annotations") or {}).get("anns") or []
    if anns and isinstance(anns[0], dict):
        task_desc = str(anns[0].get("task_desc", "")).strip()
        if task_desc:
            return task_desc

    return str(traj_data.get("task_type", "")).strip()


def _copy_runtime_assets(
    split: str,
    trial_dir: Path,
    output_games_root: Path,
) -> str:
    task_dir = trial_dir.parent.name
    trial_name = trial_dir.name
    dest_dir = output_games_root / split / task_dir / trial_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("game.tw-pddl", "traj_data.json", "initial_state.pddl"):
        src = trial_dir / filename
        if src.exists():
            shutil.copy2(src, dest_dir / filename)

    game_path = dest_dir / "game.tw-pddl"
    return str(game_path.relative_to(output_games_root))


def _build_row(
    *,
    split: str,
    row_index: int,
    traj_data: dict[str, Any],
    goal_text: str,
    game_relative_path: str,
    trial_dir: Path,
) -> dict[str, Any]:
    task_id = str(traj_data.get("task_id") or trial_dir.name)
    task_type_raw = str(traj_data.get("task_type", "")).strip()
    task_family = TASK_FAMILY_MAP[task_type_raw]

    prompt = [{"role": "user", "content": goal_text}]
    reward_model = {
        "ground_truth": {"success": None},
        "style": "rule",
    }
    extra_info = {
        "index": row_index,
        "task_id": task_id,
        "split": split,
        "task_type_raw": task_type_raw,
        "task_family": task_family,
        "goal_text": goal_text,
        "game_relative_path": game_relative_path,
        "trial_dir": str(trial_dir),
    }

    return {
        "data_source": SPLIT_TO_DATASOURCE[split],
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def _process_split(
    input_root: Path,
    output_games_root: Path,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_dir = input_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    rows: list[dict[str, Any]] = []
    family_counter: Counter[str] = Counter()
    raw_type_counter: Counter[str] = Counter()

    row_index = 0
    for traj_path in sorted(split_dir.glob("*/trial_*/traj_data.json")):
        trial_dir = traj_path.parent
        game_path = trial_dir / "game.tw-pddl"
        if not game_path.exists():
            continue

        traj_data = _load_json(traj_path)
        task_type_raw = str(traj_data.get("task_type", "")).strip()
        if task_type_raw not in TASK_FAMILY_MAP:
            continue

        game_data = _load_json(game_path)
        if not bool(game_data.get("solvable")):
            continue

        goal_text = _extract_goal_text(game_data, traj_data)
        game_relative_path = _copy_runtime_assets(split, trial_dir, output_games_root)
        row = _build_row(
            split=split,
            row_index=row_index,
            traj_data=traj_data,
            goal_text=goal_text,
            game_relative_path=game_relative_path,
            trial_dir=trial_dir,
        )
        rows.append(row)
        row_index += 1

        family_counter[row["extra_info"]["task_family"]] += 1
        raw_type_counter[task_type_raw] += 1

    stats = {
        "rows": len(rows),
        "task_family_counts": dict(family_counter),
        "task_type_raw_counts": dict(raw_type_counter),
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ALFWorld TextWorld parquet data for Agent-R1.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="alfworld_data/json_2.1.1",
        help="Raw ALFWorld data root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/alfworld",
        help="Directory to write parquet files and runtime assets.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_games_root = output_root / "games"
    output_games_root.mkdir(parents=True, exist_ok=True)

    overall_stats: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": {},
    }

    for split in ("train", "valid_seen", "valid_unseen"):
        rows, stats = _process_split(input_root, output_games_root, split)
        df = pd.DataFrame(rows)
        out_path = output_root / SPLIT_TO_OUTPUT[split]
        df.to_parquet(out_path, index=False)
        overall_stats["splits"][split] = stats

        print(f"[{split}] wrote {len(df)} rows -> {out_path}")
        print(f"[{split}] task families: {stats['task_family_counts']}")

    stats_path = output_root / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
