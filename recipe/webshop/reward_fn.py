from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def _metadata_str(value: Any) -> str:
    return "" if value is None else str(value)


def _metadata_int(value: Any) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict[str, Any]:
    if not str(data_source).startswith("webshop"):
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    extra_info = extra_info or {}
    runtime_info = extra_info.get("reward_extra_info", {}) if isinstance(extra_info, dict) else {}
    if not isinstance(runtime_info, dict):
        runtime_info = {}

    if "step_reward" in runtime_info:
        score = float(runtime_info["step_reward"])
    elif "final_reward" in runtime_info:
        score = float(runtime_info["final_reward"])
    else:
        score = float(runtime_info.get("step_env_reward", 0.0))
    task_score = float(runtime_info.get("task_score") or score)
    success = bool(runtime_info.get("success", score > 0.0))
    final_reward = float(runtime_info.get("final_reward", score))
    return {
        "score": score,
        "acc": 1.0 if success else 0.0,
        "success": success,
        "final_reward": final_reward,
        "task_score": task_score,
        "split": _metadata_str(runtime_info.get("split", extra_info.get("split"))),
        "goal_index": _metadata_int(runtime_info.get("goal_index", extra_info.get("goal_index"))),
        "asin": _metadata_str(runtime_info.get("asin", extra_info.get("asin"))),
        "selected_asin": _metadata_str(runtime_info.get("selected_asin")),
        "num_steps": _metadata_int(runtime_info.get("num_steps")),
    }
