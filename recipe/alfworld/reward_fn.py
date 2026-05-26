from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def _native_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _native_str(value: Any) -> str | None:
    value = _native_value(value)
    if value is None:
        return ""
    return str(value)


def _native_int(value: Any) -> int | None:
    value = _native_value(value)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _native_float(value: Any) -> float | None:
    value = _native_value(value)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    value = _native_value(value)
    if isinstance(value, bool):
        return value
    return bool(value)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict[str, Any]:
    if not str(data_source).startswith("alfworld"):
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    extra_info = extra_info or {}
    runtime_info = extra_info.get("reward_extra_info", {}) if isinstance(extra_info, dict) else {}
    if not isinstance(runtime_info, dict):
        runtime_info = {}

    success_flag = None
    if isinstance(ground_truth, dict):
        success_flag = _maybe_bool(ground_truth.get("success"))
    else:
        success_flag = _maybe_bool(ground_truth)

    if success_flag is None:
        success_flag = _maybe_bool(runtime_info.get("success"))
    if success_flag is None:
        success_flag = _maybe_bool(extra_info.get("success")) if isinstance(extra_info, dict) else None

    score = 1.0 if success_flag else 0.0

    result = {
        "score": float(score),
        "step_env_reward": _native_float(runtime_info.get("step_env_reward")) or 0.0,
        "dense_reward_sum": _native_float(runtime_info.get("dense_reward_sum")),
        "success": bool(success_flag),
        "num_steps": _native_int(runtime_info.get("num_steps")),
        "is_action_valid": bool(_maybe_bool(runtime_info.get("is_action_valid"))),
        "task_id": _native_str(runtime_info.get("task_id", extra_info.get("task_id"))),
        "split": _native_str(runtime_info.get("split", extra_info.get("split"))),
        "task_type_raw": _native_str(runtime_info.get("task_type_raw", extra_info.get("task_type_raw"))),
        "task_family": _native_str(runtime_info.get("task_family", extra_info.get("task_family"))),
    }
    return result
