from __future__ import annotations

from typing import Any

from verl.utils.reward_score import gsm8k


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    return float(
        gsm8k.compute_score(
            solution_str=solution_str,
            ground_truth=str(ground_truth),
            method="flexible",
            format_score=0.0,
            score=1.0,
        )
    )
