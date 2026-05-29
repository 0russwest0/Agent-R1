"""Evaluation utilities for paper search inference results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def normalize_arxiv_id(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    return arxiv_id.split("v", 1)[0]


def load_qa_pairs(file_path: Path) -> list[tuple[str, list[str]]]:
    qa_pairs: list[tuple[str, list[str]]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            qa_pairs.append((data["question"], data["answer_arxiv_id"]))
    return qa_pairs


def filter_recall_papers(details: dict[str, dict[str, Any]], score_threshold: float) -> list[str]:
    score_id_pairs: list[tuple[float, str]] = []
    for arxiv_id, paper in details.items():
        normalized_id = normalize_arxiv_id(arxiv_id)
        score = float(paper.get("score", 0.0) or 0.0)
        if normalized_id and score >= score_threshold:
            score_id_pairs.append((score, normalized_id))

    score_id_pairs.sort(key=lambda item: item[0], reverse=True)

    deduped_ids: list[str] = []
    seen_ids: set[str] = set()
    for _, arxiv_id in score_id_pairs:
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)
        deduped_ids.append(arxiv_id)
    return deduped_ids


def calc_precision_recall_f1(pred_list: list[str], gt_list: list[str]) -> tuple[float, float, float]:
    pred_set = {normalize_arxiv_id(arxiv_id) for arxiv_id in pred_list if arxiv_id}
    gt_set = {normalize_arxiv_id(arxiv_id) for arxiv_id in gt_list if arxiv_id}

    true_positive = len(pred_set & gt_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_threshold(
    *,
    qa_pairs: list[tuple[str, list[str]]],
    details_dir: Path,
    threshold: float,
    top_k_values: list[int],
    sample_prefix: str = "sample",
) -> dict[str, Any]:
    all_k_precisions = {k: [] for k in top_k_values}
    all_k_recalls = {k: [] for k in top_k_values}
    all_k_f1s = {k: [] for k in top_k_values}
    recalled_counts: list[int] = []
    missing_count = 0

    for idx, (_, answer) in enumerate(qa_pairs):
        save_path = details_dir / f"{sample_prefix}_{idx}.json"
        if not save_path.exists():
            missing_count += 1
            continue

        try:
            with save_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            recall_papers = filter_recall_papers(data.get("details", {}), threshold)
        except Exception:
            missing_count += 1
            continue

        recalled_counts.append(len(recall_papers))
        for k in top_k_values:
            precision, recall, f1 = calc_precision_recall_f1(recall_papers[:k], answer)
            all_k_precisions[k].append(precision)
            all_k_recalls[k].append(recall)
            all_k_f1s[k].append(f1)

    return {
        "evaluated_samples": len(qa_pairs) - missing_count,
        "missing_count": missing_count,
        "average_recalled_paper_count": float(np.mean(recalled_counts)) if recalled_counts else 0.0,
        "top_k": {
            str(k): {
                "precision": float(np.mean(all_k_precisions[k])) if all_k_precisions[k] else 0.0,
                "recall": float(np.mean(all_k_recalls[k])) if all_k_recalls[k] else 0.0,
                "f1": float(np.mean(all_k_f1s[k])) if all_k_f1s[k] else 0.0,
            }
            for k in top_k_values
        },
    }


def evaluate_all_thresholds(
    *,
    dataset_path: Path,
    details_dir: Path,
    output_path: Path,
    thresholds: list[float],
    top_k_values: list[int],
    sample_prefix: str = "sample",
) -> dict[str, Any]:
    qa_pairs = load_qa_pairs(dataset_path)
    results = {
        f"{threshold:.1f}": evaluate_threshold(
            qa_pairs=qa_pairs,
            details_dir=details_dir,
            threshold=threshold,
            top_k_values=top_k_values,
            sample_prefix=sample_prefix,
        )
        for threshold in thresholds
    }
    payload = {
        "dataset_path": str(dataset_path),
        "details_dir": str(details_dir),
        "thresholds": thresholds,
        "top_k_values": top_k_values,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
