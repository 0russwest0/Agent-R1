import json
import os

import numpy as np


def load_qa_pairs(file_path: str) -> list[tuple[str, list[str]]]:
    qa_pairs: list[tuple[str, list[str]]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qa_pairs.append((data["question"], data["answer_arxiv_id"]))
    return qa_pairs


def normalize_arxiv_id(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    return arxiv_id.split("v")[0]


def filter_v2_recall_papers(details: dict[str, dict], score_threshold: float) -> list[str]:
    score_id_pairs: list[tuple[float, str]] = []
    for arxiv_id, paper in details.items():
        arxiv_id = normalize_arxiv_id(arxiv_id)
        score = float(paper.get("score", 0.0) or 0.0)
        if not arxiv_id:
            continue
        if score < score_threshold:
            continue
        score_id_pairs.append((score, arxiv_id))

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
    pred_set = {normalize_arxiv_id(aid) for aid in pred_list if aid}
    gt_set = {normalize_arxiv_id(aid) for aid in gt_list if aid}

    true_positive = len(pred_set & gt_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gt_set) if gt_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


if __name__ == "__main__":
    dataset_path = os.getenv("EVAL_V2_DATASET", "datasets/RealScholarQuery/test.jsonl")
    save_dir = os.getenv("EVAL_V2_SAVE_DIR", "results/realscholar/paper_agent_lewen_local_db_pspo_160step/details")
    score_threshold = float(os.getenv("EVAL_V2_SCORE_THRESHOLD", "0.0"))

    qa_pairs = load_qa_pairs(dataset_path)
    k_values = [10000, 100, 50, 25]
    all_k_precisions = {k: [] for k in k_values}
    all_k_recalls = {k: [] for k in k_values}
    all_k_f1s = {k: [] for k in k_values}
    recall_full_len: list[int] = []
    missing_count = 0

    for i, (_, answer) in enumerate(qa_pairs):
        save_path = os.path.join(save_dir, f"Falcon_{i}.json")
        if not os.path.exists(save_path):
            missing_count += 1
            continue

        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            details = data.get("details", {})
            recall_papers = filter_v2_recall_papers(details, score_threshold)

            recall_full_len.append(len(recall_papers))

            for k in k_values:
                pred_topk = recall_papers[:k]
                precision, recall, f1 = calc_precision_recall_f1(pred_topk, answer)
                all_k_precisions[k].append(precision)
                all_k_recalls[k].append(recall)
                all_k_f1s[k].append(f1)
        except Exception as exc:
            print(f"Error in {save_path}: {exc}")

    print(f"Evaluated samples: {sum(len(v) for v in all_k_precisions.values()) // len(k_values) if k_values else 0}")
    print(f"Missing result files: {missing_count}")
    print(f"Score threshold: {score_threshold}")

    for k in k_values:
        print(f"[Top-{k}] Precision: {np.mean(all_k_precisions[k]) if all_k_precisions[k] else 0.0}")
        print(f"[Top-{k}] Recall: {np.mean(all_k_recalls[k]) if all_k_recalls[k] else 0.0}")
        print(f"[Top-{k}] F1: {np.mean(all_k_f1s[k]) if all_k_f1s[k] else 0.0}\n")

    print(f"Average recalled paper count before top-k cut: {np.mean(recall_full_len) if recall_full_len else 0.0}")
