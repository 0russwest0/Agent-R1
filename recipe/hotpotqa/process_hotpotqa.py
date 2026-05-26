#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
from FlagEmbedding import FlagAutoModel
import numpy as np

from recipe.hotpotqa.utils import DEFAULT_HOTPOTQA_EMBEDDING_MODEL

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "corpus" / "hotpotqa_corpus"
_DEFAULT_CORPUS_PATH = _DEFAULT_DATA_DIR / "hpqa_corpus.jsonl"


def _load_corpus_texts(corpus_path: Path) -> list[str]:
    corpus: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            corpus.append(f'{rec.get("title", "")} {rec.get("text", "")}'.strip())
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HotpotQA FAISS index (legacy-compatible).")
    parser.add_argument("--data_dir", type=str, default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--corpus_path", type=str, default=str(_DEFAULT_CORPUS_PATH))
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_HOTPOTQA_EMBEDDING_MODEL)
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help='Embedding devices. Examples: "cuda:0", "cuda:0,cuda:1", "cpu". Empty means library default.',
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--reuse_embeddings",
        action="store_true",
        help="Reuse existing hpqa_corpus.npy if present and skip encode_corpus.",
    )
    parser.add_argument(
        "--query_instruction",
        type=str,
        default="Represent this sentence for searching relevant passages: ",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus_path) if args.corpus_path else _DEFAULT_CORPUS_PATH
    emb_path = data_dir / "hpqa_corpus.npy"
    index_path = data_dir / "index.bin"

    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    os.makedirs(str(data_dir), exist_ok=True)
    vectors: np.ndarray

    if args.reuse_embeddings and emb_path.exists():
        print(f"[hotpotqa] reuse existing embeddings: {emb_path}")
        vectors = np.load(str(emb_path)).astype(np.float32)
    else:
        corpus = _load_corpus_texts(corpus_path)
        model_kwargs = {
            "query_instruction_for_retrieval": args.query_instruction,
        }
        if args.devices.strip():
            devices = [x.strip() for x in args.devices.split(",") if x.strip()]
            model_kwargs["devices"] = devices[0] if len(devices) == 1 else devices
        print(
            f"[hotpotqa] encoding corpus, n={len(corpus)}, batch_size={args.batch_size}, devices={args.devices or 'default'}"
        )
        model = FlagAutoModel.from_finetuned(args.embedding_model, **model_kwargs)
        try:
            vectors = model.encode_corpus(corpus, batch_size=int(args.batch_size))
        except TypeError:
            vectors = model.encode_corpus(corpus)
        vectors = np.asarray(vectors, dtype=np.float32)
        np.save(str(emb_path), vectors)
        print(f"[hotpotqa] saved embeddings to {emb_path}")

    dim = vectors.shape[-1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    print(f"[hotpotqa] saved index to {index_path}")

if __name__ == "__main__":
    main()