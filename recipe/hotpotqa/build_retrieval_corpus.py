#!/usr/bin/env python3
"""Build hpqa_corpus.jsonl for 2WikiMultiHopQA and MuSiQue from downloaded raw data.

Usage (conda env ``steppo``, repo root)::

    python recipe/hotpotqa/build_retrieval_corpus.py --dataset all

Then build FAISS indexes (requires GPU for 2Wiki ~5.9M paragraphs)::

    export PYTHONPATH=$(pwd)
    EMB=${HOTPOTQA_EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}

    python recipe/hotpotqa/process_hotpotqa.py \\
        --data_dir data/corpus/musique_corpus \\
        --corpus_path data/corpus/musique_corpus/hpqa_corpus.jsonl \\
        --embedding_model "$EMB" --devices cuda:0 --batch_size 1024

    python recipe/hotpotqa/process_hotpotqa.py \\
        --data_dir data/corpus/2wikimultihopqa_corpus \\
        --corpus_path data/corpus/2wikimultihopqa_corpus/hpqa_corpus.jsonl \\
        --embedding_model "$EMB" --devices cuda:0 --batch_size 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_CORPUS_ROOT = _REPO_ROOT / "data" / "corpus"


def _write_corpus_entry(fout, title: str, text: str) -> None:
    """Write one deduplicated corpus line if text is non-empty."""
    title = str(title).strip()
    text = str(text).strip()
    if not text:
        return
    fout.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")


def iter_2wiki_para_records(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (title, text) from 2Wiki ``para_with_hyperlink.jsonl``."""
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            title = str(rec.get("title", "")).strip()
            sentences = rec.get("sentences") or []
            if isinstance(sentences, list):
                text = " ".join(str(s).strip() for s in sentences if str(s).strip()).strip()
            else:
                text = str(sentences).strip()
            if not text:
                continue
            yield title, text
            if line_no % 500_000 == 0:
                print(f"[2wiki] scanned {line_no:,} lines...")


def build_2wiki_corpus(
    input_path: Path,
    output_path: Path,
    *,
    dedupe: bool = False,
) -> int:
    """Convert 2Wiki paragraph jsonl to ``hpqa_corpus.jsonl``.

    Args:
        input_path: Path to ``para_with_hyperlink.jsonl``.
        output_path: Destination ``hpqa_corpus.jsonl``.
        dedupe: If True, drop duplicate (title, text) pairs (uses more RAM).

    Returns:
        Number of paragraphs written.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"2Wiki corpus not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] | None = set() if dedupe else None
    count = 0

    print(f"[2wiki] reading {input_path}")
    print(f"[2wiki] writing {output_path}")
    with output_path.open("w", encoding="utf-8") as fout:
        for title, text in iter_2wiki_para_records(input_path):
            key = (title, text)
            if seen is not None:
                if key in seen:
                    continue
                seen.add(key)
            _write_corpus_entry(fout, title, text)
            count += 1
            if count % 500_000 == 0:
                print(f"[2wiki] wrote {count:,} paragraphs...")

    print(f"[2wiki] done: {count:,} paragraphs -> {output_path}")
    return count


def iter_musique_paragraphs(data_dir: Path, *, include_full: bool) -> Iterator[tuple[str, str]]:
    """Yield (title, text) from MuSiQue official jsonl files under ``data/``."""
    patterns = ["musique_ans_v1.0_*.jsonl"]
    if include_full:
        patterns.append("musique_full_v1.0_*.jsonl")

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(data_dir.glob(pattern)))

    if not paths:
        raise FileNotFoundError(f"No MuSiQue jsonl files under {data_dir}")

    for path in paths:
        print(f"[musique] reading {path.name}")
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                ex: dict[str, Any] = json.loads(line)
                for para in ex.get("paragraphs") or []:
                    if not isinstance(para, dict):
                        continue
                    title = str(para.get("title", "")).strip()
                    text = str(para.get("paragraph_text", "")).strip()
                    if text:
                        yield title, text


def build_musique_corpus(
    data_dir: Path,
    output_path: Path,
    *,
    include_full: bool = True,
) -> int:
    """Extract deduplicated paragraphs from MuSiQue jsonl into ``hpqa_corpus.jsonl``.

    Args:
        data_dir: Directory containing ``musique_*_v1.0_*.jsonl``.
        output_path: Destination ``hpqa_corpus.jsonl``.
        include_full: Include ``musique_full_*`` splits for a larger retrieval pool.

    Returns:
        Number of unique paragraphs written.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"MuSiQue data dir not found: {data_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    count = 0

    print(f"[musique] writing {output_path}")
    with output_path.open("w", encoding="utf-8") as fout:
        for title, text in iter_musique_paragraphs(data_dir, include_full=include_full):
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            _write_corpus_entry(fout, title, text)
            count += 1
            if count % 50_000 == 0:
                print(f"[musique] wrote {count:,} unique paragraphs...")

    print(f"[musique] done: {count:,} paragraphs -> {output_path}")
    return count


def main() -> None:
    """CLI entry: build one or both retrieval corpora from raw downloads."""
    parser = argparse.ArgumentParser(
        description="Build hpqa_corpus.jsonl for 2Wiki / MuSiQue retrieval indexes.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=("2wikimultihopqa", "musique", "all"),
        help="Which corpus to build.",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(_DEFAULT_RAW_ROOT),
        help="Root directory with raw/2wikimultihopqa and raw/musique.",
    )
    parser.add_argument(
        "--corpus_root",
        type=str,
        default=str(_DEFAULT_CORPUS_ROOT),
        help="Output root under data/corpus/.",
    )
    parser.add_argument(
        "--wiki2_input",
        type=str,
        default="",
        help="Override path to para_with_hyperlink.jsonl.",
    )
    parser.add_argument(
        "--musique_data_dir",
        type=str,
        default="",
        help="Override MuSiQue data/ directory with jsonl files.",
    )
    parser.add_argument(
        "--wiki2_dedupe",
        action="store_true",
        help="Deduplicate 2Wiki paragraphs (not recommended for ~6M lines; high RAM).",
    )
    parser.add_argument(
        "--musique_skip_full",
        action="store_true",
        help="Only use musique_ans_* jsonl (smaller corpus).",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    corpus_root = Path(args.corpus_root).expanduser().resolve()

    if args.dataset in ("2wikimultihopqa", "all"):
        wiki_in = (
            Path(args.wiki2_input).expanduser().resolve()
            if args.wiki2_input
            else raw_root / "2wikimultihopqa" / "para_with_hyperlink.jsonl"
        )
        wiki_out = corpus_root / "2wikimultihopqa_corpus" / "hpqa_corpus.jsonl"
        build_2wiki_corpus(wiki_in, wiki_out, dedupe=args.wiki2_dedupe)

    if args.dataset in ("musique", "all"):
        musique_dir = (
            Path(args.musique_data_dir).expanduser().resolve()
            if args.musique_data_dir
            else raw_root / "musique" / "data"
        )
        musique_out = corpus_root / "musique_corpus" / "hpqa_corpus.jsonl"
        build_musique_corpus(
            musique_dir,
            musique_out,
            include_full=not args.musique_skip_full,
        )


if __name__ == "__main__":
    main()
