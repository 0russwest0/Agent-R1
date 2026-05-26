from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from recipe.webshop.env.data import DEFAULT_SEED, normalize_options, normalize_text, product_attributes_text, product_price

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is a progress nicety, not a hard dependency.
    tqdm = None


FULL_PRODUCTS_FILE = "items_shuffle.json"
FULL_ATTRS_FILE = "items_ins_v2.json"
FULL_HUMAN_INS_FILE = "items_human_ins.json"
PRICE_RANGE = [10.0 * i for i in range(1, 100)]


def iter_json_array(path: str | Path, *, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        started = False
        while True:
            if pos >= len(buf):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf = buf[pos:] + chunk
                pos = 0

            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if not started:
                    if pos < len(buf) and buf[pos] == "[":
                        started = True
                        pos += 1
                        continue
                if started and pos < len(buf) and buf[pos] == ",":
                    pos += 1
                    continue
                break

            if started and pos < len(buf) and buf[pos] == "]":
                return

            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    raise
                if pos > chunk_size:
                    buf = buf[pos:]
                    pos = 0
                buf += chunk
                continue
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object in JSON array at {path}, got {type(obj).__name__}")
            yield obj
            pos = end
            if pos > chunk_size:
                buf = buf[pos:]
                pos = 0


def load_full_attrs(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(data_dir) / FULL_ATTRS_FILE
    with path.open("r", encoding="utf-8") as f:
        attrs = json.load(f)
    if not isinstance(attrs, dict):
        raise ValueError(f"Invalid attrs file: {path}")
    return attrs


def load_human_instructions(data_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = Path(data_dir) / FULL_HUMAN_INS_FILE
    with path.open("r", encoding="utf-8") as f:
        human = json.load(f)
    if not isinstance(human, dict):
        raise ValueError(f"Invalid human instruction file: {path}")
    return {str(k): v for k, v in human.items() if isinstance(v, list)}


def _parse_price_range(raw: Any) -> tuple[list[float], str]:
    raw_text = str(raw or "")
    prices = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw_text.replace(",", ""))]
    if not prices:
        return [100.0], "$100.0"
    prices = prices[:2]
    if len(prices) == 1:
        return prices, f"${prices[0]}"
    return prices, f"${prices[0]} to ${prices[1]}"


def normalize_full_product(raw: dict[str, Any], attr_record: dict[str, Any] | None = None) -> dict[str, Any] | None:
    asin = str(raw.get("asin") or "").strip()
    if not asin or asin == "nan" or len(asin) > 10:
        return None

    pricing, price_tag = _parse_price_range(raw.get("pricing"))
    small_description = raw.get("small_description")
    bullet_points = small_description if isinstance(small_description, list) else [small_description] if small_description else []
    options = normalize_options(raw.get("customization_options"), dedupe=True)
    attrs = list((attr_record or {}).get("attributes") or [])

    product = {
        "asin": asin,
        "name": raw.get("name") or "",
        "Title": raw.get("name") or "",
        "Description": raw.get("full_description") or "",
        "BulletPoints": bullet_points,
        "small_description": bullet_points,
        "full_description": raw.get("full_description") or "",
        "category": raw.get("category"),
        "query": normalize_text(raw.get("query")),
        "product_category": raw.get("product_category"),
        "pricing": price_tag,
        "Price": price_tag,
        "_price": min(pricing),
        "Attributes": attrs,
        "customization_options": options,
        "options": {name: [choice["value"] for choice in choices] for name, choices in options.items()},
        "average_rating": raw.get("average_rating"),
        "total_reviews": raw.get("total_reviews"),
    }
    return product


def _price_upper(price: float, rng: random.Random) -> tuple[float, str]:
    candidates = [p for p in PRICE_RANGE if p > price][:4]
    if len(candidates) >= 2:
        _, upper = sorted(rng.sample(candidates, 2))
        return upper, f", and price lower than {upper:.2f} dollars"
    return 1000000.0, ""


def build_human_goals_for_product(
    product: dict[str, Any],
    human_records: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    for record in human_records:
        attributes = list(record.get("instruction_attributes") or [])
        if not attributes:
            continue
        price_upper, price_text = _price_upper(float(product.get("_price") or 100.0), rng)
        instruction = str(record.get("instruction") or "").strip().rstrip(".")
        if not instruction:
            continue
        goal_options = [normalize_text(x) for x in record.get("instruction_options") or [] if normalize_text(x)]
        goals.append(
            {
                "asin": product["asin"],
                "category": product.get("category"),
                "query": product.get("query"),
                "name": product.get("name"),
                "product_category": product.get("product_category"),
                "instruction": instruction + price_text,
                "instruction_text": instruction + price_text,
                "attributes": attributes,
                "instruction_attributes": attributes,
                "goal_options": goal_options,
                "price_upper": price_upper,
                "reward_mode": "webshop_full",
            }
        )
    return goals


def build_full_artifacts(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    seed: int = DEFAULT_SEED,
    limit_products: int | None = None,
    build_lucene: bool = True,
    threads: int = 8,
    force_rebuild_store: bool = False,
) -> dict[str, Any]:
    if build_lucene and shutil.which("javac") is None and not os.getenv("JAVA_HOME"):
        raise RuntimeError(
            "Pyserini/Lucene indexing requires a JDK. `java` alone is not enough; "
            "install a JDK or set JAVA_HOME so pyjnius can find javac."
        )

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    docs_dir = output_root / "lucene_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / "documents.jsonl"
    db_path = output_root / "products.sqlite"
    goals_path = output_root / "goals.json"
    meta_path = output_root / "meta.json"
    lucene_index_dir = output_root / "lucene_index"

    print(f"[WebShop full] input_dir={input_root.resolve()}", flush=True)
    print(f"[WebShop full] output_dir={output_root.resolve()}", flush=True)

    existing_store_ready = (
        not force_rebuild_store
        and limit_products is None
        and meta_path.exists()
        and db_path.exists()
        and goals_path.exists()
        and docs_path.exists()
    )
    if existing_store_ready:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        print("[WebShop full] found existing completed product store/goals/docs; reusing them.", flush=True)
        print(f"[WebShop full] products={int(meta.get('num_products', 0)):,}, goals={int(meta.get('num_goals', 0)):,}", flush=True)
        if build_lucene:
            print(f"[WebShop full] building Lucene index at {lucene_index_dir} with {threads} threads...", flush=True)
            _build_lucene_index(docs_dir=docs_dir, lucene_index_dir=lucene_index_dir, threads=threads)
            print(f"[WebShop full] Lucene index complete: {lucene_index_dir}", flush=True)
        else:
            print("[WebShop full] skipped Lucene indexing (--skip_lucene).", flush=True)
        return meta

    print("[WebShop full] loading attributes...", flush=True)
    attrs = load_full_attrs(input_root)
    print(f"[WebShop full] loaded attributes: {len(attrs):,}", flush=True)
    print("[WebShop full] loading human instructions...", flush=True)
    human = load_human_instructions(input_root)
    num_human_records = sum(len(records) for records in human.values())
    print(
        f"[WebShop full] loaded human instruction ASINs: {len(human):,}; records: {num_human_records:,}",
        flush=True,
    )
    rng = random.Random(seed)

    print("[WebShop full] initializing SQLite product store and Lucene document file...", flush=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE products (asin TEXT PRIMARY KEY, product_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    seen: set[str] = set()
    goals: list[dict[str, Any]] = []
    num_raw = 0
    num_products = 0
    batch: list[tuple[str, str]] = []
    products_iter: Iterator[dict[str, Any]] = iter_json_array(input_root / FULL_PRODUCTS_FILE)
    if tqdm is not None:
        products_iter = tqdm(
            products_iter,
            total=len(attrs),
            unit="product",
            dynamic_ncols=True,
            desc="[WebShop full] products",
        )

    print("[WebShop full] scanning products, writing SQLite rows, Lucene docs, and goals...", flush=True)
    with docs_path.open("w", encoding="utf-8") as docs_f:
        for raw in products_iter:
            num_raw += 1
            asin = str(raw.get("asin") or "").strip()
            if asin in seen:
                continue
            product = normalize_full_product(raw, attrs.get(asin) or {})
            if product is None:
                continue
            seen.add(product["asin"])
            num_products += 1

            if product["asin"] in human:
                goals.extend(build_human_goals_for_product(product, human[product["asin"]], rng=rng))

            batch.append((product["asin"], json.dumps(product, ensure_ascii=False)))
            if len(batch) >= 1000:
                conn.executemany("INSERT OR REPLACE INTO products VALUES (?, ?)", batch)
                conn.commit()
                batch.clear()

            contents = product_attributes_text(product, {"attributes": product.get("Attributes") or []})
            docs_f.write(json.dumps({"id": product["asin"], "contents": contents}, ensure_ascii=False) + "\n")

            if limit_products is not None and num_products >= limit_products:
                break

    print("[WebShop full] flushing SQLite product rows...", flush=True)
    if batch:
        conn.executemany("INSERT OR REPLACE INTO products VALUES (?, ?)", batch)
        conn.commit()

    print(f"[WebShop full] shuffling and writing goals: {len(goals):,}", flush=True)
    rng.shuffle(goals)
    for i, goal in enumerate(goals):
        goal["goal_index"] = i
    with goals_path.open("w", encoding="utf-8") as f:
        json.dump(goals, f, ensure_ascii=False)

    meta = {
        "dataset_mode": "full",
        "input_dir": str(input_root.resolve()),
        "output_dir": str(output_root.resolve()),
        "num_raw_products_seen": num_raw,
        "num_products": num_products,
        "num_goals": len(goals),
        "seed": seed,
        "sqlite_path": str(db_path),
        "lucene_docs": str(docs_path),
        "lucene_index": str(lucene_index_dir),
    }
    conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    conn.commit()
    conn.close()
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(
        "[WebShop full] product scan complete: "
        f"raw_seen={num_raw:,}, products={num_products:,}, goals={len(goals):,}",
        flush=True,
    )
    print(f"[WebShop full] wrote SQLite: {db_path}", flush=True)
    print(f"[WebShop full] wrote Lucene docs: {docs_path}", flush=True)
    print(f"[WebShop full] wrote goals: {goals_path}", flush=True)
    print(f"[WebShop full] wrote meta: {meta_path}", flush=True)

    if build_lucene:
        print(f"[WebShop full] building Lucene index at {lucene_index_dir} with {threads} threads...", flush=True)
        _build_lucene_index(docs_dir=docs_dir, lucene_index_dir=lucene_index_dir, threads=threads)
        print(f"[WebShop full] Lucene index complete: {lucene_index_dir}", flush=True)
    else:
        print("[WebShop full] skipped Lucene indexing (--skip_lucene).", flush=True)

    return meta


def _build_lucene_index(*, docs_dir: Path, lucene_index_dir: Path, threads: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(docs_dir),
        "--index",
        str(lucene_index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(cmd, check=True)


@dataclass
class FullProductIndex:
    goals: list[dict[str, Any]]
    db_path: Path
    lucene_index_dir: Path
    search_top_k: int = 50

    def __post_init__(self) -> None:
        # Pyserini imports OpenAI in encode submodules at import time; Lucene BM25 does not call it.
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            os.environ["OPENAI_API_KEY"] = "local-placeholder-pyserini-import-only"
        from pyserini.search.lucene import LuceneSearcher

        uri = f"file:{self.db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._searcher = LuceneSearcher(str(self.lucene_index_dir))
        self._product_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def num_products(self) -> int:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'num_products'").fetchone()
        if row:
            return int(row[0])
        row = self._conn.execute("SELECT COUNT(*) FROM products").fetchone()
        return int(row[0])

    def get_product(self, asin: str) -> dict[str, Any] | None:
        asin = str(asin or "")
        if asin in self._product_cache:
            self._product_cache.move_to_end(asin)
            return self._product_cache[asin]
        row = self._conn.execute("SELECT product_json FROM products WHERE asin = ?", (asin,)).fetchone()
        if not row:
            return None
        product = json.loads(row[0])
        self._product_cache[asin] = product
        if len(self._product_cache) > 20000:
            self._product_cache.popitem(last=False)
        return product

    def has_product(self, asin: str) -> bool:
        return self.get_product(str(asin or "")) is not None

    def search(self, query: str, *, top_k: int = 50) -> list[dict[str, Any]]:
        if not normalize_text(query):
            return []
        hits = self._searcher.search(query, k=top_k)
        out: list[dict[str, Any]] = []
        for hit in hits:
            item = self.get_product(hit.docid)
            if item is not None:
                out.append({"item": item, "score": float(hit.score)})
        return out

    def goal(self, goal_index: int) -> dict[str, Any]:
        if goal_index < 0 or goal_index >= len(self.goals):
            raise IndexError(f"goal_index out of range: {goal_index} (num_goals={len(self.goals)})")
        return self.goals[goal_index]


def load_full_product_index(*, index_dir: str | Path, search_top_k: int = 50) -> FullProductIndex:
    root = Path(index_dir)
    goals_path = root / "goals.json"
    db_path = root / "products.sqlite"
    lucene_index_dir = root / "lucene_index"
    missing = [str(p) for p in [goals_path, db_path, lucene_index_dir] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing full WebShop artifacts: {missing}")
    with goals_path.open("r", encoding="utf-8") as f:
        goals = json.load(f)
    return FullProductIndex(goals=goals, db_path=db_path, lucene_index_dir=lucene_index_dir, search_top_k=search_top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full WebShop SQLite store, goals, and Lucene index.")
    parser.add_argument("--input_dir", default="webshop_data_full")
    parser.add_argument("--output_dir", default="data/webshop_full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit_products", type=int, default=None)
    parser.add_argument("--threads", type=int, default=int(os.getenv("WEBSHOP_INDEX_THREADS", "8")))
    parser.add_argument("--skip_lucene", action="store_true")
    parser.add_argument("--force_rebuild_store", action="store_true")
    args = parser.parse_args()
    meta = build_full_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        limit_products=args.limit_products,
        build_lucene=not args.skip_lucene,
        threads=args.threads,
        force_rebuild_store=args.force_rebuild_store,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
