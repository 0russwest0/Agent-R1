from __future__ import annotations

import itertools
import json
import math
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


DEFAULT_DATA_DIR = "webshop_data"
DEFAULT_INDEX_DIR = "data/webshop/index"
DEFAULT_SEED = 233

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean_option_value(value: Any) -> str:
    return normalize_text(str(value).strip().replace("/", " | "))


def normalize_options(raw_options: Any, *, dedupe: bool = True) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw_options, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for option_name, choices in raw_options.items():
        if not choices:
            continue
        name = normalize_text(option_name)
        normalized_choices: list[dict[str, Any]] = []
        seen: set[str] = set()
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            value = _clean_option_value(choice.get("value"))
            if not value:
                continue
            if dedupe:
                if value in seen:
                    continue
                seen.add(value)
            normalized_choices.append(
                {
                    "value": value,
                    "is_available": bool(choice.get("is_available", True)),
                    "is_selected": bool(choice.get("is_selected", False)),
                    "price": choice.get("price"),
                    "price_string": choice.get("price_string") or "",
                    "image": choice.get("image") or "",
                }
            )
        if normalized_choices:
            out[name] = normalized_choices
    return out


def product_price(item: dict[str, Any]) -> float:
    raw = str(item.get("pricing") or item.get("Price") or "")
    prices = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw.replace(",", ""))]
    return min(prices) if prices else 100.0


def product_attributes_text(item: dict[str, Any], attrs: dict[str, Any] | None = None) -> str:
    parts: list[str] = [
        item.get("name", ""),
        item.get("brand", ""),
        item.get("product_category", ""),
        item.get("category", ""),
        item.get("query", ""),
        item.get("full_description", ""),
    ]
    small_description = item.get("small_description")
    if isinstance(small_description, list):
        parts.extend(str(x) for x in small_description)
    elif small_description:
        parts.append(str(small_description))
    if attrs:
        parts.extend(attrs.get("attributes") or [])
    options = normalize_options(item.get("customization_options"))
    for option_name, choices in options.items():
        parts.append(option_name)
        parts.extend(choice["value"] for choice in choices)
    return normalize_text(" ".join(str(part) for part in parts if part))


def product_doc(item: dict[str, Any], attrs: dict[str, Any] | None = None) -> str:
    return product_attributes_text(item, attrs)


def load_products_and_attrs(data_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    root = Path(data_dir)
    items_path = root / "items_shuffle_1000.json"
    attrs_path = root / "items_ins_v2_1000.json"
    if not items_path.exists() or not attrs_path.exists():
        raise FileNotFoundError(
            f"Expected WebShop small files at {items_path} and {attrs_path}. "
            "Download items_shuffle_1000.json and items_ins_v2_1000.json first."
        )
    products = load_json(items_path)
    attrs = load_json(attrs_path)
    if not isinstance(products, list) or not isinstance(attrs, dict):
        raise ValueError("Invalid WebShop small data format.")
    return products, attrs


def build_goals(
    products: list[dict[str, Any]],
    attrs: dict[str, dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    for item in products:
        asin = str(item.get("asin") or "")
        attr_record = attrs.get(asin) or {}
        instruction = normalize_text(attr_record.get("instruction"))
        if not asin or not instruction:
            continue

        options = normalize_options(item.get("customization_options"), dedupe=False)
        option_names = list(options)
        if option_names:
            option_values = [[choice["value"] for choice in options[name]] for name in option_names]
            option_combos = itertools.product(*option_values)
        else:
            option_combos = [()]

        for combo in option_combos:
            goal_options = dict(zip(option_names, combo, strict=False))
            suffix = ""
            if goal_options:
                suffix = " with " + ", ".join(f"{name}: {value}" for name, value in goal_options.items())
            goal = {
                "asin": asin,
                "instruction": instruction + suffix,
                "base_instruction": instruction,
                "attributes": list(attr_record.get("attributes") or []),
                "instruction_attributes": list(attr_record.get("instruction_attributes") or []),
                "goal_options": goal_options,
                "category": item.get("category"),
                "query": item.get("query"),
                "price_upper": round(product_price(item) * 1.5 + 1.0, 2),
            }
            goals.append(goal)

    rng = random.Random(seed)
    rng.shuffle(goals)
    for i, goal in enumerate(goals):
        goal["goal_index"] = i
    return goals


@dataclass
class ProductIndex:
    products: list[dict[str, Any]]
    attrs: dict[str, dict[str, Any]]
    goals: list[dict[str, Any]]
    bm25: BM25Okapi
    tokenized_docs: list[list[str]]
    asin_to_product: dict[str, dict[str, Any]]
    asin_to_idx: dict[str, int]

    def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results: list[dict[str, Any]] = []
        for idx in ranked[:top_k]:
            if float(scores[idx]) <= 0 and results:
                break
            item = self.products[idx]
            results.append({"item": item, "score": float(scores[idx])})
        return results

    def get_product(self, asin: str) -> dict[str, Any] | None:
        return self.asin_to_product.get(str(asin or ""))

    def has_product(self, asin: str) -> bool:
        return str(asin or "") in self.asin_to_product

    @property
    def num_products(self) -> int:
        return len(self.products)

    def goal(self, goal_index: int) -> dict[str, Any]:
        if goal_index < 0 or goal_index >= len(self.goals):
            raise IndexError(f"goal_index out of range: {goal_index} (num_goals={len(self.goals)})")
        return self.goals[goal_index]


def build_product_index(
    products: list[dict[str, Any]],
    attrs: dict[str, dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
) -> ProductIndex:
    docs = [product_doc(item, attrs.get(str(item.get("asin") or ""))) for item in products]
    tokenized_docs = [tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    goals = build_goals(products, attrs, seed=seed)
    asin_to_product = {str(item["asin"]): item for item in products if item.get("asin")}
    asin_to_idx = {asin: i for i, asin in enumerate(asin_to_product)}
    return ProductIndex(
        products=products,
        attrs=attrs,
        goals=goals,
        bm25=bm25,
        tokenized_docs=tokenized_docs,
        asin_to_product=asin_to_product,
        asin_to_idx=asin_to_idx,
    )


def save_index(index: ProductIndex, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bm25.pkl").open("wb") as f:
        pickle.dump(index.bm25, f)
    with (out / "tokenized_docs.json").open("w", encoding="utf-8") as f:
        json.dump(index.tokenized_docs, f)
    with (out / "goals.json").open("w", encoding="utf-8") as f:
        json.dump(index.goals, f, ensure_ascii=False)
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_products": len(index.products),
                "num_goals": len(index.goals),
                "seed": DEFAULT_SEED,
            },
            f,
            indent=2,
        )


def load_product_index(
    *,
    data_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> ProductIndex:
    data_root = Path(data_dir or os.getenv("WEBSHOP_DATA_DIR", DEFAULT_DATA_DIR))
    index_root = Path(index_dir or os.getenv("WEBSHOP_INDEX_DIR", DEFAULT_INDEX_DIR))
    products, attrs = load_products_and_attrs(data_root)

    bm25_path = index_root / "bm25.pkl"
    goals_path = index_root / "goals.json"
    tokenized_path = index_root / "tokenized_docs.json"
    if bm25_path.exists() and goals_path.exists() and tokenized_path.exists():
        with bm25_path.open("rb") as f:
            bm25 = pickle.load(f)
        goals = load_json(goals_path)
        tokenized_docs = load_json(tokenized_path)
        asin_to_product = {str(item["asin"]): item for item in products if item.get("asin")}
        asin_to_idx = {asin: i for i, asin in enumerate(asin_to_product)}
        return ProductIndex(
            products=products,
            attrs=attrs,
            goals=goals,
            bm25=bm25,
            tokenized_docs=tokenized_docs,
            asin_to_product=asin_to_product,
            asin_to_idx=asin_to_idx,
        )

    return build_product_index(products, attrs, seed=seed)
