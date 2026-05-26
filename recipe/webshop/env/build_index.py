#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from recipe.webshop.env.data import DEFAULT_SEED, build_product_index, load_products_and_attrs, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WebShop small BM25 index.")
    parser.add_argument("--input_dir", default="webshop_data", help="Directory with WebShop small JSON files.")
    parser.add_argument("--output_dir", default="data/webshop/index", help="Index output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Goal shuffle seed.")
    args = parser.parse_args()

    products, attrs = load_products_and_attrs(args.input_dir)
    index = build_product_index(products, attrs, seed=args.seed)
    save_index(index, args.output_dir)
    print(f"Wrote index -> {Path(args.output_dir).resolve()}")
    print(f"Products: {len(index.products)}")
    print(f"Goals: {len(index.goals)}")


if __name__ == "__main__":
    main()
