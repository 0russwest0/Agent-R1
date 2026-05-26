# Data

This directory is reserved for generated or downloaded local datasets. Large generated artifacts such as parquet files, retrieval indexes, model outputs, logs, and environment caches should stay local and are not meant to be committed.

Agent-R1 includes lightweight raw/example data where it is practical:

| Dataset | Included in Repository | Generated Output |
|---|---|---|
| GSM8K / GSM8K Tool | No raw copy; downloaded by `datasets` in `examples/data_preprocess/`. | `~/data/gsm8k*` by default. |
| Paper Search | Yes. JSONL files are included under `recipe/paper_search/inference/datasets/`. | `data/pasa/{train,test}.parquet`. |
| HotpotQA | No raw copy; downloaded from HuggingFace by the preparation script. | `data/corpus/hotpotqa/*.parquet` and optional retrieval corpus/index. |
| ALFWorld | No raw copy; prepare from local ALFWorld raw data. | `data/alfworld/*.parquet` plus copied game files. |
| WebShop | No raw copy; prepare from local WebShop product data or prebuilt goals. | `data/webshop*/{train,test}.parquet` plus environment artifacts. |

Paper Search conversion example:

```bash
python3 recipe/paper_search/prepare_paper_search_agent_r1.py \
  --input_dir recipe/paper_search/inference/datasets/AutoScholarQuery \
  --output_dir data/pasa
```

See `docs/tutorials/datasets-and-algorithms.md` for the full matrix.
