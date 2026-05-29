# WebShop Recipe

## Overview

This recipe trains an agent for WebShop-style shopping tasks. The agent interacts with a local HTTP environment, searches products, clicks through product pages and options, and buys an item that satisfies the instruction.

Official project reference: https://github.com/princeton-nlp/WebShop. Processed Agent-R1 product catalog, goal set, index, and parquet files for this recipe are available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: WebShop agent rollout configuration.
- `webshop_agent_flow.py`: Agent-R1 rollout loop for WebShop.
- `data_preprocess/process_hotpotqa.py`: Converts WebShop goals into train/test parquet files.
- `env/server.py`: FastAPI environment server.
- `env/engine.py`: WebShop state transition logic.
- `env/catalog.py`: Small-mode product loading, goal construction, and BM25 index helpers.
- `env/full_catalog.py`: Full-mode SQLite/Lucene artifact builder and runtime index.
- `env/client.py`: Async client for the WebShop HTTP environment.
- `env/run_env_server.sh`: Environment server launcher.
- `env/build_full_artifacts.sh`: Full-mode artifact and parquet builder.
- `prompts.py`: System prompt, user prompt, and tool schema.
- `examples/webshop/*.sh`: Training launch scripts for PPO, StepPO, GRPO, RLOO, REINFORCE, GSPO, and GiGPO variants.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/webshop/requirements.txt
```

Full-mode Lucene indexing also requires a working JDK. If running inside conda, `env/run_env_server.sh` can derive `JAVA_HOME` and `JVM_PATH` from `CONDA_PREFIX`.

## Data And Resources

Expected processed files for full mode:

- `data/webshop_full/train.parquet`
- `data/webshop_full/test.parquet`
- `data/webshop_full/goals.json`
- `data/webshop_full/products.sqlite`
- `data/webshop_full/meta.json`
- Lucene index files under `data/webshop_full`

Expected raw full-mode files before artifact building:

- `webshop_data_full/items_shuffle.json`
- `webshop_data_full/items_ins_v2.json`
- `webshop_data_full/items_human_ins.json`

Small mode uses `webshop_data` plus a local BM25 index under `data/webshop/index`.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the WebShop files to the paths above. To regenerate full-mode artifacts from local raw WebShop files:

```bash
bash recipes/webshop/env/build_full_artifacts.sh
```

To prepare small-mode parquet files:

```bash
python recipes/webshop/data_preprocess/process_hotpotqa.py \
  --dataset_mode small \
  --input_dir webshop_data \
  --output_dir data/webshop
```

To prepare full-mode parquet files from an existing `goals.json`:

```bash
python recipes/webshop/data_preprocess/process_hotpotqa.py \
  --dataset_mode full \
  --input_dir webshop_data_full \
  --output_dir data/webshop_full \
  --goals_path data/webshop_full/goals.json
```

## Environment Setup

Start the WebShop environment server:

```bash
export WEBSHOP_DATASET_MODE=full
export WEBSHOP_DATA_DIR="$(pwd)/webshop_data_full"
export WEBSHOP_INDEX_DIR="$(pwd)/data/webshop_full"
bash recipes/webshop/env/run_env_server.sh
```

The default server is `http://127.0.0.1:4111`. Training scripts should point to that URL through their existing environment/config overrides.

For small mode:

```bash
export WEBSHOP_DATASET_MODE=small
export WEBSHOP_DATA_DIR="$(pwd)/webshop_data"
export WEBSHOP_INDEX_DIR="$(pwd)/data/webshop/index"
bash recipes/webshop/env/run_env_server.sh
```

## Training Scripts

```bash
bash examples/webshop/run_ppo.sh
bash examples/webshop/run_steppo.sh
bash examples/webshop/run_grpo.sh
bash examples/webshop/run_rloo.sh
bash examples/webshop/run_reinforce.sh
bash examples/webshop/run_gspo.sh
bash examples/webshop/run_gigpo.sh
```

Scripts accept trailing Hydra overrides through `"$@"`.

## Core Code Entry Points

- Rollout loop: `recipes/webshop/webshop_agent_flow.py`.
- Environment client: `recipes/webshop/env/client.py`.
- Environment server: `recipes/webshop/env/server.py`.
- Full-mode artifact builder: `recipes/webshop/env/full_catalog.py`.
- Reward: `recipes/webshop/reward_fn.py`.

## Outputs And Evaluation

The WebShop environment returns task score and purchase success through the rule reward path. Training outputs follow the common Agent-R1 trainer settings in the launch scripts.

## References

- WebShop project: https://github.com/princeton-nlp/WebShop
