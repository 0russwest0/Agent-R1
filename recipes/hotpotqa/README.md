# HotpotQA Recipe

## Overview

This recipe trains a multi-hop question-answering agent with a retrieval tool. The agent searches a FAISS index built from passage text and returns the final answer in the format expected by the HotpotQA reward function.

Official dataset references: https://hotpotqa.github.io/ and https://github.com/StonyBrookNLP/musique. Processed Agent-R1 train, validation, cross-eval, and retrieval assets for this recipe are available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: HotpotQA agent configuration.
- `hotpotqa_agent_flow.py`: Agent-R1 rollout loop for retrieval-based QA.
- `data_preprocess/process_hotpotqa.py`: Converts HotpotQA and optional cross-eval splits into Agent-R1 parquet files.
- `env/build_retrieval_corpus.py`: Builds retrieval corpora from local 2Wiki/MuSiQue-style raw files.
- `env/build_index.py`: Encodes the HotpotQA corpus and builds the FAISS index.
- `env/search_tool.py`: Runtime FAISS/BGE retrieval tool.
- `examples/hotpotqa/*.sh`: Training launch scripts for PPO, StepPO, GRPO, RLOO, REINFORCE, GSPO, and GiGPO variants.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/hotpotqa/requirements.txt
```

Building the retrieval index requires enough CPU/GPU memory to encode the corpus with the configured embedding model.

## Data And Resources

Expected processed files:

- `data/corpus/hotpotqa/train.parquet`
- `data/corpus/hotpotqa/validation.parquet`
- Optional cross-eval parquets such as `data/corpus/hotpotqa/2wikimultihopqa_validation.parquet` and `data/corpus/hotpotqa/musique_validation.parquet`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.npy`
- `data/corpus/hotpotqa_corpus/index.bin`

The FAISS index is searched by `recipes.hotpotqa.env.search_tool`. The default embedding model is `BAAI/bge-large-en-v1.5`, configurable with `HOTPOTQA_EMBEDDING_MODEL`.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the HotpotQA files to the paths above. To regenerate HotpotQA-style files from public sources for local testing:

```bash
python recipes/hotpotqa/data_preprocess/process_hotpotqa.py \
  --output_dir data/corpus/hotpotqa \
  --corpus_output_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --include_cross_eval

python recipes/hotpotqa/env/build_index.py \
  --data_dir data/corpus/hotpotqa_corpus \
  --corpus_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl
```

For local 2Wiki/MuSiQue corpus construction, use `recipes/hotpotqa/env/build_retrieval_corpus.py` with raw files placed under the paths expected by that script.

## Environment Setup

No separate HTTP service is required. The retrieval tool loads local corpus and FAISS files. Typical environment variables:

```bash
export HOTPOTQA_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

By default, retrieval uses `HOTPOTQA_EMBEDDING_DEVICE=cpu` unless configured otherwise.

## Training Scripts

```bash
bash examples/hotpotqa/run_ppo.sh
bash examples/hotpotqa/run_steppo.sh
bash examples/hotpotqa/run_grpo.sh
bash examples/hotpotqa/run_rloo.sh
bash examples/hotpotqa/run_reinforce.sh
bash examples/hotpotqa/run_gspo.sh
bash examples/hotpotqa/run_gigpo.sh
```

Scripts accept trailing Hydra overrides through `"$@"`.

## Core Code Entry Points

- Rollout loop: `recipes/hotpotqa/hotpotqa_agent_flow.py`.
- Retrieval utilities: `recipes/hotpotqa/env/search_tool.py`.
- Prompt and tool schema: `recipes/hotpotqa/prompts.py`.
- Reward: `recipes/hotpotqa/reward_fn.py`.

## Outputs And Evaluation

Validation uses normalized exact-match style reward against `reward_model.ground_truth`. Search behavior is controlled by `max_steps`, `max_parallel_calls`, and `force_first_search` in the recipe config.

## References

- HotpotQA: https://hotpotqa.github.io/
- MuSiQue: https://github.com/StonyBrookNLP/musique
