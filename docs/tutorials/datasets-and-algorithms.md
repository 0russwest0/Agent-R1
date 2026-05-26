# Datasets, Data Processing, and Algorithms

Agent-R1 is designed as a framework rather than a single benchmark implementation. A task recipe only needs to provide the same training data contract, an optional environment/tool configuration, and an agent flow. Once those pieces are present, the same trainer can run multiple RL algorithms by changing Hydra overrides.

This page documents the datasets, preparation scripts, task recipes, and algorithm entry points included in this repository. StepPO is included as a first-class Agent-R1 recipe, while the remaining entries are generic baselines that can run on the same dataset and agent-flow contracts.

## Current Runnable Scripts

| Dataset | Main StepPO Script | Baseline Scripts | Data Preparation |
|---|---|---|---|
| GSM8K Tool | `examples/run_qwen3-4b_gsm8k_tool_steppo.sh` | GRPO, PPO, token GAE, RLOO, REINFORCE++ scripts under `examples/` | `examples/data_preprocess/gsm8k_tool.py` |
| HotpotQA | `examples/run_hotpotqa_steppo.sh` | `examples/hotpotqa/run_{grpo,gspo,token_adv,rloo,gigpo,reinforce_plus_plus}.sh` | `recipe/hotpotqa/prepare_hotpotqa_agent_r1.py` |
| Paper Search | `examples/run_papersearch_steppo.sh` | `examples/papersearch/run_{grpo,gspo,token_adv,rloo,gigpo,reinforce_plus_plus}.sh` | Built-in JSONL under `recipe/paper_search/inference/datasets/` + `recipe/paper_search/prepare_paper_search_agent_r1.py` |
| ALFWorld | `examples/run_alfworld_steppo.sh` | `examples/alfworld/run_{grpo,gspo,token_adv,rloo,gigpo,reinforce_plus_plus}.sh` | `recipe/alfworld/prepare_alfworld_agent_r1.py` |
| WebShop | `examples/run_webshop_steppo.sh` | `examples/webshop/run_{grpo,gspo,token_adv,rloo,gigpo,reinforce_plus_plus}.sh` | `recipe/webshop/prepare_webshop_agent_r1.py` |

The repository includes the Paper Search raw JSONL files from the StepPO recipe:

- `recipe/paper_search/inference/datasets/AutoScholarQuery/{train,dev,test,test_lt_5}.jsonl`
- `recipe/paper_search/inference/datasets/RealScholarQuery/test.jsonl`
- `recipe/paper_search/inference/datasets/test_case.jsonl`

HotpotQA, ALFWorld, and WebShop are larger benchmark/environment datasets. They are prepared through the included scripts but are not committed as generated parquet, retrieval indexes, environment caches, or copied game/product artifacts.

## Data Preparation

```bash
# GSM8K Tool
python3 examples/data_preprocess/gsm8k_tool.py --local_save_dir ~/data/gsm8k_tool

# HotpotQA + retrieval corpus
python3 recipe/hotpotqa/prepare_hotpotqa_agent_r1.py \
  --output_dir data/corpus/hotpotqa \
  --corpus_output_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl

# Optional HotpotQA FAISS assets
python3 recipe/hotpotqa/process_hotpotqa.py

# Paper Search from bundled AutoScholarQuery JSONL files
python3 recipe/paper_search/prepare_paper_search_agent_r1.py \
  --input_dir recipe/paper_search/inference/datasets/AutoScholarQuery \
  --output_dir data/pasa

# ALFWorld from local ALFWorld raw data
python3 recipe/alfworld/prepare_alfworld_agent_r1.py \
  --input_dir alfworld_data/json_2.1.1 \
  --output_dir data/alfworld

# WebShop small/full data
python3 recipe/webshop/prepare_webshop_agent_r1.py \
  --dataset_mode small \
  --input_dir webshop_data \
  --output_dir data/webshop
```

Then choose any supported algorithm script. The StepPO entry points are:

```bash
bash examples/run_qwen3-4b_gsm8k_tool.sh
bash examples/run_qwen3-4b_gsm8k_tool_steppo.sh
bash examples/run_hotpotqa_steppo.sh
bash examples/run_papersearch_steppo.sh
bash examples/run_alfworld_steppo.sh
bash examples/run_webshop_steppo.sh
```

All scripts accept additional Hydra overrides through `"$@"`, for example:

```bash
bash examples/run_hotpotqa_steppo.sh \
  actor_rollout_ref.model.path=/path/to/model \
  trainer.n_gpus_per_node=4
```

## Training Data Contract

Agent-R1 uses parquet files compatible with the veRL trainer. For agent tasks, each row should include:

| Field | Required | Meaning |
|---|---:|---|
| `data_source` | Yes | Dataset or benchmark name. |
| `prompt` | Yes | Chat messages passed to the tokenizer and rollout engine. |
| `ability` | Recommended | Task category used for logging and reward routing. |
| `reward_model` | Yes | Rule/model reward metadata, usually including `ground_truth`. |
| `extra_info` | Recommended | Split, index, raw question, raw answer, or task-specific metadata. |
| `agent_name` | Agent tasks | Agent flow name. `agent_env_loop` is the default runnable flow. |
| `env_kwargs` | Tool/env tasks | JSON string consumed by `AgentEnvLoop._create_env`. |

`examples/data_preprocess/gsm8k_tool.py` is the smallest complete reference. It stores the environment configuration in `env_kwargs`:

```json
{
  "env_type": "tool",
  "tools": ["calc_gsm8k_reward"],
  "tool_format": "hermes",
  "tools_kwargs": {"ground_truth": "<answer>"}
}
```

This is the key extension point for new datasets. A HotpotQA recipe can point to a retrieval tool, a WebShop recipe can point to a web-shopping environment server, and an ALFWorld recipe can point to a text-world environment. The trainer does not need a dataset-specific rewrite as long as the recipe emits the same fields and the configured agent flow knows how to step the environment.

## Dataset Recipe Pattern

Agent-R1-derived recipes generally use the following layout:

```text
recipe/<task>/
  base.yaml                  # task-specific Hydra defaults
  prepare_<task>_agent_r1.py # raw data -> parquet
  <task>_agent_flow.py       # maps prompts/actions/observations into Agent-R1 steps
  reward_fn.py               # task reward
  prompts.py                 # prompt templates
  utils.py                   # task helpers
  env/                       # optional environment service or wrappers
```

This repository now includes the following concrete recipes:

| Dataset / Environment | Data Processing | Environment / Flow | Notes |
|---|---|---|---|
| HotpotQA | `recipe/hotpotqa/prepare_hotpotqa_agent_r1.py`, `process_hotpotqa.py`, `build_retrieval_corpus.py`, `verify_dataset.py` | `HotpotQAAgentFlow` | Multi-hop QA with local retrieval. |
| Paper Search | `recipe/paper_search/prepare_paper_search_agent_r1.py`, bundled AutoScholarQuery/RealScholarQuery JSONL files, inference utilities | `PaperSearchAgentFlow` | Academic search and citation expansion. |
| ALFWorld | `recipe/alfworld/prepare_alfworld_agent_r1.py` | `AlfworldAgentFlow`, ALFWorld wrapper | Text-world embodied household tasks. |
| WebShop | `recipe/webshop/prepare_webshop_agent_r1.py`, index/artifact builders, environment server | `WebShopAgentFlow` | Web shopping navigation and scoring. |

These recipes demonstrate that the Agent-R1 abstraction is not tied to GSM8K. They add task-specific data preparation and environment logic, while reusing the same rollout, reward-loop, and trainer interfaces.

## Supported Algorithms

Agent-R1 supports StepPO as a composed recipe:

```bash
algorithm.adv_estimator=gae
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
```

The first setting computes credit assignment over the agent step timeline. The GSPO policy loss then uses sequence-level importance ratios for the complete generated action at each step.

The Agent-R1 trainer also routes `algorithm.adv_estimator` to the following estimators:

| `algorithm.adv_estimator` | Granularity | Critic Required | Typical Use |
|---|---|---:|---|
| `gae` | Step-level | Yes | PPO-style actor-critic over agent steps. |
| `token_gae` | Token-level | Yes | Token-level actor-critic baseline for multi-step rollouts. |
| `grpo` | Trajectory outcome | No | Group-relative outcome optimization. |
| `rloo` | Trajectory outcome | No | Leave-one-out baseline over multiple rollouts. |
| `reinforce_plus_plus` | Token return | No | REINFORCE++ baseline with KL in reward. |
| `reinforce_plus_plus_baseline` | Trajectory outcome | No | REINFORCE++ with prompt-level mean baseline. |
| `gigpo` | Trajectory + step group | No | Requires `anchor_obs` in the agent flow for step grouping. |

The policy objective is controlled separately by `actor_rollout_ref.actor.policy_loss.loss_mode`. Agent-R1 keeps this axis separate from advantage estimation so that a task recipe can compare different credit-assignment strategies without changing the environment.

For convenience, the `examples/run_*_steppo.sh` scripts set the StepPO combination directly. Other algorithms can still be selected by overriding `algorithm.adv_estimator` and, when needed, `actor_rollout_ref.actor.policy_loss.loss_mode`.
