# Recipes and Algorithms

Agent-R1 recipes are task-level integrations. A recipe usually owns the prompts, rollout-time agent flow, reward logic, environment wrapper, data preprocessing entry points, and task-specific dependency notes needed to run a benchmark without mixing that code into the core framework.

The launch scripts live under `examples/`. The intended pattern is:

```text
examples/<task>/run_ppo.sh
examples/<task>/run_steppo.sh
examples/<task>/run_grpo.sh
examples/<task>/run_rloo.sh
examples/<task>/run_reinforce.sh
examples/<task>/run_gspo.sh
examples/<task>/run_gigpo.sh
```

Not every task needs every script. PPO and StepPO use their own launch configuration, while several other algorithm variants share the GRPO-style script structure with different Hydra overrides.

GSM8K is intentionally kept as a lightweight test recipe. It provides only `examples/gsm8k/run_steppo.sh` for the single-turn sanity check and `examples/gsm8k/run_steppo_tool.sh` for the multi-turn ToolEnv check.

## Recipe Layout

Most recipes follow this shape:

```text
recipes/<task>/
├── README.md
├── requirements.txt
├── base.yaml
├── prompts.py
├── reward_fn.py
├── <task>_agent_flow.py
├── data_preprocess/
│   └── process_<task>.py
└── env/
    └── ...
```

`reward_fn.py` is present when the task has a standalone rule or model reward function. Some online environments compute rewards during interaction instead.

## Datasets and Environments

| Recipe | What it covers | Main entry points |
| --- | --- | --- |
| `gsm8k` | Grade-school math reasoning. Plain GSM8K is kept as a single-turn sanity check, while GSM8K + Tool is the minimal `ToolEnv + BaseTool` example with recipe-local `calc_gsm8k_reward`. | `data_preprocess/process_gsm8k.py`, `data_preprocess/process_gsm8k_tool.py`, `tool.py` |
| `hotpotqa` | Multi-hop question answering with a retrieval environment. The recipe keeps preprocessing separate from retrieval-index construction. | `data_preprocess/process_hotpotqa.py`, `env/build_retrieval_corpus.py`, `env/build_index.py`, `hotpotqa_agent_flow.py` |
| `alfworld` | Text-based household task completion through an ALFWorld-style environment wrapper and tool executor. | `data_preprocess/process_alfworld.py`, `env/alfworld_wrapper.py`, `env/tool_executor.py`, `alfworld_agent_flow.py` |
| `webshop` | Shopping-agent training with a local WebShop environment server and catalog artifacts. | `data_preprocess/process_webshop.py`, `env/run_env_server.sh`, `env/full_catalog.py`, `webshop_agent_flow.py` |
| `paper_search` | Academic paper-search agents that query a paper service, expand citation/reference neighborhoods, and use a selector service for scoring. | `paper_search_agent_flow.py`, `runtime.py`, `env/paper_client.py`, `inference/run.py`, `inference/evaluation.py` |

Processed datasets are available from the Agent-R1 data release on [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data). After downloading the release, place or symlink each task's files to the paths listed in its recipe README. Recipe preprocessing scripts are still kept for local regeneration or format inspection when public raw data is available.

## Algorithms

| Algorithm | Script | Notes |
| --- | --- | --- |
| PPO | `run_ppo.sh` | Actor-critic baseline. |
| GRPO | `run_grpo.sh` | Group-relative policy optimization baseline used as the common script base for several variants. |
| [StepPO](https://arxiv.org/abs/2604.18401) | `run_steppo.sh` | Step-aligned policy optimization for multi-step agent trajectories. |
| RLOO | `run_rloo.sh` | Leave-one-out baseline variant. |
| REINFORCE | `run_reinforce.sh` | Step-level critic-free policy-gradient variant. |
| GSPO | `run_gspo.sh` | GRPO-family algorithm variant exposed through script overrides. |
| GiGPO | `run_gigpo.sh` | GRPO-family algorithm variant exposed through script overrides. |

## Where to Start

- Use `recipes/<task>/README.md` for task-specific dependencies, resource expectations, and environment setup.
- Download processed data from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data) before running the full recipe scripts.
- Use `examples/<task>/run_*.sh` for launch commands and Hydra overrides.
- Use `recipes/<task>/base.yaml` to see the recipe-local rollout configuration.
- Use `data_preprocess/process_*.py` only when you need to regenerate or inspect dataset formatting.
