# Quick Start

This quick start is a compact **sanity check** for the current Agent-R1 workflow. Its purpose is to verify that your environment, dataset path, Qwen3 model path, rollout engine, and tool loop are wired correctly.

## 1. Prepare a Tool Dataset

Use the GSM8K tool preprocessing script:

```bash
python3 examples/data_preprocess/gsm8k_tool.py --local_save_dir ~/data/gsm8k_tool
```

This produces:

- `~/data/gsm8k_tool/train.parquet`
- `~/data/gsm8k_tool/test.parquet`

## 2. Run the Sanity Check Script

Use the provided Qwen3 multi-step tool-use script:

```bash
bash examples/run_qwen3-4b_gsm8k_tool.sh
```

If needed, adjust the following values before running:

- `CUDA_VISIBLE_DEVICES`
- `actor_rollout_ref.model.path`
- dataset paths under `~/data/gsm8k_tool`

The script entrypoint is [`examples/run_qwen3-4b_gsm8k_tool.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/run_qwen3-4b_gsm8k_tool.sh), which launches `python3 -m agent_r1.trainer.main_agent_ppo`.

## 3. What to Do Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to understand the main training abstraction.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see how `AgentFlowBase`, `AgentEnvLoop`, and `ToolEnv` fit together.
- Continue to the [`Datasets and Algorithms`](../tutorials/datasets-and-algorithms.md) guide for StepPO, HotpotQA, Paper Search, ALFWorld, WebShop, and baseline scripts.
