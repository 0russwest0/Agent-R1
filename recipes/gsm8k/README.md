# GSM8K Recipe

## Overview

This recipe trains math-reasoning agents on GSM8K-style grade-school word problems. It supports both a plain PPO data path and a tool-use AgentEnv path where the model can call `calc_gsm8k_reward` before producing the final answer.

Official dataset reference: https://huggingface.co/datasets/openai/gsm8k. Processed Agent-R1 data for this recipe is available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: Agent configuration for the GSM8K agent.
- `prompts.py`: Prompt templates for plain and agent runs.
- `reward_fn.py`: Recipe-local GSM8K rule reward wrapper.
- `gsm8k_agent_flow.py`: Recipe-local agent flow using the GSM8K tool environment.
- `env/gsm8k_tool_env.py`: Dynamically builds tool-use prompts during rollout.
- `data_preprocess/process_gsm8k.py`: Converts raw GSM8K examples into standard train/test parquet files.
- `data_preprocess/process_gsm8k_agent.py`: Converts raw GSM8K examples into agent train/test parquet files.
- `examples/gsm8k/run_grpo.sh`: GRPO training script using the agent data.
- `examples/gsm8k/run_ppo.sh`: PPO training script using the plain data.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/gsm8k/requirements.txt
```

## Data And Resources

Expected processed files:

- Plain PPO path: `$HOME/data/gsm8k/train.parquet` and `$HOME/data/gsm8k/test.parquet`.
- Agent path: `$HOME/data/gsm8k_agent/train.parquet` and `$HOME/data/gsm8k_agent/test.parquet`.

Each processed row follows the verl RLHFDataset style with `prompt`, `reward_model`, and `extra_info`. The agent version also stores structured `question` and `ground_truth` fields so `GSM8KToolEnv` can build prompts dynamically during rollout.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the GSM8K files to the paths above. To regenerate from the public GSM8K source for local testing:

```bash
python recipes/gsm8k/data_preprocess/process_gsm8k.py \
  --local_save_dir "$HOME/data/gsm8k"

python recipes/gsm8k/data_preprocess/process_gsm8k_agent.py \
  --local_save_dir "$HOME/data/gsm8k_agent"
```

Use `--local_dataset_path` if the raw dataset has already been downloaded locally.

## Environment Setup

No external environment server is required. The agent path uses `recipes.gsm8k.env.gsm8k_tool_env.GSM8KToolEnv`, tool format `hermes`, and the registered `calc_gsm8k_reward` tool.

## Training Scripts

```bash
bash examples/gsm8k/run_ppo.sh
bash examples/gsm8k/run_grpo.sh
```

Both scripts accept trailing Hydra overrides through `"$@"`, for example:

```bash
bash examples/gsm8k/run_grpo.sh trainer.total_epochs=1
```

## Core Code Entry Points

- Plain data conversion: `recipes/gsm8k/data_preprocess/process_gsm8k.py`.
- Agent data conversion: `recipes/gsm8k/data_preprocess/process_gsm8k_agent.py`.
- Prompt templates: `recipes/gsm8k/prompts.py`.
- Recipe reward: `recipes/gsm8k/reward_fn.py`.
- Agent configuration: `recipes/gsm8k/base.yaml`.
- Agent flow: `recipes/gsm8k/gsm8k_agent_flow.py`.

## Outputs And Evaluation

Training outputs follow the common Agent-R1 trainer configuration in the script overrides. Validation uses the test parquet configured as `data.val_files`.

## References

- GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k
