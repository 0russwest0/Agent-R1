# ALFWorld Recipe

## Overview

This recipe trains an agent for ALFWorld household tasks in a TextWorld environment. The agent receives a task goal, observes admissible text actions, and executes one environment command per step.

Official project reference: https://github.com/alfworld/alfworld. Processed Agent-R1 data for this recipe is available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: ALFWorld agent rollout configuration.
- `alfworld_agent_flow.py`: Agent-R1 rollout loop for ALFWorld.
- `env/alfworld_wrapper.py`: Runtime wrapper around TextWorld/ALFWorld game files.
- `env/tool_executor.py`: Runtime executor for TextWorld game interaction.
- `data_preprocess/process_alfworld.py`: Converts raw ALFWorld `json_2.1.1` data into parquet files and runtime game assets.
- `prompts.py`: System prompt, user prompt, and tool schema.
- `reward_fn.py`: Rule reward for task success.
- `summarize_validation.py`: Utility for summarizing validation outputs.
- `examples/alfworld/*.sh`: Training launch scripts for PPO, StepPO, GRPO, RLOO, REINFORCE, GSPO, and GiGPO variants.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/alfworld/requirements.txt
```

ALFWorld and TextWorld can be sensitive to Python and system package versions. Use the same environment used for Agent-R1 training when installing these extras.

## Data And Resources

Expected processed files:

- `data/alfworld/train.parquet`
- `data/alfworld/valid_seen.parquet`
- `data/alfworld/valid_unseen.parquet`
- `data/alfworld/games/...`
- `data/alfworld/stats.json`

Each parquet row includes a task prompt, rule reward metadata, and `extra_info.game_relative_path`, which points to the copied `game.tw-pddl` under `data/alfworld/games`.

The recipe keeps the six supported ALFWorld task families used by the agent code: pick-and-place, examine-in-light, clean-and-place, heat-and-place, cool-and-place, and pick-two-and-place.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the ALFWorld files to the paths above. To regenerate from an official ALFWorld raw data directory for local testing:

```bash
python recipes/alfworld/data_preprocess/process_alfworld.py \
  --input_dir alfworld_data/json_2.1.1 \
  --output_dir data/alfworld
```

The script copies runtime files from each kept trial and writes train, valid-seen, and valid-unseen parquet files.

## Environment Setup

Set the data root before training if the processed data is not in the default location:

```bash
export ALFWORLD_DATA_ROOT="$(pwd)/data/alfworld"
```

The runtime wrapper loads one TextWorld game file per example. No separate HTTP service is required.

## Training Scripts

```bash
bash examples/alfworld/run_ppo.sh
bash examples/alfworld/run_steppo.sh
bash examples/alfworld/run_grpo.sh
bash examples/alfworld/run_rloo.sh
bash examples/alfworld/run_reinforce.sh
bash examples/alfworld/run_gspo.sh
bash examples/alfworld/run_gigpo.sh
```

Scripts accept trailing Hydra overrides through `"$@"`.

## Core Code Entry Points

- Rollout loop: `recipes/alfworld/alfworld_agent_flow.py`.
- Environment wrapper: `recipes/alfworld/env/alfworld_wrapper.py`.
- Prompt and tool schema: `recipes/alfworld/prompts.py`.
- Data preparation: `recipes/alfworld/data_preprocess/process_alfworld.py`.

## Outputs And Evaluation

Validation reports task success through the ALFWorld rule reward. Use `recipes/alfworld/summarize_validation.py` to aggregate validation output files when needed.

## References

- ALFWorld project: https://github.com/alfworld/alfworld
