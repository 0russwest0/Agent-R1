# Quick Start

This quick start is a **sanity check**, not the main Agent-R1 workflow. Its purpose is to verify that your environment, dataset path, model path, and training stack are wired correctly.

## 1. Prepare a Minimal Dataset

The processed Agent-R1 datasets are available on [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data). Download the release and place or symlink the GSM8K files to `~/data/gsm8k`, or use the GSM8K preprocessing script to regenerate the sanity-check data locally:

```bash
pip install modelscope
modelscope download --dataset Melmaphother/Agent-R1-data --local_dir data/agent-r1-data
```

You can also clone the dataset repository with git:

```bash
git lfs install
git clone https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data.git data/agent-r1-data
```

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k --local_save_dir ~/data/gsm8k
```

This produces:

- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

## 2. Run the Sanity Check Script

Use the provided single-step script:

```bash
bash examples/gsm8k/run_steppo.sh
```

If needed, adjust the following values before running:

- `CUDA_VISIBLE_DEVICES`
- `actor_rollout_ref.model.path`
- dataset paths under `~/data/gsm8k`

The script entrypoint is [`examples/gsm8k/run_steppo.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/gsm8k/run_steppo.sh), which launches `python3 -m agent_r1.trainer.main_agent_ppo` with the StepPO-style `gae` estimator.

## 3. What to Do Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to understand the main training abstraction.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see how `AgentFlowBase`, `AgentEnvLoop`, and `ToolEnv` fit together.
- Continue to the [`Agent Task Tutorial`](../tutorials/agent-task.md) for the minimal GSM8K + Tool example based on `ToolEnv + BaseTool`.
- Use [`Recipes and Algorithms`](../tutorials/recipes-and-algorithms.md) to find task-specific recipes and algorithm scripts.
