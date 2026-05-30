# Agent Task Tutorial

This tutorial shows the simplest multi-step tool-calling path in Agent-R1: **GSM8K + Tool** built with the generic `AgentEnvLoop`, a recipe-local `ToolEnv`, and a recipe-local `BaseTool`.

The example uses GSM8K, but the important part is not the benchmark itself. The goal is to show how Agent-R1 turns a dataset row into an environment-driven, multi-step rollout.

## What You Will Run

This tutorial uses two existing files:

- dataset preprocessing: [`recipes/gsm8k/data_preprocess/process_gsm8k_agent.py`](https://github.com/AgentR1/Agent-R1/blob/main/recipes/gsm8k/data_preprocess/process_gsm8k_agent.py)
- training script: [`examples/gsm8k/run_grpo.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/gsm8k/run_grpo.sh)

## 1. Prepare the Agent Dataset

Generate the tool-augmented GSM8K dataset:

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k_agent --local_save_dir ~/data/gsm8k_agent
```

Compared with the single-step sanity-check dataset, this preprocessing script keeps structured task fields for the agent path:

- `agent_name: "gsm8k_agent"`
- `question` and `ground_truth`, which let the recipe environment build prompts dynamically
- `env_kwargs` for per-sample tool metadata such as the ground-truth answer

Conceptually, each sample says:

1. use the configured `gsm8k_agent` entry, which points to the generic `AgentEnvLoop`
2. instantiate the GSM8K tool environment
3. expose the `calc_gsm8k_reward` tool inside that environment

## 2. Launch the Agent Task Training Script

Run:

```bash
bash examples/gsm8k/run_grpo.sh
```

This script switches the rollout from single-step generation to the generic agent-environment loop:

```bash
actor_rollout_ref.rollout.agent.default_agent_flow=gsm8k_agent \
actor_rollout_ref.rollout.agent.max_steps=5 \
```

It also points the trainer to the tool dataset:

```bash
data.train_files=$HOME/data/gsm8k_agent/train.parquet \
data.val_files=$HOME/data/gsm8k_agent/test.parquet \
```

## 3. What Happens During One Trajectory

At a high level, one sample follows this path:

```mermaid
graph TD
    datasetRow["Dataset row"] --> agentFlow["AgentEnvLoop"]
    agentFlow --> toolEnv["GSM8KToolEnv"]
    toolEnv --> llmStep["LLM response"]
    llmStep --> toolCall["Tool call parsing"]
    toolCall --> toolExec["BaseTool execution"]
    toolExec --> nextObs["Next observation"]
    nextObs --> agentFlow
```

More concretely:

1. `AgentEnvLoop` reads recipe defaults and per-sample `env_kwargs`.
2. `AgentEnv.from_config(env_type="gsm8k_agent", ...)` creates a `GSM8KToolEnv`.
3. `GSM8KToolEnv.reset()` builds prompt messages from `question` and `ground_truth`.
4. The LLM produces a response.
5. `ToolEnv.step()` parses tool calls from the response and executes the registered tool.
6. Tool output is appended to the conversation as the next observation.
7. The loop continues until the environment returns `done=True` or `max_steps` is reached.

## 4. Where the Reward Comes From

The GSM8K tool is registered as `calc_gsm8k_reward` in `agent_r1/tool/tools/gsm8k.py`.

Its role in this example is to:

- receive the model's proposed answer
- compare it with the sample's ground truth
- return tool text back into the conversation

This is what makes the tutorial useful for Agent-R1: the model is not just generating one final answer, it is interacting with an environment that can evaluate and feed back information across multiple steps.

## 5. Why This Tutorial Is Separate From the Single-Step Script

The single-step GSM8K script is still useful as a setup check. This tutorial is different: it is the minimal example for the lowest abstraction layer, where users only define tools and let `ToolEnv + BaseTool` handle standard multi-turn tool interaction. It demonstrates:

- a step-level environment transition
- a multi-step agent loop
- tool-augmented interaction
- reward signals attached to environment-mediated behavior

## 6. Where to Look Next

- Read [`Step-level MDP`](../core-concepts/step-level-mdp.md) to connect this tutorial to the core RL formulation.
- Read [`Layered Abstractions`](../core-concepts/layered-abstractions.md) to see why this example maps naturally to `ToolEnv + BaseTool`.
- Read [`Recipes and Algorithms`](recipes-and-algorithms.md) to see the other task recipes and launch scripts.
