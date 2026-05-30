# Quick Start

这个 quick start 是一个 **sanity check**，不是 Agent-R1 的主要智能体工作流。它的目标是确认环境、数据路径、模型路径和训练栈已经正确连接。

## 1. 准备最小数据集

Agent-R1 处理好的数据集已经发布在 [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data)。下载后，将 GSM8K 文件放置或软链到 `~/data/gsm8k`；也可以使用 GSM8K 数据预处理脚本在本地重新生成 sanity-check 数据：

```bash
pip install modelscope
modelscope download --dataset Melmaphother/Agent-R1-data --local_dir data/agent-r1-data
```

也可以用 git 克隆数据集仓库：

```bash
git lfs install
git clone https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data.git data/agent-r1-data
```

```bash
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k --local_save_dir ~/data/gsm8k
```

它会生成：

- `~/data/gsm8k/train.parquet`
- `~/data/gsm8k/test.parquet`

## 2. 运行 sanity check 脚本

使用单步训练脚本：

```bash
bash examples/gsm8k/run_steppo.sh
```

如果需要，请在运行前调整：

- `CUDA_VISIBLE_DEVICES`
- `actor_rollout_ref.model.path`
- `~/data/gsm8k` 下的数据路径

脚本入口是 [`examples/gsm8k/run_steppo.sh`](https://github.com/AgentR1/Agent-R1/blob/main/examples/gsm8k/run_steppo.sh)，它会使用 StepPO 风格的 `gae` estimator 启动 `python3 -m agent_r1.trainer.main_agent_ppo`。

## 3. 下一步

- 阅读 [`Step-level MDP`](../core-concepts/step-level-mdp.md)，理解主要训练抽象。
- 阅读 [`分层抽象`](../core-concepts/layered-abstractions.md)，了解 `AgentFlowBase`、`AgentEnvLoop` 与 `ToolEnv` 如何组合。
- 继续阅读 [`智能体任务教程`](../tutorials/agent-task.md)，查看基于 `ToolEnv + BaseTool` 的最小 GSM8K + Tool 示例。
- 使用 [`Recipes 与算法`](../tutorials/recipes-and-algorithms.md) 查找任务 recipe 和算法脚本。
