# Recipes 与算法

Agent-R1 的 recipe 是任务级集成。一个 recipe 通常拥有 prompt、rollout 时的 agent flow、reward 逻辑、环境 wrapper、数据预处理入口，以及运行该 benchmark 所需的任务特定依赖说明。

启动脚本统一放在 `examples/` 下。推荐结构是：

```text
examples/<task>/run_ppo.sh
examples/<task>/run_steppo.sh
examples/<task>/run_grpo.sh
examples/<task>/run_rloo.sh
examples/<task>/run_reinforce.sh
examples/<task>/run_gspo.sh
examples/<task>/run_gigpo.sh
```

不是每个任务都需要所有脚本。PPO 和 StepPO 使用自己的启动配置，其他若干算法变体会复用 GRPO 风格的脚本结构，并通过不同 Hydra overrides 区分。

GSM8K 会被刻意保持为轻量测试 recipe，只提供 `examples/gsm8k/run_steppo.sh` 作为单轮 sanity check，以及 `examples/gsm8k/run_steppo_tool.sh` 作为多轮 ToolEnv check。

## Recipe 结构

大多数 recipe 遵循以下形状：

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

当任务有独立 rule reward 或 model reward 时，通常会提供 `reward_fn.py`。部分在线环境会在交互过程中直接计算奖励，因此不需要单独的 reward function 文件。

## 数据集与环境

| Recipe | 覆盖内容 | 主要入口 |
| --- | --- | --- |
| `gsm8k` | 小学数学推理。Plain GSM8K 保留为单轮 sanity check，GSM8K + Tool 则作为最小 `ToolEnv + BaseTool` 示例，使用 recipe-local `calc_gsm8k_reward`。 | `data_preprocess/process_gsm8k.py`, `data_preprocess/process_gsm8k_tool.py`, `tool.py` |
| `hotpotqa` | 带检索环境的多跳问答。数据预处理与检索索引构建分离。 | `data_preprocess/process_hotpotqa.py`, `env/build_retrieval_corpus.py`, `env/build_index.py`, `hotpotqa_agent_flow.py` |
| `alfworld` | 基于文本的 household task，通过 ALFWorld-style environment wrapper 和 tool executor 完成。 | `data_preprocess/process_alfworld.py`, `env/alfworld_wrapper.py`, `env/tool_executor.py`, `alfworld_agent_flow.py` |
| `webshop` | 购物智能体训练，包含本地 WebShop 环境服务与商品目录 artifacts。 | `data_preprocess/process_webshop.py`, `env/run_env_server.sh`, `env/full_catalog.py`, `webshop_agent_flow.py` |
| `paper_search` | 学术论文搜索智能体，会查询 paper service、扩展 citation/reference neighborhood，并用 selector service 打分。 | `paper_search_agent_flow.py`, `runtime.py`, `env/paper_client.py`, `inference/run.py`, `inference/evaluation.py` |

处理好的数据集已经发布在 [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data)。下载后，将各任务文件放置或软链到对应 recipe README 中列出的路径。recipe 中的数据预处理脚本保留用于本地重新生成或检查数据格式。

## 算法

| 算法 | 脚本 | 说明 |
| --- | --- | --- |
| PPO | `run_ppo.sh` | Actor-critic baseline。 |
| GRPO | `run_grpo.sh` | Group-relative policy optimization baseline，也是若干变体的通用脚本基座。 |
| [StepPO](https://arxiv.org/abs/2604.18401) | `run_steppo.sh` | 面向多步智能体轨迹的 step-aligned policy optimization。 |
| RLOO | `run_rloo.sh` | Leave-one-out baseline variant。 |
| REINFORCE | `run_reinforce.sh` | Step-level critic-free policy-gradient variant。 |
| GSPO | `run_gspo.sh` | 通过脚本 overrides 暴露的 GRPO-family 算法变体。 |
| GiGPO | `run_gigpo.sh` | 通过脚本 overrides 暴露的 GRPO-family 算法变体。 |

## 从哪里开始

- 使用 `recipes/<task>/README.md` 查看任务特定依赖、资源要求和环境设置。
- 运行完整 recipe 脚本前，先从 [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data) 下载处理好的数据集。
- 使用 `examples/<task>/run_*.sh` 查看启动命令和 Hydra overrides。
- 使用 `recipes/<task>/base.yaml` 查看 recipe-local rollout 配置。
- 只有在需要重新生成或检查数据格式时，才需要使用 `data_preprocess/process_*.py`。
