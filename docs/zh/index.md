# Agent-R1

## 使用端到端强化学习训练强大的 LLM 智能体

Agent-R1 是一个面向智能体强化学习的开源框架。它支持构建自定义智能体流程、定义交互式环境与工具，并在统一的 RL 管线中训练多步智能体。

不同于把交互过程视为不断增长的单轮 prompt-response 序列，Agent-R1 将每一轮交互建模为 **step-level MDP transition**。模型观察环境、生成动作、接收工具或环境反馈，并持续交互直到任务完成或终止。

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Step-level MDP**

    ---

    用明确的 MDP 形式支持灵活上下文管理与逐步奖励信号。

    [:octicons-arrow-down-24: 了解更多](core-concepts/step-level-mdp.md)

-   :material-layers-outline:{ .lg .middle } **分层抽象**

    ---

    从完全自定义到开箱即用，根据任务复杂度选择合适的开发层次。

    [:octicons-arrow-down-24: 了解更多](core-concepts/layered-abstractions.md)

</div>

---

## 阅读路径

- 如果你想快速跑通环境，先阅读 [`快速开始`](getting-started/index.md)：复用 `verl` 环境，从 [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data) 下载处理好的数据集，并运行一个 GSM8K sanity check。
- 如果你想理解框架设计，阅读 [`Step-level MDP`](core-concepts/step-level-mdp.md) 和 [`分层抽象`](core-concepts/layered-abstractions.md)。
- 如果你想看最小工具调用示例，阅读 [`智能体任务教程`](tutorials/agent-task.md)：基于 `ToolEnv + BaseTool` 的 GSM8K + Tool。
- 如果你想了解当前 recipe 与算法脚本，阅读 [`Recipes 与算法`](tutorials/recipes-and-algorithms.md)。

## 文档范围

当前文档保持轻量，重点覆盖 Agent-R1 目前最核心的部分：智能体抽象、可运行示例、recipe 结构与算法启动脚本。

---

<div style="text-align: center; color: #888; margin-top: 2em;" markdown>
Supported by the [State Key Laboratory of Cognitive Intelligence](https://cogskl.iflytek.com/){ target=_blank }, University of Science and Technology of China (USTC).
</div>
