# 核心概念

这一部分介绍 Agent-R1 作为智能体任务训练框架时最重要的设计思想。

## 本节内容

- [`Step-level MDP`](step-level-mdp.md)：为什么 Agent-R1 把智能体训练建模为多步交互，而不是单个不断增长的 token 序列。
- [`分层抽象`](layered-abstractions.md)：`AgentFlowBase`、`AgentEnvLoop`、`AgentEnv`、`ToolEnv` 和 `BaseTool` 如何协同工作。

## 为什么这些概念重要

Agent-R1 面向的是智能体任务：LLM 与环境交互、接收新观察，并通过轨迹级强化学习改进策略。这两页解释支撑这一工作流的核心建模方式与编程模型。
