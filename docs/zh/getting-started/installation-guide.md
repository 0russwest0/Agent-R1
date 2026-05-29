# 安装指南

Agent-R1 使用与 `verl` 相同的环境设置。

## 基础环境

请参考官方 [`verl` 安装指南](https://verl.readthedocs.io/en/latest/start/install.html)，并确保最终环境中使用 `verl==0.7.0`。

如果你想先了解基础训练流程，也可以参考 [`verl` quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)。

## 这对 Agent-R1 意味着什么

当 `verl` 环境可以正常工作后，Agent-R1 应该可以在同一个环境中运行。实践中，你需要：

- 准备一个包含 `verl==0.7.0` 的 Python 环境
- 克隆本仓库
- 直接在仓库根目录运行 Agent-R1 命令

你不需要把 Agent-R1 作为单独的 Python package 安装。

本仓库文档有意不重复维护一份完整环境指南，这样基础设施设置可以持续与 `verl` 保持一致。
