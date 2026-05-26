# Installation Guide

Agent-R1 uses the same environment setup as `verl`.

## Base Environment

Follow the official [`verl` installation guide](https://verl.readthedocs.io/en/latest/start/install.html), but use a recent source checkout of `verl` rather than the old release used by earlier Agent-R1 versions. Agent-R1 relies on the newer AgentFlow / async rollout / reward-loop APIs and on `verl.trainer.config` being available as package data.

If you want a broader overview of the base training workflow, the [`verl` quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) is also useful.

## What This Means for Agent-R1

Once the `verl` environment is working, Agent-R1 should run in the same environment. In practice, that means you can:

- prepare a Python environment with a compatible recent source installation of `verl`
- clone this repository
- run Agent-R1 commands directly from the repository root

You do not need to install Agent-R1 as a separate package.

The documentation in this repository intentionally does not duplicate a separate environment guide, so that the infrastructure setup stays aligned with `verl`.
