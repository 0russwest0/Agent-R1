# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Agent-R1 local PPO critic overrides.
"""

import logging
import os

import torch

from agent_r1.trainer.ppo.core_algos import compute_value_loss
from agent_r1.trainer.ppo.trajectory_batching import get_mini_batch_global_info, split_data_proto_by_mini_batch_id
from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.workers.critic.dp_critic import DataParallelPPOCritic as VerlDataParallelPPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(VerlDataParallelPPOCritic):
    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        self.critic_module.train()
        metrics = {
            "critic/vf_loss": 0.0,
        }

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_planned_mini_batches = "mini_batch_id" in data.batch.keys()
        if has_planned_mini_batches:
            select_keys.extend(
                [
                    "mini_batch_id",
                    "mini_batch_global_size",
                    "mini_batch_global_token_num",
                    "mini_batch_global_response_token_num",
                    "sample_mask",
                ]
            )

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if has_planned_mini_batches:
            mini_batches = split_data_proto_by_mini_batch_id(data)
        else:
            mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                use_global_mini_batch_info = "mini_batch_global_size" in mini_batch.batch.keys()
                global_batch_info = {}
                if use_global_mini_batch_info:
                    global_info = get_mini_batch_global_info(mini_batch)
                    dp_size = (
                        torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
                        if torch.distributed.is_initialized()
                        else 1
                    )
                    global_batch_info = {
                        "dp_size": dp_size,
                        "batch_num_tokens": global_info["batch_num_tokens"],
                        "global_batch_size": global_info["global_batch_size"],
                    }

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if not has_planned_mini_batches:
                    micro_batches = [mb for mb in micro_batches if bool(mb.batch["response_mask"].any().item())]
                if not micro_batches:
                    append_to_dict(metrics, {"critic/grad_norm": 0.0})
                    continue

                if not self.config.use_dynamic_bsz:
                    self.gradient_accumulation = len(micro_batches)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    vpreds = self._forward_micro_batch(model_inputs)
                    vf_loss, vf_clipfrac = compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                        **global_batch_info,
                    )
                    if use_global_mini_batch_info:
                        loss_scale_factor = 1.0
                    elif self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    loss = vf_loss * loss_scale_factor
                    loss.backward()

                    append_to_dict(
                        metrics,
                        {
                            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                        },
                    )
                    metrics["critic/vf_loss"] += vf_loss.detach().item() * loss_scale_factor

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        self.critic_optimizer.zero_grad()
        return metrics
