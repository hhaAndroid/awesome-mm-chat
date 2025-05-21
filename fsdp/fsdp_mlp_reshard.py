import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
from torch import distributed as torch_dist
from torch.distributed.device_mesh import init_device_mesh
from dataclasses import dataclass
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
from datetime import datetime
from torch.distributed._tensor import DTensor
from typing import cast


def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
) -> None:
    from torch.distributed.fsdp._fully_shard import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_post_forward_mesh_info
    self_module = cast(nn.Module, self)
    modules = list(self_module.modules()) if recurse else [self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.post_forward_mesh_info = (
                    _get_post_forward_mesh_info(
                        reshard_after_forward, fsdp_param_group.mesh_info
                    )
                )


def dispatch_torch_fsdpmodule():
    from torch.distributed.fsdp._fully_shard import FSDPModule

    print("dispatch_torch_fsdpmodule")
    FSDPModule.set_reshard_after_forward = set_reshard_after_forward


def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)


class MLPDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super(MLPDecoderLayer, self).__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.down_proj(self.up_proj(hidden_states))
        return hidden_states


class MLPForCausalLM(nn.Module):
    def __init__(self, config):
        super(MLPForCausalLM, self).__init__()

        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, 0)
        self.layers = nn.ModuleList(
            [
                MLPDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, labels):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


@dataclass
class MLPConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 6
    vocab_size: int = 4096


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.embed_tokens.weight.numel()
    return num_params


def loss_fun(logits, labels, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


# torchrun --nproc_per_node=4 -m debugpy --connect 5680 a.py
if __name__ == '__main__':
    dispatch_torch_fsdpmodule()

    work_dir = 'work_dir'

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group(backend='nccl')

    world_size = torch_dist.get_world_size()
    world_mesh = init_device_mesh(
        'cuda', (world_size,), mesh_dim_names=('world',))

    rank = torch_dist.get_rank()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    torch_dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    work_dir = os.path.join(work_dir, timestamp)
    os.makedirs(os.path.expanduser(work_dir), mode=0o777, exist_ok=True)

    config = MLPConfig()
    with torch.device("meta"):
        model = MLPForCausalLM(config)

    num_params = get_num_params(model)
    print0(f"Number of parameters: {num_params / 1024 ** 2:.2f}M")

    dtype = torch.bfloat16
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
    fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy}

    reshard_after_forward = False
    for layer_id, transformer_block in enumerate(model.layers):
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)

    print0(model)

    # 前面都是 layz 的，调用这行代码才真正触发 reshard
    model.to_empty(device='cuda')

    with torch.no_grad():
        model.init_weights()

    # forward
    input_shape = (2, 8192)
    input_ids = torch.randint(1, config.vocab_size, input_shape).cuda()
    labels = torch.randint(1, config.vocab_size, input_shape).long().cuda()

    print0(f"111 {model.layers[0].up_proj.weight.shape},{model.layers[0].up_proj.weight._local_tensor.shape}")
    logits = model(input_ids, labels)
    print0(f"222 {model.layers[0].up_proj.weight.shape},{isinstance(model.layers[0].up_proj.weight, DTensor)}")

    logits = model(input_ids, labels)
    print0(f"333 {model.layers[0].up_proj.weight.shape},{isinstance(model.layers[0].up_proj.weight, DTensor)}")

    model.set_reshard_after_forward(True)

    # forward+backward
    logits = model(input_ids, labels)
    print0(f"4444 {model.layers[0].up_proj.weight.shape},{model.layers[0].up_proj.weight._local_tensor.shape}")

    loss = loss_fun(logits, labels, config.vocab_size)
    # loss.backward()

    torch.distributed.destroy_process_group()
