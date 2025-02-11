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
from torch.optim import AdamW
from datetime import datetime

from profiling import maybe_enable_profiling, maybe_enable_memory_snapshot


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
    hidden_size: int = 1024
    intermediate_size: int = 5120
    num_hidden_layers: int = 6
    vocab_size: int = 128512


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.embed_tokens.weight.numel()
    return num_params


# torchrun --nproc_per_node=4 fsdp_mlp_profile.py
if __name__ == '__main__':
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
    for layer_id, transformer_block in enumerate(model.layers):
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)

    print0(model)

    model.to_empty(device='cuda')
    with torch.no_grad():
        model.init_weights()
    model.train()

    requried_grad_params = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        requried_grad_params,
        lr=1e-5,
        weight_decay=0,
        betas=(0.9, 0.95))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    print0('[Train] Begin Train Loop. The current GPU memory is '
           f'{(max_memory / 1024 ** 3):.1f}GB')


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


    # 模拟数据集
    enable_profiling = True
    enable_snapshot = True
    profile_freq = 10

    input_shape = (32, 128)

    with maybe_enable_profiling(
            enable_profiling, work_dir, profile_freq, global_step=0
    ) as torch_profiler, maybe_enable_memory_snapshot(
        enable_snapshot, work_dir, global_step=0
    ) as memory_profiler:
        for i in range(40):
            torch.cuda.reset_peak_memory_stats()

            # 由于只开了 fsdp，所以每张卡上数据不一样即可
            input_ids = torch.randint(1, config.vocab_size, input_shape).cuda()
            labels = torch.randint(1, config.vocab_size, input_shape).long().cuda()

            logits = model(input_ids, labels)
            loss = loss_fun(logits, labels, config.vocab_size)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            max_memory = torch.cuda.max_memory_allocated()
            print0(f'[Train] Step {i}. loss: {loss.item()}. The current GPU memory is '
                   f'{(max_memory / 1024 ** 3):.1f}GB')

            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

    torch.distributed.destroy_process_group()
