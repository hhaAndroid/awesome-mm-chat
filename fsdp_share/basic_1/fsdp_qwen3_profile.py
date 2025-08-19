import torch
from torch.nn import CrossEntropyLoss
import os
from torch import distributed as torch_dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
import argparse
from torch.optim import AdamW
from datetime import datetime
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from utils.profiling import maybe_enable_profiling, \
    maybe_enable_memory_snapshot, \
    profile_time_and_memory,\
    current_max_mem
from transformers import AutoModelForCausalLM


def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.embed_tokens.weight.numel()
    return num_params


def build_fsdp_1(model, use_checkpoint=False):
    dtype = torch.bfloat16
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
    fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy}

    for idx, layer in enumerate(model.layers[:-1]):
        if use_checkpoint:
            layer = checkpoint_wrapper(layer)
        model.layers[idx] = layer

    for layer_id, transformer_block in enumerate(model.layers):
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)


# PYTHONPATH="$(pwd)" torchrun --nproc_per_node=2 basic_1/fsdp_mlp_profile.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--profile_freq", default=0)
    parser.add_argument('--enable_profiling', type=bool, default=False)
    parser.add_argument('--enable_snapshot', type=bool, default=False)

    args = parser.parse_args()

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

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    num_params = get_num_params(model)
    print0(f"Number of parameters: {num_params / 1024 ** 2:.2f}M")

    with profile_time_and_memory('[FSDP]'):
        build_fsdp_1(model, use_checkpoint=False)
    print0(model)

    with profile_time_and_memory('[FSDP Model Init]'):
        model.to_empty(device='cuda')
        with torch.no_grad():
            model.init_weights()
    model.train()

    current_max_mem('[step1]')

    requried_grad_params = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = AdamW(
        requried_grad_params,
        lr=1e-5,
        weight_decay=0,
        betas=(0.9, 0.95))

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

    enable_profiling = args.enable_profiling
    enable_snapshot = args.enable_snapshot
    profile_freq = args.profile_freq

    input_shape = (2, 8192)

    with maybe_enable_profiling(
            enable_profiling, work_dir, profile_freq, global_step=0
    ) as torch_profiler, maybe_enable_memory_snapshot(
        enable_snapshot, work_dir, global_step=0
    ) as memory_profiler:
        for i in range(20):
            print0(f'=====================step {i}==========================')
            with profile_time_and_memory('[model fwd]'):
                # 由于只开了 fsdp，所以每张卡上数据不一样即可
                input_ids = torch.randint(1, config.vocab_size, input_shape).cuda()
                labels = torch.randint(1, config.vocab_size, input_shape).long().cuda()

                logits = model(input_ids, labels)
                loss = loss_fun(logits, labels, config.vocab_size)

                del logits  # 核心，可以节省峰值显存

            with profile_time_and_memory('[model bwd]'):
                loss.backward()

            with profile_time_and_memory('[optim step]'):
                optimizer.step()
                optimizer.zero_grad()

            current_max_mem()

            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

    torch.distributed.destroy_process_group()
