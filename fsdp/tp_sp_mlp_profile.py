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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
import torch.nn.functional as F
from torch.distributed._tensor import Replicate, Shard

from profiling import maybe_enable_profiling, maybe_enable_memory_snapshot


def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)


# sp 是用于减少这些 norm 层的激活值
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLPDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super(MLPDecoderLayer, self).__init__()
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.hidden_size)

    def forward(self, x):
        # 输出序列已经减半, norm 只要确保 hidden 维度完整即可，序列可以切断，因此可以直接算
        x = self.ffn_norm(x)  # 此处新增一个 norm 层，输出序列维度也是减半
        return self.feed_forward(x)


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
        self.norm = RMSNorm(config.hidden_size)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, labels):
        hidden_states = self.embed_tokens(input_ids)  # 输出序列维度减半
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # 新增一个 norm
        hidden_states = self.norm(hidden_states)
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

    # 只开 tp
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    from torch.distributed.tensor.parallel import (
        PrepareModuleInput,
        SequenceParallel,
    )

    layer_tp_plan = {
        'ffn_norm': SequenceParallel(),
        # ffn_norm 输出丢给他，sp 输出是在序列维度切分 dtensor ，因此设置 input_layouts=(Shard(1),)
        # 对于 tp 而言，输入给 feed_forward 要一致，因此设置 desired_input_layouts=(Replicate(),)
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),  # shard1 -> replicate 会触发一次序列维度 all-gather
        ),
        "feed_forward.w1": ColwiseParallel(),
        # 因为 feed_forward 输出后就喂给 norm，因此只要设置在序列维度已经切分即可
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),  # 也只是执行 reduce-scatter 了
        "feed_forward.w3": ColwiseParallel(),
    }

    for layer_id, transformer_block in enumerate(model.layers):
        parallelize_module(
            module=transformer_block,
            device_mesh=world_mesh,
            parallelize_plan=layer_tp_plan,
        )

    model = parallelize_module(
        model,
        world_mesh,
        {
            # 对权重按照第 0 维度切分，即词表维度切分
            # 计算完成后，如果想得到完整输出，需要执行 all-reduce
            # 但是因为后面结合 sp，因此我们不需要完整输出，而是只要当前 rank 的部分序列输出即可，因此设置为 output_layouts=Shard(1)
            # 此时执行的就是 reduce-scatter, 通信量减少 (world_size-1)/world_size
            # 注意，embedding_tokens 输出的序列维度减半了，激活值也减半了
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),  # 接下来是 norm，因此设置为在序列维度切分即可，省掉了一个 all-reduce
            ),
            "norm": SequenceParallel(),
            # norm 输出是序列维度切分的输入，对于 lm_head 而已需要完整序列，因此计算前会触发一次 all-gather
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()  # 执行一次序列维度的 all-gather 在计算之前
            ),
        }
    )

    # 注意：本改进必须要用于 tp+sp 场合，如果是纯 tp 那么是无效的
    enable_async_tp = False
    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        # 我们只开了 tp
        enable_symm_mem_for_group(world_mesh.get_group().group_name)

        # 必须要启动 compile 才能开启 async_tp
        for layer_id, transformer_block in enumerate(model.layers):
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            model.layers[layer_id] = transformer_block

    print0(model)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.to_empty(device='cuda')
    with torch.no_grad():
        model.init_weights()
    model.train()

    max_memory = torch.cuda.max_memory_allocated()
    print0('[Train] 1111Begin Train Loop. The current GPU memory is '
           f'{(max_memory / 1024 ** 2):.1f}mB')

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

    input_shape = (2, 8192)

    with maybe_enable_profiling(
            enable_profiling, work_dir, profile_freq, global_step=0
    ) as torch_profiler, maybe_enable_memory_snapshot(
        enable_snapshot, work_dir, global_step=0
    ) as memory_profiler:
        for i in range(40):
            torch.cuda.reset_peak_memory_stats()

            # 在 tp 情况下数据完全相同
            if rank == 0:
                input_ids = torch.randint(1, config.vocab_size, input_shape).cuda()
                labels = torch.randint(1, config.vocab_size, input_shape).long().cuda()
            else:
                input_ids = torch.zeros(input_shape).long().cuda()
                labels = torch.zeros(input_shape).long().cuda()

            torch_dist.broadcast(input_ids, 0)
            torch_dist.broadcast(labels, 0)

            logits = model(input_ids, labels)
            loss = loss_fun(logits, labels, config.vocab_size)

            del logits  # 核心，可以节省峰值显存

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
