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


class MLPDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super(MLPDecoderLayer, self).__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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

    # 1 AX*W1 -> F.silu
    # 2 B X*W3
    # 3 C=A*B
    # 4 D D*W2
    # 这个布局是定死的，否则结果不对
    # 可确保整个模块只有一个 all-reduce forward + backward
    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        "w1": ColwiseParallel(),  # 列并行，第一个输入因此输入布局是复制[X,X]，权重在最后的维度切分[A1,A2]，输出为[XA1, XA2]
        "w2": RowwiseParallel(),  # 行并行, 输入是Shard(-1)，输出要确保tp size内的输出都是一样，因此设置为复制
        "w3": ColwiseParallel(),  # 列并行，第一个输入因此输入布局是复制[X,X]，权重在最后的维度切分[A1,A2]，输出为[XA1, XA2]
    }

    for layer_id, transformer_block in enumerate(model.layers):
        parallelize_module(
            module=transformer_block,
            device_mesh=world_mesh,
            parallelize_plan=layer_tp_plan,
        )

    # 对于这个 case，embeding_token 和 lm_head 你想用任何切分布局都可以，只要确保输入和输出都是 Replicate 即可
    # 为了推荐如此设置，是为了方便和 sp 结合以及 loss parallel 实现
    model = parallelize_module(
        model,
        world_mesh,
        {
            # 假设词表中有300个词，现在我们将word embedding拆分到两块GPU上，第一块GPU维护词表[0, 150)，第二块GPU维护词表[150, 299)。当输入X去GPU上查找时，能找到的词，就正常返回词向量，找到不到就把词向量中的全部全素都置0。按此方式查找完毕后，每块GPU上的数据做一次AllReduce，就能得到最终的输入。例如例子中，第一块GPU的查找结果为[ok, 0, ok, ok]，第二块为[0, ok, 0, 0]，两个向量一相加，变为[ok, ok, ok, ok]

            # 核心： 输入复制，权重在 vocab 维度切分，输出要进行 allreduce
            # 对 embed_tokens module 在 (vocab, hidden) 的 0 维度进行切分
            # 对 embed_tokens 输入进行复制(对于 tp 而言，输入的数据是一样的，因此input_layouts 应该设置为复制才能正确从 Tensor -> DTensor)，
            # 为了确保 tp size 内的输出都是一样，因此需要设置 output_layouts 为复制, 此时就会自动触发 allreduce
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            # 输入是一样的，因此是 Replicate()，权重第 0 维度切分，也就是在 hidden 维度切分，词表维度没有切分
            # 输出也要一样，因此在设置输出为复制布局时候，会自动触发 all-gather
            "lm_head": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        }
    )
    # 无法自动把 backward 的两个 all-reduce 变成 1 个
    # for layer_id, transformer_block in enumerate(model.layers):
    #     transformer_block = torch.compile(transformer_block, fullgraph=True)
    #     model.layers[layer_id] = transformer_block

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
