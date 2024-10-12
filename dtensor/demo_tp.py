import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
import torch.multiprocessing as mp
import os
from torch import distributed as dist
from typing import Optional
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import copy
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module,
                                               SequenceParallel,
                                               PrepareModuleInput)

from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import spawn_threads_and_init_comms
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel


@dataclass
class ModelArgs:
    dim: int = 96
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 400
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 48


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # (b, c, n_kv_heads//w, d//n_heads) -> (b, c, n_heads//w, d//n_heads)
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # (b, c, n_kv_heads//w, d//n_heads)
    #  -> (b, c, n_kv_heads//w, 1, d//n_heads)
    #  -> (b, c, n_kv_heads//w, n_heads//n_kv_heads, d//n_heads)
    #  -> (b, c, n_heads//w, d//n_heads)
    # 在第三个维度 repeat 也是一样
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # n_kv_heads 参数是用于 GQA 的，使得多个 query 共享同一个 key-value
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # tensor 并行的话，是在 head 维度进行切片，因为本来 head 计算就是独立的
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads

        # n 个 query 对应 1 个 key/value，为了方便计算，需要将 k,v 重复 n 次
        # 因此可以看出，gqa 并没有节省计算量，只是节省了参数量，也节省了 kv cache 的空间
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads  # 每个 head 的维度

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # (b,c, d//w) w=world_size

        # (b,c, n_heads//w, d//n_heads) 相当于 head 维度进行了切分
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # (b,c, n_kv_heads//w, d//n_heads) 相当于 kv_head 维度进行了切分
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        # 如果 key/value head 数小于 query head 数，那么需要重复 k/v n_rep 次
        # gqa 或者 mqa 时候出现
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        # 每个 head 对整个序列，所以分片没问题
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (b, c, d//w)
        return self.wo(output)  # 分片合并 (b,c,d)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # 为何如此复杂？ 确保 hidden_dim 是 multiple_of 的倍数，加速训练和推理
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        # norm 前置，稳定训练
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    # start_pos 是生成时候配合 key-value cache 使用的
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, start_pos=0):
        _bsz, seqlen = tokens.shape  # b,n
        h = self.tok_embeddings(tokens)  # b,n,96

        # 预计算
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[0:seqlen]

        mask = None
        if seqlen > 1:  # 下三角矩阵，下三角是 0，上三角是 -inf
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


def demo_1(transformer):
    # print(transformer)
    set_random_seed(42)
    input = torch.randint(10, 400, (3, 4))
    output = transformer(input, start_pos=0)

    label = copy.deepcopy(input)
    loss = F.cross_entropy(output.flatten(0, 1), label.flatten(0, 1))
    print('===', output.shape, output.mean(), loss)


@spawn_threads_and_init_comms
def demo_2(world_size: int, transformer):
    transformer = copy.deepcopy(transformer)

    tp_mesh = init_device_mesh('cpu', (world_size,))

    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
    }
    for layer in transformer.layers:
        attention = layer.attention
        attention.n_local_heads = attention.n_local_heads // tp_mesh.size()
        attention.n_local_kv_heads = attention.n_local_kv_heads // tp_mesh.size()
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    transformer = parallelize_module(
        module=transformer,
        device_mesh=tp_mesh,
        parallelize_plan={
            'tok_embeddings':
                RowwiseParallel(input_layouts=Replicate(), ),
            'output': ColwiseParallel(output_layouts=Replicate(), ),
        })

    set_random_seed(42)
    input = torch.randint(10, 400, (3, 4))
    output = transformer(input, start_pos=0)

    label = copy.deepcopy(input)
    loss = F.cross_entropy(output.flatten(0, 1), label.flatten(0, 1))
    print(f'{dist.get_rank()}, {output.shape}, {output.mean()}, {loss}', flush=True)


@spawn_threads_and_init_comms
def demo_3(world_size: int, transformer):
    transformer = copy.deepcopy(transformer)

    tp_mesh = init_device_mesh('cpu', (world_size,))

    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
    }
    for layer in transformer.layers:
        attention = layer.attention
        attention.n_local_heads = attention.n_local_heads // tp_mesh.size()
        attention.n_local_kv_heads = attention.n_local_kv_heads // tp_mesh.size()
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    transformer = parallelize_module(
        module=transformer,
        device_mesh=tp_mesh,
        parallelize_plan={
            'tok_embeddings':
                RowwiseParallel(input_layouts=Replicate(), ),
            'output': ColwiseParallel(use_local_output=False),
        })

    set_random_seed(42)
    input = torch.randint(10, 400, (3, 4))
    output = transformer(input, start_pos=0)

    label = copy.deepcopy(input)
    try:
        with loss_parallel():
            loss = F.cross_entropy(output.flatten(0, 1), label.flatten(0, 1))
            print(f'ttttt {dist.get_rank()}, {output.to_local().shape}, {output.mean()}, {loss}', flush=True)
    except Exception as e:
        pass


@spawn_threads_and_init_comms
def demo_4(world_size: int, transformer):
    transformer = copy.deepcopy(transformer)

    tp_mesh = init_device_mesh('cpu', (world_size,))

    # 带 sp 序列并行，减少激活值
    # 1 embeding 后输出是 (3,4,96),由于 Shard(1) 因此输出是 (3,1,96)
    # 2 attention_norm 进行序列并行输入是 (3,1,96)，输出变成 (3,4,96)
    # 3 attention 输入布局 Shard(1)，然后设置重新布局为 Replicate() 从而触发 forward 前
    layer_tp_plan = {
        # Now the input and output of SequenceParallel has Shard(1) layouts,
        # to represent the input/output tensors sharded on the sequence dimension
        "attention_norm": SequenceParallel(),
        # self.attention 有 4 个输入，这里布局也必须要 4 个，否则会报错
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None, None, None),
            desired_input_layouts=(Replicate(), None, None, None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),  # 序列维度切分
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }
    for layer in transformer.layers:
        attention = layer.attention
        attention.n_local_heads = attention.n_local_heads // tp_mesh.size()
        attention.n_local_kv_heads = attention.n_local_kv_heads // tp_mesh.size()
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    transformer = parallelize_module(
        module=transformer,
        device_mesh=tp_mesh,
        parallelize_plan=
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            )
        })

    set_random_seed(42)
    input = torch.randint(10, 400, (3, 4))
    output = transformer(input, start_pos=0)

    label = copy.deepcopy(input)
    loss = F.cross_entropy(output.flatten(0, 1), label.flatten(0, 1))
    print(f'ddddd {dist.get_rank()}, {output.shape}, {output.mean()}, {loss}', flush=True)


@spawn_threads_and_init_comms
def demo_5(world_size: int, transformer):
    transformer = copy.deepcopy(transformer)

    tp_mesh = init_device_mesh('cpu', (world_size,))

    # 带 sp 序列并行，减少激活值
    # 1 embeding 后输出是 (3,4,96),由于 Shard(1) 因此输出是 (3,1,96)
    # 2 attention_norm 进行序列并行输入是 (3,1,96)，输出变成 (3,4,96)
    # 3 attention 输入布局 Shard(1)，然后设置重新布局为 Replicate() 从而触发 forward 前
    layer_tp_plan = {
        # Now the input and output of SequenceParallel has Shard(1) layouts,
        # to represent the input/output tensors sharded on the sequence dimension
        "attention_norm": SequenceParallel(),
        # self.attention 有 4 个输入，这里布局也必须要 4 个，否则会报错
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None, None, None),
            desired_input_layouts=(Replicate(), None, None, None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),  # 序列维度切分
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }
    for layer in transformer.layers:
        attention = layer.attention
        attention.n_local_heads = attention.n_local_heads // tp_mesh.size()
        attention.n_local_kv_heads = attention.n_local_kv_heads // tp_mesh.size()
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    transformer = parallelize_module(
        module=transformer,
        device_mesh=tp_mesh,
        parallelize_plan=
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                # use DTensor as the output
                use_local_output=False,  # 核心，输出不在词表维度聚合，而是依然输出 DTensor
            )
        })

    set_random_seed(42)
    input = torch.randint(10, 400, (3, 4))
    output = transformer(input, start_pos=0)

    label = copy.deepcopy(input)
    try:  # 不知道为啥跑完会报错？
        with loss_parallel():
            loss = F.cross_entropy(output.flatten(0, 1), label.flatten(0, 1))
            print(f'cccccc {dist.get_rank()}, {output.to_local().shape}, {output.mean()}, {loss}', flush=True)
    except Exception as e:
        pass


if __name__ == '__main__':
    set_random_seed(42)
    transformer = Transformer(ModelArgs())
    demo_1(copy.deepcopy(transformer))  # 单卡
    demo_2(4, copy.deepcopy(transformer))  # tp=4
    demo_3(4, copy.deepcopy(transformer))  # tp=4 + loss parallel
    demo_4(4, copy.deepcopy(transformer))  # tp=4 + sp
    demo_5(4, copy.deepcopy(transformer))  # tp=4 + sp + loss parallel
