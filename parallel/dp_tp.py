import torch
from torch.utils.data import DataLoader, Dataset
from torch import distributed as dist
from mmengine.dist import all_gather, get_world_size, get_rank, all_reduce
import torch.multiprocessing as mp
import torch.nn as nn
from mmengine.runner import set_random_seed
import torch
import os
from column_parallel_linear_1 import ColumnParallelLinear
from row_parallel_linear_2 import RowParallelLinear
import torch
import torch.nn.functional as F
from torch import nn
from utils import _TP_GROUP, _DP_GROUP, initialize_tp_dp_group, init_process


class Attention(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        model_parallel_size = get_world_size(_TP_GROUP)
        # tensor 并行的话，是在 head 维度进行切片，因为本来 head 计算就是独立的
        self.n_local_heads = n_heads // model_parallel_size  # 可以认为是 query head 的个数

        # 输入分发，输出不聚合
        self.wq = ColumnParallelLinear(
            dim,
            dim,
            bias=False,
            gather_output=False
        )
        self.wk = ColumnParallelLinear(
            dim,
            dim,
            bias=False,
            gather_output=False
        )
        self.wv = ColumnParallelLinear(
            dim,
            dim,
            bias=False,
            gather_output=False
        )  # (b,c,d) -> (b,c, d//k) 维度输出

        # all-reduce
        self.wo = RowParallelLinear(
            dim,
            dim,
            bias=False,
            input_is_parallel=True
        )  # 聚合输出，维度不变

    def forward(
            self,
            x: torch.Tensor
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # (b,c, d//w) w=world_size

        # (b,c, n_heads//w, d//n_heads) 相当于 head 维度进行了切分
        xq = xq.view(bsz, seqlen, self.n_local_heads, -1)
        # (b,c, n_kv_heads//w, d//n_heads) 相当于 kv_head 维度进行了切分
        xk = xk.view(bsz, seqlen, self.n_local_heads, -1)
        xv = xv.view(bsz, seqlen, self.n_local_heads, -1)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, seqlen, head_dim)

        # 每个 head 对整个序列，所以分片没问题
        scores = torch.matmul(xq, keys.transpose(2, 3))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (b, c, d//w)
        return self.wo(output)  # 分片合并 (b,c,d)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # 为何如此复杂？ 确保 hidden_dim 是 multiple_of 的倍数，加速训练和推理
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 必须是这种组合，才能得到最少的数据同步通信量
        self.w1 = ColumnParallelLinear(  # 输入分发，输出不聚合
            dim, hidden_dim, bias=False, gather_output=False
        )
        self.w3 = ColumnParallelLinear(  # 输入分发，输出不聚合
            dim, hidden_dim, bias=False, gather_output=False
        )

        self.w2 = RowParallelLinear(  # 输入已经是分发后的，只需要执行一次 all-reduce
            hidden_dim, dim, bias=False, input_is_parallel=True
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 16
        self.dim = 128
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(self.n_heads, self.dim)
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=4 * self.dim,
            multiple_of=8
        )

    def forward(
            self,
            x: torch.Tensor):
        h = x + self.attention(x)
        out = h + self.feed_forward(h)
        return out


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def demo_dp(device, *args, **kwargs):
    set_random_seed(42)
    model = TransformerBlock()
    dataset = MyDataset(torch.randn(40, 48, 128))
    # 8 卡，每张卡 bs=4
    # 分布式采样
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    for i, (data) in enumerate(dataloader):
        output = model(data)
        output.mean().backward()

        # 每张卡只有部分梯度
        all_reduce(model.feed_forward.w1.weight.grad, op='mean')
        if dist.get_rank() == 0:
            print(f" 111 reduce grad mean: {model.feed_forward.w1.weight.grad.mean()}")
        break


def demo_tp_dp(device, backend):
    tp = 2
    dp = 4
    initialize_tp_dp_group(get_world_size(), tp, dp, backend)
    # print(f'{get_rank()}： dp_group={_DP_GROUP}, tp_group={_TP_GROUP}')
    # print(f'{get_rank()}： dp_rank={get_rank(_DP_GROUP)}, '
    #       f'tp_rank={get_rank(group=_TP_GROUP)}')

    set_random_seed(42)
    model = TransformerBlock()
    dataset = MyDataset(torch.randn(40, 48, 128))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False,
                                                              num_replicas=get_world_size(_DP_GROUP),
                                                              rank=get_rank(_DP_GROUP))

    # 为了保证数值一样，现在相当于只有 4 张卡，每张卡 bs 需要改成 8，但是也无法保证梯度完全一样
    # 因此只能用梯度累加的做法实现等价,但是似乎也不是一样的
    # TODO: 发现应该是数据顺序问题，比如之前 rank0 是处理 0/1 index，现在梯度累加后可能是 0/2，如果把所有数据都改成 torch.ones 可以发现此时梯度就相同了
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    for i, data in enumerate(dataloader):
        output = model(data)
        loss = output.mean()
        loss.backward()

        if i == 1:  # 梯度累加 2
            # _TP_GROUP 卡面对的数据一样，但是梯度会切分
            grads = all_gather(model.feed_forward.w1.weight.grad, group=_TP_GROUP)
            all_grads = torch.cat(grads, dim=0)

            # _DP_GROUP 卡上面数据不一样，因此梯度要进行规约
            # 最终梯度要除以 2
            all_reduce(all_grads, op='mean', group=_DP_GROUP)
            if dist.get_rank() == 0:
                print(f"222 reduce grad mean: {all_grads.mean()/2}")
            break


if __name__ == '__main__':
    # 假设一共 8 张卡
    """ global rank 是 0~7
     dp=4 的话，表示数据切成 4 份，即一次 dataloader 会输出 4 份不同数据，相当于 data 切分为 4 份，一共 2 份 copy
     tp=2 的话，表示模型切成 2 份，即面对每一份数据都在给 2 个切片模型进行处理，相当于 model 切分为 2 份，一共 4 份 copy
     
     0： dp_group=[0, 2, 4, 6], tp_group=[0, 1], dp_rank=0, tp_rank=0
     1： dp_group=[1, 3, 5, 7], tp_group=[0, 1], dp_rank=0, tp_rank=1
     2： dp_group=[0, 2, 4, 6], tp_group=[2, 3], dp_rank=1, tp_rank=0
     3： dp_group=[1, 3, 5, 7], tp_group=[2, 3], dp_rank=1, tp_rank=1
     4： dp_group=[0, 2, 4, 6], tp_group=[4, 5], dp_rank=2, tp_rank=0
     5： dp_group=[1, 3, 5, 7], tp_group=[4, 5], dp_rank=2, tp_rank=1
     6： dp_group=[0, 2, 4, 6], tp_group=[6, 7], dp_rank=3, tp_rank=0
     7： dp_group=[1, 3, 5, 7], tp_group=[6, 7], dp_rank=3, tp_rank=1
     
     假设当前 global rank=0，那么他的 dp_rank=0，因为在 dp_group([0, 2, 4, 6]) 中 dp_group.index(0)=0
     假设当前 global rank=1，那么他的 dp_rank=0，因为在 dp_group([1, 3, 5, 7]) 中 dp_group.index(1)=0
     假设当前 global rank=2，那么他的 dp_rank=1，因为在 dp_group([0, 2, 4, 6]) 中 dp_group.index(2)=1
     
     上面意思是 rank=0/1 上面所采样得到的数据一样，然后在 0/1 rank 上模型进行了切片，实现了正交功能
    """
    # 假设只开启 8 dp
    functions = [demo_dp, demo_tp_dp]
    world_size = 8
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)

