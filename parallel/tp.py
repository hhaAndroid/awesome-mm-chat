import torch
from torch.utils.data import DataLoader, Dataset
from torch import distributed as dist
from mmengine.dist import all_gather, get_world_size
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


class Attention(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        model_parallel_size = get_world_size()
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
            multiple_of=2
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


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    os.environ['RANK'] = str(rank)

    if backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        device = 'cuda'
    else:
        device = 'cpu'

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size)

    for func in functions:
        func(device)


def demo1(device):
    set_random_seed(42)
    model = TransformerBlock()
    dataset = MyDataset(torch.randn(1000, 48, 128))
    # 没有 dp，所以不需要 sampler
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for data in dataloader:
        outputs = model(data)
        outputs.mean().backward()

        grads = all_gather(model.feed_forward.w1.weight.grad)
        all_grads = torch.cat(grads, dim=0)
        print(f'{outputs.mean()},{all_grads.mean()}')
        break


if __name__ == '__main__':
    # 单卡测试
    set_random_seed(42)
    model = TransformerBlock()
    dataset = MyDataset(torch.randn(1000, 48, 128))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for data in dataloader:
        outputs = model(data)
        outputs.mean().backward()

        # 在 tp 中，数据要完全一样， output 是自动 all-gather 了，所以不用做任何处理
        # 在反向传播时候，内部参数也会自动进行处理，打印时候 all-gather 下才能保证和单卡一致

        print(f'{outputs.mean()},{model.feed_forward.w1.weight.grad.mean()}')
        break

    # tp=2 无 dp
    functions = [demo1]
    world_size = 2
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
