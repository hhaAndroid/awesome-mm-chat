# pip install torchgpipe
# 由于 gpipe 涉及到大量的 cuda stream 协作，比较复杂，因此这里考虑直接调用官方库来实现 demo 功能

from torchgpipe import GPipe
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.runner import set_random_seed
from utils import init_process, initialize_pp_dp_group, _DP_GROUP, _PP_GROUP
from torch.utils.data import DataLoader, Dataset
from torch import distributed as dist
from mmengine.dist import all_reduce, get_world_size, get_rank
import torch.multiprocessing as mp
import torch.nn as nn
from mmengine.runner import set_random_seed
import torch


class Attention(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        self.n_local_heads = n_heads

        self.wq = nn.Linear(
            dim,
            dim,
            bias=False
        )
        self.wk = nn.Linear(
            dim,
            dim,
            bias=False
        )
        self.wv = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # all-reduce
        self.wo = nn.Linear(
            dim,
            dim,
            bias=False,
        )

    def forward(
            self,
            x: torch.Tensor,
            use_sp=False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # (b,c, d//w) w=world_size

        # (b,c, n_heads, d//n_heads) 相当于 head 维度进行了切分
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

        scores = torch.matmul(xq, keys.transpose(2, 3))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        # =================================================================================
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
            x: torch.Tensor,
            use_sp=False):
        h = x + self.attention(x, use_sp)
        out = h + self.feed_forward(h)
        return out


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def demo_pp():
    # 8 卡 pp
    # 假设是 8 卡, 每1层都放到一个 device 上
    num_device = 8
    device = 'cpu'
    devices = [device] * num_device
    if device == 'cuda':
        devices = None

    num_layers = 8
    data_chunks = 8  # 可以不是 8, 可以是任意值，但是如果是 8 张卡，最好 data_chunks 要大于等于 8，否则空闲时间比较多
    set_random_seed(42)

    orig_model = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])
    model = GPipe(orig_model, balance=[1] * num_device, devices=devices, chunks=data_chunks)

    print(model.partitions)
    print(model.devices)
    set_random_seed(42)
    dataset = MyDataset(torch.randn(1000, 64, 128))
    # 假设总 bs=16, 每张卡 bs=2，然后会自动汇总
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    in_device = model.devices[0]
    out_device = model.devices[-1]
    for i, (data) in enumerate(dataloader):
        data = data.to(in_device, non_blocking=True)
        output = model(data).to(out_device, non_blocking=True)
        loss = output.mean()
        loss.backward()
        print(f"loss {loss}, grad mean: {orig_model[0].feed_forward.w1.weight.grad.mean()}")
        break


def demo_dp_run(*args, **kwargs):
    num_layers = 8
    set_random_seed(42)
    orig_model = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])

    set_random_seed(42)
    dataset = MyDataset(torch.randn(1000, 64, 128))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    # 之前是 8卡 pp=8 bs=16，现在是 8卡 dp=8 pp=1，那么 bs 就要相应减少才对上
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    for i, data in enumerate(dataloader):
        outputs = orig_model(data)
        loss = outputs.mean()
        loss.backward()
        all_reduce(loss, op='mean')
        all_reduce(orig_model[0].feed_forward.w1.weight.grad, op='mean')
        if dist.get_rank() == 0:
            print(f"reduce loss: {loss}, "
                  f"reduce grad mean: {orig_model[0].feed_forward.w1.weight.grad.mean()}")
        break


def demo_dp():
    functions = [demo_dp_run]
    world_size = 8
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)

from fairscale.nn.model_parallel import initialize_model_parallel
def demo_dp_pp_run(*args, backend='gloo', **kwargs):
    # dp=4 pp=2 意思是将数据分成 4 份，并且将模型切分为 2 段，每段 4 层，也就是
    # 0： pp_group = [0, 1], dp_group = [0, 2, 4, 6] data0+sub_model0
    # 1： pp_group = [0, 1], dp_group = [1, 3, 5, 7] data0+sub_model1
    # 2： pp_group = [2, 3], dp_group = [0, 2, 4, 6] data1+sub_model0
    # 3： pp_group = [2, 3], dp_group = [1, 3, 5, 7] data1+sub_model1
    # 4： pp_group = [4, 5], dp_group = [0, 2, 4, 6]
    # 5： pp_group = [4, 5], dp_group = [1, 3, 5, 7]
    # 6： pp_group = [6, 7], dp_group = [0, 2, 4, 6]
    # 7： pp_group = [6, 7], dp_group = [1, 3, 5, 7]
    # 0 和 1 构成一个完整模型

    # dp=2 pp=4 意思是将数据分成 2 份，并且将模型切分为 4 段，每段 2 层
    # 7： pp_group = [4, 5, 6, 7], dp_group = [3, 7]
    # 6： pp_group = [4, 5, 6, 7], dp_group = [2, 6]
    # 5： pp_group = [4, 5, 6, 7], dp_group = [1, 5]
    # 4： pp_group = [4, 5, 6, 7], dp_group = [0, 4]
    # 3： pp_group = [0, 1, 2, 3], dp_group = [3, 7] data0+sub_model4
    # 2： pp_group = [0, 1, 2, 3], dp_group = [2, 6] data0+sub_model3
    # 1： pp_group = [0, 1, 2, 3], dp_group = [1, 5] data0+sub_model1
    # 0： pp_group = [0, 1, 2, 3], dp_group = [0, 4] data0+sub_model0
    # 0~3 构成一个完整模型

    # GPipe 和 DDP 实际上是不兼容的，有点麻烦
    dp = 4
    pp = 2
    initialize_pp_dp_group(get_world_size(), pp, dp, backend)

    # TODO
    # 目前库 DDP+PP 不支持，如果想运行，只能外部启动 4 个进程，然后 PP 内部每个进程在管理 2 张卡


def demo_dp_pp():
    # 假设 pp=2，dp=4，那么 wor
    functions = [demo_dp_pp_run]
    world_size = 8
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)


if __name__ == '__main__':
    # 8卡 pp
    demo_pp()

    # 8卡 dp
    demo_dp()

    # 8卡 dp+pp
    demo_dp_pp()
