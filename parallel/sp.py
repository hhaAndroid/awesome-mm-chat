import torch
from torch.utils.data import DataLoader, Dataset
from torch import distributed as dist
from mmengine.dist import all_gather, get_world_size, get_rank, all_reduce
import torch.multiprocessing as mp
from mmengine.runner import set_random_seed
import torch
import torch.nn.functional as F
from torch import nn
from utils import _SP_GROUP, init_process, all_to_all, _ReduceLoss


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

        # =================================================================================
        # 在计算 attention 时，需要对序列维度进行合并，否则计算是不对的
        if use_sp:
            # all-to-all 在 head 维度切分，并且在 s 维度合并
            # b, h, s_div_sp, d = xq.shape
            # -> b, h/sp, s, d
            xq = all_to_all(xq, _SP_GROUP, scatter_dim=1, gather_dim=2)
            keys = all_to_all(keys, _SP_GROUP, scatter_dim=1, gather_dim=2)
            values = all_to_all(values, _SP_GROUP, scatter_dim=1, gather_dim=2)

        scores = torch.matmul(xq, keys.transpose(2, 3))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        if use_sp:
            # all-to-all 在 s 维度切分，并且在 head 维度合并
            # b, h/sp, s, d ->b, h, s_div_sp, d
            output = all_to_all(output, _SP_GROUP, scatter_dim=2, gather_dim=1)
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


def demo1(*args, **kwargs):
    set_random_seed(42)
    model = TransformerBlock()
    dataset = MyDataset(torch.ones(1000, 64, 128))

    # 没有 dp，所以不需要 sampler
    # 输出数据是完全一样的，只是会在 dataloader 输出后的数据进行序列维度切分
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for data in dataloader:
        # =========================================
        # 序列维度切分 4,64,128
        # 序列长度均分，每张卡只获取自己的部分
        world_size = get_world_size()
        rank = get_rank()
        dim_size = data.size(1)  # 序列长度
        tensor_list = torch.split(data, dim_size // world_size, dim=1)
        data = tensor_list[rank].contiguous()  # 4,32,128
        # =========================================

        outputs = model(data, use_sp=True)
        # loss 切分为 2 部分了。4,32,128
        loss = outputs.mean()
        # outputs.mean().backward()
        # =========================================
        # 以下计算非必须，只是为了让 loss 看起来一样,但是这种写法未来会没有梯度，因此不推荐
        # all_reduce(loss, op='mean', group=_SP_GROUP)
        # 1/2 是因为 sp=2
        loss.backward()
        _ReduceLoss.apply(loss, torch.tensor(1/2), _SP_GROUP)

        all_reduce(model.feed_forward.w1.weight.grad, op='mean', group=_SP_GROUP)
        if dist.get_rank() == 0:
            print(f" reduce loss {loss}, reduce grad mean: {model.feed_forward.w1.weight.grad.mean()}")
        break


if __name__ == '__main__':
    # 单卡测试
    set_random_seed(42)
    model = TransformerBlock()
    dataset = MyDataset(torch.ones(100, 64, 128))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for data in dataloader:
        outputs = model(data)
        outputs.mean().backward()

        print(f'{outputs.mean()},{model.feed_forward.w1.weight.grad.mean()}')
        break

    # 2 卡序列并行
    # 在序列长度维度进行切分，之前输入是 4x64x128，现在 dataloader 输出不变，
    # 也就是说现在同一个 sp 组内部所面对的数据是一样的，然后自己切割自己需要的部分，此时每个 llm forward  面对的数据量就减少一半了
    # sp=2 无 dp
    functions = [demo1]
    world_size = 2
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
