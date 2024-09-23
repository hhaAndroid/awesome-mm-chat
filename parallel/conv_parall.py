import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
from torch import distributed as dist
import torch.nn.init as init
import random
import numpy as np
from torch import distributed as torch_dist


def is_distributed() -> bool:
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_rank(group=None) -> int:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def get_world_size(group = None) -> int:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def set_random_seed(seed, deterministic: bool = False) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # gather
        rank = get_rank()
        tp_world_size = get_world_size()
        if tp_world_size == 1:
            return input_

        # 最后一个维度进行 gather
        last_dim = 1
        tensor_list = [torch.empty_like(input_) for _ in range(tp_world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):  # split
        tp_world_size = get_world_size()
        rank = get_rank()
        if tp_world_size == 1:
            return grad_output

        # Split along last dimension.
        last_dim = 1
        last_dim_size = grad_output.size()[last_dim] // tp_world_size  # 均分
        tensor_list = torch.split(grad_output, last_dim_size, dim=last_dim)
        output = tensor_list[rank].contiguous()
        return output


class ParallelConv2d(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            init_method=nn.init.xavier_normal_,
    ) -> None:
        super(ParallelConv2d, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        world_size = get_world_size()

        # 必须要能够均分，否则有问题
        assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)
        # 每个 rank 的 分片 dim
        self.output_size_per_partition = out_features // world_size

        # kernel=5 可以自己修改
        self.conv = nn.Conv2d(in_features, out_features, 5, bias=True)
        init_method(self.conv.weight, gain=0.1)
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)  # 偏置初始化为0

        # 切分
        self.conv.weight = nn.Parameter(self.conv.weight.chunk(world_size, dim=0)[get_rank()])
        if self.conv.bias is not None:
            self.conv.bias = nn.Parameter(self.conv.bias.chunk(world_size, dim=0)[get_rank()])

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # input_ shape: (batch_size, in_features, height, width)
        # output shape: (batch_size, out_features, height, width)
        output = self.conv(input_)
        if get_world_size() > 1:
            output = _GatherFromModelParallelRegion.apply(output)
        return output


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
    model2 = ParallelConv2d(8, 16).to(device)
    set_random_seed(42)
    input = torch.randn(2, 8, 14, 14).to(device)
    output2 = model2(input)
    output2.sum().backward()

    shard = model2.conv.weight.grad
    tensor_list = [torch.empty_like(shard) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, shard)
    shard = torch.cat(tensor_list, dim=1)
    if get_rank() == 0:
        # print(output2)
        print(output2.shape)
        print(output2.sum())
        print(shard.sum())


if __name__ == '__main__':
    # 如果想本机直接 cpu 跑
    backend = 'gloo'
    # 如果有 4 张 gpu 跑
    # backend = 'nccl'

    if backend == 'nccl':
        device = 'cuda'
    else:
        device = 'cpu'

    set_random_seed(42)
    model2 = ParallelConv2d(8, 16).to(device)
    set_random_seed(42)
    input = torch.randn(2, 8, 14, 14).to(device)
    output2 = model2(input)
    # print(output2)
    print('=======单卡结果======')
    print(output2.shape)
    print(output2.sum())
    output2.sum().backward()
    print(model2.conv.weight.grad.sum())

    # 多卡验证是否一致
    print('=======分布式结果======')
    functions = [demo1]
    world_size = 4
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
