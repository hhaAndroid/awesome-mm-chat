import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
import torch.multiprocessing as mp
import os
from torch import distributed as dist

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # 恒等变换
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # all-reduce
        if get_world_size() == 1:
            return grad_output
        torch.distributed.all_reduce(grad_output)
        return grad_output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # gather
        rank = get_rank()
        world_size = get_world_size()
        if world_size == 1:
            return input_

        # 最后一个维度进行 gather
        last_dim = input_.dim() - 1
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):  # split
        world_size = get_world_size()
        rank = get_rank()
        if world_size == 1:
            return grad_output

        # Split along last dimension.
        last_dim = grad_output.dim() - 1
        last_dim_size = grad_output.size()[last_dim] // world_size  # 均分
        tensor_list = torch.split(grad_output, last_dim_size, dim=last_dim)
        output = tensor_list[rank].contiguous()
        return output


def gather_grad(grad_output):
    rank = get_rank()
    world_size = get_world_size()
    if world_size == 1:
        return grad_output

    # 最后一个维度进行 gather
    last_dim = grad_output.dim() - 1
    tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
    tensor_list[rank] = grad_output
    torch.distributed.all_gather(tensor_list, grad_output)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


class ColumnParallelLinear(torch.nn.Module):  # 按列进行切分
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            gather_output: bool = True,
            init_method=nn.init.xavier_normal_,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        # 我们简单点，假设有多少张卡就切多少份
        world_size = get_world_size()

        # 必须要能够均分，否则有问题
        assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)
        # 每个 rank 的 分片 dim
        self.output_size_per_partition = out_features // world_size

        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        if world_size == 1:
            init_method(self.weight)
        else:
            # 先在 cpu 上初始化一个全量的，然后在切分，可以确保同步
            # 这个步骤应该只是为了和不切分时候状态一致，并不是必要的操作
            master_weight = torch.empty(out_features, in_features, dtype=self.weight.dtype, requires_grad=False)
            init_method(master_weight)
            # 输出维度切分
            weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
            rank = get_rank()  # 切分输入自己的部分
            my_weight_list = weight_list[rank::world_size]

            with torch.no_grad():  # 重新赋值
                torch.cat(my_weight_list, dim=0, out=self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # 把当前输入分发到每个 rank，同时要确保分发能反向传播，梯度是 all-reduce
        input_parallel = _CopyToModelParallelRegion.apply(input_)

        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output:
            # 合并输出
            output = _GatherFromModelParallelRegion.apply(output_parallel)
        else:
            # 输出每个 rank 结果
            output = output_parallel
        return output

    # 获取全量的权重
    def get_master_weight(self) -> torch.Tensor:
        # weight 的 shape 是 (output_size_per_partition, in_features)
        # 内部代码始终在最后一个维度聚合，所以需要先转置后聚合，得到 (in_features, output_size_per_partition)
        # 最后再转置回来 (output_size_per_partition, in_features)
        return _GatherFromModelParallelRegion.apply(self.weight.data.transpose(0, 1)).transpose_(0, 1)


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
    model1 = Linear(4, 8)
    set_random_seed(42)
    model2 = ColumnParallelLinear(4, 8, gather_output=True)

    input = torch.randn(3, 2, 4)
    output1 = model1(input)
    output2 = model2(input)
    print(f'\n====rank:{get_rank()} work size: {get_world_size()}====', output1.shape)
    assert torch.allclose(output1, output2, atol=1e-6)

    output1.sum().backward()
    output2.sum().backward()

    # 聚合梯度
    # model2.weight.grad shape 是 (output_size_per_partition, in_features)
    weights_grad = gather_grad(model2.weight.grad.transpose(0, 1)).transpose(0, 1)
    weights_bias = gather_grad(model2.bias.grad)
    assert torch.allclose(model1.weight.grad, weights_grad, atol=1e-6), 'backward error'
    assert torch.allclose(model1.bias.grad, weights_bias, atol=1e-6), 'backward error'


if __name__ == '__main__':
    # 单卡验证 forward 是否一致
    set_random_seed(42)
    model1 = Linear(4, 8)
    set_random_seed(42)
    model2 = ColumnParallelLinear(4, 8, gather_output=True)

    input = torch.randn(3, 2, 4)
    output1 = model1(input)
    output2 = model2(input)
    print(output1.shape)
    assert torch.allclose(output1, output2, atol=1e-6), 'forward error'

    # 验证反向传播梯度是否一致
    output1.sum().backward()
    output2.sum().backward()
    assert torch.allclose(model1.weight.grad, model2.weight.grad, atol=1e-6), 'backward error'
    assert torch.allclose(model1.bias.grad, model2.bias.grad, atol=1e-6), 'backward error'

    # 多卡验证是否一致
    functions = [demo1]
    world_size = 2
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
