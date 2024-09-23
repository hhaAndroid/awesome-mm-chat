import torch
from torch.distributed import all_gather
import torch.nn as nn
from utils import _SP_GROUP, init_process, all_to_all, _ReduceLoss
from mmengine.runner import set_random_seed
from mmengine.dist import get_world_size, get_rank, all_reduce
import torch.multiprocessing as mp


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # split
        world_size = get_world_size()
        rank = get_rank()
        if world_size == 1:
            return input_

        # Split along last dimension.
        last_dim = input_.dim() - 1
        last_dim_size = input_.size()[last_dim] // world_size  # 均分
        tensor_list = torch.split(input_, last_dim_size, dim=last_dim)
        output = tensor_list[rank].contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):  # all-gather
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
        grad_output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return grad_output


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


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        world_size = get_world_size()
        rank = get_rank()
        # x: bld
        x = self.fc1(x)
        # 切分
        x = torch.split(x, x.shape[1] // world_size, dim=1)
        x = x[rank].contiguous()
        x = self.fc2(x)
        # 聚合
        x = _GatherFromModelParallelRegion.apply(x)
        # tensor_list = [torch.empty_like(x) for _ in range(world_size)]
        # all_gather(tensor_list, x)
        # x = torch.cat(tensor_list, dim=1).contiguous()
        x = self.fc3(x)
        return x


def demo1(*args, **kwargs):
    set_random_seed(42)
    model = SimpleModel()
    print(model)

    x = torch.randn(2, 100, 10, requires_grad=True)
    x = model(x)
    loss = x.mean()
    loss.backward()
    rank = get_rank()
    if rank == 0:
        print(rank, loss, model.fc1.weight.grad, model.fc2.weight.grad, model.fc3.weight.grad,
              '=====================\n')


if __name__ == '__main__':
    functions = [demo1]
    world_size = 4
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
