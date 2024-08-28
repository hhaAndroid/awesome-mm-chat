import os
import torch
from torch import distributed as dist
from mmengine.dist import get_rank, all_gather
from torch import Tensor
from typing import Any, Tuple


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
        func(device, backend)


_TP_GROUP = None
_DP_GROUP = None
_SP_GROUP = None
_PP_GROUP = None


def initialize_tp_dp_group(world_size, tp, dp, backend):
    global _TP_GROUP, _DP_GROUP

    groups = torch.LongTensor(range(world_size)).reshape(dp, tp)
    found = torch.where(groups == get_rank())
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # DP
    for k in range(tp):
        group = torch.distributed.new_group(groups[:, k].tolist(), backend=backend)
        # group = groups[:, k].tolist()
        if k == found[1]:
            _DP_GROUP = group

    # TP
    for k in range(dp):
        group = torch.distributed.new_group(groups[k, :].tolist(), backend=backend)
        # group = groups[k, :].tolist()
        if k == found[0]:
            _TP_GROUP = group


def initialize_sp_dp_group(world_size, sp, dp, backend):
    global _SP_GROUP, _DP_GROUP

    groups = torch.LongTensor(range(world_size)).reshape(dp, sp)
    found = torch.where(groups == get_rank())
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # DP
    for k in range(sp):
        group = torch.distributed.new_group(groups[:, k].tolist(), backend=backend)
        # group = groups[:, k].tolist()
        if k == found[1]:
            _DP_GROUP = group

    # SP
    for k in range(dp):
        group = torch.distributed.new_group(groups[k, :].tolist(), backend=backend)
        # group = groups[k, :].tolist()
        if k == found[0]:
            _SP_GROUP = group


def initialize_pp_dp_group(world_size, pp, dp, backend):
    global _PP_GROUP, _DP_GROUP

    groups = torch.LongTensor(range(world_size)).reshape(dp, pp)
    found = torch.where(groups == get_rank())
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # DP
    for k in range(pp):
        group = torch.distributed.new_group(groups[:, k].tolist(), backend=backend)
        # group = groups[:, k].tolist()
        if k == found[1]:
            _DP_GROUP = group

    # PP
    for k in range(dp):
        group = torch.distributed.new_group(groups[k, :].tolist(), backend=backend)
        # group = groups[k, :].tolist()
        if k == found[0]:
            _PP_GROUP = group


def _all_to_all(
        input: Tensor,
        world_size: int,
        group: dist.ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
):
    # cpu 模拟实现，低效但是正确
    all_inputs = all_gather(input.contiguous(), group=group)
    all_inputs = torch.concat(all_inputs, dim=gather_dim)
    output_list = [
        t.contiguous()
        for t in torch.tensor_split(all_inputs, world_size, scatter_dim)
    ]
    return output_list[get_rank(group)]

    # nccl
    # input_list = [
    #     t.contiguous()
    #     for t in torch.tensor_split(input, world_size, scatter_dim)
    # ]
    # output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    #
    # # 不支持 gloo 后端，只能用显卡跑
    # dist.all_to_all(output_list, input_list, group=group)
    # return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, sp_group: dist.ProcessGroup,
                scatter_dim: int, gather_dim: int):
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(sp_group)
        output = _all_to_all(input, ctx.world_size, sp_group, scatter_dim,
                             gather_dim)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple:
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.sp_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
        input: Tensor,
        sp_group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
):
    return _AllToAll.apply(input, sp_group, scatter_dim, gather_dim)


class _ReduceLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mean_loss, loss_scale, process_group):
        ctx.mode = process_group
        if loss_scale == 0:
            # convert nan to 0 just for logging
            mean_loss = torch.nan_to_num(mean_loss)
        loss_sum = mean_loss * loss_scale
        dist.all_reduce(loss_sum, group=process_group)
        dist.all_reduce(loss_scale, group=process_group)
        loss = loss_sum / loss_scale
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
