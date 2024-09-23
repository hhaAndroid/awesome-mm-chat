import torch
from torch.utils.data import DataLoader, Dataset
from torch import distributed as dist
from mmengine.dist import all_gather, get_world_size, get_rank, all_reduce
import torch.multiprocessing as mp
from mmengine.runner import set_random_seed
import torch
import torch.nn.functional as F
from torch import nn
from utils import init_process

_SP_ULYESS_GROUP = None
_SP_RING_GROUP = None
_SP_ULYESS_GROUP_LIST = None
_SP_RING_GROUP_LIST = None


def set_seq_parallel_pg(
        sp_ulysses_degree, sp_ring_degree, rank, world_size, use_ulysses_low=True
):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    global _SP_ULYESS_GROUP, _SP_RING_GROUP, _SP_ULYESS_GROUP_LIST, _SP_RING_GROUP_LIST
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert (
            world_size % sp_degree == 0
    ), f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset,
                        (i + 1) * sp_ulysses_degree + offset,
                    )
                )
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    _SP_ULYESS_GROUP = group
                    _SP_ULYESS_GROUP_LIST = ulysses_ranks

            for i in range(num_ring_pgs):
                ring_ranks = list(range(i + offset, sp_degree + offset, num_ring_pgs))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    _SP_RING_GROUP = group
                    _SP_RING_GROUP_LIST = ring_ranks

    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(
                    range(
                        i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset
                    )
                )
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    _SP_RING_GROUP = group
                    _SP_RING_GROUP_LIST = ring_ranks

            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(i + offset, sp_degree + offset, num_ulysses_pgs)
                )
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    _SP_ULYESS_GROUP = group
                    _SP_ULYESS_GROUP_LIST = ulysses_ranks


def demo1(*args, **kwargs):
    set_seq_parallel_pg(8, 2, get_rank(), get_world_size(), use_ulysses_low=True)
    rank = get_rank()
    sp_ring_group = _SP_RING_GROUP
    sp_ulysess_group = _SP_ULYESS_GROUP
    sp_ring_rank = torch.distributed.get_rank(sp_ring_group)
    sp_ulysess_rank = torch.distributed.get_rank(sp_ulysess_group)
    # use_ulysses_low=True
    # 节点内使用 ulysses，节点间使用 ring
    print(
        f"rank: {rank}, {_SP_ULYESS_GROUP_LIST}, {_SP_RING_GROUP_LIST},sp_ulysess_rank: {sp_ulysess_rank}, sp_ring_rank: {sp_ring_rank}")


if __name__ == '__main__':
    functions = [demo1]
    world_size = 16
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
