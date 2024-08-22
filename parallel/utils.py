import os 
import torch
from torch import distributed as dist
from mmengine.dist import get_rank


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

