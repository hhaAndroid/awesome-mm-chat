import torch
import torch.nn as nn
from torch import distributed as dist
import os
import torch.multiprocessing as mp
from functools import partial
from torch.distributed.nn.functional import all_gather


def init_process(rank, world_size, functions, backend='gloo', **kwargs):
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
        func()


def calc_batch_loss(batch_input_ids, batch_labels):
    vocab = batch_input_ids.size(-1)
    loss_fn = nn.CrossEntropyLoss()
    batch_input_ids = batch_input_ids[:, :-1]
    batch_labels = batch_labels[:, 1:]
    print(loss_fn(batch_input_ids.reshape(-1, vocab), batch_labels.reshape(-1)))


def calc_batch_dp_loss(batch_input_ids=None, batch_labels=None):
    if dist.get_rank() == 0:
        print('=======多卡 batch 计算=======')
    # batch 维度切分
    world_size = dist.get_world_size()

    batch_input_ids = torch.split(batch_input_ids, len(batch_input_ids) // world_size, dim=0)[dist.get_rank()]
    batch_labels = torch.split(batch_labels, len(batch_labels) // world_size, dim=0)[dist.get_rank()]

    batch_input_ids = batch_input_ids[:, :-1]
    batch_labels = batch_labels[:, 1:]

    vocab = batch_input_ids.size(-1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(batch_input_ids.reshape(-1, vocab), batch_labels.reshape(-1))

    # reduce loss
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss = loss / world_size
    if dist.get_rank() == 0:
        print(loss)


def calc_batch_packing_loss(pack_input_ids, pack_labels, cumulative_lengths):
    cumulative_lengths_ = cumulative_lengths.clone()
    while cumulative_lengths_[-1] == len(pack_labels):
        cumulative_lengths_ = cumulative_lengths_[:-1]
    pack_labels[cumulative_lengths_] = -100  # 很关键, 但是因为 sft 时候始终满足条件

    pack_input_ids = pack_input_ids[..., :-1, :]
    pack_labels = pack_labels[..., 1:]
    loss_fc = nn.CrossEntropyLoss()
    loss = loss_fc(pack_input_ids, pack_labels)
    print(loss)


def rescale_sp_loss(loss_per_sp_rank,
                    labels_per_sp_rank,
                    sp_group: dist.ProcessGroup = None,
                    ignore_index=-100):
    import copy
    shift_labels = labels_per_sp_rank.view(-1)
    active_tokens = (shift_labels != ignore_index).long().sum()
    global_active_tokens = copy.deepcopy(active_tokens)
    dist.all_reduce(global_active_tokens, group=sp_group)
    loss_weight = active_tokens / global_active_tokens * dist.get_world_size(
        group=sp_group)

    if active_tokens == 0:
        # convert nan to 0 just for logging
        loss_per_sp_rank = torch.nan_to_num(loss_per_sp_rank)

    return loss_per_sp_rank * loss_weight


def calc_pack_sp_loss(pack_input_ids=None, pack_labels=None, cumulative_lengths=None):
    if dist.get_rank() == 0:
        print('=======多卡 sp pack 计算=======')
    world_size = dist.get_world_size()
    cumulative_lengths_ = cumulative_lengths.clone()
    while cumulative_lengths_[-1] == len(pack_labels):
        cumulative_lengths_ = cumulative_lengths_[:-1]
    pack_labels[cumulative_lengths_] = -100  # 很关键

    # 某个样本序列可能被切断导致无法对齐
    pack_input_ids = torch.split(pack_input_ids, len(pack_input_ids) // world_size, dim=0)[dist.get_rank()]
    pack_labels = torch.split(pack_labels, len(pack_labels) // world_size, dim=0)[dist.get_rank()]

    pack_input_ids = pack_input_ids[..., :-1, :]
    pack_labels = pack_labels[..., 1:]
    loss_fc = nn.CrossEntropyLoss()
    loss = loss_fc(pack_input_ids, pack_labels)

    # 相对更均衡
    loss = rescale_sp_loss(loss, pack_labels)

    # reduce loss
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss = loss / world_size
    if dist.get_rank() == 0:
        print(loss)


# https://zhuanlan.zhihu.com/p/721652210
# https://github.com/THUDM/LongAlign/issues/3
# 解决：当 bs > 1 时候现在主流做法都是 view(-1) 导致的问题，顺便解决了 bs=1 且 packing 的问题
# 并没有解决，多轮对话场景下，长输出和短输出存在的问题
# 如果要解决上述问题，还要指定每个样本中是由哪个轮次生成的，然后在 loss 计算时候，根据轮次进行 mask

# 以下是目前主流的各种情况 loss 计算方式，可以看出不仅仅是不对的，而且在单卡/dp/sp 情况下都不一样，存在不少问题
if __name__ == '__main__':
    a1 = torch.randn(5, 9).float()
    b1 = torch.Tensor([-100, -100, 1, 2, 3]).long()
    a2 = torch.randn(8, 9).float()
    b2 = torch.Tensor([4, -100, 3, 4, 6, -100, -100, 7]).long()
    a3 = torch.randn(3, 9).float()
    b3 = torch.Tensor([-100, 6, 8]).long()
    a4 = torch.randn(4, 9).float()
    b4 = torch.Tensor([-100, 7, 8, -100]).long()
    a5 = torch.randn(4, 9).float()
    b5 = torch.Tensor([-100, -100, 7, 4]).long()
    a6 = torch.randn(3, 9).float()
    b6 = torch.Tensor([5, 8, -100]).long()

    max_item_length = 8
    batch_input_ids = torch.zeros(8, max_item_length, 9)
    batch_labels = torch.ones(8, max_item_length).long() * -100
    for i, (a, b) in enumerate([(a1, b1), (a2, b2), (a3, b3), (a2, b2), (a6, b6), (a4, b4), (a5, b5), (a6, b6)]):
        batch_input_ids[i, :a.size(0)] = a
        batch_labels[i, :b.size(0)] = b

    pack_input_ids = torch.cat([a1, a2, a3, a2, a6, a4, a5, a6], dim=0)
    pack_labels = torch.cat([b1, b2, b3, b2, b6, b4, b5, b6], dim=0)
    num_tokens = torch.tensor([5, 8, 3, 8, 3, 4, 4, 3, 0])  # 可能有额外的一个 0
    _zero_length = torch.zeros(1)
    _pad_length = torch.cat([_zero_length, num_tokens]).int()
    # 模型只能拿到这个数据
    cumulative_lengths = torch.cumsum(_pad_length, 0).int()

    print('===========现在主流的但是不正确的 loss 计算方式===========')
    print('=======单卡 batch 计算=======')
    calc_batch_loss(batch_input_ids.clone(), batch_labels.clone())

    print('=======单卡 pack 计算=======')
    calc_batch_packing_loss(pack_input_ids.clone(), pack_labels.clone(), cumulative_lengths.clone())

    # 多卡
    partial_calc_batch_loss = partial(calc_batch_dp_loss, batch_input_ids=batch_input_ids.clone(),
                                      batch_labels=batch_labels.clone())
    partial_calc_pack_sp_loss = partial(calc_pack_sp_loss,
                                        pack_input_ids=pack_input_ids.clone(),
                                        pack_labels=pack_labels.clone(),
                                        cumulative_lengths=cumulative_lengths.clone())
    functions = [partial_calc_batch_loss, partial_calc_pack_sp_loss]
    world_size = 2
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
