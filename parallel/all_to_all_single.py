import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
from torch import distributed as dist
import torch.nn.init as init
import random
import numpy as np
from torch import distributed as torch_dist
from typing import List, Optional, Tuple, Union, Any
from xpuyu.accelerate.ops import permute_func, unpermute_func
import math
import torch.nn.functional as F
import contextlib
import time
from datetime import datetime

WARMUP = 5


def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)


@contextlib.contextmanager
def maybe_enable_profiling(enable_profiling, dump_dir, profile_freq=10, global_step: int = 0):
    # get user defined profiler settings

    if enable_profiling:
        trace_dir = os.path.join(dump_dir, 'torch_profile')

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            print0(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            print0(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        print0(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert (
                wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
                on_trace_ready=trace_handler,
                record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        yield None


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
            inputs: torch.Tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group: dist.ProcessGroup = None,
            async_op=False,
    ) -> torch.Tensor:

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return inputs, None

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs) if output_split_sizes is None else
            inputs.new_empty(size=[sum(output_split_sizes)] +
                                  list(inputs.size()[1:])))
        handle = dist.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

        return out, handle

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, _) -> Tuple[None, torch.Tensor]:
        if ctx.needs_input_grad[0]:
            world_size = dist.get_world_size(group=ctx.group)
            if world_size == 1:
                return grad_output, None, None, None, None

            grad_output = grad_output.contiguous()
            out = torch.empty(
                ctx.input_shape,
                device=grad_output.device,
                dtype=grad_output.dtype)
            dist.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(x,
               output_split_sizes=None,
               input_split_sizes=None,
               group=None,
               async_op=False):
    return _AllToAll.apply(x, output_split_sizes, input_split_sizes, group,
                           async_op)


def permute(tokens, indices, num_out_tokens: int = None):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should equal the number of tokens not dropped. By default, set to None, meaning no tokens are dropped.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


class MoEGate(nn.Module):
    def __init__(self, hidden_size, n_routed_experts, top_k, balance=True):
        super(MoEGate, self).__init__()

        self.balance = balance
        self.noise = None
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.gating_dim = hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.balance:
            if self.noise is None:
                self.noise = torch.randn_like(logits) * 50
            logits = logits + self.noise

        scores = logits.softmax(dim=-1, dtype=torch.float32)

        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
        return topk_idx, topk_weight


def old_logic(hidden_states, topk_ids, moe_gate, ep_mesh=None, ep_size=1):
    n_routed_experts = moe_gate.n_routed_experts
    experts_per_rank = n_routed_experts // ep_size

    tokens_per_expert = torch.histc(
        topk_ids,
        bins=n_routed_experts,
        min=0,
        max=n_routed_experts)

    permutated_local_input_tokens, reversed_local_input_permutation_mapping = permute_func(
        hidden_states, topk_ids.to(torch.int32))

    # if rank==0:
    #     print(rank, topk_ids, 'xxxxxx')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    ############################################################################################
    start_event.record()

    input_splits = (tokens_per_expert.reshape(ep_size, experts_per_rank).sum(dim=1).cpu().numpy())

    # (e, )
    tokens_per_expert_group = tokens_per_expert.new_empty(
        tokens_per_expert.shape[0])
    # 每个ep rank负责的experts有多少个对应tokens (n_routed_experts, )
    # (r0e0, r0e1, ..., r0ei-1, r1e0, r1e1, ..., r1ei-1, r2e0, ...)
    # print(tokens_per_expert_group.shape, tokens_per_expert.shape,'xxxxxxxxxx')
    dist.all_to_all_single(
        tokens_per_expert_group,
        tokens_per_expert,
        group=None)

    # (r0e0, r0e1, ..., r0ei-1,
    #  r1e0, r1e1, ..., r1ei-1,
    tokens_per_expert_group = tokens_per_expert_group.view(
        ep_size, -1)

    # 当前 device 要算的，来自不同ep rank的tokens数
    output_splits = (
        tokens_per_expert_group.sum(dim=-1).to(
            torch.device('cpu')).numpy())

    global_input_tokens_old, _ = all_to_all(permutated_local_input_tokens,
                                            output_splits, input_splits,
                                            None)
    end_event.record()
    torch.cuda.synchronize()

    return global_input_tokens_old, start_event.elapsed_time(end_event)


def new_logic(hidden_states, topk_ids, moe_gate, ep_mesh=None, ep_size=1):
    n_routed_experts = moe_gate.n_routed_experts
    experts_per_rank = n_routed_experts // ep_size

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    ################################################################################################
    start_event.record()

    inter_ep_size = ep_mesh.size(0)
    intra_ep_size = ep_mesh.size(1)

    # 这个地方必须用 float32 不能用 bf16，否则在 16384 情况下会出错
    input_tokens_index = torch.arange(0, hidden_states.size(0), device=device, dtype=torch.float32)
    # print(f'{input_tokens_index.max()}----------')

    # 计算后，将 2x7 个 token 扩展的 8 倍 token，按照专家0 专家1 ... 专家32 重新排列
    # 这个地方不太鲁棒，如果没有 repeat 会出现 cuda 非法内存越界，但是 repeat 多少次合适不清楚
    # permutated_local_input_tokens_index, _ = permute_func(
    #     input_tokens_index.view(-1, 1).repeat([1, 4]), topk_ids.to(torch.int32))
    # permutated_local_input_tokens_index = permutated_local_input_tokens_index[:, 0]

    permutated_local_input_tokens_index, _ = permute(input_tokens_index, topk_ids.to(torch.int32))

    # if rank==0:
    #     print(permutated_local_input_tokens_index, 'permutated_local_input_tokens_index')

    # 计算每个节点所需要处理的 token index
    num_expert_per_node = experts_per_rank * intra_ep_size

    # 需要提前计算当前 token 中每个 token 要发给每个 ep rank 多少个 token
    bins = torch.arange(0, n_routed_experts + 1, n_routed_experts // ep_size, device=device)
    # 使用 torch.bucketize 来统计每个区间的数量
    indices = torch.bucketize(topk_ids.unsqueeze(-1), bins - 1).squeeze(-1) - 1  # 减1因为我们要的是落在区间内的计数
    indices = torch.clamp(indices, 0, ep_size - 1)
    num_token_pre_ep_rank = torch.zeros((topk_ids.size(0), ep_size), dtype=topk_ids.dtype, device=device)
    num_token_pre_ep_rank.scatter_add_(1, indices, torch.ones_like(num_token_pre_ep_rank))

    # if rank==0:
    #     print(f'{num_token_pre_ep_rank} 222222')

    send_node_hidden_states_list = []
    seed_node_shape0_list = []
    num_token_pre_node_list = []
    start_node_tokens = 0
    for i in range(inter_ep_size):
        start_index = i * num_expert_per_node
        end_index = (i + 1) * num_expert_per_node

        end_node_tokens = ((topk_ids >= start_index) & (topk_ids < end_index)).sum()
        send_node_ids = permutated_local_input_tokens_index[
                        start_node_tokens:start_node_tokens + end_node_tokens].unique()
        send_node_hidden_states_list.append(hidden_states[send_node_ids.int()])
        num_token_pre_node_list.append(num_token_pre_ep_rank[send_node_ids.int()])
        seed_node_shape0_list.append(send_node_hidden_states_list[-1].shape[0])

        #    if rank==0:
        #         print(start_node_tokens,end_node_tokens, send_node_ids, send_node_hidden_states_list[-1].shape, num_token_pre_node_list[-1].shape)

        start_node_tokens = start_node_tokens + end_node_tokens

    # 执行节点间 all-to-all
    # 需要提前准备好输出 shape
    seed_node_shape0_list = torch.tensor(seed_node_shape0_list, dtype=torch.int, device=hidden_states.device)
    output_shape = torch.empty([len(seed_node_shape0_list)], dtype=torch.int, device=hidden_states.device)
    dist.all_to_all_single(
        output_shape,
        seed_node_shape0_list,
        group=ep_mesh.get_group('inter'))
    # print(f'{seed_node_shape0_list, output_shape, ep_mesh.get_local_rank(0)}==========')

    # 节点间已经去重的 token
    recv_node_token_list = [torch.empty((output_shape[i].item(), moe_gate.hidden_size),
                                        device=hidden_states.device,
                                        dtype=hidden_states.dtype)
                            for i in range(len(output_shape))]
    dist.all_to_all(recv_node_token_list, send_node_hidden_states_list, group=ep_mesh.get_group('inter'))
    # recv_node_token_list 表示当前节点所处节点内 ep rank 所处理的已经出重后的所有 token
    # print([i.shape for i in recv_node_token_list],ep_mesh.get_local_rank(0),'xxxxx')

    recv_node_num_token_list = [torch.empty((output_shape[i].item(), ep_size),
                                            device=hidden_states.device,
                                            dtype=num_token_pre_node_list[0].dtype)
                                for i in range(len(output_shape))]
    dist.all_to_all(recv_node_num_token_list, num_token_pre_node_list, group=ep_mesh.get_group('inter'))
    # print([i.shape for i in recv_node_num_token_list],ep_mesh.get_local_rank(0),'xxxxx')

    # 节点内通信
    # 需要确定在节点内，当前 rank 需要和其余 rank 交互多少 token，简而言之就是要将之前去重的还原回来
    # node_rank = ep_mesh.get_local_rank(0)

    # old 逻辑
    # hidden_states_full_list = [[0 for j in range(len(recv_node_token_list))] for i in range(intra_ep_size)]

    # for i in range(len(recv_node_token_list)):
    #     # 切分规则很有讲究，需要把当前节点内的数据重新排列，假设是 4x2，那么这个 list 长度是 4，每个里面顺序是 0-1
    #     # 因此如果直接 cat 会变成 0 1 0 1，我们需要重新变成 0 0 1 1 方便后续 all-2-all
    #     recv_node_token_i = recv_node_token_list[i]
    #     recv_node_num_token_i = recv_node_num_token_list[i]

    #     num_tokens_pre_node_current_node_i = recv_node_num_token_i[:,
    #                                          node_rank * intra_ep_size:(node_rank + 1) * intra_ep_size]
    #     for j in range(intra_ep_size):
    #         # 找出哪些 token 是本 ep rank 真正需要的
    #         current_ep_mask = num_tokens_pre_node_current_node_i[:, j] > 0
    #         hidden_states_full_i = recv_node_token_i[current_ep_mask]
    #         # 需要复制多少份
    #         num_repeat_interleave = num_tokens_pre_node_current_node_i[:, j][current_ep_mask]
    #         hidden_states_full_i = hidden_states_full_i.repeat_interleave(num_repeat_interleave, dim=0)

    #         hidden_states_full_list[j][i] = hidden_states_full_i

    # for i, hidden_states_full in enumerate(hidden_states_full_list):
    #     hidden_states_full_list[i] = torch.cat(hidden_states_full, dim=0)

    # 新逻辑
    hidden_states_pre_node = torch.cat(recv_node_token_list, dim=0)  # n,hidden_size
    num_tokens_pre_node = torch.cat(recv_node_num_token_list, dim=0)  # n,ep_size

    # # 只需要当前节点的数据即可
    node_rank = ep_mesh.get_local_rank(0)
    num_tokens_pre_node_current_node = num_tokens_pre_node[:, node_rank * intra_ep_size:(node_rank + 1) * intra_ep_size]

    hidden_states_full_list = []
    for i in range(intra_ep_size):
        num_tokens_pre_node_current_node_i = num_tokens_pre_node_current_node[:, i]
        # repeat_interleave 具备，如果重复次数是 0，则把当前数据删除功能
        hidden_states_full_i = hidden_states_pre_node.repeat_interleave(num_tokens_pre_node_current_node_i, dim=0)
        hidden_states_full_list.append(hidden_states_full_i)

    # print([i.shape for i in hidden_states_full_list],ep_mesh.get_local_rank(1),'yyyyyy')
    # 节点内 all-2-all

    # 需要提前准备好输出 shape
    seed_node_shape0_list = [hidden_states_full_i.size(0) for hidden_states_full_i in hidden_states_full_list]
    seed_node_shape0_list = torch.tensor(seed_node_shape0_list, dtype=torch.int, device=hidden_states.device)
    output_shape = torch.empty([len(seed_node_shape0_list)], dtype=torch.int, device=hidden_states.device)
    dist.all_to_all_single(
        output_shape,
        seed_node_shape0_list,
        group=ep_mesh.get_group('intra'))

    recv_node_token_list = [torch.empty((output_shape[i].item(), moe_gate.hidden_size),
                                        device=hidden_states.device,
                                        dtype=hidden_states.dtype)
                            for i in range(len(output_shape))]
    dist.all_to_all(recv_node_token_list, hidden_states_full_list, group=ep_mesh.get_group('intra'))

    global_input_tokens = torch.cat(recv_node_token_list, dim=0)

    end_event.record()
    torch.cuda.synchronize()

    return global_input_tokens, start_event.elapsed_time(end_event)


def new_logic_microbs(hidden_states, topk_ids, moe_gate, ep_mesh=None, ep_size=1, num_chunk=1):
    n_routed_experts = moe_gate.n_routed_experts
    experts_per_rank = n_routed_experts // ep_size

    inter_ep_size = ep_mesh.size(0)
    intra_ep_size = ep_mesh.size(1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    ################################################################################################
    start_event.record()

    # 按照 指定切分数 维度 split hidden_states 和 topk_ids
    assert hidden_states.size(0) % num_chunk == 0
    hidden_states_list = torch.split(hidden_states, hidden_states.size(0) // num_chunk)
    topk_ids_list = torch.split(topk_ids, hidden_states.size(0) // num_chunk)
    global_input_tokens_list = []

    chunk_recv_node_token_list = []
    chunk_recv_node_num_token_list = []
    chunk_handles = []

    for index in range(len(hidden_states_list)):
        hidden_states = hidden_states_list[index]
        topk_ids = topk_ids_list[index]

        # 这个地方必须用 float32 不能用 bf16，否则在 16384 情况下会出错
        input_tokens_index = torch.arange(0, hidden_states.size(0), device=device, dtype=torch.float32)
        # print(f'{input_tokens_index.max()}----------')

        permutated_local_input_tokens_index, _ = permute(input_tokens_index, topk_ids.to(torch.int32))

        # 计算后，将 2x7 个 token 扩展的 8 倍 token，按照专家0 专家1 ... 专家32 重新排列
        # 这个地方不太鲁棒，如果没有 repeat 会出现 cuda 非法内存越界，但是 repeat 多少次合适不清楚
        # permutated_local_input_tokens_index, _ = permute_func(
        #     input_tokens_index.view(-1, 1).repeat([1, 16]), topk_ids.to(torch.int32))
        # permutated_local_input_tokens_index = permutated_local_input_tokens_index[:, 0]

        # if rank==0:
        #     print(permutated_local_input_tokens_index, 'permutated_local_input_tokens_index')

        # 计算每个节点所需要处理的 token index
        num_expert_per_node = experts_per_rank * intra_ep_size

        # 需要提前计算当前 token 中每个 token 要发给每个 ep rank 多少个 token
        bins = torch.arange(0, n_routed_experts + 1, n_routed_experts // ep_size, device=device)
        # 使用 torch.bucketize 来统计每个区间的数量
        indices = torch.bucketize(topk_ids.unsqueeze(-1), bins - 1).squeeze(-1) - 1  # 减1因为我们要的是落在区间内的计数
        indices = torch.clamp(indices, 0, ep_size - 1)
        num_token_pre_ep_rank = torch.zeros((topk_ids.size(0), ep_size), dtype=topk_ids.dtype, device=device)
        num_token_pre_ep_rank.scatter_add_(1, indices, torch.ones_like(num_token_pre_ep_rank))

        # if rank==0:
        #     print(f'{num_token_pre_ep_rank} 222222')

        send_node_hidden_states_list = []
        seed_node_shape0_list = []
        num_token_pre_node_list = []
        start_node_tokens = 0

        for i in range(inter_ep_size):
            start_index = i * num_expert_per_node
            end_index = (i + 1) * num_expert_per_node

            end_node_tokens = ((topk_ids >= start_index) & (topk_ids < end_index)).sum()
            send_node_ids = permutated_local_input_tokens_index[
                            start_node_tokens:start_node_tokens + end_node_tokens].unique()
            send_node_hidden_states_list.append(hidden_states[send_node_ids.int()])
            num_token_pre_node_list.append(num_token_pre_ep_rank[send_node_ids.int()])
            seed_node_shape0_list.append(send_node_hidden_states_list[-1].shape[0])

            #    if rank==0:
            #         print(start_node_tokens,end_node_tokens, send_node_ids, send_node_hidden_states_list[-1].shape, num_token_pre_node_list[-1].shape)

            start_node_tokens = start_node_tokens + end_node_tokens

        # 执行节点间 all-to-all
        # 需要提前准备好输出 shape
        seed_node_shape0_list = torch.tensor(seed_node_shape0_list, dtype=torch.int, device=hidden_states.device)
        output_shape = torch.empty([len(seed_node_shape0_list)], dtype=torch.int, device=hidden_states.device)
        dist.all_to_all_single(
            output_shape,
            seed_node_shape0_list,
            group=ep_mesh.get_group('inter'))
        # print(f'{seed_node_shape0_list, output_shape, ep_mesh.get_local_rank(0)}==========')
        recv_node_num_token_list = [torch.empty((output_shape[i].item(), ep_size),
                                                device=hidden_states.device,
                                                dtype=num_token_pre_node_list[0].dtype)
                                    for i in range(len(output_shape))]
        dist.all_to_all(recv_node_num_token_list, num_token_pre_node_list, group=ep_mesh.get_group('inter'))

        # 节点间已经去重的 token
        recv_node_token_list = [torch.empty((output_shape[i].item(), moe_gate.hidden_size),
                                            device=hidden_states.device,
                                            dtype=hidden_states.dtype)
                                for i in range(len(output_shape))]
        handle = dist.all_to_all(recv_node_token_list, send_node_hidden_states_list, group=ep_mesh.get_group('inter'),
                                 async_op=True)

        chunk_recv_node_token_list.append(recv_node_token_list)
        chunk_recv_node_num_token_list.append(recv_node_num_token_list)
        chunk_handles.append(handle)
        # recv_node_token_list 表示当前节点所处节点内 ep rank 所处理的已经出重后的所有 token
        # print([i.shape for i in recv_node_token_list],ep_mesh.get_local_rank(0),'xxxxx')

        # print([i.shape for i in recv_node_num_token_list],ep_mesh.get_local_rank(0),'xxxxx')

        # 节点内通信
        # 需要确定在节点内，当前 rank 需要和其余 rank 交互多少 token，简而言之就是要将之前去重的还原回来
        # node_rank = ep_mesh.get_local_rank(0)

        # old 逻辑
        # hidden_states_full_list = [[0 for j in range(len(recv_node_token_list))] for i in range(intra_ep_size)]

        # for i in range(len(recv_node_token_list)):
        #     # 切分规则很有讲究，需要把当前节点内的数据重新排列，假设是 4x2，那么这个 list 长度是 4，每个里面顺序是 0-1
        #     # 因此如果直接 cat 会变成 0 1 0 1，我们需要重新变成 0 0 1 1 方便后续 all-2-all
        #     recv_node_token_i = recv_node_token_list[i]
        #     recv_node_num_token_i = recv_node_num_token_list[i]

        #     num_tokens_pre_node_current_node_i = recv_node_num_token_i[:,
        #                                          node_rank * intra_ep_size:(node_rank + 1) * intra_ep_size]
        #     for j in range(intra_ep_size):
        #         # 找出哪些 token 是本 ep rank 真正需要的
        #         current_ep_mask = num_tokens_pre_node_current_node_i[:, j] > 0
        #         hidden_states_full_i = recv_node_token_i[current_ep_mask]
        #         # 需要复制多少份
        #         num_repeat_interleave = num_tokens_pre_node_current_node_i[:, j][current_ep_mask]
        #         hidden_states_full_i = hidden_states_full_i.repeat_interleave(num_repeat_interleave, dim=0)

        #         hidden_states_full_list[j][i] = hidden_states_full_i

        # for i, hidden_states_full in enumerate(hidden_states_full_list):
        #     hidden_states_full_list[i] = torch.cat(hidden_states_full, dim=0)

    chunk_handles_1=[]
    chunk_recv_node_token_list=[]

    for index in range(len(hidden_states_list)):
        handle = chunk_handles[index]
        handle.wait()

        recv_node_token_list = chunk_recv_node_token_list[index]
        recv_node_num_token_list = chunk_recv_node_num_token_list[index]

        # 新逻辑
        hidden_states_pre_node = torch.cat(recv_node_token_list, dim=0)  # n,hidden_size
        num_tokens_pre_node = torch.cat(recv_node_num_token_list, dim=0)  # n,ep_size

        # # 只需要当前节点的数据即可
        node_rank = ep_mesh.get_local_rank(0)
        num_tokens_pre_node_current_node = num_tokens_pre_node[:,
                                           node_rank * intra_ep_size:(node_rank + 1) * intra_ep_size]

        hidden_states_full_list = []
        for i in range(intra_ep_size):
            num_tokens_pre_node_current_node_i = num_tokens_pre_node_current_node[:, i]
            # repeat_interleave 具备，如果重复次数是 0，则把当前数据删除功能
            hidden_states_full_i = hidden_states_pre_node.repeat_interleave(num_tokens_pre_node_current_node_i, dim=0)
            hidden_states_full_list.append(hidden_states_full_i)

        # print([i.shape for i in hidden_states_full_list],ep_mesh.get_local_rank(1),'yyyyyy')
        # 节点内 all-2-all

        # 需要提前准备好输出 shape
        seed_node_shape0_list = [hidden_states_full_i.size(0) for hidden_states_full_i in hidden_states_full_list]
        seed_node_shape0_list = torch.tensor(seed_node_shape0_list, dtype=torch.int, device=hidden_states.device)
        output_shape = torch.empty([len(seed_node_shape0_list)], dtype=torch.int, device=hidden_states.device)
        dist.all_to_all_single(
            output_shape,
            seed_node_shape0_list,
            group=ep_mesh.get_group('intra'))

        recv_node_token_list = [torch.empty((output_shape[i].item(), moe_gate.hidden_size),
                                            device=hidden_states.device,
                                            dtype=hidden_states.dtype)
                                for i in range(len(output_shape))]
        handle = dist.all_to_all(recv_node_token_list, hidden_states_full_list, group=ep_mesh.get_group('intra'), async_op=True)
        chunk_handles_1.append(handle)
        chunk_recv_node_token_list.append(recv_node_token_list)
        # global_input_tokens = torch.cat(recv_node_token_list, dim=0)
        # global_input_tokens_list.append(global_input_tokens)

    for i, handle in enumerate(chunk_handles_1):
        handle.wait()
        recv_node_token_list = chunk_recv_node_token_list[i]
        global_input_tokens = torch.cat(recv_node_token_list, dim=0)
        global_input_tokens_list.append(global_input_tokens)

    global_input_tokens = torch.cat(global_input_tokens_list, dim=0)

    end_event.record()
    torch.cuda.synchronize()

    del chunk_handles, chunk_handles_1, chunk_recv_node_token_list, chunk_recv_node_num_token_list

    return global_input_tokens, start_event.elapsed_time(end_event)


from mmengine.runner import set_random_seed

import numpy as np


def calculate_statistics(lst):
    if not lst:
        raise ValueError("列表不能为空")

    max_value = max(lst)
    min_value = min(lst)
    mean_value = np.mean(lst)
    variance_value = np.var(lst)

    return max_value, min_value, mean_value, variance_value


# torchrun --nnodes=1 --nproc_per_node=8 all_to_all_demo.py
# torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=${KUBERNETES_CONTAINER_RESOURCE_GPU} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} all_to_all_demo.py
if __name__ == '__main__':
    from xtuner._lite.parallel import setup_parallel
    from torch.distributed.device_mesh import init_device_mesh

    setup_parallel()
    # set_random_seed(1024)

    balance = True
    enable_profiling = True
    work_dir = 'all_to_all_workdirs'
    profile_freq = 10
    run_total_step = 80

    run_all_code = 0  # 0 全部运行，1 只运行 old_logic，2 只运行 new_logic 3 只运行 mcirobs
    num_chunk = 2

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    torch_dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    work_dir = os.path.join(work_dir, timestamp)
    os.makedirs(os.path.expanduser(work_dir), mode=0o777, exist_ok=True)

    hidden_size = 5120  # 16
    n_routed_experts = 256
    top_k = 8
    simpler_shape = (2, 4096, hidden_size)

    world_size = dist.get_world_size()
    ep_size = world_size

    if ep_size == 8:  # 8 卡模拟
        ep_mesh = init_device_mesh('cuda', (4, 2), mesh_dim_names=('inter', 'intra'))
    elif ep_size == 16:  # 实际的 2x8 卡配置
        ep_mesh = init_device_mesh('cuda', (2, 8), mesh_dim_names=('inter', 'intra'))
    elif ep_size == 32:  # 实际的 4x8 卡配置
        ep_mesh = init_device_mesh('cuda', (4, 8), mesh_dim_names=('inter', 'intra'))
    else:
        raise NotImplementedError()

    print0(f'ep_size: [{ep_size}]; shape: [{simpler_shape}]; ep_mesh: [{ep_mesh}]')

    moe_gate = MoEGate(hidden_size, n_routed_experts, top_k, balance).to(torch.bfloat16)
    moe_gate = moe_gate.to('cuda')

    # dim 数值不能乱写，否则 permute_func 后会出现非法越界错误
    hidden_states = torch.rand(simpler_shape).to('cuda').to(torch.bfloat16)

    device = hidden_states.device

    topk_ids, _ = moe_gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    old_times = []
    new_times = []
    torch.cuda.synchronize()
    with maybe_enable_profiling(enable_profiling, work_dir, profile_freq, global_step=0) as torch_profiler:
        for i in range(run_total_step):

            if run_all_code == 1 or run_all_code == 0:
                torch.cuda.synchronize()
                global_input_tokens_old, old_time = old_logic(hidden_states, topk_ids, moe_gate, ep_mesh, ep_size)
                torch.cuda.synchronize()

            # if run_all_code == 2 or run_all_code == 0:
            #     global_input_tokens, new_time = new_logic(hidden_states, topk_ids, moe_gate, ep_mesh, ep_size)
            #     torch.cuda.synchronize()

            if run_all_code == 3 or run_all_code == 0:
                global_input_tokens, new_time = new_logic_microbs(hidden_states, topk_ids, moe_gate, ep_mesh, ep_size,
                                                                  num_chunk)
                torch.cuda.synchronize()

            if torch_profiler:
                torch_profiler.step()

            if run_all_code == 1 or run_all_code == 0:
                old_times.append(old_time)
                print0(f"iter[{i}] old version - {old_time:.2f}ms")
            if run_all_code == 2 or run_all_code == 0:
                new_times.append(new_time)
                print0(f"iter[{i}] new version - {new_time:.2f}ms")

            if run_all_code == 0:
                # 一种取巧的判断逻辑是否正确的做法
                # 对 hidden size 维度求和，排序后判断是否完全相同。因为 premute 后顺序可能不一样，但是总的数据肯定是一样的
                are_equal = torch.equal(torch.sort(global_input_tokens.sum(1)).values,
                                        torch.sort(global_input_tokens_old.sum(1)).values)
                if not are_equal:
                    print(
                        f'error ======{global_input_tokens.shape, global_input_tokens_old.shape, global_input_tokens.sum(), global_input_tokens_old.sum()}======')
                assert are_equal, f'======{global_input_tokens.shape, global_input_tokens_old.shape, global_input_tokens.sum(), global_input_tokens_old.sum()}======'

    print0("\nPerformance Results:")
    if run_all_code == 1 or run_all_code == 0:
        max_value, min_value, mean_value, variance_value = calculate_statistics(old_times[10:])
        print0(
            f"total old version[ms] -Mix {min_value:.2f}--Max {max_value:.2f}-Mean {mean_value:.2f}-Var {variance_value:.2f}")
    if run_all_code == 2 or run_all_code == 0:
        max_value, min_value, mean_value, variance_value = calculate_statistics(new_times[10:])
        print0(
            f"total new version[ms] -Mix {min_value:.2f}--Max {max_value:.2f}-Mean {mean_value:.2f}-Var {variance_value:.2f}")

    dist.destroy_process_group()
