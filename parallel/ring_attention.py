from mmengine.runner import set_random_seed
import torch
from utils import init_process
import torch.multiprocessing as mp
from mmengine.dist import all_gather, get_world_size, get_rank, all_reduce
from torch.distributed import batch_isend_irecv, P2POp, isend, irecv


def single_demo():
    set_random_seed(42)
    q = torch.randn((20, 5))
    k = torch.randn((16, 5))
    v = torch.randn((16, 5))
    c = torch.matmul(q, k.T)
    weight = torch.softmax(c, dim=-1)
    value = torch.matmul(weight, v)
    print(value)
    print(value.mean())


def step_kv_send_recv(send_k: torch.Tensor, recv_k: torch.Tensor,
                      send_v: torch.Tensor, recv_v: torch.Tensor):
    seq_world_size = get_world_size()
    seq_rank = get_rank()

    all_handles = []

    # 倒序传送 0 <- 1 <- 2 <- 3 <- 0
    send_rank = (seq_rank - 1) % seq_world_size
    recv_rank = (seq_rank + 1) % seq_world_size

    # send and recv k, v
    all_handles.append(P2POp(op=isend, tensor=send_k, peer=send_rank % seq_world_size))
    all_handles.append(P2POp(op=irecv, tensor=recv_k, peer=recv_rank % seq_world_size))
    all_handles.append(P2POp(op=isend, tensor=send_v, peer=send_rank % seq_world_size))
    all_handles.append(P2POp(op=irecv, tensor=recv_v, peer=recv_rank % seq_world_size))

    reqs = batch_isend_irecv(all_handles)
    for req in reqs:
        req.wait()


# 不能直接原地覆盖数据，因此要设置长度为 2 的缓存区，发送和接收数据交替使用
def prepare_kv_double_buffer(k, v):
    # Move k,v into buffer
    buffer_k = []
    buffer_v = []
    buffer_k.append(k)
    buffer_k.append(torch.empty_like(k))
    buffer_v.append(v)
    buffer_v.append(torch.empty_like(v))
    return buffer_k, buffer_v


# # https://github.com/66RING/ring-attention-pytorch/blob/main/ring_attention.py
# 如果想深入理解这个代码，需要先看懂 basic.py 内部代码
def multi_demo(*args, **kwargs):
    set_random_seed(42)
    q = torch.randn((20, 5))
    k = torch.randn((16, 5))
    v = torch.randn((16, 5))

    world_size = get_world_size()
    rank = get_rank()
    # 将 qkv 在序列维度进行切分，每张卡只处理一部分
    q_bucket_size = q.size(0) // world_size
    k_bucket_size = k.size(0) // world_size

    # 只需要自己的一部分即可
    q_chunk = q[rank * q_bucket_size: (rank + 1) * q_bucket_size, :]
    k_chunk = k[rank * k_bucket_size: (rank + 1) * k_bucket_size, :]
    v_chunk = v[rank * k_bucket_size: (rank + 1) * k_bucket_size, :]

    # 中间变量
    exp_weights = []
    exp_values = []
    weight_maxes = []

    # ring 过程
    buffer_k, buffer_v = prepare_kv_double_buffer(k_chunk, v_chunk)
    for time_step in range(world_size):
        # 1. 计算当前卡的 qkv
        # send buffer
        buf_id1 = time_step % 2
        # recv buffer
        buf_id2 = (time_step - 1) % 2

        # 待发送的数据，自己先算好
        # 下一个时间步时候，就会变成已经接收到的数据
        # 从而实现发送和接收交替
        k_chunk_ = buffer_k[buf_id1]
        v_chunk_ = buffer_v[buf_id1]

        weight = torch.matmul(q_chunk, k_chunk_.T)
        weight_max = weight.amax(dim=-1, keepdim=True)
        weight_maxes.append(weight_max.squeeze(dim=-1))

        weight = weight - weight_max  # safe softmax
        exp_weight = weight.exp()
        exp_weights.append(exp_weight.sum(dim=-1))  # 注意这里

        exp_value = torch.matmul(exp_weight, v_chunk_)
        exp_values.append(exp_value)

        # 2 ring 发送和接收其余 rank 的 kv 值
        # 以 rank0 为例，其 ring 过程如下
        # 在第 0 时间步时候，rank0 把 kv 发送给 rank3，同时接收 rank1 的 kv
        # 在第 1 时间步时候，rank0 把 kv 发送给 rank3(此时的 kv 实际上是来自 rank1的)，同时接收 rank1 的 kv (此时的 kv 实际上是来自 rank2的)
        # 在第 2 时间步时候，rank0 把 kv 发送给 rank3(此时的 kv 实际上是来自 rank2的)，同时接收 rank1 的 kv (此时的 kv 实际上是来自 rank3的)
        # 在第 3 时间步时候，rank0 把 kv 发送给 rank3(此时的 kv 实际上是来自 rank3的)，同时接收 rank1 的 kv (此时的 kv 实际上是来自 rank0的)
        # 到目前为止，rank0 的 q_0 就和所有 KV 都算了一遍了
        send_rank = (rank - 1) % world_size
        recv_rank = (rank + 1) % world_size
        # print(f'{get_rank()}:{time_step}, send: {buf_id1} ->  {send_rank}, recv: {get_rank()} <- {recv_rank}', flush=True)
        step_kv_send_recv(send_k=buffer_k[buf_id1],
                          recv_k=buffer_k[buf_id2],
                          send_v=buffer_v[buf_id1],
                          recv_v=buffer_v[buf_id2])

    # 此时 q_rank_k 已经和所有的 kv 都计算过了，可以开始合并
    weight_maxes = torch.stack(weight_maxes, dim=-1)
    exp_values = torch.stack(exp_values, dim=-1)
    exp_weights = torch.stack(exp_weights, dim=-1)

    global_max = weight_maxes.amax(dim=-1, keepdim=True)
    renorm_factor = (weight_maxes - global_max).exp()

    exp_weights = exp_weights * renorm_factor
    exp_values = exp_values * renorm_factor.unsqueeze(dim=-2)

    all_values = exp_values.sum(dim=-1)
    all_weights = exp_weights.sum(dim=-1)

    out = all_values / (all_weights[:, None] + 1e-8)

    # 聚合所有卡的结，得到最终输出
    all_out = all_gather(out)
    out = torch.cat(all_out, dim=0)

    if rank == 0:
        print(out)  # (20, 5)
        print(out.float().mean())


# https://coconut-mode.com/posts/ring-attention/
if __name__ == '__main__':
    # 单卡，作为对比
    single_demo()

    # 模拟 4 卡实现 ring attention
    functions = [multi_demo]
    world_size = 4
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
