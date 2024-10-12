import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
import torch.multiprocessing as mp
import os
from torch import distributed as dist
from typing import Optional


class Embedding(nn.Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    沿着 vocab_size 方向进行 split, 这样方便计算，每个索引都可以获取完整的 embedding.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # all-reduce
        if get_world_size() == 1:
            return input_
        torch.distributed.all_reduce(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # 啥也不用做
        return grad_output


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    沿着 vocab_size 方向进行 split, 这样方便计算，每个索引都可以获取完整的 embedding.
    为何最后需要 all-reduce 因为有些rank 获取的 embedding 是 0,但是实际上不应该是0
    all-reduce-sum 可以保证一致
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            max_norm: Optional[float] = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            init_method=nn.init.xavier_normal_,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # 因为在词汇表维度进行切分，假设原先词汇表是 100，现在切分为2份，那么每份的大小是 50
        # 假设输入 id=51,那么应该是第二个 rank 里面查找才是合理的，因此需要特殊处理下，否则很低效
        world_size = get_world_size()
        rank = get_rank()
        assert num_embeddings % world_size == 0, "{} is not divisible by {}".format(num_embeddings, world_size)
        self.num_embeddings_per_partition = num_embeddings // world_size
        # 每个 rank 负责的索引位置
        self.vocab_start_index = rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        # 初始化
        if world_size == 1:
            init_method(self.weight)
        else:
            # 先在 cpu 上初始化一个全量的，然后在切分，可以确保同步
            # 这个步骤应该只是为了和不切分时候状态一致，并不是必要的操作
            master_weight = torch.empty(num_embeddings, embedding_dim, dtype=self.weight.dtype, requires_grad=False)
            init_method(master_weight)
            # 输出维度切分
            weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
            rank = get_rank()  # 切分输入自己的部分
            my_weight_list = weight_list[rank::world_size]

            with torch.no_grad():  # 重新赋值
                torch.cat(my_weight_list, dim=0, out=self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # 减掉起始索引，这样查询才是对的，类似于多卡独立
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0  # 无效索引
        # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0  # 无效索引的 embedding 为 0
        # Reduce across all the model parallel GPUs.
        output = _ReduceFromModelParallelRegion.apply(output_parallel)
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


def demo1(device):
    set_random_seed(42)
    model1 = Embedding(400, 8)  # 词汇表是 400
    set_random_seed(42)
    model2 = VocabParallelEmbedding(400, 8)

    input = torch.randint(10, 400, (3, 4))
    output1 = model1(input)
    output2 = model2(input)
    print(f'\n====rank:{get_rank()} work size: {get_world_size()}====', output1.shape)
    assert torch.allclose(output1, output2, atol=1e-6)

    output1.sum().backward()
    output2.sum().backward()

    # 聚合梯度
    # model2.weight.grad shape 是 (output_size_per_partition, in_features)
    weights_grad = gather_grad(model2.weight.grad.transpose(0, 1)).transpose(0, 1)
    assert torch.allclose(model1.weight.grad, weights_grad, atol=1e-6), 'backward error'


if __name__ == '__main__':
    # 单卡验证 forward 是否一致
    set_random_seed(42)
    model1 = Embedding(400, 8)  # 词汇表是 400
    set_random_seed(42)
    model2 = VocabParallelEmbedding(400, 8)

    input = torch.randint(10, 400, (3, 4))
    output1 = model1(input)
    output2 = model2(input)
    print(output1.shape)
    assert torch.allclose(output1, output2, atol=1e-6), 'forward error'

    # 验证反向传播梯度是否一致
    output1.sum().backward()
    output2.sum().backward()
    assert torch.allclose(model1.weight.grad, model2.weight.grad, atol=1e-6), 'backward error'

    # 多卡验证是否一致
    functions = [demo1]
    world_size = 2
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
