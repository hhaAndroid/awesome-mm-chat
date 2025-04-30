from torch.testing._internal.common_distributed import spawn_threads_and_init_comms
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor


def demo1():
    WORLD_SIZE = 4

    @spawn_threads_and_init_comms
    def shard_big_tensor(world_size):
        mesh = DeviceMesh("cpu", [0, 1, 2, 3])

        global_tensor = torch.randn((8, 10))
        dist.broadcast(global_tensor, 0)

        local_tensor = global_tensor.chunk(4)
        local_tensor_sum = [t.sum() for t in local_tensor]
        if dist.get_rank() == 0:
            print(f'rank: {dist.get_rank()}, local_tensor_sum: {local_tensor_sum}, full tensor {global_tensor.sum()}')

        # 在 mesh 上进行第 0 维度切分
        # 将 global tensor 进行 distribute 化
        dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])
        print(
            f"=====on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}, local tensor: {dtensor.to_local().sum()}, full tensor: {dtensor.full_tensor().sum()}======\n")

    shard_big_tensor(WORLD_SIZE)


def demo2():
    WORLD_SIZE = 4

    @spawn_threads_and_init_comms
    def replicate_big_tensor(world_size):
        mesh = DeviceMesh("cpu", [0, 1, 2, 3])
        global_tensor = torch.randn((8, 10))
        dist.broadcast(global_tensor, 0)
        if dist.get_rank() == 0:
            print(f'rank: {dist.get_rank()}, full tensor {global_tensor.sum()}')

        # 类似广播功能
        dtensor = distribute_tensor(global_tensor, mesh, [Replicate()])
        print(
            f"=====on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}, local tensor: {dtensor.to_local().sum()}, full tensor: {dtensor.full_tensor().sum()}======\n")

    replicate_big_tensor(WORLD_SIZE)


def demo3():
    WORLD_SIZE = 4

    @spawn_threads_and_init_comms
    def partially_mesh(world_size):
        device_mesh1 = DeviceMesh("cpu", torch.arange(world_size).reshape(2, 2), mesh_dim_names=("dp", "sp"))
        dp_mesh = device_mesh1["dp"]
        sp_mesh = device_mesh1["sp"]
        dp_rank = dp_mesh.get_local_rank()
        sp_rank = sp_mesh.get_local_rank()

        print(f" rank: {device_mesh1.get_rank()},dp_rank: {dp_rank}, sp_rank: {sp_rank}, {dp_mesh}, {sp_mesh}")
        #  rank: 0,dp_rank: 0, sp_rank: 0, DeviceMesh([0, 2], mesh_dim_names=('dp',)), DeviceMesh([0, 1], mesh_dim_names=('sp',))
        #  rank: 3,dp_rank: 1, sp_rank: 1, DeviceMesh([1, 3], mesh_dim_names=('dp',)), DeviceMesh([2, 3], mesh_dim_names=('sp',))
        #  rank: 2,dp_rank: 1, sp_rank: 0, DeviceMesh([0, 2], mesh_dim_names=('dp',)), DeviceMesh([2, 3], mesh_dim_names=('sp',)) dp mesh 中 2 位于索引 1，所以 2 就是 dp1
        #  rank: 1,dp_rank: 0, sp_rank: 1, DeviceMesh([1, 3], mesh_dim_names=('dp',)), DeviceMesh([0, 1], mesh_dim_names=('sp',)) dp mesh 中 1 位于索引0，所以 1 就是 dp0

        # 如何简单理解：
        #  device_mesh 是 [[0,1],[2,3]] 矩阵，
        #      sp0   sp1
        # dp0  0     1
        # dp1  2     3
        # 也就是说 dp rank0=device_mesh[0], 也就是 0 和 1 卡都是 dp0。dp rank1=device_mesh[1]

    partially_mesh(WORLD_SIZE)


def demo4():
    WORLD_SIZE = 4

    @spawn_threads_and_init_comms
    def partially_shard_tensor(world_size):
        # if we want to distributed a tensor with both replication and sharding
        # create a 2-d mesh
        device_mesh = DeviceMesh("cpu", torch.arange(world_size).reshape(2, 2))

        global_tensor = torch.randn((8, 10))
        dist.broadcast(global_tensor, 0)

        local_tensor = global_tensor.chunk(2, dim=1)
        local_tensor_sum = [t.sum() for t in local_tensor]
        if dist.get_rank() == 0:
            print(f'rank: {dist.get_rank()}, local_tensor_sum: {local_tensor_sum}, full tensor {global_tensor.sum()}')

        spec = [Replicate(), Shard(1)]
        dtensor = distribute_tensor(global_tensor, device_mesh=device_mesh, placements=spec)
        print(
            f"=====on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}, local tensor: {dtensor.to_local().sum()}, full tensor: {dtensor.full_tensor().sum()}======\n")

        spec = [Shard(1), Replicate()]
        dtensor = distribute_tensor(global_tensor, device_mesh=device_mesh, placements=spec)
        print(
            f"=====11111 on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}, local tensor: {dtensor.to_local().sum()}, full tensor: {dtensor.full_tensor().sum()}======\n")


    partially_shard_tensor(WORLD_SIZE)


def demo5():
    @spawn_threads_and_init_comms
    def dtensor_from_local_to_local(world_size):
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        # create a DistributedTensor that shards on dim 0, from a local torch.Tensor
        # 每个 rank 上假设已经是分片后的数据，from_local 后可以直接合并为一个大的
        local_tensor = torch.randn((8, 4), requires_grad=True)
        rowwise_placement = [Shard(0)]
        # 从 local 视图建立 dtensor
        rowwise_tensor = DTensor.from_local(local_tensor, mesh, rowwise_placement)
        print(
            f"on rank: {dist.get_rank()}, dtensor global shape: {rowwise_tensor.shape}, local shape: {rowwise_tensor.to_local().shape}")

        # 注意和这个代码区别
        global_tensor = local_tensor
        # 从 global 视图建立 dtensor
        dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])
        print(
            f"11111 on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}")

    dtensor_from_local_to_local(4)


def demo6():
    @spawn_threads_and_init_comms
    def dtensor_reshard(world_size):
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        rowwise_placement = [Shard(0)]
        colwise_placement = [Shard(1)]
        # create a rowwise tensor
        local_tensor = torch.randn(8, 8)
        rowwise_tensor = DTensor.from_local(local_tensor, mesh, rowwise_placement)
        # reshard the current row-wise tensor to a colwise tensor or replicate tensor
        replica_placement = [Replicate()]
        # 重新布局
        colwise_tensor = rowwise_tensor.redistribute(mesh, colwise_placement)
        print(
            f"on rank: {dist.get_rank()}, col-wise dtensor global shape: {colwise_tensor.shape}, local shape: {colwise_tensor.to_local().shape}")
        replica_tensor = colwise_tensor.redistribute(mesh, replica_placement)
        print(
            f"on rank: {dist.get_rank()}, replicate dtensor global shape: {replica_tensor.shape}, local shape: {replica_tensor.to_local().shape}")

    dtensor_reshard(4)


def demo7():
    WORLD_SIZE = 4

    @spawn_threads_and_init_comms
    def replicate_big_tensor(world_size):
        mesh = DeviceMesh("cpu", [0, 1, 2, 3])
        big_tensor = torch.randn((888, 10))
        big_tensor_copy = big_tensor.clone()

        # 重复 op，这个实际上类似广播，rank0 广播给别的
        # dtensor = distribute_tensor(big_tensor, mesh, [Replicate()])

        # 这种写法是先把每个 rank 的 big_tensor 进行 concat 变成 (888,40)，
        # 然后 Replicate，此时会触发 all-reduce-sum，确保每张卡上数据完全一样
        # shard 布局变成 Replicate 布局就必然会触发 all-reduce-sum
        dtensor = DTensor.from_local(big_tensor, mesh, [Shard(-1)])
        dtensor = dtensor.redistribute(mesh, [Replicate()])
        print(
            f"on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}, {dtensor.to_local().sum()}, {big_tensor_copy.sum()}")

    replicate_big_tensor(WORLD_SIZE)


# https://colab.research.google.com/drive/12Pl5fvh0eLPUrcVO7s6yY4n2_RZo8pLR#scrollTo=EOKIaL-8JcgB
# https://github.com/pytorch/pytorch/issues/88838
# https://docs.google.com/document/d/1nFeJ8NSFNhNlCkNgWK31ZGRqm1L9rd0i_XN_RprphaI/edit?tab=t.0
if __name__ == '__main__':
    demo5()
