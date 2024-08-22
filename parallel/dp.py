import torch
from torch.utils.data import DataLoader, Dataset
from torch import distributed as dist
from mmengine.dist import all_reduce
import torch.multiprocessing as mp
import torch.nn as nn
from mmengine.runner import set_random_seed
import torch
import os


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = torch.randint(0, 10, (data.shape[0],))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


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
    model = MyModel()
    dataset = MyDataset(torch.randn(1000, 100))

    criterion = nn.CrossEntropyLoss()

    # 分布式采样
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    # 为了确保 loss 完全一致，在 gpu=2 情况下，bs 要减少一半
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # ddp 训练时候梯度要进行规约，然后再进行统一参数更新
        print(f"loss: {loss.item()} grad mean: {model.fc1.weight.grad.mean()}")
        all_reduce(loss, op='mean')
        all_reduce(model.fc1.weight.grad, op='mean')
        if dist.get_rank() == 0:
            print(f"reduce loss: {loss}, "
                  f"reduce grad mean: {model.fc1.weight.grad.mean()}")
        break


def demo2(device):
    set_random_seed(42)
    model = MyModel()
    dataset = MyDataset(torch.randn(1000, 100))

    criterion = nn.CrossEntropyLoss()

    # 分布式采样
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    # 为了确保 loss 完全一致，在 gpu=2 情况下，bs 要减少一半
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if i == 1:  # 梯度累加
            # ddp 训练时候梯度要进行规约，然后再进行统一参数更新
            # 梯度累加时候梯度要除以2
            all_reduce(model.fc1.weight.grad, op='mean')
            if dist.get_rank() == 0:
                print(f"reduce loss: {loss}, "
                      f"reduce grad mean: {model.fc1.weight.grad.mean()/2}")
            break


if __name__ == '__main__':
    # 单卡测试
    set_random_seed(42)
    model = MyModel()
    dataset = MyDataset(torch.randn(1000, 100))

    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # loss 和梯度进行规约，再进行打印

        print(f"loss: {loss.item()}", model.fc1.weight.grad.mean())
        break

    # 4 卡测试
    functions = [demo1]
    world_size = 4
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)

    # 2 卡测试，并梯度累加为 2
    functions = [demo2]
    world_size = 2
    backend = 'gloo'
    start_method = 'spawn'
    mp.start_processes(init_process,
                       args=(world_size, functions, backend),
                       nprocs=world_size,
                       start_method=start_method)
