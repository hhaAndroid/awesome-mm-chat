import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的 4 层线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 40)
        self.fc4 = nn.Linear(40, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# 初始化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# 输入数据，batch_size=2
input_data = torch.randn(2, 10)
target = torch.tensor([0, 1])  # 假设的标签

# 将输入数据拆分为 a0 和 a1
a0 = input_data[0].unsqueeze(0).requires_grad_()  # 启用梯度
a1 = input_data[1].unsqueeze(0).requires_grad_()  # 启用梯度

# 缓存每一层的输出
a0_outputs = []
# 要备份 2 个激活，有点不太好，如果一个 pp 里面就一层，那么就不需要 2 个备份了
a1_outputs = []
a1_outputs_detach = []

# a1 完成一次完整的 forward
x1 = a1
x1 = model.fc1(x1)
a1_outputs.append(x1)
x1 = x1.detach().requires_grad_()
a1_outputs_detach.append(x1)
x1 = model.fc2(x1)
a1_outputs.append(x1)
x1 = x1.detach().requires_grad_()
a1_outputs_detach.append(x1)
x1 = model.fc3(x1)
a1_outputs.append(x1)
x1 = x1.detach().requires_grad_()
a1_outputs_detach.append(x1)
x1 = model.fc4(x1)

# 计算 a1 的 loss 并计算 fc4 的梯度
loss1 = criterion(x1, target[1].unsqueeze(0))
loss1.backward(retain_graph=True)

# a0 forward 第一个线性层
x0 = a0
x0 = model.fc1(x0)
a0_outputs.append(x0)  # 缓存输出

# a1 backward fc3 的梯度和输入的梯度
# a.backward(b) 表示计算 a 相关子图的梯度，包括输入和权重等，b 表示前一个子图的梯度输出，用于连接不同子图
a1_outputs[-1].backward(a1_outputs_detach[-1], retain_graph=True)

# a0 forward 第二个线性层
x0 = model.fc2(a0_outputs[-1])
a0_outputs.append(x0)

# a1 backward fc2 的梯度和输入的梯度
a1_outputs[-2].backward(a1_outputs_detach[-2], retain_graph=True)

# a0 forward 第三个线性层
x0 = model.fc3(a0_outputs[-1])
a0_outputs.append(x0)

# a1 backward fc1 的梯度和输入的梯度
a1_outputs[-3].backward(a1_outputs_detach[-3], retain_graph=True)

# a0 forward 第四个线性层
x0 = model.fc4(a0_outputs[-1])
a0_outputs.append(x0)

print("交错式 forward+backward 完成")
