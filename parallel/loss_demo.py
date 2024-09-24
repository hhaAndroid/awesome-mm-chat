import torch
import torch.nn as nn

# https://zhuanlan.zhihu.com/p/721652210
# https://github.com/THUDM/LongAlign/issues/3
# 解决：当 bs > 1 时候现在主流做法都是 view(-1) 导致的问题，顺便解决了 bs=1 且 packing 的问题
# 并没有解决，多轮对话场景下，长输出和短输出存在的问题
# 如果要解决上述问题，还要指定每个样本中是由哪个轮次生成的，然后在 loss 计算时候，根据轮次进行 mask
a1 = torch.randn(5, 9).float()
b1 = torch.Tensor([-100, -100, 1, 2, 3]).long()

a2 = torch.randn(8, 9).float()
b2 = torch.Tensor([-100, -100, 3, 4, 6, -100, -100, 7]).long()

a3 = torch.randn(3, 9).float()
b3 = torch.Tensor([-100, 6, 8]).long()

a4 = torch.randn(4, 9).float()
b4 = torch.Tensor([-100, 7, 8, -100]).long()

a5 = torch.randn(4, 9).float()
b5 = torch.Tensor([-100, -100, 7, 4]).long()

a6 = torch.randn(3, 9).float()
b6 = torch.Tensor([-100, 8, -100]).long()

# Batch 维度拼接
max_item_length = 8
a = torch.zeros(6, max_item_length, 9)
b = torch.ones(6, max_item_length).long() * -100

pack_input_ids = torch.cat([a1, a2, a3, a4, a5, a6], dim=0)
pack_labels = torch.cat([b1, b2, b3, b4, b5, b6], dim=0)
num_tokens = torch.tensor([5, 8, 3, 4, 4, 3, 0])  # 可能有额外的一个 0
_zero_length = torch.zeros(1)
_pad_length = torch.cat([_zero_length, num_tokens]).int()
cumulative_lengths = torch.cumsum(_pad_length, 0).int()

a[0, :5] = a1
b[0, :5] = b1
a[1, :8] = a2
b[1, :8] = b2
a[2, :3] = a3
b[2, :3] = b3
a[3, :4] = a4
b[3, :4] = b4
a[4, :4] = a5
b[4, :4] = b5
a[5, :3] = a6
b[5, :3] = b6

inputs_ids = a
labels = b

# 要偏移
inputs_ids = inputs_ids[:, :-1]
labels = labels[:, 1:]

print('最正确的 loss 计算方式')
loss = nn.CrossEntropyLoss()
output1 = loss(inputs_ids[0], labels[0])
output2 = loss(inputs_ids[1], labels[1])
output3 = loss(inputs_ids[2], labels[2])
output4 = loss(inputs_ids[3], labels[3])
output5 = loss(inputs_ids[4], labels[4])
output6 = loss(inputs_ids[5], labels[5])
print((output1 + output2 + output3 + output4 + output5 + output6) / 6)

print('现在主流的但是不正确的 loss 计算方式')
print(loss(inputs_ids.reshape(-1, 9), labels.reshape(-1)))
print('修正后的计算方式(单卡)')
loss = nn.CrossEntropyLoss(reduction='none')
batch_size = inputs_ids.size(0)
voc = inputs_ids.size(-1)
l = loss(inputs_ids.reshape(-1, voc), labels.reshape(-1))
l = l.reshape(batch_size, -1).sum(dim=1) / (labels != -100).sum(dim=1).float()
print(l.mean())

print('模拟 dp=2 下的计算方式')
input_ids0 = inputs_ids[0:2]
input_ids1 = inputs_ids[2:4]
input_ids2 = inputs_ids[4:6]
labels0 = labels[0:2]
labels1 = labels[2:4]
labels2 = labels[4:6]

batch_size = input_ids0.size(0)
voc = input_ids0.size(-1)
l0 = loss(input_ids0.reshape(-1, voc), labels0.reshape(-1))
l0 = l0.reshape(batch_size, -1).sum(dim=1) / (labels0 != -100).sum(dim=1).float()
l0 = l0.mean()

l1 = loss(input_ids1.reshape(-1, voc), labels1.reshape(-1))
l1 = l1.reshape(batch_size, -1).sum(dim=1) / (labels1 != -100).sum(dim=1).float()
l1 = l1.mean()

l2 = loss(input_ids2.reshape(-1, voc), labels2.reshape(-1))
l2 = l2.reshape(batch_size, -1).sum(dim=1) / (labels2 != -100).sum(dim=1).float()
l2 = l2.mean()

# reduce sum
l = (l1 + l2 + l0) / 3
print(l)

print('梯度累加情况下应该也是正确的')
# 梯度累加时候是先 l1 求mean,然后 backward, 然后 l2 求 mean, backward，等价于下面的代码
print((l1.mean() / 3 + l2.mean() / 3 + l0.mean() / 3))

# 反解 packing 计算
print('反解 packing loss 计算，正确')
# 第一种实现
pack_input_ids1 = pack_input_ids[..., :-1, :]
pack_labels1 = pack_labels[..., 1:]
loss_fc = nn.CrossEntropyLoss()
num_tokens = cumulative_lengths[1:] - cumulative_lengths[:-1]
num_tokens = num_tokens.squeeze(dim=0)
if num_tokens[-1] == 0:
    num_tokens = num_tokens[:-1]
start = 0

loss = pack_input_ids1.new_zeros(1)
for i, seq in enumerate(num_tokens.cpu().tolist()):
    if i == len(num_tokens) - 1:
        seq -= 1
    end = start + seq
    if (pack_labels1[start:end] == -100).all().item():
        # soft pack 每个序列末尾补的pad
        start = end
        continue
    loss += loss_fc(pack_input_ids1[start:end], pack_labels1[start:end])
    start = end

loss = loss / len(num_tokens)
print(loss)

# 更简单实现
# 这个计算过程正确的前提是： 每个样本的 inputs_ids[0] 对应的 label 必须是 -100，这是可以满足的，否则计算结果不正确
pack_input_ids = pack_input_ids[..., :-1, :]
pack_labels = pack_labels[..., 1:]
num_tokens = cumulative_lengths[1:] - cumulative_lengths[:-1]
num_tokens = num_tokens.squeeze(dim=0).cpu().tolist()
if num_tokens[-1] == 0:
    num_tokens = num_tokens[:-1]
num_tokens[-1] -= 1

loss_fc = nn.CrossEntropyLoss(reduction='none')
all_loss = loss_fc(pack_input_ids, pack_labels)
loss_list = all_loss.split(num_tokens)
labels_list = pack_labels.split(num_tokens)
loss_list = [loss.sum() / ((label != -100).sum().float() + 1e-12) for loss, label in zip(loss_list, labels_list)]
loss = torch.stack(loss_list).mean()
print(loss)
