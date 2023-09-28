# ZeRO 系列 
[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
https://zhuanlan.zhihu.com/p/513571706  
https://huggingface.co/docs/transformers/perf_train_gpu_many  
https://zhuanlan.zhihu.com/p/644133265  

你真的需要 ZeRO 么？

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a8b8ccb4-5a8f-4b54-a5b7-c136058b4027"/>
</div>

上面的图蛮好的，很清晰。

数据并行即最常用的 DP 策略： 假设有 N 张卡，每张卡都保存一个完整的模型副本，每一次迭代（iteration/step）都将 batch 数据分割成 N 个等大小的micro-batch，每张卡根据拿到的micro-batch数据独立计算梯度，然后调用AllReduce计算梯度均值，每张卡再独立进行参数更新。
模型并行 MP 策略，有时候也称为 tensor 并行： 水平分割，有的 tensor/layer 很大，一张卡放不下，将 tensor 分割成多块，一张卡存一块。每个张量被分割成多个块，所以不再将整个张量存储在单个GPU上，而是将每个张量的碎片分别存储在其指定的GPU上。在处理过程中，每个碎片在不同的GPU上独立且并行地进行处理，最终在步骤结束时进行同步。这可以被称为水平并行，因为分割是在水平层面上进行的。
流水线并行 PP: 模型在多个 GPU 上垂直（层级）分割，这样模型的一层或多层只会放在单个 GPU 上。每个 GPU 并行处理管道的不同阶段，并在一个小批次的数据上进行工作。注意： 为了防止”一卡工作，众卡围观“，实践中PP也会把batch数据分割成多个micro-batch(也就是在单个卡内部还会进行切分 batch)，流水线执行
流水线并行可以看 https://zhuanlan.zhihu.com/p/658773834

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/bb65bc8b-da7d-4b75-ac38-8df85c89506b"/>
</div>

三者是可以一起使用的。最常用的 DP 在每个设备上都有一个完整的备份, 我们有必要让每张卡都存一个完整的模型吗？系统中能否只有一个完整模型，每张卡都存 1/N 参数，卡数越多，每张卡的显存占用越少，这样越能训练更大规模的模型。 

ZeRO 的含义是 Zero Redundancy Optimizer，一种显存优化的数据并行(data parallelism, DP)方案，意思是零冗余优化器。其可以优化显存，大大提高了训练速度，同时增加了可以有效训练的模型大小。ZERO 在保留低通信量和高计算粒度的同时消除了数据和模型并行训练中的内存冗余，使我们能够将模型大小与持续高效率的设备数量成比例。

常用的 DP 或者 DDP 不会减少每个设备的内存，因为在每个设备上都有一个完整的备份，包括模型、激活值、优化器状态。而其他现有的解决方案，如管道并行性(PP)、模型并行性(MP) 和 CPU-Offloading 等，在功能、可用性以及内存和计算/通信效率之间进行权衡，但所有这些都对速度和规模的训练至关重要。
大模型如果要训练，最可能采用的就是模型并行，但是其只是简单的垂直分割模型，在多个设备上划分每一层的计算和参数，其需要每层之间的显著通信。

作者首先分析了大模型消耗的内存在哪里：

1）对于大模型，大部分内存被模型状态占据，其中包括优化器状态（例如 Adam 中的动量和方差）、梯度和参数。 
2) 剩余的内存由激活、临时缓冲区和不可用的碎片化内存消耗，我们将其称为剩余状态。

假设模型参数量是 a, 模型参数（fp16）、模型梯度（fp16）和 Adam 状态（fp32 的模型参数备份，fp32 的 momentum 和 fp32 的 variance）。2a+2a+4a+4a+4a=16xa 字节存储，可以看到，Adam 状态占比 75%，因此对于大模型来说，优化器状态是最大的显存消耗。
GPT-2含有1.5B个参数，如果用fp16格式，只需要3GB显存，但是模型状态实际上需要耗费24GB！相比之下，激活值可以用 activation checkpointing 来大大减少，所以模型状态就成了头号显存杀手，它也是ZeRO的重点优化对象。而其中Adam状态又是第一个要被优化的。

**(1) 模型状态优化**

ZERO-DP通过划分模型状态而不是复制它们来消除数据并行过程中的内存状态冗余，并通过在训练过程中使用动态通信调度保留DP的计算粒度和通信量来保持计算/通信效率。前半部分解决了 DP 模型状态全复制的问题，后半部分解决了 MP 不够高效灵活的问题。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/52c66ef5-533a-4b5d-b8d1-e64b9dbc1b24"/>
</div>

水平线可以理解为模型层数，baseline 的话每个 gpu 都是全量模型副本。从上图来看，一共有 3 种切分策略，统称 ZERO-DP，依次叠加，显存越来越少

- ZERO-1: 只把优化器状态进行分片 Optimizer State Partitioning (Pos), 4x memory reduction, 通信量和 DP 一样
- ZERO-2: 对优化器状态 + 梯度进行分片 Gradient Partitioning (Pos+g ): 8x memory reduction, 通信量和 DP 一样
- ZERO-3: 对优化器状态 + 梯度 + 模型参数进行分片 Parameter Partitioning (Pos+g+p): 显存消耗和 DP 卡的格式线性相关，假设是 64 张卡，那么就是减少 64x，通信量增加了 50%。

ZERO-1 情况下对优化器状态进行切片，每张卡只保存部分状态，此时显存消耗就是 4a+ 12a/N，N 是卡数，当卡数变多时候，显存消耗变少,趋于原先的 1/4
如果继续对模型梯度进行分片即为 ZERO-2 情况，此时显存消耗就是 2a+ (2a+12a)/N，N 是卡数，当卡数变多时候，显存消耗变少, 趋于原先的 1/8
如果继续对模型参数进行分片即为 ZERO-3 情况，此时显存消耗就是 2a/N+ (2a+12a)/N，N 是卡数，当卡数变多时候，显存消耗变少, 趋于 0

**(2) 剩余状态优化**

作者进一步提出了 ZERO-R，

1) 对于激活（从前向传递中存储以执行后向传递），我们注意到检查点有助于但不足以用于大模型。因此，ZeRO-R 通过激活值分片操作来识别和删除现有 MP 方法中存在的重复激活值。它还在适当的时候将激活值卸载到 CPU。
2) ZeRO-R 为临时缓冲区定义了适当的大小，以平衡内存和计算效率。
3) 由于不同张量的寿命变化，我们在训练期间观察到碎片化的内存。由于碎片化而导致的连续内存不足会导致内存分配失败，即使有足够的空闲内存可用。ZeRO-R 根据张量的不同生命周期主动管理内存，防止内存碎片。

解决了模型状态，再来看剩余状态，也就是激活值（activation）、临时缓冲区（buffer）以及显存碎片（fragmentation）。

- 激活值同样使用分片方法，并且配合checkpointing
- 模型训练过程中经常会创建一些大小不等的临时缓冲区，比如对梯度进行AllReduce啥的，解决办法就是预先创建一个固定的缓冲区，训练过程中不再动态创建，如果要传输的数据较小，则多组数据bucket后再一次性传输，提高效率
- 显存出现碎片的一大原因是时候gradient checkpointing后，不断地创建和销毁那些不保存的激活值，解决方法是预先分配一块连续的显存，将常驻显存的模型状态和checkpointed activation存在里面，剩余显存用于动态创建和销毁discarded activation

相比于传统的数据并行，ZeRO 是否会带来额外的通信（communication）成本？ 特别是在大规模训练场景下

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/bf7f99a4-a7d4-45c1-b1e0-1295de58cc04"/>
</div>

上图为 dp 情况下梯度同步计算的过程，数据数据并行在每一步（step/iteration）计算梯度后，需要进行一次 AllReduce操作来计算梯度均值，目前常用的是Ring AllReduce，分为 ReduceScatter 和 AllGather 两步。
假设现在有4张卡，在各自进行梯度计算后，每个卡都有一份的不同梯度值，现在需要对所有卡的不同参数的梯度值计算平均。计算过程就是将整个桶里面的参数分成 4 份，然后每个卡 sum 一部分，这样就可以在不同卡上得到部分同步后的值，
最后在进行一次 gather 操作，就可以让每张卡都获取到 sum 后的梯度值。 整个同步过程是 ring 环形运行。 此时每张卡的通信量包括发送和接受是 2a。

如果想清晰的理解整个运行过程，建议一定要看 https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/ 这个视频，超级好，超级清晰。
为了防止链接失效，托管了一份 https://aicarrier.feishu.cn/file/BigPbryCYoJNgvxI2Uuc8VoInIc

ZERO-1 和 ZERO-2 不会增加通信量，以 ZERO-2 为例，目前每张卡只保存了 1/4 份优化器状态和梯度，然后模型参数并没有拆分。

在模型一启动时候，模型参数会分发到不同卡上，对于每张卡只会运行自己的数据，4张卡模型一次性可以运行到底，每张卡上都有不同数据带来的激活值，并且可以得到各自的 loss值和各自的梯度。
现在开始进行反向传播，假设第 3 张卡对应模型最后部分，首先需要从3开始，向其他卡 gather 梯度，得到平均梯度，通信量是 1/N a，然后可以更新这部分的优化器状态了，此处不需要同步。
然后是第二张，不断往前，通信量是 1/N *N =a， 此时每张卡的优化器状态就都得到更新了。 为了在下一次 forward 时候各个卡参数一致，此时需要一个 all gather 操作，让每张卡的参数进行更新，通信量是 a，因此总的通信量其实没变。

对于 ZERO-3，每张卡只有部分参数梯度和优化器状态，首先在每一张卡 forward 前需要进行一次部分同步操作，然后计算 loss 后进行反向传播时候又要同步一次部分参数量(否则无法计算梯度)，同步参数量后得到梯度，然后再进行一次部分参数同步，因此通信量绘 增加 0.5 倍。

ZeRO-DP 和 ZeRO-R 结合在一起形成了一个强大的 DL 训练内存优化系统，我们统称为 ZeRO。

ZERO 系列的原生实现在 `deepspeed` 中。

扩展而来的还有 ZeRO-Offload https://www.usenix.org/system/files/atc21-ren-jie.pdf

ZeRO-Offload 的想法很简单：显存不足，内存来补。将训练阶段的某些模型状态下放（offload）到内存以及CPU计算 (ZeRO-Offload没有涉及剩余状态（比如激活值）的下放，因为在Transformer LM场景中，他比模型状态占用的显存小)。在多卡场景，ZeRO-Offload利用了ZeRO-2

# FSDP

 PyTorch 1.11 及其以上版本支持 FSDP
https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
https://zhuanlan.zhihu.com/p/644133265

模型训练的时候，显存占用大体可以分成三部分，即激活值、模型权重、模型梯度和优化器状态。对于视觉模型而言，显存占比最大的是激活值(因为通道很多)，因此使用混合精度训练能够大幅度的降低激活值的显存占用（fp16）。然而对于大语言模型或者多模态模型而言，优化后三者的显存占用则显得更重要，因此 zero 系列主要是针对这三者的优化。

FSDP 相当于 ZeRO3 的优化，原理和 ZERO3 一样。


https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html   

At high level FSDP works as follow:

1. In constructor

   - Shard model parameters and each rank only keeps its own shard

2. In forward path
   - Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
   - Run forward computation
   - Discard parameter shards it has just collected

3. In backward path
   - Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
   - Run backward computation
   - Run reduce_scatter to sync gradients
   - Discard parameters.

FSDP 核心的参数就是 auto_wrap_policy 用于定制模型的切分策略。被切分的模型会变成一个 FSDP 单元，这个单元

如果没有指定 `auto_wrap_policy`，那么默认只有一个 root fsdp module 将整个模型进行包裹。
在初始化时候会将整个模型的所有参数都进行 flatten，然后根据 `flatten_parameters` 进行均匀切分到多卡上。
一旦运行 forward，就会触发 all gather 将刚刚分发的所有参数进行聚合，此时各个卡上其实就是全量参数了，所以说等价于 ZERO-1。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

如果没有指定，那么打印模型结果为：

```python
   FullyShardedDataParallel(
   (_fsdp_wrapped_module): FlattenParamsWrapper(
       (_fpw_module): Net(
       (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
       (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
       (dropout1): Dropout(p=0.25, inplace=False)
       (dropout2): Dropout(p=0.5, inplace=False)
       (fc1): Linear(in_features=9216, out_features=128, bias=True)
       (fc2): Linear(in_features=128, out_features=10, bias=True)
       )
   )
)
```

如果对某些 layer 指定了 `auto_wrap_policy`，那么就至少有 2 个 fsdp module，一个是 root fsdp module，另一个是被指定的 layer 的 child fsdp module。


```python
my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
torch.cuda.set_device(rank)
model = Net().to(rank)

model = FSDP(model,
    fsdp_auto_wrap_policy=my_auto_wrap_policy)
```      

如果按照 size 来切，则如下所示, 上述策略的含义是： 如果一个 layer 的参数量大于 20000，那么就将其切分到一个单独的 fsdp module 中，否则就将其切分到 root fsdp module 中。

```python
  FullyShardedDataParallel(
(_fsdp_wrapped_module): FlattenParamsWrapper(
  (_fpw_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)) # 参数量 3x3x1x32 = 288
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)) # 参数量 3x3x32x64 = 18432
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): FullyShardedDataParallel(
      (_fsdp_wrapped_module): FlattenParamsWrapper(
        (_fpw_module): Linear(in_features=9216, out_features=128, bias=True) # 参数量 1179776
      )
    )
    (fc2): Linear(in_features=128, out_features=10, bias=True) # 参数量 1280
  )
)
```

一上图为例，此时有 2 个 fsdp module，也就是说实际上参数分成了两组，第一组 FlattenParams 是 fc1，第二组是其他的所有模块构成。

在 forward 时候，假设运行到 conv1 部分，那么会触发 root fsdp 的 forward，进而会 all gather 第二个组的 FlattenParams (注意 fc1 的参数不会考虑进来)，
在运行到 fc1 时候，会抛弃前面的 FlattenParams，只 all gather fc1 的 FlattenParams，这样在显存中就勇用只有其中一份 FlattenParams 全量参数。
此时才是真正的 ZERO-3 过程。

假设某个模型内部有 n 个子 fsdp 模块，那么就是相当于分成了 n+1 个 FlattenParams 组，然后每次 forward 时候都是 all gather 其中一个 FlattenParams 组的参数。

因此 `min_num_params` 参数越小，切分的 fsdp 模块就会越多，粒度就越细，all gather 次数就会越多，训练速度就越慢，显存占用就越小。
如果设置的超级大，就等价于 ZERO-1 了，只有一个 root fsdp 模块，所有参数都在其中，all gather 次数就只有一次，训练速度就会很快，但是显存占用就会很大。

因此官方建议：如果你使用的是  `size_based_auto_wrap_policy`，那么最好在 FSDP 后打印下模型结构，看下 FSDP 切分是否符合你的预期。

一个相对没那么麻烦的办法就是按照 model 名来切分，直接指定想切分的层名，这样整个层就会是一个独立的子 fsdp 模块，整个就是一个 FlattenParams 组，不会切的特别碎。

- FSDP 会对原始模型进行 warp 操作，因此优化器构建必须要在 FSDP 模型包装后，否则不正确。
- 尝试运行包含在 FSDP 实例中的子模块的前向传播是不支持的，并且会导致错误。这是因为子模块的参数将被分片，但它本身不是一个FSDP实例，因此它的前向传播将无法适当地聚集所有参数。这可能在尝试仅运行编码器的编码器-解码器模型时发生，并且编码器没有包装在自己的FSDP实例中。为解决此问题，请将子模块包装在自己的FSDP单元中。
可以看这个例子：

```python
class Layer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = nn.Linear(1, 1)
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x) + self.linear2(x)

class ToyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1, 1)
        self.layer = Layer()  # 会被 auto wrap policy 指定为 child fsdp module

    def forward(self, x):
        y = self.linear(self.layer.processor(x))
        return self.layer(y)
```

假设 Layer 被 wrap 成了 fsdp module，由于 ToyModel.forward 里，直接调用了 self.layer.processor 的 forward，此时由于 layer 的 forward 没有被触发，layer.precessor 里的参数仍然处于分配的状态，也会报错。

又例如这种情况：

```python
class A:
    ...
    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample]) -> dict:
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)
    
class B:
    ...
    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],  **kwargs) -> dict:
        cls_score = self(feats)  # self 是 B 实例，属于 A 模块内部的子模块，没有走 FSDP 的 forward
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses
```

假如 class A 中的 self.head 类型为 class B，且被 wrap 成了 child fsdp module。那么在执行 self.head.loss 的时候，会通过 FSDP 的 __getattr__ 方法直接找到 class B 的 loss，此时的局部变量 self 已经是 class B 实例而并非 FSDP，因此在执行 self(feats) 时不会进入 FSDP 的 forward 触发参数 all gather，进一步引发错误。

FullyShardedDataParallel 有几个核心参数：

- sharding_strategy：切分策略，也就是所谓的 ZERO-1 2 3，默认是 FULL_SHARD 也就是 zero2，SHARD_GRAD_OP 即为 zero-2,还有其他几种策略
- cpu_offload： 是否将参数和梯度放到 cpu 上，默认是 false
- auto_wrap_policy： 模块切分策略



## FSDP + Checkpoint

目前 FSDP 和 checkpoint 一起使用会有问题，问题本质原因和上面给的例子是一样的。以 dino+swin-l backbone 为例，训练会出现如下错误：

```text
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmengine/runner/loops.py", line 96, in run
    self.run_epoch()
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmengine/runner/loops.py", line 112, in run_epoch
    self.run_iter(idx, data_batch)
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmengine/runner/loops.py", line 128, in run_iter
    outputs = self.runner.model.train_step(
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmengine/model/wrappers/fully_sharded_distributed.py", line 278, in train_step
    optim_wrapper.update_params(parsed_loss)
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmengine/optim/optimizer/optimizer_wrapper.py", line 196, in update_params
    self.backward(loss)
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmengine/optim/optimizer/optimizer_wrapper.py", line 220, in backward
    loss.backward(**kwargs)
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/autograd/function.py", line 274, in apply
    return user_fn(self, *args)
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/utils/checkpoint.py", line 141, in backward
    outputs = ctx.run_function(*detached_inputs)
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmdet/models/backbones/swin.py", line 364, in _inner_forward
    x = self.attn(x, hw_shape)
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmdet/models/backbones/swin.py", line 232, in forward
    attn_windows = self.w_msa(query_windows, mask=attn_mask)
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/workspace/huanghaian/code/mm_rtdetr/mmdetection/mmdet/models/backbones/swin.py", line 91, in forward
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/data/mmperc/huanghaian/pt20/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat2 must be a matrix, got 1-D tensor
```

上述错误原因很明显，weight 正常来说是一个 2d tensor，现在是 1d，说明还是 flatten 的状态，说明并没有触发 all gather。

```python
    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x
```

视觉大模型训练占比最大的其实是激活，因此 checkpoint 必然需要，否则不管你啥 FSDP 都没用，因为激活值并没有切分。

解决办法是让 FSDP 和 checkpoint 兼容，目前已经解决。

# DeepSpeed
https://www.deepspeed.ai/getting-started/ 官方文档 
https://github.com/microsoft/DeepSpeed  
https://huggingface.co/docs/transformers/main/main_classes/deepspeed

单卡也可以使用 DeepSpeed，应该是使用 ZeRO-offload，将部分数据 offload 到 CPU，降低对显存的需求

- ZeRO-stage-0: stage 0会禁用所有的分片，然后把DeepSpeed当作时DDP来使用
- ZeRO-Offload 背后的核心技术是在 ZeRO-2 的基础上将优化器状态和梯度卸至 CPU 内存。这个方法让 ZeRO-Offload 能最大程度降低拷贝至 CPU 导致的计算效率损失，同时达到和 ZeRO-2 相同，甚至有时超过的效率
- ZeRO-Infinity: 利用NVMe固态硬盘打破GPU显存墙

https://zhuanlan.zhihu.com/p/630734624 一些参数说明
https://zhuanlan.zhihu.com/p/343570325 官方推文翻译
https://zhuanlan.zhihu.com/p/635358854 一些教程翻译

```shell
pip install deepspeed
```

安装完成后，你可以使用 `ds_report` 或 `python -m deepspeed.env_report` 命令查看 DeepSpeed 环境报告，以验证你的安装并查看你的机器与哪些 ops 兼容

# Megatron-LM

# FastChat

# MMEngine 相关源码记录

