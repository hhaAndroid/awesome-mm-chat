# Vicuna 小羊驼
Vicuna 是 一个开源聊天机器人，号称达到了 ChatGPT 的 90%，也是基于 LLaMA 通过指令微调而来。

Vicuna 开源代码地址：https://github.com/lm-sys/FastChat

Fork 并注释版本： https://github.com/hhaAndroid/FastChat/tree/hha

## 原理和其他

要特别注意对比 Alpaca 的区别。

博客地址：https://lmsys.org/blog/2023-03-30-vicuna/
知乎： https://zhuanlan.zhihu.com/p/618389519

没有论文。

- 训练数据不一样，Alpaca 采用的是 52K 通过 Self-Instruct 生成的数据，而 Vicuna 是用了  70K user-shared ChatGPT conversations 数据集
- 评估方式更加智能，通过构建 prompt 让 ChatGPT4 打分
- 基于 Alpaca 开源代码，改进了代码，更省显存，训练更加高效。在分布式部署方面做的更好

相同点： 都是全量微调，但是好像性能比 Alpaca 强。

训练仅使用 ShareGPT 等公开数据，而不是我们自己调用ChatGPT API 生成数据，基于 ShareGPT (70K user-shared ChatGPT conversations) 维护者的意愿，仅公开模型和训练方法，而不会公开和ShareGPT相关的训练数据，但是开源项目中包括了数据清理的部分

评估是一个比较大的问题，作者是用 ChatGPT4，通过构建 prompt 来给不同模型输出进行打分，但是也是需要 human-in-the-loop，要看下 ChatGPT4 评估的是否合理。

我们让 Vicuna 和其他模型的回答以匿名的方式合并在一起，让 GPT-4 比较它们，给每一个模型 1-10 的评分。然后我们对每一对模型回答的所有问题的所有评分各自求和，得到总分。在这个条件下，我们达到了ChatGPT-3.5 性能的90%。

The cost of training Vicuna-13B is around $300。注意他不是 PEFT 微调而是全量微调。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/02808565-70a9-4f81-8df2-94d9e71e34b9"/>
</div>

首先，我们从ShareGPT.com收集了大约7万个对话，这是一个用户可以分享他们的ChatGPT对话的网站。接下来，我们加强了Alpaca提供的训练脚本，以更好地处理多轮对话和长序列。训练是在一天内用PyTorch FSDP在8个A100 GPU上完成的。为了给演示提供服务，我们实施了一个轻量级的分布式服务系统。我们通过创建一组80个不同的问题并利用GPT-4来判断模型的输出，对模型质量进行了初步评估。为了比较两个不同的模型，我们将每个模型的输出合并为每个问题的单一提示。然后，这些提示被发送到GPT-4，由GPT-4评估哪个模型能提供更好的回答。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/d6cf5fe3-bddf-42a5-a498-8b530ed08934"/>
</div>

训练时候为了确保数据质量，我们将HTML转换回markdown，并过滤掉一些不合适或低质量的样本。此外，我们将冗长的对话分成较小的片段，以符合模型的最大上下文长度。

我们的训练配方建立在斯坦福大学羊驼的基础上，有以下改进。

- 内存优化： 为了使Vicuna能够理解长上下文，我们将最大上下文长度从alpaca的512扩展到**2048**，这大大增加了GPU的内存需求。我们通过利用gradient checkpointing and [flash attention](https://arxiv.org/abs/2205.14135) 来解决内存压力。
- 多轮对话： 我们调整训练损失以考虑到多轮对话，并仅根据聊天机器人的输出计算微调损失。
- Cost Reduction via Spot Instance： 40倍的数据集和4倍的序列长度给训练费用带来了巨大的挑战。我们采用 SkyPilot managed spot (好像是一个管理机) 来降低成本，利用较便宜的实例，自动恢复抢占和自动区域切换。这个解决方案将7B模型的训练成本从500美元降至140美元左右，13B模型的训练成本从1千美元降至300美元左右。


## CentOS 7 + 32G V100 本地部署流程
使用环境： PyTorch 1.9， CUDA 11.1

**(1) FastChat 安装**

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

我们使用的是最新的 `vicuna-7b-delta-v1.1` 权重，所以需要安装 fschat>=0.2.0 和 transformers>=4.28.0

**(2) LLAMA 7b 权重准备**

需要先安装 git-lfs
```shell
wget https://packagecloud.io/github/git-lfs/packages/el/7/git-lfs-2.13.2-1.el7.x86_64.rpm/download # 或者直接下载
sudo rpm -ivh git-lfs-2.13.2-1.el7.x86_64.rpm
```

下载 HF 已经转好的权重

```shell
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```
上述权重大概 26G(7B fp32 模型参数量的存储空间为 4B* 70 亿 = 28 GB 实际下载大概 26 G)

下载后需要修改类名，否则后续会报错：

```shell
cd llama-7b-hf
vim tokenizer_config.json # 将 LLaMATokenizer 修改为 LlamaTokenizer
```

**(3) Vicuna 7b 权重准备**

```shell
python -m fastchat.model.apply_delta \
    --base 你自己的路径/llama-7b-hf \
    --target 你自己的路径/vicuna-7b \
    --delta lmsys/vicuna-7b-delta-v1.1 # 这个 delta 权重大概 14G, 脚本自动下载，你也可以自己下载 https://huggingface.co/lmsys
```
上述程序运行需要大概 30G 内存，如果内存不够，可以使用 `--low_cpu_memory_conversion` 参数，所谓的 Low CPU Memory Conversion 是指的将每个大的模型权重文件切分的更小，每次加载小的权重进行合并

**(4) 修改开始和停止符**

如果不修改，对话过程程序停不下来，原因就是 v1.1 版本采用了新的开始和结束符

```shell
cd 你自己的路径/vicuna-7b
# 将 special_tokens_map.json 换成 https://huggingface.co/lmsys/vicuna-13b-delta-v0/blob/main/special_tokens_map.json 里面的，否则训练和测试停止符不一样，程序会停不下来
```

**(5) 运行**

```shell
# 单 GPU
python -m fastchat.serve.cli --model-path /home/huanghaian/vicuna-7b

# 8 gpu
python -m fastchat.serve.cli --model-path /home/huanghaian/vicuna-7b --num-gpus 8 # 需要 torch 大于等于 1.10

# CPU
python -m fastchat.serve.cli --model-path /home/huanghaian/vicuna-7b --device cpu
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233524677-cf89dee8-3ee8-47f9-9dc5-667aac27ee51.png"/>
</div>

后续会更新 vicuna 的一些关键代码注释，以及一些使用经验。