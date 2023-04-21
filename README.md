# awesome-mm-chat
多模态 MM +Chat 合集

## Vicuna 小羊驼
Vicuna 是 一个开源聊天机器人，号称达到了 ChatGPT 的 90%，也是基于 LLaMA 通过指令微调而来。

Vicuna 开源代码地址：https://github.com/lm-sys/FastChat

Fork 并注释版本： https://github.com/hhaAndroid/FastChat/tree/hha

### CentOS 7 + 32G V100 本地部署流程
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