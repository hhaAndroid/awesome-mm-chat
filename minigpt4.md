# MiNiGPT-4

## 环境安装
基础环境： CentOS 7 + 32G V100 

**(1) 安装基础环境**

```shell
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**(2) 安装 git-lfs**

如果你已经安装好了，可以跳过这一步，如果你是其他系统，则应该选择其他安装命令，以下为 centos 7 安装命令：

```shell
wget https://packagecloud.io/github/git-lfs/packages/el/7/git-lfs-2.13.2-1.el7.x86_64.rpm/download # 或者直接下载
sudo rpm -ivh git-lfs-2.13.2-1.el7.x86_64.rpm
```

**(3) 准备 vicuna-13b 权重**

你需要下载 vicuna-13b-delta-v0 和 llama-13b-hf 两个权重，然后合并。

注意： 虽然 vicuna 模型是基于 llama 的全量微调，但是由于 llama 协议限制，作者无法直接发布 vicuna 全量模型，因此作者先手动减掉了 llama 权重，然后发布了 vicuna-13b-delta-v0 权重，因此你需要先下载这个权重，然后再和 llama-13b-hf 权重合并。

```shell
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0 # 大概 49G
git clone https://huggingface.co/decapoda-research/llama-13b-hf  # 大概 75G
```

下载后需要修改类名，否则后续会报错：

```shell
cd llama-13b-hf
vim tokenizer_config.json # 将 LLaMATokenizer 修改为 LlamaTokenizer
```

**(4) 权重合并得到 vicuna-13b**

首先安装 FastChat

```shell
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

然后运行

```shell
python -m fastchat.model.apply_delta \
    --base 你的路径/llama-13b-hf \
    --target 你的路径/vicuna-13b \  # 合并后为 37G
    --delta 你的路径/vicuna-13b-delta-v0 
```

最后修改 https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/configs/models/minigpt4.yaml#L16 为你的 vicuna-13b 路径

**(5) 下载 minigpt-4 预训练的权重**

去 https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link 下载 13b 的权重

然后修改 https://github.com/Vision-CAIR/MiniGPT-4/blob/main/eval_configs/minigpt4_eval.yaml#L11 为你自己的路径

## 环境启动

```shell
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

如果出现如下错误： `NameError: name 'cuda_setup' is not defined`
请使用如下命令修复：  `pip install bitsandbytes==0.35.0`

如果出现如下错误： `ERROR: Your GPU does not support Int8 Matmul!`
请修改 https://github.com/Vision-CAIR/MiniGPT-4/blob/main/eval_configs/minigpt4_eval.yaml#L8 将 low_resource 设置为 False

## 效果展示

<div align=center>
<img src="https://github.com/Vision-CAIR/MiniGPT-4/assets/17425982/4408fe44-4293-497e-a30e-96e4b45b337f"/>
</div>

英文效果不错，直接就知道是来自哪一部漫画。

<div align=center>
<img src="https://github.com/Vision-CAIR/MiniGPT-4/assets/17425982/b00cf85d-e327-45f8-a80d-30b71de7f564"/>
</div>

中文效果差点，在经过提示了可以回答出来。

OCR 任务也还行

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c5c01053-f408-4df3-bd2e-eb1283c1363b"/>
</div>

整体来说 minigpt-4 13b 模型效果还是非常 ok 的。
