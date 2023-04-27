# MMGPT

Train a multi-modal chatbot with visual and language instructions!

开源地址：https://github.com/open-mmlab/Multimodal-GPT

基于开源的多模态模型 OpenFlamingo，使用开放数据集创建了各种视觉指导数据，包括 VQA、图像字幕、视觉推理、文本 OCR 和视觉对话。此外还使用仅语言指导数据训练了 OpenFlamingo 的语言模型组件。 视觉和语言指导的联合训练有效提高了模型的性能！

## 环境安装

```shell
git clone https://github.com/open-mmlab/Multimodal-GPT.git
cd Multimodal-GPT
conda create -n mmgpt python=3.8 -y
conda activate mmgpt
pip install -r requirements.txt
pip install -e . -v
```
环境安装可能需要一定时间，因为库比较多。 下面以 Linux 为例

## 权重准备

最终结构应该是：

```text
Multimodal-GPT/checkpoints
├── llama-7b_hf
│   ├── config.json
│   ├── pytorch_model-00001-of-00002.bin
│   ├── ......
│   └── tokenizer.model
├── OpenFlamingo-9B
│   └──checkpoint.pt
├──mmgpt-lora-v0-release.pt
```

**(1) llama-7b_hf 下载**

需要先安装 git-lfs, 你可以去 https://github.com/git-lfs/git-lfs/releases 下载安装包

```shell
wget https://github.com/git-lfs/git-lfs/releases/download/v2.13.1/git-lfs-linux-amd64-v2.13.1.tar.gz
tar -xzvf git-lfs-linux-amd64-v2.13.1.tar.gz
cd git-lfs-2.13.1
sudo ./install.sh
```

安装好后，可以直接下载权重

```shell
git clone https://huggingface.co/decapoda-research/llama-7b-hf
```
上述权重大概 26G(7B fp32 模型参数量的存储空间为 4B* 70 亿 = 28 GB 实际下载大概 26 G)

下载后需要修改类名，否则后续会报错：

```shell
cd llama-7b-hf
vim tokenizer_config.json # 将 LLaMATokenizer 修改为 LlamaTokenizer
```

上述只是一个典型的别人弄好的 HF FP32 权重，实际上网上还是很多版本的，大家都可以尝试一下。

**(2) OpenFlamingo-9B 下载**

```shell
git lfs install
git clone https://huggingface.co/openflamingo/OpenFlamingo-9B
```

需要输入 huggingface 账号的用户名和密码。如果实在是下载不了，可以直接登录网页点击下载。

**(3) mmgpt-lora-v0-release.pt 下载**

```shell
wget https://download.openmmlab.com/mmgpt/v0/mmgpt-lora-v0-release.pt
```

## 运行 Demo

```shell
python app.py
```
        
第一次运行会自动下载 CLIP 权重，正常启动后会出现浏览器地址，也会生成一个临时的可以公开访问的地址。界面如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234834321-e0a958c9-7552-497b-aae4-2f664f748fb5.png"/>
</div>

这个 LLM 的一大特点是无敌自信，不管你咋说他错了，他都告诉你我没有错(其他 LLM 都是立刻道歉，并给出别的答案)。～～～





