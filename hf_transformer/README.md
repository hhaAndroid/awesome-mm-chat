# NLP 课程核心摘要

官方中文链接： https://huggingface.co/course/zh-CN/chapter0/1?fw=pt
代码库地址： https://github.com/huggingface/transformers

## 0. 课程简介

```shell
pip install transformers[sentencepiece]
```

[sentencepiece] 这种安装方式会额外安装 sentencepiece 库，这是一个用于分词的工具，可以用于中文分词。

在 Pip 安装 Python 包时，可以使用方括号指定额外的依赖项。

## 1 TRANSFORMER 模型
### 1.0 本章简介

包括 5 个主要部分：

- Transformers 模型
- 🤗 Datasets 数据集
- 🤗 Tokenizers 分词库
- 🤗 Accelerate 训练加速器
- Hugging Face Hub

### 1.1 自然语言处理

以下是常见 NLP 任务的列表，每个任务都有一些示例：

- 对整个句子进行分类: 获取评论的情绪，检测电子邮件是否为垃圾邮件，确定句子在语法上是否正确或两个句子在逻辑上是否相关
- 对句子中的每个词进行分类: 识别句子的语法成分（名词、动词、形容词）或命名实体（人、地点、组织）
- 生成文本内容: 用自动生成的文本完成提示，用屏蔽词填充文本中的空白
- 从文本中提取答案: 给定问题和上下文，根据上下文中提供的信息提取问题的答案
- 从输入文本生成新句子: 将文本翻译成另一种语言，总结文本

### 1.2 Transformers能做什么？

Transformers 库中最基本的对象是 `pipeline()` 函数。它将模型与其必要的预处理和后处理步骤连接起来，使我们能够通过直接输入任何文本并获得最终的答案。其实就是一个高度封装的函数，可以直接对预训练好的模型进行推理，用户不用关心实现细节。

```python
def demo_1():
    # pip install transformers[sentencepiece]
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis") # 情感分析任务
    out = classifier("I've been waiting for a HuggingFace course my whole life.")
    print(out)
```

第一次运行会出现：

```text
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
Downloading (…)lve/main/config.json: 100%|██████████| 629/629 [00:00<00:00, 167kB/s]
Downloading pytorch_model.bin: 100%|██████████| 268M/268M [03:50<00:00, 1.16MB/s]
Downloading (…)okenizer_config.json: 100%|██████████| 48.0/48.0 [00:00<00:00, 35.1kB/s]
Downloading (…)solve/main/vocab.txt: 100%|██████████| 232k/232k [00:00<00:00, 957kB/s]
```
可以发现他会自动下载一些文件，这些文件默认是放在 `~/.cache/huggingface/hub/models--distilbert-base-uncased-finetuned-sst-2-english/snapshots/3d65bad49c7ba6f71920504507a8927f4b9db6c0/`

从上面可以看出，这个模型是基于 BERT 模型的，这个模型是在 SST-2 英文数据集上训练的，所以它的输入是英文句子，输出是情感分类。模型细节我们可以去 https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english 查看

- config.json 是模型的配置文件
- pytorch_model.bin 是模型的权重文件
- tokenizer_config.json 是分词器的配置文件
- vocab.txt 是分词器的词表文件

可以发现 HF 保存的权重文件格式是 bin，而不是常用的 pth。因为它们使用的是自己的序列化格式，不同于 PyTorch 默认的 ".pth" 格式。Hugging Face 的 ".bin" 格式是基于 Protocol Buffers（一种跨平台、跨语言的序列化框架）和 Apache Arrow（一种内存映射文件格式）实现的，可以更高效地存储和加载模型权重。
与 PyTorch 的 ".pth" 格式相比，Hugging Face 的 ".bin" 格式具有以下优点：

- 性能更高：Hugging Face 的序列化格式可以更快地读取和写入数据，而且可以在不同的平台上进行无缝转换。
- 存储空间更小：Hugging Face 的序列化格式可以更紧凑地表示数据，因此可以节省存储空间。
- 更灵活：Hugging Face 的序列化格式可以存储任意数据类型，包括模型权重、优化器状态、词汇表等等。

虽然 ".bin" 格式不是 PyTorch 默认的格式，但是 Hugging Face 提供了一些工具，可以将 ".bin" 格式的权重文件转换为 PyTorch 的 ".pth" 格式，以便与 PyTorch 中的代码集成。

上述代码做的事情包括：

- 文本被预处理为模型可以理解的格式，其实就是分词
- 预处理的输入被传递给模型。
- 模型处理后输出最终人类可以理解的结果。

目前支持了很多 pipeline，可以去 https://huggingface.co/docs/transformers/main_classes/pipelines 查看。你也可以指定特定的模型，而非默认运行的那个。

```python
# 使用默认的模型
pipe = pipeline("text-classification")
pipe("This restaurant is awesome")

# 同一个任务，但是可以指定不同的模型
pipe = pipeline("text-classification", model="roberta-large-mnli")
pipe("This restaurant is awesome")

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
```

现在也支持 CV 任务了

```python
from transformers import pipeline

depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")

# This is a tensor with the values being the depth expressed in meters for each pixel
output["predicted_depth"].shape
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233542688-0eaa8c73-3818-4a56-955e-383ec2447f95.png"/>
</div>

### 1.3 Transformers 是如何工作的？

Transformer 架构 于 2017 年 6 月推出。原本研究的重点是翻译任务。随后推出了几个有影响力的模型，包括

- 2018 年 6 月: GPT, 第一个预训练的 Transformer 模型，用于各种 NLP 任务并获得极好的结果
- 2018 年 10 月: BERT, 另一个大型预训练模型，该模型旨在生成更好的句子摘要（下一章将详细介绍！）
- 2019 年 2 月: GPT-2, GPT 的改进（并且更大）版本，由于道德问题没有立即公开发布
- 2019 年 10 月: DistilBERT, BERT 的提炼版本，速度提高 60%，内存减轻 40%，但仍保留 BERT 97% 的性能
- 2019 年 10 月: BART 和 T5, 两个使用与原始 Transformer 模型相同架构的大型预训练模型（第一个这样做）
- 2020 年 5 月： GPT-3, GPT-2 的更大版本，无需微调即可在各种任务上表现良好（称为零样本学习）

这个列表并不全面，只是为了突出一些不同类型的 Transformer 模型。大体上，它们可以分为三类：

- GPT-like (也被称作自回归Transformer模型)，预训练任务为阅读 n 个单词的句子，预测下一个单词
- BERT-like (也被称作自动编码Transformer模型)，预训练任务为遮罩语言建模，该模型预测句子中的遮住的词
- BART/T5-like (也被称作序列到序列的 Transformer 模型)

- Encoder-only models (BERT): 适用于需要理解输入的任务，如句子分类和命名实体识别。
- Decoder-only models (GPT): 适用于生成任务，如文本生成。
- Encoder-decoder models 或者 sequence-to-sequence models: 适用于需要根据输入进行生成的任务，如翻译或摘要。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233545914-38816c95-d26a-4bb5-9046-c17ddacfaa27.png"/>
</div>

“编码器”模型指仅使用编码器的Transformer模型。在每个阶段，注意力层都可以获取初始句子中的所有单词。这些模型通常具有“双向”注意力，被称为自编码模型。

这些模型的预训练通常围绕着以某种方式破坏给定的句子（例如：通过随机遮盖其中的单词），并让模型寻找或重建给定的句子。

“编码器”模型最适合于需要理解完整句子的任务，例如：句子分类、命名实体识别（以及更普遍的单词分类）和阅读理解后回答问题。

该系列模型的典型代表有：
- ALBERT
- BERT
- DistilBERT
- ELECTRA
- RoBERTa

“解码器”模型通常指仅使用解码器的Transformer模型。在每个阶段，对于给定的单词，注意力层只能获取到句子中位于将要预测单词前面的单词。这些模型通常被称为自回归模型。

“解码器”模型的预训练通常围绕预测句子中的下一个单词进行。

这些模型最适合于涉及文本生成的任务。

该系列模型的典型代表有：

- CTRL
- GPT
- GPT-2
- Transformer XL

编码器-解码器模型（也称为序列到序列模型)同时使用Transformer架构的编码器和解码器两个部分。在每个阶段，编码器的注意力层可以访问初始句子中的所有单词，而解码器的注意力层只能访问位于输入中将要预测单词前面的单词。

这些模型的预训练可以使用训练编码器或解码器模型的方式来完成，但通常涉及更复杂的内容。例如，T5通过将文本的随机跨度（可以包含多个单词）替换为单个特殊单词来进行预训练，然后目标是预测该掩码单词替换的文本。

序列到序列模型最适合于围绕根据给定输入生成新句子的任务，如摘要、翻译或生成性问答。

该系列模型的典型代表有：

- BART
- mBART
- Marian
- T5

下一部分 [README_2](README_2.md)
