# stanford_alpaca

标题： Alpaca: A Strong, Replicable Instruction-Following Model
博客： https://crfm.stanford.edu/2023/03/13/alpaca.html
GitHub: https://github.com/tatsu-lab/stanford_alpaca

没有论文。

核心特性： **仅需 52K data 指令跟随数据集微调 LLaMA 就可以类 ChatGPT 功能。**

而且是完全开源，包括训练数据集，脚本等等，非常好。

## 模型概述

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/cbfe1740-eaca-4d1d-a80d-698fb79b6e8a"/>
</div>

## 训练数据集准备

采用了 [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560) 论文方法和他们生成的 52k 数据。

指令跟随数据集定义：

1. 假设我们有 n 条指令，每条指令对应某一个 task
2. 每一个 task 有一条或者多条输入输出对
3. 训练指令跟随模型就是 M(指令中的某个 task, task 对应的输入)= task 对应的输出

样例为：

```json
{
  "id": "seed_task_0",
  "name": "breakfast_suggestion", 
  "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?", 
  "instances": [
    { 
      "input": "", 
      "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."}
  ], 
  "is_classification": false
}
```

注意一个有效的指令可以是只有 Instruction 而没有 Input，例如

1. Instruction=写一篇关于学校安全的文章，而没有 Input
2. Instruction=写一篇关于以下主题的文章，Input=学校安全

数据集生成步骤为：
- 1)指令生成
- 2)识别指令是否代表分类任务
- 3)使用输入优先或输出优先的方法生成实例
- 4)过滤低质量数据

(1) 指令生成

ChatGPT 具有比较强的 few-shot 学习能力即当在上下文中呈现一些现有指令时，大型预训练语言模型可以被提示生成新的和新颖的指令。这为我们提供了一种从一小部分人类编写的种子指令中扩充指令数据的方法。
基于此，作者以自举的方式生成一组不同的指令。具体为： 首先人工构建了 175 个任务(每个任务一个指令和一个实例)来启动任务池。对于每一步从这个池中抽取8个任务指令作为上下文示例。在8条指令中，6条来自人工编写的任务，2条来自之前步骤中模型生成的任务，以促进多样性

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/16b8f259-3cf0-4c4a-b568-ee9b7c31c451"/>
</div>

(2) 识别指令是否代表分类任务

后续需要判断上一步生成的指令是否是一个分类任务，后续有用。

作者也是借助 fow-shot 来判断，构建12条分类指令和19条非分类指令，然后用 vanilla GPT3 来判断是否为分类指令

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/8c6a532c-cc2c-46c8-b1e8-d7a6f179dcb1"/>
</div>

(3) 使用输入优先或输出优先的方法生成实例

给定指令及其是否为分类任务类型后，我们为每个类别指令独立生成实例。这是具有挑战性的，因为它要求模型根据指令理解目标任务是什么，找出需要哪些额外的输入字段并生成它们，最后通过生成输出来完成任务。我们发现预训练的语言模型可以在很大程度上实现这一点，当提示来自其他任务的上下文示例的指令-输入-输出。一种自然的方法是输入优先方法，在这种方法中，我们可以要求语言模型首先根据指令提出输入字段，然后产生相应的输出。这种生成顺序类似于模型用来响应指令和输入的方式，但这里有来自其他任务的上下文示例。

说了半天，还是因为 chatgpt 太强才可以。

对于非分类任务，则采用 input-first 方式。要模型先输出 input，然后再生成输出，这样才符合直觉。然而，我们发现这种方法可以生成偏向于一个标签的输入，特别是对于分类任务(例如，对于语法错误检测，它通常生成语法输入)。因此，我们额外提出了一种用于分类任务的输出优先方法，其中我们首先生成可能的类标签，然后在每个类标签上设置输入生成条件。提示模板如表9.4所示。我们将输出优先方法应用于前一步确定的分类任务，并将输入优先方法应用于其余的非分类任务。

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/c48d5629-26cc-4f66-a4d7-7684709c8f13"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/7c17345a-9318-4a03-a409-18d7ff26be88"/>
</div>

输入优先的样例如上所示，输出优先的样例如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/d018be8b-a729-406c-a166-fbac1ec5e428"/>
</div>

可以发现为何要对 task 进行分类，其实就是要对分类和非分类任务来构建不同的 prompt 从而生成不同的指令，保证多样性。

(4) 过滤低质量数据

为了鼓励多样性，只有当一条新指令与任何现有指令的 ROUGE-L 重叠小于 0.7 时，它才会被添加到任务池中。我们还排除了包含某些特定关键字(例如，图像、图片、图形)的指令，这些指令通常无法被语言模型处理。当为每条指令生成新实例时，我们过滤掉完全相同或具有相同输入但不同输出的实例。

迭代上述步骤就可以生成最终的数据了。后续就是用这个数据进行模型微调： 在创建大规模指令数据后，我们使用这些数据对原始语言模型(即self - instruction)进行微调。为此，我们将指令和实例输入连接起来作为提示符，并训练模型以标准的监督方式生成实例输出。为了使模型对不同格式具有健壮性，我们使用多个模板将指令和实例输入一起编码。例如，指令可以加前缀“Task:”或不加前缀，输入可以加前缀“input:”或不加前缀，提示符末尾可以加“Output:”，中间可以加不同数量的换行符，等等。

整个架构图如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/e24d3a57-c6d9-4be7-bf4b-91c10591388f"/>
</div>

这里有一篇文章有详细介绍： https://zhuanlan.zhihu.com/p/614916562
如果不想自己看代码，可以看： https://zhuanlan.zhihu.com/p/617343738

alpaca 训练数据其实就是基于上述脚本生成的 52k 指令数据集，有一些 prompt 的修改。 根据官网描述

即 fintune 时候每条数据格式大概为：

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```
或者
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```
This produced an instruction-following dataset with 52K examples obtained at a much lower cost (less than $500).

训练好后模型推理，直接用的是模板二，用户无需输入 input，更方便。

## 模型训练

全量微调。

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/e24d3a57-c6d9-4be7-bf4b-91c10591388f"/>
</div>

- fine-tunes LLaMA-7B: 4 A100 80G GPUs in FSDP full_shard mode python 3.10 用时 3 个小时。

显存还是比较大，作者说未来会采用 lora 微调。当然现在已经出现了 Alpaca-Lora，也出现了专门的中文版 Alpaca。

算是 LlaMA 指令微调的鼻祖了。

## 模型评估

为了评估Alpaca，我们对来自自建评估集的输入进行了人工评估（由5位学生作者进行）。这个评价集是由自修课作者收集的，涵盖了多样化的面向用户的指令，包括电子邮件写作、社交媒体和生产力工具。我们对text-davinci-003和Alpaca 7B进行了盲目的配对比较，我们发现这两个模型的性能非常相似： 在与text-davinci-003的比较中，Alpaca赢得了90次对89次。

考虑到模型的规模较小，而且指令的后续数据量不大，我们对这个结果感到相当惊讶。除了利用这个静态评估集，我们还对Alpaca模型进行了交互式测试，发现Alpaca在不同的输入上往往表现得与text-davinci-003类似。我们承认，我们的评估在规模和多样性方面可能是有限的。所以我们发布了Alpaca的互动演示，并鼓励读者自己评估Alpaca并给我们反馈。
