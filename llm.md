# 参考

https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard 


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
  "name": "breakfast_suggestion", # 早餐建议
  "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?", 
  "instances": [
    { 
      "input": "", 
      "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."}
  ], 
  "is_classification": false
}
```
上述格式是作者制作的，现在有不少指令微调的算法也采用了这个数据组织格式。

注意一个有效的指令可以是只有 Instruction 而没有 Input，例如

1. Instruction=写一篇关于学校安全的文章，而没有 Input
2. Instruction=写一篇关于以下主题的文章，Input=学校安全

数据集生成步骤为：
- 1)指令生成
- 2)识别指令是否代表分类任务
- 3)使用输入优先或输出优先的方法生成实例
- 4)过滤低质量数据

(1) 指令生成，无需生成实例

ChatGPT 具有比较强的 few-shot 学习能力即当在上下文中呈现一些现有指令时，大型预训练语言模型可以被提示生成新的和新颖的指令。这为我们提供了一种从一小部分人类编写的种子指令中扩充指令数据的方法。
基于此，作者以自举的方式生成一组不同的指令。具体为： 首先人工构建了 175 个任务(repo 里面可以找到，每个任务一个指令和一个实例)来启动任务池。对于每一步从这个池中抽取8个任务指令作为上下文示例。在8条指令中，6条来自人工编写的任务，2条来自之前步骤中模型生成的任务，以促进多样性

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

实际上在作者的 repo 里面说明了，其实不需要那么麻烦，可以简化掉是否分类，也不要提前生成一些指令那个步骤。

```text
You are asked to come up with a set of 20 diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The instructions should be in English.
6. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
8. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
9. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 20 tasks:
###
1. Instruction: Create a survey that asks about eating and cooking habits.
1. Input:
<noinput>
1. Output:
1. Do you enjoy cooking?
2. Do you like to eat out?
3. How often do you eat out?
4. What is your favorite food?
5. What is your favorite restaurant?
6. What is your favorite thing to cook at home?
7. Do you have any dietary restrictions?
8. Do you have any food allergies?
###
2. Instruction: Explain human's behavior.
2. Input:
Behavior: cry.
2. Output:
There could be many reasons why a person might cry. They could be feeling sad, scared, angry, or frustrated. Sometimes people cry when they are happy or relieved. There is no one answer to why people behave the way they do.
###
3. Instruction:
```

将上述输入给 GPT 即可，让他补全出 20 条，然后经过后处理，把重复度高的删掉不保存。第二次运行时候可以加载这个已经生成的，继续迭代。


迭代上述步骤就可以生成最终的数据了，一共 52k 条，格式如下

```text
    {
        "instruction": "Describe a time when you had to make a difficult decision.",
        "input": "",
        "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client\u2019s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team\u2019s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client\u2019s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
    },
    {
        "instruction": "Identify the odd one out.",
        "input": "Twitter, Instagram, Telegram",
        "output": "Telegram"
    },
```

后续就是用这个数据进行模型全量微调： 在创建大规模指令数据后，我们使用这些数据对原始语言模型(即self - instruction)进行微调。为此，我们将指令和实例输入连接起来作为提示符，并训练模型以标准的监督方式生成实例输出。为了使模型对不同格式具有健壮性，我们使用多个模板将指令和实例输入一起编码。例如，指令可以加前缀“Task:”或不加前缀，输入可以加前缀“input:”或不加前缀，提示符末尾可以加“Output:”，中间可以加不同数量的换行符，等等。

整个数据生成架构图如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/e24d3a57-c6d9-4be7-bf4b-91c10591388f"/>
</div>

alpaca 训练数据其实就是基于上述脚本生成的 52k 指令数据集，有一些 prompt 的修改。根据官网描述

即 finetune 时候每条数据输入格式大概为：

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

模型输出就是 Response 后面内容。上述的内容会全部输入到模型中，然后只有 Response 部分才计算 loss。

训练好后模型推理，直接用的是模板二，用户无需输入 input，更方便。可以发现不支持多轮对话，感觉就是类似一个为了特定任务而做的一个微调模型。

这里有一篇文章有详细介绍： https://zhuanlan.zhihu.com/p/614916562  
如果不想自己看代码，可以看： https://zhuanlan.zhihu.com/p/617343738  

## 模型训练

全量微调。

- fine-tunes LLaMA-7B: 4 A100 80G GPUs in FSDP full_shard bf16 mode python 3.10 用时 3 个小时。

显存还是比较大，作者说未来会采用 lora 微调。当然现在已经出现了 Alpaca-Lora，也出现了专门的中文版 Alpaca。

算是 LlaMA 指令微调的鼻祖了。

## 模型评估

为了评估Alpaca，对来自自建评估集的输入进行了人工评估（由5位学生作者进行）。这个评价集是由自修课作者收集的，涵盖了多样化的面向用户的指令，包括电子邮件写作、社交媒体和生产力工具。我们对text-davinci-003和Alpaca 7B进行了盲目的配对比较，我们发现这两个模型的性能非常相似： 在与text-davinci-003的比较中，Alpaca赢得了90次对89次。

考虑到模型的规模较小，而且指令的后续数据量不大，我们对这个结果感到相当惊讶。除了利用这个静态评估集，我们还对Alpaca模型进行了交互式测试，发现Alpaca在不同的输入上往往表现得与text-davinci-003类似。我们承认，我们的评估在规模和多样性方面可能是有限的。所以我们发布了Alpaca的互动演示，并鼓励读者自己评估Alpaca并给我们反馈。

# alpaca-lora
https://github.com/tloen/alpaca-lora

# Chinese-LLaMA-Alpaca

https://github.com/ymcui/Chinese-LLaMA-Alpaca  
https://github.com/ymcui/Chinese-LLaMA-Alpaca-2 

# Vicuna 小羊驼
Vicuna 是一个开源聊天机器人，号称达到了 ChatGPT 的 90%，也是基于 LLaMA 通过指令微调而来。

Vicuna 开源代码地址：https://github.com/lm-sys/FastChat

Fork 并注释版本： https://github.com/hhaAndroid/FastChat/tree/hha

## 原理和其他

要特别注意对比 Alpaca 的区别。

博客地址：https://lmsys.org/blog/2023-03-30-vicuna/  
知乎： https://zhuanlan.zhihu.com/p/618389519

没有论文。

- 训练数据不一样，Alpaca 采用的是 52K 通过 Self-Instruct 生成的数据，而 Vicuna 是用了  70K user-shared ChatGPT conversations 数据集,数据集应该是更好，且更符合人类喜好
- 评估方式更加智能，通过构建 prompt 让 ChatGPT4 打分，而不是人类来检查
- 基于 Alpaca 开源代码，改进了代码，更省显存，训练更加高效。在分布式部署方面做的更好，代码非常完善

相同点： 都是全量微调，性能比 Alpaca 强。

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

我们的训练方案建立在斯坦福大学羊驼的基础上，有以下改进。

- 内存优化： 为了使Vicuna能够理解长上下文，我们将最大上下文长度从alpaca的512扩展到**2048**，这大大增加了GPU的内存需求。我们通过利用gradient checkpointing and [flash attention](https://arxiv.org/abs/2205.14135) 来解决内存压力。
- 多轮对话： 我们调整训练损失以考虑到多轮对话(看来训练数据有多轮对话)，并仅根据聊天机器人的输出计算微调损失。
- Cost Reduction via Spot Instance： 40倍的数据集和4倍的序列长度给训练费用带来了巨大的挑战。我们采用 SkyPilot managed spot (好像是一个管理机) 来降低成本，利用较便宜的实例，自动恢复抢占和自动区域切换。这个解决方案将7B模型的训练成本从500美元降至140美元左右，13B模型的训练成本从1千美元降至300美元左右。

训练数据格式如下：

```text
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
      },
      {
        "from": "human",
        "value": "What can you do?"
      },
      {
        "from": "gpt",
        "value": "I can chat with you."
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "My name is Vicuna, and I'm a language model developed by Large Model Systems Organization (LMSYS)."
      }
    ]
  },
```

上面的案例转换为最终模型输入和输出字符如下：

```text
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Who are you? ASSISTANT: I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).</s>USER: What can you do? ASSISTANT: I can chat with you.</s>
```

包括 prompt, 输入和输出，其中输出部分才有 loss，多轮对话的话也是对应的输出部分才有 loss。

## 512 如何外推为 2048 ？

这涉及到窗口外推手段。

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

# InternLM

发布了两个开源的预训练模型：InternLM-7B 和 InternLM-20B

# Qwen-7B

https://github.com/QwenLM/Qwen-7B/  
https://github.com/QwenLM/Qwen-7B/blob/main/tech_memo.md

