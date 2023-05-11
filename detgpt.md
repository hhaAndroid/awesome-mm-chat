# DetGPT

<div align=center>
<img src="https://github.com/hhaAndroid/awesome-mm-chat/assets/17425982/928feaf4-d47c-4d81-89c3-257253347adc"/>
</div>

标题： DetGPT: Detect What You Need via Reasoning   
官方地址：https://github.com/OptimalScale/DetGPT  
project 地址： https://detgpt.github.io/   

知乎详细解读版本： https://zhuanlan.zhihu.com/p/628631687


从标题上可以大概看出，DetGPT 是通过输入文本来推理进行特定物体检测。

- 常规的目标检测是在特定类别上训练，然后给定图片将对应类别的所有物体都检测出来
- 开发词汇目标检测是给定特定类别的词汇表和一张图片，检测出包括特定类别词汇的所有物体
- Grounding 目标检测是给定特定类别词汇或者一句话，检测出包括特定类别词汇或者输入句子中蕴含的所有物体

而 DetGPT 做的其实是给定一句话，首先使用 LLM 生成句子中包括的物体，然后将类别词和图片输入到Grounding 目标检测中。可以发现 DetGPT 做的主要是事情就是通过 LLM 生成符合用户要求的类别词。

以上图为例，用户输入： 我想喝冷饮，LLM 会自动进行推理解析输出 冰箱 这个单词，从而可以通过 Grounding 目标检测算法把冰箱检测出来。

其实这个任务有点类似 visual chatgpt 或者 auto gpt，这类智能体可以将用户输出进行自动规划、解析和执行。DetGPT 任务可以自动拆解为两个任务：文本物体解析 + Grounding 目标检测。只不过由于 DetGPT 始终就是两个步骤，因此也
没有必要 auto，直接当做一个两阶段算法即可。或许可以用 langchain 试一下。

一旦 DetGPT 的过程做的很鲁棒，将其嵌入到机器人中，机器人就能够直接理解用户质量，并且执行用户的命令例如前面说的他自己给你打开冰箱拿冷饮了。这样的机器人将会非常有趣。

## 原理简析

<div align=center>
<img src="https://github.com/hhaAndroid/awesome-mm-chat/assets/17425982/12f22c88-75e7-4da8-b28d-3673bc078cb5"/>
</div>

以上是整体结构图，可以发现和 minigpt-4 类似，也是仅仅训练一个线性连接层，连接视觉特征和文本特征，其余模型参数全部估固定。由于图已经很清晰了，就不用再详细说明了。

整个过程可以说是一个 PEFT 过程，核心在于跨模态文本-图片对的构建，然后基于这个数据集进行微调即可。

根据官方描述为：针对文本推理检测任务，模型要能够实现特定格式（task-specific）的输出，而尽可能不损害模型原本的能力。为指导语言模型遵循特定的模式，在理解图像和用户指令的前提下进行推理和生成符合目标检测格式的输出，作者利用 ChatGPT 生成跨模态 instruction data 来微调模型。具体而言，基于 5000 个 coco 图片，他们利用 ChatGPT 创建了 3w 个跨模态图像 - 文本微调数据集。为了提高训练的效率，他们固定住其他模型参数，只学习跨模态线性映射。

实验效果证明，即使只有线性层被微调，语言模型也能够理解细粒度的图像特征，并遵循特定的模式来执行基于推理的图像检测任务、表现出优异的性能。

示例数据集格式为：

```text
        {
            "image_id": "000000102331",
            "task": "What is the person doing on the motorcycle?",
            "answer": "In the image, the person is riding a motorcycle and performing tricks in the air. Therefore the answer is: [motorcycle stunts, performing tricks]"
        },
        {
            "image_id": "000000102331",
            "task": "Find all the objects with wheels in the image.",
            "answer": "In the image, there is a motorcycle present, which has wheels. Therefore the answer is: [motorcycle]"
        },
        {
            "image_id": "000000103723",
            "task": "Find all animals present in the image.",
            "answer": "In the image, there is an elephant present. There is no other animal present in the image. Therefore the answer is: [elephant]"
        },
        {
            "image_id": "000000103723",
            "task": "Find all living beings present in this image.",
            "answer": "In the image, there is an elephant present, as well as a person observing it. Therefore the answer is: [elephant, person]"
        },
```

ChatGPT 本身无法处理图片，应该是利用了插件？ 

具体实现上也是一开始就构建好和微调时候尽量一致的 prompt, system_message 如下：

```text
system_message = "You must strictly answer the question step by step:\n" \
                 "Step-1. describe the given image in detail.\n" \
                 "Step-2. find all the objects related to user input, and concisely explain why these objects meet the requirement.\n" \
                 "Step-3. list out all related objects existing in the image strictly as follows: <Therefore the answer is: [object_names]>.\n" \
                 "If you did not complete all 3 steps as detailed as possible, you will be killed.\n" \
                 "You must finish the answer with complete sentences."
```

在用户输入文本描述，也会额外追加 prompt: Answer me with several sentences. End the answer by listing out target objects to my question strictly as follows: <Therefore the answer is: [object_names]>.
来然 LLM 按照特定格式返回，然后采用正则匹配把物体解析出来，然后送给 Grounding DINO 就完成了。

## 本地部署

由于其过程和 minigpt-4 非常类似，因此就没有再部署了，可以参考 [minigpt4.md](minigpt4.md) 

## 简单验证

DetGPT 核心就是基于自定义指令数据集训练了一个线性层，只要保证能正确输出类别就行。由于之前我部署了 minigpt-4 因此可以探索下如果不做一个线性层微调，并且直接使用 minigpt-4 来验证 minigpt-4 是否也能完成这个事情。下面是一些例子

输入图片如下：
<div align=center>
<img src="https://github.com/OptimalScale/DetGPT/assets/17425982/27cc403b-5135-4474-9c55-1cae9186df16"/>
</div>

输入的中文 prompt:

```text
你必须严格按步骤回答问题：

第1步详细描述给定的图像。
第2步，在给定的图像中找到所有与用户输入有关的物体，并简明地解释为什么这些物体符合要求。
第3步，严格按照以下步骤列出图像中存在的所有相关物体并用中文回复： <因此，答案是：[object_names]>。

如果你没有尽可能详细地完成所有3个步骤，你将被杀死。你必须用完整的句子完成答案。

现在用户输入是： 找到高蛋白的食物
```

输入如下所示：
<div align=center>
<img src="https://github.com/OptimalScale/DetGPT/assets/17425982/b2d9de5f-9d4b-4bcb-b5d8-4e0bbef03817"/>
</div>


输入英文 prompt：

```text
You must strictly answer the question step by step:

Step-1. describe the given image in detail.
Step-2. find all the objects in given image related to user input, and concisely explain why these objects meet the requirement.
Step-3. list out all related objects existing in the image strictly as follows: <Therefore the answer is: [object_names]>.

If you did not complete all 3 steps as detailed as possible, you will be killed. You must finish the answer with complete sentences.

user input: find the foods high in protein
```

结果如下：

<div align=center>
<img src="https://github.com/hhaAndroid/awesome-mm-chat/assets/17425982/05410f26-94b7-46e4-bdbf-b1b0bb578a1b"/>
</div>

显示格式有点小问题。可以看到 minigpt-4 其实已经有很强的指令跟随功能了，每次都能正确的按照我要求的格式输出，但是效果好像是差一点。DetGPT 设置的 beam search 参数是 5，而由于我的机器显存有限，minigpt-4 只能设置为1，否则会 OOM,因此这个参数也有点影响。

总的来说通过特定的数据集 fintune 效果确实还是比 minigpt-4 好一些，但是 minigpt-4 也还行。

## langchain+GPT 3.5 实现
