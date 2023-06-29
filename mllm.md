# OFA

论文： [OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)

在这项工作中，我们追求一种统一的多模态预训练范式，以打破复杂的任务/模态特定定制的支架。我们提出了OFA，一个支持任务全面性的任务无关性和模态无关性框架。OFA在一个简单的seq2seq的学习框架中统一了不同的跨模态和单模态的任务，包括图像生成、视觉定位、图像说明、图像分类、语言模型等。OFA在预训练和微调阶段都遵循基于指令的学习，不需要为下游任务提供额外的特定任务层。与最近的最先进的视觉和语言模型相比，OFA只在2000万个公开的图像-文本对上进行了预训练，这些模型依赖于极其庞大的跨模态数据集。尽管其简单性和相对较小的训练数据，OFA在一系列跨模态任务中取得了新的SOTA，同时在单模态任务中获得了极具竞争力的表现

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/b8958894-44ba-4dbe-8142-e7d30c41025a"/>
</div>

OFA 支持的任务如上所示。

模型架构如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/ff372f80-a704-48e6-aeee-4971e005db05"/>
</div>

可以看到架构是 a unified Seq2Seq framework。

图像 embedding 选择 ResNet, 文本分词选择 BPE。

如何统一各种模态的输入和输出？ 输入比较好统一，主要关注模态输出。一个可能的解决方案是将文本、图像和object离散化，并在一个统一的单词表中用 token 表示它们。

在文生图领域，已经有图片量化策略来实现这个功能了，可以借助这个策略实现图像 token 化。例如，256 × 256 分辨率的图像表示为长度为 16 × 16 的代码序列。每个离散代码与相应的补丁密切相关

bbox 模态表示就比较容易了，表示为一连串的离散标注，每个标注对应一个 bbox，bbox 整数坐标本质上是单词，因此可以用BPE标注表示。

最后统一词表是文本的subwords，图片的image code和物体的location tokens三者的并集。

我们选择Transformer作为主干架构，并采用编码器-解码器框架作为所有预训练、微调和zero-shot任务的统一架构。具体来说，编码器和解码器都是Transformer层的堆叠。

因为他是一个 seq2seq 任务，以 object 检测为例，他可能并不是总输出有效的 token，因此在推理时候要处理下。在分类任务中存在几个问题： 1.对整个词汇表进行优化是不必要的，而且是效率低下的；2.在推理过程中，该模型可能会从封闭的标签集中产生无效的标签。为了克服这些问题，引入一种基于前缀树的搜索策略。

预训练数据如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/bd96c0be-2e4d-401d-b4dc-830fd8cc0248"/>
</div>

下游任务微调也是采用和预训练一样的模式即采用指令微调学习。

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/be6c2a55-462e-4eca-b337-b8cef5a8dae7"/>
</div>

# VisionLLM

论文： [VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks](https://arxiv.org/abs/2305.11175)
暂时还没有开源

和 OFA 非常类似，但是模型做的更大，性能更高，同时接入了羊驼 LLM, 还可以对话啥的。没有实现通用图像分割。下面详细描述。

VisionLLM 的核心意图是一个模型可以做多种视觉和语言任务，架构统一，输入和输出也统一，不需要对不同的下游任务构建不同的输入或者增加模型等。

如果你了解 OFA，那么这个模型就比较好理解了。

结构图如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/80a54c4b-1acf-40eb-a0df-6fb1177a89c8"/>
</div>

包括 3 个模块：

- 一个统一的语言指令，为以视觉为中心的任务定义和定制提供一致的输入。简单来说就是类似 OFA 通过语言构建统一的不同任务的输入，例如图片理解任务， Describe the image <image> in details.  <image> 是 image embedding 的占位符
- 一个语言导向的图像 tokenizer，它对视觉信息进行编码，与给定的语言提示保持一致，使模型能够有效地理解和解析视觉内容； 其实就是 image embedding 转 lang feature
- 基于 LLM 的开放式任务解码器，它利用编码的视觉信息和语言指令来产生令人满意的预测或输出。 解码器是核心，也是和 OFA 不一样的主要地方。

这三种设计共同实现了一个灵活和开放的框架，可以处理各种以视觉为中心的任务，并通过语言指令进行不同程度的任务定制。通过语言指令进行任务定制。

注意：基于 LLM 的开放式任务解码器采用的是类似 DETR Query 的模式，并不是 seq2seq 输出是并行的，因此相比 OFA 效率更好。输入 Query 就是上图中的 Language Instructions <text>。下面详细说明

## 统一的语言指令
参考之前论文做法，

Vision-Language Tasks

- image captioning： The image is <image>. Please generate a caption for the image: 
- VQA：The image is <image>. Please generate an answer for the image according to the question: <question>

Vision-Only Tasks

典型的如目标检测、实例分割和姿态估计等，考虑到用户描述的不同，作者采用了 self-instruct 方法基于一些样例生成了大量的对应任务描述，训练时候随机选择一个构成训练样本。推理时候一个实例分割语义描述的例子是：

Segment all the objects of category set <class> within the <range> of the image and generate a list of the format
(c, x1, y1, x2, y2, ..., x8, y8). Here, c represents the index of the class label starting from 0, and (x1, y1, x2, y2, ..., x8, y8) correspond to the offsets of boundary points of the object relative to the center  point. The image is: <image>

range 设置为 512。

## 语言导向的图像 tokenizer

VisionLLM 认为相对于语言信息，图像是一种外语，需要将其转换为能被 llm 理解的 token。具体做法是：

1. 先用 image  backbones 对图片进行特征提取，得到多尺度图片特征
2. 使用 text encoder 提取文本特征
3. 通过交叉注意力将语言特征注入到每个尺度的视觉特征中，产生多尺度的语言感知视觉特征，使特征在不同的模式中保持一致。跨模态的特征
4. 将融合后的多尺度特征输入到 Deformable DETR Encoder 中，这个做法和 Mask2Former 是类似的

上述步骤就可以得到序列长度为 M 的图片 token

## 基于 LLM 的开放式任务解码器

Decoder 是采用 LLM 结构，实际上就是 Alpaca。但是对于以视觉为中心的任务，Alpaca 有一些固有的缺点。

(1) 它的词汇表中只有几个数字标记（如0∼9），这限制了它通过数字定位物体的能力；
(2) 它使用多个标记来表示类别名称和类名，这导致了一种低效的方案。
(3) 它是一个因果模型，对于视觉感知任务来说是低效的。在视觉感知任务中效率低下

作者的做法是扩展词汇表，增加了专门为视觉中心任务设计的标记，为以视觉为中心的任务而设计的额外标记。

- 增加了一组位置标记，表示为 {<p-512>, ..., <p0>..., <p512>}、 ..., <p512>}，其中 <p i> 表示离散偏移量。
- 我们引入了语义无关的分类代号{<c0>, <c1>, ..., <c511>}来代替类别名称代号，这就克服了使用多个代号来表示的低效率问题。例如{"人"：<c0>，"车"：<c1>，"黑猫"：<c2>，...}

为了出来语言模型串行输出低效问题，改成了采用 query 的并行预测。query 并不是随机初始化的，而是和具体任务有关系，如图所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/0b161823-db61-420a-aca1-b61b7035eca9"/>
</div>

具体实现过程可能要等开源后。训练的 loss 比较简单，因为就是一个分类任务，采用 cross entropy loss。

## 训练细节

训练数据包括 object detection, instance segmentation, visual grounding, image captioning, and visual question answering 方向，具体是：

- COCO2017
- RefCOCO
- RefCOCO+
- RefCOCOg
- COCO Caption
- LLaVA-Instruct-150K

用两个图像骨干实现了VisionLLM的两个变体，即ResNet和InternImage-H。对于语言引导的图像标记器，我们采用 BERTBase作为文本编码器和 Deformable DETR（D-DETR）来捕获高级信息。
我们将查询次数 M 设定为 100，D-DETR的编码/解码器层数为6。对于LLM，我们采用了Alpaca-7B[，并配备了LoRA进行参数有效的微调。

该模型的训练分为两个阶段。在第一阶段，我们用预先训练好的D-DETR、BERT和Alpaca-7B的权重初始化模型，并训练视觉骨干和语言引导的图像标记器，同时冻结LLM的大部分参数，只有少数LoRA参数除外。为了简化训练的复杂性，在这个阶段，我们主要关注具有随机物体类别和任务描述的物体检测任务。在第二阶段，我们冻结了视觉主干，并引入了多个任务的统一监督。引入对多个任务的统一监督。除非另有说明，训练运行的时间为在4×8的NVIDIA A100 GPU上运行50个epochs。

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/98277983-873d-45e7-bc55-898926538f2e"/>
</div>

一些例子：

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/a5d5f11b-7d2a-42d7-ad40-2cc79ee51d29"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/17425982/418dc07e-9401-4b0f-8d3f-3dab5a73f58d"/>
</div>

总的来说是一个不错的工作，期待后续开源。

# PaLI-X

[PaLI-X: On Scaling up a Multilingual Vision and Language Model](https://arxiv.org/abs/2305.18565)

很强。

摘要： 本文介绍了 PaLI-X，一种多语言视觉和语言模型的训练方法和结果，涉及模型尺度大小和训练任务混合。我们的模型在各种各样的任务中取得了新的性能水平，包括多个基于图像的图像理解和问答任务、基于图像的文档理解和少样本（上下文内）学习，以及目标检测、视频问答和视频字幕等任务。PaLI-X 在大多数视觉和语言基准测试中都取得了最新的研究成果（超过25个基准测试）。最后，我们观察到新兴的能力，如复杂计数和多语言目标检测，这些任务并未明确纳入训练组合中。

consisting of a pretrained large-capacity visual encoder (using Scaling vision transformers to 22 billion parameters as the starting point) and a pretrained language-only encoder-decoder (using UL2: Unifying language learning paradigms as the
starting point), further trained at-scale on a vision-and-language data mixture using a combination of self-supervision and full-supervision signals

需要先了解 [PaLI: A jointly-scaled multilingual language-image model](https://ai.googleblog.com/2022/09/pali-scaling-language-image-learning-in.html)

我们观察到，scaling 可以大大改善 PaLI 模型的结果，而且比专门针对某些任务进行训练的专用大模型的结果要好，这些模型通常是通过（往往更大的）仅文本 LLM 的帮助来解决问题的。也就是说多模训练更好，而且模型越大效益越大。也有类似的涌现效果。

总的贡献如下：

1. 我们扩展了一个视觉-语言模型，以在各种基准测试中实现出色的性能。我们观察到，扩展视觉和语言组件都是有利的，并且报告称，在这个规模下性能仍未饱和。
2. 我们展示了用一种混合目标的训练方法来训练这样的模型，该方法结合了 prefix-completion and masked-token completion，可以在这个规模下提高微调和少样本性能的 Pareto 前沿。
3. 我们还展示了高容量视觉编码器（ViT-22B）可以有效地进行共同训练，用于图像分类和OCR标签分类，以在需要理解图像文本的视觉-语言任务上实现显着的改进。
4. 总的来说，PaLI-X 通过对15个以上基准测试进行微调来改进SoTA结果，并且我们展示了它是第一个能够同时适应多个基准测试的模型，而且没有明显的性能下降。

采用了非常大的 ViT-22B 作为视觉编码器，以及 UL2 作为语言编码器，输出也是文本。目标检测建模为 pix2seq 任务，使用 图片的 OCR 文字作为预训练数据集，然后在各个下游任务上微调，同时也支持 few-shot 推理。

# ChatSpot

# Kosmos-2

Kosmos-2: Grounding Multimodal Large Language Models to the World

https://arxiv.org/abs/2306.14824
https://github.com/microsoft/unilm/tree/master/kosmos-2

非常好！

一个多模态大语言模型（MLLM），使多模态大语言模型（MLLM）实现了感知物体描述（如边界框）的新能力，并将文本与视觉世界联系起来。MLLM 支持 bbox+text 输出，也支持 bbox+text 输出。

提出了一个新的大规模数据集用于训练，非常有价值。

我们介绍了KOSMOS-2，一个多模态大语言模型（MLLM），能够实现多模态大语言模型（MLLM），实现了感知物体描述（如边界框）和将文本与视觉世界相结合的新能力。将文本与视觉世界联系起来。

具体来说，我们将参考表达式作为Markdown中的链接，即"[文本跨度](边界框)"，其中对象描述是位置标记序列。描述是位置标记的序列。与多模态语料库一起、 我们构建了大规模的图像-文本对数据（称为GRIT）来训练 模型。

除了MLLMs的现有能力（例如 perceiving general modalities, following instructions, and performing in-context learning）、 KOSMOS-2将grounding能力整合到下游的应用中。我们对KOSMOS-2进行了广泛的任务评估，包括（i）multimodal grounding如指代表达的理解和短语接地，(ii) multimodal referring，如指称表达的生成，(iii)感知-语言任务、 以及(iv)语言理解和生成。这项工作奠定了Embodiment AI的发展基础，并揭示了语言、多模态感知、行动和世界建模的大融合，这是向人工智能迈出的关键一步。

总的结构图如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/f2ca8ce3-2293-4802-88a4-4590c1652b89"/>
</div>

LLM 接入 grounding 能力的必要性：It enables the user to point to the object or region in the image directly rather than input detailed text descriptions to refer to it, the model can understand that image region with its spatial locations。Grounding capability also enables the model to respond with visual answers (i.e., bounding boxes), which can support more vision-language tasks such as referring expression comprehension

为了释放grounding能力，我们构建了一个网络规模的grounding图像-文本对数据集，并将其与KOSMOS-1中的多模态语料库相结合来训练该模型。这些grounding的图像-文本对建立在LAION-2B和COYO-700M的图像-文本对的子集上。我们构建了一个管道，将标题中的文本span（即名词短语和指代表达）与图像中相应对象或区域的空间位置（例如，边界框）进行提取和链接。我们将box的空间坐标转换为一串位置标记，然后将其附加在各自的文本span之后。该数据格式作为一个 "超链接"，将图像中的物体或区域与标题连接起来。

## 数据集生成过程

Step-1: Generating noun-chunk-bounding-box pairs
基于标注的图像描述文字，首先提取名词短语，然后将名词输入到 GLIP 中生成 bbox 即可

Step-2: Producing referring-expression-bounding-box pairs
扩展上面解析出来的名词短语，生成指代表达式，生成过程有点讲究，和利用到解析出来所有词之间的关系，结合前面的 bbox 就可以构成一个标注了。

我们使用spaCy来获得句子的依赖关系。然后通过递归地遍历依赖树中的子代，并将子代标记与名词块连接起来，将名词块扩展为一个指代表达。我们不扩展带有连接词的名词块。对于没有子代标记的名词块，我们为下一个过程保留它们。

图示如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/020b87fe-e1e7-4113-84eb-cd810a324e3e"/>
</div>

最后，我们得到了大约 9100 万张图片，1.15 亿个文本跨度，以及 1.37 亿个相关的边界框。

## 模型和训练

KOSMOS-2 采用了与 KOSMOS-1 相同的模型架构和训练目标。我们将 grounded 图像-文本对添加到训练数据中，使模型具有接地和referring能力。

模型的输入形式，默认训练方式本质上还是 预测下一个词，所以只要有这个句子就可以采用 KOSMOS-1 训练了。

```text
<s> <image> Image Embedding </image> <grounding> <p> It </p><box><loc44><loc863></box> seats next to <p> a campfire </p><box><loc4><loc1007></box> </s>
```

<s> 是开启标志， </s> 是结束标志，<p> and </p> are special tokens indicating the beginning and end of the text span.

其余部分和 KOSMOS 一样，无非是词表里面加入了新的 token，然后训练即可。

在新添加的基础图像-文本对、单模态文本、图像-标题对以及交错的图像-文本数据进行训练。训练过程涉及到419K标记的批量大小，包括来自文本语料库的185K标记，来自原始和基础图像-标题对的215K标记，以及来自交错数据的19K标记。我们对KOSMOS-2进行了60K步的训练，相当于大约250亿个标记。采用AdamW优化器。设置权重衰减为0.01，辍学率为0.1。在最初的375个热身步骤中，学习率增加到2e-4，并线性衰减到零。

为了告诉模型何时将 grounding 文本预测输出，我们在训练过程中，将"<grounding>"标记预置到训练期间。也就是说一旦在模型推理时候，我插入了 <grounding> 标记则表示后面的内容需要进行 grounding 预测即输出的文本中带有实体名词的需要输出同时输出 bbox。

我们在256个V100 GPU上训练模型，训练大约需要一天的时间来完成。为了告诉模型何时将文本输出与视觉世界接轨，我们在训练期间将‘<grounding>’标记预置到训练期间。

第一步预训练：
The total number of trainable parameters amounts to approximately 1.6B. The image resolution is set to 224×224 and the patch size is 14×14. We divide the width and height
of the image into 32 bins, with each bin consisting of 7×7 pixels. A total of 32×32 location tokens are added to the vocabulary. KOSMOS-2 uses the weights of KOSMOS-1 for initialization, the newly
added word embeddings of location tokens are initialized randomly. We update all the parameters during training and instruction tuning

第二步指令微调：

在模型训练完成后，我们进行指令调整，以更好地使 KOSMOS-2 与人类指令相一致。我们将视觉语言指令数据集和纯语言指令数据集与训练数据结合起来，对模型进行调整。此外，我们通过利用GRIT中的边界框和表达式（即名词短语和指代表达式）对来构建接地的指令数据。给定一个表达式-边界框对，我们使用"<p>表达式</p>"作为输入指令，并提示模型生成边界框的相应位置标记。我们还使用"<p>它</p><box><loc1><loc2</box>是 "这样的提示，要求模型根据其边界盒生成表达式。

一些指令模板如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/9919b447-c376-4f06-9c49-d5133c876f2a"/>
</div>

## 评估

• Multimodal grounding  
- Phrase grounding  
- Referring expression comprehension  
  
• Multimodal referring   
- Referring expression generation  

• Perception-language tasks  
- Image captioning  
- Visual question answering  

• Language tasks  
- Language understanding  
- Language generation  

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/2f784491-9af6-46cd-9529-2c8f7c74bddf"/>
</div>

Phrase grounding 任务要求模型基于一个或多个可能在单个标题中相互关联的短语来预测一组边界框。指代表达理解任务鼓励模型在给定图像中定位文本指代表达式中描述的对象。

在 Flickr30k Entities 上评估 Phrase grounding。在  RefCOCO , RefCOCO+  and RefCOCOg 上面进行评估 Referring expression comprehension，发现结果稍差点，这种差异可以归因于 RefCOCO 和 RefCOCO+ 中存在的数据分布，它们倾向于在双人游戏中使用更短的指代表达式（如“左下角”）。因此，我们未来的目标之一是增强多语言语言模型的能力，以便更准确地理解更多类型的人类表达。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/44df5fbf-1ce1-40ce-82ea-cf26903ac8cb"/>
</div>

referring expression generation 用于给定图片和区域，对该区域进行描述。我们在 RefCOCOg 上评估了这个任务。我们使用了两个指标，BLEU-4 和 CIDEr-D，来评估生成的指令的质量。

还评估了 image captioning 和 visual question answering 任务。最后还评估了常见的语言任务。

疑问： 看官方论文给的例子，用户在和模型对话过程中，模型似乎知道当前对话是否应该输出 bbox？ 需要等 demo 开源后确认？

看样子是有特殊的 <groundding> token 来区分。一旦输入中带有这个 token，那么表示输出的句子中需要包含 bbox。那么对话时候是如何控制呢？ 通过阅读大致是：

1. 如果想让模型预测的文本中具备对其中的名词进行 groundding bbox 输出功能，需要在输入中带有 <groundding> 特殊token，在这个 token 后面的所有文本都具备 groundding 功能，这是训练时候数据决定的
2. 在对话时候，正常是不会加这个 token， 所以只就有输入 bbox 而没有输出 bbox 功能，不过如果加上应该就有了

以 Referring Expression Comprehension 为例模型的输入实际上是： 

<s><image> Image Embedding </image><grounding><p>A man in a blue hard hat and orange safety vest</p>

模型应该应该是直接输出 bbox 坐标 <box> <loc68> <loc425> </box>。

