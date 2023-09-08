# OVR-CNN

开山之作。

弱监督和无监督离现实太遥远，开放词汇刚好！

需要先明确 OVD 的定义。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a919eb87-0fc5-40c9-ab2d-f34b0c5ae941"/>
</div>

需要区分这些概念。 关于类别一般会非常两类： base 类和 novel 或者 target 类，其中 base 类别用于训练，而 novel 类只用于测试。

- Open-set/Open-world/OOD 都是同一个含义，是指的训练在 base 类别空间，测试只需要能够检测出 novel 类就行，不需要区分到底是哪个 novel 类
- Zero-shot 是指的训练在 base 类别空间，测试需要能够检测出 novel 类别空间，可以看出这个一个比较难得任务
- open vocabulary 是基于 Zero-shot 但是它允许使用额外的低成本训练数据或预先训练的视觉语言模型如CLIP。也就是说在训练阶段，类别空间可以包括 novel，但是不允许采用直接的 bbox 监督

可以看出 OVD 是一个更加实用更加贴近实际的算法。也就是说 zero-shot 不是不允许引入文本，而是在训练中一定不能出现 novel 类别。 上述三者强调的是类别空间，图片其实没有限制，也就是说训练图片中可以出现 novel 类但是你不能用有监督训练。

还需要明白 Visual grounding 含义：给定图片和一句话，模型输出该文本对应的 bbox，一般来说这个 bbox 通常就是 1 个。

OVR-CNN 的训练过程会稍微复杂，因为当时还没有用 CLIP, 需要先预训练，然后再进行 coco 训练。

不管咋说，将任何一个封闭集目标检测器转化为 OVD，则只需要改网络输出就行，如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/ec5e71ee-6a60-4e69-92ea-3f490f5677a0"/>
</div>

将原先的 cls head 移除，然后换成直接输出 cls embedding，将该 cls embedding 和文本的 embedding 进行相似度计算，判断属于哪个类就行。

回到 OVR-CNN 算法，其训练流程如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/64e0003f-b2e3-4594-b279-12b94a4cfd82"/>
</div>

训练包括两个步骤：
1. 使用 COCO Captions 预训练 ResNet50/V2L 模块/多模态融合模块

`python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/mmss_v07.yaml --skip-test OUTPUT_DIR ~/runs/vltrain/121`

2. COCO 目标检测任务训练，需要加载第一步的 ResNet 预训练和 V2L 模块， 文本 Embedding 和 V2L 也不需要训练。也就是只是训练 ResNet 和纯检测部分。V2L 模块用做 roihead 中 emb_pred 层的权重初始值。

`python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/zeroshot_v06.yaml OUTPUT_DIR ~/runs/maskrcnn/130`

为了能够判断背景，作者增加了全0的 embedding，实验表明不需要设置为可学习，固定为 0 就行。

**(1) 预训练**

采用 PixelBERT 里面做法，有一点改动。

图片通过 ResNet-50，输出 7x7 个 2048 的特征，然后接入一个 V2L 层变成 768 维度，将其进行转换，同时将对应的 caption 描述输入到预训练好的 BERT 中；将文本和 V2L 的视觉特征 concat 一起输入到后面的多模态融合 Transformer 中。

预训练的文本 BERT 不参与训练，其余参数都要训练。

训练 Loss 包括主 loss 和辅助 loss。 主 loss 是 grounding loss 即每个文本的词嵌入应该接近它们对应的图像区域嵌入，但是因为 COCO Captions 并没有给出文本中单词和图形区域 bbox 对应关系，而是一句话对应一个 bbox。
因此我们为每个图像-标题对定义了一个全局 grounding 分数，它是词-区域对的局部 grounding 分数的加权平均值，应该最大化匹配图像标题对的全局接地分数，而对于不匹配对，它应该最小化。
所以实际上还是将整个文本当做一个单词，然后和图像特征匹配，相当于没有考虑区域信息。实际上就是一个对比 loss。

这个 grounding loss 其实就是 itc loss，感觉没有啥不一样地方。

为了避免过拟合，作者还参考 PixelBERT 加入了 MIM 训练方式：用 [MASK] 标记随机替换每个标题中的一些单词，并尝试使用掩码标记的多模式嵌入来猜测被屏蔽的单词。

然后还采用了 image-text matching loss，一共有 3 个 loss。

COCO Captions 数据集是每张图片对应 5 个描述，描述是全局描述，可能会包括多个物体，但是并没有提供对应的 bbox 标注。

可以参考 BLIP 来理解各类图片匹配的 loss。

PixelBERT 图示如下

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/ce4afbba-fa86-436b-bc70-296ceeed02bf"/>
</div>

# Towards Open Vocabulary Learning: A Survey
https://arxiv.org/abs/2306.15880

# OWL-ST

Scaling Open-Vocabulary Object Detection

# MDETR-重点

MDETR - Modulated Detection for End-to-End Multi-Modal Understanding  

图文预训练+下游 funtune

MDETR 需要对标签进行一些前处理，并且也没有做 zero-shot 相关的，都是需要 funtune。但是他做了很多任务，虽然每个下游任务都需要简单的 funtune，而且是 DETR系列。


# GLIP-重点

论文题目： GLIP: Grounded Language-Image Pre-training  
论文地址： https://arxiv.org/abs/2112.03857   
官方地址：https://github.com/microsoft/GLIP   
Fork 并注释版本： https://github.com/hhaAndroid/GLIP/tree/hha   

由于考虑在 MMDet 中支持，因此解读以 MMDet 中的为准。

## 任务说明

何谓 Grounding? 实际上就是定位的含义。Grounding 目标检测任务为：给定图片和文本，预测出文本中提到的物体的 bbox 和类别，只不过这个类别是开放的。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233949492-33664013-94aa-4e95-b0c8-f069d133ee0d.png"/>
</div>

由于 text prompt 是可以人为输入的，用户可以输入任何描述，因此这是一个开放性检测问题。如果想做闭集检测，可以采用两种方式：

1. 输入的文本不要输入开发词汇，而是只包括特定类别命名实体，例如 `there are some holes on the road`
2. 输入的文本只输入类别，例如输入 `person. bicycle.car. motorcycle.` 等等即可。text 中不包括的不应该检测出来

实际上为了方便且合理，在 COCO 目标检测任务上是输入类别序列，通过 . 拼接而成。注意： `这个符号不能随便换，因为要和训练保持一致`。

我们不会过多的关注于训练过程，因为也没有多少能成功复现。

Phrase grounding： 输入句子和图片，将句子中提到的物体都框出来。定位任务与图像检测任务非常类似，都是去图中找目标物体的位置。对于给定的 sentence，要定位其中提到的全部物体（phrase）
Referring Expression Comprehension（REC）： 每个语言描述（这里是 expression）只指示一个物体，每句话即使有上下文物体，也只对应一个指示物体的 box 标注
Visual grounding： 是包括这两个任务？ 感觉好像现在指的其实就是 REC。论文里面描述是 Visual Grounding (VG) aims to locate the most relevant object or region in an image, based on a natural language query



## 模型说明

为何要做 Grounding？ 原因是：

1. CLIP 采用了大量图文对训练，得到了一个能够进行 zero-shot 的图像分类器，但是由于是图片级别监督，实际上无法直接进行 zero-shot 目标检测或者其他密集预测任务
2. GLIP 核心是想通过大量图文对训练，得到了一个能够进行 zero-shot 的目标感知的，语义丰富的多模态模型，典型任务就是定位任务
3. 通过将 parse grounding 任何和 object detetion 任务统一，就可以利用大量的具备 bbox 标注的图文对进行训练了，相当于大量扩充了数据集

parse grounding 通常指的是将自然语言指令与场景中的物体、位置等进行对应，以实现自然语言的场景理解。这个任务通常被称为"语义解析"或"自然语言指令理解"，是人机交互和智能机器人等领域中的重要问题。 将两个任务统一建模，数据就多了。

原文里面写的是：

1. 它允许 GLIP 从检测和接地数据中学习，以改进这两个任务并引导一个良好的 grounding 模型;
2. GLIP 可以利用大量的图像-文本对,通过以自我训练的方式生成 grounding，使学习到的表示具有丰富的语义。
3. 上述两个任务的统一还允许我们使用两种类型的数据进行预训练，并使两项任务都受益

在 27M 基础数据上预训练 GLIP，包括 3M 个人工注释和 24M 个网络抓取的图像-文本对，学习到的表征在各种对象级识别任务中表现出很强的 zero-shot 和 few-shot 能力。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233951372-214e1952-4466-4c75-90e9-819f4d2e72aa.png"/>
</div>

结构图如上所示。作者发现与仅在最后一个点积层融合视觉和语言的 CLIP 不同，我们表明 GLIP 应用的深度跨模态融合对于学习高质量的语言感知视觉表示和实现卓越的迁移学习性能至关重要。

## 效果分析

模型和配置：

```python
config_file = "configs/pretrain/glip_A_Swin_T_O365.yaml"
weight_file = "glip_a_tiny_o365.pth"
```

```python
image = load('cat_remote.jpg')
caption = 'cat'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233975344-fe1ec094-b494-4a71-a157-2cdabcdee6ea.png"/>
</div>

```python
image = load('cat_remote.jpg')
caption = 'cat . remote . '
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233975607-badba29c-8fb2-4427-b0d2-63f5308ff07c.png"/>
</div>

看起来效果还行，但是如果看下面的例子：

```python
image = load('cat_remote.jpg')
caption = 'There is a cat and a remote in the picture'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233976517-ba4e9d8f-bcc2-4723-bdb2-5f723e26e121.png"/>
</div>

虽然检测好像是对的，但是由于离线的 NLP 的命名实体会将 a cat 认为是一个单词，a 不是一个数量含义，所以 grounding 训练时候 a cat 会认为都是 gt 即 gt= [1, 1] 而不是 [0, 1]

如果你换成：

```python
caption = 'There is two cat and a remote in the picture'
```
那么由于 two cat 不是一个名词，会正确检测出 cat，而不是 two cat。

## 配置解读

一共包括 4 个配置文件：

- configs/pretrain/glip_A_Swin_T_O365.yaml
- configs/pretrain/glip_Swin_T_O365.yaml
- configs/pretrain/glip_Swin_T_O365_GoldG.yaml
- configs/pretrain/glip_Swin_L.yaml

分别对应 README 中的 ABC 和 GLIP-L 模型

- A 模型： 没有采用 deepfusion 而是类似 CLIP 直接在最后进行点积融合，轻量化模型
- B 模型： 采用 deepfusion，标准模型
- C 模型： 采用 deepfusion，但是使用了 gold grounding 
- GLIP-L 模型： 采用 deepfusion，训练数据和 C 模型一样，但是 backbone 换成了 Swin_L

我们应该重点关注 AB 模型。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234162743-d4d6ce0f-7623-478e-af50-bc14b2f3e684.png"/>
</div>

左边是 A, 右边是 B。

## 推理过程分析
代码位于： https://github.com/hhaAndroid/GLIP/blob/hha/demo.py

以下面句子为例，分析推理过程：

```python
caption = 'There is two cat and a remote in the picture'
```

(1) 图片前处理
    
```python   
image = self.transforms(original_image) # 常规的 800,1333 缩放
image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(self.device)
```

(2) 文本前处理



## 训练过程分析
假设给定一张图片，和一个 text prompt，要定位其中的物体和类别，有如下几种情况：

1. 假设文本中包括类别名 traffic light 即类别是由多个单词组成
2. 假设文本中包括的类别名会切分为多个子词 toothbrush -> tooth, #brush
3. 文本中可能包括一些特殊的 token
4. 会增加一个额外的 [NoObj] token 在句子结尾


mmdet dev-3.x 已经实现了 GLIP 推理和评估。可以直接查看代码，比看本文容易些。

# OWL-ViT

重点  
Simple Open-Vocabulary Object Detection with Vision Transformers
https://arxiv.org/abs/2205.06230
https://huggingface.co/docs/transformers/model_doc/owlvit#overview

因为图片和文本特征没有进行早期交互，因此可以实现 one-shot detection

# DetCLIP

完全没法 follow

# Grounding DINO

如何可以做 REC? 应该任务不一样吧，需要确认。

# GLIP v2

论文： GLIPv2: Unifying Localization and Vision-Language Understanding

虽然没有开源，但是也可以看看，学习下。

GLIP v1 只是支持了一个 grounding object detection 任务，而 v2 支持了更多的任务，包括 localization tasks (e.g., object detection, instance segmentation) and Vision-Language (VL) understanding tasks (e.g., VQA, image captioning).

GLv2 优雅地将 localization 预训练和三个视觉语言预训练 (VLP) 任务统一起来：

- phrase grounding 作为检测任务的 VL 重构
- region-word contrastive learning 作为新的区域词级对比学习任务
- 掩码语言建模

**这种统一不仅简化了之前的多阶段 VLP 过程，而且在 localization和理解任务之间也实现了互惠互利。这算是 v2 的一个最大改动吧，不仅仅是需要 grounding 目标检测，还同时训练语言模型部分，使其具备多任务能力。**

定位任务是仅视觉的，需要细粒度的输出（例如，边界框或像素掩码），而 VL 理解任务强调两种模态的融合，需要高级语义输出（例如，答案或字幕），如何进行统一？

作者认为： Localization + VL understanding = grounded VL understanding

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/75d76c27-26f1-495f-ad1e-9bb939969ee5"/>
</div>

从上面架构图可以看出大概做法。输入包括图片和文本，在文本方面不同的任务会采用不同的输入构建形式，经过图片和文本单独的特征提前后，进行 deepfusion，然后在接不同的任务独立的 head 进行预测。

目标检测部分和 GLIP 是一样的，实例分割部分会单独加一个实例分割head。

由于没有任何开源代码，估计也不会开源，因此了解下核心思想就行。可以看出其实也不是真正的大一统，大一统应该是类似 OFA 或者 VisionLLM 或者 X-Decoder 一样。

# DetCLIP2

DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment

# MQ-Det-开源

MQ-Det: Multi-modal Queried Object Detection in the Wild

https://github.com/YifanXu74/MQ-Det

这是一个 few-shot 方法吗？ 没有做过 few-shot，不清楚这个算法推理时候是咋运行的，一定要提供 visual query 吗？ 那是不是会麻烦点？

# ViTMDETR

CVPR2023

Dynamic Inference with Grounding Based Vision and Language Models 
https://openaccess.thecvf.com/content/CVPR2023/papers/Uzkent_Dynamic_Inference_With_Grounding_Based_Vision_and_Language_Models_CVPR_2023_paper.pdf  

好像是一个加速推理的工作？ 没有开源。

# CapDet

CVPR2023  

CapDet: Unifying Dense Captioning and Open-World Detection Pretraining  

和 GLIP 比较贴合，没有开源。推理时候也要给类别的详细描述，感觉有点麻烦。要是推理时候能够支持只输入类名，或者输入类名+详细描述，那就更好了。

# FIBER-重点

Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone

https://github.com/microsoft/FIBER
开源很全面。


# DQ-DETR

图画的不错。不过这个任务不是很感兴趣。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/dcf3f151-dfff-4578-8e23-15cc43cc7bc6"/>
</div>

DQ-DETR: Dual Query Detection Transformer for  Phrase Extraction and Grounding
https://arxiv.org/pdf/2211.15516v2.pdf

# SAS-Det
https://arxiv.org/abs/2308.06412  
Improving Pseudo Labels for Open-Vocabulary Object Detection

# OVDEval

对 OVD 进行评估  
https://github.com/om-ai-lab/OVDEval  
https://arxiv.org/pdf/2308.13177.pdf  

# MMC-Det

https://arxiv.org/pdf/2308.15846.pdf 
Exploring Multi-Modal Contextual Knowledge for Open-Vocabulary Object Detection

# Open Vocabulary Semantic Segmenter
https://arxiv.org/pdf/2309.02773.pdf 

Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter 

无需训练。

# EdaDet

iccv2023 

EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment

https://arxiv.org/pdf/2309.01151.pdf

# ZERO-SHOT VISUAL GROUNDERS

VGDIFFZERO: TEXT-TO-IMAGE DIFFUSION MODELS CAN BE ZERO-SHOT VISUAL GROUNDERS
https://arxiv.org/pdf/2309.01141.pdf

# Open-Vocabulary Vision Transformer
Contrastive Feature Masking Open-Vocabulary Vision Transformer  

https://arxiv.org/pdf/2309.00775.pdf

# SLIME: SEGMENT LIKE ME
https://arxiv.org/pdf/2309.03179.pdf 
