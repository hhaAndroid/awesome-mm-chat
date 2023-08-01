# LMSEG

ICLR2023 

LMSEG: LANGUAGE-GUIDED MULTI-DATASET SEGMENTATION

https://arxiv.org/pdf/2302.13495.pdf

多数据联合训练常见的问题：

(i) 需要手动协调来构建统一的类别分类； 
(ii) 不灵活的 one-hot 形式导致模型对未标记类别进行再训练

我们引入了一个预训练的文本编码器，将类别名称映射到文本嵌入空间作为统一的分类，而不是使用不灵活的 one-hot 标签。该模型动态地将语义查询与类别嵌入对齐。
类别引导的解码模块不是用统一的分类空间重新标记每个数据集，而是旨在动态地引导对每个数据集的分类的预测。此外，我们采用了一种数据集感知增强策略，为每个数据集分配一个特定的图像增强管道，该策略可以适应来自不同数据集的图像的属性。

MSeg: A Composite Dataset for Multi-Domain Semantic Segmentation 使用 one-hot 标签手动建立统一的分类，重新标记每个数据集，然后为所有涉及的数据集训练一个分割模型，耗时且容易出错。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c85cd913-ecda-4a7e-9303-6ca3d9a2f7b4"/>
</div>

## 架构

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/59d0282e-f3d0-4a47-96e0-f0391e52a889"/>
</div>

采用了 Mask2Former 架构。文本编码器不训练，测试时候离线对所有数据集的类别处理下就行。参考 COOP 做法，也引入了一个全局共享的可学习向量，和类别名拼接一起，

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/5bc5f14c-b3e7-4739-ba8d-522a68f0818e"/>
</div>

对于每个训练样本，我们确定样本来自哪个数据集并选择相应的增强策略 Ai。此外，数据集感知增强策略允许我们与单数据集训练模型进行公平比较，同时保持每个数据集的数据增强相同。

整体训练流程看上面图其实非常清楚了。

对于语义分割，我们在四个公共语义分割数据集上进行评估：ADE20K(150 个类别，包含 20k 张图像用于训练，2k 图像用于验证）、COCO-Stuff-10K (171 个类别，包含 9k 张图像用于训练，1k 图像用于测试）、Cityscapes (19 个类别，包含 2975 张图像用于训练，500 张图像用于验证，1525 张图像用于测试），Mapillary Vistas  (65 个类别，包含 18k 张图像用于训练，2k 图像用于验证，5k 图像用于测试）。对于全景分割，我们使用 COCO-Panoptic（80 个“事物”和 53 个“东西”类别）、ADE20K-Panoptic （100 个“事物”和 50 个“东西”类别）和 Cityscapes-Panoptic  (8 个“事物”和 11 个“东西”类别)

代码是在 Mask2Former 上面改的。

由于没有开源代码，在推理时候是否也需要指定数据集？ 对于开放词汇检测没有这个问题，如果是闭集检测，那么需要将所有类名都用上? 如果类名不同含义相同的，预测成了不同的实例，是否要进行后处理？

# OneFormer

CVPR2023

OneFormer: One Transformer to Rule Universal Image Segmentation

# Towards Universal Object Detection by Domain Attention

CVPR2019

https://github.com/frank-xwang/towards-universal-object-detection

# Detection Hub

Detection Hub: Unifying Object Detection Datasets via Query Adaptation on Language Embedding

# UniDet

CVPR2022

[Simple multi-dataset detection](https://github.com/xingyizhou/UniDet)  

https://arxiv.org/abs/2102.13086v2

和 Object Detection with a Unified Label Space from Multiple Datasets 好像是同一篇文章？

开源非常全面。

# GroupSoftmax

https://zhuanlan.zhihu.com/p/73162940

# Detic-重点

ECCV2022

Detecting Twenty-thousand Classes using Image-level Supervision

代码开源非常全面且规范，非常良心，后续 follow 的人也多。

我们提出了 Detic，它简单地在图像分类任务上训练检测器的分类器(这通常是需要设定任意物体都能够被检测出来，然后才能分类)，从而将检测器的词汇量扩展到数万个概念。与之前的工作不同，Detic 不需要复杂的分配方案，根据模型预测将图像标签分配给框，从而更容易实现和兼容一系列检测架构和主干。

带标注的目标检测数据集太少，分类的话有 imagenet-21k has 21K classes and 14M images 1千 400 万张图片，因此如果弱监督可行，那是一个不错的探索。

出发点： We observe that the localization and classification sub-problems can be decoupled. 并且基于监督学习的检测器是能够很好的检测出新物体的，最大问题是不知道类别。因此 we focus on the classification sub-problem and use image-level labels to train the classifier and broaden the vocabulary of the detector.
做法也非常简单：We propose a simple classification loss that applies the image-level supervision to the proposal with the largest size, and do not supervise other outputs for image labeled data. This is easy to implement and massively expands the vocabulary.

和常规的弱监督不一样的是，本文做法仅仅关注分类分支，因此弱监督也只是作用于分类分支，回归分支不专门优化，只是和检测数据联合训练就行。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/0db24184-c30c-4778-81c5-873715dfd913"/>
</div>

流程非常简单，对于检测器就是正常训练，但是对于没有bbox标签的数据，则直接用 rpn 提取的最大 roi 作为该图像的 proposal ，然后仅仅训练分类分支。开发词汇的话，w 要变成 embeding，否则无法进行新类检测。

实验主要是在 LVIS-base、 ImageNet21k 和 Conceptual Captions 上面进行，

数据定义：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/5ffa581f-d132-4fcc-b00e-2cfce55bbaf9"/>
</div>

1. LVIS: 我们将此仅具有频繁类和常见类的部分训练集称为 LVIS-base, 罕见类作为新类进行评估。我们报告了掩码 mAP，它是 LVIS 的官方指标。虽然我们的模型是为框检测而开发的，但我们使用标准的类不可知掩码头来为框生成分割掩码。我们只在检测数据上训练掩码头
2. Image-supervised data：我们使用精确的文本匹配从 Conceptual Captions 中提取图像标签，并保留图像标签中至少包括了一个 LVIS 类的图像。生成的数据集包含 1.5M 图像，包含 992 个LVIS类。

## 实验细节

**(1) Box-Supervised: a strong LVIS baseline**

LVIS 类别很多，训练有一定技巧，作者首先基于 bbox 监督构建了一个强的 baseline，然后基于这个 baseline 再进行后续实验。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a88d6c0d-cc56-4d2d-bd16-e54463caa624"/>
</div>

在 LVIS 上训练实例分割任务，注意这个应该是全部训练集 full LVIS。最后一行是在 +IN_21K 后面继续实验，把 LVIS 中罕见类去掉训练的结果(也就是这里变成了开放词汇检测)，由于去掉了部分标注，因此测试性能会低一些，但是因为用了 CLIP classifier，所以依然可以检测出。 
最后一行的移除罕见类就是本文的 LVIS baseline。

**(2) Multi-dataset training**

我们以 1: 1 的比例对检测和分类小批量进行采样，而不管原始数据集的大小如何。我们将来自同一数据集的图像分组到同一个GPU上，以提高训练效率

我们总是首先训练一个收敛的 baseline 基类模型（4× schedule），并使用额外的图像标记数据对其进行微调，以进行另外 4 倍调度。我们确信仅使用框监督确认微调模型并不能提高性能。COCO 也是一样训练。

作者发现图片标签输入情况下图片输入分辨率对性能也有不少影响。Using smaller resolution in addition allows us to increase the batch-size with the same computation. In our implementation, we use 320×320 for ImageNet and CC。

## 改进论文 mm-ovod

https://github.com/prannaykaul/mm-ovod

已开源


# UniDetector-重点

CVPR2023

Detecting Everything in the Open World: Towards Universal Object Detection

1）它利用多个源和异构标签空间的图像通过图像和文本空间的对齐进行训练，这保证了通用表示有足够的信息
2）由于视觉和语言模式的丰富信息，它很容易推广到开放世界，同时保持可见和不可见类之间的平衡
3）通过我们提出的解耦训练方式和概率校准，进一步促进了新类别的泛化能力

这些贡献允许 UniDetector 检测超过 7k 个类别，这是迄今为止最大的可测量类别大小，只有大约 500 个类参与训练。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/d3b13027-a3e2-41fe-be1e-b2bc67f924f7"/>
</div>

## 架构

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/227e8ad5-0619-4dbb-adce-b182a440951b"/>
</div>

训练过程分成三步：

**Step1:大规模图像-文本对齐预训练**

传统的只有视觉信息的全监督学习依赖于人工注释，这限制了通用性。考虑到语言特征的泛化能力，我们引入了语言嵌入来辅助检测。受语言图像预训练最近成功的启发，我们涉及来自预训练图像-文本模型的嵌入。我们采用 RegionCLIP 预训练参数进行实验

**Step2:异构标签空间训练**

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/477ec48b-7f33-45ff-9fc6-90521da7ac71"/>
</div>

实际上有三种做法，如上所示。采样策略一般也有多种用于对付长尾问题，但是我们是要做开放集，因此长尾问题应该可以忽略，因此采用最简单的随机采样就行。

也就是训练时候虽然是 3 个数据集联合训练，但是对于每个样本都会标记其属于哪个数据集，roi 特征的视觉特征会和三个数据集的 CLIP 类别语言特征计算相似度(实际上应该是属于哪个数据集就算哪一个就行)，然后如果该数据集属于1，那么就只计算 1 数据集的 cls embeding loss。

**Step3：开放世界推理**

使用经过训练的对象检测器和来自测试词汇表的语言嵌入，我们可以直接在开放世界中执行检测进行推理，而无需任何微调。然而，由于在训练期间没有出现新类别，检测器很容易生成过度自信的预测。在这一步中，我们提出了概率校准来保持基类和新类别之间的推理平衡。

推理时候可以先提供类别信息即可。

训练数据集包括：COCO, Objects365, and OpenImages，并且在 LVIS, ImageNetBoxes, and VisualGenome 上进行评估。作者并没有用全量数据集，而是分别从 3 个数据集里面都随机抽一些出来，估计是不是因为样本量差距太大了。
不抽取估计会有偏？

作者这种训练方式应该很容易扩展，就是不知道如果用全量会不会出现问题。

## 推理时候概率校准

# DaTaSeg-重点

UniDet 和 Detic 的作者，算是改进版，没开源。

DaTaSeg: Taming a Universal Multi-Dataset Multi-Task Segmentation Model

google 作品，算是 mask2former 的改进，从标题可以看出来是利用多个数据集和通用分割多任务联合训练来提升各个任务性能。

多数据集多任务联合训练其实是一个比较难的事情，虽然论文没有开源，但是训练思路还是值得学习下。从本文来看最大贡献不是网络改进，而是提出了一个非常简单有效方便扩展的多数据集多任务训练方法，这也是本文作者一直强调的。

## 概述

核心疑问： 我们能否利用不同的分割数据集集合为所有分割任务联合训练单个模型？

**大多数工作都使用统一的词汇表在单个任务上合并数据集，而本文工作侧重于更具挑战性的问题：将具有不同词汇表和不同任务的数据集合并，方便后续用户扩展。**

和 x-decoder 做法也非常类似，不同在于类别标签的处理方面。

为了鼓励多个分割源之间的知识共享和迁移，我们的网络架构在所有数据集和任务之间共享相同的权重集。此外，我们利用文本嵌入作为类分类器，它将来自不同数据集的类标签映射到一个共享的语义嵌入空间。共享语义嵌入空间可以将不同数据集的不同标签但是含义相同的
标签尽可能靠近。

因为引入了文本，因此也是可以做开发词汇分割。

模型训练过程和常规的 Mask2Former 没有区别，也是针对不同任务采用不同的 merge 操作，然后再计算 Loss. 为了充分应用廉价的 box 标注，作者还引入了弱监督。

训练数据组成：

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/80011c7a-22cc-44d5-ba00-39641549d298"/>
</div>

包括 ADE20k semantic、COCO panoptic 和 Objects365 detection，可以发现 Objects365 detection 数据实际上只有 box，因此训练中包括部分弱监督，其实就是 boxinst 的做法。

本文的作者包括 Simple multi-dataset detection 论文 和 https://github.com/facebookresearch/Detic 作者，联合训练。

算法方面的核心做法如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmsegmentation/assets/17425982/f8cbef73-8f06-4ece-a32c-673b8bee7910"/>
</div>

网络输入是单张图片，输出是 n 个 mask proposal 和对应的类别嵌入，对于不同的数据集和任务采用不同的 merge 方式得到语义分割，全景分割和实例分割输出，然后进行正常的分割训练即可。

多数据集联合训练架构：

<div align=center>
<img src="https://github.com/open-mmlab/mmsegmentation/assets/17425982/9cb6562a-c501-4b3d-a1f9-5668d1347c98"/>
</div>

不同的数据集有不同的类名，当然可能存在类名相同，或者类含义一样但是表示不同的情况，因此作者用了一个固定的 Clip 模型对不同数据集的类别名进行嵌入。同时增加了一个全局共享的可学习的背景嵌入。

作者发现这个架构就挺好，如果你专门针对不同的数据集引入一些特别的设计，效果还会变差。

采用一个简单的协同训练策略：在每次迭代中，我们随机抽取一个数据集，然后从所选数据集中对该迭代进行采样。这可以与每次迭代中来自多个数据集的采样形成对比。我们策略的主要优点是实现起来很简单，并允许更自由地为不同的数据集使用不同的设置，并将不同的损失应用于各种任务。为了考虑不同的数据集大小，我们控制每个数据集采样率

对于全监督任务，我们采用直接掩码监督，可以从全景/语义/实例分割groundtruth中获得。对于弱边界框监督，我们采用的投影 loss 进行匹配成本和训练损失

使用掩码监督在COCO全景和ADE20k语义上训练和评估DaTaSeg，以及使用边界框弱监督的对象365-v2检测数据集。COCO全景是最流行的全景分割基准，包含118,287张训练图像和5000张验证图像。COCO有80个事物类别和53个东西类别。ADE20k 语义是使用最广泛的语义分割基准之一，包含 150 个类别、20210 个训练图像和 2,000 个验证图像

为了评估弱监督实例分割结果，我们手动标记了来自 Objects365 验证集的 1,000 张图像的子集。

我们使用 ResNet 或 ViTDet 主干进行实验。CLIP-L/14

实验的具体细节可以查看原文。是一个不错的工作，期待后面会开源。

这个模型没有实现全量数据集的模型空间，因此也不需要解决标签冲突的问题，每次推理时候应该都要选择数据集和任务。

# ScaleDet-重点

CVPR2023 

ScaleDet: A Scalable Multi-Dataset Object Detector

本文相比 DaTaSeg 解决了标签冲突问题，一次训练可以预测所有数据集的类别，更加贴合实际应用。

多数据集联合训练为在不额外注释成本的情况下利用异构大规模数据集提供了一种可行的解决方案。在这项工作中，我们提出了一种可扩展的多数据集检测器（ScaleDet），它可以在增加训练数据集数量时扩大跨数据集的泛化能力。与现有的多数据集学习器主要依赖于手动重新标记工作或复杂的优化来统一跨数据集的标签不同，我们引入了一个简单而可扩展的公式推导用于多数据集训练的统一语义标签空间。ScaleDet 通过视觉-文本对齐进行训练，以学习跨数据集具有标签语义相似性的标签分配。一旦经过训练，ScaleDet 就可以在任何给定的上游和下游数据集上很好地泛化，具有可见和不可见的类。我们使用LVIS、COCO、Objects365、OpenImages作为上游数据集进行了广泛的实验，以及来自野外目标检测的13个数据集(ODinW)作为下游数据集。我们的结果表明，ScaleDet 在 LVIS 上实现了令人信服的强大模型性能，在 COCO 上 mAP 为 50.7，COCO 为 58.8，Objects365 为 46.8，OpenImages 为 76.2，ODinW 为 71.8，超过了具有相同主干的最先进检测器。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/aa42ed6d-e70d-4443-b317-f89fa8d45a50"/>
</div>

为了训练跨多个数据集的目标检测器，我们需要解决几个挑战

1. 多数据集训练需要跨数据集统一异构标签空间，因为标签定义是特定于数据集的。来自两个数据集的标签可能表明相同或相似的对象。例如，“footwear”和“sneakers”是OpenImages和Objects365中两个不同的标签，但指的是相同类型的对象
2. 数据集之间的训练设置可能不一致，因为不同大小的数据集通常需要不同的数据采样策略和学习率
3. 多数据集模型应该在单个数据集上的性能优于单数据集模型

由于标签空间异构、数据集之间的域差异以及过度拟合更大的数据集的风险，这具有挑战性。为了解决上述挑战，现有的工作求助于手动重新标记类标签，或者训练多个特定于数据集的分类器(Simple multi-dataset detection)，这些分类器具有约束，以关联跨数据集的标签。然而，这些方法缺乏可扩展性。

我们提出了两项创新：一种可扩展的范式来统一多个标签空间，以及一种新颖的损失来学习跨数据集的硬标签和软标签分配。硬标签分配用于消除概率空间中的类标签的歧义，但软标签分配作为正则化器，在语义相似性空间中关联类标签。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c4fcf364-f075-46a5-872d-2a88a001f408"/>
</div>

从结构图来看，比较简单。标准的目标检测算法训练包括 bbox loss 和 cls loss，本算法只是修改了 cls loss 部分。整个代码是基于 Detic 做的，虽然没有开源，实际上是 centernet2

每个 roi 都可以输出对应的视觉特征 v， 每个 v 都应该匹配到一系列文本 embedding。

**(1) 离线计算标签空间和 text prompts**

也采用了 CLIP 中的 prompt engineering，对每个类名生成一系列 prompt，然后输入到 CLIP 文本编码器中，将所有文本的 emdedding 进行平均得到该类的文本嵌入。

**(2) 建立统一的标签空间**

这个是最麻烦的，因为情况比较多，可能类名不一样，但是含义相同，含义只是相似。为了避免这个问题带来的歧义，作者简单粗暴，整个标签空间就是所有数据集类名嵌入向量 concat 就行，啥也不处理。

**(3) 计算语义相似性**

CLIP 计算得到的文本嵌入向量本身就应该具备含义相近的类别嵌入更加靠近。相似性计算采用了两两类别标签的 cosine similarity，计算得到的相似性矩阵就是一个标签空间的相似性矩阵。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/51712582-27e7-4fe9-bb8b-d4823b5f4c80"/>
</div>

有了这些标签语义相似性矩阵，我们可以引入显式约束，允许检测器在统一的语义标签空间上学习。这个矩阵可以离线计算，新增数据集也是一样扩展，非常方便简洁。

**(4) 训练视觉特征和文本特征对齐**

训练过程类似 CLIP。对于每张图片输出的 roi 视觉特征，将其和所有的文本嵌入向量计算 cosine similarity， 然后分成两个 loss，一个是 hard 一个是 soft。

hard loss 就是将从视觉特征对应的标签计算的 cosine similarity 中取出，然后采用 BCE 训练，其实就是普通的分类 loss，可以看上面的图示会比较清楚。
soft loss 是作为一种正则，计算对应类别的 cosine similarity 和前面定义的语义相似性矩阵的 MSE Loss，某个类别的语义相似性向量实际上是该类别和其他所有类别的语义相似度。

一旦使用 ScaleDet 进行训练，ScaleDet 就可以部署在任何包含可见或不可见类的上游或下游数据集上。通过将统一的标签空间替换为任何给定测试数据集的标签空间，ScaleDet 可以基于视觉语言相似性计算标签分配。

训练数据包括 LVIS/COCO/Objects365/OpenImages，训练后作者在 Object Detection in the Wild 任务上面测试。

我们使用CenterNet2和在ImageNet21k上预训练的骨干。我们使用来自CLIP或OpenCLIP的提示文本嵌入来编码类标签。对于增强，我们使用大规模抖动和使用ResNet50、Swin Transformer作为骨干时输入大小为640×640,896×896的高效调整大小裁剪。我们使用 800×1333 的输入大小进行测试。我们使用 Adam 优化器并在 8 个 V100 GPU 上训练。对于多数据集训练，我们直接结合所有数据集并使用LVIS中的重复因子采样，而不使用任何多数据集采样策略。

作者论文说会开源预训练权重，但是由于代码库是专用的，因此不会开源训练部分。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/32c07592-8568-4891-b9f7-a14d50515b13"/>
</div>

测试时候是将视觉特征和所有类别的文本特征计算相似度，取最大的作为预测类别。

# Cross-dataset Training for Class Increasing Object Detection

https://arxiv.org/pdf/2001.04621.pdf

# Towards a category-extended object detector without relabeling or conflicts

# OmDet: Language-Aware Object Detection with Large-scale Vision-Language Multi-dataset Pre-training

# Language-aware Multiple Datasets Detection Pretraining for DETRs

https://arxiv.org/pdf/2304.03580.pdf

# Multi-Task Heterogeneous Training

https://arxiv.org/pdf/2306.17165.pdf
An Efficient General-Purpose Modular Vision Model via Multi-Task Heterogeneous Training

MOE 系统。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/3071aaa4-3635-48d2-b51a-f104362dbf4f"/>
</div>

如何进行异构空间多任务多模型训练？ 难点在于 data distribution, architectures, task-specific modules, dataset scales, and sampling strategies 的设计上。

# UOVN

Unified Open-Vocabulary Dense Visual Prediction

https://arxiv.org/pdf/2307.08238.pdf

基于 mask2former 没有开源。

# UniT
UniT: Multimodal Multitask Learning with a Unified Transformer
https://arxiv.org/abs/2102.10772v3

# PolyViT

PolyViT: Co-training Vision Transformers on Images, Videos and Audio
https://arxiv.org/abs/2111.12993


# Prompt Guided Transformer for Multi-Task Dense Prediction

https://arxiv.org/pdf/2307.15362.pdf


