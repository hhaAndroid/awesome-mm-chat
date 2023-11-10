# AlignDet

ICCV2023

https://liming-ai.github.io/AlignDet/

https://arxiv.org/abs/2307.11077

大规模预训练然后下游微调的范式已广泛应用于各种物体检测算法。在本文中,我们发现现有做法中的预训练和微调阶段的数据、模型和任务之间存在差异,这隐式地限制了检测器的性能、泛化能力和收敛速度。为此,我们提出 AlignDet, 一个统一的预训练框架, 可以应用于各种现有检测器来缓解这些差异。 AlignDet将预训练过程分成了两步,即图像域和框域预训练。图像域预训练优化检测器骨干以捕获整体视觉抽象,框域预训练学习实例级语义和任务相关概念来初始化骨干之外的部分。通过结合自监督预训练的骨干,我们可以在无监督范式下预训练各种检测器的所有模块。广泛的实验表明 AlignDet 可以在不同的协议下获得显著的提高, 如检测算法、模型骨干、数据设置和训练时间表。例如,AlignDet可以在较少的轮次下使FCOS提高5.3 mAP, RetinaNet提高2.1 mAP, Faster R-CNN 提高3.3 mAP, DETR 提高2.3mAP。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a25952ee-4e99-4620-98f6-b3a7e4f42e29"/>
</div>

预训练和微调阶段的数据、模型和任务都不一样。

作者想解决的问题是： how to design a pre-training framework that can address the discrepancies in data, model, and task, and is applicable to all detection algorithms?

AlignDet将预训练过程分成两步:

- 图像域预训练: 优化检测骨干捕获整体视觉抽象。
- 框域预训练:学习对象级概念来初始化骨干外的部分。

检测器是通过框级对比学习和相关回归损失来优化的。 这有助于完全适应各种检测器,进一步提高随后的微调过程中的性能。

总的来说,图像域预训练专注于捕获通用视觉特征,框域预训练则关注检测任务特有的知识,如实例语义和位置信息。 两步预训练相互补充, 能更有效地初始化检测器, 在微调时更快收敛到高性能。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/6886c35b-95f3-4441-b6f1-0a3487c39a82"/>
</div>

相比于原先的 imagenet+coco fintune 步骤，现在变成了 imagenet + coco box pretrain + coco fintune, 但是在全量数据集下，涨的不多，训练代价应该翻了一倍。

# Description Detection Dataset 

https://arxiv.org/abs/2307.12813  
https://github.com/shikras/d-cube

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/0870af80-f983-4a0d-8f8b-3ae15cadc3d7"/>
</div>

首先指出了目前 OVD 和 REC 数据集存在的问题：

1. OVD 只能用固定的简单名词短语来定位物体
2. REC 它关注的是通过语言表达式定位对象的空间位置,并假设目标对象必须出现在给定图像中。但是在实际场景中,如果描述不在图像中存在的对象,REC算法也会输出假阳性边界框

基于此，提出了 DOD 数据集，采用类似 coco 格式组织，并且标注更加完善，一个语句可以指示多个物体。

如上说说，REC 无法表示不存在的物体，并且一句话只能对应一个物体，而 DOD 更加贴近实际，一句话可以表示多个物体，同时也可以表示不存在的物体。属于是 OVD 和 REC 数据集的合并+扩展。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c94c1a79-efe4-473c-af8e-413ad62a9ef0"/>
</div>

更多案例如上所属。

# Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection

ICCV2023

https://arxiv.org/pdf/2308.14286.pdf

# Unified Open-Vocabulary Dense Visual Prediction

https://arxiv.org/pdf/2307.08238.pdf

没有开源

解决不同源数据的开放训练问题，采用了 DA 策略，基于mask2former。

# DRR

https://arxiv.org/pdf/2309.00227.pdf  360 研究院 KDD workshop 2023
What Makes Good Open-Vocabulary Detector: A  Disassembling Perspective

# IoU-Aware Calibration

Do We Still Need Non-Maximum Suppression? Accurate Confidence Estimates
and Implicit Duplication Modeling with IoU-Aware Calibration

https://arxiv.org/pdf/2309.03110.pdf

# DAT++

DAT++: Spatially Dynamic Vision Transformer with Deformable Attention

https://arxiv.org/pdf/2309.01430.pdf
https://github.com/LeapLabTHU/DAT

看起来很强，但是为何没有 DINO 的结果？

# Beyond Generation

Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation

和下面一文做的事情类似。

https://arxiv.org/pdf/2309.05956.pdf

https://github.com/gyhandy/Text2Image-for-Detection

# DiffusionEngine

DiffusionEngine: Diffusion Model is Scalable Data Engine for Object Detection

https://arxiv.org/pdf/2309.03893.pdf
https://github.com/bytedance/DiffusionEngine

1. Diffusion 模块可以直接作为 backbone，实现目标检测功能，效果还不错，如果这条路可行？那么应该有不少可以优化的，不知道推理成本多大？
2. 同时 Diffusion 还可以用于生成数据，同时输出检测框，从而 scale up 数据集。作者基于这个原则采用 coco train2017 数据集生成了 COCO-DE 数据集，然后联合训练，在 DINO 上可以提升 3 个点

算是把 Diffusion 的优势都发挥出来了，不仅仅是做检测任务，而且可以无缝生成数据。

疑问：baseline 训练都是 6x 的，有点久？
疑问：基于 coco 2017 训练的数据看起来和原始图片其实非常类似，所以这个扩充其实只是相当于训练图片的简单扩展，多样性好像不多？如果数据集很小，扩展后依然差不多吧，效果有多明显？

# MOCAE

MOCAE: Mixture of Calibrated Experts Significantly Improves Object Detection

https://arxiv.org/pdf/2309.14976v2.pdf

代码说会开源。

# UniHead

https://arxiv.org/pdf/2309.13242.pdf
UniHead: Unifying Multi-Perception for Detection Heads

做纯检测的论文，后续会开源。

# DETR Doesn’t Need Multi-Scale or Locality Design

ICCV2023

https://arxiv.org/abs/2308.01904

# SIMPLR

比 ViTDet 更加 plain，只需要单层特征图就行。代码说后续会开源

https://arxiv.org/pdf/2310.05920.pdf
SIMPLR: A SIMPLE AND PLAIN TRANSFORMER FOR OBJECT DETECTION AND SEGMENTATION

# Anchor-Intermediate Detector

https://arxiv.org/pdf/2310.05666.pdf
Anchor-Intermediate Detector: Decoupling and Coupling Bounding Boxes for Accurate Object Detection

# Rank-DETR 

https://arxiv.org/pdf/2310.08854.pdf

Rank-DETR for High Quality Object Detection

# RichSem-重点

https://arxiv.org/pdf/2310.12152.pdf

Learning from Rich Semantics and Coarse Locations for Long-tailed Object Detection

基于 Detic 和 DINO, 用了CLIP作为辅助，但是是一个闭集算法，不过作者说应该很容易拓展为开集

后续会开源

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/da0f9931-f05d-4db8-a308-5636eba69882"/>
</div>

输入图片包括2部分：检测数据和额外数据，额外数据不一定是 imagenet-21k，可以是分类数据，也可以是图文对描述数据。

1. 将全部数据都输入到 DINO 里面进行正常的预测，输出定位和分类分支
2. DINO 上额外扩展一个语义向量分支作为辅助分支，也就是现在同时输出 3 个分支
3. 对于检测数据来说，对应匹配的 bbox 的语义向量 target 是将该图片经过 CLIP 图像特征提取器后进行 RoIAlign 后的视觉语义特征
4. 对于图像分类数据，是直接将原图不采用随机裁剪增强后的图片输入到 CLIP 中得到的全局图像特征
5. 对于其他数据(包括图像分类数据)，将其进行随机裁剪增强后输入给 CLIP 得到局部图像特征
6. 将所有 LVIS 类别经过 CLIP 文本特征后得到文本特征
7. 语义分支 loss 采用的是将预测语义向量和目标语义向量进行对比计算，同时将文本特征和视觉语义特征经过对比学习后分布进行 KL 散度计算。并不是直接计算向量的相似性

大概流程应该是这样。

# Query-adaptive DETR for Crowded Pedestrian Detection

https://arxiv.org/pdf/2310.15725.pdf

# Decoupled DETR

Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection

https://arxiv.org/abs/2310.15955

性能很低，而且都没有和现在 SOTA 对比。

#  DAC-DETR

https://github.com/huzhengdongcs/DAC-DETR 
DAC-DETR: Divide the Attention Layers and Conquer

NeurIPS 2023

# Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details

https://arxiv.org/pdf/2102.01066.pdf

评估大词汇量目标检测器：细节决定成败


# DAMEX

DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets

https://arxiv.org/pdf/2311.04894.pdf  
https://github.com/jinga-lala/DAMEX  

Neurips 2023






