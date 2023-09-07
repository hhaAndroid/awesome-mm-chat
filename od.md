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

DAT++: Spatially Dynamic Vision Transformer  with Deformable Attention

https://arxiv.org/pdf/2309.01430.pdf

