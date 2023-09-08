# BEiT

# BEiT v2

# BEiT v3

# EVA

EVA: Exploring the Limits of Masked Visual Representation Learning at Scale

受模型缩放在 NLP 中的成功启发，我们如果能够将这种成功从语言翻译成视觉则很有吸引力，即扩大以视觉为中心的基础模型，有利于视觉和多模态下游任务。在视觉里面也有 MIM 方法，
然而，最具竞争力的十亿级视觉预训练模型仍然严重依赖于具有数亿(通常是公开无法访问)标记数据的监督或弱监督训练。在经过严重监督的预训练之前，MIM在某种程度上只被用作初始化阶段，或者说纯粹 MIM 预训练的十亿级视觉模型性能依然不够。
作者认为原因是：自然图像是原始图像和信息稀疏的事实。同时，理想的视觉pretext任务不仅需要抽象低级几何和结构信息，还需要高级语义，而高级语义很难通过像素级恢复任务捕获。

然后作者发现简单地使用图像-文本对齐(即CLIP)视觉特征作为MIM中的预测目标可以很好地扩展，并在广泛的下游基准上取得了良好的性能。这种预训练任务利用了图像-文本对比学习的高级语义抽象以及MIM中几何和结构的良好捕获的好处，这通常涵盖了大多数视觉感知任务所需的信息。
通过这个MIM pretext任务，我们可以有效地将被称为EVA的普通ViT编码器扩展到具有强大视觉表示的10亿个参数，可以很好地转移到广泛的下游任务。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/436b6cfa-21ef-4cf7-9812-8e8c142092a7"/>
</div>

从精度来看其实没有涨啥。因此本文的重点不是刷点，而是说明视觉模块可以通过 MIM 成功训练到 10 亿参数。

不过 LVIS 实例分割结果性能很强，LVIS 是长尾问题，这说明 EVA 模型确实学到了很多图文的信息。

**EVA 没有直接用到图文对数据，而是仅仅通过 MIM 来重构 mask 掉的视觉语义特征，视觉语义特征的 target 来自 CLIP 输出的视觉特征。**

EVA 模型本身就是一个 10 亿参数的 ViT 模型。EVA 经过预训练，以可见图像块为条件重建被屏蔽的图像-文本对齐视觉特征。我们用 [MASK] 标记破坏输入补丁，我们使用掩码比为 40% 的逐块掩码。MIM预训练的目标来自公开可用的 OpenAI CLIP-L/14 视觉特征，在224×224像素图像上训练。EVA 的输出特征首先被归一化，然后通过线性层投影到与 CLIP 特征相同的维度。我们使用负余弦相似度作为损失函数。

EVA-CLIP 也是一个亮点：EVA 不仅是广泛的视觉下游任务的强大编码器，也是在视觉和语言之间建立桥梁的多模态枢轴。
对于我们的 CLIP 模型，我们通过预训练的 EVA 和 OpenAI CLIP-L 中的语言编码器来初始化视觉编码器。预训练实现基于 Open CLIP。我们还采用DeepSpeed优化库和ZeRO stage-1优化器来保存内存。我们发现，在整个训练过程中使用动态损失缩放的 fp16 格式足够稳定，而使用 bfloat16 格式是不必要的。这些修改使我们能够在 256× NVIDIA A100 40GB GPU 上训练批量大小为 41k 的 1.1B CLIP。

# EVA-02

EVA-02: A Visual Representation for Neon Genesis

EVA 还是太大了，作者觉得应该可以充分压榨下性能。因此提出了下一代 Transformer ViT 结构，并优化了 MIM 预训练策略，从而实现在少 3 倍参数量情况下，性能还类似甚至高一点。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/5e338d63-31e3-46f4-a6fe-c774b500b7c7"/>
</div>

## 结构修改

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/40e45302-2f10-4323-b083-3d612f48d9f9"/>
</div>

这些技术都是在前人文章中参考过来的，证明了其有效性。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/d7b4c31f-1d3b-4161-b374-71110b7d0722"/>
</div>

## MIM 预训练优化

需要更强的 CLIP 模型作为 teacher，并且要加入更多的训练 epoch 会更好。

# InstructDiffusion

InstructDiffusion: A Generalist Modeling Interface for Vision Tasks  
https://arxiv.org/pdf/2309.03895.pdf
https://github.com/cientgu/InstructDiffusion 
微软



