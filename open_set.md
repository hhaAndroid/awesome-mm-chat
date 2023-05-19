# OpenSet

X-Decoder 基于 Mask2Former，因此必须要理解 MaskFormer 和 Mask2Former。

## 前置 MaskFormer

论文名： Per-Pixel Classification is Not All You Need for Semantic Segmentation

由于论文非常有名，知乎有不少不错的解释，例如 https://zhuanlan.zhihu.com/p/389457610 和 https://zhuanlan.zhihu.com/p/532168933
本文只写几个核心部分。

图像分割是一个像素级分类per-pixel classification问题，即假设 80 类，那么输出就是 (81,h,w) 或者 (80,h,w)。
作者发现把语义分割看成一个mask classification问题不仅更自然的把语义级分割(semantic-level segmentation)和实例级分割(instance-level segmentation)联系在了一起，并且在语义分割上取得了比像素级分类方法更好的方法，
也就是只要重新构建分割任务，那么就可以非常容易的用统一架构实现语义分割和实例分割，进而也可以完成全景分割。

Mask classification 和 per-pixel classification 最大的不同在于：mask classification里面的每一个binary mask都只需要一个global的类别(而不是每个像素都需要类别)

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/fa1f02c0-d12a-481e-b8cf-e7cc26d71bf4"/>
</div>

常见的语义分割就是 per-pixel classification，而 Mask rcnn 就是 Mask classification 即输出不固定数量的二值 mask，每个 mask 对应一个类别。我们只要提前设置这个不固定数量就可以。这样就统一了语义分割和实例分割。

架构图如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/705d9e99-daf7-4313-9296-c9b10f9b197b"/>
</div>

受到DETR的启发，用“object query”的概念去预测binary mask以及每个binary mask的类别。这里想强调的一点是：虽然语义分割有固定个数的类别，我们发现query的个数不一定需要等于类别数。相反，在实验中我们发现最优的query个数其实跟类别数没有关系。

因为query个数和类别数不一样，所以我们也借鉴了DETR中bipartite matching loss的设计思想来训练我们的模型。

MMDet 中已经实现了，因此可以直接看 MMDet 的实现。

从代码来看，代码和架构图是对不上的。以 `configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py` 为例

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/ca0f53cd-a55a-4af8-8b21-e9b10422f92f"/>
</div>

简单看下没有融合前的推理过程：

(1) backbone 是 ResNet50, 输出 4 个尺度特征图

假设输入是 (1,3,800, 1199),那么 backbone 输出 4 个尺度特征图：

- (1,256,200,300)
- (1,512,100,150)
- (1,1024,50,75)
- (1,2048,25,38)

(2) MaskFormerHead 中的 pixel_decoder 是一个 6 层的 TransformerEncoderPixelDecoder，实际上就是常规的 Encoder

pixel_decoder 是一个类似 FPN 的模块进行特征融合，输出 2 个特征图

- mask_feature (1,256,200,300) 特征融合后，最大尺度对应输出经过 1 个 3x3 卷积后的特征图
- memory (1,256,25,38) 特征融合后，没有经过投影层的最小尺度特征图

```python
def forward(self, feats: List[Tensor],
            batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
    feat_last = feats[-1]
    bs, c, h, w = feat_last.shape
    input_img_h, input_img_w = batch_img_metas[0]['batch_input_shape']
    padding_mask = feat_last.new_ones((bs, input_img_h, input_img_w),
                                      dtype=torch.float32)
    for i in range(bs):
        img_h, img_w = batch_img_metas[i]['img_shape']
        padding_mask[i, :img_h, :img_w] = 0
    padding_mask = F.interpolate(
        padding_mask.unsqueeze(1),
        size=feat_last.shape[-2:],
        mode='nearest').to(torch.bool).squeeze(1)
    pos_embed = self.positional_encoding(padding_mask)
    feat_last = self.encoder_in_proj(feat_last)
    # (batch_size, c, h, w) -> (batch_size, num_queries, c)
    feat_last = feat_last.flatten(2).permute(0, 2, 1)
    pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
    # (batch_size, h, w) -> (batch_size, h*w)
    padding_mask = padding_mask.flatten(1)
    
    # 最小尺度特征图经过 encoder 处理，得到 memory
    memory = self.encoder(
        query=feat_last,
        query_pos=pos_embed,
        key_padding_mask=padding_mask)
    # (batch_size, num_queries, c) -> (batch_size, c, h, w)
    memory = memory.permute(0, 2, 1).view(bs, self.encoder_embed_dims, h,
                                          w)
    
    # 其余层继续进行特征融合
    y = self.encoder_out_proj(memory)
    for i in range(self.num_inputs - 2, -1, -1):
        x = feats[i]
        cur_feat = self.lateral_convs[i](x)
        y = cur_feat + \
            F.interpolate(y, size=cur_feat.shape[-2:], mode='nearest')
        y = self.output_convs[i](y)
    mask_feature = self.mask_feature(y)
    return mask_feature, memory
```

(3) transformer_decoder 实际上是 DetrTransformerDecoder

```python
# when backbone is swin, memory is output of last stage of swin.
# when backbone is r50, memory is output of tranformer encoder.
mask_features, memory = self.pixel_decoder(x, batch_img_metas)

pos_embed = self.decoder_pe(padding_mask)
memory = self.decoder_input_proj(memory)
# shape (batch_size, c, h, w) -> (batch_size, h*w, c)
memory = memory.flatten(2).permute(0, 2, 1)
pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
# shape (batch_size, h * w)
padding_mask = padding_mask.flatten(1)
# shape = (num_queries, embed_dims)
query_embed = self.query_embed.weight
# shape = (batch_size, num_queries, embed_dims)
query_embed = query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
target = torch.zeros_like(query_embed)

# shape (6, 100, 1, 256) 
# 核心逻辑
out_dec = self.transformer_decoder(
    query=target, # 开始初始化为0
    key=memory, # 输入
    value=memory,
    query_pos=query_embed, # query
    key_pos=pos_embed,
    key_padding_mask=padding_mask)

# cls_scores 简单线性层 ->  (6, 100, 1, num_cls) 
all_cls_scores = self.cls_embed(out_dec)
# mask_preds MLP ->  (6, 100, 1, 256) 
mask_embed = self.mask_embed(out_dec)

# (6, 100, 1, 256) 和 (1,256, 200,300)
all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed,
                              mask_features)
# (6,1,100,200,300)
return all_cls_scores, all_mask_preds
```

最终用的还只是最后一个layer输出。 MaskFormer 还是比较简单的，容易理解，只要你熟悉 DETR 的话。

## 前置 Mask2Former

论文： Masked-attention Mask Transformer for Universal Image Segmentation

相比于 MaskFormer 主要是提升训练效率和性能。

- 首先，我们在Transformer解码器中使用 mask attention，它将注意力限制在以预测片段为中心的局部特征上，这些特征可以是对象或区域，具体取决于分组的特定语义。与标准Transformer解码器中使用的交叉注意(关注图像中的所有位置)相比，我们的 mask attention 可以更快地收敛并提高性能.
- 其次，我们使用 multi-scale high-resolution features 来帮助模型分割小物体/区域。
- 第三，我们提出了 optimization improvements，如切换自注意力和交叉注意力的顺序，使 query features 可学习，并 移除 dropout，所有这些都可以在不增加计算的情况下提高性能
- 最后，我们通过 calculating mask loss on few randomly sampled points 来节省 3×training 内存，而不影响性能。

结构图如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/73310ac7-7cc7-4082-b7e6-f808d4d9f35d"/>
</div>

大致结构差不多，但是内部有些区别。

一个简单的体系结构由三个组件组成。

- 从图像中提取低分辨率特征的backbone
- pixel decoder，它从主干的输出逐渐上采样低分辨率特征，以生成高分辨率的逐像素嵌入
- 最后是 Transformer decoder，它操作图像特征来处理对象查询。最终的二进制掩码预测是用对象查询从逐像素嵌入中解码出来的

下面对要点详细说明。

### Mask Attention

基于transformer的模型的缓慢收敛是由于交叉注意层中的全局上下文，因为交叉注意需要许多次训练才能学会关注局部对象区域。

Mask2Former 主要改动在于 Decoder, 提出了所谓的 masked attention 操作。其作用是通过将交叉注意力限制在每个query的预测掩码的前景区域内来提取局部特征，而不是关注整个特征图，可以大幅减少计算量和减少显存，提升收敛速度。

masked attention 的 mask 来自上一次 decoder 的输出并通过预测后的语义 mask 通过阈值 0.5 来二值化，实现自举。

### 推理代码解读

pixel decoder 换成了 MSDeformAttnPixelDecoder，更高效，收敛更快。

```python
# shape (num_total_queries, batch_size, c)
# Mask2FormerTransformerEncoder 其实就是 DeformableDetrTransformerEncoder
# encoder_inputs 是 cat 了 backbone 输出的多尺度特征的，相当于已经包括了多尺度特征输入
memory = self.encoder(
    query=encoder_inputs,
    query_pos=level_positional_encodings,
    key_padding_mask=padding_masks,
    spatial_shapes=spatial_shapes,
    reference_points=reference_points,
    level_start_index=level_start_index,
    valid_ratios=valid_radios)

# 后续就和 maskformer 没有多大区别，主要是输出有区别
# (batch_size, c, num_total_queries)
memory = memory.permute(0, 2, 1)
# from low resolution to high resolution
num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
outs = torch.split(memory, num_queries_per_level, dim=-1)
outs = [
    x.reshape(batch_size, -1, spatial_shapes[i][0],
              spatial_shapes[i][1]) for i, x in enumerate(outs)
]
for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
               -1):
    x = feats[i]
    cur_feat = self.lateral_convs[i](x)
    y = cur_feat + F.interpolate(
        outs[-1],
        size=cur_feat.shape[-2:],
        mode='bilinear',
        align_corners=False)
    y = self.output_convs[i](y)
    outs.append(y)
multi_scale_features = outs[:self.num_outs]
mask_feature = self.mask_feature(outs[-1])

# mask_feature 没有变，依然是最大尺度尺度，但是 feature 变成了多尺度特征，而且只取了后面的 3 个
return mask_feature, multi_scale_features
```

## X-Decoder

论文： Generalized Decoding for Pixel, Image, and Language
链接： https://arxiv.org/pdf/2212.11270.pdf
project: https://x-decoder-vl.github.io/

一个模型完成 8 个任务，其中包括纯视觉和视觉语言任务，基本上都是开发集任务。

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/41522d22-b874-4dcd-b458-5d7588e96dad"/>
</div>

上图中对应的 8 个任务分别是：

- image captioning 图片描述: 给定图片，输出该图片的文字描述
- Referring captioning 指示型图片描述: 给定图片和一个文本指示，输出该指示对应物体的文字描述
- Image-Text Region Retrieval 图文区域检索: 给定一系列图片和文本描述，输出与文本描述最相似的图片，同时将该描述物体分割出来
- Open-Vocabulary Segmentation 开放词汇语义分割: 给定图片和固定数量的名词短语，输出图片中对应的所有物体的语义 mask
- Referring Segmentation 基于指示文本而非固定名词短语的开放语义分割： 通过自然的语言表达来分割一个参考物，给定图片和文字描述，输出图片中对应的所有物体的 mask
- Open-Vocabulary Instance Segmentation 开放词汇实例分割: 给定图片和固定数量的名词短语，输出图片中对应的所有物体的实例 mask
- Open-Vocabulary Panoptic Segmentation 开放词汇全景分割: 给定图片和固定数量的名词短语，输出图片中对应的所有物体的实例 mask 和语义 mask
- Referring Image Editing 基于指示文本的开放图像编辑: 给定图片和文本，输出文本所指示物体的编辑后的图片，需要借助 Stable Diffusion 完成(因为它不会文生图)

可以仔细查看上述图片，方便理解任务。Open-Vocabulary Segmentation 和 Referring Segmentation 的区别在于文本输入的不同，如果输入是名称短语那么就是 Open-Vocabulary Segmentation，如果输入是自然语言那么就是 Referring Segmentation。
但是自然语言也可以是名称短语，所以简单来看 Open-Vocabulary Segmentation 是 Referring Segmentation 的一个特例，但是实际上不是，实测发现如果在输入名称短语情况下，Open-Vocabulary Segmentation 的效果要好于 Referring Segmentation，因为
Referring Segmentation 会倾向于分割一个参考物，例如我想分割苹果，图片中存在一堆连起来的苹果和一个单独放的苹果，Referring Segmentation 一般只会分割出其中一个。但是原理上 Referring Segmentation 是可以分割所有苹果的，估计是训练数据集本身的问题。
需要进一步确认。

后面会有图片演示效果，可以更清晰的看出。

**Grounding 目标检测任务等价于： Referring 目标检测 + Open-Vocabulary 目标检测，一个任务可以做两件事情，如果输入是名称短语，那么是 Open-Vocabulary 目标检测，如果输入是自然语言，那么是 Referring 目标检测。**

X-Decoder 在诸多数据集上联合预训练，使其具备了各个任务的 zero-shot 能力，同时作者也在下游任务上进行了特定数据集微调，下游任务性能实现了 SOTA。

### 算法概述

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/b8c21373-9189-447e-b491-761e3b553843"/>
</div>

论文贡献可以分成三点：

**(1) 统一的解码框架**
基于 Mask2Former 并进一步扩展提出了一个统一的解码框架，一个模型可以完成像素级图像分割，图像级检索和视觉语言任务，不再局限在闭集。

为了实现这个统一性，作者定义了两种查询类型（潜在查询和文本查询）和两种输出类型（语义输出和像素级输出）。类似于Mask2Former，潜在查询即通用的非语义查询旨在解码通用分割的分割掩码，而新引入的文本查询使解码器对各种语言相关的视觉任务具有语言意识。同时包括两种类型的输出即像素级掩码和 token 级语义，以及它们的不同组合可以无缝地支持所有感兴趣的任务。

**(2) 端到端的学习范式**

将所有任务一起预训练学习。我们统一了三种类型的数据:全景分割、参考分割和图像-文本对。具体训练细节后续说明

**(3) 在广泛的分割和VL任务上实现了强大的 zero-shot 和任务特定的微调性能**

总结来说，核心设计包括三点：

- 有两种查询类型（潜在查询和文本查询）和输出类型（语义输出和像素级输出）
- 使用单一的文本编码器处理所有的文本语料库，包括类别概念、Referring 短语和图像描述。
- 解耦了图像和文本编码器以适应跨图像任务（例如图像-文本检索）和图像内任务（例如分割和 captioning）的需求。

基于上述 8 个任务，作者分成了两大任务，并分成了两种查询类型和输出类型。

**(1) Pixel-Level Understanding**

- 通用分割任务：语义分割，实例分割，全景分割
- _开放词汇分割_：开放词汇语义分割、实例分割、全景分割，论文做的实际上是这个
- **Referring Segmentation**: 本质上是开放词汇，但是它在训练和推理时间中不假设固定的短语数量，也可以分成语义分割、实例分割、全景分割三个细方向，实际上作者做的只是 Referring 语义分割

**(2) Visual-Language Understanding**

- _图像-文本检索_
- _图片理解_

基于上述划分，作者进行了输入和输出统一，如下所示：

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/fcc3df36-d31b-4741-9016-b9af6d519573"/>
</div>

如果要非常清晰的理解这个图，需要对 Mask2Former 架构图有一定了解。待补充...

通用语义分割和 Referring Segmentation 比较好理解，基于语义输出然后转换为像素级输出即可完成像素级任务。对于图像-文本检索和图片描述稍微不一样，需要语义输出，后续结合代码再分析。

### 模型架构详述
### 推理过程分析
### 训练过程分析
### 数据集整理
我们在包括全景分割、图像-文本对 (itp) 和参考分割在内的三种类型的数据上预训练 X-Decoder。

- 对于全景和 Referring 分割，使用带有分割注释的 COCO2017，并排除 Ref-COCOg UMD 和 COCO Karpathy 的验证集。总共有 104k 张图像用于分割预训练，其中 30k 张图像带有 Referring 分割注释。
- 对于图文对，我们使用标准的 4M 语料库，包括 Conceptual Captions 、SBU Captions 、Visual Genome  和 COCO Captions。

- RefCOCO: https://arxiv.org/pdf/1608.00272.pdf
- RefCOCO+
- G-Ref

全景分割数据集
- ADE 
- Cityscapes 
- COCO 
- Flickr30k

### 结果分析


## OpenSeeD

论文： A Simple Framework for Open-Vocabulary Segmentation and Detection
链接：https://arxiv.org/pdf/2303.08131.pdf
github: https://github.com/IDEA-Research/OpenSeeD

