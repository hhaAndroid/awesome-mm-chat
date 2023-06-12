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
- 第三，我们提出了 optimization improvements，如切换自注意力和交叉注意力的顺序，使 query features 可学习，并移除 dropout，所有这些都可以在不增加计算的情况下提高性能
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

基于 transformer 的模型的缓慢收敛是由于交叉注意层中的全局上下文，因为交叉注意需要许多次训练才能学会关注局部对象区域。

Mask2Former 主要改动在于 Decoder, 提出了所谓的 masked attention 操作。其作用是通过将交叉注意力限制在每个query的预测掩码的前景区域内来提取局部特征，而不是关注整个特征图，可以大幅提升收敛速度。

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

在得到多尺度特征后，按照 multi_scale_features 层的顺序一次和 decoder 中的 transformer layer 进行交互。

```python
# multi_scale_memorys (from low resolution to high resolution)
decoder_inputs = []
decoder_positional_encodings = []
# 对 multi_scale_features 进行处理，加上 可学习的 level embed 和 decoder_positional_encoding
for i in range(self.num_transformer_feat_level):
    decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
    # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
    decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
    level_embed = self.level_embed.weight[i].view(1, 1, -1)
    decoder_input = decoder_input + level_embed
    # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
    mask = decoder_input.new_zeros(
        (batch_size, ) + multi_scale_memorys[i].shape[-2:],
        dtype=torch.bool)
    decoder_positional_encoding = self.decoder_positional_encoding(
        mask)
    decoder_positional_encoding = decoder_positional_encoding.flatten(
        2).permute(0, 2, 1)
    decoder_inputs.append(decoder_input)
    decoder_positional_encodings.append(decoder_positional_encoding)
```

为了得到 mask attention，需要先推理一遍

```python
# shape (num_queries, c) -> (batch_size, num_queries, c)
query_feat = self.query_feat.weight.unsqueeze(0).repeat(
    (batch_size, 1, 1))
query_embed = self.query_embed.weight.unsqueeze(0).repeat(
    (batch_size, 1, 1))
cls_pred_list = []
mask_pred_list = []
cls_pred, mask_pred, attn_mask = self._forward_head(
    query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
```

query_feat 可学习是 mask2former 的改进，maskformer 里面这个值是不可以学习的，初始化是0，假装 query_feat 已经更新了，可以推理一遍得到 atten_mask

开始对 decoder layer 进行推理

```python
# self.num_transformer_decoder_layers=9
# self.num_transformer_feat_level=3
# 说明 0 decoder -> 0 多尺度特征； 1 decoder -> 1 多尺度特征； 2 decoder -> 2 多尺度特征； 3 decoder -> 0 多尺度特征, 交错排布
for i in range(self.num_transformer_decoder_layers):
    level_idx = i % self.num_transformer_feat_level
    # if a mask is all True(all background), then set it all False.
    attn_mask[torch.where(
        attn_mask.sum(-1) == attn_mask.shape[-1])] = False
    # cross_attn + self_attn
    layer = self.transformer_decoder.layers[i]
    query_feat = layer(
        query=query_feat, # 可学习
        key=decoder_inputs[level_idx],
        value=decoder_inputs[level_idx],
        query_pos=query_embed, # 可学习
        key_pos=decoder_positional_encodings[level_idx],
        cross_attn_mask=attn_mask,
        query_key_padding_mask=None,
        # here we do not apply masking on padded region
        key_padding_mask=None)
    cls_pred, mask_pred, attn_mask = self._forward_head(
        query_feat, mask_features, multi_scale_memorys[
            (i + 1) % self.num_transformer_feat_level].shape[-2:])
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

如果要非常清晰的理解这个图，需要对 Mask2Former 架构图有一定了解。

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

只开源了推理代码。基于 xdecoder 思想，提出了改进版本，支持目标检测，实例分割，全景分割和语义分割。

Our OpenSeeD is the first open-vocabulary model that jointly learn on segmentation and detection.

这里的分割是指的通用分割。

1. 我们首先利用单个文本编码器对数据中出现的所有概念进行编码，并训练我们的模型将视觉标记与公共空间中的语义对齐。
2. 我们将解码器中的 object queries 显式划分为两个子类型：前景和背景查询，其中第一组负责分割和检测中的前景对象，而第二组仅用于分割中的背景内容
3. 我们引入了条件掩码解码，它从分割数据中学习从真实框解码掩码，并生成用于检测数据的 mask 辅助数据。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/18fb603c-4e7a-4944-8730-bdd6d2b4d1e0"/>
</div>

模型输入包括图片和文本词汇表，模型输出为预测的 mask、bbox 和类别概率。

```text
O = EncI(I), T = EncT(V)
< Pm, Pb, Ps〉 = Dec (Q; O)
Pc = Sim(Ps, T)
```

Q 是 object query,Ps 是 decoded semantics，Pc 是 visual-semantic matching scores。其中的 Pm 是一个统一表示，包括实例 mask 和语义 mask。

考虑到不同任务要求的语义粒度不一样，例如全景分割和语义分割是需要预测背景的，而实例分割和目标检测不需要，如果用同样的 query 来预测不同的任务性能较差。因此作者将输入
query 分成了 2 组： 前景 query 和背景 query。背景 query 用于预测 stuff mask 即为 Bridge Task Gap: Decoupled Foreground and Background Decoding 部分。

但是这样会有一个新问题没有解决： 虽然我们是用了不同 query 且共享同一个 decoder， 但是训练数据有 gap，检测数据是没有 mask 的,因此联合训练时候就只能只训练检测部分，我们的最终目的是训练一个统一模型
并且用一个统一的 loss 来联合训练。因此我们需要解决检测数据没有 mask 的问题。即为 Bridge Data Gap: Conditioned Mask Decoding 部分。实际上应该是检测数据太大了，而有mask的数据太少了，如果检测数据
不提供 mask，那么性能会低一些。

因此作者提出了 Conditioned Mask Decoding 模块，也就是说 query 实际上分成了 3个部分： 前景 query 和背景 query 和 conditioned queries。conditioned queries 的作用就是给定一个 gt bbox 和对应的类别词汇，生成对应的 mask。

对于已经有 mask 标注的数据集，则可以直接进行端到端检测训练，对于没有 mask 标注的数据集则并不是直接使用预测 mask，而是将预测 mask 加入到正负样本匹配过程中，因为这个 mask 不太强，特别是早期时候。

因此实际上这个模型的 loss 就是包括任意数据集的分割+bbox+cls loss。

考虑到这个模型实际上是开放词汇任务，因此不能过拟合到特定类别，因此加入了 Language-guided foreground query selection 模型，实际上就是前期 query 不是随机初始化的，而是需要结合 text 和图像特征进行简单 bbox head 预测而来
对于训练收敛也会更好更快。

作者在 coco 全景分割数据集和 object365 v1 和 v2数据集上面训练。v1 训练小模型，v2训练大模型。

代码是基于 mask dino 和 x-decoder 构建。

## OFA 简单阅读

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

## VisionLLM

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

### 统一的语言指令
参考之前论文做法，

Vision-Language Tasks

- image captioning： The image is <image>. Please generate a caption for the image: 
- VQA：The image is <image>. Please generate an answer for the image according to the question: <question>

Vision-Only Tasks

典型的如目标检测、实例分割和姿态估计等，考虑到用户描述的不同，作者采用了 self-instruct 方法基于一些样例生成了大量的对应任务描述，训练时候随机选择一个构成训练样本。推理时候一个实例分割语义描述的例子是：

Segment all the objects of category set <class> within the <range> of the image and generate a list of the format
(c, x1, y1, x2, y2, ..., x8, y8). Here, c represents the index of the class label starting from 0, and (x1, y1, x2, y2, ..., x8, y8) correspond to the offsets of boundary points of the object relative to the center  point. The image is: <image>

range 设置为 512。

### 语言导向的图像 tokenizer

VisionLLM 认为相对于语言信息，图像是一种外语，需要将其转换为能被 llm 理解的 token。具体做法是：

1. 先用 image  backbones 对图片进行特征提取，得到多尺度图片特征
2. 使用 text encoder 提取文本特征
3. 通过交叉注意力将语言特征注入到每个尺度的视觉特征中，产生多尺度的语言感知视觉特征，使特征在不同的模式中保持一致。跨模态的特征
4. 将融合后的多尺度特征输入到 Deformable DETR Encoder 中，这个做法和 Mask2Former 是类似的

上述步骤就可以得到序列长度为 M 的图片 token

### 基于 LLM 的开放式任务解码器

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

### 训练细节

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

## PaLI-X

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

## Open-World Entity Segmentation

https://github.com/dvlab-research/Entity/blob/main/README.md

Entity Segmentation 新任务：开放集的无需预测类别的全景分割算法。

We introduce a new image segmentation task, called Entity Segmentation (ES), which aims to segment all visual entities (objects and stuffs) in an image without predicting their semantic labels. By removing the need of class label prediction, the models trained for such task can focus more on improving segmentation quality. It has many practical applications such as image manipulation and editing where the quality of segmentation masks is crucial but class labels are less important.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/cee48864-a274-4262-9cdb-7dd5621757b6"/>
</div>

作者认为在图像编辑领域，分割质量比类别预测更重要，因此如果我们可以抛弃类别，那么模型也更容易学。

标注数据其实就是全景分割数据转化而来。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/599ead3f-617c-41de-8f5f-ae145210a907"/>
</div>

因为要做开发集，因此输出就不能固定通道数，作者采用的是 condinst 改造版本，采用动态 head 输出不定数据的不重叠的二值 mask。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/fe347c18-d773-412e-a5ed-ef6af7264a69"/>
</div>

## AIMS: All-Inclusive Multi-Level Segmentation

https://arxiv.org/pdf/2305.17768.pdf

进一步提高了 Open-World Entity Segmentation 在图片编辑领域的可用性。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c316342f-b972-40ad-ac3c-f09c22467756"/>
</div>

训练时候会直接输出3层粒度层级的实例分割图，一个是部件，一个是实体，一个是 relation levels 和两个关系图即为(relation and entity levels) 和 (entity and part levels) 。 训练完成后，不仅可以进行 One-step Inference 还可以进行 Prompt Inference。
用户提供相应的 mask prompt，结合三个不同层级输出图，可以实现比较好的图像编辑功能。

Relation-Level 不知道是啥样子的？

## DaTaSeg

DaTaSeg: Taming a Universal Multi-Dataset Multi-Task Segmentation Model

google 作品，算是 mask2former 的改进，从标题可以看出来是利用多个数据集和通用分割多任务联合训练来提升各个任务性能。暂时没有开源。

多数据集多任务联合训练其实是一个比较难的事情，虽然论文没有开源，但是训练思路还是值得学习下。从本文来看最大贡献不是网络改进，而是提出了一个非常简单有效方便扩展的多数据集多任务训练方法，这也是本文作者一直强调的。

大多数工作都使用统一的词汇表在单个任务上合并数据集，而本文工作侧重于更具挑战性的问题：将具有不同词汇表和不同任务的数据集合并，方便后续用户扩展。

和 x-decoder 做法也非常类似，不同在于类别标签的处理方面。

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

不同的数据集有不同的类名，当然可能存在类名相同，或者类含义一样但是表示不同的情况，因此作者用了一个固定的 Clip 模型对不同数据集的类别名进行嵌入。

同时增加了一个全局共享的可学习的背景嵌入。

作者发现这个架构就挺好，如果你专门针对不同的数据集引入一些特别的设计，效果还会变差。

采用一个简单的协同训练策略：在每次迭代中，我们随机抽取一个数据集，然后从所选数据集中对该迭代进行采样。这可以与每次迭代中来自多个数据集的采样形成对比。我们策略的主要优点是实现起来很简单，并允许更自由地为不同的数据集使用不同的设置，并将不同的损失应用于各种任务。为了考虑不同的数据集大小，我们控制每个数据集采样率

是一个不错的工作，期待后面会开源。

## GRES

GRES: Generalized Referring Expression Segmentation

https://github.com/henghuiding/ReLA

https://github.com/MarkMoHR/Awesome-Referring-Image-Segmentation

## RAM

[Recognize Anything: A Strong Image Tagging Model](https://recognize-anything.github.io/)

https://github.com/xinyu1205/Recognize_Anything-Tag2Text

