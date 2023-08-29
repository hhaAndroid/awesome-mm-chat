# SAM

主要贡献可以看下图：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/83d56a9b-86ae-4dce-bdf2-26c93efdf85d"/>
</div>

1. 分割任何物体
2. 可以支持点框和参考mask输入，然后进行分割
3. 提供了一个超级大的 SA-1B 数据集

数据标注如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c30aa269-ab87-4998-8fea-1fa0e7467464"/>
</div>

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/6409bb34-3bc2-40dc-aa89-0c6aefb21c82"/>
</div>

可以看出标注数据集偏向于部件分割，一个人不是一个整体，而是分成了多个部件，并且是没有提供类别的。

考虑到用户点击时候，对模型预测的mask 会存在歧义，例如点击人，那么是想分割人还是分割其中某个部件呢？因此为了避免歧义也为了更好训练，会输出三个层级的有效 mask

模型架构如下：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/b5804212-0f81-45f9-b69c-3c361004b3c6"/>
</div>

- an image encoder
- a flexible prompt encoder
-  a fast mask decoder

图片编码器： 使用 MAE 预训练的 ViT 作为 image encoder，使用一个 transformer 作为 prompt encoder，使用一个 CNN 作为 mask decoder
提示编码器: 我们考虑两组提示：稀疏（点、框、文本）和密集（掩码）。我们通过位置编码来编码点和框，并与CLIP中现成的文本编码器的文本的学习嵌入和每一种 prompt 可学习 embedding 相加。使用卷积来编码 mask，并与图像嵌入按元素求和。
mask解码器。掩码解码器有效地将图像嵌入、提示嵌入和输出标记映射到掩码。对 Transformer 解码器块进行了修改，然后是动态掩码预测头。我们修改后的解码器块在两个方向上使用提示自注意力和交叉注意力（提示到图像嵌入，反之亦然）来更新所有嵌入。在运行两个块后，我们对图像嵌入进行上采样，MLP将输出标记映射到动态线性分类器，然后计算每个图像位置的掩码前景概率。
解决输出歧义问题： 输出 3 个 mask,训练时候仅仅反向传播 loss 最小的 预测mask
训练过程： 我们使用 focal loss 和 dice loss 的线性组合来监督掩码预测。我们使用几何提示的混合来训练可提示的分割任务。我们通过在每个掩码的 11 次随机采样提示来模拟交互式设置，允许 SAM 无缝集成到我们的数据引擎中。

## 代码分析

以 ViT_b 为基础，输入图片大小为 1024x1024，图像编码器输出为 (1,256, 64, 64)，

然后对输入的点或者 bbox 或者 mask 进行编码。位置编码方面本身是不可学习的，但是引入了可学习参数来区分正负点、bbox 的两个角点

```python
super().__init__()
self.embed_dim = embed_dim
self.input_image_size = input_image_size
self.image_embedding_size = image_embedding_size
self.pe_layer = PositionEmbeddingRandom(embed_dim // 2) # 随机高斯编码，无需训练

self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
self.point_embeddings = nn.ModuleList(point_embeddings)
self.not_a_point_embed = nn.Embedding(1, embed_dim)

self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
self.mask_downscaling = nn.Sequential(
    nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
    LayerNorm2d(mask_in_chans // 4),
    activation(),
    nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
    LayerNorm2d(mask_in_chans),
    activation(),
    nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
)
self.no_mask_embed = nn.Embedding(1, embed_dim)
```

编码代码如下所示：

```python
def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
) -> torch.Tensor:
    """Embeds point prompts."""
    points = points + 0.5  # Shift to center of pixel
    if pad:  # 作者的逻辑是如果没有输入box，则为 true，在 points 后面加一个假的角点，是否有必要？
        padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
    # 将输入点坐标通过可学习的位置编码进行编码
    point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
    # 需要区分正负点
    point_embedding[labels == -1] = 0.0
    point_embedding[labels == -1] += self.not_a_point_embed.weight  # 加入的假的角点
    point_embedding[labels == 0] += self.point_embeddings[0].weight  # point 负点的位置编码
    point_embedding[labels == 1] += self.point_embeddings[1].weight  # point 正点的位置编码
    return point_embedding

def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
    """Embeds box prompts."""
    boxes = boxes + 0.5  # Shift to center of pixel
    coords = boxes.reshape(-1, 2, 2)  # box 也变成 point 即可
    corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
    corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # 两个角点的位置编码
    corner_embedding[:, 1, :] += self.point_embeddings[3].weight
    return corner_embedding

def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
    """Embeds mask inputs."""
    mask_embedding = self.mask_downscaling(masks)  # 直接卷积计算即可
    return mask_embedding
```

point 和 bbox 合并为稀疏编码，mask 为密集编码。

编码后输入到 mask decoder 中

```python
low_res_masks, iou_predictions = self.model.mask_decoder(
    image_embeddings=self.features,
    image_pe=self.model.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=multimask_output,
)
```

核心逻辑如下：

```python
def predict_masks(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts masks. See 'forward' for more details."""
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    # 一共有7个序列的token，第一个是mask iou 质量预测
    # 第2 3 4 5 个是 mask embedding
    # 第 6 个 是交互的点或者框的可学习的 embedding, 让模型学习到这些点或者框的位置信息
    # 模型是预测3个不同粒度的mask,为啥 mask_tokens 是4个？ 原文中说道
    # 如果用户输入的 prompt 很多其实是没有模糊问题的，mask 应该是很明确的，这时候监督就应该是单 mask 监督
    # 所以作者用第 0 个 token 来学习这种情况。训练时候，当用户如果多个prompt时候就是训练第 0 个 token
    # 否则训练第 1～3 个 token。
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # 一般是加法，这里是 cat ?
    # Expand per-image data in batch direction to be per-mask
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    src = src + dense_prompt_embeddings  # dense 的加到 image_embeddings 上而不是 tokens
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape
    # Run the transformer
    hs, src = self.transformer(src, pos_src, tokens)  # src=key, pos_src=key_pos, tokens=query
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = self.output_upscaling(src)
    hyper_in_list: List[torch.Tensor] = []
    # 学习的是 mask embeddding，然后和 upscaled_embedding 做点积，得到 mask，参考了 maskformer 做法
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)
    return masks, iou_pred
```

总的来说，如果是单点，那么输出就是模糊的，输出应该是 multimask_output 模式，其他的例如两个点，或者提供了 bbox，或者多种 prompt 情况下就是单 mask 模式

```python
# 模型预测有两种模式，如果是单 mask 模式，那么就取第 0 个，如果是多 mask 模式，就取 1～3个
# 这个应该和训练模式有关系
if multimask_output:
    mask_slice = slice(1, None)
else:
    mask_slice = slice(0, 1)
masks = masks[:, mask_slice, :, :]
iou_pred = iou_pred[:, mask_slice]
# Prepare output
return masks, iou_pred
```

## 全自动分割模式

在图像上均匀采样 nxn 个网格点，每个点当做一个单点输入提示,从每个点SAM都可以预测多个掩模。然后,通过将mask转化为 bbox，应用非最大抑制来过滤掩模的质量并去重。
你可以采用额外的选项可以进一步改进掩模的质量和数量,例如在图像的多个裁剪上运行预测,或者对掩模进行后处理以移除小的不连通区域和空洞，就是类似大图滑窗检测一样。

# RSPrompter

RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model

https://arxiv.org/pdf/2306.16269.pdf

基于遥感数据，扩展 SAM 进行实例分割。整个训练过程中都不调整 SAM,而是仅仅调整其余部件。为了体现算法优异性，作者一共提出了 4 种基于 sam 的实例分割算法

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/fff656cc-b147-46bf-8d62-22ed5d69c4c3"/>
</div>

前三个是一个参考，第 4 个是本身主推的实例分割实现。针对 CNN 和 Transformer 的不同，作者提出了两者结构： Anchor-based Prompter 针对 Faster RCNN，Query-based Prompter 针对 DETR 系列。

在三个遥感数据集上进行评估，发现两种结构各有优点，并没有一家独大。

本文属于工程类型文章，实验做的比较多，应该是社区用户需要的，是一篇不错的文章。

SAM 作为一个分割基模型，要想取得好的实例分割效果，严重依赖于用户输入的交互式的 point bbox 或者 mask，并且在遥感领域效果不是特别好。基于此我们需要一个全自动的实例分割算法，当然是可以预测类别的。

在下游任务上微调，由于数据集比较少，效果是不太好的，因此作者实际上是从 prompt embedding 的角度出发，类似于注入新的 prompt 到 SAM 中，而不是试图去改变 SAM 基础模型本身。


问题2： RSPrompter 应该是一个通用算法，之前分析到的无法全自动问题在通用场景下也是存在，后续是否有计划在通用场景下测试并进行改进优化？ 我想这个问题是社区用户最关心的，因为用户一般会想在自己场景下采用你这套策略提升性能
问题3： 虽然 prompt 学习方法是一个不错的选择，但是如果只调整 prompt 可能不一定是最优的，是否探索过一些更多参数的微调方式？据我所知虽然 PEFT 式的微调比较高效，但是在多模态里面可学习参数量多一些应该会提示性能？
问题5： 通过多个数据集来看，anchor-based 和 query_based 各有千秋，你觉得这个现象如何解释？ 
问题4： 我看你使用看了大量 MM 系列 2.0 架构代码，看起来应用已经非常熟悉了，作为深度用户，是否可以给 openmmlab 一些建议，例如用户体验，哪里设计不合理等等任何相关的
问题1： SAM 是一个万能分割模型，为何选择做实例分割，而不是做通用分割？ 如果做通用分割是不是用途更广性能也会更好呢？ 后续是否有做通用分割的打算？

本质是上学习了新的 point embedding，然后将其注入到 SAM 中，从而实现实例分割。

# Semantic-SAM

https://arxiv.org/pdf/2307.04767.pdf
https://github.com/UX-Decoder/Semantic-SAM

利用 sam 数据进行通用图像分割+部件分割。

X-Decoder OpenSeeD SEEM 作者，已经开源代码，并且复现了 SAM。

# SAM-PARSER

https://arxiv.org/pdf/2308.14604.pdf

SAM-PARSER: Fine-tuning SAM Efficiently by Parameter Space Reconstruction


