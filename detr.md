我正在飞书上面整理 DETR 系列相关进展，因此本文档内容不会写的很细。后续会公布详细的DETR 系列相关进展飞书链接

# DETR

MultiheadAttention 中有两个核心参数：

- attn_mask 只用于Decoder训练时的解码过程，作用是掩盖掉当前时刻之后的信息，让模型只能看到当前时刻（包括）之前的信息。维度是 (L, S) , `L` is the target sequence length, and :math:`S` is the source sequence length
- key_padding_mask 指的是在 encoder 和 Decoder 的输入中，由于每个batch的序列长短不一，被padding的内容需要用key_padding_mask来标识出来，然后在计算注意力权重的时候忽略掉这部分信息。维度是 (B, S)

a fixed small set of learned object queries 才是关键。

点集预测模式，主要是两个贡献：

1. 引入了双边匹配 loss
2. 并行解码，而非传统的 transformer 的序列解码方式

DETR 的训练设置与标准对象检测器不同。并且辅助loss非常关键，我们彻底探讨了哪些组件对性能至关重要。

DETR 在算 loss 时候使用的是普通的 CE Loss，但是考虑到前后景不平衡，专门对背景类进行 0.1 权重衰减。

```python
bbox_head=dict(
    type='DETRHead',
    num_classes=80,
    embed_dims=256,
    loss_cls=dict(
        type='CrossEntropyLoss', 
        bg_cls_weight=0.1, # factor
        use_sigmoid=False,
        loss_weight=1.0,
        class_weight=1.0),
    loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
```

那如果直接换成 Focal Loss 会有多少提升？ 后续论文好像换了

在 bbox loss 方面，作者是直接预测归一化的 cxcywh 坐标，如果只采用 l1 loss那么会有尺度问题，因此作者还引入了 giou loss，一定程度上克服这个目标尺度问题。

在 DETR 系列中, auxiliary losses 非常重要。我们在每个解码器层之后添加预测 FFN 和匈牙利损失。所有预测 FFN 共享参数。我们使用一个额外的共享层范数将输入归一化为来自不同解码器层的预测 FFN。 
也就是不同的 decoder 层都是共享同一套 head。 共享还是不共享？ 是一个问题？

DETR 系列中需要用强增强，否则性能不行。为了帮助通过编码器的自注意力学习全局关系，我们还在训练期间应用了随机裁剪增强，将性能提高了大约 1 AP。

注意：At inference time, some slots predict empty class. To optimize for AP, we override the prediction of these slots with the second highest scoring class, using the corresponding confidence. This improves AP by 2 points compared to filtering out empty slots.
其意思是在评估或者说刷点时候，不做任何 score thr 处理，而是直接强行取前 100 个检测框，虽然里面肯定有不少是背景，但是这样可以提升大概 2 个点 mAP。实际应用肯定还是要用阈值处理的。

DETR 虽然说训练的整体 mAP 性能和单尺度的 Faster RCNN 差不多，但是小物体差距巨大，持平的原因是大物体性能很高。原因是啥？

作者分析了 decoder 层参数的重要性。可以发现第一个层输出性能和最后一层性能差了 8 个点，而且前面层更需要 NMS 后处理，到最后的层实际上就不太需要 NMS了。
Transformer 的单个解码层无法计算输出元素之间的任何互相关，因此容易对同一对象进行多次预测。在第二层和后续层中，激活上的自注意力机制允许模型抑制重复预测。我们观察到 NMS 带来的改进随着深度的增加而减少。
在最后一层，我们观察到应用了 NMS 还会给 AP 带来一些下降，因为 NMS 错误地删除了真阳性预测。也就是说 NMS 最好不要，对性能还有点影响。

DETR 论文里面也推广到了全景分割领域，但是由于后续有更好的 maskformer，因此就不是重点了。

# DeformableDETR

https://zhuanlan.zhihu.com/p/372116181

- 多尺度输出特征图
- 新的 attention 
- 训练过程几乎没变

DETR 有两个缺点：

- Transformer 注意力模块在初始化时，分配给所有特征像素的注意力权重几乎是均等的，这就造成了模型需要长时间去学习关注真正有意义的位置，这些位置应该是稀疏的
- Transformer 在计算注意力权重时，伴随着高计算量与空间复杂度。特别是在编码器部分，与特征像素点的数量成平方级关系，因此难以处理高分辨率的特征

所以 DETR 没有用多尺度特征，而是仅仅用了 backbone 最小输出特征，否则计算量非常大。

一个训练好的注意力模块应该是稀疏只关注特定区域的，因此 DeformableDETR 提出了多尺度可变形注意力模块，具体来说是：每个特征像素不必与所有特征像素交互计算，只需要与部分基于采样获得的其它像素“亲密接触”（交互）即可，这就大大加速了模型收敛，同时也降低了计算复杂度与所需的空间资源。另外，该模块能够很方便地应用到多尺度特征上，连FPN都不需要。

每个参考点仅关注邻域(包括多层特征图)的一组采样点，这些采样点的位置并非固定，而是可学习的（和可变形卷积一样），从而实现了一种局部(local)&稀疏(sparse)的高效注意力机制。

举个例子: 假设特征图维度是 100x100，那么 Transformer 输入序列长度就是 10000，内部计算自注意力时候有一个 10000x10000 的 attention 的计算过程，不仅计算量大而且收敛慢，现在对于 10000 个点，每个点学习 4 个 offset 点
每个点就只需要和这个 4 个点进行交互就可以了，这样计算量就大大减少了。实际上 attention 权重都不qk之间需要计算，作者直接用 q 通过一个 linear 层就得到了，计算量更小。

并且对于每个参考点也不仅仅是学习 4 个 offset，他还会跨多尺度层，也就是每个点会学习 4 层 x 4 个点，一共 16 个 offset，这样就相当于同时学到了跨多尺度的信息了，因此 FPN 层也不需要了。


## deformable-detr_r50_16xb2-50e_coco.py

backbone 输出 4 个尺度的特征，输出维度都是 256。mmdet 内部代码运行顺序是;

```text
img_feats & batch_data_samples
              |
              V
     +-----------------+
     | pre_transformer | transformer 运行前的预处理
     +-----------------+
         |          |
         |          V
         |    +-----------------+
         |    | forward_encoder | # encoder forward
         |    +-----------------+
         |             |
         |             V
         |     +---------------+
         |     |  pre_decoder  | # decoder 运行前的预处理
         |     +---------------+
         |         |       |
         V         V       |
     +-----------------+   |
     | forward_decoder |   | # decoder forward
     +-----------------+   |
               |           |
               V           V
              head_inputs_dict # 输入给 head 的字典
```

### pre_transformer
```python
...
encoder_inputs_dict = dict(
    feat=feat_flatten, # b,19947,256 # 19947 = 100x100 + 50x50 + 25x25 + 13x13, 多尺度特征拉伸而来
    feat_mask=mask_flatten, # b,19947 实际上是 key_padding_mask 参数，用于忽略 padding 的位置
    feat_pos=lvl_pos_embed_flatten, # b,19947,256 sincos2d 位置编码+ level embedding
    spatial_shapes=spatial_shapes, # 这些信息在计算 attention 的时候需要用到
    level_start_index=level_start_index,
    valid_ratios=valid_ratios) # 类似于 anchor-based 计算 Loss 时候的 vaild anchor，非常 trick
decoder_inputs_dict = dict(
    memory_mask=mask_flatten,
    spatial_shapes=spatial_shapes,
    level_start_index=level_start_index,
    valid_ratios=valid_ratios)
return encoder_inputs_dict, decoder_inputs_dict
```

### forward encoder  
```python
# forward encoder   
memory = self.encoder(
    query=feat, # 和上面名字一一对应
    query_pos=feat_pos,
    key_padding_mask=feat_mask,  # for self_attn
    spatial_shapes=spatial_shapes,
    level_start_index=level_start_index,
    valid_ratios=valid_ratios)
encoder_outputs_dict = dict(
    memory=memory,
    memory_mask=feat_mask,
    spatial_shapes=spatial_shapes)
return encoder_outputs_dict
```

重点要看一下 encoder

```python
encoder=dict(  # DeformableDetrTransformerEncoder
    num_layers=6,
    layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
        self_attn_cfg=dict(  # MultiScaleDeformableAttention
            embed_dims=256,
            batch_first=True),
        ffn_cfg=dict(
            embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
```

新加的 MultiScaleDeformableAttention 需要提供参考点坐标，对于 encoder 而已，实际上就是预定义的 anchor point

```python
# b,19947,4,2 - 4 个参考点
reference_points = self.get_encoder_reference_points(
    spatial_shapes, valid_ratios, device=query.device)
for layer in self.layers:
    query = layer(
        query=query,
        query_pos=query_pos,
        key_padding_mask=key_padding_mask,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        reference_points=reference_points,
        **kwargs)
return query


reference_points_list = []
for lvl, (H, W) in enumerate(spatial_shapes):
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            0.5, H - 0.5, H, dtype=torch.float32, device=device),
        torch.linspace(
            0.5, W - 0.5, W, dtype=torch.float32, device=device))
    ref_y = ref_y.reshape(-1)[None] / (
        valid_ratios[:, None, lvl, 1] * H)
    ref_x = ref_x.reshape(-1)[None] / (
        valid_ratios[:, None, lvl, 0] * W)
    ref = torch.stack((ref_x, ref_y), -1)
    reference_points_list.append(ref)
reference_points = torch.cat(reference_points_list, 1)
# [bs, sum(hw), num_level, 2]
reference_points = reference_points[:, :, None] * valid_ratios[:, None]
```

### MultiScaleDeformableAttention
最核心模块: 

```python
class MultiScaleDeformableAttention(BaseModule):
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        # 类似可变型卷积，学习基于输入参考点的偏移量，参考点可以是我们说的 anchor point，也可以是 rpn 等，
        self.sampling_offsets = nn.Linear(
              embed_dims, num_heads * num_levels * num_points * 2)
        
        # 可以看出实际上对于每个点，学习的不只是当前层的 4 个 offset，还包括 4 个多尺度层，实现了跨尺度功能
        self.attention_weights = nn.Linear(embed_dims,
                                  num_heads * num_levels * num_points)
        
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        # 这个初始化也很有讲究
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query: torch.Tensor, # b,n,256
                key: Optional[torch.Tensor] = None, # b,n,256
                value: Optional[torch.Tensor] = None, # b,m,256
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None, # b,n,256
                key_padding_mask: Optional[torch.Tensor] = None, # b,n
                reference_points: Optional[torch.Tensor] = None, # 外部提供参考点，长度和序列长度要一致 b,n,4,2
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos # 相加即可
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
            
        value = value.view(bs, num_value, self.num_heads, -1)
        # 通过 query 预测 offset 
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # 通过 query 预测 attention weights, 不需要通过 key 和  q 计算了，更暴力
        # b,n,8,16
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        # 参考点加上 offset 得到归一化采样点坐标
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            # 
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

```

来看一下 pytorch 实现

```python
output = multi_scale_deformable_attn_pytorch(
    value, spatial_shapes, sampling_locations, attention_weights)

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, # b,n,8,256
        value_spatial_shapes: torch.Tensor, # 4, 2， 4 是 4 个 level
        sampling_locations: torch.Tensor, # b,n,8,4,4,2 4 个层，每个层 4 个采样点
        attention_weights: torch.Tensor) -> torch.Tensor: # b,n,8,4,4
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    # 将之前 flatten 的 value 恢复成原来的 4 个尺度
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    # 遍历每个输出尺度
    for level, (H_, W_) in enumerate(value_spatial_shapes):

        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        # 对 value 进行双线性插值，b, c, h, w
        sampling_value_l_ = F.grid_sample(
            value_l_, # bx8, 32, h, w
            sampling_grid_l_, # bx8, hw, 4, 2
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_) #  bx8, 32, hw, 4
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points) # bx8, 1, hw, 16
    
    # (bx8,32,hw,16) X (bx8, 1, hw, 16) -> bx8, 32, hw, 16 -> b,8x32,hw 
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()
```        

编码器部分就分析完了。接下来是 pre_decoder

### pre_decoder

```python
enc_outputs_class, enc_outputs_coord = None, None

query_embed = self.query_embedding.weight

query_pos, query = torch.split(query_embed, c, dim=1)
query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)

query = query.unsqueeze(0).expand(batch_size, -1, -1)

# 解码器的参考点不是预设的，而是学习出来的，300 个 query 那么这个参考点维度就是 300x2
reference_points = self.reference_points_fc(query_pos).sigmoid()

decoder_inputs_dict = dict(
    query=query, # bx300x256
    query_pos=query_pos, # bx300x256
    memory=memory, # b,hw,256
    reference_points=reference_points) # bx300x2
head_inputs_dict = dict( # 如果不是 two-stage，就不需要这两个
    enc_outputs_class=enc_outputs_class,
    enc_outputs_coord=enc_outputs_coord) if self.training else dict()
return decoder_inputs_dict, head_inputs_dict
```

### forward_decoder

接下来是 forward_decoder

```python
# 网络学习是基于参考点的，因此中间的可学习参考点都需要返回，不然多级监督就没法做了
# 不过如果没有开启 with_box_refine，那么中间参考点是不会变的，始终和初始参考点一样
inter_states, inter_references = self.decoder(
    query=query, # bx300x256
    value=memory, # b,hw,256
    query_pos=query_pos,
    key_padding_mask=memory_mask,  # for cross_attn
    reference_points=reference_points, # bx300x2
    spatial_shapes=spatial_shapes,
    level_start_index=level_start_index,
    valid_ratios=valid_ratios,
    reg_branches=self.bbox_head.reg_branches
    if self.with_box_refine else None) # 先不考虑 with_box_refine 参数，此时 inter_references 和 reference_points完全一样
references = [reference_points, *inter_references]
decoder_outputs_dict = dict(
    hidden_states=inter_states, references=references)
return decoder_outputs_dict
```

### DeformableDETRHead

此时就完成了 Transformer 的前向传播，接下来就是 head 的前向传播了

```python
def forward(self, hidden_states: Tensor,
            references: List[Tensor]) -> Tuple[Tensor]:
    all_layers_outputs_classes = []
    all_layers_outputs_coords = []
    # 6 层 decoder 就有 6 个输出
    for layer_id in range(hidden_states.shape[0]):
        
        # references 之前是归一化的，现在要还原
        reference = inverse_sigmoid(references[layer_id])
        
        # NOTE The last reference will not be used.
        hidden_state = hidden_states[layer_id]
        
        # 不同层的cls_branches和reg_branches可以通过参数设置为共享或者不共享
        outputs_class = self.cls_branches[layer_id](hidden_state)
        tmp_reg_preds = self.reg_branches[layer_id](hidden_state) # b,300, 4
        if reference.shape[-1] == 4:
            # When `layer` is 0 and `as_two_stage` of the detector
            # is `True`, or when `layer` is greater than 0 and
            # `with_box_refine` of the detector is `True`.
            tmp_reg_preds += reference
        else:
            # When `layer` is 0 and `as_two_stage` of the detector
            # is `False`, or when `layer` is greater than 0 and
            # `with_box_refine` of the detector is `False`.
            assert reference.shape[-1] == 2
            tmp_reg_preds[..., :2] += reference  # cxcywh 格式
        outputs_coord = tmp_reg_preds.sigmoid()
        all_layers_outputs_classes.append(outputs_class)
        all_layers_outputs_coords.append(outputs_coord)
    all_layers_outputs_classes = torch.stack(all_layers_outputs_classes) # 6，b,300, 80
    all_layers_outputs_coords = torch.stack(all_layers_outputs_coords) # 6，b,300, 4
    return all_layers_outputs_classes, all_layers_outputs_coords
```

后面 loss 计算过程和 DETR 完全一样了。

## deformable-detr-refine_r50_16xb2-50e_coco
refine the references in the decoder

```python
bbox_head['share_pred_layer'] = False
```

如果是 refine 模式，那么分类回归头权重必须要不共享，非 refine 模式是共享的。

差异主要在于 DeformableDetrTransformerDecoder 中的 forward 函数

```python
# decoder 运行完成后
output = layer(
    output,
    query_pos=query_pos,
    value=value,
    key_padding_mask=key_padding_mask,
    spatial_shapes=spatial_shapes,
    level_start_index=level_start_index,
    valid_ratios=valid_ratios,
    reference_points=reference_points_input,
    **kwargs)

if reg_branches is not None:
    # 利用 reg_branches 对参考点进行修正
    tmp_reg_preds = reg_branches[layer_id](output)
    if reference_points.shape[-1] == 4:
        new_reference_points = tmp_reg_preds + inverse_sigmoid(
            reference_points)
        new_reference_points = new_reference_points.sigmoid()
    else:
        assert reference_points.shape[-1] == 2
        new_reference_points = tmp_reg_preds
        new_reference_points[..., :2] = tmp_reg_preds[
            ..., :2] + inverse_sigmoid(reference_points)
        new_reference_points = new_reference_points.sigmoid()
        
    reference_points = new_reference_points.detach() # 要 detach，不然会影响 reg_branches 梯度
    
if self.return_intermediate:
    intermediate.append(output)
    intermediate_reference_points.append(reference_points)
```

注意 decoder 有 6 个，参考点实际上是 7 个，第一个是初始参考点，最后一个参考点不需要。

## deformable-detr-refine-twostage_r50_16xb2-50e_coco

as_two_stage (bool, optional): Whether to generate the proposal from the outputs of encoder. Defaults to `False`.

这个模式是将 encoder 比作 rpn 网络，decoder 比作 roi 网络了。

The last prediction layer is used to generate proposal

bbox_head['num_pred_layer'] = decoder['num_layers'] + 1 额外加一个预测层

在这个模式下输入到 Decoder 的参考点和 object query & query embedding 会有所不同，不再是直接通过 embeding 得到

```python
if not self.as_two_stage:
    # 不需要
    self.query_embedding = nn.Embedding(self.num_queries,
                                        self.embed_dims * 2)
    # NOTE The query_embedding will be split into query and query_pos
    # in self.pre_decoder, hence, the embed_dims are doubled.

    
if self.as_two_stage:
    self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
    self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
    
    self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                  self.embed_dims * 2)
    self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
else:
    self.reference_points_fc = nn.Linear(self.embed_dims, 2) # decoder 参考点 fc 层
```

pre_decoder 处理有所不同

```python
if self.as_two_stage:
    # output_proposals 实际上就是 anchor point
    output_memory, output_proposals = \
        self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
    
    # 额外的 cls_branches 和 reg_branches 当做 encoder rpn 的预测头
    # b,hw,80
    enc_outputs_class = self.bbox_head.cls_branches[
        self.decoder.num_layers](
            output_memory)
    
    # 基于 anchor point 预测的新的 proposals，最后维度是 4
    enc_outputs_coord_unact = self.bbox_head.reg_branches[
        self.decoder.num_layers](output_memory) + output_proposals
    enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
    # We only use the first channel in enc_outputs_class as foregroun
    # the other (num_classes - 1) channels are actually not used.
    # Its targets are set to be 0s, which indicates the first
    # class (foreground) because we use [0, num_classes - 1] to
    # indicate class labels, background class is indicated by
    # num_classes (similar convention in RPN).
    # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet
    # This follows the official implementation of Deformable DETR.
    # 取 topk 个 proposals
    topk_proposals = torch.topk(
        enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
    topk_coords_unact = torch.gather(
        enc_outputs_coord_unact, 1,
        topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
    topk_coords_unact = topk_coords_unact.detach()
    reference_points = topk_coords_unact.sigmoid() # b,300,4
    
    # 基于 topk_coords_unact 生成 query_pose 和 query
    pos_trans_out = self.pos_trans_fc(
        self.get_proposal_pos_embed(topk_coords_unact))
    pos_trans_out = self.pos_trans_norm(pos_trans_out)
    
    query_pos, query = torch.split(pos_trans_out, c, dim=2)
else:
    # 
    enc_outputs_class, enc_outputs_coord = None, None
    query_embed = self.query_embedding.weight
    # 切分而来
    query_pos, query = torch.split(query_embed, c, dim=1)
    query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
    query = query.unsqueeze(0).expand(batch_size, -1, -1)
    # 直接生成
    reference_points = self.reference_points_fc(query_pos).sigmoid()

decoder_inputs_dict = dict(
    query=query,
    query_pos=query_pos,
    memory=memory,
    reference_points=reference_points)
# enc_outputs_class 和 enc_outputs_coord 相当于 rpn head 是需要训练的，因此需要输入给后面的 head
head_inputs_dict = dict(
    enc_outputs_class=enc_outputs_class,
    enc_outputs_coord=enc_outputs_coord) if self.training else dict()
return decoder_inputs_dict, head_inputs_dict
```

以上就是全部内容。可以发现实现的非常巧妙和 trick。

# ConditionalDETR

https://zhuanlan.zhihu.com/p/401916664

针对 DEtection Transformer (DETR) 训练收敛慢的问题(需要训练500 epoch才能获得比较好的效果) 提出了conditional cross-attention mechanism，通过 conditional spatial query 显式地寻找物体的 extremity 区域，从而缩小搜索物体的范围，加速了收敛。结构上只需要对 DETR 的 cross-attention 部分做微小的改动，就能将收敛速度提高 6~10 倍。

对 DETR 的 decoder cross-attention 中 attention map 的可视化。我们可以看到，DETR decoder cross-attention 里的 query 查询到的区域都是物体的 extremity 区域

我们认为，DETR 在计算 cross-attention 时，query 中的 content embedding 要同时和 key 中的 content embedding 以及 key 中的 spatial embedding 做匹配，这就对 content embedding 的质量要求非常高。而训练了 50 epoch 的DETR，因为 content embedding 质量不高，无法准确地缩小搜寻物体的范围，导致收敛缓慢。所以用一句话总结 DETR 收敛慢的原因，就是 DETR 高度依赖高质量的 content embedding 去定位物体的 extremity 区域，而这部分区域恰恰是定位和识别物体的关键。

为了解决对高质量 content embedding 的依赖，我们将 DETR decoder cross-attention 的功能进行解耦，并提出 conditional spatial embedding。Content embedding 只负责根据外观去搜寻跟物体相关的区域，而不用考虑跟 spatial embedding 的匹配; 对于 spatial 部分，conditional spatial embedding 可以显式地定位物体的 extremity 区域，缩小搜索物体的范围。

```python
self.encoder = DetrTransformerEncoder(**self.encoder)
self.decoder = ConditionalDetrTransformerDecoder(**self.decoder)
```

可以看到主要是改了下 decoder，具体推理时候 forward_decoder

```python
hidden_states, references = self.decoder(
    query=query,
    key=memory,
    query_pos=query_pos,
    key_pos=memory_pos,
    key_padding_mask=memory_mask)
head_inputs_dict = dict(
    hidden_states=hidden_states, references=references)
return head_inputs_dict
```

相比 DETR 多了一个参考点 references 的输出。

对于自注意力层，实际上和 DETR 没有区别，代码写的是和交叉注意力统一而已

```python
# 都是线性投影层
# query 认为是 content，query_pos 认为是 spatial
q_content = self.qcontent_proj(query)
q_pos = self.qpos_proj(query_pos)

k_content = self.kcontent_proj(query)
k_pos = self.kpos_proj(query_pos)

v = self.v_proj(query) # query就是论文中的 decoder embedding
q = q_content if q_pos is None else q_content + q_pos
k = k_content if k_pos is None else k_content + k_pos
# 内部和 DETR 一样
sa_output = self.forward_attn(
    query=q,
    key=k,
    value=v,
    attn_mask=attn_mask,
    key_padding_mask=key_padding_mask)[0]
query = query + self.proj_drop(sa_output)
```

对于 cross-attention 层，需要将 query 和 key 都解耦为 content 和 spatial，对于 spatial 显示引入可学习的参考点

```python
q_content = self.qcontent_proj(query)
k_content = self.kcontent_proj(key) # 来自 encoder 的输出
v = self.v_proj(key)

bs, nq, c = q_content.size()
_, hw, _ = k_content.size()
k_pos = self.kpos_proj(key_pos)
if is_first or self.keep_query_pos:
    q_pos = self.qpos_proj(query_pos)
    q = q_content + q_pos
    k = k_content + k_pos
else:
    q = q_content
    k = k_content

q = q.view(bs, nq, self.num_heads, c // self.num_heads)

# ref_sine_embed 是来自 query 经过 MLP 得来，并经过了 sincos embedding 处理，并不是坐标
# reference_unsigmoid = self.ref_point_head(
#            query_pos)  # [bs, num_queries, 2]
# reference = reference_unsigmoid.sigmoid()
# ref_sine_embed = coordinate_to_encoding(coord_tensor=reference)
query_sine_embed = self.qpos_sine_proj(ref_sine_embed) # 来自外部参考点

query_sine_embed = query_sine_embed.view(bs, nq, self.num_heads,
                                         c // self.num_heads)
# concat，也就是分成了 content 和 spatial 两部分
q = torch.cat([q, query_sine_embed], dim=3).view(bs, nq, 2 * c)
k = k.view(bs, hw, self.num_heads, c // self.num_heads)

# key 要分成 content 和 spatial 两部分
k_pos = k_pos.view(bs, hw, self.num_heads, c // self.num_heads)
k = torch.cat([k, k_pos], dim=3).view(bs, hw, 2 * c)

ca_output = self.forward_attn(
    query=q,
    key=k,
    value=v,
    attn_mask=attn_mask,
    key_padding_mask=key_padding_mask)[0]
query = query + self.proj_drop(ca_output)
```

仔细看论文图，实际上是对应的

<div align=center>
<img src="https://github.com/Atten4Vis/ConditionalDETR/raw/main/.github/conditional-detr.png" width="48%"/>
</div>

# DAB-DETR

https://zhuanlan.zhihu.com/p/560513044
https://arxiv.org/pdf/2201.12329v4.pdf

DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR

可以看出显式的引入了 anchor 的概念，相比 ConditionalDETR 更加解耦，可解释性更好。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/18b17bf5-6f87-4a5d-8d17-abf5e117e2ad">
</div>

详细结构：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/3b05e38e-ef6a-4a8b-9d90-24a71b367ef5">
</div>

看起来是 ConditionalDETR + DeformableDETR 中的 iter_refine_bbox 模式

query_pos 只是设置 b,n,4 维度，而不是之前说的 256 无法解释的维度。也就是论文图片中的 anchor bboxes

# DINO

DINO=DAB-DETR+DN-DETR

DAB-DETR是在思考DETR query理解的问题。它直接把DETR的 query pos 显示地建模为四维的框，同时每一层decoder中都会去预测相对偏移量并去更新检测框，得到一个更加精确的检测框预测，动态更新这个检测框并用它来帮助 decoder cross-attention 来抽取feature。
DN-DETR是在思考DETR中的二分图匹配问题，或者说标签分配问题。我们发现DETR中的二分匹配在早期十分不稳定，这会导致优化目标不一致引起收敛缓慢的问题。因此，我们使用一个 denoising task 直接把带有噪声的真实框输入到decoder中，作为一个shortcut来学习相对偏移，它跳过了匹配过程直接进行学习

需要先简单了解 DN-DETR。提出了全新的去噪训练(DeNoising training)解决DETR decoder二分图匹配 （bipartite graph matching）不稳定的问题，可以让模型收敛速度翻倍，并对检测结果带来显著提升（+1.9AP）。该方法简易实用，可以广泛运用到各种DETR模型当中，以微小的训练代价带来显著提升。

DETR把目标检测做成了一个set prediction的问题，并利用匈牙利匹配（Hungarian matching）算法来解决decoder输出的objects和ground-truth objects的匹配。因此，匈牙利算法匹配的离散性和模型训练的随机性，导致ground-truth的匹配变成了一个动态的、不稳定的过程。

我们可以把decoder看成在学习两个东西： 可学习 anchor+ 可学习 offset。

decoder queries可以看成是anchor位置的学习，而不稳定的匹配会导致不稳定的anchor，从而使得相对偏移的学习变得困难。因此，我们使用一个denoising task作为一个shortcut来学习相对偏移，它跳过了匹配过程直接进行学习。如果我们把query像上述一样看作四维坐标，可以通过在真实框附近添加一个微小的扰动作为噪声，这样我们的denoising task就有了一个清晰的目标--直接重建真实框而不需要匈牙利匹配。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/43cff0b7-d352-471e-8f56-b7b88dcecc08">
</div>

DN 是一种训练方法，类似 Faster RCNN 中的 RCNN 随机采样 add_gt_as_proposals 参数，也是为了早期稳定收敛。

将之前的 Decoder embedding 分成2部分，一部分和之前一样，然后额外加入一些对 gt 加了噪声的 query，并记录下 indictor 标记，当然 query pos 也可以加随机噪声。 在训练时候对于这些噪声的 query，不计算需要进行二分匹配，直接和 GT 计算 loss，其余部分和之前一样训练。

核心类是 CdnQueryGenerator

pre_decoder 核心代码如下：

```python
if self.training:
    # 有一个额外的加噪声的 query 生成过程
    dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
        self.dn_query_generator(batch_data_samples)
    # 和之前的 query 拼接起来
    query = torch.cat([dn_label_query, query], dim=1)
    # query pos 也是一样处理
    reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                 dim=1)
else:
    # 测试时候流程不变
    reference_points = topk_coords_unact
    dn_mask, dn_meta = None, None
```

dn_mask 是用于区分哪些是加了 DN 的

# Mask DINO

CVPR2023

https://github.com/IDEAResearch/MaskDINO

Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation

基于 DINO 和 Mask2Former 并拓展到检测和通用图像分割领域。

首先分析了 DINO 直接做分割和 Mask2Former 直接做目标检测的问题，然后给出了几个解决方案。

Mask DINO 做通用图像分割的做法是和 Mask2Former 一样，但是额外针对检测和 mask 任务进行了一些设计。

作者主要是基于以下两个问题进行了改进：

1）为什么不能检测和分割任务在基于 Transformer 的模型中相互帮助
2) 是否有可能开发一种统一的架构来取代专门的架构？

作者首先尝试对 DINO 和 Mask2Former 进行简单扩展，使其可以同时完成检测和分割任务，发现效果都不行。

作者觉得主要原因还是任务的不同，一个是关注像素级别，一个是区域级别。

Mask2Former中的query只需要比较每像素与图像特征的相似性，它们可能不知道每个实例的区域级别位置。相反，DINO 中的query并非旨在与这种低级特征交互以学习像素级表示。相反，它们编码丰富的位置信息和高级语义进行检测。

Why cannot Mask2Former do detection well?

1. 它的查询遵循 DETR中的设计，而无需像条件 DETR、Anchor DETR 和 DABDETR 中研究的那样利用更好的位置先验
2. Mask2Former 在 Transformer 解码器中采用了掩码注意（带有注意掩码的多头注意力）。从具有高分辨率的前一层预测的注意掩码用作注意力计算的硬约束。它们既不高效也不灵活用于框预测
3. Mask2Former 不能逐层显式执行框细化。

Why cannot DETR/DINO do segmentation well?

1. DETR的分割头不是最优的。DETR 让每个查询嵌入和具有最小特征图的点积来计算注意力图，然后对它们进行上采样以获得掩码预测。这种设计缺乏来自主干的查询和更大的特征图之间的交互。也就是没有 pixdecoder
2. 改进的检测模型中的特征与分割不一致即特征不对齐

总而言之，就是 DINO 和 Mask2Former 都是对特定领域进行了大量专门优化，如果只是简单的扩展任务而没有特别设计，性能不会最好。需要一个能够将两者优点结合并能够对齐特征的操作。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/c7e13dfa-8551-4cad-9a1c-4b0816997efb">
</div>

红色的地方表示是相比于 DINO 的改进，主要是3个模块。也引入了 pix decoder。

**(1) 改进1：Unified and Enhanced Query Selection**

原先的 DINO encoder 预测头输出是 cls 和 bbox 预测，现在多了额外的 mask 预测，然后 Query Selection 就可以利用预测的 mask 进行增强。

每个token的分类分数被认为是选择排名靠前的特征并将它们馈送到解码器作为内容查询的置信度。所选特征还使用高分辨率特征图回归框和点积来预测掩码。

请注意，我们在 Mask DINO 中初始化 content 和 anchor pos query，而 DINO 仅初始化 pos query, 其中 content 是直接来自 query_embedding，而不是来自 topk mem 输出值。

在统一查询选择之后的初始阶段，掩码预测比框更准确。因此，在统一的查询选择之后，我们从预测掩码中导出框作为解码器更好的锚点框初始化。通过这种有效的任务合作，增强的框初始化可以大大提高检测性能。

模型推理部分改动就是这个，其余两个改动是训练相关的。

**(2) 改进二： Unified denoising for mask**

由于掩码可以被视为框的更细粒度的表示，因此框和掩码是自然连接的。因此，我们可以将框视为掩码的噪声版本，并训练模型预测给定框的掩码作为去噪任务。给定用于掩码预测的框也被随机噪声，以实现更有效的掩码去噪训练。

**(3) Hybrid matching**

我们添加了一个掩码预测损失来鼓励更准确和一致匹配。实际上就是匹配代价考虑了 mask。

## 源码分析

MaskDINO 目前开源了三个任务：全景分割，实例分割和语义分割，需要单独训练。如果只用检测数据集那就是 DINO 了，联合检测和分割数据训练，作者没有提供配置。

一个需要注意的细节：为了统一三个不同任务，让其只是在数据层面有差异，其余部分没有差异。作者设计的 loss 始终是 mask loss + bbox loss + cls loss，
也就是说即使是语义分割，也会强行计算 gt bbox，然后进行监督。这个是否合理有待商榷？ mask2former 是没有考虑 bbox 监督的。对于全景分割，stuff seg 标注也要强行计算 bbox，然后计算 bbox loss。
作者如此设计一来是想统一整个架构，而来是觉得检测和分割可以共同提升，但是事实真的是这样吗？


# DDQ
CVPR2023    
Dense Distinct Query for End-to-End Object Detection  
https://zhuanlan.zhihu.com/p/631607906  官方解读


# CO-DETR

ICCV2023

https://arxiv.org/pdf/2211.12860.pdf
https://github.com/Sense-X/Co-DETR

DETRs with Collaborative Hybrid Assignments Training

在DETR中分配为正样本的查询太少，具有一对一集匹配会导致对编码器输出的稀疏监督，这大大损害了编码器和解码器的判别特征学习。 我们提出了一种新的协作混合分配训练方案，即Co-DETR，从通用的标签分配方式中学习更高效和有效的基于DETR的检测器。

这种新的训练方案可以通过训练由 ATSS、FCOS 和 Faster RCNN 等一对多标签分配监督的多个并行辅助头来轻松增强编码器在端到端检测器中的学习能力。此外，我们通过从这些辅助头中提取正坐标来进行额外的定制正查询，以提高解码器中正样本的训练效率。在推理中，这些辅助头被丢弃，因此我们的方法不需要对原始检测器引入额外的参数和计算成本，而不需要手工制作的非最大抑制(NMS)。

作者通过一些分析，发现核心原因在于分配为正样本的查询太少，因此核心还是希望加入更多的正样本 query。因此引入了 one-to-many 的匹配方式，并且不能抛弃不需要nms这个特点，所以是以辅助头的形式加入的。

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/81edbb1d-b98d-4b0e-bcde-5feea67991ab">
</div>

全文思想和做法非常简单，但是因为用到的模块比较多。

(1) 图片输入到 resnet50 中提取多尺度特征
(2) 将多尺度特征输入到 deformable detr encoder 中进行特征提取
(3) 将输出的特征输入到 deformable detr decoder 中进行 bbox 和 cls 预测，此次采用的是 one-to-one 匹配规则
(4) 参考 vitdet，将输出的特征进行变换即输入到 ms adapter 里面得到多尺度特征
(5) 将多尺度特征直接输入到 one-stage head 例如 atss head 里面进行 cls 和 bbox 预测，这个地方采用正常的 atss loss 训练即可
(6) 将多尺度特征直接输入到 two-stage rpn 和 roi head 里面进行 cls 和 bbox 预测
(7) 将各个辅助 head 预测的 bbox 经过处理作为高质量正样本 query 再次输入到 deformable detr decoder 中进行 bbox 和 cls 预测，此时不需要再次匹配，直接预测即可

辅助 head 可以是任意类型，也可以是任意多个，可以是 one-stage 也可以是 two-stage 混合等等。

推理时候可以只需要 deformable detr decoder head，而不需要辅助 head。当然如果你想验证辅助 head 性能，也可以的。

总体来看，思想还是比较好理解的，就是如果用了非常多的辅助 head，训练成本会增加不少，而且因为引入了第 7 步骤，代码复杂度会增加不少。

测试了 detr one-stage 和 two-stage， 为啥 two-stage 性能比前面两个差了很多，前面两个是 52.1 49.4，最后这个只有 47.9，这个是为啥呢？
原因是： faster rcnn 本身的性能就比 atss 低 2 个点，所以可能会出现这个问题。不过原则上 LSJ 情况 faster rcnn 性能应该和 atss 类似才是。估计是 faster rcnn baseline 太低或者说 lsj 下需要特别设置下参数。
经查证，不管再啥设置下，atss 始终比 faster rcnn 高 2 个点。


# Cascade-DETR

iccv2023

Cascade-DETR: Delving into High-Quality Universal Object Detection
https://arxiv.org/abs/2307.11035

# Focus-DETR

ICCV2023

Less is More: Focus Attention for Efficient DETR
https://arxiv.org/abs/2307.12612

meishayis

# RefineBox

Enhancing Your Trained DETRs with Box Refinement  
https://arxiv.org/pdf/2307.11828.pdf

可以接任意的 DETR 类检测器，然后进行 refine，从而提高性能。采用的是 two-stage 思想。

一个亮点是，检测器不需要训练，只需要训练 FPN 和 refine 模块，算是一个插件，比较方便

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/8d9603f4-c94d-4057-a25f-6ed2a039649b">
</div>

完整结构图如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/a08901ef-666d-46d1-b1aa-a6915816995c">
</div>

训练成本其实也不算少，是训练 DAB-DETR 的一半。

# ViTDet

Exploring Plain Vision Transformer Backbones for Object Detection

https://zhuanlan.zhihu.com/p/548147089

#  ViT-Adapter 系列

# Group DETR 和 Group DETR V2

# Align DETR

# H-DETR

# Anchor DETR


# DETR 应用于密集场景

DDQ

DETR 研究分支，包括预训练。

# Mask Frozen-DETR

Mask Frozen-DETR: High Quality Instance Segmentation with One GPU

https://arxiv.org/pdf/2308.03747.pdf

# FoLR

Focus on Local Regions for Query-based Object Detection
https://arxiv.org/abs/2310.06470

不需要 encoder，但是并没有和 DINO 等比，性能也没有他高


