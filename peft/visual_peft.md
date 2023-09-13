## LLaMA-Adapter

论文： [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/pdf/2303.16199.pdf)

### 原理

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236591154-bc2f75f8-4acb-43e5-b86b-0d376d796faf.png"/>
</div>

从原理图上看没有特别大的区别，也是在自注意力模块处加了可学习的虚拟 token，并且为了保证初始化时候不破坏原始信息流，会设计为 zero-init 初始化，特别的是设置了一个 zero-gate 门控模块。

并且不是所有 transformer 层都会加入虚拟 token，只是高层会加入，这是一个超参数。

同时还构造了一个多模态版本，可以支持图片和文本输入，输出依然是文本。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236591372-aaecf273-5382-4c89-b70d-e1e9dd075bc0.png"/>
</div>

### 实践

实现上也比较简单，将要换到的 attention 层全部替换为 AdaptedAttention 即可

```python
class AdaptedAttention(nn.Module):
    def __init__(self, model_type: str, adapter_len: int, model):
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model # 这个是被替换前的 attention 模块
        self.adapter_len = adapter_len
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        self.adaption_prompt = nn.Parameter(
            torch.empty(1, adapter_len, self.model.hidden_size, device=device).normal_()
        )
        # Initialize the gate to 0 as this is "zero-init".
        self.adaption_gate = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, **kwargs):
        # 原先模块正常运行
        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        # 获取到原先模块中的 layer
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        
        # 利用原先模块，对 adaption 进行投影计算
        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_k = (
            key.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_v = (
            value.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )

        # Recompute query states.
        # 因为目前的设计中没有返回 q 的值，因此需要重新对原始模块计算一下
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        # (bsz, num_heads, q_len, head_dim)
        query_states = compute_query_states(model=self.model, **kwargs)
        
        # 以下计算需要对照公式，具体也是简单的 attention 计算过程
        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3)) / math.sqrt(self.model.head_dim)
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        # (bsz, q_len, hidden_size)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)
        
        # 构成残差
        # Add adaption prompt output to original output.
        output = output + adapter_output
        return output, None, past_key_value
```

## LLaMA-Adapter v2

LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model

重点对 LLaMA-Adapter 中的 图文多模态模型进行更新。
