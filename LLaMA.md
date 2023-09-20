# LLaMA

论文题目： LLaMA: Open and Efficient Foundation Language Models
开源地址：https://github.com/facebookresearch/llama

LLaMA 的火热程度以及重要性，我想不需要多说一个字。虽然我们无法复现，但是了解他的网络结构设计还是关键的。

官方号称： LLaMA-13B outperforms GPT-3 (175B) on most benchmarks 这个就不得了。

## 模型结构和其他

LLaMA 是一个自回归模型, 模型超参如下所示：

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/17425982/5d5888b7-1fc0-4271-9450-3d86432e1a10"/>
</div>

预训练数据如下：

<div align=center>
<img src="https://github.com/open-mmlab/playground/assets/17425982/01a9c90f-7e30-4f97-ae51-3a610cc6f2ea"/>
</div>

只用了公开数据，没有私有数据。

Tokenizer： BPE 分词算法，具体是采用了 SentencePiece 这个库。SentencePiece 是一个开源的分词工具，实现了多种分词算法，其中包括 BPE 算法。

模型结构作者做了不少改动。

1. Pre-normalization [GPT3]： 为了提高训练的稳定性，我们对每个 transformer 子层的输入进行归一化，而不是对输出进行归一化。我们使用 RMSNorm 归一化函数，RMSNorm是对LayerNorm的一个改进，没有做re-center操作（移除了其中的均值项），可以看作LayerNorm在均值为0时的一个特例
2. SwiGLU activation function [PaLM]：
3. Rotary Embeddings [GPTNeo]: 移除绝对位置嵌入，取而代之的是添加旋转位置嵌入(RoPE)。通过绝对位置编码的方式实现相对位置编码，苏神做的，具有良好的外推性 https://zhuanlan.zhihu.com/p/359502624

我们的模型使用AdamW优化器进行训练，具有以下超参数:β1 = 0.9， β2 = 0.95。我们使用余弦学习率 schedule，这样最终的学习率等于最大学习率的10%。我们使用0.1的权重衰减和1.0的梯度裁剪,2000 warmup

Efficient implementation

1. 我们使用 causal multi-head attention 的实现来减少内存使用和运行时间。该实现可在 xformers 库中获得
2. 我们通过检查点减少了在向后传递期间重新计算的激活量。我们保存了计算成本高的激活，比如线性层的输出。这是通过手动实现transformer层的backward函数来实现的，而不是依赖于PyTorch的autograd
3. 我们还尽可能地重叠激活的计算和gpu之间通过网络的通信(由于toall_reduce操作)

在训练65b参数模型时，我们的代码在2048 A100 GPU和80GB RAM上处理大约380个token/秒/GPU。这意味着对包含1.4T令牌的数据集进行训练大约需要21天。

## 简单看下源码

https://github.com/facebookresearch/llama/blob/main/llama/

```python
from sentencepiece import SentencePieceProcessor
class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
```

```python
@dataclass
class ModelArgs:
    dim: int = 512
    ...
    max_seq_len: int = 2048 # 最大输入 token 2048
```

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # 相乘
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
```

````python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
    
    # 自注意力模块
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        # 旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 因为这是自回归模型，每次都是基于之前预测来预测下一个词
        # 假设开始输入： i love my country and, 预测出 i
        # 下一次输入是： i love my country and i, 预测出 love
        # 为了节省计算量，不需要再次计算 i love my country and，直接取出来就行，作者使用 cache_k 和 cache_v 来存储
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)
````

```python
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) # 激活函数好像是 silu ？
```

ColumnParallelLinear 和 Linear 都是神经网络中常用的层类型，但是它们的计算方式略有不同。

Linear 层是全连接层，将输入张量与权重矩阵相乘，再加上偏置向量，最后通过激活函数输出结果。这个过程通常是在单个设备上完成的，比如 CPU 或 GPU。

而 ColumnParallelLinear 层是用于分布式训练的一种特殊层类型。在分布式训练中，每个设备只能访问一部分权重矩阵，因此需要将输入张量在列（column）维度上进行分割，并分配给不同的设备进行计算。在 ColumnParallelLinear 层中，权重矩阵在列维度上被分割，并分配给不同的设备。每个设备计算自己所分配到的权重子矩阵与输入张量的乘积，然后将结果通过横向拼接（concatenate）的方式合并起来。最后再加上偏置向量并通过激活函数输出结果。

因此，ColumnParallelLinear 层和 Linear 层在计算方式上有所不同，主要是为了适应分布式训练的需求。

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

完整推理过程：

```python
def forward(self, tokens: torch.Tensor, start_pos: int):
        # seqlen=1 表示不是第一次运行
        # =1 表示是第一次运行
        _bsz, seqlen = tokens.shape
        
        h = self.tok_embeddings(tokens)
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            # 第一次运行构建 mask，后续运行每次都是一个 token 输入，而非一个序列
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
```

最外层生成过程

```python
class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int, # 最大生成长度，为啥不是自动停止？
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # 没有 batch 算
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens]) # batch 里面最小的 prompt 长度
        max_prompt_size = max([len(t) for t in prompt_tokens])  # batch 里面最大的 prompt 长度
        
        # 最大不超过 2048
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        
        # 自己 padding 
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
            
        input_text_mask = tokens != self.tokenizer.pad_id
        
        # 这个地方稍微有点奇怪，正常应该是输入最长的 batch 序列+ mask 的，作者是输入了最短序列，这样可以不提供 mask
        # 在后面，如果当前预测后发现不是 Padding 位置，说明其实不用预测，直接用当前的 token 代替就行
        start_pos = min_prompt_size 
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            
            # 预测下一个 token
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p) # 采用
            else:
                next_token = torch.argmax(logits, dim=-1)
                
            next_token = next_token.reshape(-1)
            # 如果还不需要预测，则还是用当前 token 代替，而非预测
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            
            # 这样的话，tokens[:, prev_pos:cur_pos] 每次就只输入了一个 token
            # 但是作者输入了 prev_pos，而且内部有缓存，也实现了类似功能
            prev_pos = cur_pos
        
        # 解码整个序列
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
```

```python
# 以一定概率选择 top 预测
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
```

# LLAMA2

