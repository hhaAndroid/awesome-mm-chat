# Tokenizer

见 `tokenizer` 文件夹


# 张量并行线性层

## ColumnParallelLinear

<div align=center>
<img src="https://github.com/user-attachments/assets/127c8fd2-cac4-4867-8963-7ba44f9ca774"/>
</div>

对权重进行列切分，也就是 W, shape 是 (out_dim, in_dim)，变成 [W1, W2, W3...]

- 将输入通过 f(恒等变换) 函数分发到每个模型并行 rank 内部
- 每个 rank 内部的模型并行线性层计算，得到部分输出
- 通过 g(all gather) 操作 concat 到一起，得到最终输出
- 反向传播时候，g 函数的梯度是 split 后原样返回，f 函数的梯度是 (all reduce) add 操作

## RowParallelLinear

<div align=center>
<img src="https://github.com/user-attachments/assets/5958d7f1-0b89-4364-88d3-6012f7cf9f0d"/>
</div>

对权重进行行切分

- 将输入通过 f(split) 函数均匀切分到每个模型并行 rank 内部
- 每个 rank 内部的模型并行线性层计算，得到部分输出
- 通过 g(all reduce) 操作 add 到一起，得到最终输出
- 反向传播时候，g 函数的梯度是原样返回，无需进行任何操作，f 函数的梯度是 (all gather) concat 操作

## VocabParallelEmbedding

沿着词汇表维度进行切割即可，由于输入并没有切分，因此最后只需要一次 all-reduce 操作

# 多头注意力模块

- n_query_heads=n_kv_heads 则是常规的 MHA
- n_kv_heads=1 则是 MQA
- n_kv_heads>1 and n_query_heads!=n_kv_heads 则是 GQA

# KV Cache

请结合本文件下两个 gif 来理解。

LLM 是序列解码方式，暂时无法并行解码。其解码过程为：

- 假设第一次预测时候有 prompt，假设长度是 10，那么开始解码时候输入的 embeding 是 (10, embed_dim)
- 在 attention 中，qkv 计算后会得到 (10, embed_dim) 输出，注意输入要传入下三角 attention mask
- 假设全部运行完可以得到 (10, vocab_size) 的输出，然后取最后一个 token id 作为下一次输入,得到 (11, embed_dim)
- 依次运行，直到得到 eos token

可以发现在第一次和第二次时候，都需要计算前 10 个 token 的 attention，而且由于有因果注意力，导致每次算前 10 个序列
输出都是一样的，这就出现了重复计算。 kv cache 就是为了避免掉这个重复计算

- 假设第一次预测时候有 prompt，假设长度是 10，那么开始解码时候输入的 embeding 是 (10, embed_dim)
- 在 attention 中，qkv 计算后会得到 (10, embed_dim) 输出，注意输入要传入下三角 attention mask
- 假设全部运行完可以得到 (10, vocab_size) 的输出，然后取最后一个 token id 作为下一次输入。此时需要把当前算出来的 key value 保存下来，其维度是 (10, embed_dim)
- 下一次迭代时候，只需要输入上一次解码的 token id 即为 (1, embed_dim) 即可，经过 qkv 投影后得到 (1, embed_dim)，然后从 kv cache 中取出上一次的 key value，其维度是 (10, embed_dim)，构造成 kv (11, embed_dim)，但是 q 还是 (1, embed_dim) 输入到 attention 中，得到 (1, vocab_size) 输出
- **后续的所有计算都是 (1, dim) 流转**，相比于之前计算量少了很多，attention mlp 部分都少了不少。

可以看出，区别就是，**使用 kvcache 后，每次只需要输入 1 个 token id ，当然输出序列长度也始终是 1**, 大幅减少了计算代价，没有任何重复计算。
缺点就是要缓存全部的 key value，因此会占用一定的显存。当上下文很长时候会是大头。

# RoPE 旋转位置编码


# 模型生成过程

llama3_generate_7.py

# DP 和 TP

DP 是 data parallel，TP 是 tensor parallel, 假设我一共有 32 卡，那么 rank 就是 0~31，

- 如果 dp=32，tp=1，那么每个 rank 内部有一个完整模型，也就是我们常用的 DDP 模式，数据层面每张卡都是不同数据
- 如果 dp=1, tp=32，也就是模型切分为 32 份，数据只有一份，每张卡上面都是拿到一部分模型，数据层面必须每张卡完全一样，训练时候每个 rank 运行一部分结果，然后通过一些特定的切分规则实现结果合并
- 如果 dp=8, tp=4，也就是模型切分为 4 份，数据切分为 8 份，也就是每张4张卡上面组成一个完整模型，这 4 张卡上面数据完全一样，类似于 0~3 rank 上组成一个完整模型，数据完全一样为 data1, 4~7 rank 上组成一个完整模型，数据完全一样为 data2, 以此类推
- 如果 dp=4, tp=8, 也就是模型切分为 8 份，数据切分为 4 份，每个节点组成一个完整模型，这个节点上数据完全一样，类似于 0~7 rank 上组成一个完整模型，数据完全一样为 data1, 8~15 rank 上组成一个完整模型，数据完全一样为 data2, 以此类推

对于专家并行，如果将其当做普通 tp 处理那么就类似 dp=1, ep=32 (假设一共 32 个专家)，那么将这 32 个专家均分到不同卡上，每个卡上面都是完整的某个独立专家，数据层面所有卡都是一样数据，
在训练时候，对于 rank0 -专家 0 来说，他只需要把经过路由 topk 后应该分发给 rank0 的 token 进行专家计算就可以了，没有任何其余通信。整个流程非常类似比 tp 还简单，而且没有通信过程。

考虑更通用的情况，假设一共 8 个专家，可以设置 ep=8, dp=4, e0~e7 对应 rank0~rank7，这个节点的数据完全一样，eo~-7 对应 rank8~rank15，这个节点的数据完全一样，以此类推。也就是 ep 一共会复制 4 份。

但是这样的问题是：每个专家接收到的 token 数其实是变小了，因为只有 1/32 的 token 会被分配到这个专家上，因此这个专家的训练效果可能会变差,因此实际上专家并行并不是这样做的，专家并行就是为了解决这个问题。

专家并行维度和 dp 维度可以没有任何联系，例如依然将有 32 张卡，dp 可以直接设置为 32，专家数可以任意，假设是 2 4 8

- 如果 ep 数等于 2，那么专家并行数就是 2，也就是每两张卡组成一个完整模型，rank0~1 上面是一个完整模型，但是数据也不一样，此时在训练时候 all-to-all dispatch 是会在 rank0~1 之间聚合数据
- 如果 ep 数等于 4，那么专家并行数就是 4，也就是每 4 张卡组成一个完整模型，rank0~4 上面是一个完整模型，但是数据也不一样，此时在训练时候 all-to-all dispatch 是会在 rank0~4 之间聚合数据
- 如果专家数非常多(通常也不会)为 128，但是只有 32 张卡，那么就要分组，比如 ep=32，此时每张卡上面有 4 个专家， all-to-all 的通信量非常大，因为所有数据都要 gather 过来一起算(rank0~31 的数据)。为了效率和通信量，可能会做折中，例如设置 ep=4，此时每张卡上面有 32 个专家，数据通信量会小很多(只需要同步 rank0~3 的数据)，但是每个专家接收到的 token 数会变少，但是显存会增加。

# PP

PP 是 pipeline parallel, 

# huggingface 生成逻辑

```generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)```

generate 位于 site-packages/transformers/generation/utils.py/generate 函数中

```python
@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
```

因果推理模型除了要重写 forward 外，一般还需要 prepare_inputs_for_generation，这个函数会在每一次 forward 之前都会调用一遍，然后才进行 forward
这个函数会不断的处理数据方便后面 foward。


对于不同的解码策略都会有一个专门的函数进行全局处理，以最简单的贪婪选择策略为例：

```python
batch_size, cur_len = input_ids.shape
this_peer_finished = False
unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    if synced_gpus and this_peer_finished:
        continue  # don't waste resources running the code we don't need

    next_token_logits = outputs.logits[:, -1, :]

    # pre-process distribution
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    # argmax
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)

    # finished sentences should have their next token be a padding token
    if eos_token_id is not None:
        if pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

    # update generated ids, model inputs, and length for next step
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=self.config.is_encoder_decoder,
    )
    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
    this_peer_finished = unfinished_sequences.max() == 0
```

缓存是通过初始化 model_kwargs["past_key_values"] = DynamicCache() 实现 transformers/cache_utils.py/DynamicCache 类实现的


```python
class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):  # 每一层单独缓存
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx: # 最外面一层是 layer_idx
            # 当前层没有缓存，直接添加
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 当前层有缓存，直接拼接
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx] # 返回的是包括历史的所有 key value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]
```


