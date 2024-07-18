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


