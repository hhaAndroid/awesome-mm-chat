# 并行 tiny example

- 单卡
- DDP
- TP
- DDP+TP
- SP
- DDP+SP
- PP
- Ring Attention： 分布式版本的 Flash Attention

TODO
- Context Parallelism
- DDP+PP
- DDP+TP+PP
- EP
- FSDP
- Flash Attention 原理版

# 矩阵乘法分块计算

- 矩阵分块计算
- softmax
- online softmax
- parallel online softmax
- python_qkv_online_softmax
- torch_qkv_chunk_online_softmax_one_pass
- online_qkv_attention_parallel


FSDP1 不管是 zero 几，实际上都会全切分模型，并且实际上也没有 zero1

- zero2: forward 时候会 all gather 参数，此时会保留全量参数，backward 后就不需要 all-gather 参数了，完成后会把模型全量参数再切开
- zero3: forward 时候会 all gather 参数,完成后全量参数释放，backward 时候再 all gather 参数，完成后再释放

