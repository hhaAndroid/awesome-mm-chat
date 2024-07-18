import math
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_query_heads: int,
            n_kv_heads: int,
            split_impl: bool = False,
            # kv cache
            use_kvcache: bool = False,
            max_batch_size: int = 20,
            max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model  # 这里的 d_model 是包括了 n_heads 的

        self.n_query_heads = n_query_heads
        # n_query_heads=n_kv_heads 则是常规的 MHA
        # n_kv_heads=1 则是 MQA
        # n_kv_heads>1 and n_query_heads!=n_kv_heads 则是 GQA
        self.n_kv_heads = n_kv_heads

        self.head_dim = d_model // n_query_heads
        self.n_repeat = n_query_heads // n_kv_heads

        # 分开写
        self.split_impl = split_impl
        if self.split_impl:
            self.wq = nn.Linear(d_model, self.head_dim * n_query_heads)
            self.wk = nn.Linear(d_model, self.head_dim * n_kv_heads)
            self.wv = nn.Linear(d_model, self.head_dim * n_kv_heads)
        else:
            # 合并写
            self.Wqkv = nn.Linear(d_model, self.head_dim * (n_query_heads + 2 * n_kv_heads))

        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # kv cache
        self.use_kvcache = use_kvcache
        if self.use_kvcache:
            self.max_batch_size = max_batch_size
            self.max_seq_len = max_seq_len
            self.cache_k = torch.zeros(
                (
                    max_batch_size,
                    max_seq_len,
                    self.n_kv_heads,
                    self.head_dim,
                )
            )
            self.cache_v = torch.zeros(
                (
                    max_batch_size,
                    max_seq_len,
                    self.n_kv_heads,
                    self.head_dim,
                )
            )

    # start_pos 是为了支持 kv cache 的
    def forward(self, x, attention_mask=None, start_pos=0):
        # (b, c, d)
        bsz, seqlen, _ = x.shape
        if self.split_impl:
            # (b,c,d1) (b,c,d2) (b,c,d2)
            query, key, value = self.wq(x), self.wk(x), self.wv(x)
        else:
            qkv = self.Wqkv(x)
            query = qkv[..., :self.head_dim * self.n_query_heads]
            key = qkv[..., self.head_dim * self.n_query_heads:self.head_dim * (self.n_query_heads + self.n_kv_heads)]
            value = qkv[..., self.head_dim * (self.n_query_heads + self.n_kv_heads):]

        # (b,c, q_heads, d) (b,c kv_heads,d) (b,c kv_heads,d)
        query = query.view(bsz, seqlen, self.n_query_heads, self.head_dim)
        key = key.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        value = value.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.use_kvcache:
            # 当前值存起来
            self.cache_k[:bsz, start_pos: start_pos + seqlen] = key
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = value
            # 当前和之前的全部取出来
            key = self.cache_k[:bsz, :start_pos + seqlen]
            value = self.cache_v[:bsz, :start_pos + seqlen]

        if self.n_repeat > 1:
            # MQA 和 GQA 场景需要 repeat 成 query 一样的 shape
            # (b, c, kv_heads, d) -> (b,c, q_heads, d)
            bs, slen, n_kv_heads, head_dim = key.shape
            key = key[:, :, :, None, :].expand(bs, slen, n_kv_heads, self.n_repeat, head_dim) \
                .reshape(bs, slen, n_kv_heads * self.n_repeat, head_dim)
            value = value[:, :, :, None, :].expand(bs, slen, n_kv_heads, self.n_repeat, head_dim) \
                .reshape(bs, slen, n_kv_heads * self.n_repeat, head_dim)

        # (b, c, q_heads, d) -> (b, q_heads, c, d)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # (b, q_heads, c, c)
        attn_weight = query.matmul(key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weight = attn_weight + attention_mask  # 0 有效位置， -inf 屏蔽位置

        attn_weight = torch.softmax(attn_weight, dim=-1)
        # (b, q_heads, c, c) * (b, q_heads, c, d) -> (b, q_heads, c, d)
        output = attn_weight.matmul(value)
        # (b, q_heads, c, d) -> (b, c, q_heads, d) -> (b, c, d)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.out_proj(output)


if __name__ == '__main__':
    seqlen = 5
    x = torch.randn(2, 5, 128)
    total_len = 10  # 假设最多预测 10 个 token

    mha_attention = MultiheadAttention(128, 4, 4, use_kvcache=False)

    # 不使用 kv cache
    for i in range(total_len):
        # seqlen 可以理解是 prompt 的长度
        attention_mask = torch.full((seqlen + i, seqlen + i), float("-inf"))
        attention_mask = torch.triu(attention_mask, diagonal=1)
        output = mha_attention(x, attention_mask)
        x = torch.cat([x, output[:, -1, :][:, None]], dim=1)
        print(x.shape)

    print('--- kv cache ---')
    x = torch.randn(2, 5, 128)
    mha_attention = MultiheadAttention(128, 4, 4, use_kvcache=True)
    for i in range(total_len):
        if seqlen > 1:
            attention_mask = torch.full((seqlen, seqlen), float("-inf"))
            attention_mask = torch.triu(attention_mask, diagonal=1)
        x = mha_attention(x, start_pos=seqlen + i)
        x = x[:, -1, :][:, None]
        print(x.shape)
