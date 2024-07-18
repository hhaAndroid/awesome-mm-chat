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

    def forward(self, x, attention_mask=None):
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

    attention_mask = torch.full((seqlen, seqlen), float("-inf"))
    attention_mask = torch.triu(attention_mask, diagonal=1)

    mha_attention = MultiheadAttention(128, 4, 4)
    output = mha_attention(x, attention_mask)
    print(output.shape)

    mqa_attention = MultiheadAttention(128, 4, 1)
    output = mqa_attention(x, attention_mask)
    print(output.shape)

    gqa_attention = MultiheadAttention(128, 4, 2)
    output = gqa_attention(x, attention_mask)
    print(output.shape)

    gqa_attention = MultiheadAttention(128, 4, 2, split_impl=False)
    output = gqa_attention(x, attention_mask)
    print(output.shape)
