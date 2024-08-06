import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Union
from fairscale.nn.checkpoint import checkpoint_wrapper
import torch.nn.functional as F
from torch.nn import Linear, Embedding


@dataclass
class YOCOArgs:
    dim: int = 64
    n_layers: int = 6
    hidden_dim: int = 256
    n_self_heads: int = 8
    n_attn_heads: int = 8
    n_attn_kv_heads: int = 8
    vocab_size: int = 1000

    max_batch_size: int = 10
    max_seq_len: int = 100
    model_parallel_size: int = 1
    load_checkpoint: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    sliding_window: Optional[int] = 7


class SelfDecoder(nn.Module):
    def __init__(
            self,
            args: YOCOArgs,
            checkpoint_activations: bool = False
    ):
        super().__init__()
        self.args = args
        layers = [DecoderLayer(args, is_cross_layer=False, ) for idx in range(args.n_layers // 2)]
        if checkpoint_activations:
            layers = [checkpoint_wrapper(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self.head_dim = args.dim // args.n_self_heads
        self.block_size = 256
        self._precomputed_freqs_cis = None

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.args.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float,
                                                                  device=x.device))
            index = torch.arange(self.args.max_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos + x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos + x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def get_index_mask(self, x, length, pad_length):
        return torch.arange(pad_length, device=x.device) >= length

    def forward(
            self,
            x,
            incremental_state=None,
            is_prefilling=False,
            start_pos=0
    ):
        if is_prefilling and x.size(1) % self.block_size != 0 and self.args.sliding_window is None:
            padding_len = self.block_size - x.size(1) % self.block_size
            x = F.pad(x, (0, 0, 0, padding_len), value=0)
        else:
            padding_len = 0

        if incremental_state is not None and is_prefilling:
            index_mask = self.get_index_mask(x, x.size(1) - padding_len, x.size(1))

        rel_pos = self.build_rel_pos(x, start_pos)
        for idx, layer in enumerate(self.layers):
            if incremental_state is not None:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
                if is_prefilling:
                    incremental_state[idx]["index_mask"] = index_mask
            x = layer(
                x,
                start_pos=start_pos,
                rel_pos=rel_pos,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
                is_prefilling=is_prefilling, )

        x = x[:, :x.size(1) - padding_len, :]
        return x


def flash_attn_func(q, key, value, causal=True):
    # Compute the dot product between the query and the key
    attn_weights = torch.matmul(q, key.transpose(-1, -2)) / (key.size(-1) ** 0.5)

    if causal:
        # Apply causal masking
        seq_len = q.size(-2)
        key_len = key.size(-2)
        if seq_len > 1:
            mask = torch.full((seq_len, key_len), float("-inf"), device=q.device)
            mask = torch.triu(mask, diagonal=1)
            attn_weights = attn_weights + mask

    # Apply softmax to get the attention weights
    attn_weights = F.softmax(attn_weights, dim=-1)
    # Compute the weighted sum of the values
    attn_output = torch.matmul(attn_weights, value)

    return attn_output


class CrossAttention(nn.Module):
    def __init__(
            self,
            args,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.dim
        self.num_heads = args.n_attn_heads
        self.num_kv_heads = args.n_attn_kv_heads

        self.head_dim = args.dim // args.n_attn_heads
        self.q_proj = Linear(args.dim, args.dim, bias=False)
        self.out_proj = Linear(args.dim, args.dim, bias=False)

    def forward(
            self,
            x,
            key,
            value,
            rel_pos
    ):
        bsz, tgt_len, _ = x.size()

        q = self.q_proj(x)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        # 暂时关掉
        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)

        # TODO 暂时自己写
        q = q.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = flash_attn_func(q, key, value, causal=True)

        output = output.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
        output = self.out_proj(output)
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class CrossDecoder(nn.Module):
    def __init__(
            self,
            args: YOCOArgs,
            checkpoint_activations: bool = False
    ):
        super().__init__()
        self.args = args
        self.num_heads = args.n_attn_kv_heads
        self.head_dim = args.dim // args.n_attn_heads
        self.k_proj = Linear(args.dim, self.head_dim * args.n_attn_kv_heads, bias=False)
        self.v_proj = Linear(args.dim, self.head_dim * args.n_attn_kv_heads, bias=False)
        self.kv_layer_norm = RMSNorm(args.dim, eps=args.norm_eps)
        layers = [DecoderLayer(args, is_cross_layer=True) for idx in range(args.n_layers // 2)]
        if checkpoint_activations:
            layers = [checkpoint_wrapper(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self._precomputed_freqs_cis = None

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.args.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float,
                                                                  device=x.device))
            index = torch.arange(self.args.max_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos + x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos + x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def forward(
            self,
            x,
            incremental_state=None,
            start_pos=0,
            skip_cross_decoder=False,
    ):
        bsz, seqlen, embed_dim = x.size()
        x_norm = self.kv_layer_norm(x)
        key, value = self.k_proj(x_norm), self.v_proj(x_norm)
        key = key.view(bsz, seqlen, self.num_heads, self.head_dim)
        value = value.view(bsz, seqlen, self.num_heads, self.head_dim)
        rel_pos = self.build_rel_pos(x, start_pos)
        # key = apply_rotary_emb(key, *rel_pos, interleaved=True)
        if incremental_state is not None:
            if "prev_key" not in incremental_state:
                incremental_state["prev_key"] = torch.empty(bsz, self.args.max_seq_len, self.num_heads, self.head_dim,
                                                            device=x.device, dtype=x.dtype)
                incremental_state["prev_value"] = torch.empty(bsz, self.args.max_seq_len, self.num_heads, self.head_dim,
                                                              device=x.device, dtype=x.dtype)
            incremental_state["prev_key"][:, start_pos: start_pos + seqlen] = key
            incremental_state["prev_value"][:, start_pos: start_pos + seqlen] = value
            key = incremental_state["prev_key"][:, : start_pos + seqlen]
            value = incremental_state["prev_value"][:, : start_pos + seqlen]

        if skip_cross_decoder:
            return torch.zeros(bsz, 1, embed_dim, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(
                x,
                key=key,
                value=value,
                rel_pos=rel_pos)

        return x


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            load_checkpoint=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = Linear(self.embed_dim, ffn_dim, bias=False)
        self.gate = Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = Linear(ffn_dim, self.embed_dim, bias=False)

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc2(F.silu(self.fc1(x)) * self.gate(x))
        output = x.view(x_shape)
        return output


class SlidingWindowAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.dim
        self.num_heads = args.n_self_heads // args.model_parallel_size
        self.window_size = args.sliding_window - 1  # compatible with flash attention

        self.head_dim = args.dim // args.n_self_heads

        self.q_proj = Linear(args.dim, args.dim, bias=False)
        self.k_proj = Linear(args.dim, args.dim, bias=False)
        self.v_proj = Linear(args.dim, args.dim, bias=False)
        self.out_proj = Linear(args.dim, args.dim, bias=False)

    def forward(
            self,
            x,
            rel_pos,
            start_pos=0,
            incremental_state=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)
        if incremental_state is not None:
            if "prev_key" not in incremental_state:
                incremental_state["prev_key"] = torch.empty(self.args.max_batch_size, self.window_size, self.num_heads,
                                                            self.head_dim, device=x.device, dtype=x.dtype)
                incremental_state["prev_value"] = torch.empty(self.args.max_batch_size, self.window_size,
                                                              self.num_heads, self.head_dim, device=x.device,
                                                              dtype=x.dtype)

            key = torch.cat([incremental_state["prev_key"][:bsz, :start_pos], k], dim=1)
            value = torch.cat([incremental_state["prev_value"][:bsz, :start_pos], v], dim=1)
            if key.shape[1] > self.window_size:
                incremental_state["prev_key"][:bsz] = key[:, -self.window_size:]
                incremental_state["prev_value"][:bsz] = value[:, -self.window_size:]
            else:
                incremental_state["prev_key"][:bsz, start_pos: start_pos + tgt_len] = k
                incremental_state["prev_value"][:bsz, start_pos: start_pos + tgt_len] = v

        # attn = flash_attn_func(q, k, v, causal=True, window_size=(self.window_size - 1, 0))
        attn = flash_attn_func(q, k, v, causal=True)
        attn = attn.reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        attn = self.out_proj(attn)
        return attn


class DecoderLayer(nn.Module):
    def __init__(
            self,
            args: YOCOArgs,
            is_cross_layer=False
    ):
        super().__init__()
        self.args = args
        self.is_cross_layer = is_cross_layer

        if is_cross_layer:
            self.mixer = CrossAttention(args)
        elif args.sliding_window is not None:
            self.mixer = SlidingWindowAttention(args)
        else:
            pass

        self.mixer_layer_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.ffn = FeedForwardNetwork(
            args.dim,
            args.hidden_dim,
            args.load_checkpoint
        )

        self.final_layer_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x,
            start_pos=0,
            key=None,
            value=None,
            rel_pos=None,
            incremental_state=None,
            is_prefilling=False,
    ):
        residual = x
        x = self.mixer_layer_norm(x)

        if self.is_cross_layer:
            x = self.mixer(
                x,
                key,
                value,
                rel_pos=rel_pos,
            )
        elif self.args.sliding_window is not None:
            x = self.mixer(
                x,
                rel_pos=rel_pos,
                start_pos=start_pos,
                incremental_state=incremental_state,
            )
        else:
            x = self.mixer(
                x,
                rel_pos=rel_pos,
                incremental_state=incremental_state,
                is_prefilling=is_prefilling, )

        x = x + residual
        residual = x
        x = self.final_layer_norm(x)

        x = self.ffn(x)

        x = x + residual
        return x


class YOCO(nn.Module):
    def __init__(
            self,
            args,
            checkpoint_activations: bool = False,
            share_input_output_embed: bool = False,
    ):
        super().__init__()
        self.args = args
        self.embed_scale = math.sqrt(args.dim)
        self.embed_tokens = Embedding(args.vocab_size, args.dim)
        self.output_projection = nn.Linear(args.dim, args.vocab_size, bias=False)
        if share_input_output_embed:
            self.output_projection.weight = self.embed_tokens.weight

        self.self_decoder = SelfDecoder(args, checkpoint_activations)
        self.cross_decoder = CrossDecoder(args, checkpoint_activations)
        self.layer_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x,
            start_pos=0,
            incremental_state=None,
            is_prefilling=True,
            skip_cross_decoder=False
    ):
        x = self.embed_scale * self.embed_tokens(x)

        x = self.self_decoder(
            x,
            incremental_state=incremental_state,
            is_prefilling=is_prefilling,
            start_pos=start_pos,
        )

        x = self.cross_decoder(
            x,
            start_pos=start_pos,
            incremental_state=incremental_state,
            skip_cross_decoder=skip_cross_decoder,
        )

        x = self.layer_norm(x)
        x = self.output_layer(x)

        return x

    def output_layer(self, features):
        return self.output_projection(features)


if __name__ == '__main__':
    args = YOCOArgs()
    model = YOCO(args)

    # print(model)
    # x = torch.randint(0, 1000, (2, 4))
    # print(model(x).shape)

    # 生成模式
    with torch.no_grad():
        pad_id = 999
        bos_id = 999
        eos_id = 998
        prompt_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device="cpu")
        output_tokens = torch.cat((prompt_tokens, torch.full((prompt_tokens.shape[0], 50), pad_id).long()), dim=1)
        begin_pad_index = torch.where(output_tokens == pad_id)[1].min().item()
        incremental_state = {}
        eos_reached = torch.tensor([False] * prompt_tokens.shape[0], device="cpu")
        chunk_length = 32768
        for begin_index in range(0, begin_pad_index - 1, chunk_length):
            end_index = min(begin_index + chunk_length, begin_pad_index - 1)
            _ = model(output_tokens[:, begin_index: end_index], incremental_state=incremental_state,
                      start_pos=begin_index, skip_cross_decoder=True, is_prefilling=True)
        # generation
        for index in range(begin_pad_index, output_tokens.shape[1]):
            generation_net_output = model(output_tokens[:, index - 1].unsqueeze(-1),
                                          incremental_state=incremental_state, start_pos=index - 1,
                                          skip_cross_decoder=False, is_prefilling=False)
            generation_net_output[:, :, bos_id] = -math.inf
            generation_net_output[:, :, pad_id] = -math.inf
            next_tokens = torch.argmax(generation_net_output[:, -1, :], dim=-1)
            pad_tokens = output_tokens[:, index]
            next_tokens = torch.where((pad_tokens == pad_id) & ~eos_reached, next_tokens, pad_tokens)
            output_tokens[:, index] = next_tokens
            eos_reached |= (
                    next_tokens == eos_id
            )
        print(output_tokens)
