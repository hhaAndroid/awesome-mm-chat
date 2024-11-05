from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

DEFAULT_PRECISION = 2


def number_to_string(num, units=None, precision=DEFAULT_PRECISION):
    if units is None:
        if num >= 1e12:
            magnitude, units = 1e12, "T"
        elif num >= 1e9:
            magnitude, units = 1e9, "G"
        elif num >= 1e6:
            magnitude, units = 1e6, "M"
        elif num >= 1e3:
            magnitude, units = 1e3, "K"
        elif num >= 1 or num == 0:
            magnitude, units = 1, ""
        elif num >= 1e-3:
            magnitude, units = 1e-3, "m"
        else:
            magnitude, units = 1e-6, "u"
    else:
        if units == "T":
            magnitude = 1e12
        elif units == "G":
            magnitude = 1e9
        elif units == "M":
            magnitude = 1e6
        elif units == "K":
            magnitude = 1e3
        elif units == "m":
            magnitude = 1e-3
        elif units == "u":
            magnitude = 1e-6
        else:
            magnitude = 1
    return f"{round(num / magnitude, precision):g} {units}"


def params_to_string(params_num, units=None, precision=DEFAULT_PRECISION):
    units = units.replace("B", "G") if units else units
    return number_to_string(params_num, units=units, precision=precision).replace("G", "B").strip()


# 通过公式计算，而不是直接读取模型的参数
def calc_params(model_name):
    config = AutoConfig.from_pretrained(model_name)
    if 'head_dim' not in config.__dict__:
        config.head_dim = config.hidden_size // config.num_attention_heads

    # 1 embedding
    embed_params = config.hidden_size * config.vocab_size

    # 2 self attention
    q_params = config.hidden_size * (config.num_attention_heads * config.head_dim)
    kv_params = 2 * config.hidden_size * (config.num_key_value_heads * config.head_dim)
    o_params = config.hidden_size * (config.num_attention_heads * config.head_dim)
    total_self_attn_params = q_params + kv_params + o_params

    # 2 mlp
    mlp_params = 3 * config.hidden_size * config.intermediate_size

    # 2 decoder layers
    decoder_params = (total_self_attn_params + mlp_params) * config.num_hidden_layers

    # 3 output layer
    if config.tie_word_embeddings is False:
        output_params = config.hidden_size * config.vocab_size
    else:
        output_params = 0

    return embed_params + decoder_params + output_params


# FLOPs 指的是浮点运算（加法，乘法）次数
def calc_FLOPs(model_name, bs=1, seq_len=1024):
    config = AutoConfig.from_pretrained(model_name)
    if 'head_dim' not in config.__dict__:
        config.head_dim = config.hidden_size // config.num_attention_heads

    # 1 embedding
    # (b,s) -> (b,s,h) 查表操作
    embed_flops = bs * seq_len * config.hidden_size

    # k,n X n,m -> k,m，FLOPs = 2 * k * n * m 2 是因为乘法和加法
    # 2 self attention
    # q: (b,s,h) * (h, config.num_attention_heads * config.head_dim) -> (b,s,config.num_attention_heads * config.head_dim)
    q_flops = 2 * bs * seq_len * config.hidden_size * (config.num_attention_heads * config.head_dim)
    # kv
    kv_flops = 2 * 2 * bs * seq_len * config.hidden_size * (config.num_key_value_heads * config.head_dim)

    # rotary_emb 计算量很小，不考虑

    # qk
    # (b,num_head,s, config.head_dim) * (b,num_head,config.head_dim, s) -> (b,num_head,s,s)
    qk_flops = 2 * bs * config.num_attention_heads * config.head_dim * seq_len * seq_len

    # dropout 忽略， softmax flops 也很小，忽略，也没有考虑激活

    # v (b,num_head,s,s) * (b,num_head,s, config.head_dim) -> (b,num_head, s, config.head_dim)
    # 除以 2 是假设是 casual attention，实际上在使用了 flash att 后也不准
    atten_weight_flops = 2 * bs * config.num_attention_heads * config.head_dim * seq_len * seq_len / 2

    # o_proj
    # (b,s,config.num_attention_heads * config.head_dim) * (config.num_attention_heads * config.head_dim, h) -> (b,s,h)
    o_proj_flops = 2 * bs * seq_len * (config.num_attention_heads * config.head_dim) * config.hidden_size

    total_self_attn_flops = q_flops + kv_flops + qk_flops + atten_weight_flops + o_proj_flops

    # 2 mlp
    gate_proj_flops = 2 * bs * seq_len * config.hidden_size * config.intermediate_size
    down_proj_flops = 2 * bs * seq_len * config.intermediate_size * config.hidden_size
    up_proj_flops = 2 * bs * seq_len * config.hidden_size * config.intermediate_size
    total_mlp_flops = gate_proj_flops + down_proj_flops + up_proj_flops

    # 3 output
    # (b,s,h) * (h, vocab_size) -> (b,s,vocab_size)
    output_flops = 2 * bs * seq_len * config.hidden_size * config.vocab_size

    total_flops = embed_flops + (total_self_attn_flops + total_mlp_flops) * config.num_hidden_layers + output_flops

    # 计算每个部分占比
    embed_ratio = embed_flops / total_flops
    self_attn_ratio = (total_self_attn_flops / total_flops) * config.num_hidden_layers
    self_attn_ratio_pre_layer = total_self_attn_flops / total_flops
    mlp_ratio = (total_mlp_flops / total_flops) * config.num_hidden_layers
    mlp_ratio_pre_layer = total_mlp_flops / total_flops
    output_ratio = output_flops / total_flops
    print('======================')
    print(f" Embedding FLOPs 占比: {embed_ratio:.2f},\n",
          f"Self Attention FLOPs 占比: {self_attn_ratio:.2f},\n",
          f"Self Attention FLOPs pre layer 占比: {self_attn_ratio_pre_layer:.2f},\n",
          f"MLP FLOPs 占比: {mlp_ratio:.2f},\n",
          f"MLP FLOPs pre layer 占比: {mlp_ratio_pre_layer:.2f},\n",
          f"Output FLOPs 占比: {output_ratio:.2f},")
    print('======================')
    return total_flops


def calc_inference_memory(model_name, bs=1, prompt_len=1024, output_len=1024):
    config = AutoConfig.from_pretrained(model_name)
    if 'head_dim' not in config.__dict__:
        config.head_dim = config.hidden_size // config.num_attention_heads

    # 如果是 1B 参数那么就是 2G
    params = calc_params(model_name)

    # kv cache
    kv_memory = 2 * bs * (prompt_len + output_len) * config.num_key_value_heads * config.head_dim

    total_memory = params + kv_memory * config.num_hidden_layers

    params_ratio = params / total_memory
    kv_ratio_pre_layer = kv_memory / total_memory
    kv_ratio = kv_ratio_pre_layer * config.num_hidden_layers

    print('vvvvvvvvvvvvvvvvvvvvvv')
    print(f" 参数占比: {params_ratio:.2f},\n",
          f"kv cache 占比: {kv_ratio:.2f} \n",
          f"kv cache pre layer 占比: {kv_ratio_pre_layer:.2f},\n",
          f"kv cache : {number_to_string(kv_memory * config.num_hidden_layers, 'G')}")
    print('vvvvvvvvvvvvvvvvvvvv')

    return total_memory * 2


# https://zhuanlan.zhihu.com/p/4083427292
def calc_train_memory(model_name, bs=1, seq_len=1024, dp=4, use_ckpt=True):
    config = AutoConfig.from_pretrained(model_name)
    if 'head_dim' not in config.__dict__:
        config.head_dim = config.hidden_size // config.num_attention_heads

    # 训练显存包括模型参数，梯度，优化器状态，激活值和临时 buffer
    # bf16格式，如果是 1b 参数，那么显存就是 2
    params = calc_params(model_name)

    # 梯度
    grad = params

    # adam 优化器状态,fp32 格式，每个参数有 2 个参数，因此是 4 倍数
    adam_state = params * 4

    # 如果还有一个 fp32 的备份，用于梯度更新
    params_bk = 2 * params

    # 只切分优化器
    zero1_memory = params + grad + (adam_state + params_bk) // dp
    # 切分优化器和梯度
    zero2_memory = params + (grad + adam_state + params_bk) // dp
    # 全部切分
    zero3_memory = (params + grad + adam_state + params_bk) // dp

    # 激活值分析：激活（activations）指的是：前向传递过程中计算得到的，并在后向传递过程中需要用到的所有张量
    # 保存的激活值是以某一层的输入来算的(如果后续 bwd 时候需要用到这个输入，则保存，至于他的输出会当做下一层的输入)，
    # 由于 embedding 是第一层，其前面没有参数，因此不需要保存激活值
    embedding_act = 0

    # 2 self attention
    # q
    # 输入 x (b,s,h) 后续 bwd 肯定要用，因此首先就有一个 x 的激活值
    q_input_act = bs * seq_len * config.hidden_size
    # q_proj 输出
    q_output_act = bs * seq_len * (config.num_attention_heads * config.head_dim)
    # kv proj 输出
    kv_output_act = 2 * bs * seq_len * (config.num_key_value_heads * config.head_dim)

    # softmax 由于用了 flash attention2 实际上激活值远远小于这个数
    # 暴力的算法就是把平方去掉
    # softmax_act = bs * config.num_attention_heads * seq_len * seq_len
    softmax_act = bs * config.num_attention_heads * seq_len

    # dropout 一般是不开的
    dropout_act = 0
    # o_proj
    o_proj_act = bs * seq_len * config.hidden_size

    total_self_attn_act = q_input_act + q_output_act + kv_output_act + softmax_act + dropout_act + o_proj_act

    # 3 mlp
    mlp_inputs = bs * seq_len * config.hidden_size
    gate_proj_act = bs * seq_len * config.intermediate_size
    up_proj_act = bs * seq_len * config.intermediate_size
    down_proj_act = bs * seq_len * config.hidden_size

    total_mlp_act = mlp_inputs + gate_proj_act + up_proj_act + down_proj_act

    # output
    output_input_act = bs * seq_len * config.hidden_size
    # 有一个 float 操作
    output_output_act = bs * seq_len * config.vocab_size * 2
    # output_output_act = 0

    if use_ckpt:
        # 实际上这里也不能直接乘以 1，因为在开启 ckpt 后，在每个 layer 输入前都会额外保存一份激活，用于连接前后两个 layer
        total_act = embedding_act + (
                total_self_attn_act + total_mlp_act) * 1 + output_input_act + output_output_act
        total_act += (config.num_hidden_layers-1) * q_input_act
        config.num_hidden_layers = 1
    else:
        total_act = embedding_act + (
                total_self_attn_act + total_mlp_act) * config.num_hidden_layers + output_input_act + output_output_act

    zero1_total_memory = zero1_memory + total_act
    zero2_total_memory = zero2_memory + total_act
    zero3_total_memory = zero3_memory + total_act

    # 显存占比
    zero1_ratio = zero1_memory / zero1_total_memory
    zero1_act_ratio = total_act / zero1_total_memory
    zero2_ratio = zero2_memory / zero2_total_memory
    zero2_act_ratio = total_act / zero2_total_memory
    zero3_ratio = zero3_memory / zero3_total_memory
    zero3_act_ratio = total_act / zero3_total_memory

    att_ratio_pre_layer = total_self_attn_act / total_act
    att_ratio = att_ratio_pre_layer * config.num_hidden_layers
    mlp_ratio_pre_layer = total_mlp_act / total_act
    mlp_ratio = mlp_ratio_pre_layer * config.num_hidden_layers
    out_ratio = (output_input_act + output_output_act) / total_act
    print(
        f" [zero1] zero占比: {zero1_ratio:.2f},\n",
        f"[zero1] 激活占比: {zero1_act_ratio:.2f} \n",
        f"[zero2] zero占比: {zero2_ratio:.2f},\n",
        f"[zero2] 激活占比: {zero2_act_ratio:.2f} \n",
        f"[zero3] zero占比: {zero3_ratio:.2f},\n",
        f"[zero3] 激活占比: {zero3_act_ratio:.2f} \n",
        f"attn pre layer 激活占比: {att_ratio_pre_layer:.2f},\n",
        f"attn 激活占比: {att_ratio:.2f},\n",
        f"mlp pre layer 激活占比: {mlp_ratio_pre_layer:.2f},\n",
        f"mlp 激活占比: {mlp_ratio:.2f},\n",
        f"输出 激活占比: {out_ratio:.2f},\n",
    )
    return zero1_total_memory * 2, zero2_total_memory * 2, zero3_total_memory * 2


if __name__ == '__main__':
    model_name = 'Qwen/Qwen2.5-72B'
    params = calc_params(model_name)
    print(f"模型参数量: {params_to_string(params, 'M')}")

    bs = 2
    seq_len = 2048
    dp = 32

    flops = calc_FLOPs(model_name, bs=bs, seq_len=seq_len)
    print(f"模型 Forward FLOPs: {number_to_string(flops, 'T')}")

    # 推理显存不用分析
    memory_ = calc_inference_memory(model_name, bs=bs, prompt_len=seq_len, output_len=seq_len)
    print(f"模型推理显存: {number_to_string(memory_, 'G')}")

    print('=============================================')
    zero1_memory_, zero2_memory_, zero3_memory_ = calc_train_memory(model_name, bs=bs, seq_len=seq_len, dp=dp)
    print(f"[zero1]单卡模型训练显存: {number_to_string(zero1_memory_, 'G')}")
    print(f"[zero2]单卡模型训练显存: {number_to_string(zero2_memory_, 'G')}")
    print(f"[zero3]单卡模型训练显存: {number_to_string(zero3_memory_, 'G')}")
