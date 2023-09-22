# PEFT

Parameter-Efficient Fine-Tuning

官方地址： https://github.com/huggingface/peft  
官方文档：https://huggingface.co/docs/peft/index
fork 并注释版本: https://github.com/hhaAndroid/peft/tree/hha1

这个库现在代码量很少，没有几个 py 文件，阅读起来比较容易。examples 里面也有很多可以直接跑的案例

统一视角理解 PEFT: [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/pdf/2110.04366v3.pdf)

下图可以帮助理解：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236398391-d452b785-8d53-4979-a016-4945c4930ea6.png"/>
</div>

## LORA 

### 原理

论文题目： LORA: Low-Rank Adaptation of Large Language Models
论文地址：https://arxiv.org/abs/2106.09685

通过冻结预训练模型权重和注入可训练的秩分解矩阵到 Transformer 的每一层大大减少了下游任务训练参数的数量。与通过 Adam 微调的 175B GPT-3 相比，LoRA 将可训练参数减少了 10.000 倍并且 GPU 所需内存减少了 3 倍。LoRA 在 RoBERTa, DeBERTa, GPT-2 和 GPT-3 模型上展现了比微调模型相提并论或者更好的能力，并且 LoRA 具有更少的训练参数，更高的训练吞吐量并且不像适配器那样有额外的推理延迟

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236161808-c5f3cdb1-db8a-4d24-9c43-ffb88fd0e2c7.png"/>
</div>

r 表示 the rank of a LoRA module,远远小于 d 这个维度且不会增加推理延迟。。在原始PLM旁边增加一个旁路，做一个降维再升维的操作，来模拟所谓的intrinsic rank。训练的时候固定PLM的参数，只训练降维矩阵A与升维矩阵B。而模型的输入输出维度不变，输出时将BA与PLM的参数叠加。用随机高斯分布初始化A，用0矩阵初始化B，保证训练的开始此旁路矩阵依然是0矩阵。

已有方案的问题：

- Adapters引入额外的推理延迟 (由于增加了模型层数)
- Prefix-Tuning难于训练，且预留给prompt的序列挤占了下游任务的输入序列空间，影响模型性能

LoRA与Transformer的结合也很简单，仅在QKV attention的计算中增加一个旁路，而不动MLP模块。论文其实比较简单

### 代码解读

核心模块是 LoraConfig + LoraModel + LoraLayer，其中 LoraConfig 会自动控制生成 LoraModel，并自动替换为 LoraLayer，用户只需要操作 LoraConfig 即可。


以一个最简单的例子说明 lora 的具体做法

```python
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType

# model_name_or_path = "bigscience/mt0-large"
model_name_or_path = "bigscience/mt0-small"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
print(model)
model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters()
```

首先可以将前面 model 保存下来，进行对比：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236185425-3eac2f36-1d6f-4215-942c-2b621ea5b768.png"/>
</div>

左边为原始模型，右边为加了 lora 后的微调模型。可以发现 mt0 这个模型加入 lora 实际上就是在所有的 SelfAttention 的 Q 和 V 投影层加了 LOra 层而已,其他地方没有修改.

```python
(q): Linear(in_features=512, out_features=384, bias=False)
```

变成:

```python
(q): Linear(in_features=512, out_features=384, bias=False)
  (lora_dropout): ModuleDict(
    (default): Dropout(p=0.1, inplace=False)
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=512, out_features=8, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=8, out_features=384, bias=False)
  )
  (lora_embedding_A): ParameterDict() # 可以忽略
  (lora_embedding_B): ParameterDict() # 可以忽略
)
```

在设置 peft_config 时候,我们没有传入想插入 lora 层 target_modules 层的名字, 可以自动推断为 qv,是因为代码里面其实写死了:

```python
# src/peft/utils/other.py
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"], # 对应这个模型
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
}
```

因为模型是 HF 本身的,因为这些 hard code 是允许的。

下面先分析最底层的带有 lora 的 linear 

```python
# src/peft/tuners/lora.py
class Linear(nn.Linear, LoraLayer):
```

只要将找到的原始 qv 层，全部替换为这个 Linear 就实现了模型动态替换，后续就可以训练了。这个层实现了推理时候 merge 和 unmerge 的功能，方便用户使用。

```python
class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0, # 中间的维度
        lora_alpha: int = 1, # scaling 系数
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        
        # 初始化原先的线性层
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化 lora 
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        # 原先的线性层赋予原始值
        nn.Linear.reset_parameters(self)
        
        # Initialize the Lora layer A kaiming_uniform_ 初始化，B 全0初始化，构造恒等变换
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
```

```python
def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()
    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
    # Actual trainable parameters
    if r > 0:
        self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
        self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
        # lora_alpha 是一个缩放系数
        self.scaling[adapter_name] = lora_alpha / r
    if init_lora_weights:
        self.reset_lora_parameters(adapter_name)
    self.to(self.weight.device)
```

如果在微调阶段，计算流程为：

```python
result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

x = x.to(self.lora_A[self.active_adapter].weight.dtype)

result += (
    self.lora_B[self.active_adapter](
        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
    )
    * self.scaling[self.active_adapter]
)
```

如果权重要合并也比较简单，因为都是线性变换，所以直接相加就可以了。

```python
self.weight.data += (
    transpose(
        self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
        self.fan_in_fan_out,
    )
    * self.scaling[self.active_adapter]
)
```

模型+lora 后，实际上会套一层 LoraModel，在这个类里面实现了层的替换

```python
# src/peft/tuners/lora.py
class LoraModel(torch.nn.Module):
    config = self._prepare_lora_config(config, model_config) # 得到所有 lora 所需配置
    self._find_and_replace(adapter_name) # 替换层
    mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
```

```python
def _find_and_replace(self, adapter_name):
    # 遍历模型所有层
    key_list = [key for key, _ in self.model.named_modules()]
    for key in key_list:
        # 查看是否在 target_modules 中
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        
        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
        # 替换原先层
        self._replace_module(parent, target_name, new_module, target)
```

完整 demo 代码：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# model_name_or_path = "bigscience/mt0-large"
model_name_or_path = "bigscience/mt0-small"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
print(model)

model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters() # trainable params: 344064 || all params: 300520832 || trainable%: 0.11448923447676333

# once forward--权重没有估计，也没有merge
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>


# inference mode--这样只是权重固定，但是没有 merge
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # trainable params: 0 || all params: 300520832 || trainable%: 0.0

# 一旦设置 merge_adapter 就会自动合并了
model.base_model.merge_adapter()

# once forward
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>
```

具体见 https://github.com/hhaAndroid/peft/tree/hha1

## Prefix tuning 

论文：Prefix-Tuning: Optimizing Continuous Prompts for Generation
地址： https://arxiv.org/abs/2101.00190

### 原理

只训练 prefix 部分，其他地方全部固定。

- 把预训练大模型freeze住，因为大模型参数量大，精调起来效率低，毕竟prompt的出现就是要解决大模型少样本的适配
- 作者发现直接优化Prompt参数不太稳定，加了个更大的MLP，训练完只保存MLP变换后的参数就行了
- 实验证实只加到embedding上的效果不太好，因此作者在每层都加了prompt的参数，改动较大

### 实践

可以采用同样的模型，然后对比 peft 前面的模型结构差异。不过发现代码实现比较复杂，可能有错误理解的地方。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236375207-4fadad2e-710b-4492-a6a5-b6a15cf55260.png"/>
</div>

代码为：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, PrefixTuningConfig, TaskType

# model_name_or_path = "bigscience/mt0-large"
model_name_or_path = "bigscience/mt0-small"

# 对于这个模型，token_dim 必须要设置为 384，否则由于维度不匹配， model.generate 会报错
# 因为每个向量又分成了 6 个 head，所以 token_dim 必须要能被 6 整除，默认是 512，无法整除
peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20, token_dim=384)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
# print(model)

model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters() # trainable params: 122880 || all params: 300299648 || trainable%: 0.04091912888289499

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>
```

可以发现相比 lora，Prefix tuning 方法改动更少，只需要在模型前面新增 prompt_encoder + word_embeddings 就可以了。新增部分如下：

```python
  (prompt_encoder): ModuleDict(
    (default): PrefixEncoder(
      (embedding): Embedding(20, 6144)
    )
  )
```

核心逻辑代码为 PrefixEncoder

```python
class PrefixEncoder(torch.nn.Module):
     self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
```
如果不使用 mlp 层投影，那么核心就是加了一个 embeding 层。

- num_virtual_tokens 表示新增的可以学习的虚拟 token 数(按照论文应该是编码器和解码器都要加虚拟 token，为啥看起来只是 decoder 加了，encoder 没有加？)
- num_layers 表示有多少个 decoder transformer 层，实际上是 8 个
- token_dim 表示每个 token 的维度
- 2 表示 编码器输出的 vk 两个向量都要加虚拟 token

也就是说虽然代码实现方便 embedding 就是一个大的矩阵，但是实际上虚拟 token 要加到每个 decoder transformer 层 vk 向量前面

但是如果又不想直接改动 MT5ForConditionalGeneration 模型(其本身参数见 https://huggingface.co/bigscience/mt0-small/blob/main/config.json)，则需要在 PeftModelForSeq2SeqLM 中进行一些前后处理

具体代码实现比较复杂，下面列一下核心代码，实际上就是在自注意力层进行操作

```python
# site-packages/transformers/models/mt5/modeling_mt5.py#MT5Attention
# get key/value states
# past_key_value 就是额外追加的 20 个 token 的 embeding
key_states = project(
    hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
)
value_states = project(
    hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
)

def project(hidden_states, proj_layer, key_value_states, past_key_value):
    """projects hidden states correctly to key/query states"""
    if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(hidden_states))
    elif past_key_value is None:
        # cross-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(key_value_states))
    if past_key_value is not None:
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, key_length, dim_per_head)
            hidden_states = torch.cat([past_key_value, hidden_states], dim=2) # 核心代码
        elif past_key_value.shape[2] != key_value_states.shape[1]:
            # checking that the `sequence_length` of the `past_key_value` is the same as
            # the provided `key_value_states` to support prefix tuning
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))
        else:
            # cross-attn
            hidden_states = past_key_value
    return hidden_states
```

## P-Tuning

论文： [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
参考：https://huggingface.co/docs/peft/task_guides/ptuning-seq-classification

### 原理

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236403330-0136530d-63e6-42b1-bc12-c9ff3f7bb97a.png"/>
</div>

核心思想就是：人为构建好的 prompt 是比较难的，不同任务还不一样，设置为可学习是非常合理的。PromptEncoder 的位置其实不一定是前面，可以是任意位置，如上图所示，只不过 PEFT 里面实现的前面。

- Prefix Tuning 是将额外的 embedding 加在开头，看起来更像是模仿 Instruction 指令；而 P-Tuning 的位置则不固定。
- Prefix Tuning 通过在每个 Attention 层都加入 Prefix Embedding 来增加额外的参数，通过 MLP 来初始化；而 P-Tuning 只是在输入的时候加入 Embedding，并通过 LSTM+MLP 来初始化。

实现上相比 Prefix Tuning 简单很多。

### 实践

```python
(prompt_encoder): ModuleDict(
    (default): PromptEncoder(
      (embedding): Embedding(20, 768)
      (mlp_head): Sequential(
        (0): Linear(in_features=768, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=128, bias=True)
        (3): ReLU()
        (4): Linear(in_features=128, out_features=768, bias=True)
      )
    )
  )
```

因为这个随机的 PromptEncoder 是很离散的，要想得到连续的 embeding，作者说需要对初始化特别设置，例如在 mlp_head 前面再接入一个 lstm_head。

整个运行过程非常好理解;

```python
# peft/peft_model.py#PeftModelForSequenceClassification
if inputs_embeds is None:
    inputs_embeds = self.word_embeddings(input_ids) # 输入进行 embedding

prompts = self.get_prompt(batch_size=batch_size) # 获得 prompt 的 embeding
prompts = prompts.to(inputs_embeds.dtype)

inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)  # 拼接作为输入即可 （1,20+n,768）
return self.base_model(inputs_embeds=inputs_embeds, **kwargs) # 和 base 模型一样运行即可
```

```python
# get_prompt 方法
prompt_tokens = torch.arange(config.num_virtual_tokens * 1).long()
prompts = prompt_encoder(prompt_tokens)
```

## Prompt tuning

### 原理
[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)

Causal Language Modeling： 因果语言建模，因果语言建模关注于根据给定的上下文生成文本序列。在这种建模方法中，模型试图预测给定上下文中的下一个单词，该上下文通常包括在当前单词之前的所有单词。这种建模方法遵循因果原则，即当前单词只受到其前面单词的影响，而不受后面单词的影响。

因果语言建模的一个经典应用是GPT（如GPT-2和GPT-3），它主要用于生成连贯的文本。在这种建模方法中，模型接收一个输入序列，然后生成一个自然且语法正确的输出序列。

代表模型：GPT2、Bloom、OPT、GPT-Neo、GPT-J、LLaMA、ChatGLM。

Prompt tuning 和 P-tuning 非常类似，做法几乎可以说一样，只不过初始化策略不一样，Prompt tuning 可以直接用文本初始化虚拟 token，帮助训练

### 实践

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType

model_name_or_path = "bert-base-cased"

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()   # trainable params: 1413636 || all params: 125468676 || trainable%: 1.1266844004953076

# TODO 训练中...

# 假装训练好了
text_column = "Tweet text"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
inputs = tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)

# 因为没有训练，所以实际上是瞎输出
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    # max_new_tokens=10,
    eos_token_id=102
)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
```

核心代码为：

```python
class PromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer
            
            # 对于输入的文本进行 token，并作为 embedding 的初始化权重
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]

            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
```

训练时候也是作为虚拟 token 加入到输入 embedding 前面拼接起来。

## adalora

AdaLoRA: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)  

### 原理

How can we allocate the parameter budget adaptively according to importance of modules to improve the performance of parameter-efficient fine-tuning?

核心出发点是： lora 会对所有的 attention 都加入同样 rank 的可学习参数。但是实际上实验表明不同层的 attention 所需要的可学习参数不一样。在给定可学习参数预算情况下，如何最大化性能是一个值得关注的问题、
本文提出了 adalora 算法，在类似 LoRA 的微调期间动态分配权重矩阵之间的参数预算。具体来说，AdaLoRA 调整增量矩阵的rank 以控制它们的预算。关键增量矩阵被赋予高等级，以便它们可以捕获更细粒度的和特定于任务的信息。

AdaLoRA根据重要性评分自适应地分配参数预算，其实就是每次训练时候都动态的修改每个层的 rank，并且参数要基于新的 rank 进行 resize。在AdaLoRA中，以奇异值分解的形式对权重矩阵的增量更新进行参数化。然后，根据新的重要性指标，通过操纵奇异值，在增量矩阵之间动态地分配参数预算。这种方法有效地提高了模型性能和参数效率。

### 实践

由于目前用的不是很多，暂时没有细看。

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

# Distribution-Aware Prompt Tuning for Vision-Language Models

https://arxiv.org/pdf/2309.03406.pdf

# Dynamic Visual Prompt Tuning

Dynamic Visual Prompt Tuning for Parameter Efficient Transfer Learning

https://arxiv.org/pdf/2309.06123.pdf

为啥验证实验都是只有分类这一个任务？

# LongLoRA

https://github.com/dvlab-research/LongLoRA

