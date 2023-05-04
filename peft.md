# PEFT

Parameter-Efficient Fine-Tuning

官方地址： https://github.com/huggingface/peft  
官方文档：https://huggingface.co/docs/peft/index
fork 并注释版本: https://github.com/hhaAndroid/peft/tree/hha

这个库现在代码量很少，没有几个 py 文件，阅读起来比较容易。

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

具体见 https://github.com/hhaAndroid/peft/tree/hha

## Prefix tuning 
### 原理

### 实践

