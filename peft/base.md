# PEFT

## 基本概念

### trust_remote_code

HF 支持调用外部类的功能。您可以使用 auto-classes 和 from_pretrained 方法在其存储库中使用任何带有自定义代码文件的配置、模型或标记器。所有上传到 Hub 的文件和代码都会被扫描以防恶意软件(更多信息请参阅 Hub 安全文档) ，但是你仍然应该检查模型代码和作者，以避免在你的机器上执行恶意代码。设置 trust_remote_code = True 来使用带有自定义代码的模型

```python
# https://huggingface.co/internlm/internlm-7b
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-7b", trust_remote_code=True).cuda()
```

可以看到实际上 InternLMForCausalLM 类的代码实际上并没有包括在 transformer 包里面，但是你依然可以运行，是因为这个类放到了远程 hub 上面了，设置了 trust_remote_code 后，就会自动调用，而不需要 transformer 包中存在这个类。非常方便

如果打印 config，那么是这样：

```text
InternLMConfig {
  "_name_or_path": "internlm/internlm-7b",
  "architectures": [
    "InternLMForCausalLM"
  ],
  "auto_map": { # 核心这个字段
    "AutoConfig": "internlm/internlm-7b--configuration_internlm.InternLMConfig",
    "AutoModel": "internlm/internlm-7b--modeling_internlm.InternLMForCausalLM",
    "AutoModelForCausalLM": "internlm/internlm-7b--modeling_internlm.InternLMForCausalLM"
  },
  "bias": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
```

configuration_internlm 是 py 文件名，InternLMConfig 是类名，他的隐射规则是这样。

此时实例化类是：

```python
model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, **hub_kwargs, **kwargs
            )
return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
```

实际上代码会同步下载到 `.cache/huggingface/modules/transformers_modules/internlm` 路径下。虽然这样比较好用，但是有被坑的风险，要小心。

```python
def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, ".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

### Causal Language Modeling

https://medium.com/@tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjdjMGI2OTEzZmUxMzgyMGEzMzMzOTlhY2U0MjZlNzA1MzVhOWEwYmYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMTY3NzgyNDczMTIxMjg0NzUzNzYiLCJlbWFpbCI6Imh1YW5naGFpYW4xMjNAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5iZiI6MTY5NDU5MDM1NCwibmFtZSI6ImhhaWFuIGh1YW5nIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0pYcXpya1dSbVlLUC1kbGlndC05MXpPRTdsRnRpbVJSR005dVdtQktjTT1zOTYtYyIsImdpdmVuX25hbWUiOiJoYWlhbiIsImZhbWlseV9uYW1lIjoiaHVhbmciLCJsb2NhbGUiOiJ6aC1UVyIsImlhdCI6MTY5NDU5MDY1NCwiZXhwIjoxNjk0NTk0MjU0LCJqdGkiOiI3MGI4ZjhiYjVjZGU2YmQ4ODBkNWJkZDc2NDFhN2MyYzBjODIxNWNmIn0.K5bv7LuclRjox6D5-3calbXMZhQeaKdt7cUL4e2-g5sqH5-UQQqj5L-8WzTaS-2nsOBmietQ6cGvYc5Cg-RObC0lWeLXuw7W-9qCVBWwA8WG9zQQBnuqm8FpkVnTi9m7iAo-9WMe8tx0GBzr2Q8syhlMCAp61PBK6VBM_IAU5a0XPRD_f9w8pKtoIEGGYfoJL0EPhwSiNbaGWrg-G5MVvFjZXd7WxfArpVqEXVfafFQZrcbsIGF6u6E5-NjJ4IDz1Vb9sfy7CLOJyudm1-gUeVxKMddR3Nt9eCVMb7c-Mu1YuXo4v0ZaLQ5m3HXE49yulWDZgCuiGvQTn40dsOQJDQ

CLM 是一种自回归的方法，模型通过给定前面的标记来训练以预测序列中的下一个标记。CLM 在像 GPT-2 和 GPT-3 这样的模型中使用，非常适合文本生成和摘要等任务。然而，CLM 模型具有单向上下文，这意味着在生成预测时只考虑过去的上下文而不考虑未来的上下文。

### Masked Language Modeling 

MLM（Masked Language Model）是一种训练方法，广泛应用于像 BERT 这样的模型中。在 MLM 中，输入序列中的一些标记被掩盖（masked），模型通过周围的上下文来预测被掩盖的标记。MLM 具有双向上下文的优势，让模型在进行预测时可以考虑到过去和未来的标记。这种方法在文本分类、情感分析和命名实体识别等任务中特别有用。

### Sequence-to-Sequence

Seq2Seq 模型由编码器-解码器结构组成，其中编码器处理输入序列，解码器生成输出序列。这种方法通常用于机器翻译、摘要和问答等任务。Seq2Seq 模型能够处理涉及输入-输出转换的更复杂任务，使其在各种自然语言处理任务中具有广泛的适用性。

三者的主要区别在于架构，训练方法和输出。

实现方式：
1 CLM：CLM 是一种自回归模型，通过给定前面的标记来预测序列中的下一个标记。它使用的是单向上下文，只考虑过去的标记。
2 MLM：MLM 是一种通过掩码标记来预测缺失标记的模型。它使用的是双向上下文，可以同时考虑过去和未来的标记。
3 seq2seq：seq2seq 模型包含编码器和解码器，编码器处理输入序列，解码器生成输出序列。它适用于输入和输出之间的转换任务。

架构：
1 CLM：CLM 使用自回归架构，通常基于 Transformer 模型，如 GPT 系列。它通过预测下一个标记来生成序列。通常是只包括解码器,如 GPT 系列
2 MLM：MLM 通常基于 Transformer 模型，如 BERT 和 RoBERTa。它通过掩码标记并预测它们来学习上下文表示。通常是只包括编码器，如 BERT
3 seq2seq：seq2seq 模型通常使用编码器-解码器架构，可以是基于循环神经网络（RNN）或 Transformer 的变体。编码器将输入序列编码为固定长度的向量表示，解码器根据该表示生成输出序列。通常是包括编码器和解码器，如 T5, BART

输出模型：
1 CLM：CLM 输出的是根据给定上下文预测的下一个标记。它适用于生成连贯的文本序列。
2 MLM：MLM 输出的是对掩码标记的预测，以填补缺失的标记。它适用于文本分类、命名实体识别等任务。
3 seq2seq：seq2seq 输出的是根据输入序列生成的输出序列。它适用于机器翻译、摘要、问答等任务。

### Conditional Generation

https://docs.gretel.ai/reference/synthetics/conditional-generation-faq

例如 HF 里面的 MT5ForConditionalGeneration 类。

有条件的数据生成（有时称为提示）是一种技术，其中一个生成模型被要求根据预先指定的条件生成数据，例如主题、情感或使用表格、文本或基于图像的数据集中的一个或多个字段值。

## 打开 from_pretrained 

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name_or_path = "bigscience/mt0-small"  # mt5 的 funtune 版本
tokenizer_name_or_path = "bigscience/mt0-small"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
print(model)

# from_pretrained 原理和流程也一样,但是更简单
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 因为输入是 prompt + text，所以实际上是条件生成模型
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
outputs = model.generate(input_ids=inputs)
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>
```

transformers的三个核心抽象类是Config, Tokenizer和Model，这些类根据模型种类的不同，派生出一系列的子类。构造这些派生类的对象也很简单，transformers为这三个类都提供了自动类型，即AutoConfig, AutoTokenizer和AutoModel。三个AutoClass都提供了from_pretrained方法，这个方法则一气完成了模型类别推理、模型文件列表映射、模型文件下载及缓存、类对象构建等一系列操作。

以以下为例

```python
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-small")
```

这一行代码实际上做了非常多事情，第一步是自动生成配置


```python
config, kwargs = AutoConfig.from_pretrained(
    "bigscience/mt0-small",
    return_unused_kwargs=True,
    trust_remote_code=trust_remote_code,
    **hub_kwargs,
    **kwargs_copy,
)
```

这个 config 对象非常重要，里面存储了非常多信息。配置里面也保留了该文件的 hash 值(_commit_hash)，下次如果读取缓存时候会根据这个值来读取。因为可能会存在多份权重。

```text
MT5Config {
  "_name_or_path": "bigscience/mt0-small",
  "architectures": [
    "MT5ForConditionalGeneration" # 类别名
  ],
  "d_ff": 1024,
  "d_kv": 64,
  "d_model": 512,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "mt5",
  "num_decoder_layers": 8,
  "num_heads": 6,
  "num_layers": 8,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "tokenizer_class": "T5Tokenizer", # tokenizer 类别名
  "torch_dtype": "float32",
  "transformers_version": "4.29.1", # 版本
  "use_cache": true,
  "vocab_size": 250112
}
```

然后利用这个类实例化模型

```python
model_class = _get_model_class(config, cls._model_mapping)
# model_class= transformers.models.mt5.modeling_mt5.MT5ForConditionalGeneration
return model_class.from_pretrained(
    pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
)
```

MT5ForConditionalGeneration 是继承自 transformers/modeling_utils.py/PreTrainedModel，内部实现了很多有用的方法，包括 from_pretrained

这个方法有非常多参数可以设置，

1. 可以传入本地路径，他就不会走网络
2. 如果已经下载，会直接从缓存读取,会进行 hash 校验,实际上下载是通过 hf_hub_download 工具实现的

模型下载后才可以实例化 

```python
model = cls(config, *model_args, **model_kwargs)
```

```python
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model) # token embeding 层

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared) # 编码器

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared) # 解码器

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # head

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
```

上述就将类自动构建好了,下面看生成函数 transformers/generation/utils.py/GenerationMixin/generate, 所有的生成相关的逻辑都是在这个类里面处理的。

MT5 是一个编码器解码器网络,因此生成时候需要先运行编码器,然后运行解码器,这个流程都是在 generate 里面控制,而不是 forward 里面。

因果模型推理时候可以通过 `past_key_values` 参数来加速,这个参数的含义是历史计算的 kv 值。

