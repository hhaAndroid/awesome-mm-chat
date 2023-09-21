# Transformer 进阶

前面基于全部重写的最简实践，我们已经知道了整个训练过程，但是代码过于陈旧，并且不如 Transformers 库方便，因此本文将基于 Transformers 库进行进一步的实践。

## Seq2Seq 模型推理

参考链接： peft 库中的 `examples/conditional_generation/peft_lora_seq2seq.ipynb`

选择的模型为 `bigscience/mt0-small`

### 模型介绍

https://huggingface.co/bigscience/mt0-small

mt0 模型是基于 mt5 模型并且在跨语言任务混合数据集（xP3）上微调所得模型，具体强大的人类指令跟随能力。他是一个生成式预训练模型，只不过是一个 encoder-decoder 结构
而 mt5 模型是基于 t5 模型训练而来，论文为 Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

<div align=center>
<img src="https://github.com/QwenLM/Qwen-7B/assets/17425982/ae3ea4a2-874e-41b0-bc44-e226797c9665"/>
</div>

所谓的条件生成就是指的用户会通过 prompt 来完成特定任务。

T5（Text-to-Text Transfer Transformer）模型结构仍然是一个由 Transformer 层堆叠而成的 Encoder-Decoder 结构。

T5与原生的Transformer结构非常类似，区别在于：

- 采用了一种简化版的 Layer Normalization，去除了 Layer Norm 的 bias
- 将Layer Norm放在残差连接外面
- 位置编码： T5 使用了一种简化版的相对位置编码，即每个位置编码都是一个标量，被加到 logits 上用于计算注意力权重。各层共享位置编码，但是在同一层内，不同的注意力头的位置编码都是独立学习的。一定数量的位置Embedding，每一个对应一个可能的 key-query 位置差。作者学习了 32 个 Embedding，至多适用于长度为 128 的位置差，超过位置差的位置编码都使用相同的 Embedding。

采用 SentencePiece工具 将文本切词为 WordPiece 词元，词典大小为32000。考虑到下游有翻译任务，作者按照英语:德语:法语:罗马尼亚语=10:1:1:1的比例构造语料库，重新训练 SentencePiece 模型。

简单用法：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "bigscience/mt0-small"

# model: MT5ForConditionalGeneration
# tokenizer_class: T5TokenizerFast
# "vocab_size": 250112
# pad_token_id": 0,
# eos_token_id": 1,
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
# inputs 后面会自动追加一个结束符，不需要开始符，代表输入结束。不同模型可能不一样
outputs = model.generate(inputs)
# 在推理时候，decoder 首先会注入开始解码符号，token_id=0,也就是和 pad_token_id 一样，所以输出的第一个 token 是 <pad>
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>
```

## GPT 模型推理
GPT 也是一个生成式模型，只有 decoder，没有 encoder，适合做生成任务。

以 GPT2 为例，虽然 GPT2 论文里面声称只需要进行无监督预训练即可，但是有些人为了在特定任务上更好，还是会有一些监督的训练，从代码上可以看出来

`transformers/models/gpt2/modeling_gpt2.py`


```text
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
```

这个是不包括 text prediction head 的 GPT2 模型。如果你直接调用，只能输出最后一层隐含层的输出。

```text
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
```

这个才是真正有用的 head，可以进行 text prediction。

当然也有其他一些特定的 head

```text
@add_start_docstrings(
    """
The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
```

```text
@add_start_docstrings(
    """
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
```

还有两个特定的 head。

因此我们要做模型推理，就必须要用 GPT2LMHeadModel

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Replace me by any text you'd like.", return_tensors="pt").to("cuda")
# 这个模型就没有在后面追加结束符
outputs = model.generate(inputs)
# 在推理时候，decoder 不需要注入额外的开始解码符，因为这个是 decoder-only 模型，直接基于前面的输入进行预测解码即可
print(tokenizer.decode(outputs[0]))  # I'm not sure if you're aware
```

## BERT 模型推理

BERT是另一个系列的模型，不同于 mt0 和 gpt，是一个掩码预训练模型，是只有 encoder 的模型。

