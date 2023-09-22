# BERT

论文： [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  

最大贡献是验证了在 NLP 中也可以采用先无监督预训练后微调的方式，而且效果非常好。微调时候只需要简单的增加一些输出层即可对下游任务进行微调，不需要修改 BERT 架构。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236760268-a86a31fa-406f-4b93-8e09-7c25c8616218.png"/>
</div>

图片来自 李宏毅 bert ppt 和 [链接](https://hackmd.io/@shaoeChen/Bky0Cnx7L#Bidirectional-Encoder-Representations-from-TransformersBERT) 感谢～

## 原理简析

https://huggingface.co/bert-base-uncased

示意图如下：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236749418-e5823526-8319-4a89-8262-215b09798d68.png"/>
</div>

- NSP： "next sentence prediction" 判断输入的两句话在语义上是否是连续的 (这两个句子可以是连续的语篇文本，也可以是随机选择的两个不相关的句子)，注意并不是 next token prediction, 是一个2分类任务

BERT 是基于 GPT1 来进行改进的。GPT1 是单向的 Transformer，作者认为单向的 Transformer 会导致模型对于上下文的理解不够，因此提出了 BERT (当然从现在来看，其实 GPT 系列效果其实可以非常好)。 BERT 是双向的 Transformer。

BERT 通过使用 “掩码语言模型”（MLM）预训练目标来缓解前面提到的单向约束。掩码语言模型从输入中随机屏蔽一些标记，目标是基于上下文预测掩码处的原始词汇 id。除了掩码语言模型之外，还使用类似的 “下一句预测”任务来联合预训练文本对表示。

网络结构参数：

- BERT_BASE (L=12, H=768, A=12, Total Parameters=110M)
- BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M)
- 最大支持长度为 512，即输入 token 最长是 512 个

A: the number of self-attention heads

为了让 BERT 可以尽可能的方便处理各种下游任务，作者在输入表示方面进行了一些设计，图示如下

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236750824-546cb490-118a-4812-ab77-373ee641addf.png"/>
</div>

典型的 NLP 任务例如 NSP，输入是一句话，但是对于判断两句话是否相似则输入是两句话。因此作者这里的输入句子不是真的是指的真正含义上的句子，也可以是两个句子然后通过特殊符号连接的一个句子作为输入。这样就可以统一输入形式，方便下游微调。

作者使用 WordPiece 嵌入算法进行分词和 30,000 个标记词汇，也就是词汇表大概是 3w。每个输入序列的第一个token始终是一个特殊的分类标记（[CLS]）。与该标记对应的最终隐藏状态被用作分类任务的聚合序列表示。如果是多个句子，则会打包成一个序列。作者以两种方式区分句子。首先，将它们与特殊标记 ([SEP]) 分开。其次，为每个标记添加一个可学习嵌入，指示它是否属于句子 A 或句子 B。

## 预训练过程

作者采用了两个任务联合训练的方式进行无监督预训练。

**(1) Task #1: Masked LM**

简单地随机屏蔽一定百分比的输入标记，然后预测这些掩码标记。与掩码标记对应的最终隐藏向量被输入到词汇表的输出 softmax 中进行训练。在我们所有的实验中，随机屏蔽每个序列中 15% 的所有 WordPiece 标记，并仅仅预测 masked token 而非整个句子。

但是这样会有一个问题：训练时候使用的 [MASK] token 在微调期间不会出现，这会导致预训练和微调期间的不一致。

解决办法是： 我们并不总是用实际的 [MASK] 标记替换“屏蔽”单词。训练数据生成器随机选择 15% 的标记位置进行预测。如果选择第 i 个令牌，我们将第 i 个令牌替换为 

- (1) 80% 的概率替换为 [MASK] token 
- (2) 10% 的概率替换为 随机token 
- (3) 10% 的概率未更改的第 i 个token 。然后，T_i 将用于预测具有交叉熵损失的原始标记

**(2) Task #2: Next Sentence Prediction (NSP)**

这个任务不是说文本生成。而是说给定两个句子，判断这两个句子是否是连续的。这个任务的目的是让模型学习到句子之间的关系。具体为在为每个预训练示例选择句子 A 和 B 时，50% 的时间 B 是遵循 A 的实际下一个句子（标记为 IsNext），50% 的时间是语料库中的随机句子（标记为 NotNext）。
所以实际上预训练时候这实际上是一个二分类问题。作者验证了虽然看起来好像非常简单，但是对于 Question Answering 等任务是非常有用的，因为这些任务要理解句子之间的联系。

对于预训练语料库，我们使用 BooksCorpus（8 亿字）和英语维基百科（2,500M 字）。对于 Wikipedia，我们只提取文本段落而忽略列表、表格和标题。使用文档级语料库而不是打乱的句子级语料库（例如十亿字基准）以提取长的连续序列至关重要。

## Fine-tuning BERT

对于每个任务，我们只需将特定于任务的输入和输出插入 BERT 中并端到端微调所有参数。作者在 11 个任务上进行了微调。

典型任务构造方式:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236755681-1ffc0bc5-ef09-4a78-89a4-c35085ca07e8.png"/>
</div>

单句/句子对分类任务：直接使用 [CLS] 的 hidden state 过一层分类层+ softmax 进行预测；

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236761255-8f6de0c6-5573-41aa-a022-69a97194177c.png"/>
</div>

QA 任务：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236761610-ae657cdd-2b56-403e-8347-f4a75b2f4e66.png"/>
</div>

给定一段内容和问题，输出答案的起始和结束位置。这里的输出是两个位置，所以需要两个分类层。 红蓝向量是新增的可学习 embedding。

## 代码详解

我们选择最常用的 https://huggingface.co/bert-base-uncased 来进行源码分析。也有中文版的 https://huggingface.co/bert-base-chinese

uncased 表示不区分大小写，统一转为小写输入和输出。

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer.vocab_size)  # 30522
print(tokenizer.get_vocab()) # 第 0 个是 [PAD]，中间是 100 个 [unused], 然后是 '[CLS]': 101, '[SEP]': 102, '[MASK]': 103，后面一大堆又是 [unused]
```

架构可以查看文件 config.json

```text
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072, # feedforward 层的维度
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512, # 最大输入长度
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute", # 位置编码方式
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

只有编码器。详细结构见 [bert model](hf_transformer/bert_base_model.txt),具体代码见 [code](hf_transformer/bert/bert_model.py)

下面提供了一个简单的 demo，并进行实现说明

```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
o = unmasker("Hello I'm a [MASK] model.")
print(o)
# [{'score': 0.10731077939271927, 'token': 4827, 'token_str': 'fashion', 'sequence': "hello i'm a fashion model."},
# {'score': 0.0877445638179779, 'token': 2535, 'token_str': 'role', 'sequence': "hello i'm a role model."},
# {'score': 0.05338413640856743, 'token': 2047, 'token_str': 'new', 'sequence': "hello i'm a new model."},
# {'score': 0.046672213822603226, 'token': 3565, 'token_str': 'super', 'sequence': "hello i'm a super model."},
# {'score': 0.027095887809991837, 'token': 2986, 'token_str': 'fine', 'sequence': "hello i'm a fine model."}]
```

实际上内部调用的是 FillMaskPipeline 类，并依次调用 preprocess _forward postprocess 三个方法

```python
def preprocess(self, inputs, return_tensors=None, **preprocess_parameters) -> Dict[str, GenericTensor]:
    if return_tensors is None:
        return_tensors = self.framework
    # Token 过程
    model_inputs = self.tokenizer(inputs, return_tensors=return_tensors)
    self.ensure_exactly_one_mask_token(model_inputs)
    return model_inputs
```

```python
def _forward(self, model_inputs):
    # 就是 BertForMaskedLM =BertModel+分类层BertOnlyMLMHead，结构见 hf_transformer/bert_base_for_maskedlm.txt
    model_outputs = self.model(**model_inputs)
    model_outputs["input_ids"] = model_inputs["input_ids"]
    return model_outputs

(cls): BertOnlyMLMHead(
  (predictions): BertLMPredictionHead(
    (transform): BertPredictionHeadTransform(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (transform_act_fn): GELUActivation()
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    )
    (decoder): Linear(in_features=768, out_features=30522, bias=True)
  )
)
```

输出 30522 维度

```python
def postprocess(self, model_outputs, top_k=5, target_ids=None):
    
    input_ids = model_outputs["input_ids"][0]
    outputs = model_outputs["logits"]
    
    # 取出哪个 token 位置才是 mask 标记
    masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    
    # 只需要这个位置预测就可以 (1,30522)
    logits = outputs[0, masked_index, :]
    
    probs = logits.softmax(dim=-1)
    if target_ids is not None:
        probs = probs[..., target_ids]
    # 取 token 个预测
    values, predictions = probs.topk(top_k)
    
    # 把预测的 masked token 和原来的 token 合并，重新解码，得到最终结果
    # hello i'm a fashion model.
    sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
```

