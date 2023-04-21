## 2 使用 TRANSFORMERS

创建🤗 Transformers库就是为了解决这个问题。它的目标是提供一个API，通过它可以加载、训练和保存任何Transformer模型。这个库的主要特点是：

- 易于使用：下载、加载和使用最先进的NLP模型进行推理只需两行代码即可完成。
- 灵活：所有型号的核心都是简单的PyTorch nn.Module 或者 TensorFlow tf.kears.Model，可以像它们各自的机器学习（ML）框架中的任何其他模型一样进行处理。
- 简单：当前位置整个库几乎没有任何摘要。“都在一个文件中”是一个核心概念：模型的正向传递完全定义在一个文件中，因此代码本身是可以理解的，并且是可以破解的。

### 2.1 管道的内部

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233547821-14f9269f-84d4-4700-b721-26006e3f6fb1.png"/>
</div>

(1) 分词

Transformer 模型无法直接处理原始文本， 因此我们管道的第一步是将文本输入转换为模型能够理解的数字。 为此，我们使用tokenizer(标记器)，负责：

- 将输入拆分为单词、子单词或符号（如标点符号），称为标记(token)
- 将每个标记(token)映射到一个整数
- 添加可能对模型有用的其他输入

最终输出就是一系列的 int id。注意：所有这些预处理都需要以与模型预训练时完全相同的方式完成，也就是说分词要和模型绑定，不能随便换

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# {'input_ids': tensor([
#         [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
#         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0, 0,     0,     0,     0,     0,     0]]), 
#  'attention_mask': tensor([
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}

sequences = ["Hello!", "Cool.", "Nice!"]
encoded_input = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(tokenizer.decode(encoded_input['input_ids'][0]))  # [CLS] hello! [SEP]
```

模型架构图如下：
<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233549177-7f140ce0-25db-4263-abe8-83dd43d2c933.png"/>
</div>

一旦我们有了标记器，我们就可以直接将我们的句子传递给它，然后我们就会得到一本字典，它可以提供给我们的模型！剩下要做的唯一一件事就是将输入ID列表转换为张量。

上述模型词表长度为 30522，所以我们的输入 ID 列表长度为 16，这是因为我们的输入句子长度为 15，但是我们还需要一个额外的标记来表示句子的开始和结束，所以我们的输入ID列表长度为 16。

(2) 标记器（Tokenizer）

标记器(Tokenizer)是 NLP 管道的核心组件之一。它们有一个目的：将文本转换为模型可以处理的数据。

1. 基于词的(Word-based)
它通常很容易设置和使用，只需几条规则，并且通常会产生不错的结果。有多种方法可以拆分文本。例如，我们可以通过应用Python的split()函数，使用空格将文本标记为单词：

使用这种标记器，我们最终可以得到一些非常大的“词汇表”，其中词汇表由我们在语料库中拥有的独立标记的总数定义。每个单词都分配了一个 ID，从 0 开始一直到词汇表的大小。该模型使用这些 ID 来识别每个单词。

如果我们想用基于单词的标记器(tokenizer)完全覆盖一种语言，我们需要为语言中的每个单词都有一个标识符，这将生成大量的标记。例如，英语中有超过 500,000 个单词，因此要构建从每个单词到输入 ID 的映射，我们需要跟踪这么多 ID。此外，像“dog”这样的词与“dogs”这样的词的表示方式不同，模型最初无法知道“dog”和“dogs”是相似的：它会将这两个词识别为不相关。这同样适用于其他相似的词，例如“run”和“running”，模型最初不会认为它们是相似的。

最后，我们需要一个自定义标记(token)来表示不在我们词汇表中的单词。这被称为“未知”标记(token)，通常表示为“[UNK]”或”“。如果你看到标记器产生了很多这样的标记，这通常是一个不好的迹象，因为它无法检索到一个词的合理表示，并且你会在这个过程中丢失信息。制作词汇表时的目标是以这样一种方式进行，即标记器将尽可能少的单词标记为未知标记。

减少未知标记数量的一种方法是使用更深一层的标记器(tokenizer)，即基于字符的(character-based)标记器(tokenizer)。

2. 基于字符的(Character-based)
基于字符的标记器(tokenizer)将文本拆分为字符，而不是单词。这有两个主要好处：

- 词汇量要小得多。
- 词汇外（未知）标记(token)要少得多，因为每个单词都可以从字符构建。

但是，基于字符的标记器(tokenizer)也有一些缺点： 

- 从直觉上讲，它的意义不大：每个字符本身并没有多大意义
- 另一件要考虑的事情是，我们的模型最终会处理大量的词符(token)：虽然使用基于单词的标记器(tokenizer)，单词只会是单个标记，但当转换为字符时，它很容易变成 10 个或更多的词符(token)。也就是反向过程会比较麻烦

为了两全其美，我们可以使用结合这两种方法的第三种技术：子词标记化(subword tokenization)。

3. 子词标记化(Subword tokenization)

子词分词算法依赖于这样一个原则，即不应将常用词拆分为更小的子词，而应将稀有词分解为有意义的子词。

例如，“annoyingly”可能被认为是一个罕见的词，可以分解为“annoying”和“ly”。这两者都可能作为独立的子词出现得更频繁，同时“annoyingly”的含义由“annoying”和“ly”的复合含义保持。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233552922-3a8fa568-b401-4abf-88f9-8fc3a160924a.png"/>
</div>

`</w>` 表示单词的结尾，例如 tokenization 拆分为 token 和 ization</w>, 通过 </w> 可以知道ization是一个单词的最后一部分。

还有更多的技术。仅举几例：

- Byte-level BPE, 用于 GPT-2
- WordPiece, 用于 BERT
- SentencePiece or Unigram, 用于多个多语言模型

(3) 编码

将文本翻译成数字被称为编码(encoding).编码分两步完成：标记化，然后转换为输入 ID。

- 第一步是将文本拆分为单词（或单词的一部分、标点符号等），通常称为标记(token)
- 第二步是将这些标记转换为数字，这样我们就可以用它们构建一个张量并将它们提供给模型。为此，标记器(tokenizer)有一个词汇(vocabulary)，这是我们在实例化它时下载的部分 from_pretrained() 方法。同样，我们需要使用模型预训练时使用的相同词汇。

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

sequence = "Using a Transformer network is simple"

# 分词： ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
tokens = tokenizer.tokenize(sequence)

# 去词汇表里面查询
# [7993, 170, 13809, 23763, 2443, 1110, 3014]
ids = tokenizer.convert_tokens_to_ids(tokens)

# 解码
# "Using a Transformer network is simple"
decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])

# Using a Trans network is simple
decoded_string = tokenizer.decode([7993, 170, 13809, 2443, 1110, 3014])
```
内部会先对每个 id 进行查表，得到  ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']，然后再将这些 token 拼接起来。规则比较简单，如下所示：

```python
# 在 BertTokenizer 代码里面会把 ## 去掉,并且和前一个单词合并，从而形成正确的单词
def convert_tokens_to_string(self, tokens):
    # ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
    out_string = " ".join(tokens).replace(" ##", "").strip()
    #  Using a Transformer network is simple
    return out_string
```

如果想 batch 推理，需要考虑 padding 

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
print(model(torch.tensor(sequence1_ids)).logits)
# tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
print(model(torch.tensor(sequence2_ids)).logits)
# tensor([[ 1.5694, -1.3895],[ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
print(model(torch.tensor(batched_ids)).logits)
```

可以发现一个问题，`sequence2_ids` padding 和不 padding 结果不一样，这不符合预期。Transformer模型的关键特性是关注层，它将每个标记上下文化。这些将考虑填充标记，因为它们涉及序列中的所有标记。为了在通过模型传递不同长度的单个句子时，或者在传递一批应用了相同句子和填充的句子时获得相同的结果，我们需要告诉这些注意层忽略填充标记。这是通过使用 attention mask来实现的。

(4) Attention masks

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

这样就合理了。


几个必须要知道的参数：

```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

```python
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
```
"pt" 返回PyTorch 张量，"tf"返回TensorFlow张量，"np"返回NumPy数组：


下一部分 [README_3](README_3.md)
