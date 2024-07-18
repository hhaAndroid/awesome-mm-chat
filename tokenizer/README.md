# Tokenizer

## 如何生成 llama3 相同的 tokenzier 文件结构

https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main

- special_tokens_map.json
- tokenizer_config.json
- tokenizer.json

详情见 `1_from_llama3.py` 文件

## 如何生成 qwen2 相同的 tokenzier 文件结构

- merges.txt
- vocab.json
- tokenizer_config.json
- tokenizer.json

详情见 `2_from_qwen2.py` 文件

## 如何生成 internlm2 相同的 tokenzier 文件结构

- special_tokens_map.json
- tokenizer_config.json
- tokenizer.json
- tokenizer.model
- tokenization_internlm2.py
- tokenization_internlm2_fast.py

详情见 `3_from_internlm2.py` 文件

## 说明
现在主流生成 tokenizer 方法的主要是 3 个

- huggingface 的 tokenizer 库
- SentencePiece 库
- openai 的 tiktoken 库

第三个库没有用过，不过从目前来看，前两个库结果都可以相互转换。如果是类似 llama3 直接用 huggingface 的 tokenizer 库训练的，那么其实非常简单，不用做啥转换，
如果是 SentencePiece 库训练的，那么需要做一次转换，写法非常灵活，能满足需求就行，其实很多文件可能都不是必要的，只是为了方面查看。

注意： vocab size 和模型 embeding size 其实不要求完全一致。

- 假设 vocab size 长度大于 embedding size，那么说明可能是词表里面预留了很多 UNK token id 方便后面扩展。正常情况下这些 id 不会输入到模型中，在没有扩展情况下
- 大部分情况下都是 vocab size 小于 embedding size，也就是说很多 embedding 位置没有用到，是为了扩充词表而不改变 embedding 用的
- 各种情况都可能出现，并没有统一范式


## 从头训练 tokenizer

https://huggingface.co/docs/tokenizers/index
https://huggingface.co/docs/tokenizers/pipeline

运行顺序：

- normalization 对输入字符进行前处理，例如去除奇怪字符，全部小写啥的
- pre-tokenization 预分词是将文本拆分成更小的对象的过程,这些对象为最终训练得到的词元设定了一个上限。一个好的思路是,预分词器会将你的文本拆分成"单词",然后你最终得到的词元将是这些单词的部分。
- model
- post-processing 后处理是分词流程的最后一步,在返回编码之前对其进行任何额外的转换,比如添加潜在的特殊标记。


normalization：

规范化,简而言之,就是一组你应用于原始字符串的操作,以使其变得更干净。常见的操作包括去除空白字符、删除带重音的字符或将所有文本转换为小写。如果你熟悉 Unicode 规范化,它也是大多数分词器中非常常见的规范化操作。

在 🤗 Tokenizers 库中,每个规范化操作都由一个 Normalizer 来表示,你可以通过使用 normalizers.Sequence 来组合多个这样的操作。

解码过程是： 将 ID 转换回词元(使用分词器的词汇表),然后删除所有特殊标记,最后用空格连接这些词元:


## 參考

https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt
https://zhuanlan.zhihu.com/p/657047389
https://medium.com/@awaldeep/hugging-face-understanding-tokenizers-1b7e4afdb154
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb


## 基础

https://zhuanlan.zhihu.com/p/631463712
https://zhuanlan.zhihu.com/p/656115828

Tokenizer 中文称为分词器，为何需要这个东西？原因是现在的模型还不能直接将人类字符串语言直接输入到模型中进行学习，需要一个编码转换过程。
因此 Tokenizer 作用就是将字符序列转化为数字序列，对应模型的输入。这个转换过程有非常多写法，但是哪一种转换范式才是我们需要的？

一个好的 tokenizer 能够将输入文本准确的切分为合适的单元。一个训练好的 tokenizer 会输出一个词汇表，代表切分后的字符单元和 id 的对应关系。
在使用时候，输入任何一句话，都会按照之前确定的切分规则，切分为单元，然后查表得到数字 id，这样就把一句话转换为一维向量。

一个好的 tokenizer 应该满足：

1、无损重构：分词结果应该可以无损还原为输入；
2、高压缩率：词表大小相同时，同一批数据的 tokens 数应该尽可能少；
3、语言无关：基于统计，训练和分词过程都不应引入语言特性；
4、数据驱动：可以直接基于原始语料进行无监督训练；
5、训练友好：能够在合理的时间和配置上完成训练过程。

无损重构是必备要求，因为如果你无法根据模型预测的id还原为字符串，那用户如何用呢？
高压缩率是希望词汇表大小刚刚好，不能太少也不能太大。根据这个原则，实际上 tokenizer 算法一般分成 3 大类：

1. word-based tokenizer
2. character-based tokenizer
3. subword-based tokenizer

word 级别切分的话，就是按单词来进行切分，通俗易懂。英文的话就是按照空格来进行切分，中文就是一条有意义的单元，例如你好
这样存在的一个很大问题就是词表的长度会变的很大，特别是对跨语言的模型来说，每种语言的word都不一样，需要维护的词库大小呈数量级增长，那么模型就要学习一个巨大的embedding matrix，增加了空间复杂度度和时间复杂度。

把每一个单词看成一个token，然后对其进行编号。这种编码方式十分符合人类语言习惯，因为人类语言也经常以单词为单位进行交流，但这并不一定是最合理的编码方式。
我们知道，一门语言中，通常有几万到几十万量级的单词数。若使用这种编码方式，在语言模型预测的时候需要在这个拥有几万个单词的列表上计算一个概率分布，那样的计算量是非常恐怖的，而且过大的token列表十分影响模型的预测准确度。

character 字符级别是以字符为最小粒度，对于英文而言是一个个英文字母，对于中文来说就是一个一个词。这样做的词表规模要小得多（通常为几十到几百），但是字符本身的意义不大，无法学习到单词的向量表征。

可以发现上述两者都属于两个极端，word 的词汇表超级大，而 character 的词汇表超级小。

首先我们应该知道在现在 LLM 场景，我们应该还是希望词汇表大一点好。因为通过增大Tokenizer的词表来提高压缩率，从而缩短序列长度、降低解码成本，是大家都喜闻乐见的事情。毕竟增大词表只需要增大Embedding层和输出的Dense层，这部分增加的计算量几乎不可感知，但缩短序列长度之后带来的解码速度提升却是实打实的。当然，增加词表大小也可能会对模型效果带来一些负面影响，所以也不能无节制地增加词表大小

增加词表大小的好处是显而易见的。由于LLM是自回归的，它的解码会越来越慢，而“增大词表 → 提高压缩率 → 缩短序列长度”，换言之相同文本对应的tokens数变少了，也就是解码步数变少了，从而解码速度提升了；

不过增大词表的缺点也很明显，最直接的就是会割裂token与token之间在字符层面之间的联系，从而可能会影响泛化，甚至会损失做某些任务的能力。比如“太阳能”和“太阳”都是词表中的一个词的话，模型是不知道“太阳能”是由“太阳”和“能”组成，也不知道“太阳”是“太”和“阳”，这样如果要做一些子词相关的任务就会比较艰难，比如最经典的问“‘太阳能’反过来怎么读？”，期望回答时“能阳太”，但由于模型不知道它是“太”、“阳”、“能”三个字组成，从而很难回答正确。

比如当“太阳能”和“太阳”都成为了一个独立的token时，用户输入“太阳”后，接下来续写的字就基本不会是“能”了，这可能不符合用户的分布期望；又比如“白云”、“白云山”、“白云机场”都是一个独立的token时，用户输入“广州的白云”后，接下来也几乎不会续写出“广州的白云机场”、“广州的白云山”，等等。

现在 llm 模型用的全部是 subword 分词方法，因为可以兼顾上述两者的优点。以子词为最小粒度（例如flying会被分为fly和ing两部分），是对前两种粒度的一种平衡。遵循的原则是**尽量不切分常用词，生僻词应该拆分成子词以共享token压缩空间**
这种方案一定程度的保留了语义独立性的同时还能保证词表不会太大，是相对较优的方案。transformer 类的模型使用的是这种方式。

传统构造词表的方法，是先对各个句子进行分词，然后再统计并选出频数最高的前N个词组成词表。通常训练集中包含了大量的词汇，
以英语为例，总的单词数量在17万到100万左右。出于计算效率的考虑，通常N的选取无法包含训练集中的所有词。
因而，这种方法构造的词表存在着如下的问题：

1. 实际应用中，模型预测的词汇是开放的，对于未在词表中出现的词(Out Of Vocabulary, OOV)，模型将无法处理及生成；
2. 词表中的低频词/稀疏词在模型训练过程中无法得到充分训练，进而模型不能充分理解这些词的语义；
3. 一个单词因为不同的形态会产生不同的词，如由"look"衍生出的"looks", "looking", "looked"，显然这些词具有相近的意思，但是在词表中这些词会被当作不同的词处理，一方面增加了训练冗余，另一方面也造成了大词汇量问题。

一种解决思路是使用字符粒度来表示词表，虽然能够解决OOV问题，但单词被拆分成字符后，一方面丢失了词的语义信息，另一方面，模型输入会变得很长，这使得模型的训练更加复杂难以收敛。针对上述问题，Subword(子词)模型方法横空出世。它的划分粒度介于词与字符之间，比如可以将”looking”划分为”look”和”ing”两个子词，而划分出来的"look"，”ing”又能够用来构造其它词，如"look"和"ed"子词可组成单词"looked"，因而Subword方法能够大大降低词典的大小，同时对相近词能更好地处理。

现在主流的 subword 切词方法主要是 BPE（Byte-Pair Encoding，字节对编码法），WordPiece，SentencePiece，Unigram，其中前3种最常用，并且第一种应用最广泛。

## BPE 分词方法 
https://zhuanlan.zhihu.com/p/191648421  

GPT 系列用的就是这个，最常用。

它的目标是找到一种最优的字符组合方式，使得整个数据集中不同单词的字符组合尽可能的少。这种算法最初被设计用于字节级的数据压缩，后来被应用于NLP。

BPE获得 Subword 的步骤如下：

- 准备足够大的训练语料，并确定期望的Subword词表大小；
- 将单词拆分为成最小单元。比如英文中26个字母加上各种符号，这些作为初始词表；
- 在语料上统计单词内相邻单元对的频数，选取频数最高的单元对合并成新的Subword单元；
- 重复第3步直到达到第1步设定的Subword词表大小或下一个最高频数为1.

一个更具体的案例为：

1. 统计语料中输入中所有出现的单词并在每个单词后加一个单词结束符</w> -> ['hello</w>': 6, 'world</w>': 8, 'peace</w>': 2]
2. 将所有单词拆成单字 -> {'h': 6, 'e': 10, 'l': 20, 'o': 14, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
3. 合并最频繁出现的单字(l, o) -> {'h': 6, 'e': 10, 'lo': 14, 'l': 6, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
4. 合并最频繁出现的单字(lo, e) -> {'h': 6, 'lo': 4, 'loe': 10, 'l': 6, 'w': 8, 'r': 8, 'd': 8, 'p': 2, 'a': 2, 'c': 2, '</w>': 3}
5. 反复迭代直到满足停止条件

在获得子词词表后，就可以将句子分割成子词了

```text
输入单词序列
["the</w>", "highest</w>", "mountain</w>"]

# 从一个很大的corpus中排好序的subword表如下
# 长度 6         5           4        4         4       4          2
["errrr</w>", "tain</w>", "moun", "est</w>", "high", "the</w>", "a</w>"]

# 迭代结果
"the</w>" -> ["the</w>"]
"highest</w>" -> ["high", "est</w>"]
"mountain</w>" -> ["moun", "tain</w>"]
```

注意，在上述算法执行后，如果句子中仍然有子字符串没被替换但所有subword都已迭代完毕，则将剩余的子词替换为特殊token，如<unk> 。从这里大家也可以发现了，原则上<unk>这个token出现的越少越好，所以我们也往往用<unk>的数量来评价一个tokenizer的好坏程度，这个token出现的越少，tokenizer的效果往往越好。

https://zhuanlan.zhihu.com/p/191648421 中也有一个详细的例子。
https://zhuanlan.zhihu.com/p/639701636 有代码
https://zhuanlan.zhihu.com/p/424631681  
https://leimao.github.io/blog/Byte-Pair-Encoding/  

中文 bpe 是咋做。
https://www.zhihu.com/question/600945415/answer/3058758758 

Byte-level BPE（BBPE）

BBPE整体和BPE的逻辑类似，不同的是，粒度更细致，BPE最多做到字符级别，但是BBPE是做到byte级别，按照unicode 编码作为最小粒度。

chatgpt等部分大语言模型使用的就是这种tokenize方式，所以对于中文调用API收费时token数并不是指按字收费。同时在使用chatgpt时你的Prompt即使限定了输出的字数，但是和实际的输出字数也存在一些差异，tokenize也是影响因素之一。

优点： 不会出现 OOV 的情况。不管是怎样的汉字，只要可以用 unicode 表示，就都会存在于词表中。
缺点： 增加了学习的成本，对于中文还需要学习 unicode 的组合，会导致模型在训练不够充足的时候，会输出一些乱码（不合法的 unicode 序列）。

## 词汇扩充


## 微调

在第三章，我们研究了如何在给定任务上微调模型，但是我们可以发现 tokenizer 我们是直接使用的，没有进行训练，这会存在不足。因为使用在来自其他领域或语言的语料库上预训练的标记器通常不是最理想的。 例如，在英语语料库上训练的标记器在日语文本语料库上表现不佳，因为两种语言中空格和标点符号的使用非常不同。
假设我们要微调的语言不太一样，那么我们就需要重新训练 tokenizer，这就是 TOKENIZERS 库的作用。它是一个用于训练和使用自定义分词器的库，它可以让我们轻松地训练自己的分词器，然后将其与 Hugging Face 的 Transformer 库一起使用。

标记器需要仔细查看语料库中的所有文本——我们称之为 training 的过程。

注意：训练标记器与训练模型不同！模型训练使用随机梯度下降使每个batch的loss小一点。它本质上是随机的（这意味着在进行两次相同的训练时，您必须设置一些随机数种子才能获得相同的结果）。训练标记器是一个统计过程，它试图确定哪些子词最适合为给定的语料库选择，用于选择它们的确切规则取决于分词算法。它是确定性的，这意味着在相同的语料库上使用相同的算法进行训练时，您总是会得到相同的结果。

### 1 微调 tokenizer

可以先看一个简单的训练例子

```python
import os
# 会自动下载到  ~/.cache/huggingface/dataset, 您可以通过设置 HF_HOME 环境变量来自定义缓存的文件夹
# 必须要放到 from datasets import load_dataset 前面，否则无效
os.environ['HF_HOME'] = '../'  # 所有缓存都会放到这个路径下

from datasets import load_dataset
# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("espejelomar/code_search_net_python_10000_examples", "python")
print(raw_datasets["train"])
print(raw_datasets["train"][3456]["whole_func_string"])

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        yield samples["whole_func_string"]
        
from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
print(len(old_tokenizer))  # 50257

training_corpus = get_training_corpus()
# 训练出一个新的 tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokens = tokenizer.tokenize(example)
print(len(tokens))  # 27
print(len(old_tokenizer.tokenize(example)))  # 44

print(len(tokenizer))  # 52000
# 可以发现总的词汇表发现了变化(保存到了 merges.txt 里面), finetune 的时候应该要改输入 embedding 参数，否则可能越界, 可以看 demo_2
tokenizer.save_pretrained("code-search-net-tokenizer")
```

训练完成后，词汇表肯定是发生变化了，此时如果连接上模型进行推理，可能会出现越界的情况，因为模型的输入 embedding 参数是固定的，所以需要重新训练模型。

如果你是简单的新增 token，则有快捷办法，无需训练，但是不清楚上面的情况下是否一定要重新训练 embedding 

```python
import os
os.environ['HF_HOME'] = '../'  # 所有缓存都会放到这个路径下

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "c_1四处张望。"
print(text)  # c_1四处张望。
# ['c', '_', '1', '四', '[UNK]', '[UNK]', '[UNK]', '。']  c_1 会切分为三个单词，后续解码时候也是变成三个单词，而不是一个单词，也就是说有缺陷
print(tokenizer.tokenize(text))
print(tokenizer.encode(text))
# [CLS] c _ 1 四 [UNK] [UNK] [UNK] 。 [SEP]
print(tokenizer.decode(tokenizer.encode(text)))

# 可以通过新增 token 来解决这个问题
characters = ["c_1"]
print(tokenizer.vocab_size) # 30522

tokenizer.add_tokens(characters)

print(tokenizer.vocab_size) # 30522
# 可以发现实际上打印的词汇表并没有改变，原因是他是单独存放的，而不是合并到词汇表
tokenizer.save_pretrained('aa')

# 会生成一个新的 added_tokens.json 文件
print(len(tokenizer))  # 长度+1

# ['c_1', '四', '[UNK]', '[UNK]', '[UNK]', '。']
print(tokenizer.tokenize(text))
# [CLS] c_1 四 [UNK] [UNK] [UNK] 。 [SEP]
print(tokenizer.decode(tokenizer.encode(text)))

# 这个步骤必须，因为总的词汇表变了，需要对新增的部分随机初始化
# 模型需要调用 resize_token_embeddings，预训练的 Embedding 不变，添加的 token 随机初始化进 Embedding 矩阵中。
# 假设原先词汇表大小是 100，维度是 512,那么原先的 embeding 维度就是 100x512, 现在增加了 10 个 token，那么新的 embeding 维度就是 110x512
# 但是原先的 100 个 token 的 embeding 是预训练好的，所以不需要改变，只需要把新增的 10 个 token 的 embeding 随机初始化进去就可以了
model = BertModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(tokenizer))
```

`resize_token_embeddings` 的过程非常暴力，就是直接 copy+剩下的随机初始化。


