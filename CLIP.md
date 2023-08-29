# CLIP 解读

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233788808-1f2b26bd-82f0-47f5-88f1-7526dba94c0e.png"/>
</div>

论文题目： Learning Transferable Visual Models From Natural Language Supervision  
论文地址： https://arxiv.org/abs/2103.00020   
官方地址： https://github.com/openai/CLIP (没有训练代码)   
Fork 并注释版本： https://github.com/hhaAndroid/CLIP/tree/hha  

CLIP 带动了视觉多模态的繁荣，可以说是第一篇真正意义上的成功进行大规模图文多模态训练然后直接 zero-shot 用于下游任务的算法，其泛华性极强。

从上图可以看出，其训练过程非常朴素，对图文对采用对比学习即可，训练完成后即可进行 zero-shot 图像分类。

由于这个 repo 没有开源训练代码，因此我们重点研究推理过程，关于论文细节在分析完代码后给出。

## 1 推理代码解读

官方给出了一个 8 分类的 zero-shot 图像分类的 notebook 例子。 

**(1) 基于类别构建 text prompt** 

这个部分非常重要，构建一个好的 text prompt 会严重影响分类性能, 必须要给予足够重视。

为了简单和足够通用性，我们可以先采用一种简单的方式将类别转化为 prompt 即

任何类别名都变成 `a photo of a {类别名}` 即

```python
descriptions = {
    "page": "a photo of a page",
    "chelsea": "a photo of a chelsea",
    "astronaut": "a photo of a astronaut",
    "rocket": "a photo of a rocket",
    "motorcycle_right": "a photo of a motorcycle_right",
    "camera": "a photo of a camera",
    "horse": "a photo of a horse", 
    "coffee": "a photo of a coffee"
}
```

**(2) 将这些类别 descriptions 通过 CLIP 中使用的 tokenizer 对输入的文本进行编码**

因为模型无法直接处理文本，因此需要将输入文本先进行分词，然后结合词汇表转换为索引 id，即完成了文本编码过程。核心代码如下

```python
# 输出 (8,77)，内部是 id
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
# 输出 (8,512)，代表每个句子的文本编码向量
text_features = model.encode_text(text_tokens).float()
```
text_tokens 输出维度是 (8,77) 即一共 8 个句子，每个句子最多支持 77 个 token 输入，如果不够就后续补 PAD id(默认是 0)。text_features 是 (8,512)，每个句子用 512 维度向量表示。

特别注意： 每个句子都加了开始符和结束符，因此实际上你能输入的 token 最多是 75 个。

`clip.tokenize` 计算过程为：

```python
# 输入 8 个句子
if isinstance(texts, str):
    texts = [texts]

# 添加开始和结束符，并进行编码
sot_token = _tokenizer.encoder["<|startoftext|>"]  # 49406
eot_token = _tokenizer.encoder["<|endoftext|>"]  # 49407
# 文本进行编码，得到所有 tokens
all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

# 每个句子的 token如果大于77，则截断，但是 eot 是肯定要保留的
# 每个句子的 token 如果小于 77，则后面补 0
if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
else:
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
for i, tokens in enumerate(all_tokens):
    if len(tokens) > context_length:
        if truncate:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
    result[i, :len(tokens)] = torch.tensor(tokens)
return result
```

model.encode_text 的内部计算过程为：

```python
def encode_text(self, text):
    # 利用 id 提取训练好的 embedding -> (8, 77, 512)
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # (8,77,512) 提前每个句子的结束符对应位置的预测值即可
    #  text.argmax(dim=-1) 就是结束符位置
    # 从而得到 (8, 512) 然后进行投影，输出 (8, 512)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x
```

**(3) zero-shot 图像分类**

分类过程如上图所示，将图像编码向量和所以类别的文本编码向量归一化后计算余弦相似度，然后 softmax，当前图片和哪个文本的编码向量最相似就属于那个类。

```python
image_features = model.encode_image(image_input).float()

# 归一化，计算余弦相似度
# (8,512)
image_features /= image_features.norm(dim=-1, keepdim=True)
# (8,512)
text_features /= text_features.norm(dim=-1, keepdim=True)
# (8, 8)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
```

图片编码器就是一个常用的 ViT 模型，计算过程比较简单

```python
def forward(self, x: torch.Tensor):
    x = self.conv1(x)  # input shape = [8, 3, 224, 224]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [8, 768, 49]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    # 在前面加上分类头的 embedding  ->  [8, 49 + 1, 768]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
    
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_post(x[:, 0, :])  # 取分类头 embedding 对应的输出
    if self.proj is not None:
        x = x @ self.proj
    return x  # (8,512)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233835070-6b962c84-963c-4636-b8b0-bfc55904b216.png"/>
</div>

有 6 个是分类正确的

可以简单的修改下 descriptions，看看分类效果

```python
descriptions = {
    "page": "This is a photo of a page",
    "chelsea": "This is a photo of a chelsea",
    "astronaut": "This is a photo of a astronaut",
    "rocket": "This is a photo of a rocket",
    "motorcycle_right": "This is a photo of a motorcycle_right",
    "camera": "This is a photo of a camera",
    "horse": "This is a photo of a horse", 
    "coffee": "This is a photo of a coffee"
}
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233835262-d925032b-43da-4c1d-96c6-dbdd75a38e1b.png"/>
</div>

可以发现对每个句子加上一个 This is, 模型的输出值就变了。

我们再次考虑一个非常好的描述，如下所示：

```python
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}
```

可以发现效果就无敌了，全对。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233835380-fa97871a-5f75-4ef5-bc4a-8289224c754c.png"/>
</div>

实际上如果真正用于分类，需要进行 softmax 计算，并且由于作者训练了 logit_scale，实际上的推理代码应该是：

```python
with torch.no_grad():
    # 对图片进行编码
    image_features = model.encode_image(image_input).float()
    # 对本文序列进行编码 (8, 512)
    text_features = model.encode_text(text_tokens).float()
    
    # 归一化，计算余弦相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    similarity = logits_per_image.t()
    
    similarity = torch.softmax(similarity, dim=-1).cpu().numpy()
```

虽然不影响准确率，但是分值会高很多。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233835999-abc62f97-75ca-41df-9976-d0025604b827.png"/>
</div>

这充分反应了 text prompt 的重要性。现如今对于一个已经发布的大模型，掌握了 prompt 就掌握了财富！！！

作者在官方代码里面也提供了 notebooks/Prompt_Engineering_for_ImageNet.ipynb，内部提供了多个 prompt 进行集成测试，可以提高准确率。部分 prompt 如下：

```python
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
]
```

## 2 tokenizer 

tokenizer 是 SimpleTokenizer，就是朴素的 BPE 子词分词法。在大量词汇上统计好的词汇表位于 `clip/bpe_simple_vocab_16e6.txt.gz`，在实际使用的时候只需要采用同样的 BPE 进行切词，然后查表转换为 id 即可。

```python
def encode(self, text):
    bpe_tokens = []
    text = whitespace_clean(basic_clean(text)).lower()
    # 单词粒度切分实际上就是模式匹配即可
    for token in re.findall(self.pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        # 对每个单词在使用 bpe 计算，看下是否要切分为子词
        bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
    return bpe_tokens
def decode(self, tokens):
    # 对每个最小粒度的 token 进行解码
    text = ''.join([self.decoder[token] for token in tokens])
    # 如果多个 token 才组成一个词，则需要继续处理
    text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
    return text
```

下面结合具体例子来直观理解

```python
from clip.simple_tokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer()
text=''''''
bpe_tokens, bpe_text = tokenizer.encode(text)
print('bpe 分词后的结果:', bpe_text)
print('token 2 id:', bpe_tokens)
str1 = tokenizer.decode(bpe_tokens)
print('得到句子：', str1)
```

我们对 SimpleTokenizer 进行简单改造，以便理解计算过程

```python
text = "Using a Transformer network is simple"

# bpe 分词后的结果: ['using</w>', 'a</w>', 'transformer</w>', 'network</w>', 'is</w>', 'simple</w>']
# token 2 id: [1996, 320, 38235, 3304, 533, 4129]
# 解码每个子 token： using</w>a</w>transformer</w>network</w>is</w>simple</w>
# 得到句子： using a transformer network is simple 
```

可以看出每个词都是一个单词，没有子词。

```python
text = "Transformers support framework interoperability between PyTorch, TensorFlow, and JAX"

# bpe 分词后的结果: ['transformers</w>', 'support</w>', 'framework</w>', 'interoper', 'ability</w>', 'between</w>', 'py', 'torch</w>', ',</w>', 'ten', 'sor', 'flow</w>', ',</w>', 'and</w>', 'jax</w>']
# token 2 id: [17843, 1425, 13195, 45754, 3024, 1957, 4005, 15820, 267, 1149, 1852, 5608, 267, 537, 12390]
# 解码每个子 token： transformers</w>support</w>framework</w>interoperability</w>between</w>pytorch</w>,</w>tensorflow</w>,</w>and</w>jax</w>
# 得到句子： transformers support framework interoperability between pytorch , tensorflow , and jax 
```

可以发现 PyTorch 被分词了两个子词，解码时候也正确还原了，但是无法区分大小写。

## 3 HF 中的 CLIP

https://huggingface.co/openai/clip-vit-base-patch32

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
```

pipeline 也已经支持

```python
from transformers import pipeline

classifier = pipeline(model="openai/clip-vit-large-patch14")
classifier(
    "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    candidate_labels=["animals", "humans", "landscape"],
)

classifier(
    "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    candidate_labels=["black and white", "photorealist", "painting"],
)
```

## 3 论文重点记录

- 数据集由于不够，因此是自己构建的，包含了4亿个从互联网上多个公开来源收集来的（图像，文本）对。
- 作者最终采用对比学习方式训练，是因为发现这种训练效果最高，其余方式会低一些。简单来说就是： jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N 2 − N incorrect pairings.
- 由于数据集太大，不太可能会过拟合，因此 We train CLIP from scratch without initializing the image encoder with ImageNet weights or the text encoder with pre-trained weights.
- 关于 text transformer 部分描述为： 作为base 模型，我们使用一个具有8个注意头的63m参数的12层512宽模型。transformer在 49152 词汇表大小的文本的小写字节对编码(BPE)表示上运行。为了计算效率，最大序列长度被限制在76。文本序列用[SOS]和[EOS] token 括起来。[EOS]令牌上transformer最高层的激活被视为文本的特征表示，文本被层归一化，然后线性投影到多模态嵌入空间中。
- 文本编码器中使用了mask 自注意力，以保留使用预先训练的语言模型进行初始化的能力。意思应该是说由于文本编码器是使用的 BERT，并且用于预训练权重，因此也使用 [MASK] 这种训练方式。具体要看开源训练代码了。
- 我们使用非常大的32,768个小批量。混合精度用于加速训练和节省内存。为了节省额外的内存，梯度检查点、半精度Adam 统计和半精度随机四舍五入文本编码器权重。嵌入相似度的计算也被分片，单个gpu只计算其局部批次嵌入所需的成对相似度的子集。可以看出要想训练起来需要不少 trick


## Improved baselines for vision-language pre-training

论文地址：https://arxiv.org/pdf/2305.08675.pdf
未开源





