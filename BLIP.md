# BLIP 和 BLIP2 和 InstructBLIP

## 前置

在开始多模态前，可以先看看 B 站李沫老师的多模态串讲视频，可以事半功倍。下面是视频笔记。


## BLIP
论文： BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation   
论文地址：https://arxiv.org/abs/2201.12086      
代码地址：https://github.com/salesforce/BLIP      
官方博客： https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/   

- 提出一个统一的视觉语言模型,可以同时完成理解和生成任务。而之前大部分都是无法在一个模型中完成
- 大部分 SOTA 算法都是直接使用了网络爬取的数据，但是实际上这些数据都是有噪声的，有结果显示有噪声的网络文本对于视觉语言学习来说只能得到次优的结果。

基于上述两点，作者提出了一个统一的混合视觉和语言多模态算法，可以同时完成理解和生成任务。并且提出了一种 数据集 Bootstrapping 策略来减少噪声的影响。

**要想非常清楚的理解文本部分，需要熟悉 BERT 技巧下游任务做法。我们会在 BERT 源码里面详细说明**

### 原理

网络结构和训练 loss 如下所示:

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/579b00fb-cfa2-452b-a33b-74ca48206728"/>
</div>

整个架构包括 4 个大模块，虽然模型是一个统一的多模态模型，但是实际上依然遵循 **多个任务预训练+ 特定任务微调的范式**，而非 zer0-shot (不过有一定的视频语言任务的 zero-shot 能力)

先来看不同模块能做什么事情，再来分析不同模型如何预训练，最后如何微调。

1. ImageEncoder 即 ViT 模块，负责对图片进行编码，输出单尺度 image embedding 特征
2. TextEncoder 即 BERT 不包括下游 task 的基础模块，负责对文本进行编码， 1 + 2 其实就是 CLIP 算法，可以实现 zero-shot 的图片分类任务或者说图文检索。会追加 [CLS] 符号在文本开头
3. Image-groundedText encoder 也是采用了 BERT 模型，只不过相比于 2，是一个匹配任务而非对比任务，该编码器会接收 ImageEncoder 的输出并通过交叉注意力模块进行交互，输出当前图片和文本是否匹配，是一个二分类任务，也可以实现类似 2 的效果
4. Image-groundedText decoder 也是采用了 BERT 模型，不过因为是解码任务，采用的自注意力模块是 Causal Self-Att (Causal 因果，其实就是自回归，其实只是训练时候有差别，推理和 3 没有区别)，防止训练时候模型看到整个句子。这个模块可以实现生成任务，例如 captioning，VQA 等

需要注意图片的颜色，相同颜色表示权重共享哦。因此并不是有 4 个独立的模块，实际上 2/3/4 共享了很多模块。

ViT 部分没有啥写的，我们来看下 2/3/4 模型结构差异。在代码层面作者实际上 2/3/4 是直接用的 HF 的 BertModel 来构建 BERT 模块，文本相关的 loss 计算和复杂计算其实都是这个代码本身处理的，作者代码没有进行啥特殊操作。

2. TextEncoder 即为 N 个不包括 crossattention 的 BERT 模型，输出的是文本特征，没有和图片特征进行交互。在计算 loss 时候和 CLIP 一样计算对比 loss 即可 (itc loss)
3. Image-groundedText encoder即为 N 个包括 crossattention 的 BERT 模型，crossattention 接受 image embedding，输出是一个二分类任务，采用的 loss 为 bce (itm loss)
4. Image-groundedText decoder即为 N 个包括 crossattention 的 BERT 模型，crossattention 接受 image embedding。输出是一个生成的自回归任LM任务。采用的 loss 为 crossentropy (llm loss)

3 和 4 的 Bi Self-Att 和 Causal Self-Att 自注意力层结构是一样的只不过计算方式不一样，会在 BERT 里面说明

itc 和 itm 看起来是同一个任务，只是侧重点不同。

Image-Text Matching Loss 和 Image-Text Contrastive Loss 是两种不同的损失函数，用于训练视觉-语言模型中的编码器。它们的主要区别在于损失函数的形式和训练方式。
Image-Text Matching Loss 通常用于训练图像引导的文本编码器，其目的是通过学习图像-文本之间的相似度来训练模型。具体来说，给定一组图像和相应的文本描述，该损失函数将正样本（即图像和文本描述之间有语义关联的样本）与负样本（即图像和文本描述之间没有语义关联的样本）进行区分。通常，这种损失函数采用交叉熵损失函数或二元交叉熵损失函数。
Image-Text Contrastive Loss 也用于训练图像引导的文本编码器，其目的是通过学习图像-文本之间的距离来训练模型。具体来说，该损失函数将正样本（即图像和相应文本描述）从负样本（即图像和不相关文本描述）中分离出来，并使它们在特征空间中彼此更接近。通常，这种损失函数采用余弦相似度或欧几里得距离等度量来计算图像-文本之间的距离，并使用对比损失函数（如 triplet loss）来训练模型。
因此，Image-Text Matching Loss 和 Image-Text Contrastive Loss 的区别在于对正负样本之间的关系处理方式不同。前者是通过区分正负样本来训练模型，后者则是通过将正样本聚集在一起并与负样本分离来训练模型。

按照论文的说法，其实是关注的粒度不一样。 Image-Text Matching Loss 是要求正样本和负样本要区分开，而 Image-Text Contrastive Loss 是希望正样本对拉进，负样本对推远。

上面介绍预训练和模型的整体结构。

### 下游任务

基于预训练完成后，需要对不同的下游任务构建新的 Head 或者直接微调。

下游任务包括：

**(1) Image-Text Retrieval**

无需新增结构，只需要微调 2/3 模型即可即 using ITC and ITM losses on COCO and Flickr30K. 为了实现更快的推理速度，我们遵循Li等人(2021a)的方法，首先根据图像-文本特征相似度选择 k 个候选对象，然后根据候选对象的成对 ITM 分数对其重新排序。我们为 COCO 设置k = 256，为 Flickr30K 设置k = 128。

**(2) Image Captioning**

也无需新增结构，只需要训练 4 即可。使用了 NoCaps (Agrawal et al., 2019) and COCO 数据集 with the LM loss. , we add a prompt "a picture of" at the beginning of each caption, which leads to slightly better results.

**(3) Visual Question Answering**

需要稍微重新组织下模型结构，但无需新增结构

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/383e7ecf-e20b-43a3-b1ae-4bdf8101c46a"/>
</div>

VQA 要求模型在给定图像和问题的情况下预测答案。是一个多答案分类任务。
在微调期间，我们重新排列预训练的模型，其中 image-question 首先被编码为多模态嵌入，然后给出答案解码器。The VQA model is finetuned with the LM loss using ground-truth answers as targets.

**(4) Natural Language Visual Reasoning (NLVR2)**

NLVR2 要求模型预测一个句子是否描述了一对图像。

**(5) Visual Dialog (VisDial)**

VisDial 在自然对话环境中扩展了VQA，其中模型不仅需要根据图像-问题对预测答案，还需要考虑对话历史和图像的标题。


### 下游任务推理过程

**(1) 图文匹配**

```python
from models.blip_itm import blip_itm

image_size = 384
image = load_demo_image(image_size=image_size,device=device)

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    
model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device='cpu')

caption = 'a woman sitting on the beach with a dog'

print('text: %s' %caption)

itm_output = model(image,caption,match_head='itm')
itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
print('The image and text is matched with a probability of %.4f'%itm_score)

itc_score = model(image,caption,match_head='itc')
print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

# load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
# text: a woman sitting on the beach with a dog
# The image and text is matched with a probability of 0.9960
# The image feature and text feature has a cosine similarity of 0.5262
```

核心代码

```python
def forward(self, image, caption, match_head='itm'):
    image_embeds = self.visual_encoder(image) 
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
  
    text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                          return_tensors="pt").to(image.device) 
    
    # 论文里面应该是说 2 模型会加入 [CLS] 作为开始符号，3 模型会加入 [ENC] 为了开始标注，但是好像是一样的？
    if match_head=='itm':
        output = self.text_encoder(text.input_ids,
                                   attention_mask = text.attention_mask,
                                   encoder_hidden_states = image_embeds,
                                   encoder_attention_mask = image_atts,      
                                   return_dict = True,
                                  )
        # 可以看出 itm 会多加一个 Head 进行二分类
        itm_output = self.itm_head(output.last_hidden_state[:,0,:])     
        return itm_output
        
    elif match_head=='itc':
        # 和 clip 完全一样
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                     
                                        return_dict = True, mode = 'text')                     
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    
        
        sim = image_feat @ text_feat.t()
        return sim


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'}) # 解码开始特殊符号
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})  # 编码特殊符好
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer
```


**(2) 图文描述**

```python
# Image Captioning
from models.blip import blip_decoder
device = 'cuda:0'
image_size = 384
image = load_demo_image(image_size=image_size, device=device)

model_url = 'model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')

model.eval()
model = model.to(device)
with torch.no_grad():
    # beam search
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    print('caption: ' + caption[0])
```

核心源码

```python
def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
    image_embeds = self.visual_encoder(image)
    if not sample:
        image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
    
    # self.prompt： a picture of 
    prompt = [self.prompt] * image.size(0)
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
    
    # 最开始追加特殊解码符号 [DEC]
    input_ids[:,0] = self.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1] 
    
    if sample:
        #nucleus sampling 
        # 在 CapFilt 中核采样产生了更多样化和令人惊讶的说明，其中包含更多模型可以从中受益的新信息，作者实验表明这个方法比 beam search 效果好
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                              max_length=max_length,
                                              min_length=min_length,
                                              do_sample=True,
                                              top_p=top_p,
                                              num_return_sequences=1,
                                              eos_token_id=self.tokenizer.sep_token_id,
                                              pad_token_id=self.tokenizer.pad_token_id, 
                                              repetition_penalty=1.1,                                            
                                              **model_kwargs)
    else:
        #beam search
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                              max_length=max_length,
                                              min_length=min_length,
                                              num_beams=num_beams,
                                              eos_token_id=self.tokenizer.sep_token_id,
                                              pad_token_id=self.tokenizer.pad_token_id,     
                                              repetition_penalty=repetition_penalty,
                                              **model_kwargs)            
        
    captions = []    
    for output in outputs:
        caption = self.tokenizer.decode(output, skip_special_tokens=True)    
        captions.append(caption[len(self.prompt):])
    return captions
```

**(3) visual question answering**

```python
from models.blip_vqa import blip_vqa

image_size = 480
image = load_demo_image(image_size=image_size, device=device)     

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    
model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

# 对输入图片进行提问，模型回答问题
question = 'where is the woman sitting?'

with torch.no_grad():
    answer = model(image, question, train=False, inference='generate') 
    print('answer: '+answer[0])

# answer: on beach
```

代码有点长，就不贴了。

### CapFilt

算是一个带噪声数据前处理自举过程吧，首先利用带噪声数据进行预训练，然后将 Image-groundedText Encoder 作为 Filter 模块，而 Image-groundedText Decoder 作为 Captioner 模块，相当于构建了一个新的下游模型，
然后仅仅利用 COCO 数据集进行微调，训练完成后，就可以对带噪声数据进行 Captioning和 Filtering 得到去噪的新数据。得到新数据集后再重新预训练模型即可，不断的重构这个过程即可。

也就是 先预训练，然后 CapFilt，在预训练，再  CapFilt 不断 Bootstrapping，觉得差不多后就可以真的训练下游任务了。

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/2069cb5f-3fe6-4473-9670-65989dea7016"/>
</div>

详细流程图如下所示;

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/c1add1dc-dd96-40e0-8de4-02fa3cdf806f"/>
</div>

## BLIP-2

论文： BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
地址： https://arxiv.org/abs/2301.12597
github: https://github.com/salesforce/LAVIS/tree/main/projects/blip2

BLIP 和其他多模态算法一样，预训练成本太高了。如果可以利用强大的预训练好的视觉和语言模型，并且估计这些大模型，而仅仅训练连接层，那将可以极大的提升训练效率。

BLIP-2采用了一种通用且高效的预训练策略，它从现成的固定的预训练图像编码器和固定的大型语言模型中引导视觉语言预训练。BLIP-2使用轻量级 Q-Former 弥补了模式上的差距，该转换器分两个阶段进行预训练。

1. 第一阶段从固定图像编码器中引导视觉语言表示学习，强制 Q-Former 学习与文本最相关的视觉表示
2. 第二阶段从一个固定的语言模型中引导视觉到语言的生成学习，使其输出的视觉表示可以被LLM解释。

尽管比现有方法具有更少的可训练参数，但 BLIP-2 在各种视觉语言任务上实现了最先进的性能。

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/5d945855-8db4-432e-91c8-735fcac3eee9"/>
</div>

### 原理说明

从上面简要的结构图可以大概看出原理。

- Q-Former 算一个模态桥接器，用户将视觉特征转换为后续 LLM 能够理解的语言特征，如果 Q-Former 也同时输入了 text，那么 text 算是一个辅助 query, 首先将 text 特征和可学习 queries 进行信息融合，然后再将视觉特征转换为后续 LLM 能够理解的语言特征
- Q-Former 输出语言特征后，输入到 LLM 中即可进行自回归进行生成任务，如果是检索任务，则不需要 LLM，直接将 Q-Former 的输出，然后进行检索即可
- 可学习 queries 学习全局信息，同时还可以不同规模的视觉编码器和 LLM 都是统一大小输入和输出，因为不管模型多大，可学习 queries 输出维度是固定的

**Q-Former 实际上就是 BERT 模型。**

下面分析两个阶段的预训练过程。

**(1) 第一阶段预训练**

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/456deee4-efa4-4b6c-a39c-43775100b318"/>
</div>

可以发现和 BLIP 训练过程非常类似。也是包括了 3 个 loss

1. 图片输入到固定的 ViT 中抽取视觉特征
2. 对于 Image-Text Contrastive Learning Loss 分支， 将可学习 queries 输入到 Q-Former 的自注意力层中，然后将 text 也输入到 Q-Former 的自注意力层中，注意： 对比学习是希望视觉特征和文本特征对齐，
如果在计算自注意力时候，存在 queries 和 text 之间的计算，那么必然会导致 text 信息泄露进来，后续和图像特征进行交叉注意力时候，其实就存在信息泄露了，这违背了对比学习过程，其实就变成了后面的匹配学习。 故为了避免信息泄露，在计算
自注意力时候，只能让 queries 和 queries 之间计算，text 和 text 之间计算，不能让 queries 和 text 进行计算。 如果你依然不能理解，那么可以看下 BLIP 的 Image-Text Contrastive Learning Loss 过程
3. 对于 Image-TextMatching loss 则没有上述限制，因为图片匹配学习本身就是需要融合，text 信息和 queries 都是平等的
4. 对于 Image-GroundedText Generation loss，因此这是生成任务，输入的 text 是整个句子，当前 text 在计算自注意力时候不能知道后面的词，否则就是信息泄露了，例如 text =a cat wearing sunglasses 时候，如果要
计算 cat 后面的单词 wearing，那么 cat 这个词计算自注意力时候只能和前面的 a 单词计算，不能和后面的 wearing sunglasses 计算，那就是数据泄露了，答案都知道了还训练啥。

如果不太理解这个 mask 的操作，可以回去看下 BLIP 的 预训练过程，应该就会有清晰的理解。

**(2) 第二阶段预训练**

<div align=center>
<img src="https://github.com/salesforce/BLIP/assets/17425982/82d72627-0dfb-4239-8bea-fb2bfd154a89"/>
</div>

这个训练过程就比较清晰易懂了。针对两种不同的 LLM 模型，有两种不同的训练方式。

