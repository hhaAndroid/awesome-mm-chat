# 参考

https://zhuanlan.zhihu.com/p/350017443

# GPT 1

https://static.aminer.cn/upload/pdf/1319/1601/76/5f8eab579e795e9e76f6f6a0_0.pdf
Improving Language Understanding by Generative Pre-Training

在GPT-1之前（和ELMo同一年），传统的NLP模型往往使用大量的数据对有监督的模型进行任务相关的模型训练,这必然存在非常多问题。

GPT-1的思想是先通过在无标签的数据上学习一个生成式的语言模型，然后再根据特定任务进行微调，处理的有监督任务包括

- 自然语言推理（Natural Language Inference 或者 Textual Entailment）：判断两个句子是包含关系（entailment），矛盾关系（contradiction），或者中立关系（neutral）；
- 问答和常识推理（Question answering and commonsense reasoning）：类似于多选题，输入一个文章，一个问题以及若干个候选答案，输出为每个答案的预测概率；
- 语义相似度（Semantic Similarity）：判断两个句子是否语义上是相关的；
- 分类（Classification）：判断输入文本是指定的哪个类别。

将无监督学习作为有监督模型的预训练目标，因此叫做生成式预训练（Generative Pre-training，GPT）。

GPT-1的训练分为无监督的预训练和有监督的模型微调，当时还是需要微调的。虽然这一套在 CV 里面非常成熟，但是在 NLP 领域当年还是不行。

**(1) 无监督预训练**
这个部分就是常规的 next token prediction，即给定前面的 token，预测下一个 token。 假设词汇表长度是 768，那么在 decoder 隐含层输出后接一个 embedding 矩阵，将其输出维度映射为 768，然后进行 loss 计算。

**(2) 有监督微调**
这里的微调就是和 CV 里面一样，假设是分类任务，那么 decoder 输出后再接一个新的分类权重矩阵，假设类别是3，那么输出维度就是 3，然后计算 loss。
注意：有监督微调并不是我们现在常说的 next token prediction。

但是作者实际微调时候发现，如果将原先的 next token prediction head 保留并作为辅助 loss 分支，效果更好。

<div align=center>
<img src="https://github.com/QwenLM/Qwen-7B/assets/17425982/c260f041-f3ab-4a2c-8000-7292bab49df2"/>
</div>

左边从下到上，text prediction 就是无监督预训练流程。右边从下到上，task classification 就是有监督分类微调流程(微调时候也要用 text prediction 分支作为辅助，此时只是分隔符的嵌入值才可训练的，其余部分不训练)。

为了确保微调和预训练时候输入保持一致。作者将不同任务的输入构造成一个统一的文本模板，如右图所示。

从上面可以看出，一次预训练后，对于不同的任务都需要再进行一次微调，并且 3 分类和 4 分类情况下也要分别微调两次，和 CV 里面的情况一样。

GPT-1 在未经微调的任务上虽然也有一定效果，但是其泛化能力远远低于经过微调的有监督任务，说明了 GPT-1 只是一个简单的领域专家，而非通用的语言学家。

# GPT2

Language models are unsupervised multitask learners
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
https://huggingface.co/gpt2

GPT-2是一个在自监督方式下，以非常大的英文数据语料为基础进行预训练的Transformer模型。这意味着它仅仅在原始文本上进行预训练，没有以任何方式由人类进行标注（这就是为什么它可以使用大量的公开可用数据），而是通过自动化过程从这些文本中生成输入和标签。更准确地说，它被训练来猜测句子中的下一个词。
具体而言，输入是一定长度的连续文本序列，目标是相同的序列，向右移动一个标记（单词或词片段）。模型在内部使用掩码机制，以确保对于标记i的预测仅使用从1到i的输入，而不使用未来的标记。通过这种方式，模型学习到了英语语言的内部表示，这可以用来提取对下游任务有用的特征。然而，该模型在其预训练任务上表现最佳，即根据提示生成文本。

GPT-2的核心思想概括为：任何有监督任务都是语言模型的一个子集，当模型的容量非常大且数据量足够丰富时，仅仅靠训练语言模型的学习便可以完成其他有监督学习的任务。

GPT-2的模型结构和GPT-1的模型结构类似，都是基于Transformer的。相对于GPT-1，做的修改有：

1. 调整Transformer的decoder： 将归一化层移动到block的输入位置并且在最后一个self-attention之后加了一层归一化。
2. 数据量扩增：GPT1利用了约5GB，GPT2利用了40GB，并且质量更高
3. 词典被扩展到了50257，context的维度从512提高到了1024并且batchsize采用了512。
4. 堆叠的层数增加：GPT1使用的12层的 TransformerDecoder，GPT2分别使用了24、36、48层。
5. **去掉了Fine-tune部分：使用了完全的无监督训练。这样使得预训练和Fine-tuning的结构完全一致。**

总结来说，就是 GPT2 认为只需要做预训练就可以了，只要训练语料足够大，模型足够大，那么就不需要所谓的 funetune了，可以通过类似 prompt 的方式来轻松实现各种任务。现在这个思想已经是主流了。

也就是说 GPT2 只有 text prediction head。

GPT-2的最大贡献是验证了通过海量数据和大量参数训练出来的词向量模型有迁移到其它类别任务中而不需要额外的训练。但是很多实验也表明，GPT-2的无监督学习的能力还有很大的提升空间，甚至在有些任务上的表现不比随机的好。尽管在有些zero-shot的任务上的表现不错，但是我们仍不清楚GPT-2的这种策略究竟能做成什么样子。GPT-2表明随着模型容量和数据量的增大，其潜能还有进一步开发的空间，基于这个思想，诞生了我们下面要介绍的GPT-3。

# GPT3
https://arxiv.org/abs/2005.14165  
Language Models are Few-Shot Learners

虽然 GPT-2 主推的 zero-shot 在创新度上有比较高的水平，但是由于其在效果上表现平平，所以在业界并没有取得比较大的影响力，而 GPT-3 正是为了解决效果上的问题而提出的。GPT-3 不再去追求那种极致的不需要任何样本就可以表现很好的模型，而是考虑像人类的学习方式那样，仅仅使用极少数样本就可以掌握某一个任务，因此就引出了 GPT-3 标题 Language Models are Few-Shot Learners。

这里的 few-shot 不是像之前的方式那样，使用少量样本在下游任务上去做微调，因为在 GPT-3 那样的参数规模下，即使是参数微调的成本也是高到无法估计。他其实是我们现在说的 In-context learning。

<div align=center>
<img src="https://github.com/QwenLM/Qwen-7B/assets/17425982/7cba5316-3f33-41df-ad9c-7644ed933888"/>
</div>

上图解释的非常清楚了。

模型参数如下所示，gpt3 一般指的就是 1750 亿参数的最大模型

<div align=center>
<img src="https://github.com/QwenLM/Qwen-7B/assets/17425982/95c10bcc-50d3-49fc-8269-74eb379821ba"/>
</div>

相比于 gpt2 ，整个结构几乎没有变。但是引入了 Sparse Transformer 中的 sparse attention 模块（稀疏注意力）

使用 sparse attention 的好处主要有以下两点：

1. 减少注意力层的计算复杂度，节约显存和耗时，从而能够处理更长的输入序列；
2. 具有“局部紧密相关和远程稀疏相关”的特性，对于距离较近的上下文关注更多，对于距离较远的上下文关注较少；

关于 sparse attention 详情可参考《Generating Long Sequences with Sparse Transformers》

GPT-3共训练了5个不同的语料，分别是低质量的Common Crawl，高质量的WebText2，Books1，Books2和Wikipedia，GPT-3根据数据集的不同的质量赋予了不同的权值，权值越高的在训练的时候越容易抽样到

<div align=center>
<img src="https://github.com/QwenLM/Qwen-7B/assets/17425982/c2d48c51-e90a-4f0a-8148-646b77f6b29c"/>
</div>

# InstructGPT

Training language models to follow instructions with human feedback   
https://arxiv.org/abs/2203.02155


