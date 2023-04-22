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

因为模型无法直接处理文本，因此需要将输入文本先进行分词，然后结合词汇表转换为索引 id，即完成了文本编码过程

**(3) zero-shot 图像分类**





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
即 
