# GLIP 解读

论文题目： GLIP: Grounded Language-Image Pre-training  
论文地址： https://arxiv.org/abs/2112.03857   
官方地址：https://github.com/microsoft/GLIP   
Fork 并注释版本： https://github.com/hhaAndroid/GLIP/tree/hha   

由于考虑在 MMDet 中支持，因此解读以 MMDet 中的为准。
## 任务说明

何谓 Grounding? 实际上就是定位的含义。Grounding 目标检测任务为：给定图片和文本，预测出文本中提到的物体的 bbox 和类别，只不过这个类别是开放的。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233949492-33664013-94aa-4e95-b0c8-f069d133ee0d.png"/>
</div>

由于 text prompt 是可以人为输入的，用户可以输入任何描述，因此这是一个开放性检测问题。如果想做闭集检测，可以采用两种方式：

1. 输入的文本不要输入开发词汇，而是只包括特定类别命名实体，例如 `there are some holes on the road`
2. 输入的文本只输入类别，例如输入 `person. bicycle.car. motorcycle.` 等等即可。text 中不包括的不应该检测出来

实际上为了方便且合理，在 COCO 目标检测任务上是输入类别序列，通过 . 拼接而成。注意： `这个符号不能随便换，因为要和训练保持一致`。

我们不会过多的关注于训练过程，因为也没有多少能成功复现。

## 模型说明

为何要做 Grounding？ 原因是：

1. CLIP 采用了大量图文对训练，得到了一个能够进行 zero-shot 的图像分类器，但是由于是图片级别监督，实际上无法直接进行 zero-shot 目标检测或者其他密集预测任务
2. GLIP 核心是想通过大量图文对训练，得到了一个能够进行 zero-shot 的目标感知的，语义丰富的多模态模型，典型任务就是定位任务
3. 通过将 parse grounding 任何和 object detetion 任务统一，就可以利用大量的具备 bbox 标注的图文对进行训练了，相当于大量扩充了数据集

parse grounding 通常指的是将自然语言指令与场景中的物体、位置等进行对应，以实现自然语言的场景理解。这个任务通常被称为"语义解析"或"自然语言指令理解"，是人机交互和智能机器人等领域中的重要问题。 将两个任务统一建模，数据就多了。

原文里面写的是：

1. 它允许 GLIP 从检测和接地数据中学习，以改进这两个任务并引导一个良好的 grounding 模型;
2. GLIP 可以利用大量的图像-文本对,通过以自我训练的方式生成 grounding，使学习到的表示具有丰富的语义。
3. 上述两个任务的统一还允许我们使用两种类型的数据进行预训练，并使两项任务都受益

在 27M 基础数据上预训练 GLIP，包括 3M 个人工注释和 24M 个网络抓取的图像-文本对，学习到的表征在各种对象级识别任务中表现出很强的 zero-shot 和 few-shot 能力。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233951372-214e1952-4466-4c75-90e9-819f4d2e72aa.png"/>
</div>

结构图如上所示。作者发现与仅在最后一个点积层融合视觉和语言的 CLIP 不同，我们表明 GLIP 应用的深度跨模态融合对于学习高质量的语言感知视觉表示和实现卓越的迁移学习性能至关重要。

## 效果分析

模型和配置：

```python
config_file = "configs/pretrain/glip_A_Swin_T_O365.yaml"
weight_file = "glip_a_tiny_o365.pth"
```

```python
image = load('cat_remote.jpg')
caption = 'cat'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233975344-fe1ec094-b494-4a71-a157-2cdabcdee6ea.png"/>
</div>

```python
image = load('cat_remote.jpg')
caption = 'cat . remote . '
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233975607-badba29c-8fb2-4427-b0d2-63f5308ff07c.png"/>
</div>

看起来效果还行，但是如果看下面的例子：

```python
image = load('cat_remote.jpg')
caption = 'There is a cat and a remote in the picture'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233976517-ba4e9d8f-bcc2-4723-bdb2-5f723e26e121.png"/>
</div>

虽然检测好像是对的，但是由于离线的 NLP 的命名实体会将 a cat 认为是一个单词，a 不是一个数量含义，所以 grounding 训练时候 a cat 会认为都是 gt 即 gt= [1, 1] 而不是 [0, 1]

如果你换成：

```python
caption = 'There is two cat and a remote in the picture'
```
那么由于 two cat 不是一个名词，会正确检测出 cat，而不是 two cat。

## 配置解读

一共包括 4 个配置文件：

- configs/pretrain/glip_A_Swin_T_O365.yaml
- configs/pretrain/glip_Swin_T_O365.yaml
- configs/pretrain/glip_Swin_T_O365_GoldG.yaml
- configs/pretrain/glip_Swin_L.yaml

分别对应 README 中的 ABC 和 GLIP-L 模型

- A 模型： 没有采用 deepfusion 而是类似 CLIP 直接在最后进行点积融合，轻量化模型
- B 模型： 采用 deepfusion，标准模型
- C 模型： 采用 deepfusion，但是使用了 gold grounding 
- GLIP-L 模型： 采用 deepfusion，训练数据和 C 模型一样，但是 backbone 换成了 Swin_L

我们应该重点关注 AB 模型。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/234162743-d4d6ce0f-7623-478e-af50-bc14b2f3e684.png"/>
</div>

左边是 A, 右边是 B。

## 推理过程分析
代码位于： https://github.com/hhaAndroid/GLIP/blob/hha/demo.py

以下面句子为例，分析推理过程：

```python
caption = 'There is two cat and a remote in the picture'
```

(1) 图片前处理
    
```python   
image = self.transforms(original_image) # 常规的 800,1333 缩放
image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(self.device)
```

(2) 文本前处理



## 训练过程分析
假设给定一张图片，和一个 text prompt，要定位其中的物体和类别，有如下几种情况：

1. 假设文本中包括类别名 traffic light 即类别是由多个单词组成
2. 假设文本中包括的类别名会切分为多个子词 toothbrush -> tooth, #brush
3. 文本中可能包括一些特殊的 token
4. 会增加一个额外的 [NoObj] token 在句子结尾




