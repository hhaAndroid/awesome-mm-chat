# Diffusers

官方文档：https://huggingface.co/docs/diffusers/index  

## DreamBooth

官方论文： [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)  
Dreambooth on Stable Diffusion： https://github.com/XavierXiao/Dreambooth-Stable-Diffusion   
HF 链接： https://huggingface.co/docs/diffusers/training/dreambooth   
解释：https://huggingface.co/blog/dreambooth   


DreamBooth 是一种全量参数微调方法(个性化通用 Diffusion 方法)，只需要非常少的图片就可以实现将特定物体或者风格编码到 Diffusion Model 中。但是需要全量微调，一般会配合 LoRA 一起使用。

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236808584-ef683841-2135-43bf-8b84-64ca0a5ecb22.png"/>
</div>

效果如上所示： 只需要提供几张太阳镜图片(参考图)，最好是不同位置的，在微调完成后就可以通过 [V]+太阳镜 生成带有你微调的特定的太阳镜的不同图片。[V] 特指你输入的独特风格的太阳镜，作为唯一标识符，在训练时候需要指定。[V] 的作用需要接着看才能理解。

完整描述为： 给定作为主题的几个图像作为输入，我们微调预训练的文本到图像模型，以便它学习将唯一标识符 [V] 与特定主题例如前面说的太阳镜绑定。一旦主题嵌入到模型的输出域中，唯一标识符就可以用于合成在不同场景中上下文化的主题的新颖逼真图像。

language drift： 使模型将类名(例如，“太阳镜”)与特定实例相关联，例如如果模型存在的 language drift，那么我输入其他风格的太阳镜文本 prompt，也会生成带有我微调风格的太阳镜的图片。也就是微调把语言模型带偏了的意思。

### 微调训练过程

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236811328-d300b041-865a-4c76-9307-0be67f1bf93f.png"/>
</div>

我们的目标是将一个新的（唯一标识符 [V]可自定义、主题如太阳镜）对 “植入” 到扩散模型的“字典”中。

- 训练数据集： 3~5 张或者更多的参考图片
- 文本输入： `a [identifier] [class noun]` 所有图片都是用这一个同样的 prompt， identifier 就是用来表示这个独特物体风格或者主题的关键词，用户可以自己输入，class noun 就是物体的类别，比如太阳镜，这个是固定的。

我们试图利用特定class noun类的模型的先验，并将其与我们主题的唯一标识符的嵌入纠缠在一起，这样我们就可以利用视觉先验来生成不同背景下主题的新姿势。

我们通过给出3-5张主体的图像，在两个步骤中微调了一个文本到图像的扩散模型：

- (a) 使用包含唯一标识符和主体所属类别名称（例如，“一张[T]狗的照片”）的文本提示来微调低分辨率的文本到图像模型，并同时应用类别特定的先验保持损失，利用模型对类别的语义先验知识，鼓励其通过在文本提示中注入类别名称（例如，“一张狗的照片”）生成属于主体类别的多样实例，防止出现语义偏移
- (b) 使用来自我们输入图像集的低分辨率和高分辨率图像对来微调超分辨率组件，从而能够保持对主体小细节的高保真度。

### 推理过程

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236814311-6679e802-ab20-4f45-803c-dd457f8e1100.png"/>
</div>

疑问： 原则上 V 这个唯一标识符应该是要罕见的不是常用含义的词，论文里面也说要通过一个函数学出来，但是我看一些解释和例子似乎都是直接给的一个单词，并没有所谓的学习过程。
答案： V 这个唯一标识符你可以选择学习，也可以选择不学习，如果不学习那么就是用户输入一个词即可，不一定要是 sks 这个词，最早的一个实现使用了它，因为它是词汇表中一个罕见的标记，实验表明，选择你会自然地用来描述你的目标的术语是可以的。
如果你要学习 V 那么其实其他不变，但是放开了 text encoder，这个模块也一起学习了，效果会更好，但是通常需要消耗非常多的显存。

我们的方法以几张主体（如一只特定的狗）的图片（根据我们的实验，通常3-5张图片就足够了）和相应的类名（如 "狗"）作为输入，并返回一个经过微调"个性化 "的文本-图像模型，该模型编码了一个指代主体的独特标识符。然后，在推理时，我们可以在不同的句子中植入唯一的标识符，以合成不同语境中的主体。

[知乎](https://zhuanlan.zhihu.com/p/612992813) 这篇文章绘制的图蛮好的，比较清晰易懂

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/236815070-97e1b1dd-b368-40c1-98f4-c0f6d576a30f.png"/>
</div>

对应的例子是 https://huggingface.co/docs/diffusers/training/dreambooth

假设是微调狗，那么输入 text 核心参数：

```text
--instance_prompt="a photo of sks dog" \
--class_prompt="a photo of dog" \
```

sks 是唯一标识符，用户可以自定义，但是最好不要和其他的常用的词汇重复。

训练的过程：

1. 给训练图片添加 n 步噪声，使其变成较为嘈杂的图片（测试图左侧的噪声图)。
2. 另外再给训练图片添加较少一点的噪声(n-1)，使其成为一张 target 图片（测试图右侧的图片）。
3. 然后我们来训练SD模型以左侧较嘈杂的图片作为输入，再加上特殊关键词指令的输入，能输出右侧较为清晰的图片。
4. 一开始，由于模型可能根本就不识别新增的特殊关键词sks，他可能输出了一个不是很好的结果。此时我们将该结果与目标图片（右侧较少噪声的图片）进行比较，得出一个loss结果，用以描述生成图像与目标图像的差异程度。
5. 当训练重复了一段时间后，整个模型会逐渐认识到：当它收到sks的词语输入时，生成的结果应该看起来比较像训练者所提供的柯基犬的图片，由此我们便完成了对模型的调校。

也就是包括两个 loss，一个是 diffuser 本身的 loss，一个是先验保留 loss。

知乎上有很多实践课程，例如 https://zhuanlan.zhihu.com/p/620577688?utm_id=0

## ControlNet

https://huggingface.co/docs/diffusers/training/controlnet


## ControlGPT

[Controllable Text-to-Image Generation with GPT-4](https://arxiv.org/pdf/2305.18583.pdf)

