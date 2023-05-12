# InternGPT

标题： InternGPT: Solving Vision-Centric Tasks by Interacting with ChatGPT Beyond Language
官方地址： https://github.com/OpenGVLab/InternGPT
论文： https://arxiv.org/pdf/2305.05662.pdf

InternGPT（简称 iGPT） / InternChat（简称 iChat） 是一种基于指向语言驱动的视觉交互系统，允许您使用指向设备通过点击、拖动和绘制与 ChatGPT 进行互动。internGPT 的名称代表了 interaction（交互）、nonverbal（非语言）和 ChatGPT。与依赖纯语言的现有交互系统不同，通过整合指向指令，iGPT 显著提高了用户与聊天机器人之间的沟通效率，以及聊天机器人在视觉为中心任务中的准确性，特别是在复杂的视觉场景中。此外，在 iGPT 中，采用辅助控制机制来提高 LLM 的控制能力，并对一个大型视觉-语言模型 Husky 进行微调，以实现高质量的多模态对话（在ChatGPT-3.5-turbo评测中达到 93.89% GPT-4 质量）

简单来说就是多了用户点框等交互，可以快速帮助完成一些复杂的仅靠语言描述很难完成的任务。

看了下代码，看起来是基于 VisualGPT,然后多加入了一些模型，可以将用户交互点转换为 mask，然后将图片和 mask 以前丢给后面的模型，实现更高效的计算。

