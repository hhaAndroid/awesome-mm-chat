# MMPreTrain

MMPreTrain 最近新增了多模态算法，后续也会新增不少多模态相关的内容。MMDetection 也会增加不少多模态内容，考虑到会有参考 MMPreTrain 的地方，因此需要对 MMPreTrain 多模态部分进行分析，吸收一些比较好的设计。

## Inference

https://github.com/open-mmlab/mmpretrain/blob/main/docs/zh_CN/user_guides/inference.md

不同任务都有独立的 Inference，并且提供了类似 HF 的对外简单接口。
