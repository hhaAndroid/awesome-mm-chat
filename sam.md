# SAM

# RSPrompter

RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model

https://arxiv.org/pdf/2306.16269.pdf

基于遥感数据，扩展 SAM 进行实例分割。整个训练过程中都不调整 SAM,而是仅仅调整其余部件。为了体现算法优异性，作者一共提出了 4 种基于 sam 的实例分割算法

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/fff656cc-b147-46bf-8d62-22ed5d69c4c3"/>
</div>

前三个是一个参考，第 4 个是本身主推的实例分割实现。针对 CNN 和 Transformer 的不同，作者提出了两者结构： Anchor-based Prompter 针对 Faster RCNN，Query-based Prompter 针对 DETR 系列。

在三个遥感数据集上进行评估，发现两种结构各有优点，并没有一家独大。

本文属于工程类型文章，实验做的比较多，应该是社区用户需要的，是一篇不错的文章。

