# Transformer 最简入门和实践

参考链接和代码
1. https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
2. https://github.com/SamLynnEvans/Transformer
3. 大规模语言模型 从理论到实践 PDF

## 说明
本文件夹提供了一个英文翻译为法语的最小实践案例。基于 https://github.com/SamLynnEvans/Transformer，将其升级为新版torch，并进行了一些修改和注释

具体功能：

1. Transformer 的最小实现,非常简单
2. 从头训练一个英法翻译模型，数据集已经在 data 路径下，无需再下载
3. 包括 beam search 解码过程

## 依赖安装
以 pytorch 1.8.1 为例，假设已经安装好了

```text
pip install torchtext==0.9.1
pip install spacy==3.6.1 
```        
## 训练

```text
spacy download en && spacy download fr
python train.py -src_data data/english.txt -trg_data data/french.txt -src_lang en_core_web_sm -trg_lang fr_core_news_sm -batchsize 500
```

会训练 2 个 epoch ，大约 5 分钟

## 推理(翻译)

```text
python translate.py -load_weights weights/ -src_lang en_core_web_sm -trg_lang fr_core_news_sm
```

