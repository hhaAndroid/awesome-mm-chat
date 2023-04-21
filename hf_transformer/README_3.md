## 3 微调一个预训练模型

基于预训练好的 BERT 我们构建一个线性层进行分类微调

```python
from transformers import AutoModelForSequenceClassification,AutoModel

checkpoint = "bert-base-uncased"

model = AutoModel.from_pretrained(checkpoint)
print(model)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print(model)
```

将上述两个打印的 model 保存到两个 txt 例如，然后可以通过对比查看区别

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/233567478-295ed5b9-f82e-478f-9c4e-b824e6c9389b.png"/>
</div>

可以发现 AutoModelForSequenceClassification 只是在 BERT 后面接了 dropout+linear 进行分类

上述程序运行后会输出如下警告：

```text
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight']

- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']

You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

第一个警告是因为 Bert 和 BertForSequenceClassification 中都不存在这些 key(暂时不清楚这些 key 为何会存在权重里面) ，因此无法加载，这是合理的。
第二个警告是因为 BertForSequenceClassification 有新增的层，自然没有加载预训练的参数。
他还贴心的告诉你可能是在微调。

如果想对上述模型进行微调训练，可以自己准备数据，也可以用 https://huggingface.co/datasets 中的非常方便。

让我们使用 MRPC 数据集中的 GLUE 基准测试数据集，它是构成 MRPC 数据集的 10 个数据集之一，这是一个学术基准，用于衡量机器学习模型在10个不同文本分类任务中的性能。

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

可以发现这个数据集的任务是：输入两个句子，判断是否是同义句， 0 对应于not_equivalent，1对应于equivalent。

要想利用 BERT 进行微调，我们需要将数据集转换为 BERT 可以处理的格式，即将句子转换为 BERT 的输入格式即  [CLS] sentence1 [SEP] sentence2 [SEP] 

幸运的是，标记器不仅仅可以输入单个句子还可以输入一组句子，并按照我们的BERT模型所期望的输入进行处理：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
# {'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], # 告诉模型输入的哪一部分是第一句，哪一部分是第二句。
# 请注意，如果选择其他的检查点，则不一定具有类型标记ID(token_type_ids)（例如，如果使用DistilBERT模型，就不会返回它们）。只有当它在预训练期间使用过这一层，模型在构建时依赖它们，才会返回它们。
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
# ['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```

Transformers 提供了一个 Trainer 类来帮助您在自己的数据集上微调任何预训练模型，你也自己写，或者将其修改为 accelerator 来训练。

```python

import os
# 会自动下载到  ~/.cache/huggingface/dataset, 您可以通过设置 HF_HOME 环境变量来自定义缓存的文件夹
# 必须要放到 from datasets import load_dataset 前面，否则无效
os.environ['HF_HOME'] = '../'  # 所有缓存都会放到这个路径下

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 一个样本一个处理
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 移除一些不需要的数据，否则 Model 无法处理
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

from torch.utils.data import DataLoader
# batch 内 padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
import evaluate
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
metric.compute()
```

如果不想自己处理分布式，或者 to device 这种操作，可以用 accelerator 来改写。或者用更高级的封装 Trainer

```python
import os
# 会自动下载到  ~/.cache/huggingface/dataset, 您可以通过设置 HF_HOME 环境变量来自定义缓存的文件夹
# 必须要放到 from datasets import load_dataset 前面，否则无效
os.environ['HF_HOME'] = '../'  # 所有缓存都会放到这个路径下

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 一个样本一个处理
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# batch 内 padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
import evaluate  # pip install evaluate
import numpy as np
from transformers import Trainer
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```

感觉 trainer 封装程度非常高，如果不熟悉内部代码，出现不符合预期的情况就很难处理。

```python
