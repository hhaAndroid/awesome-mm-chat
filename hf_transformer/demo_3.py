
def demo_1():
    import torch
    from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification,AutoModel

    # 这是一个二类分类
    # Same as before
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # 这个模型的最后新增层是没有训练过的，因此如果直接测试性能很差
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    print(model)

    # model = AutoModel.from_pretrained(checkpoint)
    # print(model)

    # 新样本
    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This course is amazing!",
    ]
    batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    # 新标签
    batch["labels"] = torch.tensor([1, 1])

    optimizer = AdamW(model.parameters())
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()


def demo_2():
    import os
    # 会自动下载到  ~/.cache/huggingface/dataset, 您可以通过设置 HF_HOME 环境变量来自定义缓存的文件夹
    # 必须要放到 from datasets import load_dataset 前面，否则无效
    os.environ['HF_HOME'] = '../'

    # pip install datasets
    # from datasets import load_dataset
    #
    # # MRPC数据集中的GLUE 基准测试数据集，它是构成MRPC数据集的10个数据集之一，
    # # 这是一个学术基准，用于衡量机器学习模型在10个不同文本分类任务中的性能
    # # 该数据集由5801对句子组成，每个句子对带有一个标签，指示它们是否为同义
    # # 训练集中有3668对句子，验证集中有408对，测试集中有1725对
    # raw_datasets = load_dataset("glue", "mrpc")
    # print(raw_datasets)

    # 0 对应于 not_equivalent，1 对应于 equivalent。

    # 需要预处理才能训练
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


def demo_3():
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


# 不使用 Trainer，用原先的实现
def demo_4():
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

    # 这个 finetune 应该是全部层都 finetune
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


if __name__ == '__main__':
    num = 3
    eval(f"demo_{num}()")
