from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset


def train():
    device = "cuda"
    model_name_or_path = "bigscience/mt0-small"
    text_column = "sentence"
    label_column = "text_label"
    max_length = 128
    lr = 1e-3
    num_epochs = 3
    batch_size = 4

    # creating model
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                             inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model)

    # loading dataset
    # 包含金融领域短语和表达的数据集，用于构建金融情感分析、文本分类、语义角色标注和关系提取等任务的机器学习模型
    # 这里其实是一个3分类问题：'positive', 'neutral', 'negative'
    # .cache/huggingface/datasets/
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=1,
    )

    print(dataset["train"][0])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True,
                                 return_tensors="pt")
        labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # -100 表示忽略，在 loss 里面也写死了
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,  # 并行处理 1000 (默认)条数据，而不是一条一条处理
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # training and evaluation
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            # 他这个模型分类微调，其实也用的是 next sentence prediction 的方式，而不是常规的分类方式，也就是和预训练的方式一致
            # 假设 bs=2，词表大小为 250112， 输入 shape 为 (2, 128)
            # 编码器输出是 (2, 128, 512)，解码器端的 target 需要进行右移，第一位填充 decoder_start_token_id =0
            # target 的 shape 为 (2, 3)，其中 3 为最大长度，也就是每个类别名都采用 3 个 token id 代替，内部的每个元素为词表的 id，填充的位置设置为 -100
            # 解码器输出为 (2,3,512)， 然后经过分类头，输出 (2, 3, 250112)，然后和 target 计算 loss
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                       skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)  # 困惑度指标，越小越好
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    # print accuracy
    correct = 0
    total = 0
    for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total * 100
    print(f"{accuracy=} % on the evaluation dataset")
    print(f"{eval_preds[:10]=}")
    print(f"{dataset['validation']['text_label'][:10]=}")

    # saving model
    peft_model_id = f"financial_sentiment_analysis_lora_v1.pth"
    model.save_pretrained(peft_model_id)

    val()


def val():
    from peft import PeftModel, PeftConfig

    text_column = "sentence"
    peft_model_id = f"financial_sentiment_analysis_lora_v1.pth"

    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model.eval()

    # loading dataset
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=1,
    )

    i = 13
    inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
    print(dataset["validation"][text_column][i])
    print(inputs)

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
        print(outputs)
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


if __name__ == '__main__':
    is_train = True
    if is_train:
        train()
    else:
        val()

