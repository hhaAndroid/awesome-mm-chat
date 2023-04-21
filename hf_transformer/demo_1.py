def demo_1():
    # pip install transformers[sentencepiece]
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    out = classifier("I've been waiting for a HuggingFace course my whole life.")
    print(out)  # [{'label': 'POSITIVE', 'score': 0.9598048329353333}]


def demo_2():
    from transformers import pipeline

    # No model was supplied, defaulted to gpt2
    # Using a pipeline without specifying a model name and revision in production is not recommended.
    # 您可以使用参数 num_return_sequences 控制生成多少个不同的序列，并使用参数 max_length 控制输出文本的总长度。
    generator = pipeline("text-generation")
    out = generator("In this course, we will teach you how to")
    # [{'generated_text': 'In this course, we will teach you how to
    # build an HTTP server-style HTTP client application.
    # While the fundamentals are familiar, the techniques
    # involve understanding how a large set of mechanisms
    # (like HTTP APIs) works and how to provide the most efficient,'}]
    print(out)


def demo_3():
    from transformers import pipeline
    # dbmdz/bert-large-cased-finetuned-conll03-englis
    # 命名实体识别（NER）是指识别文本中的人名、地名、机构名、专有名词等实体。
    ner = pipeline('ner', grouped_entities=True)
    out = ner('My name is Sylvain and I work at Hugging Face in Brooklyn')
    print(out)
    # [{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18},
    #  {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45},
    #  {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
    # ]
    # Sylvain 是一个人 (PER)，Hugging Face 是一个组织 (ORG)，而布鲁克林是一个位置 (LOC)


def demo_4():
    from transformers import AutoTokenizer

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)  # ID 列表长度为 16
    # {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
    #           2607,  2026,  2878,  2166,  1012,   102],
    #         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
    #              0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}

    from transformers import AutoModel
    import torch

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    # batch 是2，序列长度是 16，隐藏层大小是 768
    print(outputs.last_hidden_state.shape)  # torch.Size([2, 16, 768])

    from transformers import AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    # AutoModel + SequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.logits.shape)  # torch.Size([2, 2])
    print(outputs.logits)
    predictions = torch.softmax(outputs.logits, dim=-1)
    print(predictions)

    print(model.config.id2label)  # {0: 'NEGATIVE', 1: 'POSITIVE'}


def demo_5():
    from transformers import BertConfig, BertModel

    # Building the config
    config = BertConfig()
    print(config)

    # Building the model from the config
    model = BertModel(config)
    print(model)


def demo_6():
    from transformers import AutoTokenizer
    import torch

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    sequences = ["Hello!", "Cool.", "Nice!"]
    encoded_input = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    print(encoded_input['input_ids'])
    # tensor([[ 101, 7592,  999,  102],
    #         [ 101, 4658, 1012,  102],
    #         [ 101, 3835,  999,  102]])
    print(tokenizer.decode(encoded_input['input_ids'][0]))  # [CLS] hello! [SEP]
    # 101 是 [CLS] 的 ID，102 是 [SEP] 的 ID，999 是 [UNK] 的 ID，7592 是 hello 的 ID, 999 是 ! 的 ID
    print(tokenizer.decode(torch.tensor([999])))  # !


def demo_7():
    from transformers import AutoTokenizer

    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequence = "Using a Transformer network is simple"

    tokens = tokenizer.tokenize(sequence)

    # ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
    print(tokens)

    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(ids)  # [7993, 170, 13809, 23763, 2443, 1110, 3014]

    decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
    print(decoded_string)  # Using a Transformer network is simple

    # 在 BertTokenizer 代码里面会把 ## 去掉,并且和前一个单词合并，从而形成正确的单词
    def convert_tokens_to_string(self, tokens):
        # ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
        out_string = " ".join(tokens).replace(" ##", "").strip()
        #  Using a Transformer network is simple
        return out_string

    decoded_string = tokenizer.decode([7993, 170, 13809, 2443, 1110, 3014])
    print(decoded_string)  # Using a Trans network is simple

    # 故意修改了，可以发现解码就不符合预期了， 170, 23763 被解码为 aer
    decoded_string = tokenizer.decode([7993, 170, 23763, 2443, 1110, 3014])
    print(decoded_string)  # Using aformer network is simple


if __name__ == '__main__':
    num = 7
    eval(f"demo_{num}()")
