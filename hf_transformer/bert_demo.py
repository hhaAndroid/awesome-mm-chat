from transformers import pipeline


def demo_1():
    # 这个是没有进行 fintune 的模型，实际上只能用于特征提取
    # FillMaskPipeline
    # model_inputs = self.preprocess(inputs, **preprocess_params)
    #  BertForMaskedLM = BertModel+分类层 BertOnlyMLMHead
    # model_outputs = self._forward(model_inputs, **forward_params)
    # outputs = self.postprocess(model_outputs, **postprocess_params)
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    o = unmasker("Hello I'm a [MASK] model.")
    print(o)

    # [{'score': 0.10731077939271927, 'token': 4827, 'token_str': 'fashion', 'sequence': "hello i'm a fashion model."},
    # {'score': 0.0877445638179779, 'token': 2535, 'token_str': 'role', 'sequence': "hello i'm a role model."},
    # {'score': 0.05338413640856743, 'token': 2047, 'token_str': 'new', 'sequence': "hello i'm a new model."},
    # {'score': 0.046672213822603226, 'token': 3565, 'token_str': 'super', 'sequence': "hello i'm a super model."},
    # {'score': 0.027095887809991837, 'token': 2986, 'token_str': 'fine', 'sequence': "hello i'm a fine model."}]


def demo_2():
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(tokenizer.vocab_size)  # 30522
    print(tokenizer.get_vocab()) # 第 0 个是 [PAD]，中间是 100 个 [unused], 然后是 '[CLS]': 101, '[SEP]': 102, '[MASK]': 103，后面一大堆又是 [unused]

    model = BertModel.from_pretrained("bert-base-uncased")
    print(model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # print(output) # BaseModelOutputWithPoolingAndCrossAttentions 对象
    print([{k: v.shape} for k, v in output.items()]) # [{'last_hidden_state': torch.Size([1, 12, 768])}, {'pooler_output': torch.Size([1, 768])}]


def demo_3():
    from hf_transformer.bert.bert_model import BertModel
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(tokenizer.vocab_size)  # 30522
    print(tokenizer.get_vocab()) # 第 0 个是 [PAD]，中间是 100 个 [unused], 然后是 '[CLS]': 101, '[SEP]': 102, '[MASK]': 103，后面一大堆又是 [unused]

    model = BertModel.from_pretrained("bert-base-uncased")
    print(model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # print(output) # BaseModelOutputWithPoolingAndCrossAttentions 对象
    print([{k: v.shape} for k, v in output.items()]) # [{'last_hidden_state': torch.Size([1, 12, 768])}, {'pooler_output': torch.Size([1, 768])}]


if __name__ == '__main__':
    num = 1
    eval(f'demo_{num}()')

