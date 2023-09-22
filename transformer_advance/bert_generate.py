# https://huggingface.co/bert-base-uncased
import torch


def mlm_head():
    from transformers import BertTokenizer, BertForMaskedLM
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 如果手动调用这个类，那就需要自己解码，可以直接用 pipeline 会全部封装好流程
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    inputs = tokenizer("I love the [MASK] pies.", return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    predicted_token = tokenizer.convert_ids_to_tokens(predictions[0, 4].item())
    print(predicted_token)  # 'apple'


def nsp_head():
    from transformers import BertTokenizer, BertForNextSentencePrediction
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    sentence_A = "The sun is a huge ball of gases. It has a diameter of 1,392,000 km."
    sentence_B = "It is mainly made up of hydrogen and helium gas. The surface of the Sun is known as the photosphere."

    inputs = tokenizer(sentence_A, sentence_B, return_tensors='pt')
    # inputs = tokenizer(sentence_A, return_tensors='pt')
    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits)
    if prediction == 0:
        print("True Pair")  # ture
    else:
        print("False Pair")


def lm_head():
    from transformers import BertTokenizer, BertLMHeadModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertLMHeadModel.from_pretrained('bert-base-uncased', is_decoder=True)

    # 没有对应的权重，因此性能不好
    inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt")
    # inputs 后面会自动追加一个结束符，不需要开始符，代表输入结束。不同模型可能不一样
    outputs = model.generate(inputs)
    # 在推理时候，decoder 首先会注入开始解码符号，token_id=0,也就是和 pad_token_id 一样，所以输出的第一个 token 是 <pad>
    print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>


def token_head():
    from transformers import BertTokenizer, BertForTokenClassification
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

    inputs = tokenizer("HuggingFace is a company based in Paris and New York", return_tensors="pt")
    outputs = model(**inputs)
    predicted_token_class_ids = outputs.logits.argmax(-1)
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    print(predicted_tokens_classes)  # ['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']


if __name__ == '__main__':
    # mlm_head()
    # nsp_head()
    lm_head()
    # token_head()
