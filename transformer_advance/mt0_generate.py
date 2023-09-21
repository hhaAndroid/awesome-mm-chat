from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "bigscience/mt0-small"

# model: MT5ForConditionalGeneration
# tokenizer_class: T5TokenizerFast
# "vocab_size": 250112
# pad_token_id": 0,
# eos_token_id": 1,
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
# inputs 后面会自动追加一个结束符，不需要开始符，代表输入结束。不同模型可能不一样
outputs = model.generate(inputs)
# 在推理时候，decoder 首先会注入开始解码符号，token_id=0,也就是和 pad_token_id 一样，所以输出的第一个 token 是 <pad>
print(tokenizer.decode(outputs[0]))  # <pad> I love you.</s>

