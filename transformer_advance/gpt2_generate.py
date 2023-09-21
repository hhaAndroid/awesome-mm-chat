from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Replace me by any text you'd like.", return_tensors="pt").to("cuda")
# 这个模型就没有在后面追加结束符
outputs = model.generate(inputs)
# 在推理时候，decoder 不需要注入额外的开始解码符，因为这个是 decoder-only 模型，直接基于前面的输入进行预测解码即可
print(tokenizer.decode(outputs[0]))  # I'm not sure if you're aware
