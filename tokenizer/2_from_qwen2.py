from tokenizers import Tokenizer, decoders, AddedToken, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# bbpe 编码不需要 UNK token
# 因为 BBPE 是一种 open-vocabulary 的分词方法,它能够处理任何输入词汇,不会出现无法识别的词语
tokenizer = Tokenizer(BPE())

from tokenizers.normalizers import NFC

tokenizer.normalizer = NFC()

# 定义预分词处理器
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel

# 可以很好地处理英文文本的分词问题,能够识别单词、数字、标点符号、特殊字符以及空白字符等
splitter = Split(pattern="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+["
                         "\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", behavior="isolated")
byte_level = ByteLevel(add_prefix_space=False, use_regex=False)
# 所有输入文本都会被拆分成单词、数字、标点符号、特殊字符以及空白字符等，然后转换为字节级别的编码

# 如果没有一个预分词器来将我们的输入拆分成单词,我们可能会得到跨越多个单词的 token,
# 例如,我们可能会得到一个 "it is" 的 token,因为这两个单词经常出现在彼此旁边。
# 使用预分词器可以确保没有任何一个 token 跨越多个单词。也比较符合子词的概念
tokenizer.pre_tokenizer = Sequence([splitter, byte_level])

from tokenizers.processors import ByteLevel

byte_level = ByteLevel()

# 字节级别的编码
tokenizer.post_processor = ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

# 不包括特殊 token
trainer = BpeTrainer(vocab_size=151642)
# 准备语料，开始训练
files = ['wiki_corpus.txt']
tokenizer.train(files, trainer)

import os
save_dir = "my_Qwen2-7B/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 拆分出 vocab.json 和 merges.txt
tokenizer.save("temp.json")
import json

data = json.load(open("temp.json", "r", encoding="utf-8"))
vocab = data['model']['vocab']
merges = data['model']['merges']

if os.path.exists("temp.json"):
    os.remove("temp.json")

with open(save_dir + 'vocab.json', "w", encoding="utf-8") as f:
    f.write(json.dumps(vocab, ensure_ascii=False) + "\n")

with open(save_dir + 'merges.txt', "w", encoding="utf-8") as writer:
    for merge in merges:
        writer.write(merge + "\n")

# # 转换为 huggingface 的 tokenizer 格式，方便发布
from transformers import Qwen2TokenizerFast

# 实际 vocab size 可能不等于词表长度
# 预定义特殊 token
added_tokens_decoder = {151643: AddedToken("<|endoftext|>", special=True),
                        151644: AddedToken("<|im_start|>", special=True),
                        151645: AddedToken("<|im_end|>", special=True)}

wrapped_tokenizer = Qwen2TokenizerFast(
    vocab_file=save_dir + 'vocab.json',
    merges_file=save_dir + 'merges.txt',
    added_tokens_decoder=added_tokens_decoder,
    bos_token=None,  # 注意，他没有 bos token，只有 eos token, 有点特殊
    unk_token=None,
    eos_token="<|endoftext|>",
    additional_special_tokens=["<|im_start|>", "<|im_end|>"],  # 使用这个作为开始结束标志
    model_max_length=32768,
    # 预训练时候先调用这个模板，然后才 tokenizer
    chat_template="{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

wrapped_tokenizer.save_pretrained(save_dir)

chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
prompt = wrapped_tokenizer.apply_chat_template(chat, tokenize=False)
print(prompt, wrapped_tokenizer(prompt, return_tensors="pt"))

