from tokenizers import Tokenizer, decoders, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# bbpe 编码不需要 UNK token
# 因为 BBPE 是一种 open-vocabulary 的分词方法,它能够处理任何输入词汇,不会出现无法识别的词语
tokenizer = Tokenizer(BPE())

# 定义预分词处理器
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel

# 可以很好地处理英文文本的分词问题,能够识别单词、数字、标点符号、特殊字符以及空白字符等
splitter = Split(pattern="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{"
                         "N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", behavior="isolated")
byte_level = ByteLevel(add_prefix_space=False, use_regex=False)
# 所有输入文本都会被拆分成单词、数字、标点符号、特殊字符以及空白字符等，然后转换为字节级别的编码

# 如果没有一个预分词器来将我们的输入拆分成单词,我们可能会得到跨越多个单词的 token,
# 例如,我们可能会得到一个 "it is" 的 token,因为这两个单词经常出现在彼此旁边。
# 使用预分词器可以确保没有任何一个 token 跨越多个单词。也比较符合子词的概念
tokenizer.pre_tokenizer = Sequence([splitter, byte_level])

from tokenizers.processors import TemplateProcessing, ByteLevel, Sequence

byte_level = ByteLevel()

# 在 tokenizer 后输出前如何处理？
# 注意 llama3 在预训练时候并没有训练过 eos token，所以这里不需要加 eos token
# 只使用 bos token 即可，每句开头也会加上 bos token
# 在预习训练完成后如果想评测，只需要把 eos token=bos token 即可，可以实现类似 eos token 效果
template_processing = TemplateProcessing( # 这个实际上就是起到了类似 apply_chat_template 作用
    single="<|begin_of_text|> $A",  # 单句如何处理？
    pair="<|begin_of_text|> $A <|begin_of_text|> $B:1",  # 多句如何处理？
    special_tokens=[
        ("<|begin_of_text|>", 128000)
    ],
)

# 字节级别的编码
tokenizer.post_processor = Sequence([ByteLevel(), template_processing])
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(vocab_size=128000)
# 准备语料，开始训练
files = ['wiki_corpus.txt']
tokenizer.train(files, trainer)

# 验证效果
x = "Hey how are you doing today?"
output = tokenizer.encode(x)
print(output.tokens, output.ids)

# 编码效率很低
x = "今天天气真的要呀，你觉得呢?"
output = tokenizer.encode(x)
print(output.tokens, output.ids)

# 转换为 huggingface 的 tokenizer 格式，方便发布
from transformers import PreTrainedTokenizerFast

# 预定义特殊 token
added_tokens_decoder = {"128000": AddedToken("<|begin_of_text|>", special=True),
                        "128001": AddedToken("<|end_of_text|>", special=True)}

for i in range(128002, 128256):
    added_tokens_decoder[str(i)] = AddedToken(f'<|reserved_special_token_{i - 128002}|>', special=True)

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    added_tokens_decoder=added_tokens_decoder,
    bos_token="<|begin_of_text|>",
    eos_token="<|end_of_text|>",
    model_input_names=["input_ids", "attention_mask"],
)

wrapped_tokenizer.save_pretrained("./my_Meta-Llama-3-8B")

x = "Hey how are you doing today?"
print(wrapped_tokenizer(x, return_tensors="pt"))
