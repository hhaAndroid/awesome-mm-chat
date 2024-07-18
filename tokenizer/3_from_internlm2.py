import os
import sentencepiece as spm
import shutil
from tokenizers import AddedToken

# 如果有多个文件，可以用逗号分割，例如 corpus_path = 'a.txt,b.txt'
corpus_path = 'wiki_corpus.txt,test.txt'
vocab_size = 92544  # 数据如果不够的话 92544 会报错

save_model = 'my_internlm2/tokenizer'
if not os.path.exists(os.path.dirname(save_model)):
    os.makedirs(os.path.dirname(save_model))

# model_type 表示使用的训练算法，默认是 unigram
# LLaMA 使用的算法是 bpe，所以这里也使用 bpe 作为训练算法
# 另外，LLaMA 在训练分词器时设置了 byte_fallback=True，这样在解码时如果遇到了词表以外的词，会用
# UTF-8 编码将这些字符转换成字节表示，可以避免 out of vocabulary 问题

# 应该有很多参数要调，而且应该要和 tokenizer 前后处理对齐才合理？ 估计默认就是对齐了
model_type = 'bpe'
spm.SentencePieceTrainer.train(input=corpus_path,
                               model_prefix=save_model,
                               vocab_size=vocab_size,
                               model_type=model_type,
                               byte_fallback=True)

# 转换为 huggingface 的 tokenizer 格式，方便发布
# 内部定义了 convert，可以得到 tokenizer.json
from tokenization_internlm2_fast import InternLM2TokenizerFast

added_tokens_decoder = {92538: AddedToken("<|plugin|>", special=True),
                        92539: AddedToken("<|interpreter|>", special=True),
                        92540: AddedToken("<|action_end|>", special=True),
                        92541: AddedToken("<|action_start|>", special=True),
                        92542: AddedToken("<|im_end|>", special=True),
                        92543: AddedToken("<|im_start|>", special=True),
                        }

tokenizer = InternLM2TokenizerFast(save_model+'.model',
                                   added_tokens_decoder=added_tokens_decoder,
                                   # 虽然定义了，但是用不上，因为已经注入了 bos token
                                   chat_template="{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                                   )
tokenizer.save_pretrained('my_internlm2_7b/')

shutil.copyfile('tokenization_internlm2.py', 'my_internlm2_7b/tokenization_internlm2.py')
if os.path.exists(os.path.dirname(save_model)):
    shutil.rmtree(os.path.dirname(save_model))

