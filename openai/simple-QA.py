import openai
import faiss
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
from mmengine.utils import scandir

EXTENSIONS = ('md',)
MAX_TOKEN_LEN = 4096  # 切分文本时候，每段文本最大长度
text_embedding_model = "text-embedding-ada-002"
OUT_EMBEDDING_DIM = 1536  # text_embedding_model 输出维度
chatgpt_model = 'gpt-3.5-turbo'

# TOPK×EXTRA_PADDING_NUM=25,也就是说每次查询会输入最多 25 和片段进行，最大不超过 MAX_INPUT_TOKEN
TOPK = 3  # 选择数据库里面 topk 条组成 prompt
EXTRA_PADDING_NUM = 5  # 核心参数，对于数据库中被选择的每条参考文本，额外扩展连续的 n 个 item，因为考虑在处理时候会非常多个 item
MAX_INPUT_TOKEN = 1000  # 输入给 gpt 的最大 token 长度, 理论上越大越好，例如 3000

# 为了避免 openai key 暴露，我采用了环境变量注入方式，例如
# export OPENAI_API_KEY=YOUR_API_KEY

# refer from https://github.com/fierceX/Document_QA
# faiss 安装命令： conda install -c faiss-cpu
# 第一次运行会创建 embedding 文件，第二次运行直接读取
# python simple-QA.py --source ‘你的包括大量 md 文件夹的源文件夹路径，也可以是单个 md 文件’ --embedding-path `保存的 pkl 文件名`
# python simple-QA.py --source /mmyolo/docs/zh_cn --embedding-path source_embedding.pkl


def parse_args():
    parser = argparse.ArgumentParser(description='Simple QA based on GPT-3.5')
    parser.add_argument('--source', help='source dst')
    parser.add_argument('--embedding-path', default='source_embeding.pkl',
                        help='Embedding file path. If not specified, '
                             'it will be regenerated based on the source')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def get_file_list(source_root, extensions=EXTENSIONS) -> [list, dict]:
    is_dir = os.path.isdir(source_root)
    is_file = os.path.splitext(source_root)[-1].lower() in extensions

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(
                source_root, extensions, recursive=True,
                case_sensitive=False):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find file.')
    return source_file_path_list


def create_embeddings_one_file(texts):
    result = []
    lens = [len(text) for text in texts]
    query_len = 0
    start_index = 0
    tokens = 0

    def get_embedding(input_slice):
        embedding = openai.Embedding.create(model=text_embedding_model, input=input_slice)
        return [(text, data.embedding) for text, data in zip(input_slice, embedding.data)], embedding.usage.total_tokens

    for index, l in tqdm(enumerate(lens)):
        query_len += l
        if query_len > MAX_TOKEN_LEN:
            ebd, tk = get_embedding(texts[start_index:index + 1])
            query_len = 0
            start_index = index + 1
            tokens += tk
            result.extend(ebd)

    if query_len > 0:
        ebd, tk = get_embedding(texts[start_index:])
        tokens += tk
        result.extend(ebd)
    return result, tokens


def create_embeddings(source, embedding_path):
    source_file_path_list = get_file_list(source)
    print(f'find {len(source_file_path_list)} files')
    total_tokens = 0
    embedding_results = []
    for source_file in source_file_path_list:
        with open(source_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts if text.strip()]
            embedding, tokens = create_embeddings_one_file(texts)
            embedding_results.extend(embedding)
            total_tokens = total_tokens + tokens
    pickle.dump(embedding_results, open(embedding_path, 'wb'))
    print("文档嵌入预处理总共消耗 {} tokens".format(total_tokens))
    return embedding_results


def create_embedding(text):
    """Create an embedding for the provided text."""
    embedding = openai.Embedding.create(model=text_embedding_model, input=text)
    return text, embedding.data[0].embedding


class QA:
    def __init__(self, doc_embeddings, verbose=False) -> None:
        # https://github.com/facebookresearch/faiss/wiki/Getting-started
        # 直接计算 L2 距离
        index = faiss.IndexFlatL2(OUT_EMBEDDING_DIM)
        embe = np.array([emm[1] for emm in doc_embeddings])
        data = [emm[0] for emm in doc_embeddings]
        index.add(embe)
        self.index = index
        self.data = data
        self.topk = TOPK
        self.verbose = verbose

    def __call__(self, query):
        embedding = create_embedding(query)
        context = self.get_texts(embedding[1])
        answer = self.completion(query, context)
        return answer, context

    def get_texts(self, embeding):
        _, text_index = self.index.search(np.array([embeding]), self.topk)
        context = []
        for i in list(text_index[0]):
            # 一条有用的信息可能会非常多段，因为我们是原格式读取，或存在不少换行啥的
            context.extend(self.data[i:i + EXTRA_PADDING_NUM])
        return context

    def completion(self, query, context):
        """Create a completion."""
        lens = [len(text) for text in context]

        maximum = MAX_INPUT_TOKEN
        for index, l in enumerate(lens):
            maximum -= l
            if maximum < 0:
                context = context[:index + 1]
                print("超过最大长度，截断到前", index + 1, "个片段")
                break

        text = "\n".join(f"{index}. {text}" for index, text in enumerate(context))
        response = openai.ChatCompletion.create(
            model=chatgpt_model,
            messages=[
                {'role': 'system',
                 'content': f'你是一个有帮助的AI文章助手，从下文中提取有用的内容进行回答，不能回答不在下文提到的内容，相关性从高到底排序：\n\n{text}'},
                {'role': 'user', 'content': query},
            ],
        )
        print("使用的 tokens 数：", response.usage.total_tokens)
        return response.choices[0].message.content


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.embedding_path):
        assert args.source
        print('The embedding of the document is being generated, please wait')
        doc_embeddings = create_embeddings(args.source, args.embedding_path)
    else:
        doc_embeddings = pickle.load(open(args.embedding_path, 'rb'))

    qa = QA(doc_embeddings, args.verbose)

    while True:
        query = input("请输入查询(help可查看指令)：")
        if query == "quit":
            break
        elif query == "help":
            print("输入 quit 退出")
            continue
        answer, context = qa(query)
        if args.verbose:
            print("已找到相关片段：")
            for text in context:
                print("-------片段-----")
                print('\t', text)
            print("=====================================")
        print("回答如下\n\n")
        print(answer.strip())
        print("=====================================")
