import openai
from utils import load_paper_info
import time
import pandas as pd
import tqdm

system_prompt = """我希望你能充当一个论文摘要总结和英语翻译助手，我将给你一段英文文本， 请严格按照如下步骤进行分析：

1. 对输入的英文文本进行总结，要求不超过 200 个单词
2. 将总结的内容翻译为中文输出

你只需要输出第二步的中文内容即可，其余请不要输出。
好的，现在我的输入为文本是：
"""

src_file = 'filted_cvpr2023.csv'

paper_infos = load_paper_info(src_file)
print('The total number of papers is', len(paper_infos))

filter_papers = []
all_check = False
total_tokens = 0
for i, paper in enumerate(tqdm.tqdm(paper_infos)):
    if paper.tran_flag != -1:
        all_check = True
        continue

    print('====================================')
    print('paper title is : ', paper.title)
    content = system_prompt.replace('输入为文本是：', f'输入为文本是： {paper.abstract}')

    prompt = [
        {
            'role': 'system',
            'content': content,
        }
    ]

    try:
        # 可能会出现 openai.error.RateLimitError
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=prompt, temperature=0, max_tokens=300)
        total_token = response['usage']['total_tokens']
        total_tokens += total_token
        text_prompt = response['choices'][0]['message']['content']
        print(f'\n num token is {total_token}', text_prompt)
        paper.tran_flag = text_prompt
        time.sleep(10)  # 为了尽可能的减少 500 请求错误
        all_check = True
    except Exception as e:
        all_check = False
        print(e)
        time.sleep(20)  # 再次延迟

    filter_papers.append(paper)

df = pd.DataFrame.from_dict(filter_papers)

# 因为我们没有修改源文件，只是加了一个新的 item，所以可以原地覆盖，不要紧
df.to_csv(src_file, index=True, header=True)

print('The total number of tokens is', total_tokens)

if all_check:
    print('All papers have been checked.')
