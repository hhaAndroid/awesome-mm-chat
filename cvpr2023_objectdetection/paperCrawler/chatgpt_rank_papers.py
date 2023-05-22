import openai
from utils import load_paper_info
import time
import pandas as pd
import tqdm

system_prompt = """You are an AI assistant. I will provide you with the title and abstract of a paper and ask for your help to analyze its relevance to general object detection, which includes object detection, instance segmentation, and semantic segmentation tasks. 
The relevance is divided into three levels: 

1. highly relevant. It's means that the paper's title or abstract includes relevant datasets such as COCO, LVIS, Objects365, and Open Images.
2. moderately relevant. It's means that it includes general object detection but also some additional irrelevant information.
3. not relevant. It's means that it includes mostly content unrelated to general object detection tasks.

Your output format must be one of the following and do not output anything else: [highly relevant], [moderately relevant], [not relevant]. 

title is : 
abstract is : 
"""

src_file = 'filted_cvpr2023.csv'

paper_infos = load_paper_info(src_file)
print('The total number of papers is', len(paper_infos))

filter_papers = []
all_check = False

for i, paper in enumerate(tqdm.tqdm(paper_infos)):
    if paper.relevant is not None:
        all_check = True
        continue

    print('====================================')
    print('paper title is : ', paper.title)
    content = system_prompt.replace('title is : ', f'title is : {paper.title}')
    content = content.replace('abstract is : ', f'abstract is : {paper.abstract}')

    prompt = [
        {
            'role': 'system',
            'content': content,
        }
    ]

    try:
        # 可能会出现 openai.error.RateLimitError
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=prompt, temperature=0, max_tokens=200)
        text_prompt = response['choices'][0]['message']['content']
        print(text_prompt)

        if 'highly relevant' in text_prompt:
            paper.relevant = 1
        elif 'moderately relevant' in text_prompt:
            paper.relevant = 2
        else:
            paper.relevant = 3

        time.sleep(30)  # 为了尽可能的减少 500 请求错误
        all_check = True
    except Exception as e:
        all_check = False
        print(e)
        time.sleep(60)  # 再次延迟一分钟

    filter_papers.append(paper)

df = pd.DataFrame.from_dict(paper_infos)
df.to_csv(f'chatgpt_filted_cvpr2023.csv', index=True, header=True)

if all_check:
    print('All papers have been checked.')
