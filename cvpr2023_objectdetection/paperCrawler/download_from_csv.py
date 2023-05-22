import os
import os.path as osp
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm
from utils import load_paper_info


def download_paper(paper_info):
    year = paper_info.year
    title = paper_info.title
    pdf_url = paper_info.pdf_url

    folder = f'cvpr{year}'
    if not osp.exists(folder):
        os.makedirs(folder, exist_ok=True)

    filename = osp.join(folder, title.replace('/', '_') + '.pdf')
    if osp.exists(filename):
        return

    try:
        r = requests.get(pdf_url)
        with open(filename, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(filename)
        print(e)


if __name__ == '__main__':
    parser = ArgumentParser('Download papers')
    parser.add_argument('--src', default='filted_cvpr2023.csv')
    parser.add_argument('--threads', default=10)
    args = parser.parse_args()

    paper_infos = load_paper_info(args.src)

    # 不下载无关论文
    filterd_paper_infos = []
    for paper_info in paper_infos:
        if int(paper_info.relevant) != 3:
            filterd_paper_infos.append(paper_info)

    print('The total number of filted paper is', len(filterd_paper_infos))

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for _ in tqdm(
                executor.map(download_paper, filterd_paper_infos),
                total=len(filterd_paper_infos)):
            pass
