import csv
import os
import os.path as osp
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm


def download_paper(paper_info):
    year = paper_info[1]
    title = paper_info[2]
    pdf_url = paper_info[6]

    folder = f'{conference}{year}'
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
    parser.add_argument('conference')
    parser.add_argument('--threads', default=10)
    args = parser.parse_args()
    conference = args.conference

    with open(f'{conference}.csv', 'r') as f:
        paper_infos = list(csv.reader(f))[1:]

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for _ in tqdm(
                executor.map(download_paper, paper_infos),
                total=len(paper_infos)):
            pass
