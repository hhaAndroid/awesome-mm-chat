import pandas as pd
import tqdm
from dataclasses import dataclass
from typing import Optional


@dataclass
class paper_info:
    title: str
    conference: str
    year: int
    authors: str
    abstract: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_url: Optional[str] = None
    code_url: Optional[str] = None
    relevant: Optional[int] = None
    tran_flag: Optional[int] = None


def load_paper_info(src_file):
    print(f'load paper index from {src_file}')
    df = pd.read_csv(src_file)

    papers = []
    for _, item in tqdm.tqdm(df.iterrows(), total=len(df) - 1):
        paper = paper_info(title=item['title'],
                           conference=item['conference'],
                           year=item['year'],
                           authors=item['authors'],
                           abstract=item['abstract'])

        if isinstance(item['pdf_url'], str):
            paper.pdf_url = item['pdf_url']

        if isinstance(item['code_url'], str):
            paper.code_url = item['code_url']

        paper.relevant = -1
        if 'relevant' in item:
            try:
                paper.relevant = int(item['relevant'])
            except Exception as e:
                paper.relevant = -1

        paper.tran_flag = -1
        if 'relevant' in item:
            try:
                paper.tran_flag = str(item['tran_flag'])
            except Exception as e:
                paper.tran_flag = -1

        papers.append(paper)

    print(f'load {len(papers)} papers in total.')
    return papers


def filter_papers(papers, keywords, reversed_keywords):
    filtered_papers = []
    for paper in papers:
        for keyword in keywords:
            is_in_reversed = False
            if not type(paper.abstract) == str:
                print(f'invalid paper: {paper}')
                continue
            if not type(paper.title) == str:
                print(f'invalid paper: {paper}')
                continue
            if keyword in paper.abstract.lower() or keyword in paper.title.lower():
                for r_keyword in reversed_keywords:
                    if r_keyword in paper.abstract.lower() or r_keyword in paper.title.lower():
                        is_in_reversed = True
                        break
                if not is_in_reversed:
                    filtered_papers.append(paper)
                    break
    return filtered_papers
