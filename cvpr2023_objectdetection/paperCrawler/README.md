## Paper Crawler for Top AI Conferences

This is a Scrapy-based crawler. A tutorial is [at this url](https://www.logx.xyz/scrape-papers-using-scrapy). The scraped information includes:

```text
conference, year, title, authors, abstract, code_url, pdf_url, matched_queries
```

The crawler scrapes accepted papers from top AI conferences, including:

- CVPR and ICCV since 2013.
- ECCV since 2018.
- AAAI since 1980.
- IJCAI since 2017.
- NIPS since 1987.
- ICML since 2017.
- ICLR 2018, 2019, 2021 and 2022.

### Change Log

- 4-Dec-2022
  - This project was rewritten by @hellock.
- 28-OCT-2022
  - Added a feature in which the target conferences can be specified in `main.py`. See Example 4.
- 27-OCT-2022
  - Added the crawler for ACM Multimedia.
- 20-OCT-2022
  - Fixed a bug that falsely locates the paper pdf url for NIPS.
- 7-OCT-2022
  - Rewrote `main.py` so that the crawler can run over all the conferences!
- 6-OCT-2022
  - Removed the use of `PorterStemmer()` from `nltk` as it involves false negative when querying.

### Install

```shell
pip install scrapy git+https://github.com/hellock/paperCrawler.git
```

### Usage

It's a [Scrapy](https://docs.scrapy.org/en/latest/intro/tutorial.html) project. Simply cd to `PaperCrawler/conf_crawler/`
then call the spider. Some examples are provided below.

#### For single conference

```shell
scrapy crawl [conference name] -a years=[year1,year2,...,yearn] -a queries=[key1,key2,...,keyn] -o [output_filename.csv] -s JOBDIR=[checkpoint_folder]
```

- `conference name`: cvpr, iccv, eccv, aaai, ijcai, nips, icml, iclr. Must be lowercase.

- `year`: Four-digit numbers, use comma to separate.
- `keys`: The abstract must contain at least one of the keywords. Use comma to separate.
- `output_filename`: The output csv filename. The outputs will attach to previous file if two commands share the same
  output filename.
- `checkpoint_folder`: The folder to store the spider state.

##### Example 1

```shell
scrapy crawl iccv -a years=2021,2019,2017 -a queries=video,emotion -o output.csv
```

The command above can scrape the information of all papers from ICCV2017, ICCV2019, ICCV2021, with "video" OR "emotion"
appeared in the abstracts. The results will be saved in `output.csv`. It won't count the citations and save the
checkpoint. It scrapes really fast.

##### Example 2

```shell
scrapy crawl ijcai  -a years=2021,2020 -a query=video -a cc=1 -o output.csv -s JOBDIR=folder1
```

The command above will save the scraping [checkpoint](https://docs.scrapy.org/en/latest/topics/jobs.html#topics-jobs) in
a folder named `folder1`. If the scraping process is interrupted by `CTRL+C` or other incidents, simply execute the same
command so that the scraping can continue.

#### For multiple conferences

```shell
python main.py [conf1,conf2,...,confn] [year1,year2,...,yearn] -query [key1,key2,...,keyn]
```

##### Example 3

```shell
python main cvpr,iccv,eccv 2018,2019,2020,2021,2022 -query emotion,multimodal,multi-modal
```

The command would scrape the papers, whose abstracts contain at least one query, from CVPR, ICCV, and ECCV since 2018. In this case, the scraped data will be saved in `data.csv`, which is defined in `settings.py`.
