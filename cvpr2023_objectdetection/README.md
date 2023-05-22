# CVPR2023 目标检测领域分析

## 1. 爬取所有 CVPR2023 论文

使用工具： https://www.logx.xyz/scrape-papers-using-scrapy

不过由于官方代码有些小 bug，因此我采用的是 fork 版本： https://github.com/hellock/paperCrawler
注意为了方便我其实已经把代码全部上传了，因此你直接 pull 下本仓库并且安装 scrapy 就可以了。不用严格按照如下步骤

(1) 环境准备

```shell
cd cvpr2023_objectdetection
pip install scrapy
git clone git@github.com:hellock/paperCrawler.git
cd paperCrawler
```

(2) 修复 代码 bug
可能是由于我的库版本都比较新，直接运行，总是会出现无法全部下载完成的情况，下载一会就失败了，而且无法使用 checkpoints。因此需要简单修复下

```shell
cd paperCrawler/conf_crawler
```

打开 pipelines.py 将 45～最后 行代码进行 try 处理，替换为如下即可

```python
if len(matched_queries) or matched_queries == "":
    item["matched_queries"] = matched_queries
    if not item.get("code_url"):
        try:
            # try to get the code url from the abstract
            url = re.findall(r"(https?://github.com\S+)", abstract)[-1]
            url = url.rstrip('".})')
            # try to get the paper url from paperswithcode
            if not url:
                url = get_code_url(title)
            item["code_url"] = url
        except Exception as e:
            item["code_url"] = ''
    return item
else:
    raise DropItem(f"No query hit in {item['title']}")
```
为了方便处理，我直接把修改后的代码直接上传了

(3) 开始下载

```shell
# 位于 paperCrawler/conf_crawler 目录下
scrapy crawl cvpr -a years=2023  -s JOBDIR=out_jobs
```

会在当前路径下生成 data.csv 文件(重命名为 cpr2023，并且已经上传)，包含所有论文的信息，一共 2358 篇论文(其中有一篇论文官方 url 有问题，因此自动跳过了)。
