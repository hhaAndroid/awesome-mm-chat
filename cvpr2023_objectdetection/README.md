# CVPR2023 目标检测领域分析

## 1. 爬取所有 CVPR2023 论文的信息

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

会在当前路径下生成 data.csv 文件(重命名为 cvpr2023.csv，并且已经上传)，包含所有论文的信息，一共 2358 篇论文(其中有一篇论文官方 url 有问题，因此自动跳过了)。

## 2. 从 CSV 中提取和目标检测相关的论文

从标题和摘要中提取即可，进行初筛。为了尽可能不会将相关论文漏掉，我设置了关键词和反向关键词，请注意:反向关键词要小心设置，其中的词表示一定不关注的论文

关键词如下：

```python
keywords = [
    'object detection',
    'instance segmentation',
    'panoptic segmentation',
]

# 反向关键词
reversed_keywords = [
    '3d',
    'active detection',
    'anomaly',
    'oriented',
    'point cloud',
    'attribute recognition',
    '4d instance segmentation',
    'salient object detection',
    'pose estimation',
    'LiDAR',
    'few-shot',
    'cross-domain',
    'video segmentation'
]
```

用法
```shell
cd paperCrawler
python filter_with_keyword.py
```

会在当前路径下生成 filted_cvpr2023.csv 文件。 反向关键词可以根据自己的需求进行修改。经过处理，将从 2358 篇论文中刷选出了 136篇。你可能疑惑为这么多，我大概看了下，原因如下：

1. 一些通用技术，例如提出一个新的 backbone，然后应用于目标检测，这类论文没有被删，也是合理的
2. 一些非常小众的检测方向，我没有特意设置反向关键词，因此也被保留了

