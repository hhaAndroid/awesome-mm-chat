# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import re

import requests
from scrapy.exceptions import DropItem


def get_code_url(title):
    if not title:
        return ""
    r = requests.get("https://paperswithcode.com/api/v1/search", params={"q": title})
    res = r.json()
    if (
        res["count"] > 0
        and res["results"][0]["repository"] is not None
        and res["results"][0]["is_official"]
    ):
        code_url = res["results"][0]["repository"]["url"]
    else:
        code_url = ""
    return code_url


class PaperItemPipeline:
    def process_item(self, item, spider):
        # Process the item one at a time.
        abstract = item["abstract"]
        title = item["title"]
        # replace any special characters from the abstract with a space.
        clean_title = re.sub(r"\W+", " ", title).lower()
        clean_abstract = re.sub(r"\W+\-", " ", abstract).lower()
        tokens = set(clean_title.split(" ") + clean_abstract.split(" "))

        if spider.queries:
            matched_queries = [query for query in spider.queries if query in tokens]
        else:
            matched_queries = ""

        # If the queries are found in the abstract, then return the item
        # otherwise drop it.
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
